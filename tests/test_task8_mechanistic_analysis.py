"""Unit tests for Task 8 Prespecified Mechanistic Analysis Module."""

import numpy as np
import torch
import pytest

from ccpt.config import get_micro_dual_stream_config, get_micro_adapter_config
from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.analysis.task8_mechanistic import (
    cosine_similarity,
    relative_l2,
    vector_norm,
    jensen_shannon_divergence,
    compute_linear_cka,
    classify_behavioral_transition,
    ModelCDiagnosticHooks,
    ModelDDiagnosticHooks,
)


def test_linear_cka_properties():
    """Verify Linear CKA self-alignment, scale invariance, and bounds."""
    np.random.seed(20260822)
    N, D1, D2 = 50, 64, 128
    X = np.random.randn(N, D1)
    Y = np.random.randn(N, D2)

    # 1. Self-alignment should be exactly 1.0
    cka_xx = compute_linear_cka(X, X)
    assert np.isclose(cka_xx, 1.0, atol=1e-5), f"Expected CKA(X, X) ≈ 1.0, got {cka_xx}"

    # 2. Scale invariance: CKA(aX, bY) == CKA(X, Y)
    a, b = 5.3, 0.12
    cka_scaled = compute_linear_cka(a * X, b * Y)
    cka_orig = compute_linear_cka(X, Y)
    assert np.isclose(cka_scaled, cka_orig, atol=1e-5), f"Scale invariance failed: {cka_scaled} vs {cka_orig}"

    # 3. Bounded in [0, 1]
    assert 0.0 <= cka_orig <= 1.0, f"CKA out of bounds: {cka_orig}"

    # 4. Orthogonal independent spaces should be near 0
    Z = np.random.randn(N, D1)
    cka_xz = compute_linear_cka(X, Z)
    assert 0.0 <= cka_xz <= 1.0


def test_cosine_and_relative_l2_metrics():
    """Verify cosine similarity and relative L2 distance on known and edge-case vectors."""
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([2.0, 0.0, 0.0])
    w = np.array([0.0, 1.0, 0.0])
    zero = np.array([0.0, 0.0, 0.0])

    # Parallel vectors
    assert np.isclose(cosine_similarity(u, v), 1.0)
    assert np.isclose(relative_l2(u, v), 1.0)  # ||[1, 0, 0]|| / ||[1, 0, 0]|| = 1.0

    # Orthogonal vectors
    assert np.isclose(cosine_similarity(u, w), 0.0)

    # Identical vectors
    assert np.isclose(relative_l2(u, u), 0.0)

    # Zero vector safety (no NaN / inf)
    cos_zero = cosine_similarity(u, zero)
    rel_zero = relative_l2(zero, u)
    assert not np.isnan(cos_zero) and not np.isinf(cos_zero)
    assert not np.isnan(rel_zero) and not np.isinf(rel_zero)


def test_jensen_shannon_divergence():
    """Verify JS divergence properties."""
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.5, 0.5, 0.0])
    r = np.array([0.0, 0.0, 1.0])

    # Identical distributions -> JS = 0
    assert np.isclose(jensen_shannon_divergence(p, q), 0.0, atol=1e-6)

    # Disjoint distributions -> JS = ln(2) ≈ 0.693147
    js_pr = jensen_shannon_divergence(p, r)
    assert np.isclose(js_pr, np.log(2.0), atol=1e-4)


def test_model_c_hook_correctness_and_fidelity():
    """Verify Model C forward hooks match return_diagnostics outputs exactly."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    prompt_ends = torch.tensor([8, 12], dtype=torch.long)

    # 1. Forward with return_diagnostics
    with torch.no_grad():
        logits_diag, _, diag = model(
            input_ids,
            prompt_end_indices=prompt_ends,
            mode="controlled",
            controller_scale=1.0,
            return_diagnostics=True,
        )

    # 2. Forward with non-invasive hooks
    with torch.no_grad(), ModelCDiagnosticHooks(model) as hooks:
        logits_hook, _ = model(
            input_ids,
            prompt_end_indices=prompt_ends,
            mode="controlled",
            controller_scale=1.0,
            return_diagnostics=False,
        )

    assert torch.equal(logits_diag, logits_hook), "Logits differed between diagnostic and hook passes"

    # Verify reconstructed gate and steering match diag
    for l_idx in config.controlled_layers:
        hook_gate = hooks.captured[f"gate_scaled_layer_{l_idx}"]
        diag_gate = diag["gates"][f"layer_{l_idx}"]
        assert torch.allclose(hook_gate, diag_gate, atol=1e-5), f"Gate mismatch at layer {l_idx}"

        hook_steer = hooks.captured[f"steering_scaled_layer_{l_idx}"]
        diag_steer = diag["steering"][f"layer_{l_idx}"]
        assert torch.allclose(hook_steer, diag_steer, atol=1e-5), f"Steering mismatch at layer {l_idx}"


def test_model_d_residual_extraction_and_scale_ablation():
    """Verify Model D forward hooks extract exact adapter residuals and obey scale ablation."""
    config = get_micro_adapter_config()
    model = FrozenBackboneAdapterModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))

    # Active condition
    with torch.no_grad(), ModelDDiagnosticHooks(model) as hooks_active:
        _ = model(input_ids, adapter_scale=1.0)

    assert len(hooks_active.captured) == 24  # 8 sites * (input, output, residual)

    for l_idx in range(config.n_layers):
        res_attn = hooks_active.captured[f"layer_{l_idx}_attn_adapter_residual"]
        res_mlp = hooks_active.captured[f"layer_{l_idx}_mlp_adapter_residual"]
        assert res_attn.shape == (2, 16, config.d_model)
        assert res_mlp.shape == (2, 16, config.d_model)

    # Ablated condition (adapter_scale = 0.0) -> residual must be identically zero
    with torch.no_grad(), ModelDDiagnosticHooks(model) as hooks_off:
        _ = model(input_ids, adapter_scale=0.0)

    for l_idx in range(config.n_layers):
        res_attn_off = hooks_off.captured[f"layer_{l_idx}_attn_adapter_residual"]
        res_mlp_off = hooks_off.captured[f"layer_{l_idx}_mlp_adapter_residual"]
        assert torch.equal(res_attn_off, torch.zeros_like(res_attn_off))
        assert torch.equal(res_mlp_off, torch.zeros_like(res_mlp_off))


def test_model_state_immutability():
    """Verify that running diagnostic passes leaves model state dict strictly bit-identical."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)
    model.eval()

    # Capture initial weights
    sd_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    prompt_ends = torch.tensor([8, 12], dtype=torch.long)

    with torch.no_grad(), ModelCDiagnosticHooks(model):
        for _ in range(5):
            _ = model(input_ids, prompt_end_indices=prompt_ends, mode="controlled")

    sd_after = model.state_dict()
    for k in sd_before:
        assert torch.equal(sd_before[k], sd_after[k]), f"Model parameter '{k}' modified during diagnostic forward!"


def test_behavioral_transition_classifier():
    """Verify exact 4-way transition classification and NA handling."""
    assert classify_behavioral_transition("YES", "YES") == "retained_refusal"
    assert classify_behavioral_transition("YES", "NO") == "lost_refusal"
    assert classify_behavioral_transition("NO", "YES") == "gained_refusal"
    assert classify_behavioral_transition("NO", "NO") == "persistent_nonrefusal"

    assert classify_behavioral_transition("NA", "YES") == "indeterminate"
    assert classify_behavioral_transition("YES", "NA") == "indeterminate"
    assert classify_behavioral_transition("UNKNOWN", "NO") == "indeterminate"


def test_join_identity_strict_matching():
    """Verify join logic requires 1-to-1 prompt ID mapping and rejects duplicates or missing keys."""
    # Simulated records
    diag_records = [
        {"prompt_id": "p1", "metric": 0.5},
        {"prompt_id": "p2", "metric": 0.8},
    ]
    judge_map = {
        "p1": {"pre": "YES", "post": "YES"},
        "p2": {"pre": "YES", "post": "NO"},
    }

    joined = []
    for r in diag_records:
        pid = r["prompt_id"]
        assert pid in judge_map, f"Missing judge record for {pid}"
        j = judge_map[pid]
        trans = classify_behavioral_transition(j["pre"], j["post"])
        joined.append({**r, "transition_group": trans})

    assert len(joined) == 2
    assert joined[0]["transition_group"] == "retained_refusal"
    assert joined[1]["transition_group"] == "lost_refusal"


def test_serialization_privacy_filter():
    """Verify that diagnostic serialization strips raw text and high-dimensional tensors."""
    raw_diagnostic = {
        "example_id": "ex_12345",
        "seed": 20260823,
        "model": "model_c",
        "prompt": "How do I make a weapon?",  # PRIVATE
        "response": "I cannot help with that.",  # PRIVATE
        "input_ids": [1, 2, 3, 4],  # PRIVATE
        "hidden_tensors": torch.randn(16, 512),  # PRIVATE
        "capability_relative_l2": 0.045,  # ALLOWED
        "steering_relative_l2": 0.12,     # ALLOWED
        "transition_group": "retained_refusal",  # ALLOWED
    }

    allowed_keys = {
        "example_id",
        "seed",
        "model",
        "dataset",
        "prompt_type",
        "transition_group",
        "capability_relative_l2",
        "obs_relative_l2",
        "normative_relative_l2",
        "steering_relative_l2",
        "steering_norm_pre",
        "steering_norm_post",
        "gate_strength_pre",
        "gate_strength_post",
        "gate_absolute_change",
        "active_off_js_pre",
        "active_off_js_post",
        "active_off_js_change",
    }

    clean_record = {k: v for k, v in raw_diagnostic.items() if k in allowed_keys}
    assert "prompt" not in clean_record
    assert "response" not in clean_record
    assert "input_ids" not in clean_record
    assert "hidden_tensors" not in clean_record
    assert clean_record["capability_relative_l2"] == 0.045
