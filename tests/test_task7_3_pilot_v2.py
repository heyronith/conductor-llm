"""Unit and integration tests for Task 7.3: Authoritative Pilot-v2 Execution."""

import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from ccpt.config import (
    get_smoke_adapter_config,
    get_smoke_baseline_config,
    get_smoke_dual_stream_config,
    get_micro_adapter_config,
    get_micro_baseline_config,
    get_micro_dual_stream_config,
)
from ccpt.data.canonical_materializer import (
    TARGET_PERSISTENCE_BLOCKS,
    TARGET_TRAIN_PREFIX_BLOCKS,
    TARGET_VAL_BLOCKS,
    materialize_authoritative_fineweb_stream,
)
from ccpt.data.wildguard import (
    RiskRecord,
    SafeGenerationRecord,
    sample_wildguard_id_behavior_prompts,
)
from ccpt.evaluation.behavioral import (
    extract_raw_prompt,
    format_eval_prompt,
    wilson_score_interval,
)
from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
from ccpt.training.engine import (
    create_identical_dual_stream_models,
    snapshot_parameters,
    assert_parameters_equal,
    count_changed_parameters,
)
from ccpt.training.losses import compute_causal_lm_loss, compute_safe_generation_loss
from ccpt.training.preflight_proofs import scan_production_paths
from ccpt.training.safety_schedule import (
    generate_authoritative_safety_schedule,
)


# =============================================================================
# 1. Safety Schedule Determinism & Properties
# =============================================================================

def _create_dummy_safety_records(n_risk: int = 100, n_gen: int = 50):
    risk_recs = []
    for i in range(n_risk):
        risk_recs.append(RiskRecord(
            example_id=f"risk_{i:04d}",
            prompt_group_key=f"group_{i}",
            input_ids=[10 + (i % 50)] * 30,  # 30 tokens each
            prompt_end_index=15,
            risk_label=1 if (i % 2 == 0) else 0,
            is_adversarial=False,
            subcategory="test",
            split="train",
        ))
    gen_recs = []
    for j in range(n_gen):
        gen_recs.append(SafeGenerationRecord(
            example_id=f"gen_{j:04d}",
            prompt_group_key=f"group_{j}",
            input_ids=[100 + (j % 50)] * 40,  # 40 tokens each
            prompt_end_index=20,
            risk_label=1 if (j % 2 == 0) else 0,
            is_refusal=True,
            is_adversarial=False,
            subcategory="test",
            split="train",
        ))
    return risk_recs, gen_recs


def test_safety_schedule_determinism_and_alternation():
    """Verify schedule generation is bit-deterministic and strictly 1:1 alternating."""
    risk_recs, gen_recs = _create_dummy_safety_records(n_risk=80, n_gen=40)
    target_tokens = 5000  # small target

    sched_1 = generate_authoritative_safety_schedule(
        risk_records=risk_recs,
        gen_records=gen_recs,
        target_safety_tokens=target_tokens,
        batch_size=8,
        seed=20260821,
    )

    sched_2 = generate_authoritative_safety_schedule(
        risk_records=risk_recs,
        gen_records=gen_recs,
        target_safety_tokens=target_tokens,
        batch_size=8,
        seed=20260821,
    )

    # Determinism
    assert sched_1["schedule_hash"] == sched_2["schedule_hash"]
    assert sched_1["total_batches"] == sched_2["total_batches"]
    assert sched_1["total_valid_input_tokens"] == sched_2["total_valid_input_tokens"]
    assert sched_1["total_valid_input_tokens"] >= target_tokens

    # 1:1 Alternation check
    batches = sched_1["batches"]
    for idx, b in enumerate(batches):
        expected_type = "risk" if (idx % 2 == 0) else "generation"
        assert b["batch_type"] == expected_type, f"Batch {idx} expected {expected_type}, got {b['batch_type']}"
        assert len(b["example_ids"]) == 8


# =============================================================================
# 2. Model D Identity Initialization & Parameter Matching
# =============================================================================

def test_model_d_identity_initialization():
    """Verify Model D adapter scale 1.0 vs 0.0 gives exact identical logits at initialization."""
    cfg = get_smoke_adapter_config()
    model_d = FrozenBackboneAdapterModel(cfg)
    model_d.eval()

    torch.manual_seed(20260821)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))

    with torch.no_grad():
        logits_scale_1, _ = model_d(input_ids, adapter_scale=1.0)
        logits_scale_0, _ = model_d(input_ids, adapter_scale=0.0)

    # Maximum absolute difference must be strictly 0.0
    max_diff = (logits_scale_1 - logits_scale_0).abs().max().item()
    assert max_diff == 0.0, f"Expected 0.0 logit difference, got {max_diff}"


def test_model_d_parameter_matching_tolerances():
    """Verify Model D matches Model C parameters within 0.1% tolerance."""
    smoke_dual = get_smoke_dual_stream_config()
    smoke_adapter = get_smoke_adapter_config()

    model_c = CCPTDualStreamModel(smoke_dual)
    model_d = FrozenBackboneAdapterModel(smoke_adapter)

    c_total = sum(p.numel() for p in model_c.parameters())
    c_theta_c = sum(p.numel() for p in model_c.theta_C)
    c_theta_n = sum(p.numel() for p in model_c.theta_N)

    d_total = sum(p.numel() for p in model_d.parameters())
    d_backbone = sum(p.numel() for p in model_d.backbone_parameters)
    d_safety = sum(p.numel() for p in model_d.safety_parameters)

    assert d_backbone == c_theta_c == 33_165_824
    assert abs(d_total - c_total) / c_total < 0.001
    assert abs(d_safety - c_theta_n) / c_theta_n < 0.001


# =============================================================================
# 3. Model B/C Initialization Equality
# =============================================================================

def test_model_b_and_c_initialization_equality():
    """Verify Model B and Model C start bit-for-bit identical."""
    cfg = get_smoke_dual_stream_config()
    model_b, model_c = create_identical_dual_stream_models(cfg, seed=20260821)

    state_b = model_b.state_dict()
    state_c = model_c.state_dict()

    assert set(state_b.keys()) == set(state_c.keys())
    for k in state_b:
        assert torch.equal(state_b[k], state_c[k]), f"Mismatch in {k}"


# =============================================================================
# 4. Model B LM Gradient Behavior
# =============================================================================

def test_model_b_lm_gradients_natural_flow():
    """Verify Model B in controlled mode gives natural LM gradients to participating parameters."""
    cfg = get_micro_dual_stream_config()
    model_b = JointTrainingDualStreamModel(cfg)
    model_b.train()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, risk_logits = model_b(input_ids, mode="controlled")
    assert risk_logits is None, "risk_logits should be None when prompt_end_indices is not provided"

    loss = compute_causal_lm_loss(logits, input_ids)
    loss.backward()

    # Capability params must have gradients
    for p in model_b.theta_C:
        assert p.grad is not None and not torch.all(p.grad == 0), "Capability parameter missing gradient"

    # Controller gate & steering projections participate in generation -> must have gradients
    for gate_proj in model_b.gate_projections:
        assert gate_proj.weight.grad is not None

    for steer_proj in model_b.steering_projections:
        assert steer_proj.weight.grad is not None

    # Risk head does NOT participate in promptless LM next-token logits -> grad is None
    assert model_b.risk_head.weight.grad is None
    # Normative final norm does NOT participate -> grad is None
    assert model_b.normative_final_norm.weight.grad is None


# =============================================================================
# 5. Model C and D Freezing and Snapshot Invariants
# =============================================================================

def test_model_c_and_d_freezing_invariants():
    """Verify theta_C / backbone freezing during safety training."""
    cfg_c = get_micro_dual_stream_config()
    model_c = CCPTDualStreamModel(cfg_c)
    model_c.train()

    # Freeze theta_C
    for p in model_c.theta_C:
        p.requires_grad = False
    for p in model_c.theta_N:
        p.requires_grad = True

    c_snap = snapshot_parameters(model_c.theta_C)

    input_ids = torch.randint(0, cfg_c.vocab_size, (2, 16))
    prompt_end = torch.tensor([8, 8])
    labels = torch.tensor([1.0, 0.0])

    logits, risk_logits = model_c(input_ids, prompt_end_indices=prompt_end, mode="controlled")
    l_risk = torch.nn.functional.binary_cross_entropy_with_logits(risk_logits, labels)
    l_gen = compute_safe_generation_loss(logits, input_ids, prompt_end)
    (l_risk + l_gen).backward()

    opt_c = torch.optim.AdamW([p for p in model_c.theta_N if p.requires_grad], lr=1e-3)
    opt_c.step()

    # Assert theta_C unchanged
    assert count_changed_parameters(model_c.theta_C, c_snap) == 0

    # Model D
    cfg_d = get_micro_adapter_config()
    model_d = FrozenBackboneAdapterModel(cfg_d)
    model_d.train()
    model_d.freeze_backbone()

    d_backbone_snap = snapshot_parameters(model_d.backbone_parameters)

    logits_d, risk_logits_d = model_d(input_ids, prompt_end_indices=prompt_end, adapter_scale=1.0)
    l_risk_d = torch.nn.functional.binary_cross_entropy_with_logits(risk_logits_d, labels)
    l_gen_d = compute_safe_generation_loss(logits_d, input_ids, prompt_end)
    (l_risk_d + l_gen_d).backward()

    opt_d = torch.optim.AdamW(model_d.safety_parameters, lr=1e-3)
    opt_d.step()

    # Assert backbone unchanged
    assert count_changed_parameters(model_d.backbone_parameters, d_backbone_snap) == 0


# =============================================================================
# 6. Streaming Production Materializer & Manifest Cleanliness
# =============================================================================

def test_production_materializer_no_raw_bytes_b64(tmp_path):
    """Verify production streaming materializer produces clean manifest without raw_bytes_b64."""
    from transformers import AutoTokenizer
    # Mock tokenizer
    class SimpleMockTokenizer:
        vocab_size = 1000
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [10 + (len(text) % 50)] * max(1, len(text.split()))

    tok = SimpleMockTokenizer()
    docs = [
        {"id": f"doc_{i}", "text": f"Document text sample number {i} with many words to fill buffers."}
        for i in range(500)
    ]

    res = materialize_authoritative_fineweb_stream(
        output_dir=tmp_path / "fineweb_stream_test",
        tokenizer=tok,
        document_iterable=docs,
        train_prefix_blocks=10,
        persistence_blocks=5,
        val_blocks=3,
        sequence_length=16,
        shard_size_blocks=4,
        val_modulo=10,
    )

    manifest = res["manifest"]
    manifest_str = json.dumps(manifest)

    assert "raw_bytes_b64" not in manifest_str
    assert manifest["train_prefix"]["target_blocks"] == 10
    assert manifest["persistence_continuation"]["target_blocks"] == 5
    assert manifest["validation"]["target_blocks"] == 3


# =============================================================================
# 7. Wilson Score Confidence Interval
# =============================================================================

def test_wilson_score_interval():
    """Verify Wilson score confidence interval mathematical properties."""
    low, high = wilson_score_interval(50, 100)
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert low < high

    # Edge cases
    assert wilson_score_interval(0, 100)[0] == 0.0
    assert wilson_score_interval(100, 100)[1] == 1.0


# =============================================================================
# 8. ID Behavioral Salted Hash Sampling
# =============================================================================

def test_wildguard_id_behavior_salted_sampling():
    """Verify salted hash ranking for ID behavioral benchmark is deterministic and balanced."""
    risk_recs = []
    for i in range(100):
        risk_recs.append(RiskRecord(
            example_id=f"ex_{i:04d}",
            prompt_group_key=f"group_{i}",
            input_ids=[1, 10, 11, 12, 2],
            prompt_end_index=3,
            risk_label=1 if i < 50 else 0,
            is_adversarial=False,
            subcategory="test",
            split="validation",
        ))

    class DummyTok:
        def decode(self, tokens, skip_special_tokens=False):
            return f"Prompt for tokens {tokens}"

    h_prompts, b_prompts, manifest = sample_wildguard_id_behavior_prompts(
        records=risk_recs,
        tokenizer=DummyTok(),
        n_harmful=20,
        n_benign=20,
    )

    assert len(h_prompts) == 20
    assert len(b_prompts) == 20
    assert manifest["selection_algorithm"] == "salted_hash_ranking_v1"
    assert manifest["n_harmful"] == 20
    assert manifest["n_benign"] == 20


# =============================================================================
# 9. Prompt Framing Verification
# =============================================================================

def test_prompt_framing_no_double_wrapping():
    """Verify prompt extraction and formatting prevents double framing."""
    raw = "What is the capital of France?"
    framed = format_eval_prompt(raw)
    assert framed == "<s>User: What is the capital of France?\nAssistant:"

    # If prompt was already framed, extracting raw prompt recovers exact text
    extracted = extract_raw_prompt(framed)
    assert extracted == "What is the capital of France?"
    assert format_eval_prompt(extracted) == framed
