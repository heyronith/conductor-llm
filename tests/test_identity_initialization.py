"""Tests verifying identity controller initialization and mathematical output bounds."""

import torch

from ccpt.config import get_micro_dual_stream_config, get_smoke_dual_stream_config
from ccpt.modeling.dual_stream import CCPTDualStreamModel


def test_identity_initialization_equivalence():
    """Controlled CCPT must initially produce identical outputs to its capability-only forward path."""
    config = get_smoke_dual_stream_config()
    model = CCPTDualStreamModel(config)
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits_lm, _ = model(input_ids, mode="lm")
        logits_controlled, _, diags = model(input_ids, mode="controlled", return_diagnostics=True)

    # 1. Output equivalence assertion (bit-for-bit / high precision)
    assert torch.allclose(logits_lm, logits_controlled, atol=1e-5, rtol=1e-5), (
        "Controlled CCPT forward does not match LM forward at zero-initialization"
    )

    # 2. Diagnostic controller values assertion
    for layer_idx in config.controlled_layers:
        gate = diags["gates"][f"layer_{layer_idx}"]
        steer = diags["steering"][f"layer_{layer_idx}"]

        expected_gate = torch.ones_like(gate)
        expected_steer = torch.zeros_like(steer)

        assert torch.allclose(gate, expected_gate, atol=1e-7), f"Gate at layer {layer_idx} is not exactly 1.0"
        assert torch.allclose(steer, expected_steer, atol=1e-7), f"Steering at layer {layer_idx} is not exactly 0.0"


def test_model_d_identity_initialization_exact():
    """Fresh Model D must produce bit-for-bit / exact identical logits for adapter_scale=1.0 vs 0.0."""
    from ccpt.config import get_smoke_adapter_config
    from ccpt.modeling.adapter import FrozenBackboneAdapterModel

    config = get_smoke_adapter_config()
    model = FrozenBackboneAdapterModel(config)
    model.eval()

    # Verify all adapter up-projections are strictly zeros
    for layer_idx, layer in enumerate(model.layers):
        assert torch.equal(layer.attn_adapter.up_proj.weight, torch.zeros_like(layer.attn_adapter.up_proj.weight)), (
            f"Layer {layer_idx} attn_adapter up_proj is not strictly zero!"
        )
        assert torch.equal(layer.mlp_adapter.up_proj.weight, torch.zeros_like(layer.mlp_adapter.up_proj.weight)), (
            f"Layer {layer_idx} mlp_adapter up_proj is not strictly zero!"
        )

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits_scale_1, _ = model(input_ids, adapter_scale=1.0)
        logits_scale_0, _ = model(input_ids, adapter_scale=0.0)

    # Exact equality because adapter output residual is strictly 0
    max_diff = (logits_scale_1 - logits_scale_0).abs().max().item()
    assert max_diff == 0.0, f"Model D scale 1.0 vs 0.0 max diff is non-zero: {max_diff}"
    assert torch.equal(logits_scale_1, logits_scale_0), "Model D logits differ between scale 1.0 and 0.0!"


def test_controller_mathematical_bounds():
    """Verify that controller outputs strictly obey their analytical bounds (0.9 < g < 1.1, -1.0 < s < 1.0)."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)
    model.eval()

    # Perturb controller weights to extreme values to test saturation behavior
    with torch.no_grad():
        for gate_proj in model.gate_projections:
            gate_proj.weight.fill_(1000.0)
        for steer_proj in model.steering_projections:
            steer_proj.weight.fill_(-1000.0)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        _, _, diags_pos = model(input_ids, mode="controlled", return_diagnostics=True)

    for layer_idx in config.controlled_layers:
        gate_pos = diags_pos["gates"][f"layer_{layer_idx}"]
        steer_pos = diags_pos["steering"][f"layer_{layer_idx}"]

        # alpha = 0.1 -> g must be in (0.9, 1.1)
        assert torch.all(gate_pos >= 0.9 - 1e-6) and torch.all(gate_pos <= 1.1 + 1e-6)
        # beta = 1.0 -> s must be in (-1.0, 1.0)
        assert torch.all(steer_pos >= -1.0 - 1e-6) and torch.all(steer_pos <= 1.0 + 1e-6)

    # Test reverse perturbation
    with torch.no_grad():
        for gate_proj in model.gate_projections:
            gate_proj.weight.fill_(-1000.0)
        for steer_proj in model.steering_projections:
            steer_proj.weight.fill_(1000.0)

    with torch.no_grad():
        _, _, diags_neg = model(input_ids, mode="controlled", return_diagnostics=True)

    for layer_idx in config.controlled_layers:
        gate_neg = diags_neg["gates"][f"layer_{layer_idx}"]
        steer_neg = diags_neg["steering"][f"layer_{layer_idx}"]

        assert torch.all(gate_neg >= 0.9 - 1e-6) and torch.all(gate_neg <= 1.1 + 1e-6)
        assert torch.all(steer_neg >= -1.0 - 1e-6) and torch.all(steer_neg <= 1.0 + 1e-6)
