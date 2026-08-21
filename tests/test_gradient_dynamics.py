"""Tests verifying gradient propagation through frozen capability operations and zero-initialization dynamics."""

import torch

from ccpt.config import get_micro_dual_stream_config
from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.training.gradients import gradient_summary, set_requires_grad
from ccpt.training.losses import (
    risk_classification_loss,
    safe_generation_loss,
)


def get_deterministic_synthetic_batch(config):
    torch.manual_seed(1234)
    B, T = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    prompt_end_indices = torch.tensor([2, 4])
    risk_labels = torch.tensor([1, 0])
    return input_ids, prompt_end_indices, risk_labels


def test_frozen_capability_operations_remain_differentiable():
    """Test 6: Safe-generation gradients flow through frozen capability operations back to controller and deeper N."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    # Freeze capability parameters
    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)

    # Perturb controllers away from zero so gradients propagate back into N
    with torch.no_grad():
        for steer in model.steering_projections:
            steer.weight.normal_(mean=0.0, std=1e-3)
        for gate in model.gate_projections:
            gate.weight.normal_(mean=0.0, std=1e-3)

    input_ids, prompt_end_indices, _ = get_deterministic_synthetic_batch(config)
    logits, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")

    loss = safe_generation_loss(logits, input_ids, prompt_end_indices)
    loss.backward()

    # Verify that capability parameters have no gradient
    summary_C = gradient_summary(model.theta_C)
    assert summary_C["num_grad_nonzero"] == 0

    # Verify that controller projections received gradients
    summary_controllers = gradient_summary(
        list(model.gate_projections.parameters()) + list(model.steering_projections.parameters())
    )
    assert summary_controllers["grad_norm"] > 0.0

    # Verify that deeper normative parameters received gradients via backprop through frozen C operations
    summary_norm_layers = gradient_summary(model.normative_layers.parameters())
    summary_pin = gradient_summary(model.p_in.parameters())
    summary_obs = gradient_summary(model.obs_projections.parameters())

    assert summary_norm_layers["grad_norm"] > 0.0
    assert summary_pin["grad_norm"] > 0.0
    assert summary_obs["grad_norm"] > 0.0


def test_zero_initialization_gradient_dynamics():
    """Test 7: At exact zero-controller initialization, safe-generation loss trains controller matrices but not deeper N."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)

    input_ids, prompt_end_indices, _ = get_deterministic_synthetic_batch(config)
    logits, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")

    loss = safe_generation_loss(logits, input_ids, prompt_end_indices)
    loss.backward()

    # Controller projections must receive non-zero gradient from generation loss
    summary_controllers = gradient_summary(
        list(model.gate_projections.parameters()) + list(model.steering_projections.parameters())
    )
    assert summary_controllers["grad_norm"] > 0.0

    # Deeper normative parameters must have exactly zero gradient because W_s = 0 and W_g = 0 initially
    summary_norm_layers = gradient_summary(model.normative_layers.parameters())
    summary_pin = gradient_summary(model.p_in.parameters())
    summary_obs = gradient_summary(model.obs_projections.parameters())

    assert summary_norm_layers["num_grad_nonzero"] == 0
    assert summary_pin["num_grad_nonzero"] == 0
    assert summary_obs["num_grad_nonzero"] == 0


def test_risk_loss_trains_deeper_normative_network_immediately():
    """Test 8: At exact zero controller initialization, pure risk loss immediately trains deeper normative blocks."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)

    input_ids, prompt_end_indices, risk_labels = get_deterministic_synthetic_batch(config)
    _, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")

    loss = risk_classification_loss(risk_logits, risk_labels)
    loss.backward()

    summary_risk_head = gradient_summary(model.risk_head.parameters())
    summary_norm_layers = gradient_summary(model.normative_layers.parameters())
    summary_obs = gradient_summary(model.obs_projections.parameters())
    summary_pin = gradient_summary(model.p_in.parameters())

    assert summary_risk_head["grad_norm"] > 0.0
    assert summary_norm_layers["grad_norm"] > 0.0
    assert summary_obs["grad_norm"] > 0.0
    assert summary_pin["grad_norm"] > 0.0


def test_generation_gradient_reaches_deeper_n_after_controllers_move():
    """Test 9: Once controller weights move away from zero, generation loss reaches deeper normative layers."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)

    # 1. Perturb controller matrices
    with torch.no_grad():
        for steer in model.steering_projections:
            steer.weight.fill_(0.05)
        for gate in model.gate_projections:
            gate.weight.fill_(0.05)

    input_ids, prompt_end_indices, _ = get_deterministic_synthetic_batch(config)
    logits, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")

    loss = safe_generation_loss(logits, input_ids, prompt_end_indices)
    loss.backward()

    summary_pin = gradient_summary(model.p_in.parameters())
    summary_obs = gradient_summary(model.obs_projections.parameters())
    summary_norm_layers = gradient_summary(model.normative_layers.parameters())

    assert summary_pin["grad_norm"] > 0.0
    assert summary_obs["grad_norm"] > 0.0
    assert summary_norm_layers["grad_norm"] > 0.0


def test_combined_normative_loss_reaches_all_intended_components():
    """Test 10: Combined loss (L_risk + L_safe) at initialization reaches all required logical components."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)

    input_ids, prompt_end_indices, risk_labels = get_deterministic_synthetic_batch(config)
    logits, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")

    loss = risk_classification_loss(risk_logits, risk_labels) + 1.0 * safe_generation_loss(
        logits, input_ids, prompt_end_indices
    )
    loss.backward()

    # Check each logical component
    components = {
        "p_in": model.p_in,
        "obs_0": model.obs_projections[0],
        "obs_1": model.obs_projections[1],
        "norm_block_0": model.normative_layers[0],
        "norm_block_1": model.normative_layers[1],
        "gate_0": model.gate_projections[0],
        "gate_1": model.gate_projections[1],
        "steer_0": model.steering_projections[0],
        "steer_1": model.steering_projections[1],
        "normative_final_norm": model.normative_final_norm,
        "risk_head": model.risk_head,
    }

    for name, module in components.items():
        summary = gradient_summary(module.parameters())
        assert summary["grad_norm"] > 0.0, f"Module {name} did not receive gradients from combined loss"
        assert summary["all_finite"] is True, f"Module {name} has non-finite gradients"
