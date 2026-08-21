"""Tests verifying gradient firewall invariants, observation edge detach, and optimization isolation."""

import torch

from ccpt.config import get_micro_baseline_config, get_micro_dual_stream_config
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
from ccpt.training.gradients import (
    gradient_summary,
    parameters_bit_identical,
    set_requires_grad,
    snapshot_parameters,
)
from ccpt.training.losses import (
    causal_lm_loss,
    risk_classification_loss,
    safe_generation_loss,
)


def get_deterministic_synthetic_batch(config):
    """Generates a deterministic synthetic batch for gradient topology tests."""
    torch.manual_seed(1234)
    B, T = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    prompt_end_indices = torch.tensor([2, 4])
    risk_labels = torch.tensor([1, 0])
    return input_ids, prompt_end_indices, risk_labels


def test_ccpt_lm_mode_firewall():
    """Test 1: In CCPT LM mode, L_LM must update theta_C while theta_N receives zero gradients."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    set_requires_grad(model.theta_C, True)
    set_requires_grad(model.theta_N, True)

    input_ids, _, _ = get_deterministic_synthetic_batch(config)
    logits, risk_logits = model(input_ids, mode="lm")
    assert risk_logits is None

    loss = causal_lm_loss(logits, input_ids)
    loss.backward()

    summary_C = gradient_summary(model.theta_C)
    summary_N = gradient_summary(model.theta_N)

    # theta_C must have valid non-zero gradients
    assert summary_C["num_grad_nonzero"] > 0
    assert summary_C["grad_norm"] > 0.0
    assert summary_C["all_finite"] is True

    # theta_N must have zero gradients (all grad is None because N pathway is bypassed)
    assert summary_N["num_grad_nonzero"] == 0
    assert summary_N["grad_norm"] == 0.0
    assert summary_N["num_grad_none"] == summary_N["num_params"]


def test_pure_risk_loss_cannot_update_capability_through_observation_edge():
    """Test 3: Pure risk loss cannot backpropagate into theta_C through the detached observation edge."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    # Leave both parameter groups fully trainable to test architectural detach invariant
    set_requires_grad(model.theta_C, True)
    set_requires_grad(model.theta_N, True)

    input_ids, prompt_end_indices, risk_labels = get_deterministic_synthetic_batch(config)
    _, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")

    loss = risk_classification_loss(risk_logits, risk_labels)
    loss.backward()

    summary_C = gradient_summary(model.theta_C)
    summary_N = gradient_summary(model.theta_N)

    # theta_N must receive risk gradients
    assert summary_N["num_grad_nonzero"] > 0
    assert summary_N["grad_norm"] > 0.0
    assert summary_N["all_finite"] is True

    # theta_C must receive zero gradients through the observation edge
    assert summary_C["num_grad_nonzero"] == 0
    assert summary_C["grad_norm"] == 0.0


def test_normative_training_with_frozen_capability_parameters():
    """Test 4: Normative training with frozen capability parameters updates theta_N with zero gradient on theta_C."""
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

    summary_C = gradient_summary(model.theta_C)
    summary_N = gradient_summary(model.theta_N)

    # theta_C has requires_grad=False, so all grad are None / zero
    assert summary_C["num_grad_nonzero"] == 0
    assert summary_C["grad_norm"] == 0.0

    # theta_N receives active non-zero gradients
    assert summary_N["num_grad_nonzero"] > 0
    assert summary_N["grad_norm"] > 0.0
    assert summary_N["all_finite"] is True


def test_model_b_ordinary_lm_gradients():
    """Test 11: In Model B (Joint-Training Control), LM loss updates both theta_C and the controller."""
    config = get_micro_dual_stream_config()
    model_b = JointTrainingDualStreamModel(config)

    set_requires_grad(model_b.theta_C, True)
    set_requires_grad(model_b.theta_N, True)

    input_ids, _, _ = get_deterministic_synthetic_batch(config)
    logits, _ = model_b(input_ids, mode="controlled")

    loss = causal_lm_loss(logits, input_ids)
    loss.backward()

    summary_C = gradient_summary(model_b.theta_C)
    controller_params = list(model_b.gate_projections.parameters()) + list(model_b.steering_projections.parameters())
    summary_controllers = gradient_summary(controller_params)

    assert summary_C["grad_norm"] > 0.0
    assert summary_controllers["grad_norm"] > 0.0
    assert summary_C["all_finite"] is True
    assert summary_controllers["all_finite"] is True


def test_model_b_after_nonzero_controller_initialization():
    """Test 12: In Model B after controller perturbation, LM loss gradients reach deeper normative layers."""
    config = get_micro_dual_stream_config()
    model_b = JointTrainingDualStreamModel(config)

    # Perturb controller matrices
    with torch.no_grad():
        for steer in model_b.steering_projections:
            steer.weight.normal_(0.0, 0.01)

    set_requires_grad(model_b.theta_C, True)
    set_requires_grad(model_b.theta_N, True)

    input_ids, _, _ = get_deterministic_synthetic_batch(config)
    logits, _ = model_b(input_ids, mode="controlled")

    loss = causal_lm_loss(logits, input_ids)
    loss.backward()

    summary_C = gradient_summary(model_b.theta_C)
    summary_N = gradient_summary(model_b.theta_N)
    summary_norm_layers = gradient_summary(model_b.normative_layers.parameters())

    assert summary_C["grad_norm"] > 0.0
    assert summary_N["grad_norm"] > 0.0
    assert summary_norm_layers["grad_norm"] > 0.0


def test_model_a_reference_behavior():
    """Test 13: Model A baseline receives standard gradients for both LM and risk supervision."""
    config = get_micro_baseline_config()
    model_a = ParameterMatchedBaselineModel(config)

    input_ids, prompt_end_indices, risk_labels = get_deterministic_synthetic_batch(config)

    # 1. LM loss
    logits, _ = model_a(input_ids)
    loss_lm = causal_lm_loss(logits, input_ids)
    loss_lm.backward()

    summary_core = gradient_summary(model_a.core_lm_parameters())
    assert summary_core["grad_norm"] > 0.0
    assert summary_core["all_finite"] is True

    model_a.zero_grad(set_to_none=True)

    # 2. Risk loss
    _, risk_logits = model_a(input_ids, prompt_end_indices=prompt_end_indices)
    loss_risk = risk_classification_loss(risk_logits, risk_labels)
    loss_risk.backward()

    summary_risk = gradient_summary(model_a.risk_head_parameters())
    assert summary_risk["grad_norm"] > 0.0
    assert summary_risk["all_finite"] is True


def test_tied_embedding_behavior():
    """Test 14: LM projection shares storage with token embedding and belongs exclusively to theta_C."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    # Verify tied embedding storage
    assert model.embedding.weight.data_ptr() is not None

    theta_C_ids = {id(p) for p in model.theta_C}
    theta_N_ids = {id(p) for p in model.theta_N}

    assert id(model.embedding.weight) in theta_C_ids
    assert id(model.embedding.weight) not in theta_N_ids


def test_freeze_unfreeze_reversibility():
    """Test 15: Freezing and unfreezing theta_C preserves parameter values and grad flags reversibly."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    snap_C = snapshot_parameters(model.theta_C)
    snap_N = snapshot_parameters(model.theta_N)

    # Freeze capability
    set_requires_grad(model.theta_C, False)
    assert all(not p.requires_grad for p in model.theta_C)
    assert all(p.requires_grad for p in model.theta_N)

    # Restore capability
    set_requires_grad(model.theta_C, True)
    assert all(p.requires_grad for p in model.theta_C)
    assert all(p.requires_grad for p in model.theta_N)

    # Verify bit-identical values
    assert parameters_bit_identical(snap_C, model.theta_C)
    assert parameters_bit_identical(snap_N, model.theta_N)


def test_stale_gradient_protection():
    """Test 16: Alternating mode switches with zero_grad(set_to_none=True) prevent stale gradient leakage."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)
    input_ids, prompt_end_indices, risk_labels = get_deterministic_synthetic_batch(config)

    # Step 1: LM forward and backward
    set_requires_grad(model.theta_C, True)
    set_requires_grad(model.theta_N, False)
    logits, _ = model(input_ids, mode="lm")
    loss_lm = causal_lm_loss(logits, input_ids)
    loss_lm.backward()

    # Clear gradients
    model.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())

    # Step 2: Normative forward and backward
    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)
    logits, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
    loss_norm = risk_classification_loss(risk_logits, risk_labels) + 1.0 * safe_generation_loss(
        logits, input_ids, prompt_end_indices
    )
    loss_norm.backward()

    summary_C = gradient_summary(model.theta_C)
    summary_N = gradient_summary(model.theta_N)

    # No residual LM gradients masquerading on theta_C
    assert summary_C["num_grad_nonzero"] == 0
    assert summary_N["num_grad_nonzero"] > 0

    # Clear gradients again
    model.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())
