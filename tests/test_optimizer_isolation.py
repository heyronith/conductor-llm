"""Tests asserting strict, bit-for-bit optimizer parameter update isolation."""

import torch
from torch.optim import AdamW

from ccpt.config import get_micro_dual_stream_config
from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.training.gradients import (
    count_changed_tensors,
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
    torch.manual_seed(1234)
    B, T = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    prompt_end_indices = torch.tensor([2, 4])
    risk_labels = torch.tensor([1, 0])
    return input_ids, prompt_end_indices, risk_labels


def test_lm_optimizer_step_isolation():
    """Test 2: An LM optimizer step updates theta_C while theta_N remains bit-for-bit identical."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    set_requires_grad(model.theta_C, True)
    set_requires_grad(model.theta_N, False)

    snap_C = snapshot_parameters(model.theta_C)
    snap_N = snapshot_parameters(model.theta_N)

    optimizer_C = AdamW(model.theta_C, lr=1e-3, weight_decay=0.0)

    input_ids, _, _ = get_deterministic_synthetic_batch(config)
    logits, _ = model(input_ids, mode="lm")
    loss = causal_lm_loss(logits, input_ids)

    optimizer_C.zero_grad(set_to_none=True)
    loss.backward()
    optimizer_C.step()

    # Verify capability parameters changed
    c_changed, c_unchanged = count_changed_tensors(snap_C, model.theta_C)
    assert c_changed > 0, "No capability parameters changed after LM optimizer step"

    # Verify normative parameters are strictly bit-for-bit identical
    n_changed, n_unchanged = count_changed_tensors(snap_N, model.theta_N)
    assert n_changed == 0, f"{n_changed} normative tensors were mutated during LM optimizer step"
    assert parameters_bit_identical(snap_N, model.theta_N) is True


def test_normative_optimizer_step_isolation():
    """Test 5: A normative optimizer step updates theta_N while theta_C remains bit-for-bit identical."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)

    snap_C = snapshot_parameters(model.theta_C)
    snap_N = snapshot_parameters(model.theta_N)

    optimizer_N = AdamW(model.theta_N, lr=1e-3, weight_decay=0.0)

    input_ids, prompt_end_indices, risk_labels = get_deterministic_synthetic_batch(config)
    logits, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
    loss = risk_classification_loss(risk_logits, risk_labels) + 1.0 * safe_generation_loss(
        logits, input_ids, prompt_end_indices
    )

    optimizer_N.zero_grad(set_to_none=True)
    loss.backward()
    optimizer_N.step()

    # Verify capability parameters remain strictly bit-for-bit identical
    c_changed, c_unchanged = count_changed_tensors(snap_C, model.theta_C)
    assert c_changed == 0, f"{c_changed} capability tensors were mutated during normative optimizer step"
    assert parameters_bit_identical(snap_C, model.theta_C) is True

    # Verify normative parameters changed
    n_changed, n_unchanged = count_changed_tensors(snap_N, model.theta_N)
    assert n_changed > 0, "No normative parameters changed after normative optimizer step"
