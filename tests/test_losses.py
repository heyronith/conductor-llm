"""Tests verifying mathematical correctness and masking boundary logic of loss functions."""

import torch
import torch.nn.functional as F

from ccpt.training.losses import (
    causal_lm_loss,
    risk_classification_loss,
    safe_generation_loss,
)


def test_causal_lm_loss_computation():
    """Verify standard next-token causal language modeling loss."""
    B, T, V = 2, 4, 10
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))

    loss = causal_lm_loss(logits, input_ids)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    # Manual shifted cross entropy check
    manual_loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), input_ids[:, 1:].reshape(-1))
    assert torch.allclose(loss, manual_loss)


def test_risk_classification_loss_computation():
    """Verify binary risk classification loss."""
    risk_logits = torch.tensor([2.0, -1.5, 0.0])
    risk_labels = torch.tensor([1, 0, 1])

    loss = risk_classification_loss(risk_logits, risk_labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    manual_loss = F.binary_cross_entropy_with_logits(risk_logits, risk_labels.float())
    assert torch.allclose(loss, manual_loss)


def test_safe_generation_loss_masking_exact_positions():
    """Verify exact token-by-token masking for safe-generation loss.

    Example:
    Tokens: [P0, P1, P2, S3, S4, S5, S6, S7] (length 8)
    Prompt boundary: prompt_end_index = 2 (last prompt token is P2)
    Safe continuation tokens: S3, S4, S5, S6, S7 (positions 3, 4, 5, 6, 7)
    Corresponding prediction logits: positions 2, 3, 4, 5, 6 (predicting targets 3, 4, 5, 6, 7)
    Prompt predictions to exclude: position 0 (predicts 1) and position 1 (predicts 2).
    """
    B, T, V = 1, 8, 16
    logits = torch.randn(B, T, V, requires_grad=True)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    prompt_end = torch.tensor([2])  # Prompt ends at index 2 (token value 3)

    loss = safe_generation_loss(logits, input_ids, prompt_end)

    # Compute manual loss on targets 3, 4, 5, 6, 7 from logits 2, 3, 4, 5, 6
    target_positions = [3, 4, 5, 6, 7]
    logit_positions = [2, 3, 4, 5, 6]

    manual_loss_terms = [
        F.cross_entropy(logits[0, lp : lp + 1, :], input_ids[0, tp : tp + 1])
        for lp, tp in zip(logit_positions, target_positions)
    ]
    expected_loss = sum(manual_loss_terms) / len(manual_loss_terms)

    assert torch.allclose(loss, expected_loss, atol=1e-6)

    # Gradient check: gradients with respect to prompt-only logits (positions 0, 1) and final unused logit (position 7) must be 0
    loss.backward()
    assert torch.all(logits.grad[0, 0, :] == 0.0)
    assert torch.all(logits.grad[0, 1, :] == 0.0)
    assert torch.all(logits.grad[0, 7, :] == 0.0)

    # Gradients with respect to active safe prediction positions (2, 3, 4, 5, 6) must be non-zero
    for lp in logit_positions:
        assert torch.any(logits.grad[0, lp, :] != 0.0)


def test_safe_generation_loss_variable_prompt_boundaries():
    """Verify safe generation loss with variable prompt boundaries across a batch."""
    B, T, V = 2, 8, 16
    logits = torch.randn(B, T, V, requires_grad=True)
    input_ids = torch.randint(0, V, (B, T))
    # Sample 0: prompt ends at 0 (all targets 1..7 are safe continuations)
    # Sample 1: prompt ends at 6 (penultimate token; only target 7 is safe continuation)
    prompt_end_indices = torch.tensor([0, 6])

    loss = safe_generation_loss(logits, input_ids, prompt_end_indices)
    assert torch.isfinite(loss)

    loss.backward()

    # For Sample 0 (prompt_end = 0): logit 0 (predicts 1) must have non-zero gradient
    assert torch.any(logits.grad[0, 0, :] != 0.0)

    # For Sample 1 (prompt_end = 6): logits 0..5 predict prompt tokens and must have exactly 0 gradient
    for lp in range(6):
        assert torch.all(logits.grad[1, lp, :] == 0.0)
    # Logit 6 (predicts target 7) must have non-zero gradient
    assert torch.any(logits.grad[1, 6, :] != 0.0)
