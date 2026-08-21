"""Exact loss functions for causal language modeling, risk classification, and safe generation."""

from typing import Optional

import torch
import torch.nn.functional as F


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Standard shifted next-token cross-entropy loss for causal language modeling.

    Args:
        logits: Output logits of shape [B, T, vocab_size].
        input_ids: Ground-truth token ids of shape [B, T].

    Returns:
        Scalar cross-entropy loss.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def risk_classification_loss(risk_logits: torch.Tensor, risk_labels: torch.Tensor) -> torch.Tensor:
    """Binary classification loss for safety risk prediction.

    Args:
        risk_logits: Predicted risk logits of shape [B].
        risk_labels: Ground-truth binary labels (0 = benign, 1 = harmful) of shape [B].

    Returns:
        Scalar binary cross-entropy with logits loss.
    """
    return F.binary_cross_entropy_with_logits(risk_logits, risk_labels.float())


def safe_generation_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_end_indices: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked shifted next-token cross-entropy for safe-continuation training.

    Only prediction positions whose target token is strictly after the prompt boundary
    (i.e., target position > prompt_end_index, or equivalently prediction logit position >= prompt_end_index)
    and where attention_mask[target_position] == 1 (if attention_mask is provided)
    are included in the loss computation.

    Args:
        logits: Output logits of shape [B, T, vocab_size].
        input_ids: Ground-truth token ids of shape [B, T].
        prompt_end_indices: Tensor of shape [B] indicating the index of the last prompt token.
        attention_mask: Optional binary attention mask of shape [B, T] (1 = valid token, 0 = padding).

    Returns:
        Scalar masked cross-entropy loss over valid safe continuation tokens.
    """
    B, T, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()  # [B, T - 1, V]
    shift_labels = input_ids[:, 1:].contiguous()   # [B, T - 1]

    # Target position for prediction index p in [0, T-2] is p + 1.
    # Target position must be > prompt_end_index <=> p >= prompt_end_index.
    pos = torch.arange(T - 1, device=input_ids.device).unsqueeze(0)  # [1, T - 1]
    prompt_mask = pos >= prompt_end_indices.unsqueeze(1)  # [B, T - 1]

    if attention_mask is not None:
        target_mask = attention_mask[:, 1:].bool()  # [B, T - 1]
        mask = prompt_mask & target_mask
    else:
        mask = prompt_mask

    loss_unreduced = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
    ).view(B, T - 1)

    masked_loss = loss_unreduced * mask.float()
    num_valid = mask.sum().clamp(min=1.0)
    return masked_loss.sum() / num_valid


def token_weighted_continuation_nll_and_count(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_end_indices: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[float, int]:
    """Computes total continuation negative log-likelihood and valid continuation token count.

    Args:
        logits: Output logits of shape [B, T, vocab_size].
        input_ids: Ground-truth token ids of shape [B, T].
        prompt_end_indices: Tensor of shape [B] indicating the index of the last prompt token.
        attention_mask: Optional binary attention mask of shape [B, T].

    Returns:
        Tuple of (total_continuation_nll: float, total_continuation_tokens: int).
    """
    B, T, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()  # [B, T - 1, V]
    shift_labels = input_ids[:, 1:].contiguous()   # [B, T - 1]

    pos = torch.arange(T - 1, device=input_ids.device).unsqueeze(0)  # [1, T - 1]
    prompt_mask = pos >= prompt_end_indices.unsqueeze(1)  # [B, T - 1]

    if attention_mask is not None:
        target_mask = attention_mask[:, 1:].bool()
        mask = prompt_mask & target_mask
    else:
        mask = prompt_mask

    loss_unreduced = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
    ).view(B, T - 1)

    masked_nll = float((loss_unreduced * mask.float()).sum().item())
    valid_tokens = int(mask.sum().item())
    return masked_nll, valid_tokens


# Aliases for explicit semantic naming
compute_causal_lm_loss = causal_lm_loss
compute_risk_loss = risk_classification_loss
compute_risk_classification_loss = risk_classification_loss
compute_safe_generation_loss = safe_generation_loss
token_weighted_continuation_loss = token_weighted_continuation_nll_and_count



