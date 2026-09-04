"""Distillation losses and tiny training helpers for successor Task 1."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def continuation_token_mask(
    input_ids: torch.Tensor,
    prompt_end_indices: torch.Tensor,
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    """Boolean mask [B, T-1] selecting positions that predict continuation tokens.

    For causal LM, logit at index t predicts token t+1. We supervise predictions
    of tokens strictly after the prompt boundary (assistant continuation).
    """
    B, T = input_ids.shape
    # position t (0..T-2) predicts token t+1; keep when (t+1) > prompt_end
    pos = torch.arange(T - 1, device=input_ids.device).unsqueeze(0).expand(B, -1)
    ends = prompt_end_indices.unsqueeze(1)
    mask = (pos + 1) > ends
    if pad_id is not None:
        mask = mask & (input_ids[:, 1:] != pad_id)
    return mask


def masked_kl_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(teacher || student) averaged over masked next-token positions."""
    # logits: [B, T, V]; mask: [B, T-1] applies to predictions of tokens 1..T-1
    t = teacher_logits[:, :-1, :] / temperature
    s = student_logits[:, :-1, :] / temperature
    log_p_t = F.log_softmax(t, dim=-1)
    log_p_s = F.log_softmax(s, dim=-1)
    p_t = log_p_t.exp()
    kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)  # [B, T-1]
    kl = kl * mask.to(kl.dtype)
    denom = mask.to(kl.dtype).sum().clamp_min(1.0)
    return kl.sum() / denom


def risk_mse(
    teacher_risk: Optional[torch.Tensor],
    student_risk: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, bool]:
    if teacher_risk is None or student_risk is None:
        z = torch.zeros((), device=teacher_risk.device if teacher_risk is not None else "cpu")
        if student_risk is not None:
            z = z.to(student_risk.device)
        return z, False
    return F.mse_loss(student_risk, teacher_risk.detach()), True


def fit_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    teacher_risk: Optional[torch.Tensor],
    student_risk: Optional[torch.Tensor],
    *,
    risk_weight: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    l_logits = masked_kl_logits(teacher_logits.detach(), student_logits, mask)
    l_risk, has_risk = risk_mse(teacher_risk, student_risk)
    if not has_risk:
        l_risk = l_logits.new_zeros(())
    total = l_logits + (risk_weight * l_risk if has_risk else 0.0)
    return total, {
        "L_logits": float(l_logits.detach().item()),
        "L_risk": float(l_risk.detach().item()) if has_risk else None,
        "has_risk": has_risk,
        "L_fit": float(total.detach().item()),
    }
