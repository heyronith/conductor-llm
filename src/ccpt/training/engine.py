"""Training engine, parameter snapshots, optimization loops, and evaluation routines."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ccpt.config import DualStreamConfig
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
from ccpt.training.losses import compute_causal_lm_loss, compute_risk_loss, compute_safe_generation_loss


def create_identical_dual_stream_models(
    config: DualStreamConfig,
    seed: int = 20260821,
) -> Tuple[JointTrainingDualStreamModel, CCPTDualStreamModel]:
    """Instantiates Model B and Model C with bit-for-bit identical initial parameters."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    canonical = CCPTDualStreamModel(config)
    initial_state = {k: v.clone() for k, v in canonical.state_dict().items()}

    model_b = JointTrainingDualStreamModel(config)
    model_b.load_state_dict(initial_state)

    model_c = CCPTDualStreamModel(config)
    model_c.load_state_dict(initial_state)

    # Verify bit-for-bit equality
    for (k_b, p_b), (k_c, p_c) in zip(model_b.state_dict().items(), model_c.state_dict().items()):
        assert k_b == k_c, f"Key mismatch: {k_b} vs {k_c}"
        assert torch.equal(p_b, p_c), f"Initial parameter mismatch for {k_b}"

    return model_b, model_c


def snapshot_parameters(params: List[nn.Parameter]) -> List[torch.Tensor]:
    """Creates detached, cloned snapshots of a list of parameters."""
    return [p.detach().clone() for p in params]


def assert_parameters_equal(params: List[nn.Parameter], snapshots: List[torch.Tensor]) -> None:
    """Asserts that parameters are bit-for-bit identical to their snapshots."""
    assert len(params) == len(snapshots), f"Length mismatch: {len(params)} vs {len(snapshots)}"
    for idx, (p, s) in enumerate(zip(params, snapshots)):
        assert torch.equal(p.data, s), f"Parameter at index {idx} modified!"


def count_changed_parameters(params: List[nn.Parameter], snapshots: List[torch.Tensor]) -> int:
    """Counts how many parameter tensors differ from their snapshots."""
    assert len(params) == len(snapshots), f"Length mismatch: {len(params)} vs {len(snapshots)}"
    changed = 0
    for p, s in zip(params, snapshots):
        if not torch.equal(p.data, s):
            changed += 1
    return changed


def clip_and_measure_gradients(parameters: List[nn.Parameter], max_norm: float = 1.0) -> float:
    """Measures total gradient norm before clipping, asserts finiteness, and applies clipping.

    Returns:
        Total gradient norm before clipping.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    # Compute total L2 norm across all parameters
    total_norm = float(torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2).item())

    if torch.isnan(torch.tensor(total_norm)) or torch.isinf(torch.tensor(total_norm)):
        raise FloatingPointError(f"Non-finite gradient norm detected: {total_norm}")

    if max_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    return total_norm


@torch.no_grad()
def evaluate_lm_loss_and_acc(
    model: nn.Module,
    sequences: torch.Tensor,
    mode: str = "controlled",
    controller_scale: float = 1.0,
) -> Tuple[float, float]:
    """Evaluates causal LM cross-entropy loss and next-token prediction accuracy.

    Args:
        model: Model to evaluate.
        sequences: Token tensor of shape [B, T].
        mode: Forward mode for dual-stream models ('controlled' or 'lm').
        controller_scale: Controller ablation scale.

    Returns:
        (mean_loss, token_accuracy)
    """
    model.eval()
    if isinstance(model, CCPTDualStreamModel):
        logits, _ = model(sequences, mode=mode, controller_scale=controller_scale)
    else:
        logits, _ = model(sequences)

    # Next-token prediction targets
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = sequences[:, 1:].contiguous()

    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

    preds = shift_logits.argmax(dim=-1)
    acc = (preds == shift_labels).float().mean().item()

    return float(loss), float(acc)


@torch.no_grad()
def evaluate_risk_loss_and_acc(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_end_indices: torch.Tensor,
    labels: torch.Tensor,
    mode: str = "controlled",
    controller_scale: float = 1.0,
) -> Tuple[float, float]:
    """Evaluates binary risk prediction loss and accuracy.

    Args:
        model: Model to evaluate.
        input_ids: Token tensor [B, T].
        prompt_end_indices: Boundary index tensor [B].
        labels: Float risk labels [B] (1.0 = harmful, 0.0 = benign).
        mode: Forward mode.
        controller_scale: Controller scale.

    Returns:
        (bce_loss, accuracy)
    """
    model.eval()
    if isinstance(model, CCPTDualStreamModel):
        _, risk_logits = model(
            input_ids,
            prompt_end_indices=prompt_end_indices,
            mode=mode,
            controller_scale=controller_scale,
        )
    else:
        _, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices)

    assert risk_logits is not None, "risk_logits must not be None when prompt_end_indices is provided"

    loss = F.binary_cross_entropy_with_logits(risk_logits, labels).item()
    preds = (risk_logits > 0.0).float()
    acc = (preds == labels).float().mean().item()

    return float(loss), float(acc)


@torch.no_grad()
def evaluate_safe_gen_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_end_indices: torch.Tensor,
    risk_labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    mode: str = "controlled",
    controller_scale: float = 1.0,
) -> Tuple[float, float, float]:
    """Evaluates combined normative loss: L_norm = L_risk + 1.0 * L_safe_generation.

    Returns:
        (total_loss, risk_loss, safe_gen_loss)
    """
    model.eval()
    if isinstance(model, CCPTDualStreamModel):
        logits, risk_logits = model(
            input_ids,
            prompt_end_indices=prompt_end_indices,
            mode=mode,
            controller_scale=controller_scale,
        )
    else:
        logits, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices)

    assert risk_logits is not None, "risk_logits must not be None"

    l_risk = compute_risk_loss(risk_logits, risk_labels)
    l_gen = compute_safe_generation_loss(logits, input_ids, prompt_end_indices, attention_mask=attention_mask)
    l_tot = l_risk + l_gen

    return float(l_tot.item()), float(l_risk.item()), float(l_gen.item())


evaluate_safe_generation_loss = evaluate_safe_gen_loss


