"""Transparent gradient inspection and parameter management utilities."""

from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


def gradient_summary(parameters: Iterable[nn.Parameter]) -> Dict[str, Any]:
    """Computes transparent diagnostic summary over an iterable of parameters.

    Returns:
        num_params: total number of parameters in the group
        num_grad_none: number of parameters with grad is None
        num_grad_zero: number of parameters with grad not None and all elements exactly zero
        num_grad_nonzero: number of parameters with grad not None and at least one non-zero element
        grad_norm: total L2 norm of all gradient tensors
        max_abs_grad: maximum absolute value across all gradients
        all_finite: boolean indicating whether all computed gradients are finite (no NaN / Inf)
    """
    param_list = list(parameters)
    num_params = len(param_list)
    num_grad_none = 0
    num_grad_zero = 0
    num_grad_nonzero = 0

    squared_norm_sum = 0.0
    max_abs = 0.0
    all_finite = True

    for p in param_list:
        if p.grad is None:
            num_grad_none += 1
        else:
            grad = p.grad
            if not torch.all(torch.isfinite(grad)):
                all_finite = False

            abs_max = torch.max(torch.abs(grad)).item()
            if abs_max > max_abs:
                max_abs = abs_max

            if torch.all(grad == 0):
                num_grad_zero += 1
            else:
                num_grad_nonzero += 1
                squared_norm_sum += torch.sum(grad.float() ** 2).item()

    total_grad_norm = squared_norm_sum ** 0.5

    return {
        "num_params": num_params,
        "num_grad_none": num_grad_none,
        "num_grad_zero": num_grad_zero,
        "num_grad_nonzero": num_grad_nonzero,
        "grad_norm": total_grad_norm,
        "max_abs_grad": max_abs,
        "all_finite": all_finite,
    }


def set_requires_grad(parameters: Iterable[nn.Parameter], requires_grad: bool) -> None:
    """Sets requires_grad flag on an iterable of parameters."""
    for p in parameters:
        p.requires_grad_(requires_grad)


def snapshot_parameters(parameters: Iterable[nn.Parameter]) -> List[torch.Tensor]:
    """Creates cloned, detached snapshots of parameter tensors for bit-identical verification."""
    return [p.detach().clone() for p in parameters]


def parameters_bit_identical(snapshot: List[torch.Tensor], parameters: Iterable[nn.Parameter]) -> bool:
    """Verifies that all parameters are bit-for-bit identical to their snapshot."""
    for snap, p in zip(snapshot, parameters):
        if not torch.equal(snap, p.data):
            return False
    return True


def count_changed_tensors(snapshot: List[torch.Tensor], parameters: Iterable[nn.Parameter]) -> Tuple[int, int]:
    """Counts the number of changed and unchanged parameter tensors compared to a snapshot.

    Returns:
        (num_changed, num_unchanged)
    """
    changed = 0
    unchanged = 0
    for snap, p in zip(snapshot, parameters):
        if torch.equal(snap, p.data):
            unchanged += 1
        else:
            changed += 1
    return changed, unchanged
