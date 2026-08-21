"""Metrics logging and diagnostic helpers for CCPT training."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from ccpt.modeling.dual_stream import CCPTDualStreamModel


def compute_gate_diagnostics(gates: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Computes distribution statistics for multiplicative gates g_l."""
    if not gates:
        return {}

    all_gates = torch.cat([g.flatten() for g in gates.values()])
    mean_val = float(all_gates.mean().item())
    min_val = float(all_gates.min().item())
    max_val = float(all_gates.max().item())
    abs_dev = float((all_gates - 1.0).abs().mean().item())

    # Bounds are 0.9 (1.0 - alpha) and 1.1 (1.0 + alpha) for alpha=0.1
    # Near bound defined as within 0.001 of bound
    near_lower = float((all_gates <= 0.901).float().mean().item())
    near_upper = float((all_gates >= 1.099).float().mean().item())

    return {
        "gate_mean": mean_val,
        "gate_min": min_val,
        "gate_max": max_val,
        "gate_abs_deviation_mean": abs_dev,
        "gate_near_lower_bound_fraction": near_lower,
        "gate_near_upper_bound_fraction": near_upper,
    }


def compute_steering_diagnostics(steering: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Computes distribution statistics for additive steering vectors s_l."""
    if not steering:
        return {}

    all_steering = torch.cat([s.reshape(-1, s.shape[-1]) for s in steering.values()], dim=0)  # [N, d_C]
    l2_norms = torch.norm(all_steering, p=2, dim=-1)  # [N]
    abs_vals = all_steering.abs()

    l2_mean = float(l2_norms.mean().item())
    l2_max = float(l2_norms.max().item())
    abs_max = float(abs_vals.max().item())
    saturate_fraction = float((abs_vals > 0.99).float().mean().item())

    return {
        "steering_l2_mean": l2_mean,
        "steering_l2_max": l2_max,
        "steering_abs_max": abs_max,
        "steering_saturate_fraction": saturate_fraction,
    }


def compute_gradient_group_norms(model: nn.Module) -> Dict[str, float]:
    """Computes gradient norms for distinct parameter sub-groups before clipping."""
    metrics: Dict[str, float] = {}

    if isinstance(model, CCPTDualStreamModel):
        # theta_C
        c_grads = [p.grad for p in model.theta_C if p.grad is not None]
        if c_grads:
            metrics["theta_C_grad_norm"] = float(torch.norm(torch.stack([torch.norm(g) for g in c_grads])).item())
        else:
            metrics["theta_C_grad_norm"] = 0.0

        # theta_N
        n_grads = [p.grad for p in model.theta_N if p.grad is not None]
        if n_grads:
            metrics["theta_N_grad_norm"] = float(torch.norm(torch.stack([torch.norm(g) for g in n_grads])).item())
        else:
            metrics["theta_N_grad_norm"] = 0.0

        # Controller projections (gate and steering projections)
        ctrl_params: List[nn.Parameter] = []
        for g_proj in model.gate_projections:
            ctrl_params.extend(g_proj.parameters())
        for s_proj in model.steering_projections:
            ctrl_params.extend(s_proj.parameters())

        ctrl_grads = [p.grad for p in ctrl_params if p.grad is not None]
        if ctrl_grads:
            metrics["controller_grad_norm"] = float(torch.norm(torch.stack([torch.norm(g) for g in ctrl_grads])).item())
        else:
            metrics["controller_grad_norm"] = 0.0

        # Deeper normative layers (Transformer blocks and observation projections)
        deep_n_params: List[nn.Parameter] = []
        deep_n_params.extend(model.p_in.parameters())
        for obs_proj in model.obs_projections:
            deep_n_params.extend(obs_proj.parameters())
        for n_layer in model.normative_layers:
            deep_n_params.extend(n_layer.parameters())

        deep_grads = [p.grad for p in deep_n_params if p.grad is not None]
        if deep_grads:
            metrics["deep_normative_grad_norm"] = float(torch.norm(torch.stack([torch.norm(g) for g in deep_grads])).item())
        else:
            metrics["deep_normative_grad_norm"] = 0.0

    return metrics


class MetricLogger:
    """Logs training metrics to JSON Lines format and tracks real-time progress."""

    def __init__(self, log_path: Union[str, Path]):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def log(self, record: Dict[str, Any]) -> None:
        """Appends a record to memory and writes it to disk."""
        # Sanity assertion: no NaN or Inf allowed in any logged float metric
        for k, v in record.items():
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    raise ValueError(f"Non-finite metric encountered in logger for key '{k}': {v}")

        self.records.append(record)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
