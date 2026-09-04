"""Adaptive-interface wrapper around a frozen CCPTDualStreamModel."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.successor.adapters import (
    ActuatorRepairAdapter,
    GenericResidualRepairControl,
    ObserverRepairAdapter,
)


class RepairVariant(str, Enum):
    NONE = "none"
    OBSERVER = "observer"
    ACTUATOR = "actuator"
    COMBINED = "observer_plus_actuator"
    GENERIC = "matched_generic"


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def hash_existing_parameters(module: nn.Module) -> str:
    """Canonical hash over every parameter tensor in ``module`` (sorted by name)."""
    hasher = hashlib.sha256()
    for name, p in sorted(module.named_parameters(), key=lambda kv: kv[0]):
        t = p.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(t.dtype).encode("utf-8"))
        hasher.update(str(tuple(t.shape)).encode("utf-8"))
        hasher.update(t.view(torch.uint8).numpy())
    return hasher.hexdigest()


def _bottleneck_params(dim: int, rank: int) -> int:
    # RMSNorm(dim) + Linear(dim,rank) + Linear(rank,dim)
    return dim + dim * rank + rank * dim


def observer_param_count(d_N: int, n_obs: int, rank: int) -> int:
    return (1 + n_obs) * _bottleneck_params(d_N, rank)


def actuator_param_count(d_N: int, d_C: int, n_ctrl: int, rank: int) -> int:
    # per site: gate norm+down+up + steer norms+down+up
    gate = d_N + d_N * rank + rank * 1
    steer = d_N + d_C + (d_N + d_C) * rank + rank * d_C
    return n_ctrl * (gate + steer)


def generic_param_count(d_C: int, n_sites: int, rank: int) -> int:
    return n_sites * _bottleneck_params(d_C, rank)


def match_generic_rank(
    *,
    d_C: int,
    n_sites: int,
    target_params: int,
    max_rank: int = 512,
) -> Tuple[int, int, float]:
    """Choose generic bottleneck rank minimizing |params - target| / target."""
    best_r, best_n, best_err = 1, generic_param_count(d_C, n_sites, 1), float("inf")
    for r in range(1, max_rank + 1):
        n = generic_param_count(d_C, n_sites, r)
        err = abs(n - target_params) / max(target_params, 1)
        if err < best_err:
            best_r, best_n, best_err = r, n, err
    return best_r, best_n, best_err


class AdaptiveInterfaceWrapper(nn.Module):
    """Wraps a frozen CCPT model with optional observer/actuator/generic repairs.

    Historical CCPT equations are preserved; repairs are additive residual modules.
    Actuator corrections are applied in raw pre-bound space and then multiplied by
    ``controller_scale`` so ``controller_scale=0`` reproduces historical off semantics.
    """

    def __init__(
        self,
        base: CCPTDualStreamModel,
        variant: RepairVariant,
        *,
        observer_rank: int = 32,
        actuator_rank: int = 32,
        generic_rank: Optional[int] = None,
    ):
        super().__init__()
        if not isinstance(base, CCPTDualStreamModel):
            raise TypeError("base must be CCPTDualStreamModel")
        self.base = base
        self.variant = variant if isinstance(variant, RepairVariant) else RepairVariant(variant)
        freeze_module(self.base)

        cfg = base.config
        n_ctrl = len(cfg.controlled_layers)
        self.observer: Optional[ObserverRepairAdapter] = None
        self.actuator: Optional[ActuatorRepairAdapter] = None
        self.generic: Optional[GenericResidualRepairControl] = None

        if self.variant in (RepairVariant.OBSERVER, RepairVariant.COMBINED):
            self.observer = ObserverRepairAdapter(cfg.d_N, n_ctrl, observer_rank, eps=cfg.rms_norm_eps)
        if self.variant in (RepairVariant.ACTUATOR, RepairVariant.COMBINED):
            self.actuator = ActuatorRepairAdapter(
                cfg.d_N, cfg.d_C, n_ctrl, actuator_rank, eps=cfg.rms_norm_eps
            )
        if self.variant == RepairVariant.GENERIC:
            if generic_rank is None:
                raise ValueError("generic_rank required for GENERIC variant")
            self.generic = GenericResidualRepairControl(
                cfg.d_C, n_ctrl, generic_rank, eps=cfg.rms_norm_eps
            )

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for mod in (self.observer, self.actuator, self.generic):
            if mod is not None:
                params.extend(list(mod.parameters()))
        return params

    def trainable_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        out: List[Tuple[str, nn.Parameter]] = []
        for prefix, mod in (
            ("observer", self.observer),
            ("actuator", self.actuator),
            ("generic", self.generic),
        ):
            if mod is None:
                continue
            for name, p in mod.named_parameters():
                out.append((f"{prefix}.{name}", p))
        return out

    def assert_optimizer_owns_only_repairs(self, optimizer: torch.optim.Optimizer) -> None:
        opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        repair_ids = {id(p) for p in self.trainable_parameters()}
        base_ids = {id(p) for p in self.base.parameters()}
        if opt_ids != repair_ids:
            raise RuntimeError("Optimizer parameter set must equal repair parameters exactly")
        if opt_ids & base_ids:
            raise RuntimeError("Historical base parameters entered the optimizer")

    def forward(
        self,
        input_ids: torch.Tensor,
        prompt_end_indices: Optional[torch.Tensor] = None,
        mode: str = "controlled",
        controller_scale: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]],
    ]:
        base = self.base
        if mode == "lm":
            return base.forward(
                input_ids,
                prompt_end_indices=prompt_end_indices,
                mode="lm",
                controller_scale=controller_scale,
                return_diagnostics=return_diagnostics,
            )

        assert mode == "controlled"
        B, T = input_ids.shape
        cfg = base.config

        c_0 = base.embedding(input_ids)
        n = base.p_in(c_0.detach())
        if self.observer is not None:
            n = self.observer.repair_p_in(n)

        c = c_0
        diag_gates: Dict[str, torch.Tensor] = {}
        diag_steering: Dict[str, torch.Tensor] = {}
        diag_normative: Dict[str, torch.Tensor] = {}

        for l_idx, cap_layer in enumerate(base.capability_layers, start=1):
            prev_c = c
            c_tilde = cap_layer(prev_c)

            if l_idx in cfg.controlled_layers:
                k = cfg.controlled_layers.index(l_idx)

                if self.generic is not None:
                    c_tilde = self.generic.repair(k, c_tilde)

                obs = base.obs_projections[k](c_tilde.detach())
                if self.observer is not None:
                    obs = self.observer.repair_obs(k, obs)
                n_in = n + obs
                n = base.normative_layers[k](n_in)

                g_raw = base.gate_projections[k](n)
                s_raw = base.steering_projections[k](n)
                if self.actuator is not None:
                    g_raw = g_raw + self.actuator.gate_delta(k, n)
                    s_raw = s_raw + self.actuator.steer_delta(k, n, c_tilde.detach())

                g_raw_scaled = cfg.alpha * torch.tanh(g_raw)
                g_l = 1.0 + controller_scale * g_raw_scaled
                s_raw_scaled = cfg.beta * torch.tanh(s_raw)
                s_l = controller_scale * s_raw_scaled

                c = prev_c + g_l * (c_tilde - prev_c) + s_l

                if return_diagnostics:
                    diag_gates[f"layer_{l_idx}"] = g_l.detach()
                    diag_steering[f"layer_{l_idx}"] = s_l.detach()
                    diag_normative[f"normative_block_{k+1}"] = n.detach()
            else:
                c = c_tilde

        logits = F.linear(base.capability_final_norm(c), base.embedding.weight)

        risk_logits: Optional[torch.Tensor] = None
        if prompt_end_indices is not None:
            normed_n = base.normative_final_norm(n)
            batch_indices = torch.arange(B, device=input_ids.device)
            boundary_repr = normed_n[batch_indices, prompt_end_indices]
            risk_logits = base.risk_head(boundary_repr).squeeze(-1)

        if return_diagnostics:
            return logits, risk_logits, {
                "gates": diag_gates,
                "steering": diag_steering,
                "normative_states": diag_normative,
            }
        return logits, risk_logits


def build_variant_bundle(
    base: CCPTDualStreamModel,
    *,
    observer_rank: int = 32,
    actuator_rank: int = 32,
) -> Dict[str, Any]:
    """Construct all trainable variants with matched generic rank."""
    cfg = base.config
    n_ctrl = len(cfg.controlled_layers)
    obs_n = observer_param_count(cfg.d_N, n_ctrl, observer_rank)
    act_n = actuator_param_count(cfg.d_N, cfg.d_C, n_ctrl, actuator_rank)
    combined_n = obs_n + act_n
    g_rank, g_n, g_err = match_generic_rank(d_C=cfg.d_C, n_sites=n_ctrl, target_params=combined_n)
    base_n = count_parameters(base)
    return {
        "base_parameters": base_n,
        "observer_parameters": obs_n,
        "actuator_parameters": act_n,
        "combined_parameters": combined_n,
        "generic_parameters": g_n,
        "generic_rank": g_rank,
        "combined_percent_of_base": 100.0 * combined_n / base_n,
        "generic_match_error_percent": 100.0 * g_err,
        "observer_rank": observer_rank,
        "actuator_rank": actuator_rank,
        "within_1pct_budget": combined_n <= 0.01 * base_n,
        "generic_within_1pct_match": g_err <= 0.01,
    }
