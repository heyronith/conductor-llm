"""Tiny identity-preserving residual repair modules for successor Task 1."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccpt.modeling.layers import RMSNorm


class ResidualBottleneck(nn.Module):
    """R(x) = x + W_up(silu(W_down(RMSNorm(x)))) with W_up zero-initialized."""

    def __init__(self, dim: int, rank: int, eps: float = 1e-6):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.dim = dim
        self.rank = rank
        self.norm = RMSNorm(dim, eps=eps)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(self.norm(x))
        h = F.silu(h)
        return x + self.up(h)


class ObserverRepairAdapter(nn.Module):
    """Residual repairs on p_in and each obs_projection output (d_N space)."""

    def __init__(self, d_N: int, n_obs: int, rank: int, eps: float = 1e-6):
        super().__init__()
        self.p_in_repair = ResidualBottleneck(d_N, rank, eps=eps)
        self.obs_repairs = nn.ModuleList(
            [ResidualBottleneck(d_N, rank, eps=eps) for _ in range(n_obs)]
        )

    def repair_p_in(self, z: torch.Tensor) -> torch.Tensor:
        return self.p_in_repair(z)

    def repair_obs(self, k: int, z: torch.Tensor) -> torch.Tensor:
        return self.obs_repairs[k](z)


class ActuatorRepairAdapter(nn.Module):
    """Additive corrections in pre-bound raw gate/steering space (zero-init)."""

    def __init__(self, d_N: int, d_C: int, n_ctrl: int, rank: int, eps: float = 1e-6):
        super().__init__()
        self.n_ctrl = n_ctrl
        self.gate_norm = nn.ModuleList([RMSNorm(d_N, eps=eps) for _ in range(n_ctrl)])
        self.gate_down = nn.ModuleList([nn.Linear(d_N, rank, bias=False) for _ in range(n_ctrl)])
        self.gate_up = nn.ModuleList([nn.Linear(rank, 1, bias=False) for _ in range(n_ctrl)])
        self.steer_norm_n = nn.ModuleList([RMSNorm(d_N, eps=eps) for _ in range(n_ctrl)])
        self.steer_norm_h = nn.ModuleList([RMSNorm(d_C, eps=eps) for _ in range(n_ctrl)])
        self.steer_down = nn.ModuleList(
            [nn.Linear(d_N + d_C, rank, bias=False) for _ in range(n_ctrl)]
        )
        self.steer_up = nn.ModuleList([nn.Linear(rank, d_C, bias=False) for _ in range(n_ctrl)])
        for up in list(self.gate_up) + list(self.steer_up):
            nn.init.zeros_(up.weight)

    def gate_delta(self, k: int, n: torch.Tensor) -> torch.Tensor:
        h = self.gate_down[k](self.gate_norm[k](n))
        return self.gate_up[k](F.silu(h))

    def steer_delta(self, k: int, n: torch.Tensor, h_cap: torch.Tensor) -> torch.Tensor:
        nn_ = self.steer_norm_n[k](n)
        hh = self.steer_norm_h[k](h_cap)
        h = self.steer_down[k](torch.cat([nn_, hh], dim=-1))
        return self.steer_up[k](F.silu(h))


class GenericResidualRepairControl(nn.Module):
    """Matched-capacity residual adapters on capability states only (no N access)."""

    def __init__(self, d_C: int, n_sites: int, rank: int, eps: float = 1e-6):
        super().__init__()
        self.adapters = nn.ModuleList(
            [ResidualBottleneck(d_C, rank, eps=eps) for _ in range(n_sites)]
        )

    def repair(self, k: int, h: torch.Tensor) -> torch.Tensor:
        return self.adapters[k](h)
