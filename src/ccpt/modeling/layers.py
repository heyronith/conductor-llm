"""Core architectural primitives for CCPT and baseline Transformers."""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (bias-free)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Compute in float32 for fp16/bf16 stability, or maintain float64 if input is float64
        calc_dtype = torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype
        x_calc = x.to(calc_dtype)
        variance = x_calc.pow(2).mean(-1, keepdim=True)
        x_normed = x_calc * torch.rsqrt(variance + self.eps)
        return (self.weight.to(calc_dtype) * x_normed).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute table up to max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            # Dynamically compute for lengths exceeding precomputed buffer if needed
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(dtype)
            sin = emb.sin().to(dtype)
        else:
            cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
            sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos.unsqueeze(0).unsqueeze(1), sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors of shape [B, n_heads, T, head_dim]."""
        seq_len = q.shape[2]
        cos, sin = self._get_cos_sin(seq_len, q.device, q.dtype)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention with RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, T, head_dim]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, T, head_dim]
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, T, head_dim]

        q, k = self.rotary_emb(q, k)

        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(attn_out)


class SwiGLU(nn.Module):
    """Genuine 3-matrix SwiGLU MLP (bias-free)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-RMSNorm Residual Transformer Block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 1024,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            dropout=dropout,
        )
        self.mlp_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.mlp = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x + self.attn(self.attn_norm(x))
        out = u + self.mlp(self.mlp_norm(u))
        return out
