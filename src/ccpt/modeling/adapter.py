"""Model D: Parameter-matched Frozen-Backbone Safety Adapter Control for CCPT.

Standard decoder-only capability backbone with residual bottleneck adapters
(Houlsby-style dual adapter: post-attention and post-MLP)
matched to Model C theta_N parameter budget (~2.75M trainable parameters).
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccpt.config import BaselineConfig
from ccpt.modeling.layers import (
    CausalSelfAttention,
    RMSNorm,
    SwiGLU,
)


class ResidualBottleneckAdapter(nn.Module):
    """Standard residual bottleneck adapter with RMSNorm, down-projection, SiLU, and up-projection."""

    def __init__(self, d_model: int = 512, d_mid: int = 336, eps: float = 1e-6, init_std: float = 1e-3) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model, eps=eps)
        self.down_proj = nn.Linear(d_model, d_mid, bias=False)
        self.act = nn.SiLU()
        self.up_proj = nn.Linear(d_mid, d_model, bias=False)

        nn.init.normal_(self.down_proj.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=init_std)
        nn.init.ones_(self.norm.weight)

    def forward(self, x: torch.Tensor, adapter_scale: float = 1.0) -> torch.Tensor:
        if adapter_scale == 0.0:
            return x
        h = self.act(self.down_proj(self.norm(x)))
        return x + adapter_scale * self.up_proj(h)


class FrozenBackboneAdapterBlock(nn.Module):
    """Transformer block wrapping frozen capability attention/FFN with trainable Houlsby adapters."""

    def __init__(self, config: BaselineConfig, d_mid: int = 336) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )
        self.attn_adapter = ResidualBottleneckAdapter(
            d_model=config.d_model,
            d_mid=d_mid,
            eps=config.rms_norm_eps,
            init_std=config.init_std,
        )

        self.mlp_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(d_model=config.d_model, d_ff=config.d_ff)
        self.mlp_adapter = ResidualBottleneckAdapter(
            d_model=config.d_model,
            d_mid=d_mid,
            eps=config.rms_norm_eps,
            init_std=config.init_std,
        )

    def forward(
        self,
        x: torch.Tensor,
        adapter_scale: float = 1.0,
    ) -> torch.Tensor:
        # Backbone Self-Attention + Trainable Attention Adapter
        u = x + self.attn(self.attn_norm(x))
        u = self.attn_adapter(u, adapter_scale=adapter_scale)

        # Backbone Feed-Forward + Trainable MLP Adapter
        out = u + self.mlp(self.mlp_norm(u))
        out = self.mlp_adapter(out, adapter_scale=adapter_scale)
        return out


class FrozenBackboneAdapterModel(nn.Module):
    """Model D: Conventional language model with frozen backbone and trainable safety adapters.

    Tied embeddings match Model A / Model C capability backbone.
    Trainable safety parameters match Model C theta_N (~2.75M).
    """

    def __init__(self, config: BaselineConfig, d_mid: int = 336) -> None:
        super().__init__()
        self.config = config
        self.d_mid = d_mid

        # Backbone (Frozen during safety training) - Tied Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            FrozenBackboneAdapterBlock(config, d_mid=d_mid) for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Trainable Prompt-Boundary Risk Head (Trainable during safety training)
        self.risk_head = nn.Linear(config.d_model, 1, bias=False)

        self._init_backbone_weights()

    def _init_backbone_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    @property
    def backbone_parameters(self) -> List[nn.Parameter]:
        """Returns all frozen backbone parameters (embeddings, attention, FFN, final_norm)."""
        params = []
        params.extend(self.embedding.parameters())
        for layer in self.layers:
            params.extend(layer.attn_norm.parameters())
            params.extend(layer.attn.parameters())
            params.extend(layer.mlp_norm.parameters())
            params.extend(layer.mlp.parameters())
        params.extend(self.final_norm.parameters())
        return params

    @property
    def safety_parameters(self) -> List[nn.Parameter]:
        """Returns all trainable safety parameters (adapters across layers + risk head)."""
        params = []
        for layer in self.layers:
            params.extend(layer.attn_adapter.parameters())
            params.extend(layer.mlp_adapter.parameters())
        params.extend(self.risk_head.parameters())
        return params

    def freeze_backbone(self) -> None:
        """Freezes all backbone parameters and enables gradients only for safety adapters."""
        for p in self.backbone_parameters:
            p.requires_grad = False
        for p in self.safety_parameters:
            p.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        prompt_end_indices: Optional[torch.Tensor] = None,
        adapter_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through frozen backbone and adapters.

        Returns:
            logits: Next token logits [batch_size, seq_len, vocab_size]
            risk_logits: Optional prompt-boundary risk classification logits [batch_size]
        """
        B, T = input_ids.shape
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, adapter_scale=adapter_scale)

        normed_x = self.final_norm(x)
        logits = F.linear(normed_x, self.embedding.weight)

        risk_logits: Optional[torch.Tensor] = None
        if prompt_end_indices is not None:
            batch_indices = torch.arange(B, device=input_ids.device)
            boundary_repr = normed_x[batch_indices, prompt_end_indices]
            risk_logits = self.risk_head(boundary_repr).squeeze(-1)

        return logits, risk_logits
