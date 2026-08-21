"""Model A: Parameter-matched standard Transformer baseline."""

from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccpt.config import BaselineConfig
from ccpt.modeling.layers import RMSNorm, TransformerBlock


class ParameterMatchedBaselineModel(nn.Module):
    """Standard causal decoder-only Transformer with tied embeddings and auxiliary prompt-boundary risk head."""

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    max_seq_len=config.max_seq_len,
                    rms_norm_eps=config.rms_norm_eps,
                    rope_theta=config.rope_theta,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Auxiliary risk classifier (512 params in linear projection, operating on final normalized hidden state)
        self.baseline_risk_head = nn.Linear(config.d_model, 1, bias=False)

        self._init_weights()

    @property
    def baseline_risk_norm(self) -> RMSNorm:
        """Alias to the final RMSNorm for consistency with risk head specification."""
        return self.final_norm

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        prompt_end_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for Model A.

        Args:
            input_ids: Input token ids of shape [B, T].
            prompt_end_indices: Optional tensor of shape [B] indicating the prompt-boundary token index.

        Returns:
            logits: Next-token logits of shape [B, T, vocab_size].
            risk_logits: Binary classification risk logits of shape [B] if prompt_end_indices is provided, else None.
        """
        B, T = input_ids.shape
        x = self.embedding(input_ids)  # [B, T, d_model]

        for layer in self.layers:
            x = layer(x)

        normed_x = self.final_norm(x)
        logits = F.linear(normed_x, self.embedding.weight)  # [B, T, vocab_size]

        risk_logits: Optional[torch.Tensor] = None
        if prompt_end_indices is not None:
            batch_indices = torch.arange(B, device=input_ids.device)
            boundary_repr = normed_x[batch_indices, prompt_end_indices]  # [B, d_model]
            risk_logits = self.baseline_risk_head(boundary_repr).squeeze(-1)  # [B]

        return logits, risk_logits

    def core_lm_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters belonging to the core language model (excluding auxiliary risk head)."""
        for param in self.embedding.parameters():
            yield param
        for layer in self.layers:
            for param in layer.parameters():
                yield param
        for param in self.final_norm.parameters():
            yield param

    def risk_head_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters belonging exclusively to the auxiliary risk head."""
        for param in self.baseline_risk_head.parameters():
            yield param
