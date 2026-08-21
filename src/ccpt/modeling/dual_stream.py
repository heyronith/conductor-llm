"""Model B (Joint-Training Dual-Stream) and Model C (CCPT Protected Dual-Stream)."""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccpt.config import DualStreamConfig
from ccpt.modeling.layers import RMSNorm, TransformerBlock


class CCPTDualStreamModel(nn.Module):
    """Constitutional Control-Plane Transformer (CCPT) Dual-Stream Architecture."""

    def __init__(self, config: DualStreamConfig):
        super().__init__()
        self.config = config

        # --- Capability Pathway (C) ---
        self.embedding = nn.Embedding(config.vocab_size, config.d_C)
        self.capability_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_C,
                    n_heads=config.n_heads_C,
                    d_ff=config.d_ff_C,
                    max_seq_len=config.max_seq_len,
                    rms_norm_eps=config.rms_norm_eps,
                    rope_theta=config.rope_theta,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers_C)
            ]
        )
        self.capability_final_norm = RMSNorm(config.d_C, eps=config.rms_norm_eps)

        # --- Normative Pathway (N) & Controllers ---
        self.p_in = nn.Linear(config.d_C, config.d_N, bias=False)
        self.obs_projections = nn.ModuleList(
            [nn.Linear(config.d_C, config.d_N, bias=False) for _ in config.controlled_layers]
        )
        self.normative_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_N,
                    n_heads=config.n_heads_N,
                    d_ff=config.d_ff_N,
                    max_seq_len=config.max_seq_len,
                    rms_norm_eps=config.rms_norm_eps,
                    rope_theta=config.rope_theta,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers_N)
            ]
        )
        self.gate_projections = nn.ModuleList(
            [nn.Linear(config.d_N, 1, bias=False) for _ in config.controlled_layers]
        )
        self.steering_projections = nn.ModuleList(
            [nn.Linear(config.d_N, config.d_C, bias=False) for _ in config.controlled_layers]
        )
        self.normative_final_norm = RMSNorm(config.d_N, eps=config.rms_norm_eps)
        self.risk_head = nn.Linear(config.d_N, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        # 1. Standard normal initialization for linear and embedding layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

        # 2. Mandatory zero initialization for controller projections
        for gate_proj in self.gate_projections:
            nn.init.zeros_(gate_proj.weight)
        for steer_proj in self.steering_projections:
            nn.init.zeros_(steer_proj.weight)

    @property
    def theta_C(self) -> List[nn.Parameter]:
        """Parameters belonging strictly to the capability pathway."""
        params: List[nn.Parameter] = []
        params.extend(self.embedding.parameters())
        for layer in self.capability_layers:
            params.extend(layer.parameters())
        params.extend(self.capability_final_norm.parameters())
        return params

    @property
    def theta_N(self) -> List[nn.Parameter]:
        """Parameters belonging strictly to the normative pathway and controllers."""
        params: List[nn.Parameter] = []
        params.extend(self.p_in.parameters())
        for obs_proj in self.obs_projections:
            params.extend(obs_proj.parameters())
        for layer in self.normative_layers:
            params.extend(layer.parameters())
        for gate_proj in self.gate_projections:
            params.extend(gate_proj.parameters())
        for steer_proj in self.steering_projections:
            params.extend(steer_proj.parameters())
        params.extend(self.normative_final_norm.parameters())
        params.extend(self.risk_head.parameters())
        return params

    def forward(
        self,
        input_ids: torch.Tensor,
        prompt_end_indices: Optional[torch.Tensor] = None,
        mode: str = "controlled",
        controller_scale: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]]:
        """Forward pass for CCPT.

        Args:
            input_ids: Input token ids of shape [B, T].
            prompt_end_indices: Optional tensor of shape [B] indicating the prompt-boundary token index.
            mode: Forward mode, either 'controlled' (full C->N->controller->C) or 'lm' (pure capability stream).
            controller_scale: Multiplier for controller interventions (1.0 = normal CCPT, 0.0 = ablated steering).
            return_diagnostics: If True, returns detached diagnostics dictionary.

        Returns:
            If return_diagnostics is False:
                (logits, risk_logits)
            If return_diagnostics is True:
                (logits, risk_logits, diagnostics)
        """
        assert mode in ("controlled", "lm"), f"Unsupported mode '{mode}', must be 'controlled' or 'lm'"

        B, T = input_ids.shape

        if mode == "lm":
            # Ordinary capability LM forward path (normative pathway skipped entirely)
            c = self.embedding(input_ids)
            for layer in self.capability_layers:
                c = layer(c)
            logits = F.linear(self.capability_final_norm(c), self.embedding.weight)
            risk_logits = None

            if return_diagnostics:
                return logits, risk_logits, {}
            return logits, risk_logits

        # --- Controlled Dual-Stream Forward ---
        c_0 = self.embedding(input_ids)  # [B, T, d_C]
        n = self.p_in(c_0.detach())  # N_0: [B, T, d_N]

        c = c_0
        diag_gates: Dict[str, torch.Tensor] = {}
        diag_steering: Dict[str, torch.Tensor] = {}
        diag_normative: Dict[str, torch.Tensor] = {}

        for l_idx, cap_layer in enumerate(self.capability_layers, start=1):
            prev_c = c
            # Compute intermediate unsteered capability proposal
            c_tilde = cap_layer(prev_c)

            if l_idx in self.config.controlled_layers:
                k = self.config.controlled_layers.index(l_idx)

                # Normative observation with detached capability proposal
                n_in = n + self.obs_projections[k](c_tilde.detach())

                # Normative Transformer Block
                n = self.normative_layers[k](n_in)

                # Controller outputs with bounded parameterizations and controller_scale ablation support
                g_raw = self.gate_projections[k](n)  # [B, T, 1]
                g_raw_scaled = self.config.alpha * torch.tanh(g_raw)
                g_l = 1.0 + controller_scale * g_raw_scaled

                s_raw = self.steering_projections[k](n)  # [B, T, d_C]
                s_raw_scaled = self.config.beta * torch.tanh(s_raw)
                s_l = controller_scale * s_raw_scaled

                # Residual capability steering
                c = prev_c + g_l * (c_tilde - prev_c) + s_l

                if return_diagnostics:
                    diag_gates[f"layer_{l_idx}"] = g_l.detach()
                    diag_steering[f"layer_{l_idx}"] = s_l.detach()
                    diag_normative[f"normative_block_{k+1}"] = n.detach()
            else:
                c = c_tilde

        logits = F.linear(self.capability_final_norm(c), self.embedding.weight)

        risk_logits: Optional[torch.Tensor] = None
        if prompt_end_indices is not None:
            normed_n = self.normative_final_norm(n)  # [B, T, d_N]
            batch_indices = torch.arange(B, device=input_ids.device)
            boundary_repr = normed_n[batch_indices, prompt_end_indices]  # [B, d_N]
            risk_logits = self.risk_head(boundary_repr).squeeze(-1)  # [B]

        if return_diagnostics:
            diagnostics = {
                "gates": diag_gates,
                "steering": diag_steering,
                "normative_states": diag_normative,
            }
            return logits, risk_logits, diagnostics

        return logits, risk_logits


class JointTrainingDualStreamModel(CCPTDualStreamModel):
    """Model B: Joint-Training Dual-Stream Control Model.

    Structurally identical to Model C (CCPT), but defaults to controlled forward mode for all batches.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        prompt_end_indices: Optional[torch.Tensor] = None,
        mode: str = "controlled",
        controller_scale: float = 1.0,
        return_diagnostics: bool = False,
    ):
        return super().forward(
            input_ids=input_ids,
            prompt_end_indices=prompt_end_indices,
            mode=mode,
            controller_scale=controller_scale,
            return_diagnostics=return_diagnostics,
        )

