"""Task 8: Prespecified Mechanistic Heterogeneity Analysis Module.

Provides non-invasive PyTorch forward hooks, Linear CKA calculation,
vector drift metrics, selectivity definitions, and behavioral transition classification.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccpt.config import DualStreamConfig, AdapterConfig
from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.modeling.adapter import FrozenBackboneAdapterModel


# =========================================================================
# 1. CORE VECTOR & DISTRIBUTION METRICS
# =========================================================================

def cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    """Computes cosine similarity between two 1D or 2D vectors in float64."""
    u_f64 = np.asarray(u, dtype=np.float64)
    v_f64 = np.asarray(v, dtype=np.float64)
    norm_u = np.linalg.norm(u_f64)
    norm_v = np.linalg.norm(v_f64)
    if norm_u < eps or norm_v < eps:
        return 0.0
    dot = np.dot(u_f64, v_f64)
    sim = dot / (norm_u * norm_v + eps)
    return float(np.clip(sim, -1.0, 1.0))


def relative_l2(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    """Computes relative L2 distance ||v - u||_2 / (||u||_2 + eps)."""
    u_f64 = np.asarray(u, dtype=np.float64)
    v_f64 = np.asarray(v, dtype=np.float64)
    diff_norm = np.linalg.norm(v_f64 - u_f64)
    base_norm = np.linalg.norm(u_f64)
    return float(diff_norm / (base_norm + eps))


def vector_norm(u: np.ndarray) -> float:
    """Computes Euclidean L2 norm of a vector in float64."""
    u_f64 = np.asarray(u, dtype=np.float64)
    return float(np.linalg.norm(u_f64))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Computes Jensen-Shannon divergence with natural logarithm between two discrete probability distributions.

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q).
    """
    p_f64 = np.asarray(p, dtype=np.float64)
    q_f64 = np.asarray(q, dtype=np.float64)

    # Normalize to ensure valid simplex
    p_f64 = np.clip(p_f64, eps, 1.0)
    q_f64 = np.clip(q_f64, eps, 1.0)
    p_f64 = p_f64 / np.sum(p_f64)
    q_f64 = q_f64 / np.sum(q_f64)

    m_f64 = 0.5 * (p_f64 + q_f64)

    # KL(P || M)
    kl_p_m = np.sum(p_f64 * (np.log(p_f64) - np.log(m_f64)))
    # KL(Q || M)
    kl_q_m = np.sum(q_f64 * (np.log(q_f64) - np.log(m_f64)))

    js = 0.5 * kl_p_m + 0.5 * kl_q_m
    return float(max(0.0, js))


# =========================================================================
# 2. LINEAR CENTERED KERNEL ALIGNMENT (CKA)
# =========================================================================

def compute_linear_cka(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """Computes Linear Centered Kernel Alignment (CKA) between representations X and Y.

    Reference: Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019.
    X: [N, D1] representation matrix across N prompts.
    Y: [N, D2] representation matrix across N prompts.
    """
    X_f64 = np.asarray(X, dtype=np.float64)
    Y_f64 = np.asarray(Y, dtype=np.float64)

    assert X_f64.ndim == 2 and Y_f64.ndim == 2, f"Expected 2D matrices, got {X_f64.shape} and {Y_f64.shape}"
    assert X_f64.shape[0] == Y_f64.shape[0], f"Sample dimension mismatch: {X_f64.shape[0]} vs {Y_f64.shape[0]}"

    n = X_f64.shape[0]
    if n < 2:
        return 1.0

    # Center columns across samples (mean subtraction across rows)
    X_c = X_f64 - np.mean(X_f64, axis=0, keepdims=True)
    Y_c = Y_f64 - np.mean(Y_f64, axis=0, keepdims=True)

    # Compute Frobenius norms of cross-covariance and self-covariances
    # ||X_c^T Y_c||_F^2
    cov_xy = np.dot(X_c.T, Y_c)
    hsic_xy = float(np.sum(cov_xy ** 2))

    # ||X_c^T X_c||_F^2
    cov_xx = np.dot(X_c.T, X_c)
    hsic_xx = float(np.sum(cov_xx ** 2))

    # ||Y_c^T Y_c||_F^2
    cov_yy = np.dot(Y_c.T, Y_c)
    hsic_yy = float(np.sum(cov_yy ** 2))

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < eps:
        return 0.0

    cka = hsic_xy / (denom + eps)
    return float(np.clip(cka, 0.0, 1.0))


# =========================================================================
# 3. BEHAVIORAL TRANSITION CLASSIFIER
# =========================================================================

def classify_behavioral_transition(pre_decision: str, post_decision: str) -> str:
    """Classifies refusal transition group for a prompt based on tri-state judge decisions.

    Returns one of:
      - 'retained_refusal'      (YES -> YES)
      - 'lost_refusal'          (YES -> NO)
      - 'gained_refusal'        (NO -> YES)
      - 'persistent_nonrefusal' (NO -> NO)
      - 'indeterminate'         (if either decision is NA / unknown)
    """
    pre = str(pre_decision).strip().upper()
    post = str(post_decision).strip().upper()

    if pre not in ("YES", "NO") or post not in ("YES", "NO"):
        return "indeterminate"

    if pre == "YES" and post == "YES":
        return "retained_refusal"
    elif pre == "YES" and post == "NO":
        return "lost_refusal"
    elif pre == "NO" and post == "YES":
        return "gained_refusal"
    else:
        return "persistent_nonrefusal"


# =========================================================================
# 4. FORWARD HOOKS FOR MODEL C (CCPT DUAL-STREAM)
# =========================================================================

class ModelCDiagnosticHooks:
    """Non-invasive forward hooks capturing Model C internal diagnostics at controlled layers."""

    def __init__(self, model: CCPTDualStreamModel):
        self.model = model
        self.handles: List[Any] = []
        self.captured: Dict[str, Any] = {}

    def __enter__(self):
        self.captured.clear()
        cfg = self.model.config
        alpha = cfg.alpha
        beta = cfg.beta

        # 1. Capability Proposal hooks (C_tilde_l at controlled layers)
        for l_idx in cfg.controlled_layers:
            layer_mod = self.model.capability_layers[l_idx - 1]
            def make_cap_hook(layer_num: int):
                def hook(module, args, output):
                    # output is c_tilde: [B, T, d_C]
                    self.captured[f"c_tilde_layer_{layer_num}"] = output.detach()
                return hook
            h = layer_mod.register_forward_hook(make_cap_hook(l_idx))
            self.handles.append(h)

        # 2. Observation Projection hooks (OBS_k)
        for k, obs_mod in enumerate(self.model.obs_projections):
            l_idx = cfg.controlled_layers[k]
            def make_obs_hook(layer_num: int):
                def hook(module, args, output):
                    # output is OBS_k: [B, T, d_N]
                    self.captured[f"obs_layer_{layer_num}"] = output.detach()
                return hook
            h = obs_mod.register_forward_hook(make_obs_hook(l_idx))
            self.handles.append(h)

        # 3. Normative Layer hooks (N_k)
        for k, norm_mod in enumerate(self.model.normative_layers):
            l_idx = cfg.controlled_layers[k]
            def make_norm_hook(layer_num: int):
                def hook(module, args, output):
                    # output is N_k: [B, T, d_N]
                    self.captured[f"normative_layer_{layer_num}"] = output.detach()
                return hook
            h = norm_mod.register_forward_hook(make_norm_hook(l_idx))
            self.handles.append(h)

        # 4. Gate Projection hooks (g_raw -> g_l)
        for k, gate_mod in enumerate(self.model.gate_projections):
            l_idx = cfg.controlled_layers[k]
            def make_gate_hook(layer_num: int):
                def hook(module, args, output):
                    # output is raw gate: [B, T, 1]
                    g_raw = output.detach()
                    g_l = 1.0 + alpha * torch.tanh(g_raw)
                    self.captured[f"gate_raw_layer_{layer_num}"] = g_raw
                    self.captured[f"gate_scaled_layer_{layer_num}"] = g_l
                return hook
            h = gate_mod.register_forward_hook(make_gate_hook(l_idx))
            self.handles.append(h)

        # 5. Steering Projection hooks (s_raw -> s_l)
        for k, steer_mod in enumerate(self.model.steering_projections):
            l_idx = cfg.controlled_layers[k]
            def make_steer_hook(layer_num: int):
                def hook(module, args, output):
                    # output is raw steering: [B, T, d_C]
                    s_raw = output.detach()
                    s_l = beta * torch.tanh(s_raw)
                    self.captured[f"steering_raw_layer_{layer_num}"] = s_raw
                    self.captured[f"steering_scaled_layer_{layer_num}"] = s_l
                return hook
            h = steer_mod.register_forward_hook(make_steer_hook(l_idx))
            self.handles.append(h)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# =========================================================================
# 5. FORWARD HOOKS FOR MODEL D (FROZEN-BACKBONE ADAPTER)
# =========================================================================

class ModelDDiagnosticHooks:
    """Non-invasive forward hooks capturing Model D internal diagnostics across all 8 adapter sites."""

    def __init__(self, model: FrozenBackboneAdapterModel):
        self.model = model
        self.handles: List[Any] = []
        self.captured: Dict[str, Any] = {}

    def __enter__(self):
        self.captured.clear()

        for l_idx, layer in enumerate(self.model.layers):
            # Attention Adapter
            def make_attn_hook(layer_num: int):
                def hook(module, args, output):
                    # input is args[0]: [B, T, d_model]
                    # output is [B, T, d_model]
                    in_t = args[0].detach()
                    out_t = output.detach()
                    res_t = out_t - in_t
                    self.captured[f"layer_{layer_num}_attn_adapter_input"] = in_t
                    self.captured[f"layer_{layer_num}_attn_adapter_output"] = out_t
                    self.captured[f"layer_{layer_num}_attn_adapter_residual"] = res_t
                return hook
            h_attn = layer.attn_adapter.register_forward_hook(make_attn_hook(l_idx))
            self.handles.append(h_attn)

            # MLP Adapter
            def make_mlp_hook(layer_num: int):
                def hook(module, args, output):
                    in_t = args[0].detach()
                    out_t = output.detach()
                    res_t = out_t - in_t
                    self.captured[f"layer_{layer_num}_mlp_adapter_input"] = in_t
                    self.captured[f"layer_{layer_num}_mlp_adapter_output"] = out_t
                    self.captured[f"layer_{layer_num}_mlp_adapter_residual"] = res_t
                return hook
            h_mlp = layer.mlp_adapter.register_forward_hook(make_mlp_hook(l_idx))
            self.handles.append(h_mlp)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()
