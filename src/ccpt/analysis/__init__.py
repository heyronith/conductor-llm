"""CCPT Analysis and Mechanistic Diagnostics Package."""

from ccpt.analysis.task8_mechanistic import (
    cosine_similarity,
    relative_l2,
    vector_norm,
    jensen_shannon_divergence,
    compute_linear_cka,
    classify_behavioral_transition,
    ModelCDiagnosticHooks,
    ModelDDiagnosticHooks,
)

__all__ = [
    "cosine_similarity",
    "relative_l2",
    "vector_norm",
    "jensen_shannon_divergence",
    "compute_linear_cka",
    "classify_behavioral_transition",
    "ModelCDiagnosticHooks",
    "ModelDDiagnosticHooks",
]
