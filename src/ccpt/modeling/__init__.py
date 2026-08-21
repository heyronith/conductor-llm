"""Modeling primitives and architectures for CCPT research."""

from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
from ccpt.modeling.layers import (
    CausalSelfAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
)

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "CausalSelfAttention",
    "SwiGLU",
    "TransformerBlock",
    "ParameterMatchedBaselineModel",
    "CCPTDualStreamModel",
    "JointTrainingDualStreamModel",
    "FrozenBackboneAdapterModel",
]
