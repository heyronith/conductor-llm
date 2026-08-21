"""CCPT package root."""

from ccpt.config import (
    AdapterConfig,
    BaselineConfig,
    DualStreamConfig,
    get_micro_adapter_config,
    get_micro_baseline_config,
    get_micro_dual_stream_config,
    get_smoke_adapter_config,
    get_smoke_baseline_config,
    get_smoke_dual_stream_config,
    get_task5_micro_adapter_config,
    get_task5_micro_baseline_config,
    get_task5_micro_dual_stream_config,
)
from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel

__all__ = [
    "AdapterConfig",
    "BaselineConfig",
    "DualStreamConfig",
    "get_micro_adapter_config",
    "get_micro_baseline_config",
    "get_micro_dual_stream_config",
    "get_smoke_adapter_config",
    "get_smoke_baseline_config",
    "get_smoke_dual_stream_config",
    "get_task5_micro_adapter_config",
    "get_task5_micro_baseline_config",
    "get_task5_micro_dual_stream_config",
    "ParameterMatchedBaselineModel",
    "CCPTDualStreamModel",
    "JointTrainingDualStreamModel",
    "FrozenBackboneAdapterModel",
]
