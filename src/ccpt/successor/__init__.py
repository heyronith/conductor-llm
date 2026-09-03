"""Successor architecture experiments (non-historical CCPT namespace)."""

from ccpt.successor.adapters import (
    ActuatorRepairAdapter,
    GenericResidualRepairControl,
    ObserverRepairAdapter,
    ResidualBottleneck,
)
from ccpt.successor.retrofit import (
    AdaptiveInterfaceWrapper,
    RepairVariant,
    count_parameters,
    freeze_module,
    hash_existing_parameters,
    match_generic_rank,
)

__all__ = [
    "ResidualBottleneck",
    "ObserverRepairAdapter",
    "ActuatorRepairAdapter",
    "GenericResidualRepairControl",
    "AdaptiveInterfaceWrapper",
    "RepairVariant",
    "freeze_module",
    "hash_existing_parameters",
    "count_parameters",
    "match_generic_rank",
]
