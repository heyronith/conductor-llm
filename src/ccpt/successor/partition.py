"""Machine-readable historical CCPT parameter → conceptual bucket mapping."""

from __future__ import annotations

from typing import Any, Dict, List

from ccpt.modeling.dual_stream import CCPTDualStreamModel


BUCKETS = (
    "CAPABILITY",
    "OLD_OBSERVER_INTERFACE",
    "NORMATIVE_SEMANTIC_CORE",
    "OLD_ACTUATOR_INTERFACE",
    "RISK_READOUT",
)


def classify_parameter_name(name: str) -> str:
    if name.startswith("embedding.") or name.startswith("capability_layers.") or name.startswith(
        "capability_final_norm."
    ):
        return "CAPABILITY"
    if name.startswith("p_in.") or name.startswith("obs_projections."):
        return "OLD_OBSERVER_INTERFACE"
    if name.startswith("normative_layers.") or name.startswith("normative_final_norm."):
        return "NORMATIVE_SEMANTIC_CORE"
    if name.startswith("gate_projections.") or name.startswith("steering_projections."):
        return "OLD_ACTUATOR_INTERFACE"
    if name.startswith("risk_head."):
        return "RISK_READOUT"
    raise KeyError(f"Unclassified CCPT parameter name: {name}")


def build_parameter_partition(model: CCPTDualStreamModel) -> Dict[str, Any]:
    by_bucket: Dict[str, List[str]] = {b: [] for b in BUCKETS}
    for name, _ in model.named_parameters():
        by_bucket[classify_parameter_name(name)].append(name)
    for b in BUCKETS:
        by_bucket[b] = sorted(by_bucket[b])
    return {
        "task": "successor_task1_parameter_partition",
        "model_class": type(model).__name__,
        "buckets": by_bucket,
        "counts": {b: len(by_bucket[b]) for b in BUCKETS},
        "total_named_parameters": sum(len(by_bucket[b]) for b in BUCKETS),
        "retrofit_policy": "EVERY_EXISTING_PARAMETER_FROZEN",
        "trainable_in_this_experiment": "NEW_REPAIR_MODULES_ONLY",
    }
