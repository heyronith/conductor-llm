"""Machine-derived Seed-4 retention / behavior / ablation summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SEED4 = 20260825
STEPS = (0, 250, 1000, 4000)
MODELS = ("model_b", "model_c", "model_d")
PRIMARY_ENDPOINT_STEP = 1000
SECONDARY_ENDPOINT_STEP = 4000


def _group_key(model: str, step: int, condition: str, cohort: str) -> str:
    return f"{model}__step_{step}__cond_{condition}__cohort_{cohort}"


def extract_rate(summary: dict[str, Any], model: str, step: int, condition: str, cohort: str) -> dict[str, Any] | None:
    key = _group_key(model, step, condition, cohort)
    g = summary.get(key)
    if g is None:
        return None
    return {
        "key": key,
        "refusal_yes": g["refusal_yes"],
        "refusal_no": g["refusal_no"],
        "refusal_na": g["refusal_na"],
        "harmful_yes": g.get("harmful_yes"),
        "harmful_no": g.get("harmful_no"),
        "harmful_na": g.get("harmful_na"),
        "determinate_refusal_rate": g["determinate_refusal_rate"],
        "na_as_refusal_rate": g.get("na_as_refusal_rate"),
        "na_as_nonrefusal_rate": g.get("na_as_nonrefusal_rate"),
        "harmful_response_rate": g.get("harmful_response_rate"),
        "determinate_denominator": g.get("determinate_denominator"),
        "total": g["total"],
    }


def build_behavior_summary(
    judge_summary: dict[str, Any],
    capability_by_model: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = judge_summary["summary"]
    models_out: dict[str, Any] = {}
    for model in MODELS:
        traj: dict[str, Any] = {}
        for step in STEPS:
            harmful = extract_rate(summary, model, step, "active", "harmful")
            benign = extract_rate(summary, model, step, "active", "benign")
            if harmful is None and benign is None:
                continue
            over_refusal = None
            if benign is not None:
                # Benign over-refusal = determinate refusal rate on benign cohort
                over_refusal = benign["determinate_refusal_rate"]
            cap = None
            if capability_by_model and model in capability_by_model:
                cap = capability_by_model[model].get(f"step_{step}") or capability_by_model[model].get(str(step))
            traj[str(step)] = {
                "harmful_refusal": harmful,
                "benign_over_refusal": {
                    "determinate_refusal_rate": over_refusal,
                    "raw": benign,
                },
                "capability": cap,
            }
        models_out[model] = traj
    return {
        "task": "strengthening_seed4_behavior_summary",
        "seed": SEED4,
        "primary_endpoint_step": PRIMARY_ENDPOINT_STEP,
        "secondary_endpoint_step": SECONDARY_ENDPOINT_STEP,
        "models": models_out,
        "source_evaluation_sha": judge_summary.get("evaluation_sha"),
    }


def build_ablation_summary(judge_summary: dict[str, Any]) -> dict[str, Any]:
    summary = judge_summary["summary"]
    ablation_steps = (0, 1000, 4000)
    out: dict[str, Any] = {}
    for model in MODELS:
        model_out: dict[str, Any] = {}
        for step in ablation_steps:
            active = extract_rate(summary, model, step, "active", "harmful")
            ablated = extract_rate(summary, model, step, "ablated", "harmful")
            if active is None or ablated is None:
                continue
            gap = active["determinate_refusal_rate"] - ablated["determinate_refusal_rate"]
            model_out[str(step)] = {
                "active": active,
                "ablated": ablated,
                "active_minus_ablated_determinate_refusal_rate": gap,
                "na_sensitivity": {
                    "active_na_as_refusal": active.get("na_as_refusal_rate"),
                    "active_na_as_nonrefusal": active.get("na_as_nonrefusal_rate"),
                    "ablated_na_as_refusal": ablated.get("na_as_refusal_rate"),
                    "ablated_na_as_nonrefusal": ablated.get("na_as_nonrefusal_rate"),
                },
            }
        out[model] = model_out
    return {
        "task": "strengthening_seed4_ablation_summary",
        "seed": SEED4,
        "models": out,
        "source_evaluation_sha": judge_summary.get("evaluation_sha"),
    }


def build_retention_summary(behavior: dict[str, Any]) -> dict[str, Any]:
    retentions: dict[str, Any] = {}
    for model in MODELS:
        traj = behavior["models"].get(model, {})
        base = traj.get("0", {}).get("harmful_refusal")
        if base is None:
            continue
        rate0 = base["determinate_refusal_rate"]
        model_ret: dict[str, Any] = {"rate_0": rate0}
        for step in (250, 1000, 4000):
            cur = traj.get(str(step), {}).get("harmful_refusal")
            if cur is None:
                continue
            rate = cur["determinate_refusal_rate"]
            model_ret[f"rate_{step}"] = rate
            model_ret[f"retention_{step}"] = rate - rate0
            model_ret[f"retention_{step}_pp"] = (rate - rate0) * 100.0
        retentions[model] = model_ret

    comparisons: dict[str, Any] = {}
    for step in (250, 1000, 4000):
        b = retentions.get("model_b", {}).get(f"retention_{step}")
        c = retentions.get("model_c", {}).get(f"retention_{step}")
        d = retentions.get("model_d", {}).get(f"retention_{step}")
        if c is not None and b is not None:
            comparisons[f"C_minus_B_{step}"] = c - b
            comparisons[f"C_minus_B_{step}_pp"] = (c - b) * 100.0
        if c is not None and d is not None:
            comparisons[f"C_minus_D_{step}"] = c - d
            comparisons[f"C_minus_D_{step}_pp"] = (c - d) * 100.0

    return {
        "task": "strengthening_seed4_retention_summary",
        "seed": SEED4,
        "definition": "retention_S = determinate_harmful_refusal_rate_S - determinate_harmful_refusal_rate_0",
        "primary_endpoint_step": PRIMARY_ENDPOINT_STEP,
        "secondary_endpoint_step": SECONDARY_ENDPOINT_STEP,
        "retentions": retentions,
        "comparisons": comparisons,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
