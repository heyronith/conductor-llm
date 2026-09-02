"""Seed-4 live cash ledger and pre-pipeline gate helpers (zero-GPU / CPU-only)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

HARD_AUTHORIZATION_USD = 27.00
REMAINING_MODAL_CREDITS_USD = 0.0
SEED4 = 20260825
MODEL_ORDER = ("model_d", "model_b", "model_c")

# Audited historical successful aggregate envelope (ap-TaUU), not fallback 6000/305/780.
HISTORICAL_SUCCESSFUL_BCD_H100_USD = 20.73914211
PER_MODEL_H100_ENVELOPE_USD = HISTORICAL_SUCCESSFUL_BCD_H100_USD / 3.0
HISTORICAL_CORRECTED_EVAL_TOTAL_USD = 1.90400049
HISTORICAL_OTHER_INCREMENTAL_USD = 0.3687


def repo_root(start: Path | None = None) -> Path:
    return start or Path(__file__).resolve().parents[3]


def ledger_path(root: Path | None = None) -> Path:
    return repo_root(root) / "artifacts" / "seed4_execution_live_cost_ledger.json"


def fetch_workspace_rates() -> dict[str, float]:
    raw = subprocess.check_output(
        ["uv", "run", "modal", "billing", "rates", "--json"],
        text=True,
    )
    rates = json.loads(raw)
    return {
        "h100_hourly_usd": float(rates["gpu_hour_cost_h100"]),
        "l40s_hourly_usd": float(rates["gpu_hour_cost_l40s"]),
        "cpu_hourly_usd": float(rates["cpu_hour_cost"]),
        "mem_gib_hourly_usd": float(rates["mem_gib_hour_cost"]),
        "raw": rates,
    }


def estimate_cost_from_seconds(seconds: float, hourly_rate: float) -> float:
    return float(Decimal(str(seconds)) / Decimal("3600") * Decimal(str(hourly_rate)))


def new_ledger(execution_sha: str, rates: dict[str, float]) -> dict[str, Any]:
    return {
        "task": "seed4_execution_live_cost_ledger",
        "seed": SEED4,
        "execution_sha": execution_sha,
        "hard_authorization_usd": HARD_AUTHORIZATION_USD,
        "remaining_modal_credits_usd": REMAINING_MODAL_CREDITS_USD,
        "out_of_pocket_equals_metered": True,
        "rates": {
            "h100_hourly_usd": rates["h100_hourly_usd"],
            "l40s_hourly_usd": rates["l40s_hourly_usd"],
            "cpu_hourly_usd": rates["cpu_hourly_usd"],
            "mem_gib_hourly_usd": rates["mem_gib_hourly_usd"],
        },
        "planning_envelopes_usd": {
            "per_model_h100": PER_MODEL_H100_ENVELOPE_USD,
            "full_bcd_h100": HISTORICAL_SUCCESSFUL_BCD_H100_USD,
            "corrected_eval_total": HISTORICAL_CORRECTED_EVAL_TOTAL_USD,
            "other_incremental": HISTORICAL_OTHER_INCREMENTAL_USD,
        },
        "model_order": list(MODEL_ORDER),
        "completed_models": [],
        "current_model": None,
        "stages": [],
        "accrued": {
            "h100_seconds_measured": 0.0,
            "h100_cost_estimated_usd": 0.0,
            "l40s_seconds_measured": 0.0,
            "l40s_cost_estimated_usd": 0.0,
            "cpu_memory_cost_estimated_usd": 0.0,
            "total_estimated_usd": 0.0,
            "total_billed_usd_when_available": None,
        },
        "remaining_authorization_usd": HARD_AUTHORIZATION_USD,
        "projected_final_total_usd": None,
        "ceiling_breached": False,
        "budget_hard_stop": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_ledger(ledger: dict[str, Any], root: Path | None = None) -> Path:
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2)
        handle.write("\n")
    return path


def load_ledger(root: Path | None = None) -> dict[str, Any]:
    path = ledger_path(root)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def remaining_mandatory_cost_usd(ledger: dict[str, Any], next_model: str | None) -> float:
    """Estimate cost of next_model (if any) + all later models + eval + other."""
    envelopes = ledger["planning_envelopes_usd"]
    remaining_models = []
    if next_model is not None:
        order = ledger["model_order"]
        idx = order.index(next_model)
        remaining_models = order[idx:]
    else:
        remaining_models = [m for m in ledger["model_order"] if m not in ledger["completed_models"]]

    # If evaluation not done, reserve full corrected eval
    eval_done = any(s.get("stage") == "evaluation_complete" for s in ledger.get("stages", []))
    eval_reserve = 0.0 if eval_done else envelopes["corrected_eval_total"]
    other_reserve = envelopes["other_incremental"] * (len(remaining_models) / 3.0) if remaining_models else 0.0
    h100_reserve = envelopes["per_model_h100"] * len(remaining_models)
    return h100_reserve + eval_reserve + other_reserve


def pre_pipeline_gate(ledger: dict[str, Any], next_model: str) -> dict[str, Any]:
    accrued = float(ledger["accrued"]["total_estimated_usd"])
    remaining_auth = float(ledger["hard_authorization_usd"]) - accrued
    needed = remaining_mandatory_cost_usd(ledger, next_model)
    projected_final = accrued + needed
    ok = projected_final <= float(ledger["hard_authorization_usd"]) + 1e-9
    return {
        "next_model": next_model,
        "accrued_usd": accrued,
        "remaining_authorization_usd": remaining_auth,
        "reserved_for_remaining_mandatory_usd": needed,
        "projected_final_total_usd": projected_final,
        "hard_authorization_usd": ledger["hard_authorization_usd"],
        "allowed": ok,
        "reason": (
            "OK"
            if ok
            else (
                f"Projected final ${projected_final:.4f} exceeds hard ceiling "
                f"${ledger['hard_authorization_usd']:.2f}; refusing to launch {next_model}."
            )
        ),
    }


def pre_evaluation_gate(ledger: dict[str, Any]) -> dict[str, Any]:
    accrued = float(ledger["accrued"]["total_estimated_usd"])
    eval_reserve = float(ledger["planning_envelopes_usd"]["corrected_eval_total"])
    projected = accrued + eval_reserve
    ok = projected <= float(ledger["hard_authorization_usd"]) + 1e-9
    return {
        "accrued_usd": accrued,
        "eval_reserve_usd": eval_reserve,
        "projected_final_total_usd": projected,
        "allowed": ok,
        "reason": "OK" if ok else "Evaluation would breach $27.00 hard ceiling",
    }


def record_training_stage(
    ledger: dict[str, Any],
    *,
    model_type: str,
    training_result: dict[str, Any],
    h100_hourly: float,
) -> dict[str, Any]:
    timing = training_result.get("timing", {})
    # Prefer measured total; never invent fallbacks here — use returned total_h100_seconds.
    h100_seconds = float(timing.get("total_h100_seconds", 0.0))
    h100_cost = estimate_cost_from_seconds(h100_seconds, h100_hourly)
    stage = {
        "stage": f"training_{model_type}",
        "model_type": model_type,
        "status": training_result.get("status"),
        "h100_seconds_measured": h100_seconds,
        "h100_cost_estimated_usd": h100_cost,
        "timing": timing,
        "final_state_hash": training_result.get("final_state_hash"),
        "initial_state_hash": training_result.get("initial_state_hash"),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    ledger["stages"].append(stage)
    if model_type not in ledger["completed_models"]:
        ledger["completed_models"].append(model_type)
    ledger["current_model"] = None
    ledger["accrued"]["h100_seconds_measured"] += h100_seconds
    ledger["accrued"]["h100_cost_estimated_usd"] += h100_cost
    ledger["accrued"]["total_estimated_usd"] = (
        ledger["accrued"]["h100_cost_estimated_usd"]
        + ledger["accrued"]["l40s_cost_estimated_usd"]
        + ledger["accrued"]["cpu_memory_cost_estimated_usd"]
    )
    ledger["remaining_authorization_usd"] = (
        float(ledger["hard_authorization_usd"]) - float(ledger["accrued"]["total_estimated_usd"])
    )
    ledger["ceiling_breached"] = ledger["accrued"]["total_estimated_usd"] > float(
        ledger["hard_authorization_usd"]
    )
    return ledger


def record_evaluation_stage(
    ledger: dict[str, Any],
    *,
    eval_seconds_by_model: dict[str, float],
    judge_seconds: float,
    l40s_hourly: float,
) -> dict[str, Any]:
    total_eval = sum(float(v) for v in eval_seconds_by_model.values())
    total_l40s = total_eval + float(judge_seconds)
    l40s_cost = estimate_cost_from_seconds(total_l40s, l40s_hourly)
    ledger["stages"].append(
        {
            "stage": "evaluation_complete",
            "eval_seconds_by_model": eval_seconds_by_model,
            "judge_seconds": judge_seconds,
            "l40s_seconds_measured": total_l40s,
            "l40s_cost_estimated_usd": l40s_cost,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    ledger["accrued"]["l40s_seconds_measured"] += total_l40s
    ledger["accrued"]["l40s_cost_estimated_usd"] += l40s_cost
    ledger["accrued"]["total_estimated_usd"] = (
        ledger["accrued"]["h100_cost_estimated_usd"]
        + ledger["accrued"]["l40s_cost_estimated_usd"]
        + ledger["accrued"]["cpu_memory_cost_estimated_usd"]
    )
    ledger["remaining_authorization_usd"] = (
        float(ledger["hard_authorization_usd"]) - float(ledger["accrued"]["total_estimated_usd"])
    )
    ledger["ceiling_breached"] = ledger["accrued"]["total_estimated_usd"] > float(
        ledger["hard_authorization_usd"]
    )
    return ledger


def volume_seed4_model_terminal_exists(model_type: str) -> bool:
    """CPU-only Modal volume check for persistence_4000.pt."""
    path = f"/ccpt/strengthening_task2/seed_{SEED4}/{model_type}/persistence_4000.pt"
    proc = subprocess.run(
        ["uv", "run", "modal", "volume", "ls", "ccpt-authoritative-runs", path],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0
