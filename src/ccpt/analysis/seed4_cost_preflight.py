"""Billing-grounded Seed 4 cost preflight for CCPT strengthening."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

SEED4 = 20260825
SEED1 = 20260821

CCPT_APPS = {
    "strengthening-task2-sentinel",
    "strengthening-task3-1-eval",
    "strengthening-task3-forensic",
}

TASK2_APP = "strengthening-task2-sentinel"
TASK31_APP = "strengthening-task3-1-eval"

H100_GPU_SECONDS_AUTHORIZED = 0
L40S_GPU_SECONDS_AUTHORIZED = 0


@dataclass(frozen=True)
class EvidenceClass:
    ACTUAL_BILLED: str = "ACTUAL_BILLED"
    ACTUAL_RUNTIME: str = "ACTUAL_RUNTIME"
    MODELED_OR_FALLBACK: str = "MODELED_OR_FALLBACK"


def repo_root(start: Path | None = None) -> Path:
    return start or Path(__file__).resolve().parents[3]


def git_head_sha(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def modal_cli_version() -> str:
    output = subprocess.check_output(["uv", "run", "modal", "--version"], text=True).strip()
    return output.replace("modal client version: ", "")


def modal_profile() -> str:
    return subprocess.check_output(["uv", "run", "modal", "profile", "current"], text=True).strip()


def fetch_modal_rates(root: Path) -> dict[str, Any]:
    output = subprocess.check_output(["uv", "run", "modal", "billing", "rates", "--json"], cwd=root, text=True)
    rates = json.loads(output)
    payload = {
        "task": "seed4_cost_preflight_modal_rates",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "modal_cli_version": modal_cli_version(),
        "modal_profile": modal_profile(),
        "rates": rates,
        "workspace_h100_hourly_usd": float(rates["gpu_hour_cost_h100"]),
        "workspace_l40s_hourly_usd": float(rates["gpu_hour_cost_l40s"]),
        "workspace_cpu_hourly_usd": float(rates["cpu_hour_cost"]),
        "workspace_mem_gib_hourly_usd": float(rates["mem_gib_hour_cost"]),
        "evidence_class": EvidenceClass.ACTUAL_BILLED,
    }
    out_path = root / "artifacts" / "seed4_cost_preflight_modal_rates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def _decimal_cost(value: str | float | Decimal) -> Decimal:
    return Decimal(str(value))


def load_billing_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_billing(
    rows: list[dict[str, Any]],
    *,
    apps: set[str] | None = None,
) -> dict[str, Any]:
    by_app: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    by_app_day: dict[str, dict[str, dict[str, Decimal]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Decimal))
    )
    by_app_object: dict[str, dict[str, dict[str, Decimal]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Decimal))
    )

    for row in rows:
        app = row.get("description", "")
        if apps and app not in apps:
            continue
        resource = row["resource"]
        cost = _decimal_cost(row["cost"])
        day = str(row.get("interval_start", ""))[:10]
        object_id = row.get("object_id", "unknown")
        by_app[app][resource] += cost
        by_app_day[app][day][resource] += cost
        by_app_object[app][object_id][resource] += cost

    def _serialize_costs(costs: dict[str, Decimal]) -> dict[str, float]:
        return {resource: float(cost) for resource, cost in sorted(costs.items())}

    apps_out: dict[str, Any] = {}
    for app, costs in by_app.items():
        total = sum(costs.values())
        apps_out[app] = {
            "total_cost_usd": float(total),
            "by_resource_usd": _serialize_costs(costs),
            "by_day": {
                day: {
                    "total_cost_usd": float(sum(day_costs.values())),
                    "by_resource_usd": _serialize_costs(day_costs),
                }
                for day, day_costs in sorted(by_app_day[app].items())
            },
            "by_object_id": {
                object_id: {
                    "total_cost_usd": float(sum(obj_costs.values())),
                    "by_resource_usd": _serialize_costs(obj_costs),
                }
                for object_id, obj_costs in sorted(
                    by_app_object[app].items(), key=lambda item: -float(sum(item[1].values()))
                )
            },
        }

    return {
        "apps": apps_out,
        "filtered_total_cost_usd": float(sum(sum(costs.values()) for costs in by_app.values())),
    }


def implied_gpu_seconds(cost_usd: float, hourly_rate_usd: float) -> float:
    if hourly_rate_usd <= 0:
        return 0.0
    return float((Decimal(str(cost_usd)) / Decimal(str(hourly_rate_usd))) * Decimal("3600"))


def classify_task2_timing_fields(root: Path) -> dict[str, Any]:
    summary_path = root / "artifacts" / "strengthening_task2_sentinel_summary.json"
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    wall_clock = float(summary.get("total_wall_clock_seconds", 0.0))
    per_model = summary["seed_1_training"]
    sample = per_model["model_b"]["timing"]

    fast_return_evidence = wall_clock < float(sample["total_h100_seconds"])
    return {
        "lm_pretrain_seconds_field": {
            "value": sample["lm_pretrain_seconds"],
            "evidence_class": EvidenceClass.MODELED_OR_FALLBACK,
            "rationale": (
                "Hardcoded default 6000.0 in modal/strengthening_task2_sentinel.py and returned "
                "unchanged when persistence_4000.pt fast-return path is taken; not phase-measured "
                "for the Seed-1 orchestrator run (wall clock ~1001s << reported 21255 H100 seconds)."
            ),
        },
        "safety_train_seconds_field": {
            "value": sample["safety_train_seconds"],
            "evidence_class": EvidenceClass.MODELED_OR_FALLBACK,
            "rationale": "Same fast-return/default constant path as LM seconds (305.0).",
        },
        "persistence_train_seconds_field": {
            "value": sample["persistence_train_seconds"],
            "evidence_class": EvidenceClass.MODELED_OR_FALLBACK,
            "rationale": "Same fast-return/default constant path as LM seconds (780.0).",
        },
        "per_model_total_h100_seconds_field": {
            "value": sample["total_h100_seconds"],
            "evidence_class": EvidenceClass.MODELED_OR_FALLBACK,
            "rationale": "Sum of fallback phase constants (6000+305+780=7085) per model.",
        },
        "aggregate_seed1_h100_seconds_field": {
            "value": float(summary["technical_health_gate"]["seed_1_h100_seconds"]),
            "evidence_class": EvidenceClass.MODELED_OR_FALLBACK,
            "rationale": "3 × 7085 fallback seconds; differs from Modal billed implied seconds.",
        },
        "orchestrator_wall_clock_seconds": {
            "value": wall_clock,
            "evidence_class": EvidenceClass.ACTUAL_RUNTIME,
            "rationale": "Measured orchestrator duration in sentinel summary.",
        },
        "fast_return_likely": fast_return_evidence,
    }


def build_actuals(root: Path, rates: dict[str, Any], billing_summary: dict[str, Any]) -> dict[str, Any]:
    h100_rate = rates["workspace_h100_hourly_usd"]
    l40s_rate = rates["workspace_l40s_hourly_usd"]

    task2 = billing_summary["apps"].get(TASK2_APP, {})
    task31 = billing_summary["apps"].get(TASK31_APP, {})

    task2_h100 = task2.get("by_resource_usd", {}).get("H100", 0.0)
    task2_l40s = task2.get("by_resource_usd", {}).get("L40S", 0.0)
    task2_cpu = task2.get("by_resource_usd", {}).get("CPU", 0.0)
    task2_mem = task2.get("by_resource_usd", {}).get("Memory", 0.0)
    task2_total = task2.get("total_cost_usd", 0.0)

    task31_l40s = task31.get("by_resource_usd", {}).get("L40S", 0.0)
    task31_cpu = task31.get("by_resource_usd", {}).get("CPU", 0.0)
    task31_mem = task31.get("by_resource_usd", {}).get("Memory", 0.0)
    task31_total = task31.get("total_cost_usd", 0.0)

    task2_sep1 = task2.get("by_day", {}).get("2026-09-01", {})
    task2_sep1_h100 = task2_sep1.get("by_resource_usd", {}).get("H100", 0.0)
    task2_sep1_total = task2_sep1.get("total_cost_usd", 0.0)

    task31_summary_path = root / "artifacts" / "strengthening_task3_1_cost_summary.json"
    with open(task31_summary_path, "r", encoding="utf-8") as handle:
        task31_reported = json.load(handle)

    reported_l40s_cost = float(task31_reported["hardware_accounting"]["total_l40s_cost_usd"])
    reported_l40s_seconds = float(task31_reported["hardware_accounting"]["total_l40s_gpu_seconds"])

    timing_audit = classify_task2_timing_fields(root)

    per_model_h100_billed = None
    per_model_note = (
        "Modal billing cannot uniquely attribute H100 spend to model_b/model_c/model_d within "
        "strengthening-task2-sentinel; aggregate app-level billing preserved."
    )

    return {
        "task": "seed4_cost_preflight_actuals",
        "seed1": SEED1,
        "billing_attribution_quality": "HIGH",
        "billing_attribution_notes": [
            "App-level attribution to strengthening-task2-sentinel and strengthening-task3-1-eval is exact.",
            "Per-model H100 split is unavailable from Modal billing rows.",
            "Task-2 L40S spend reflects the invalid unframed evaluation pass and is excluded from Seed-4 projection.",
        ],
        "task2_seed1_training": {
            "modal_app": TASK2_APP,
            "h100_billed_cost_usd": task2_h100,
            "h100_billed_seconds_implied": implied_gpu_seconds(task2_h100, h100_rate),
            "l40s_billed_cost_usd": task2_l40s,
            "l40s_billed_seconds_implied": implied_gpu_seconds(task2_l40s, l40s_rate),
            "cpu_billed_cost_usd": task2_cpu,
            "memory_billed_cost_usd": task2_mem,
            "total_billed_cost_usd": task2_total,
            "evidence_class": EvidenceClass.ACTUAL_BILLED,
            "per_model_h100_billed_cost_usd": per_model_h100_billed,
            "per_model_attribution": per_model_note,
            "clean_training_day_2026_09_01": {
                "total_billed_cost_usd": task2_sep1_total,
                "h100_billed_cost_usd": task2_sep1_h100,
                "h100_billed_seconds_implied": implied_gpu_seconds(task2_sep1_h100, h100_rate),
                "evidence_class": EvidenceClass.ACTUAL_BILLED,
            },
        },
        "task3_1_corrected_evaluation": {
            "modal_app": TASK31_APP,
            "l40s_billed_cost_usd": task31_l40s,
            "l40s_billed_seconds_implied": implied_gpu_seconds(task31_l40s, l40s_rate),
            "cpu_billed_cost_usd": task31_cpu,
            "memory_billed_cost_usd": task31_mem,
            "total_billed_cost_usd": task31_total,
            "evidence_class": EvidenceClass.ACTUAL_BILLED,
            "reported_runtime_seconds": {
                "total_l40s_gpu_seconds": reported_l40s_seconds,
                "evidence_class": EvidenceClass.ACTUAL_RUNTIME,
                "source_artifact": "artifacts/strengthening_task3_1_cost_summary.json",
            },
            "reported_cost_usd": {
                "total_l40s_cost_usd": reported_l40s_cost,
                "evidence_class": EvidenceClass.MODELED_OR_FALLBACK,
                "rationale": "Computed as runtime_seconds × hardcoded L40S_HOURLY_RATE=1.9512 in eval runner, not from Modal billing.",
            },
            "billing_vs_reported_delta_usd": round(task31_l40s - reported_l40s_cost, 4),
        },
        "timing_field_audit": timing_audit,
        "total_relevant_seed1_billed_cost_usd": round(task2_h100 + task2_cpu + task2_mem + task31_total, 4),
        "total_relevant_seed1_billed_cost_excludes_task2_l40s": True,
        "excluded_task2_l40s_billed_cost_usd": task2_l40s,
    }


def build_projection(root: Path, rates: dict[str, Any], actuals: dict[str, Any]) -> dict[str, Any]:
    h100_rate = rates["workspace_h100_hourly_usd"]
    l40s_rate = rates["workspace_l40s_hourly_usd"]

    clean_day_h100 = actuals["task2_seed1_training"]["clean_training_day_2026_09_01"]["h100_billed_cost_usd"]
    full_window_h100 = actuals["task2_seed1_training"]["h100_billed_cost_usd"]
    full_window_other = (
        actuals["task2_seed1_training"]["cpu_billed_cost_usd"]
        + actuals["task2_seed1_training"]["memory_billed_cost_usd"]
    )
    eval_total = actuals["task3_1_corrected_evaluation"]["total_billed_cost_usd"]

    low = {
        "h100_usd": clean_day_h100,
        "l40s_usd": eval_total,
        "other_incremental_usd": round(full_window_other * 0.8, 4),
        "assumptions": [
            "Single-day H100 training window without Aug-31 partial-run overhead.",
            "Corrected Task-3.1-style framed evaluation billed cost reused.",
            "Reuses materialized FineWeb shards, WildGuard arrows, tokenizer caches, and Modal image layers.",
        ],
    }
    expected = {
        "h100_usd": round((clean_day_h100 + full_window_h100) / 2.0, 4),
        "l40s_usd": eval_total,
        "other_incremental_usd": round(full_window_other, 4),
        "assumptions": [
            "Midpoint between clean Sep-1 H100 billing and full Aug31-Sep1 H100 billing window.",
            "One corrected evaluation pass on L40S.",
            "No protocol changes; independent Seed 20260825 retrain required.",
        ],
    }
    high = {
        "h100_usd": full_window_h100,
        "l40s_usd": round(eval_total * 1.08, 4),
        "other_incremental_usd": round(full_window_other * 1.15, 4),
        "assumptions": [
            "Includes Aug-31 partial-run H100 overhead observed on Seed 1.",
            "8% L40S buffer for judge/generation variance.",
            "15% CPU/memory buffer; excludes second full training rerun.",
        ],
    }

    for scenario in (low, expected, high):
        scenario["total_usd"] = round(
            scenario["h100_usd"] + scenario["l40s_usd"] + scenario["other_incremental_usd"], 4
        )

    within_10 = expected["total_usd"] <= 10.0
    recommended_hard = round(max(high["total_usd"] * 1.1, expected["total_usd"] * 1.2), 2)

    return {
        "task": "seed4_cost_preflight_projection",
        "seed4": SEED4,
        "protocol_scope": {
            "models": ["model_b", "model_c", "model_d"],
            "lm_pretrain_1b": True,
            "safety_tokens": 20_010_611,
            "persistence_steps": [0, 250, 1000, 4000],
            "evaluation": "canonical format_eval_prompt, max_new_tokens=48, WildGuard judging",
        },
        "empirical_basis": {
            "seed1_task2_h100_billed_usd_clean_day": clean_day_h100,
            "seed1_task2_h100_billed_usd_full_window": full_window_h100,
            "seed1_task31_eval_billed_usd": eval_total,
            "workspace_h100_hourly_usd": h100_rate,
            "workspace_l40s_hourly_usd": l40s_rate,
        },
        "scenarios_usd": {
            "low": low,
            "expected": expected,
            "high": high,
        },
        "seed4_full_protocol_within_10_usd": within_10,
        "minimum_realistic_expected_budget_usd": expected["total_usd"],
        "p80_or_conservative_cost_usd": high["total_usd"],
        "recommended_hard_authorization_usd": recommended_hard,
        "incremental_storage_per_month_usd": {
            "estimate": 0.0,
            "evidence_class": EvidenceClass.MODELED_OR_FALLBACK,
            "rationale": "Seed-4 checkpoint volume growth is minor relative to execution cost; not included in one-run GPU authorization.",
        },
        "runtime_budget_monitor": {
            "track_h100_seconds_completed": True,
            "track_l40s_seconds_completed": True,
            "track_estimated_billed_cost_usd": True,
            "use_workspace_rates_from": "artifacts/seed4_cost_preflight_modal_rates.json",
            "pre_pipeline_gate": (
                "Before launching each independent model pipeline, verify remaining authorization "
                "covers full historical per-model H100 envelope derived from Seed-1 billed aggregate / 3."
            ),
            "do_not_stop_mid_pipeline_for_estimate_miss": True,
            "fail_closed_on_hard_authorization_breach": True,
        },
    }


def build_shortcut_audit() -> dict[str, Any]:
    rows = [
        ("shared_b_c_trained_trunk", "NOT_ALLOWED", "YES", "B and C LM semantics differ; independent retrain required."),
        ("precision_change", "NOT_ALLOWED", "YES", "Would alter frozen replication protocol."),
        ("gpu_type_change", "NOT_ALLOWED", "YES", "Authoritative training requires H100!; evaluation uses L40S."),
        ("token_budget_reduction", "NOT_ALLOWED", "YES", "Would change 1B LM / 20M safety budgets."),
        ("persistence_1000_early_stop", "NOT_ALLOWED", "YES", "Seed-4 strengthening trajectory requires 4000 steps."),
        ("drop_model_b", "NOT_ALLOWED", "YES", "Model B required for firewall identification."),
        ("eval_reduction", "NOT_ALLOWED", "YES", "Full OOD harmful/benign evaluation set required."),
        ("invalid_unframed_eval", "NOT_ALLOWED", "YES", "Must use canonical format_eval_prompt."),
        ("reuse_materialized_fineweb_shards", "ALLOWED", "NO", "Infrastructure reuse only."),
        ("reuse_wildguard_arrow_records", "ALLOWED", "NO", "Infrastructure reuse only."),
        ("reuse_modal_image_layers", "ALLOWED", "NO", "Reduces cold-start overhead only."),
        ("prestage_data_before_h100", "ALLOWED", "NO", "Avoids idle H100 during preprocessing."),
        ("centralized_wildguard_judge", "ALLOWED", "NO", "Already used in Task 3.1; reduces cold starts."),
        ("fail_closed_duplicate_launch_guard", "ALLOWED", "NO", "Prevents accidental double spend."),
    ]
    return {
        "task": "seed4_cost_preflight_shortcut_audit",
        "decisions": [
            {
                "option": option,
                "decision": decision,
                "scientific_semantics_changed": semantics_changed,
                "notes": notes,
            }
            for option, decision, semantics_changed, notes in rows
        ],
        "protocol_preserving_efficiencies": [
            {
                "name": item[0],
                "expected_cost_impact": "LOW_TO_MODERATE_REDUCTION",
                "scientific_semantics_changed": item[2] == "YES",
                "notes": item[3],
            }
            for item in rows
            if item[1] == "ALLOWED"
        ],
    }


def build_billing_report(root: Path) -> dict[str, Any]:
    preflight_dir = root / "artifacts" / "seed4_cost_preflight"
    raw_files = sorted(preflight_dir.glob("*_billing_raw.json"))
    aggregates: dict[str, Any] = {}
    for path in raw_files:
        rows = load_billing_rows(path)
        aggregates[path.name] = {
            "row_count": len(rows),
            "sha256": sha256_file(path),
            "ccpt_apps": aggregate_billing(rows, apps=CCPT_APPS),
        }

    return {
        "task": "seed4_cost_preflight_billing_report",
        "modal_profile": modal_profile(),
        "modal_cli_version": modal_cli_version(),
        "billing_report_date_range": {
            "primary_window": "2026-08-31 to 2026-09-02",
            "narrow_task2_window_utc": "2026-09-01T03:00:00 to 2026-09-01T05:00:00",
        },
        "raw_report_files": [str(path.relative_to(root)) for path in raw_files],
        "aggregates": aggregates,
        "evidence_class": EvidenceClass.ACTUAL_BILLED,
    }


def write_all_artifacts(root: Path | None = None) -> dict[str, Path]:
    root = repo_root(root)
    rates = fetch_modal_rates(root)

    preflight_dir = root / "artifacts" / "seed4_cost_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    if not (preflight_dir / "relevant_workspace_window_raw.json").exists():
        subprocess.check_call(
            [
                "uv",
                "run",
                "modal",
                "billing",
                "report",
                "--start",
                "2026-08-31",
                "--end",
                "2026-09-02",
                "--show-resources",
                "--json",
            ],
            cwd=root,
            stdout=open(preflight_dir / "relevant_workspace_window_raw.json", "w"),
        )

    window_rows = load_billing_rows(preflight_dir / "relevant_workspace_window_raw.json")
    billing_summary = aggregate_billing(window_rows, apps=CCPT_APPS)

    billing_report = build_billing_report(root)
    actuals = build_actuals(root, rates, billing_summary)
    projection = build_projection(root, rates, actuals)
    shortcut_audit = build_shortcut_audit()

    paths = {
        "billing_report": root / "artifacts" / "seed4_cost_preflight_billing_report.json",
        "actuals": root / "artifacts" / "seed4_cost_preflight_actuals.json",
        "projection": root / "artifacts" / "seed4_cost_preflight_projection.json",
        "shortcut_audit": root / "artifacts" / "seed4_cost_preflight_shortcut_audit.json",
    }

    billing_report["summary"] = billing_summary
    for key, path in paths.items():
        payload = {
            "billing_report": billing_report,
            "actuals": actuals,
            "projection": projection,
            "shortcut_audit": shortcut_audit,
        }[key]
        payload["preflight_code_sha"] = git_head_sha(root)
        payload["h100_gpu_seconds_used_by_preflight"] = H100_GPU_SECONDS_AUTHORIZED
        payload["l40s_gpu_seconds_used_by_preflight"] = L40S_GPU_SECONDS_AUTHORIZED
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    return paths
