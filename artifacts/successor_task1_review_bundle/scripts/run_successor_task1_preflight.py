#!/usr/bin/env python3
"""Zero-GPU preflight for successor Task 1 adaptive-interface falsification."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccpt.config import get_smoke_dual_stream_config, get_micro_dual_stream_config
from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.successor.calibration import assert_zero_eval_overlap, build_successor_calibration_reference
from ccpt.successor.cohort import resolve_cohort, write_cohort_artifact
from ccpt.successor.partition import build_parameter_partition
from ccpt.successor.retrofit import build_variant_bundle

HARD_AUTHORIZATION_USD = 5.00
TARGET_USD = 3.00
ART = ROOT / "artifacts"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()


def _fetch_l40s_rate() -> float:
    try:
        raw = subprocess.check_output(
            ["uv", "run", "modal", "billing", "rates", "--json"], text=True, cwd=ROOT
        )
        rates = json.loads(raw)
        return float(rates["gpu_hour_cost_l40s"])
    except Exception as e:  # noqa: BLE001
        print(f"WARN: could not fetch Modal rates ({e}); using documented 1.95", flush=True)
        return 1.95


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    now = datetime.now(timezone.utc).isoformat()

    # Cohort
    print("Resolving checkpoint cohort on Modal volume...", flush=True)
    cohort = resolve_cohort(check_volume=True)
    write_cohort_artifact(ART / "successor_task1_checkpoint_cohort.json", cohort)
    print(
        f"primary_valid_pairs={cohort['primary_valid_pairs']} seeds={cohort['primary_seeds']}",
        flush=True,
    )

    # Calibration
    cal = build_successor_calibration_reference()
    assert_zero_eval_overlap(cal)
    (ART / "successor_task1_calibration_manifest.json").write_text(json.dumps(cal, indent=2) + "\n")

    # Parameter partition + adapter budgets (smoke geometry = production)
    model = CCPTDualStreamModel(get_smoke_dual_stream_config())
    part = build_parameter_partition(model)
    (ART / "successor_task1_parameter_partition.json").write_text(json.dumps(part, indent=2) + "\n")
    bundle = build_variant_bundle(model, observer_rank=32, actuator_rank=32)
    configs = {
        "task": "successor_task1_adapter_configs",
        "created_at_utc": now,
        "code_sha_at_preflight": sha,
        "observer_rank": 32,
        "actuator_rank": 32,
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "training_steps": 500,
            "risk_loss_weight": 0.1,
        },
        "budget": bundle,
    }
    (ART / "successor_task1_adapter_configs.json").write_text(json.dumps(configs, indent=2) + "\n")

    if not bundle["within_1pct_budget"]:
        print("STOP: combined repair exceeds 1% of base parameters", flush=True)
        return 2
    if not bundle["generic_within_1pct_match"]:
        print("STOP: generic parameter match error > 1%", flush=True)
        return 3

    l40s = _fetch_l40s_rate()
    # Conservative envelope: 4 seeds × 4 variants × ~12 min fit + eval ~90 min total L40S
    # Prefer target $3; hard stop $5.
    projected_l40s_hours = 1.4
    projected_usd = projected_l40s_hours * l40s
    # also micro sanity that geometry works
    _ = CCPTDualStreamModel(get_micro_dual_stream_config())

    preflight = {
        "task": "successor_task1_preflight",
        "created_at_utc": now,
        "SUCCESSOR_BASE_SHA_hint": sha,
        "primary_valid_pairs": cohort["primary_valid_pairs"],
        "primary_seeds": cohort["primary_seeds"],
        "gpu_allowed_by_cohort": cohort["gpu_allowed"],
        "calibration_overlap": cal["CALIBRATION_FINAL_TEST_OVERLAP"],
        "combined_percent_of_base": bundle["combined_percent_of_base"],
        "generic_match_error_percent": bundle["generic_match_error_percent"],
        "hard_authorization_usd": HARD_AUTHORIZATION_USD,
        "target_usd": TARGET_USD,
        "l40s_hourly_usd": l40s,
        "projected_l40s_hours_conservative": projected_l40s_hours,
        "projected_usd_conservative": projected_usd,
        "h100_authorized": False,
        "within_hard_authorization": projected_usd <= HARD_AUTHORIZATION_USD,
        "within_target": projected_usd <= TARGET_USD,
        "stop_before_gpu": (not cohort["gpu_allowed"])
        or cal["CALIBRATION_FINAL_TEST_OVERLAP"] != 0
        or not bundle["within_1pct_budget"]
        or projected_usd > HARD_AUTHORIZATION_USD,
    }
    (ART / "successor_task1_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print(json.dumps(preflight, indent=2), flush=True)

    if preflight["stop_before_gpu"]:
        print("SUCCESSOR_TASK1_PREFLIGHT_BLOCK", flush=True)
        return 4
    print("SUCCESSOR_TASK1_PREFLIGHT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
