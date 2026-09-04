#!/usr/bin/env python3
"""Cost-gated orchestrator for successor Task 1 (L40S only; hard $5)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccpt.successor.calibration import build_successor_calibration_reference, load_calibration_manifest
from ccpt.successor.cohort import resolve_cohort
from ccpt.successor.criteria import assess_hypothesis

HARD_AUTHORIZATION_USD = 5.00  # IMMUTABLE
ART = ROOT / "artifacts"
VOLUME = "ccpt-authoritative-runs"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()


def _run(cmd: List[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout)


def fetch_l40s_rate() -> float:
    proc = _run(["uv", "run", "modal", "billing", "rates", "--json"])
    rates = json.loads(proc.stdout)
    return float(rates["gpu_hour_cost_l40s"])


def query_app_billing_today() -> float:
    proc = _run(
        ["uv", "run", "modal", "billing", "report", "--for", "today", "--show-resources", "--json"]
    )
    if proc.returncode != 0:
        return 0.0
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return 0.0
    total = 0.0
    for r in rows if isinstance(rows, list) else []:
        if "successor-task1" in str(r.get("description", "")):
            total += float(r.get("cost") or 0)
    return total


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def upload_calibration_fit_records() -> Dict[str, Any]:
    cal_ref = build_successor_calibration_reference()
    man = load_calibration_manifest()
    held = int(cal_ref["held_out_diagnostic_count"])
    fit_records = man["records"][held:]
    local = ART / "successor_task1_calibration_fit_records.jsonl"
    with local.open("w") as f:
        for r in fit_records:
            f.write(json.dumps(r) + "\n")
    # put on volume
    remote = "ccpt/successor_task1/calibration_fit_records.jsonl"
    _run(["uv", "run", "modal", "volume", "put", VOLUME, str(local), remote])
    return cal_ref


def cost_gate(accrued: float, projected_next: float) -> Dict[str, Any]:
    ok = (accrued + projected_next) <= HARD_AUTHORIZATION_USD + 1e-9
    return {
        "accrued_usd": accrued,
        "projected_next_usd": projected_next,
        "hard_authorization_usd": HARD_AUTHORIZATION_USD,
        "allowed": ok,
    }


def main() -> int:
    code_sha = _git_sha()
    print(f"SUCCESSOR_TASK1_CODE_SHA={code_sha}", flush=True)

    cohort = resolve_cohort(check_volume=True)
    write_json(ART / "successor_task1_checkpoint_cohort.json", cohort)
    if not cohort["gpu_allowed"]:
        print("STOP: fewer than 4 valid PRE/POST pairs", flush=True)
        return 2

    cal_ref = upload_calibration_fit_records()
    write_json(ART / "successor_task1_calibration_manifest.json", cal_ref)
    if cal_ref["CALIBRATION_FINAL_TEST_OVERLAP"] != 0:
        print("STOP: calibration overlap", flush=True)
        return 3

    rates = {"l40s_hourly_usd": fetch_l40s_rate()}
    # Conservative per-seed fit envelope (~0.35 L40S-hr)
    per_seed_fit_usd = 0.35 * rates["l40s_hourly_usd"]
    accrued = query_app_billing_today()
    ledger = {
        "hard_authorization_usd": HARD_AUTHORIZATION_USD,
        "rates": rates,
        "accrued_usd": accrued,
        "stages": [],
        "h100_gpu_seconds": 0,
    }

    adapter_cfg = json.loads((ART / "successor_task1_adapter_configs.json").read_text())
    training_cfg = adapter_cfg["training"]
    training_cfg["max_fit_records"] = 256
    training_cfg["teacher_continuation_tokens"] = 32
    ranks = {
        "observer_rank": adapter_cfg["observer_rank"],
        "actuator_rank": adapter_cfg["actuator_rank"],
    }

    fit_results = []
    for pair in cohort["primary_pairs"]:
        if not pair["pair_valid"]:
            continue
        seed = pair["seed"]
        gate = cost_gate(accrued, per_seed_fit_usd)
        write_json(ART / f"successor_task1_cash_gate_before_seed_{seed}.json", gate)
        if not gate["allowed"]:
            print("SUCCESSOR_TASK1_BLOCKED_BY_COST_GATE", flush=True)
            ledger["stages"].append({"stage": "blocked", "gate": gate})
            write_json(ART / "successor_task1_cost_summary.json", ledger)
            return 4

        # Invoke Modal function via modal run
        # Pass args through a temp json consumed by a thin runner
        payload = {
            "seed": seed,
            "pre_rel": pair["pre"]["volume_path"],
            "post_rel": pair["post_1000"]["volume_path"],
            "code_sha": code_sha,
            "calibration": cal_ref,
            "training_cfg": training_cfg,
            "adapter_ranks": ranks,
        }
        payload_path = ART / f"successor_task1_payload_seed_{seed}.json"
        write_json(payload_path, payload)

        cmd = [
            "uv",
            "run",
            "modal",
            "run",
            "modal/successor_task1_falsification.py::fit_and_eval_one_seed",
            "--seed",
            str(seed),
            "--pre-rel",
            pair["pre"]["volume_path"],
            "--post-rel",
            pair["post_1000"]["volume_path"],
            "--code-sha",
            code_sha,
        ]
        # modal run with complex dicts is awkward; use a helper entrypoint
        print(f"Launching fit seed={seed}", flush=True)
        # Direct python modal invocation via written runner
        runner = ROOT / "scripts" / "_successor_task1_modal_fit_seed.py"
        proc = _run(
            [
                "uv",
                "run",
                "python",
                str(runner),
                str(payload_path),
            ],
            timeout=4 * 3600,
        )
        print(proc.stdout[-4000:] if proc.stdout else "", flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-4000:], flush=True)
            ledger["stages"].append({"stage": f"fit_failed_{seed}", "rc": proc.returncode})
            write_json(ART / "successor_task1_cost_summary.json", ledger)
            return 5

        result_path = ART / f"successor_task1_fit_result_seed_{seed}.json"
        if result_path.exists():
            fit_results.append(json.loads(result_path.read_text()))
        accrued = query_app_billing_today()
        ledger["accrued_usd"] = accrued
        ledger["stages"].append({"stage": f"fit_seed_{seed}", "accrued_usd": accrued})

    write_json(ART / "successor_task1_training_summary.json", {"fits": fit_results, "code_sha": code_sha})
    write_json(ART / "successor_task1_cost_summary.json", ledger)
    print("FIT_PHASE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
