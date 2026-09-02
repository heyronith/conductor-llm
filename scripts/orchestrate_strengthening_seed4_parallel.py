"""Seed-4 parallel orchestration: D (already running) → [B || C] → corrected eval.

Orchestration-only. Scientific training entrypoints remain frozen at SCIENTIFIC_EXECUTION_SHA.
Does NOT launch or disturb Model D — polls until D terminal checkpoint, then concurrent B+C.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ccpt.analysis.seed4_execution_ledger import (
    HARD_AUTHORIZATION_USD,
    SEED4,
    fetch_workspace_rates,
    load_ledger,
    pre_evaluation_gate,
    record_evaluation_stage,
    record_training_stage,
    save_ledger,
    volume_seed4_model_terminal_exists,
)
from ccpt.config import get_smoke_dual_stream_config
from ccpt.evaluation.forensics import compute_canonical_state_dict_hash
from ccpt.training.engine import create_identical_dual_stream_models

# Import orchestration helpers from sequential controller (no scientific code).
from orchestrate_strengthening_seed4 import (  # noqa: E402
    ROOT,
    _git_sha,
    _load_training_artifact_if_success,
    _training_artifact,
    _write_training_result,
    launch_orphaned_modal,
    query_today_billing_for_apps,
    reload_hard_authorization,
    run_corrected_evaluation,
    sentinel_training_active,
    sync_billed_spend_to_ledger,
    synthesize_post_eval_artifacts,
    wait_for_terminal_checkpoint,
    write_json,
)

SCIENTIFIC_EXECUTION_SHA = "e062271628c3c4434fde6310aa5e0b9024c3dadf"
D_MODAL_APP_ID = "ap-7GuG5S7ZgfLoJQ5jblKaSd"
PARALLEL_LOG = ROOT / "artifacts/strengthening_seed4_parallel_orchestrator.log"

REQUIRED_D_CHECKPOINTS = (
    "lm_1b_final.pt",
    "safety_20m_final.pt",
    "persistence_0000.pt",
    "persistence_0250.pt",
    "persistence_1000.pt",
    "persistence_4000.pt",
)


def _log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PARALLEL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PARALLEL_LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def volume_checkpoint_exists(model_type: str, filename: str) -> bool:
    path = f"/ccpt/strengthening_task2/seed_{SEED4}/{model_type}/{filename}"
    proc = subprocess.run(
        ["uv", "run", "modal", "volume", "ls", "ccpt-authoritative-runs", path],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def verify_d_completion(execution_sha: str) -> dict[str, Any]:
    missing = [f for f in REQUIRED_D_CHECKPOINTS if not volume_checkpoint_exists("model_d", f)]
    if missing:
        raise RuntimeError(f"Model D missing checkpoints on Volume: {missing}")
    return {
        "model_type": "model_d",
        "seed": SEED4,
        "execution_sha": execution_sha,
        "checkpoints_present": list(REQUIRED_D_CHECKPOINTS),
        "terminal": "persistence_4000.pt",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def model_job_active(model_type: str) -> bool:
    """Best-effort: model-specific orphaned Modal client pid still running."""
    pid_path = ROOT / f"artifacts/strengthening_seed4_modal_{model_type}.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            if subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0:
                return True
        except (ValueError, OSError):
            pass
    return False


def verify_bc_init_parity() -> dict[str, Any]:
    cfg = get_smoke_dual_stream_config()
    mb, mc = create_identical_dual_stream_models(cfg, seed=SEED4)
    hb = compute_canonical_state_dict_hash(mb.state_dict())
    hc = compute_canonical_state_dict_hash(mc.state_dict())
    if hb != hc:
        raise RuntimeError(f"B/C init parity failed for seed {SEED4}: {hb} != {hc}")
    return {"hash_b": hb, "hash_c": hc, "parity": "BIT_IDENTICAL", "seed": SEED4}


def pre_bc_parallel_cash_gate(ledger: dict[str, Any]) -> dict[str, Any]:
    """Joint gate: complete B + C + eval + overhead must fit under hard authorization."""
    billing_accrued = float(ledger["accrued"].get("total_estimated_usd") or 0.0)
    remaining = float(ledger["hard_authorization_usd"]) - billing_accrued
    env = ledger["planning_envelopes_usd"]
    per_model = float(env["per_model_h100"])
    eval_total = float(env["corrected_eval_total"])
    other = float(env["other_incremental"]) * (2.0 / 3.0)
    # Parallel wall-clock uses max(B,C) time, but GPU-seconds ≈ B + C (two H100s).
    needed = per_model * 2.0 + eval_total + other
    projected = billing_accrued + needed
    ok = projected <= float(ledger["hard_authorization_usd"]) + 1e-9
    return {
        "accrued_after_d_usd": billing_accrued,
        "remaining_authorization_usd": remaining,
        "reserved_bc_eval_overhead_usd": needed,
        "projected_total_final_usd": projected,
        "hard_authorization_usd": ledger["hard_authorization_usd"],
        "allowed": ok,
        "reason": "OK" if ok else "B_C_PARALLEL_LAUNCH_BLOCKED_BY_CASH_GATE",
        "mode": "parallel_bc_joint",
    }


def launch_model_if_needed(model_type: str, execution_sha: str) -> None:
    if volume_seed4_model_terminal_exists(model_type):
        _log(f"[{model_type}] persistence_4000.pt already present — no launch.")
        return
    if _load_training_artifact_if_success(model_type):
        _log(f"[{model_type}] success artifact already present — no launch.")
        return
    if model_job_active(model_type):
        _log(f"[{model_type}] job appears active — no duplicate launch.")
        return

    modal_cmd = [
        "uv",
        "run",
        "modal",
        "run",
        "-d",
        "modal/strengthening_task2_sentinel.py::run_seed4_single_model_training",
        "--model-type",
        model_type,
        "--expected-code-sha",
        execution_sha,
        "--seed",
        str(SEED4),
    ]
    log_path = ROOT / f"artifacts/strengthening_seed4_modal_{model_type}.log"
    launch_orphaned_modal(modal_cmd, log_path)
    _log(f"[{model_type}] orphaned Modal launch issued.")


def wait_for_model(model_type: str, execution_sha: str) -> dict[str, Any]:
    if volume_seed4_model_terminal_exists(model_type):
        result = {
            "seed": SEED4,
            "model_type": model_type,
            "code_sha": execution_sha,
            "status": "ALREADY_COMPLETE",
            "timing": {"total_h100_seconds": 0.0, "already_complete_no_new_work": True},
        }
        _write_training_result(result)
        return result
    existing = _load_training_artifact_if_success(model_type)
    if existing:
        return existing
    if not wait_for_terminal_checkpoint(model_type):
        raise RuntimeError(f"{model_type} failed to produce persistence_4000.pt")
    artifact = _load_training_artifact_if_success(model_type)
    if artifact:
        return artifact
    result = {
        "seed": SEED4,
        "model_type": model_type,
        "code_sha": execution_sha,
        "status": "SUCCESS",
        "timing": {"total_h100_seconds": 0.0, "completed_on_volume_poll": True},
    }
    _write_training_result(result)
    return result


def wait_for_d_only(execution_sha: str) -> None:
    _log(f"Polling Model D (Modal app {D_MODAL_APP_ID}); will NOT launch D.")
    if volume_seed4_model_terminal_exists("model_d"):
        _log("Model D persistence_4000.pt already on Volume.")
        return
    if not wait_for_terminal_checkpoint("model_d", timeout_seconds=12 * 3600):
        raise RuntimeError("Model D did not complete within poll window.")
    _log("Model D terminal checkpoint observed on Volume.")


def poll_modal_d_status() -> dict[str, Any]:
    proc = subprocess.run(
        ["uv", "run", "modal", "app", "list", "--json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {"status": "UNKNOWN", "app_id": D_MODAL_APP_ID}
    apps = json.loads(proc.stdout)
    for app in apps:
        if app.get("app_id") == D_MODAL_APP_ID:
            return {
                "status": "RUNNING" if int(str(app.get("tasks", "0"))) > 0 else "STOPPED",
                "app_id": D_MODAL_APP_ID,
                "state": app.get("state"),
                "tasks": app.get("tasks"),
            }
    return {"status": "NOT_FOUND", "app_id": D_MODAL_APP_ID}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wait-d-only",
        action="store_true",
        help="Only poll for D completion (default full pipeline after D).",
    )
    args = parser.parse_args()

    orchestration_sha = _git_sha()
    execution_sha = SCIENTIFIC_EXECUTION_SHA
    if execution_sha not in orchestration_sha and orchestration_sha != execution_sha:
        # Orchestration correction may be ahead of frozen scientific SHA on branch tip.
        _log(f"ORCHESTRATION_CORRECTION_SHA={orchestration_sha} (scientific frozen {execution_sha})")

    _log("=== Seed-4 parallel orchestrator start ===")
    _log(f"SCIENTIFIC_EXECUTION_SHA={execution_sha}")
    _log(f"ORCHESTRATION_CORRECTION_SHA={orchestration_sha}")
    _log("SCIENTIFIC_SOURCE_CHANGED=NO ORCHESTRATION_POLICY_CHANGED=YES")

    d_status = poll_modal_d_status()
    _log(f"MODEL D REMOTE: {json.dumps(d_status)}")

    ledger = load_ledger()
    ledger["hard_authorization_usd"] = HARD_AUTHORIZATION_USD
    ledger["execution_sha"] = execution_sha
    ledger["stages"].append(
        {
            "stage": "orchestration_correction",
            "original_policy": "D → B → C sequential",
            "corrected_policy": "D → [B || C] → eval",
            "reason": "Wall-clock reduction; GPU-seconds unchanged",
            "d_modal_app_id": D_MODAL_APP_ID,
            "sequential_orchestrator_auto_advance_disabled": True,
            "orchestration_correction_sha": orchestration_sha,
            "scientific_execution_sha": execution_sha,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    rates = fetch_workspace_rates()
    ledger["rates"]["h100_hourly_usd"] = rates["h100_hourly_usd"]
    ledger["rates"]["l40s_hourly_usd"] = rates["l40s_hourly_usd"]
    save_ledger(ledger)

    wait_for_d_only(execution_sha)
    verify_d_completion(execution_sha)
    ledger = sync_billed_spend_to_ledger(reload_hard_authorization(ledger))
    ledger["completed_models"] = ["model_d"]
    ledger["current_model"] = None
    save_ledger(ledger)
    _log(f"Model D verified. ACCRUED=${ledger['accrued']['total_estimated_usd']:.4f}")

    if args.wait_d_only:
        return 0

    gate = pre_bc_parallel_cash_gate(reload_hard_authorization(ledger))
    write_json(ROOT / "artifacts/strengthening_seed4_cash_gate_before_bc_parallel.json", gate)
    if not gate["allowed"]:
        _log(gate["reason"])
        synthesize_post_eval_artifacts(
            execution_sha=execution_sha,
            ledger=ledger,
            judge_summary=None,
            completed_models=["model_d"],
            final_status="B_C_PARALLEL_LAUNCH_BLOCKED_BY_CASH_GATE",
        )
        return 4

    parity = verify_bc_init_parity()
    write_json(ROOT / "artifacts/strengthening_seed4_bc_init_parity.json", parity)
    _log(f"B/C init parity OK hash={parity['hash_b']}")

    # Concurrent launch guards + spawn.
    for m in ("model_b", "model_c"):
        launch_model_if_needed(m, execution_sha)

    launch_ts = datetime.now(timezone.utc).isoformat()
    write_json(
        ROOT / "artifacts/strengthening_seed4_bc_parallel_launch.json",
        {"model_b": launch_ts, "model_c": launch_ts, "scientific_execution_sha": execution_sha},
    )

    results: dict[str, Any] = {}
    pending = {"model_b", "model_c"}
    deadline = time.time() + 12 * 3600
    while pending and time.time() < deadline:
        for m in list(pending):
            if volume_seed4_model_terminal_exists(m) or _load_training_artifact_if_success(m):
                results[m] = wait_for_model(m, execution_sha)
                record_training_stage(
                    ledger,
                    model_type=m,
                    training_result=results[m],
                    h100_hourly=float(ledger["rates"]["h100_hourly_usd"]),
                )
                ledger = sync_billed_spend_to_ledger(reload_hard_authorization(ledger))
                pending.discard(m)
                _log(f"{m} complete.")
        if pending:
            billing = query_today_billing_for_apps(["strengthening-task2-sentinel"])
            billed = billing.get("total_usd")
            _log(
                f"Waiting for {sorted(pending)}; billed_strengthening=${billed:.4f}"
                if billed is not None
                else f"Waiting for {sorted(pending)}."
            )
            if ledger.get("ceiling_breached"):
                break
            time.sleep(120)

    if pending:
        raise RuntimeError(f"B/C parallel phase incomplete: {sorted(pending)}")

    completed = ["model_d", "model_b", "model_c"]
    ledger["completed_models"] = completed
    save_ledger(ledger)

    egate = pre_evaluation_gate(reload_hard_authorization(ledger))
    write_json(ROOT / "artifacts/strengthening_seed4_cash_gate_before_eval.json", egate)
    if not egate["allowed"]:
        synthesize_post_eval_artifacts(
            execution_sha=execution_sha,
            ledger=ledger,
            judge_summary=None,
            completed_models=completed,
            final_status="PARTIAL_EXECUTION_DUE_TO_HARD_CASH_CEILING",
        )
        return 6

    _log("Launching corrected L40S evaluation...")
    judge_summary = run_corrected_evaluation(execution_sha, ["model_b", "model_c", "model_d"])
    timing = judge_summary.get("timing") or {}
    record_evaluation_stage(
        ledger,
        eval_seconds_by_model=timing.get("eval_seconds_by_model") or {},
        judge_seconds=float(timing.get("judge_l40s_seconds") or judge_summary.get("judge_seconds") or 0.0),
        l40s_hourly=float(ledger["rates"]["l40s_hourly_usd"]),
    )
    ledger = sync_billed_spend_to_ledger(reload_hard_authorization(ledger))

    final_status = "SEED 4 AUTHORITATIVE EXECUTION COMPLETE — READY FOR SCIENTIFIC REVIEW"
    if ledger.get("ceiling_breached"):
        final_status = "BUDGET_HARD_STOP"
    synthesize_post_eval_artifacts(
        execution_sha=execution_sha,
        ledger=ledger,
        judge_summary=judge_summary,
        completed_models=completed,
        final_status=final_status,
    )
    _log(final_status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
