"""Resume Seed-4 Model B persistence from step 1000, then corrected eval for B/C/D.

Infrastructure recovery after Modal 7200s timeout. Does not relaunch D or C.
Scientific checkpoint SHA remains e062271...; this script only orchestrates resume + eval.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ccpt.analysis.seed4_execution_ledger import HARD_AUTHORIZATION_USD, SEED4
from orchestrate_strengthening_seed4 import (  # noqa: E402
    _git_sha,
    _write_training_result,
    launch_orphaned_modal,
    load_ledger,
    pre_evaluation_gate,
    query_today_billing_for_apps,
    record_evaluation_stage,
    record_training_stage,
    reload_hard_authorization,
    run_corrected_evaluation,
    save_ledger,
    sync_billed_spend_to_ledger,
    synthesize_post_eval_artifacts,
    volume_seed4_model_terminal_exists,
    wait_for_terminal_checkpoint,
    write_json,
)

SCIENTIFIC_EXECUTION_SHA = "e062271628c3c4434fde6310aa5e0b9024c3dadf"
LOG = ROOT / "artifacts/strengthening_seed4_model_b_resume.log"


def _log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    orch_sha = _git_sha()
    _log("=== Seed-4 Model B persistence resume + eval ===")
    _log(f"SCIENTIFIC_EXECUTION_SHA={SCIENTIFIC_EXECUTION_SHA}")
    _log(f"ORCHESTRATION_SHA={orch_sha}")
    _log("REASON=Modal FunctionTimeoutError at 7200s after persistence_1000; resume 1001→4000")

    if not volume_seed4_model_terminal_exists("model_d"):
        _log("FAIL: model_d persistence_4000 missing")
        return 2
    if not volume_seed4_model_terminal_exists("model_c"):
        _log("FAIL: model_c persistence_4000 missing")
        return 2

    ledger = load_ledger()
    ledger["hard_authorization_usd"] = HARD_AUTHORIZATION_USD
    ledger = sync_billed_spend_to_ledger(reload_hard_authorization(ledger))
    accrued = float(ledger["accrued"]["total_estimated_usd"])
    remaining = float(ledger["hard_authorization_usd"]) - accrued
    rates = ledger["rates"]
    # Remaining work: persistence steps 1001→4000 only (~≤1.1 H100-hr observed) + corrected eval.
    resume_reserve = (1.1 * float(rates["h100_hourly_usd"])) + float(
        ledger["planning_envelopes_usd"]["corrected_eval_total"]
    )
    gate = {
        "accrued_usd": accrued,
        "remaining_authorization_usd": remaining,
        "resume_plus_eval_reserve_usd": resume_reserve,
        "projected_final_usd": accrued + resume_reserve,
        "hard_authorization_usd": HARD_AUTHORIZATION_USD,
        "allowed": accrued + resume_reserve <= HARD_AUTHORIZATION_USD + 1e-9,
    }
    write_json(ROOT / "artifacts/strengthening_seed4_cash_gate_before_model_b_resume.json", gate)
    _log(json.dumps(gate))
    if not gate["allowed"]:
        _log("B_RESUME_BLOCKED_BY_CASH_GATE")
        return 4

    if volume_seed4_model_terminal_exists("model_b"):
        _log("model_b persistence_4000 already present — skip resume launch")
    else:
        # Confirm mid-checkpoint exists
        proc = subprocess.run(
            [
                "uv",
                "run",
                "modal",
                "volume",
                "ls",
                "ccpt-authoritative-runs",
                f"/ccpt/strengthening_task2/seed_{SEED4}/model_b/persistence_1000.pt",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            _log("FAIL: model_b persistence_1000.pt missing; cannot resume")
            return 3

        ledger["stages"].append(
            {
                "stage": "model_b_infrastructure_resume",
                "from_checkpoint": "persistence_1000.pt",
                "to_checkpoint": "persistence_4000.pt",
                "prior_failure": "FunctionTimeoutError 7200s",
                "scientific_execution_sha": SCIENTIFIC_EXECUTION_SHA,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        save_ledger(ledger)

        modal_cmd = [
            "uv",
            "run",
            "modal",
            "run",
            "-d",
            "modal/strengthening_task2_sentinel.py::run_seed4_single_model_training",
            "--model-type",
            "model_b",
            "--expected-code-sha",
            SCIENTIFIC_EXECUTION_SHA,
            "--seed",
            str(SEED4),
        ]
        launch_orphaned_modal(modal_cmd, ROOT / "artifacts/strengthening_seed4_modal_model_b_resume.log")
        _log("Launched Model B resume H100 job; waiting 90s for Modal spin-up")
        time.sleep(90)

        if not wait_for_terminal_checkpoint("model_b", timeout_seconds=4 * 3600, poll_seconds=60):
            _log("FAIL: model_b resume did not produce persistence_4000.pt")
            ledger = sync_billed_spend_to_ledger(reload_hard_authorization(ledger))
            synthesize_post_eval_artifacts(
                execution_sha=SCIENTIFIC_EXECUTION_SHA,
                ledger=ledger,
                judge_summary=None,
                completed_models=["model_d", "model_c"],
                final_status="SEED 4 EXECUTION INVALID — DO NOT USE RESULTS",
            )
            return 5

        result = {
            "seed": SEED4,
            "model_type": "model_b",
            "code_sha": SCIENTIFIC_EXECUTION_SHA,
            "status": "SUCCESS",
            "timing": {"total_h100_seconds": 0.0, "infrastructure_resume_from_1000": True},
            "infrastructure_note": "Resumed after Modal 7200s timeout from persistence_1000",
        }
        _write_training_result(result)
        record_training_stage(
            ledger,
            model_type="model_b",
            training_result=result,
            h100_hourly=float(ledger["rates"]["h100_hourly_usd"]),
        )
        ledger = sync_billed_spend_to_ledger(reload_hard_authorization(ledger))
        _log(f"Model B resume complete. Accrued=${ledger['accrued']['total_estimated_usd']:.4f}")

    completed = ["model_d", "model_b", "model_c"]
    for m in completed:
        if not volume_seed4_model_terminal_exists(m):
            _log(f"FAIL: missing terminal for {m}")
            return 6
    ledger["completed_models"] = completed
    save_ledger(ledger)

    egate = pre_evaluation_gate(reload_hard_authorization(ledger))
    write_json(ROOT / "artifacts/strengthening_seed4_cash_gate_before_eval.json", egate)
    if not egate["allowed"]:
        _log("PARTIAL_EXECUTION_DUE_TO_HARD_CASH_CEILING — training complete, eval blocked")
        synthesize_post_eval_artifacts(
            execution_sha=SCIENTIFIC_EXECUTION_SHA,
            ledger=ledger,
            judge_summary=None,
            completed_models=completed,
            final_status="PARTIAL_EXECUTION_DUE_TO_HARD_CASH_CEILING",
        )
        return 7

    _log("Launching corrected L40S evaluation for B/C/D...")
    judge_summary = run_corrected_evaluation(SCIENTIFIC_EXECUTION_SHA, ["model_b", "model_c", "model_d"])
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
        execution_sha=SCIENTIFIC_EXECUTION_SHA,
        ledger=ledger,
        judge_summary=judge_summary,
        completed_models=completed,
        final_status=final_status,
    )
    _log(final_status)
    return 0 if "COMPLETE" in final_status else 1


if __name__ == "__main__":
    sys.exit(main())
