"""Cash-controlled Seed-4 authoritative orchestrator (D → B → C, then corrected eval).

Hard ceiling: $35.00 out-of-pocket (operator-authorized 2026-09-02). Credits = $0 ⇒ metered = OOP.
No H100 spawn when valid persistence_4000.pt already exists on Volume.
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

from ccpt.analysis.seed4_execution_ledger import (
    HARD_AUTHORIZATION_USD,
    MODEL_ORDER,
    SEED4,
    fetch_workspace_rates,
    load_ledger,
    pre_evaluation_gate,
    pre_pipeline_gate,
    record_evaluation_stage,
    record_training_stage,
    save_ledger,
    volume_seed4_model_terminal_exists,
)
from ccpt.analysis.seed4_execution_summaries import (
    build_ablation_summary,
    build_behavior_summary,
    build_retention_summary,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip().lower()


def _run(cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout)


def query_today_billing_for_apps(app_name_substrings: list[str]) -> dict[str, Any]:
    proc = _run(["uv", "run", "modal", "billing", "report", "--for", "today", "--show-resources", "--json"])
    if proc.returncode != 0:
        return {"status": "UNAVAILABLE", "stderr": proc.stderr[-2000:], "total_usd": None}
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "PARSE_ERROR", "raw_tail": proc.stdout[-2000:], "total_usd": None}

    # Modal billing report schema can vary; best-effort extract.
    total = 0.0
    matched = []
    items = report if isinstance(report, list) else report.get("items") or report.get("rows") or []
    if isinstance(report, dict) and not items:
        # Sometimes nested under workspaces / days
        items = report.get("resources") or report.get("apps") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("app_name") or item.get("description") or item.get("name") or "")
        if any(s in name for s in app_name_substrings):
            cost = item.get("cost") or item.get("amount") or item.get("total_cost") or 0
            try:
                cost_f = float(cost)
            except (TypeError, ValueError):
                cost_f = 0.0
            total += cost_f
            matched.append({"name": name, "cost": cost_f, "raw": item})
    return {
        "status": "OK",
        "matched": matched,
        "total_usd": total,
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def reload_hard_authorization(ledger: dict[str, Any]) -> dict[str, Any]:
    """Pick up operator authorization changes written to the live ledger file."""
    on_disk = load_ledger()
    ledger["hard_authorization_usd"] = float(on_disk["hard_authorization_usd"])
    billed = float(ledger["accrued"].get("total_estimated_usd") or 0.0)
    ledger["remaining_authorization_usd"] = float(ledger["hard_authorization_usd"]) - billed
    ledger["ceiling_breached"] = billed > float(ledger["hard_authorization_usd"])
    return ledger


def sentinel_training_active() -> bool:
    """True if any strengthening-task2-sentinel Modal app has active tasks."""
    proc = subprocess.run(
        ["uv", "run", "modal", "app", "list", "--json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False
    try:
        apps = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False
    for app in apps:
        if "strengthening-task2-sentinel" not in str(app.get("description", "")):
            continue
        state = str(app.get("state", "")).lower()
        if state == "stopped":
            continue
        try:
            tasks = int(str(app.get("tasks", "0")))
        except ValueError:
            tasks = 0
        if tasks > 0:
            return True
    return False


# Backward-compatible alias
def sentinel_detached_training_active() -> bool:
    return sentinel_training_active()


def launch_orphaned_modal(cmd: list[str], log_path: Path) -> int:
    """Start Modal CLI in a new session so orchestrator restarts do not cancel H100."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n--- launch {datetime.now(timezone.utc).isoformat()} ---\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path = log_path.with_suffix(".pid")
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    print(f"Orphaned Modal client pid={proc.pid} log={log_path}", flush=True)
    return proc.pid


def _training_artifact(model_type: str) -> Path:
    return ROOT / f"artifacts/strengthening_seed4_training_{model_type}.json"


def _load_training_artifact_if_success(model_type: str) -> dict[str, Any] | None:
    art = _training_artifact(model_type)
    if not art.exists():
        return None
    try:
        data = json.loads(art.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if data.get("status") in ("SUCCESS", "ALREADY_COMPLETE"):
        return data
    return None


def wait_for_terminal_checkpoint(
    model_type: str,
    *,
    timeout_seconds: int = 10 * 3600,
    poll_seconds: int = 120,
) -> bool:
    """Poll Volume / local artifact until model pipeline completes or Modal job ends."""
    deadline = time.time() + timeout_seconds
    idle_grace_started: float | None = None
    while time.time() < deadline:
        if volume_seed4_model_terminal_exists(model_type):
            print(f"[{model_type}] persistence_4000.pt found on Volume.", flush=True)
            return True
        artifact = _load_training_artifact_if_success(model_type)
        if artifact is not None:
            print(f"[{model_type}] Training artifact reports {artifact.get('status')}.", flush=True)
            return True
        if sentinel_training_active():
            idle_grace_started = None
            billing = query_today_billing_for_apps(["strengthening-task2-sentinel"])
            billed = billing.get("total_usd")
            print(
                f"[{model_type}] Modal training active; billed_strengthening=${billed:.4f}"
                if billed is not None
                else f"[{model_type}] Modal training active.",
                flush=True,
            )
        else:
            if idle_grace_started is None:
                idle_grace_started = time.time()
            elif time.time() - idle_grace_started > 180:
                print(
                    f"[{model_type}] Modal app stopped without terminal checkpoint/artifact.",
                    flush=True,
                )
                return False
        time.sleep(poll_seconds)
    print(f"[{model_type}] Timed out waiting for persistence_4000.pt.", flush=True)
    return False


def _write_training_result(result: dict[str, Any]) -> None:
    art = _training_artifact(result["model_type"])
    with open(art, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def train_model(model_type: str, execution_sha: str) -> dict[str, Any]:
    if volume_seed4_model_terminal_exists(model_type):
        print(f"[{model_type}] persistence_4000.pt already on Volume — skipping H100 spawn.", flush=True)
        result = {
            "seed": SEED4,
            "model_type": model_type,
            "code_sha": execution_sha,
            "status": "ALREADY_COMPLETE",
            "timing": {
                "lm_pretrain_seconds": 0.0,
                "safety_train_seconds": 0.0,
                "persistence_train_seconds": 0.0,
                "total_h100_seconds": 0.0,
                "already_complete_no_new_work": True,
            },
            "skipped_h100_allocation": True,
        }
        _write_training_result(result)
        return result

    existing = _load_training_artifact_if_success(model_type)
    if existing is not None:
        return existing

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

    if not sentinel_training_active():
        launch_orphaned_modal(modal_cmd, log_path)
    else:
        print(f"[{model_type}] strengthening-task2-sentinel already active — polling only.", flush=True)

    if wait_for_terminal_checkpoint(model_type):
        artifact = _load_training_artifact_if_success(model_type)
        if artifact is not None:
            return artifact
        result = {
            "seed": SEED4,
            "model_type": model_type,
            "code_sha": execution_sha,
            "status": "SUCCESS",
            "timing": {
                "total_h100_seconds": 0.0,
                "completed_on_volume_poll": True,
            },
            "infrastructure_note": "Terminal checkpoint verified on Volume; timing from Modal billing.",
        }
        _write_training_result(result)
        return result

    raise RuntimeError(
        f"Training for {model_type} ended without persistence_4000.pt. "
        f"See {log_path} and Modal app logs."
    )


def sync_billed_spend_to_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    """Use Modal billing as authoritative accrued spend when available."""
    billing = query_today_billing_for_apps(
        ["strengthening-task2-sentinel", "strengthening-task3-1-eval"]
    )
    ledger["stages"].append({"stage": "billing_sync", "billing": billing})
    if billing.get("total_usd") is not None:
        billed = float(billing["total_usd"])
        ledger["accrued"]["total_billed_usd_when_available"] = billed
        ledger["accrued"]["total_estimated_usd"] = billed
        ledger["remaining_authorization_usd"] = float(ledger["hard_authorization_usd"]) - billed
        ledger["ceiling_breached"] = billed > float(ledger["hard_authorization_usd"])
    return ledger


def pre_pipeline_gate_resume(ledger: dict[str, Any], next_model: str) -> dict[str, Any]:
    """Resume gate: allow next model if remaining auth covers its H100 envelope (sequential)."""
    billing_accrued = float(ledger["accrued"].get("total_estimated_usd") or 0.0)
    remaining_auth = float(ledger["hard_authorization_usd"]) - billing_accrued
    envelopes = ledger["planning_envelopes_usd"]
    per_model = float(envelopes["per_model_h100"])
    order = ledger["model_order"]
    idx = order.index(next_model)
    tail_models = order[idx:]
    eval_done = any(s.get("stage") == "evaluation_complete" for s in ledger.get("stages", []))
    eval_reserve = 0.0 if eval_done else float(envelopes["corrected_eval_total"])
    tail_h100 = per_model * len(tail_models)
    tail_total = tail_h100 + eval_reserve
    projected_if_full_tail = billing_accrued + tail_total
    # Launch this model if we can afford its envelope; full-tail projection is advisory only.
    ok = remaining_auth >= per_model and (billing_accrued + per_model) <= float(
        ledger["hard_authorization_usd"]
    ) + 1e-9
    return {
        "next_model": next_model,
        "accrued_usd": billing_accrued,
        "remaining_authorization_usd": remaining_auth,
        "per_model_h100_envelope_usd": per_model,
        "reserved_for_full_tail_usd": tail_total,
        "projected_if_full_tail_usd": projected_if_full_tail,
        "hard_authorization_usd": ledger["hard_authorization_usd"],
        "allowed": ok,
        "reason": (
            "OK"
            if ok
            else (
                f"Remaining ${remaining_auth:.2f} < per-model envelope ${per_model:.2f} "
                f"or next launch would exceed ${ledger['hard_authorization_usd']:.2f}."
            )
        ),
        "mode": "resume_sequential",
        "full_protocol_feasible": projected_if_full_tail <= float(ledger["hard_authorization_usd"]) + 1e-9,
    }


def wait_for_eval_summary(*, timeout_seconds: int = 3 * 3600, poll_seconds: int = 60) -> bool:
    art = ROOT / "artifacts/strengthening_seed4_task3_1_summary.json"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if art.exists():
            try:
                data = json.loads(art.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = None
            if data and data.get("summary"):
                return True
        time.sleep(poll_seconds)
    return False


def run_corrected_evaluation(execution_sha: str, models: list[str]) -> dict[str, Any]:
    model_types = ",".join(models)
    art = ROOT / "artifacts/strengthening_seed4_task3_1_summary.json"
    if art.exists():
        try:
            existing = json.loads(art.read_text(encoding="utf-8"))
            if existing.get("summary"):
                return existing
        except json.JSONDecodeError:
            pass

    modal_cmd = [
        "uv",
        "run",
        "modal",
        "run",
        "-d",
        "modal/strengthening_task3_1_eval.py::run_seed4_corrected_evaluation",
        "--expected-code-sha",
        execution_sha,
        "--model-types",
        model_types,
    ]
    log_path = ROOT / "artifacts/strengthening_seed4_modal_eval.log"
    launch_orphaned_modal(modal_cmd, log_path)
    if not wait_for_eval_summary():
        raise RuntimeError(f"Evaluation failed without summary artifact. See {log_path}.")
    payload = json.loads(art.read_text(encoding="utf-8"))
    return payload


def build_checkpoint_manifest(completed: list[str], execution_sha: str) -> dict[str, Any]:
    files = [
        "lm_1b_final.pt",
        "safety_20m_final.pt",
        "persistence_0000.pt",
        "persistence_0250.pt",
        "persistence_1000.pt",
        "persistence_4000.pt",
    ]
    models: dict[str, Any] = {}
    for m in completed:
        present = {}
        for fname in files:
            path = f"/ccpt/strengthening_task2/seed_{SEED4}/{m}/{fname}"
            proc = subprocess.run(
                ["uv", "run", "modal", "volume", "ls", "ccpt-authoritative-runs", path],
                capture_output=True,
                text=True,
            )
            present[fname] = proc.returncode == 0
        models[m] = present
    return {
        "task": "strengthening_seed4_checkpoint_manifest",
        "seed": SEED4,
        "execution_sha": execution_sha,
        "models": models,
    }


def synthesize_post_eval_artifacts(
    *,
    execution_sha: str,
    ledger: dict[str, Any],
    judge_summary: dict[str, Any] | None,
    completed_models: list[str],
    final_status: str,
) -> None:
    write_json(
        ROOT / "artifacts/strengthening_seed4_execution_manifest.json",
        {
            "task": "strengthening_seed4_execution_manifest",
            "seed": SEED4,
            "execution_sha": execution_sha,
            "model_order": list(MODEL_ORDER),
            "completed_models": completed_models,
            "final_status": final_status,
            "hard_authorization_usd": HARD_AUTHORIZATION_USD,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(ROOT / "artifacts/strengthening_seed4_checkpoint_manifest.json", build_checkpoint_manifest(completed_models, execution_sha))

    training_summary = {
        "task": "strengthening_seed4_training_summary",
        "seed": SEED4,
        "execution_sha": execution_sha,
        "completed_models": completed_models,
        "per_model": {},
    }
    for m in completed_models:
        p = ROOT / f"artifacts/strengthening_seed4_training_{m}.json"
        if p.exists():
            training_summary["per_model"][m] = json.loads(p.read_text(encoding="utf-8"))
    write_json(ROOT / "artifacts/strengthening_seed4_training_summary.json", training_summary)

    if judge_summary is not None:
        capability = (judge_summary.get("timing") or {}).get("capability_by_model") or {}
        behavior = build_behavior_summary(judge_summary, capability)
        ablation = build_ablation_summary(judge_summary)
        retention = build_retention_summary(behavior)
        write_json(ROOT / "artifacts/strengthening_seed4_behavior_summary.json", behavior)
        write_json(ROOT / "artifacts/strengthening_seed4_ablation_summary.json", ablation)
        write_json(ROOT / "artifacts/strengthening_seed4_retention_summary.json", retention)

    cost = {
        "task": "strengthening_seed4_cost_summary",
        "seed": SEED4,
        "execution_sha": execution_sha,
        "hard_authorization_usd": HARD_AUTHORIZATION_USD,
        "ceiling_breached": ledger.get("ceiling_breached"),
        "accrued": ledger.get("accrued"),
        "rates": ledger.get("rates"),
        "stages": ledger.get("stages"),
        "remaining_authorization_usd": ledger.get("remaining_authorization_usd"),
        "final_status": final_status,
    }
    write_json(ROOT / "artifacts/strengthening_seed4_cost_summary.json", cost)
    save_ledger(ledger)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume after abort: sync billed spend, use billing-aware gates, keep Modal jobs on disconnect.",
    )
    args = parser.parse_args()

    execution_sha = _git_sha()
    print(f"SEED4_EXECUTION_SHA={execution_sha}", flush=True)

    # Ensure preflight artifact + ledger exist
    preflight_p = ROOT / "artifacts" / "strengthening_seed4_preflight.json"
    if not preflight_p.exists():
        print("Running zero-GPU preflight first...", flush=True)
        proc = _run(["uv", "run", "python", "scripts/run_strengthening_seed4_preflight.py"])
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return 2

    preflight = json.loads(preflight_p.read_text(encoding="utf-8"))
    if preflight.get("overall_status") != "PASSED":
        print("PREFLIGHT FAILED — STOP BEFORE GPU", flush=True)
        return 2
    if not preflight["checks"]["cash_gate_pre_first_h100"]["allowed_to_launch_first_h100"]:
        print("CASH GATE FAILED — STOP BEFORE GPU", flush=True)
        return 2

    ledger = load_ledger()
    rates = fetch_workspace_rates()
    ledger["rates"]["h100_hourly_usd"] = rates["h100_hourly_usd"]
    ledger["rates"]["l40s_hourly_usd"] = rates["l40s_hourly_usd"]
    ledger["execution_sha"] = execution_sha
    if args.resume:
        ledger = sync_billed_spend_to_ledger(ledger)
        ledger["stages"].append(
            {
                "stage": "resume_after_abort",
                "reason": "Prior model_d run cancelled by local client disconnect (~LM 25k/30517).",
                "prior_waste_usd": ledger["accrued"].get("total_estimated_usd"),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    save_ledger(ledger)

    gate_fn = pre_pipeline_gate if float(ledger.get("hard_authorization_usd", HARD_AUTHORIZATION_USD)) >= 35.0 else (
        pre_pipeline_gate_resume if args.resume else pre_pipeline_gate
    )

    completed: list[str] = list(ledger.get("completed_models") or [])
    stopped_reason = None

    for model in MODEL_ORDER:
        if model in completed:
            print(f"[{model}] already recorded complete in ledger; skipping.", flush=True)
            continue

        gate = gate_fn(reload_hard_authorization(ledger), model)
        ledger = reload_hard_authorization(ledger)
        write_json(ROOT / f"artifacts/strengthening_seed4_cash_gate_before_{model}.json", gate)
        if not gate["allowed"]:
            print(gate["reason"], flush=True)
            stopped_reason = "PARTIAL_EXECUTION_DUE_TO_HARD_CASH_CEILING"
            ledger["budget_hard_stop"] = True
            ledger["stages"].append({"stage": "cash_gate_block", "gate": gate})
            save_ledger(ledger)
            break

        ledger["current_model"] = model
        save_ledger(ledger)
        print(f"=== Launching {model} (H100!, detached) ===", flush=True)
        try:
            result = train_model(model, execution_sha)
        except Exception as exc:  # noqa: BLE001
            ledger = sync_billed_spend_to_ledger(ledger)
            ledger["stages"].append(
                {
                    "stage": f"training_{model}_failed",
                    "error": str(exc),
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            save_ledger(ledger)
            # Do not hard-stop on disconnect if Volume poll might still succeed later.
            if volume_seed4_model_terminal_exists(model):
                result = {
                    "seed": SEED4,
                    "model_type": model,
                    "code_sha": execution_sha,
                    "status": "SUCCESS",
                    "timing": {"total_h100_seconds": 0.0, "completed_on_volume_after_failure": True},
                }
            else:
                stopped_reason = "SEED 4 EXECUTION INVALID — DO NOT USE RESULTS"
                synthesize_post_eval_artifacts(
                    execution_sha=execution_sha,
                    ledger=ledger,
                    judge_summary=None,
                    completed_models=completed,
                    final_status=stopped_reason,
                )
                print(stopped_reason, flush=True)
                return 3

        record_training_stage(
            ledger,
            model_type=model,
            training_result=result,
            h100_hourly=float(ledger["rates"]["h100_hourly_usd"]),
        )
        ledger = sync_billed_spend_to_ledger(ledger)
        if ledger["ceiling_breached"]:
            ledger["budget_hard_stop"] = True
            stopped_reason = "BUDGET_HARD_STOP"
            save_ledger(ledger)
            break

        billing = query_today_billing_for_apps(
            ["strengthening-task2-sentinel", "strengthening-task3-1-eval"]
        )
        ledger["stages"].append({"stage": f"billing_after_{model}", "billing": billing})
        if billing.get("total_usd") is not None:
            ledger["accrued"]["total_billed_usd_when_available"] = billing["total_usd"]
        save_ledger(ledger)
        completed.append(model)
        print(f"=== {model} complete. Accrued est ${ledger['accrued']['total_estimated_usd']:.4f} ===", flush=True)

    judge_summary = None
    if stopped_reason is None and completed:
        egate = pre_evaluation_gate(reload_hard_authorization(ledger))
        write_json(ROOT / "artifacts/strengthening_seed4_cash_gate_before_eval.json", egate)
        if not egate["allowed"]:
            print(egate["reason"], flush=True)
            stopped_reason = "PARTIAL_EXECUTION_DUE_TO_HARD_CASH_CEILING"
        else:
            print("=== Launching corrected L40S evaluation ===", flush=True)
            judge_summary = run_corrected_evaluation(execution_sha, completed)
            timing = judge_summary.get("timing") or {}
            record_evaluation_stage(
                ledger,
                eval_seconds_by_model=timing.get("eval_seconds_by_model") or {},
                judge_seconds=float(timing.get("judge_l40s_seconds") or judge_summary.get("judge_seconds") or 0.0),
                l40s_hourly=float(ledger["rates"]["l40s_hourly_usd"]),
            )
            billing = query_today_billing_for_apps(
                ["strengthening-task2-sentinel", "strengthening-task3-1-eval"]
            )
            ledger["stages"].append({"stage": "billing_after_evaluation", "billing": billing})
            if billing.get("total_usd") is not None:
                ledger["accrued"]["total_billed_usd_when_available"] = billing["total_usd"]
            save_ledger(ledger)

    if stopped_reason is None:
        if set(completed) == set(MODEL_ORDER) and judge_summary is not None:
            final_status = "SEED 4 AUTHORITATIVE EXECUTION COMPLETE — READY FOR SCIENTIFIC REVIEW"
        else:
            final_status = "PARTIAL_EXECUTION_DUE_TO_HARD_CASH_CEILING"
    else:
        final_status = stopped_reason

    synthesize_post_eval_artifacts(
        execution_sha=execution_sha,
        ledger=ledger,
        judge_summary=judge_summary,
        completed_models=completed,
        final_status=final_status,
    )
    print(final_status, flush=True)
    return 0 if "COMPLETE" in final_status or "PARTIAL" in final_status else 1


if __name__ == "__main__":
    sys.exit(main())
