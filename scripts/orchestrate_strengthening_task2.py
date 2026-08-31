"""Authoritative orchestration script for CCPT Strengthening Round Task 2 Sentinel Execution.

Executes:
1. Preflight and Lineage validation
2. Staged execution topology:
   - Seed 1 (B || C || D) on 3 x H100!
   - Seed 1 Technical Health Gate & Smoke Eval
   - Seed 4 (B || C || D) on 3 x H100!
   - Seed 1 & Seed 4 Evaluation Workers (L40S)
   - Centralized WildGuard 7B Moderation Judge (L40S)
3. Full artifact generation, behavioral metrics, and cost accounting.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import modal

app = modal.App("strengthening-task2-orchestrator")


def get_git_sha() -> str:
    res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return res.stdout.strip()


def run_staged_sentinel_experiment(code_sha: str) -> Dict[str, Any]:
    sys.path.insert(0, str(Path("modal").resolve()))
    from strengthening_task2_sentinel import (
        run_strengthening_single_model_training,
        run_strengthening_eval_smoke,
        run_strengthening_evaluation_worker,
        run_strengthening_centralized_judge,
    )

    t0_sentinel = time.time()
    results = {
        "task_name": "CCPT_STRENGTHENING_TASK2_SENTINEL",
        "execution_sha": code_sha,
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "seed_1_training": {},
        "seed_4_training": {},
        "technical_health_gate": {},
        "evaluation": {},
        "judging": {},
        "cost_accounting": {},
    }

    # =========================================================================
    # STAGE 1: Launch Seed 1 (20260821) B || C || D on 3 x H100!
    # =========================================================================
    seed_1 = 20260821
    print(f"\n==================================================", flush=True)
    print(f"STAGE 1: Launching Seed 1 ({seed_1}) Models B, C, D concurrently on H100!", flush=True)
    print(f"==================================================", flush=True)

    models = ["model_b", "model_c", "model_d"]
    t0_s1 = time.time()

    # Launch concurrently using modal.Function.spawn()
    s1_handles = {
        m: run_strengthening_single_model_training.spawn(
            seed=seed_1,
            model_type=m,
            expected_code_sha=code_sha,
        )
        for m in models
    }

    print(f"Spawned Seed 1 jobs: {[f'{m}:{h.object_id}' for m, h in s1_handles.items()]}")

    # Gather results
    s1_results = {}
    for m, handle in s1_handles.items():
        print(f"Waiting for Seed 1 {m}...")
        res = handle.get()
        s1_results[m] = res
        print(f"-> Seed 1 {m} completed! H100 GPU seconds: {res['timing']['total_h100_seconds']:.1f}")

    results["seed_1_training"] = s1_results
    s1_elapsed = time.time() - t0_s1
    print(f"Stage 1 complete in {s1_elapsed:.1f}s wall clock.", flush=True)

    # =========================================================================
    # STAGE 2: Technical Health Gate & Small Evaluation Smoke
    # =========================================================================
    print(f"\n==================================================", flush=True)
    print(f"STAGE 2: Executing Technical Health Gate Audit on Seed 1", flush=True)
    print(f"==================================================", flush=True)

    smoke_results = {}
    for m in models:
        print(f"Running evaluation smoke for Seed 1 {m} on L40S...")
        smoke_res = run_strengthening_eval_smoke.remote(
            seed=seed_1,
            model_type=m,
            expected_code_sha=code_sha,
        )
        smoke_results[m] = smoke_res
        print(f"-> Smoke status: {smoke_res['status']} | Checkpoint hash: {smoke_res['checkpoint_hash'][:16]}...")

    gate_passed = all(sr["status"] == "PASSED" for sr in smoke_results.values())
    s1_h100_secs = sum(r["timing"]["total_h100_seconds"] for r in s1_results.values())
    h100_hourly_rate = 3.50
    s1_cost = (s1_h100_secs / 3600.0) * h100_hourly_rate
    projected_total_h100_secs = s1_h100_secs * 2.0
    projected_total_cost = s1_cost * 2.0

    print(f"Seed 1 H100 GPU seconds: {s1_h100_secs:.1f} (~${s1_cost:.2f})")
    print(f"Projected total sentinel H100 GPU seconds: {projected_total_h100_secs:.1f} (~${projected_total_cost:.2f})")

    budget_gate_passed = projected_total_cost <= 14.00 and projected_total_h100_secs <= 10800.0

    results["technical_health_gate"] = {
        "eval_smoke": smoke_results,
        "seed_1_h100_seconds": s1_h100_secs,
        "seed_1_estimated_cost_usd": s1_cost,
        "projected_total_cost_usd": projected_total_cost,
        "technical_gate_passed": gate_passed,
        "budget_gate_passed": budget_gate_passed,
        "gate_status": "GO" if (gate_passed and budget_gate_passed) else "STOP",
    }

    if not (gate_passed and budget_gate_passed):
        raise RuntimeError(f"Technical Health Gate STOP: technical={gate_passed}, budget={budget_gate_passed}")

    print(f"TECHNICAL HEALTH GATE: GO! Proceeding to Stage 3.", flush=True)

    # =========================================================================
    # STAGE 3: Seed 4 Training (H100!) in parallel with Seed 1 Evaluation (L40S)
    # =========================================================================
    seed_4 = 20260825
    print(f"\n==================================================", flush=True)
    print(f"STAGE 3: Launching Seed 4 ({seed_4}) Training & Seed 1 Evaluation in parallel", flush=True)
    print(f"==================================================", flush=True)

    # Launch Seed 4 training
    s4_handles = {
        m: run_strengthening_single_model_training.spawn(
            seed=seed_4,
            model_type=m,
            expected_code_sha=code_sha,
        )
        for m in models
    }

    # Launch Seed 1 evaluation workers
    s1_eval_handles = {
        m: run_strengthening_evaluation_worker.spawn(
            seed=seed_1,
            model_type=m,
            expected_code_sha=code_sha,
        )
        for m in models
    }

    # Gather Seed 4 training
    s4_results = {}
    for m, handle in s4_handles.items():
        print(f"Waiting for Seed 4 {m} training...")
        res = handle.get()
        s4_results[m] = res
        print(f"-> Seed 4 {m} training completed! H100 GPU seconds: {res['timing']['total_h100_seconds']:.1f}")

    results["seed_4_training"] = s4_results

    # Gather Seed 1 eval
    s1_eval_results = {}
    for m, handle in s1_eval_handles.items():
        print(f"Waiting for Seed 1 {m} evaluation...")
        res = handle.get()
        s1_eval_results[m] = res
        print(f"-> Seed 1 {m} evaluation completed! Total responses: {res['total_responses_generated']}")

    # =========================================================================
    # STAGE 4: Seed 4 Evaluation (L40S)
    # =========================================================================
    print(f"\n==================================================", flush=True)
    print(f"STAGE 4: Launching Seed 4 Evaluation Workers (L40S)", flush=True)
    print(f"==================================================", flush=True)

    s4_eval_handles = {
        m: run_strengthening_evaluation_worker.spawn(
            seed=seed_4,
            model_type=m,
            expected_code_sha=code_sha,
        )
        for m in models
    }

    s4_eval_results = {}
    for m, handle in s4_eval_handles.items():
        print(f"Waiting for Seed 4 {m} evaluation...")
        res = handle.get()
        s4_eval_results[m] = res
        print(f"-> Seed 4 {m} evaluation completed! Total responses: {res['total_responses_generated']}")

    results["evaluation"] = {
        "seed_1": s1_eval_results,
        "seed_4": s4_eval_results,
    }

    # =========================================================================
    # STAGE 5: Centralized WildGuard Moderation Judging (L40S)
    # =========================================================================
    print(f"\n==================================================", flush=True)
    print(f"STAGE 5: Launching Centralized WildGuard Moderation Judge", flush=True)
    print(f"==================================================", flush=True)

    s1_paths = [r["responses_path"] for r in s1_eval_results.values()]
    s4_paths = [r["responses_path"] for r in s4_eval_results.values()]

    print(f"Judging Seed 1 ({len(s1_paths)} response files)...")
    s1_judge = run_strengthening_centralized_judge.remote(
        seed=seed_1,
        responses_jsonl_paths=s1_paths,
        expected_code_sha=code_sha,
    )
    print(f"-> Seed 1 judging complete: {s1_judge['total_judged']} responses judged in {s1_judge['judge_seconds']:.1f}s")

    print(f"Judging Seed 4 ({len(s4_paths)} response files)...")
    s4_judge = run_strengthening_centralized_judge.remote(
        seed=seed_4,
        responses_jsonl_paths=s4_paths,
        expected_code_sha=code_sha,
    )
    print(f"-> Seed 4 judging complete: {s4_judge['total_judged']} responses judged in {s4_judge['judge_seconds']:.1f}s")

    results["judging"] = {
        "seed_1": s1_judge,
        "seed_4": s4_judge,
    }

    # =========================================================================
    # STAGE 6: Final Cost Accounting & Artifact Packaging
    # =========================================================================
    s4_h100_secs = sum(r["timing"]["total_h100_seconds"] for r in s4_results.values())
    total_h100_secs = s1_h100_secs + s4_h100_secs
    total_h100_cost = (total_h100_secs / 3600.0) * h100_hourly_rate

    eval_l40s_secs = sum(r["eval_seconds"] for r in s1_eval_results.values()) + sum(r["eval_seconds"] for r in s4_eval_results.values())
    judge_l40s_secs = s1_judge["judge_seconds"] + s4_judge["judge_seconds"]
    total_l40s_secs = eval_l40s_secs + judge_l40s_secs
    l40s_hourly_rate = 1.95
    total_l40s_cost = (total_l40s_secs / 3600.0) * l40s_hourly_rate
    total_experiment_cost = total_h100_cost + total_l40s_cost

    results["cost_accounting"] = {
        "h100_gpu_seconds": total_h100_secs,
        "h100_cost_usd": total_h100_cost,
        "l40s_gpu_seconds": total_l40s_secs,
        "l40s_cost_usd": total_l40s_cost,
        "total_sentinel_cost_usd": total_experiment_cost,
        "h100_budget_limit_usd": 14.00,
        "within_budget": total_h100_cost <= 14.00,
    }

    results["end_time_utc"] = datetime.now(timezone.utc).isoformat()
    results["total_wall_clock_seconds"] = time.time() - t0_sentinel

    return results


def main():
    code_sha = get_git_sha()
    print(f"=== Starting Task 2 Sentinel Execution with Code SHA: {code_sha} ===")

    # Run preflight verification first
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.run_strengthening_task2_preflight import run_preflight

    pre_res = run_preflight()
    if pre_res["overall_status"] != "PASSED":
        raise RuntimeError("Preflight audit did not pass. Aborting execution.")

    with app.run():
        summary = run_staged_sentinel_experiment(code_sha=code_sha)

    # Save summary artifact
    out_p = Path("artifacts/strengthening_task2_sentinel_summary.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved Task 2 Sentinel Summary to: {out_p}")
    print(f"Total Sentinel Cost: ${summary['cost_accounting']['total_sentinel_cost_usd']:.2f}")


if __name__ == "__main__":
    main()
