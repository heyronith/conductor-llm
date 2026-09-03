#!/usr/bin/env python3
"""Cost-gated L40S eval+judge for successor Task 1 (adapters already fitted)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
VOLUME = "ccpt-authoritative-runs"
HARD = 5.00


def _run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout)


def billing_succ() -> float:
    proc = _run(["uv", "run", "modal", "billing", "report", "--for", "today", "--show-resources", "--json"])
    if proc.returncode != 0:
        return 0.0
    rows = json.loads(proc.stdout)
    return sum(
        float(r.get("cost") or 0)
        for r in rows
        if "successor-task1" in str(r.get("description", ""))
    )


def main() -> int:
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    cohort = json.loads((ART / "successor_task1_checkpoint_cohort.json").read_text())
    rate = 1.95
    # Conservative: ~0.55 L40S-hr per seed eval+judge
    per_seed = 0.55 * rate
    for pair in cohort["primary_pairs"]:
        if not pair["pair_valid"]:
            continue
        seed = pair["seed"]
        accrued = billing_succ()
        gate = {
            "accrued_usd": accrued,
            "projected_next_usd": per_seed,
            "hard_authorization_usd": HARD,
            "allowed": accrued + per_seed <= HARD,
        }
        (ART / f"successor_task1_cash_gate_before_eval_{seed}.json").write_text(
            json.dumps(gate, indent=2) + "\n"
        )
        if not gate["allowed"]:
            print("SUCCESSOR_TASK1_BLOCKED_BY_COST_GATE", flush=True)
            return 4
        print(f"EVAL seed={seed}", flush=True)
        proc = _run(
            [
                "uv",
                "run",
                "modal",
                "run",
                "modal/successor_task1_falsification.py",
                "--seed",
                str(seed),
                "--mode",
                "eval",
                "--code-sha",
                code_sha,
            ],
            timeout=5 * 3600,
        )
        print(proc.stdout[-3000:], flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-3000:], file=sys.stderr)
            return 5
        _run(
            [
                "uv",
                "run",
                "modal",
                "volume",
                "get",
                VOLUME,
                f"ccpt/successor_task1/seed_{seed}/eval_result.json",
                str(ART / f"successor_task1_eval_result_seed_{seed}.json"),
            ]
        )
        print(f"JUDGE seed={seed}", flush=True)
        proc = _run(
            [
                "uv",
                "run",
                "modal",
                "run",
                "modal/successor_task1_falsification.py",
                "--seed",
                str(seed),
                "--mode",
                "judge",
                "--code-sha",
                code_sha,
            ],
            timeout=5 * 3600,
        )
        print(proc.stdout[-3000:], flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-3000:], file=sys.stderr)
            return 6
        _run(
            [
                "uv",
                "run",
                "modal",
                "volume",
                "get",
                VOLUME,
                f"ccpt/successor_task1/seed_{seed}/judge_summary.json",
                str(ART / f"successor_task1_judge_summary_seed_{seed}.json"),
            ]
        )
        _run(
            [
                "uv",
                "run",
                "modal",
                "volume",
                "get",
                VOLUME,
                f"ccpt/successor_task1/seed_{seed}/judged_responses.jsonl",
                str(ART / f"successor_task1_judged_seed_{seed}.jsonl"),
            ]
        )
    print("EVAL_JUDGE_PHASE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
