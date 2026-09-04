#!/usr/bin/env python3
"""Resume-aware cost-gated L40S eval+judge for successor Task 1.

Uses ``modal run -d`` so laptop disconnect does not cancel GPU work.
Skips steps whose Volume artifacts already exist.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
VOLUME = "ccpt-authoritative-runs"
# Operator-authorized hard ceiling (2026-09-03). Must not be raised programmatically.
HARD = 9.00
CODE_SHA = "c834b5ed2c81dba6c5d53c40ed25593b00c146a3"

# Measured from seed 20260821 EVAL_COMPLETE (~1647 L40S-s @ $1.95/hr ≈ $0.89)
EVAL_ENVELOPE_USD = 1.05
JUDGE_ENVELOPE_USD = 0.75
# Full experiment window (UTC) so day-rollover does not understate accrued spend.
BILLING_START = "2026-09-03"
BILLING_END = "2026-09-05"


def _run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout)


def billing_succ() -> float:
    """Sum successor-task1 metered cost over the full experiment billing window."""
    proc = _run(
        [
            "uv",
            "run",
            "modal",
            "billing",
            "report",
            "--start",
            BILLING_START,
            "--end",
            BILLING_END,
            "--show-resources",
            "--json",
        ]
    )
    if proc.returncode != 0:
        return 0.0
    rows = json.loads(proc.stdout)
    return sum(
        float(r.get("cost") or 0)
        for r in rows
        if "successor-task1" in str(r.get("description", ""))
    )


def volume_has(rel: str) -> bool:
    parent = str(Path(rel).parent)
    name = Path(rel).name
    proc = _run(["uv", "run", "modal", "volume", "ls", VOLUME, parent])
    if proc.returncode != 0:
        return False
    return any(line.rstrip().endswith(name) for line in proc.stdout.splitlines())


def pull(rel: str, local: Path) -> bool:
    proc = _run(["uv", "run", "modal", "volume", "get", VOLUME, rel, str(local)])
    return proc.returncode == 0 and local.exists()


def launch_detached(seed: int, mode: str) -> subprocess.Popen[str]:
    log = ART / f"successor_task1_modal_{mode}_{seed}.log"
    cmd = [
        "uv",
        "run",
        "modal",
        "run",
        "-d",
        "modal/successor_task1_falsification.py",
        "--seed",
        str(seed),
        "--mode",
        mode,
        "--code-sha",
        CODE_SHA,
    ]
    print(f"$ {' '.join(cmd)}  (log={log})", flush=True)
    f = log.open("a")
    f.write(f"\n--- launch {datetime.now(timezone.utc).isoformat()} ---\n")
    f.flush()
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_volume(rel: str, timeout_s: int = 4 * 3600, poll_s: int = 45) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if volume_has(rel):
            # reload volume listing can lag; try pull
            return True
        print(
            f"  waiting for {rel} ({int(time.time() - t0)}s); billed=${billing_succ():.4f}",
            flush=True,
        )
        time.sleep(poll_s)
    return False


def gate(accrued: float, need: float, label: str) -> bool:
    g = {
        "label": label,
        "accrued_usd": accrued,
        "projected_next_usd": need,
        "hard_authorization_usd": HARD,
        "allowed": accrued + need <= HARD + 1e-9,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = ART / f"successor_task1_cash_gate_{label}.json"
    path.write_text(json.dumps(g, indent=2) + "\n")
    print(json.dumps(g), flush=True)
    if not g["allowed"]:
        print("SUCCESSOR_TASK1_BLOCKED_BY_COST_GATE", flush=True)
        return False
    return True


def ensure_eval(seed: int) -> int:
    eval_rel = f"ccpt/successor_task1/seed_{seed}/eval_result.json"
    resp_rel = f"ccpt/successor_task1/seed_{seed}/responses.jsonl"
    local = ART / f"successor_task1_eval_result_seed_{seed}.json"
    if volume_has(eval_rel) and volume_has(resp_rel):
        print(f"SKIP EVAL seed={seed} (already on Volume)", flush=True)
        pull(eval_rel, local)
        return 0
    accrued = billing_succ()
    if not gate(accrued, EVAL_ENVELOPE_USD, f"before_eval_{seed}"):
        return 4
    print(f"EVAL seed={seed} (detached)", flush=True)
    proc = launch_detached(seed, "eval")
    # Give Modal time to start; then poll Volume (detach survives local death)
    time.sleep(90)
    if not wait_for_volume(eval_rel):
        print(f"FAIL: eval timeout seed={seed}", flush=True)
        proc.poll()
        return 5
    pull(eval_rel, local)
    print(f"EVAL DONE seed={seed}", flush=True)
    return 0


def ensure_judge(seed: int) -> int:
    judge_rel = f"ccpt/successor_task1/seed_{seed}/judge_summary.json"
    judged_rel = f"ccpt/successor_task1/seed_{seed}/judged_responses.jsonl"
    local_j = ART / f"successor_task1_judge_summary_seed_{seed}.json"
    local_r = ART / f"successor_task1_judged_seed_{seed}.jsonl"
    if volume_has(judge_rel) and volume_has(judged_rel):
        print(f"SKIP JUDGE seed={seed} (already on Volume)", flush=True)
        pull(judge_rel, local_j)
        pull(judged_rel, local_r)
        return 0
    # Require responses present
    if not volume_has(f"ccpt/successor_task1/seed_{seed}/responses.jsonl"):
        print(f"FAIL: cannot judge seed={seed}; responses missing", flush=True)
        return 7
    accrued = billing_succ()
    if not gate(accrued, JUDGE_ENVELOPE_USD, f"before_judge_{seed}"):
        return 4
    print(f"JUDGE seed={seed} (detached)", flush=True)
    launch_detached(seed, "judge")
    time.sleep(90)
    if not wait_for_volume(judge_rel):
        print(f"FAIL: judge timeout seed={seed}", flush=True)
        return 6
    pull(judge_rel, local_j)
    pull(judged_rel, local_r)
    print(f"JUDGE DONE seed={seed}", flush=True)
    return 0


def main() -> int:
    print(f"RESUME CODE_SHA={CODE_SHA}", flush=True)
    print(f"accrued_at_start=${billing_succ():.4f} hard=${HARD:.2f}", flush=True)
    cohort = json.loads((ART / "successor_task1_checkpoint_cohort.json").read_text())
    seeds = [p["seed"] for p in cohort["primary_pairs"] if p["pair_valid"]]
    for seed in seeds:
        rc = ensure_eval(seed)
        if rc != 0:
            return rc
        rc = ensure_judge(seed)
        if rc != 0:
            return rc
    (ART / "successor_task1_eval_judge_resume_status.json").write_text(
        json.dumps(
            {
                "status": "EVAL_JUDGE_PHASE_COMPLETE",
                "code_sha": CODE_SHA,
                "accrued_usd": billing_succ(),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "seeds": seeds,
            },
            indent=2,
        )
        + "\n"
    )
    print("EVAL_JUDGE_PHASE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
