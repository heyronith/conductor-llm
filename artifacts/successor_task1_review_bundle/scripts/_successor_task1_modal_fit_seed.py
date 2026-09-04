#!/usr/bin/env python3
"""Thin Modal CLI wrapper for one successor Task-1 seed fit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "ccpt-authoritative-runs"


def main() -> int:
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text())
    seed = int(payload["seed"])
    # Upload payload to volume for the remote function
    remote = f"ccpt/successor_task1/payload_seed_{seed}.json"
    put = subprocess.run(
        ["uv", "run", "modal", "volume", "put", VOLUME, str(payload_path), remote],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    if put.returncode != 0:
        print(put.stderr, file=sys.stderr)
        return put.returncode

    proc = subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "modal/successor_task1_falsification.py",
            "--seed",
            str(seed),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    # Pull fit_result.json from volume
    local = ROOT / "artifacts" / f"successor_task1_fit_result_seed_{seed}.json"
    get = subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "volume",
            "get",
            VOLUME,
            f"ccpt/successor_task1/seed_{seed}/fit_result.json",
            str(local),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    if get.returncode != 0:
        # Fallback: parse last JSON object from stdout
        print(get.stderr, file=sys.stderr)
        local.write_text(proc.stdout[proc.stdout.rfind("{") :] if "{" in proc.stdout else "{}")
    print(json.dumps({"wrote": str(local)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
