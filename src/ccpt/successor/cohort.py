"""Resolve PRE/POST Model-C checkpoint cohort for successor Task 1."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


VOLUME = "ccpt-authoritative-runs"


@dataclass
class CheckpointRef:
    seed: int
    role: str  # PRE | POST_1000 | POST_4000
    volume_path: str  # path inside volume without leading /runs
    persistence_step: int
    source_lineage: str
    notes: str = ""


PRIMARY_COHORT: List[Dict[str, Any]] = [
    {
        "seed": 20260821,
        "source_lineage": "strengthening_task2",
        "pre": "ccpt/strengthening_task2/seed_20260821/model_c/persistence_0000.pt",
        "post_1000": "ccpt/strengthening_task2/seed_20260821/model_c/persistence_1000.pt",
        "post_4000": "ccpt/strengthening_task2/seed_20260821/model_c/persistence_4000.pt",
        "pre_step": 0,
        "notes": "Strengthening Seed-1 corrected lineage; PRE=persistence_0000 post-safety.",
    },
    {
        "seed": 20260823,
        "source_lineage": "task7_4_multiseed_replication_v1",
        "pre": "ccpt/task7_4/multiseed_replication_v1/seed_20260823/model_c/safety_20m_final.pt",
        "post_1000": "ccpt/task7_4/multiseed_replication_v1/seed_20260823/model_c/persistence_1000_final.pt",
        "post_4000": None,
        "pre_step": 0,
        "notes": "Task-7.4 Seed-2; PRE=safety_20m_final (post-safety / persistence step 0 equivalent).",
    },
    {
        "seed": 20260824,
        "source_lineage": "task7_4_multiseed_replication_v1",
        "pre": "ccpt/task7_4/multiseed_replication_v1/seed_20260824/model_c/safety_20m_final.pt",
        "post_1000": "ccpt/task7_4/multiseed_replication_v1/seed_20260824/model_c/persistence_1000_final.pt",
        "post_4000": None,
        "pre_step": 0,
        "notes": "Task-7.4 Seed-3; PRE=safety_20m_final.",
    },
    {
        "seed": 20260825,
        "source_lineage": "strengthening_task2_seed4",
        "pre": "ccpt/strengthening_task2/seed_20260825/model_c/persistence_0000.pt",
        "post_1000": "ccpt/strengthening_task2/seed_20260825/model_c/persistence_1000.pt",
        "post_4000": "ccpt/strengthening_task2/seed_20260825/model_c/persistence_4000.pt",
        "pre_step": 0,
        "notes": "Strengthening Seed-4; PRE=persistence_0000.",
    },
]


def volume_path_exists(rel_path: str) -> bool:
    parent = str(Path(rel_path).parent)
    name = Path(rel_path).name
    proc = subprocess.run(
        ["uv", "run", "modal", "volume", "ls", VOLUME, parent],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False
    return any(line.rstrip().endswith(name) or line.rstrip().endswith("/" + name) for line in proc.stdout.splitlines())


def resolve_cohort(*, check_volume: bool = True) -> Dict[str, Any]:
    primary: List[Dict[str, Any]] = []
    exploratory_4000: List[Dict[str, Any]] = []
    missing: List[str] = []

    for row in PRIMARY_COHORT:
        seed = row["seed"]
        pre_ok = (not check_volume) or volume_path_exists(row["pre"])
        post_ok = (not check_volume) or volume_path_exists(row["post_1000"])
        if not pre_ok:
            missing.append(row["pre"])
        if not post_ok:
            missing.append(row["post_1000"])
        entry = {
            "seed": seed,
            "model_class": "CCPTDualStreamModel",
            "model_type": "model_c",
            "source_lineage": row["source_lineage"],
            "notes": row["notes"],
            "pre": {
                "role": "PRE",
                "persistence_step": row["pre_step"],
                "volume_path": row["pre"],
                "runs_path": f"/runs/{row['pre']}",
                "exists_on_volume": pre_ok,
            },
            "post_1000": {
                "role": "POST_1000",
                "persistence_step": 1000,
                "persistence_tokens": 32_768_000,
                "volume_path": row["post_1000"],
                "runs_path": f"/runs/{row['post_1000']}",
                "exists_on_volume": post_ok,
            },
            "pair_valid": bool(pre_ok and post_ok),
        }
        primary.append(entry)

        if row["post_4000"]:
            p4_ok = (not check_volume) or volume_path_exists(row["post_4000"])
            exploratory_4000.append(
                {
                    "seed": seed,
                    "pre_volume_path": row["pre"],
                    "post_4000_volume_path": row["post_4000"],
                    "persistence_step": 4000,
                    "persistence_tokens": 131_072_000,
                    "exists_on_volume": p4_ok,
                    "label": "EXPLORATORY_4000",
                }
            )

    n_valid = sum(1 for e in primary if e["pair_valid"])
    return {
        "task": "successor_task1_checkpoint_cohort",
        "primary_post_horizon": 1000,
        "required_primary_pairs": 4,
        "primary_valid_pairs": n_valid,
        "primary_seeds": [e["seed"] for e in primary if e["pair_valid"]],
        "primary_pairs": primary,
        "exploratory_4000_pairs": exploratory_4000,
        "missing_paths": missing,
        "gpu_allowed": n_valid >= 4,
        "stop_reason": None if n_valid >= 4 else "FEWER_THAN_FOUR_VALID_PRE_POST_1000_PAIRS",
    }


def write_cohort_artifact(path: Path, cohort: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cohort, indent=2) + "\n")
