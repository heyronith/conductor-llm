#!/usr/bin/env python3
"""
Package the review bundle for Task 3.1: Corrected Seed-1 Evaluation Replay.
"""

import json
import zipfile
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
ZIP_PATH = ARTIFACTS_DIR / "strengthening_task3_1_review_bundle.zip"

FILES = [
    ("artifacts/strengthening_task3_1_preflight.json", "Preflight execution confirmation and environment verification"),
    ("artifacts/strengthening_task3_1_generation_manifest.json", "Generation config manifest with SHA256 hashes"),
    ("artifacts/strengthening_task3_1_summary.json", "Full raw evaluation and judging summary output from Modal"),
    ("artifacts/strengthening_task3_1_behavior_summary.json", "Detailed behavioral metrics across all conditions and steps"),
    ("artifacts/strengthening_task3_1_reproducibility_summary.json", "Historical reproducibility comparison table and verdict"),
    ("artifacts/strengthening_task3_1_cost_summary.json", "Hardware accounting and Zero-H100 invariant verification"),
    ("docs/research/strengthening_task3_1_corrected_evaluation_report.md", "Authoritative scientific research report"),
    ("modal/strengthening_task3_1_eval.py", "Dedicated L40S Modal runner for evaluation replay and judging"),
    ("tests/test_strengthening_task3_1_preflight.py", "Preflight test suite verifying canonical prompt framing and hashes"),
    ("tests/test_strengthening_task3_1_regression.py", "Regression test suite verifying reproducibility, causality, and cost"),
    ("scripts/generate_task3_1_preflight_artifacts.py", "Artifact generator for preflight verification"),
    ("scripts/generate_task3_1_post_eval_artifacts.py", "Artifact generator for post-evaluation summaries"),
    ("artifacts/strengthening_task3_1_external_assets_manifest.json", "Manifest of external/gitignored assets, remote paths, and SHA256 hashes"),
    ("docs/research/strengthening_task3_1_editor_handoff.md", "Editor handoff note detailing repository status, lineage, and Task 3.2 instructions"),
]


def main():
    manifest_items = []
    for rel_path, desc in FILES:
        p = Path(rel_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing file for review bundle: {p}")
        manifest_items.append({
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "description": desc,
        })

    manifest = {
        "task": "strengthening_task3_1_review_bundle",
        "evaluation_sha": "751c7b7e52572501cf4fdfe728afc9ff9b0db7a7",
        "seed": 20260821,
        "files": manifest_items,
    }

    manifest_p = ARTIFACTS_DIR / "strengthening_task3_1_review_manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_p, arcname="manifest.json")
        for rel_path, _ in FILES:
            zf.write(Path(rel_path), arcname=rel_path)

    print(f"Created {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
