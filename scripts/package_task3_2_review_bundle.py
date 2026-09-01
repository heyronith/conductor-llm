#!/usr/bin/env python3
"""Package the review bundle for Task 3.2 evidence reconciliation."""

import json
import zipfile
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
ZIP_PATH = ARTIFACTS_DIR / "strengthening_task3_2_review_bundle.zip"

FILES = [
    (
        "src/ccpt/analysis/strengthening_task3_2_reconciliation.py",
        "Authoritative historical resolver and reconciliation builder",
    ),
    (
        "scripts/generate_task3_2_reconciliation_artifacts.py",
        "Generator for reconciliation artifact and report",
    ),
    (
        "tests/test_strengthening_task3_2_reconciliation.py",
        "Task 3.2 reconciliation regression tests",
    ),
    (
        "artifacts/strengthening_task3_2_reconciliation.json",
        "Superseding machine reconciliation artifact",
    ),
    (
        "docs/research/strengthening_task3_2_historical_reconciliation.md",
        "Task 3.2 research report derived from reconciliation artifact",
    ),
    (
        "docs/research/strengthening_task3_2_onboarding_audit.md",
        "New-editor onboarding audit note",
    ),
    (
        "artifacts/strengthening_task3_1_behavior_summary.json",
        "Task 3.1 corrected behavioral machine summary (unchanged evidence)",
    ),
    (
        "artifacts/strengthening_task3_1_summary.json",
        "Task 3.1 corrected evaluation summary (unchanged evidence)",
    ),
    (
        "artifacts/strengthening_task3_1_reproducibility_summary.json",
        "Task 3.1 synthesis layer preserved for audit comparison",
    ),
    (
        "artifacts/task7_3_1a_forensic_summary.json",
        "Authoritative historical Seed-1 behavioral source",
    ),
    (
        "artifacts/task7_4_multiseed_replication_summary.json",
        "Task 7.4 Seed-1 forensic cross-check reference",
    ),
    (
        "docs/research/strengthening_task3_1_editor_handoff.md",
        "Task 3.1 editor handoff note",
    ),
    (
        "artifacts/strengthening_task3_1_external_assets_manifest.json",
        "External assets provenance manifest",
    ),
]


def main() -> None:
    manifest_items = []
    for rel_path, desc in FILES:
        path = Path(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing file for review bundle: {path}")
        manifest_items.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "description": desc,
            }
        )

    manifest = {
        "task": "strengthening_task3_2_review_bundle",
        "supersedes": "artifacts/strengthening_task3_1_reproducibility_summary.json",
        "relationship": (
            "Task 3.2 reconciliation supersedes Task 3.1 historical persistence synthesis only; "
            "Task 3.1 raw evaluation evidence remains immutable."
        ),
        "files": manifest_items,
    }

    manifest_path = ARTIFACTS_DIR / "strengthening_task3_2_review_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, arcname="manifest.json")
        for rel_path, _ in FILES:
            archive.write(Path(rel_path), arcname=rel_path)

    print(f"Created {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
