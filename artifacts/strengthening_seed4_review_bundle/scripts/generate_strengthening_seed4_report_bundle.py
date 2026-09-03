"""Generate Seed-4 research report + review bundle from machine artifacts."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs" / "research"


def _load(name: str) -> dict | None:
    p = ARTIFACTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_report() -> Path:
    exec_m = _load("strengthening_seed4_execution_manifest.json") or {}
    train = _load("strengthening_seed4_training_summary.json") or {}
    ckpt = _load("strengthening_seed4_checkpoint_manifest.json") or {}
    behavior = _load("strengthening_seed4_behavior_summary.json") or {}
    ablation = _load("strengthening_seed4_ablation_summary.json") or {}
    retention = _load("strengthening_seed4_retention_summary.json") or {}
    cost = _load("strengthening_seed4_cost_summary.json") or {}
    ledger = _load("seed4_execution_live_cost_ledger.json") or {}
    preflight = _load("strengthening_seed4_preflight.json") or {}

    lines = [
        "# Strengthening Seed 4 — Authoritative Execution Report",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. Execution provenance",
        "",
        "```json",
        json.dumps(
            {
                "seed": 20260825,
                "execution_sha": exec_m.get("execution_sha"),
                "final_status": exec_m.get("final_status"),
                "completed_models": exec_m.get("completed_models"),
                "preflight_overall": preflight.get("overall_status"),
            },
            indent=2,
        ),
        "```",
        "",
        "## 2. Exact data / protocol",
        "",
        "Seed `20260825`. Models B/C/D. Capability 999,981,056 tokens / 30,517 steps.",
        "Safety 20,010,611 tokens / 2,344 batches. Persistence 0/250/1000/4000 continuous.",
        "Training `H100!`. Corrected eval `L40S` with `format_eval_prompt` and `max_new_tokens=48`.",
        "",
        "## 3. Model checkpoints",
        "",
        "```json",
        json.dumps(ckpt, indent=2),
        "```",
        "",
        "## 4–11. Behavioral / retention / ablation / capability (machine tables)",
        "",
        "### Training summary",
        "",
        "```json",
        json.dumps(train, indent=2),
        "```",
        "",
        "### Behavior summary",
        "",
        "```json",
        json.dumps(behavior, indent=2),
        "```",
        "",
        "### Ablation summary",
        "",
        "```json",
        json.dumps(ablation, indent=2),
        "```",
        "",
        "### Retention summary",
        "",
        "```json",
        json.dumps(retention, indent=2),
        "```",
        "",
        "## 12. Execution anomalies / retries",
        "",
        "```json",
        json.dumps([s for s in (ledger.get("stages") or []) if "fail" in str(s).lower() or "retry" in str(s).lower() or "abort" in str(s).lower()], indent=2),
        "```",
        "",
        "## 13. Exact Modal billing / cost",
        "",
        "```json",
        json.dumps(cost, indent=2),
        "```",
        "",
        "## 14. Scientific limitations",
        "",
        "- Single independent seed (`20260825`); do not generalize superiority claims from Seed 4 alone.",
        "- Primary endpoint remains 1000 steps; 4000 is secondary long-horizon.",
        "- Hard cash ceiling was `$27.00`; partial completion may omit later models/eval.",
        "",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "strengthening_seed4_authoritative_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_review_bundle() -> Path:
    report = render_report()
    include = [
        "modal/strengthening_task2_sentinel.py",
        "modal/strengthening_task3_1_eval.py",
        "src/ccpt/analysis/seed4_execution_ledger.py",
        "src/ccpt/analysis/seed4_execution_summaries.py",
        "scripts/run_strengthening_seed4_preflight.py",
        "scripts/orchestrate_strengthening_seed4.py",
        "scripts/generate_strengthening_seed4_report_bundle.py",
        "tests/test_seed4_execution.py",
        "artifacts/strengthening_seed4_preflight.json",
        "artifacts/seed4_execution_live_cost_ledger.json",
        "artifacts/seed4_cost_reduction_projection.json",
        "artifacts/strengthening_task3_2_reconciliation.json",
        "artifacts/strengthening_seed4_execution_manifest.json",
        "artifacts/strengthening_seed4_training_summary.json",
        "artifacts/strengthening_seed4_checkpoint_manifest.json",
        "artifacts/strengthening_seed4_behavior_summary.json",
        "artifacts/strengthening_seed4_ablation_summary.json",
        "artifacts/strengthening_seed4_retention_summary.json",
        "artifacts/strengthening_seed4_cost_summary.json",
        "artifacts/strengthening_seed4_task3_1_summary.json",
        "docs/research/strengthening_seed4_authoritative_report.md",
        "docs/research/seed4_cost_reduction_forensic_audit.md",
    ]
    manifest = {"task": "strengthening_seed4_review_bundle_manifest", "files": []}
    zip_path = ARTIFACTS / "strengthening_seed4_review_bundle.zip"
    ARTIFACTS.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            p = ROOT / rel
            if not p.exists():
                continue
            zf.write(p, rel)
            manifest["files"].append({"path": rel, "sha256": _sha256(p), "bytes": p.stat().st_size})
        man_path = ARTIFACTS / "strengthening_seed4_review_bundle_manifest.json"
        man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        zf.write(man_path, "artifacts/strengthening_seed4_review_bundle_manifest.json")
    return zip_path


def main() -> None:
    report = render_report()
    bundle = build_review_bundle()
    print(f"Wrote {report}")
    print(f"Wrote {bundle}")


if __name__ == "__main__":
    main()
