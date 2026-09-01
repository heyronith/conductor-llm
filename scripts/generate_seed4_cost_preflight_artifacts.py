#!/usr/bin/env python3
"""Generate Seed 4 billing-grounded cost preflight artifacts and report."""

from __future__ import annotations

import json
from pathlib import Path

from ccpt.analysis.seed4_cost_preflight import repo_root, write_all_artifacts


def render_report(root: Path) -> str:
    rates = json.loads((root / "artifacts" / "seed4_cost_preflight_modal_rates.json").read_text())
    actuals = json.loads((root / "artifacts" / "seed4_cost_preflight_actuals.json").read_text())
    projection = json.loads((root / "artifacts" / "seed4_cost_preflight_projection.json").read_text())
    billing = json.loads((root / "artifacts" / "seed4_cost_preflight_billing_report.json").read_text())
    shortcuts = json.loads((root / "artifacts" / "seed4_cost_preflight_shortcut_audit.json").read_text())

    t2 = actuals["task2_seed1_training"]
    t31 = actuals["task3_1_corrected_evaluation"]
    timing = actuals["timing_field_audit"]
    scenarios = projection["scenarios_usd"]

    lines = [
        "# Seed 4 Billing-Grounded Cost Preflight",
        "",
        f"**Preflight code SHA:** `{projection['preflight_code_sha']}`",
        f"**Modal profile:** `{billing['modal_profile']}`",
        f"**Modal CLI version:** `{billing['modal_cli_version']}`",
        f"**Billing attribution quality:** `{actuals['billing_attribution_quality']}`",
        "",
        "## 1. Workspace billing rates (Modal CLI)",
        "",
        f"- H100: `${rates['workspace_h100_hourly_usd']:.5f}/hr`",
        f"- L40S: `${rates['workspace_l40s_hourly_usd']:.5f}/hr`",
        f"- CPU: `${rates['workspace_cpu_hourly_usd']:.5f}/hr`",
        f"- Memory: `${rates['workspace_mem_gib_hourly_usd']:.5f}/GiB-hr`",
        "",
        "## 2. Seed-1 actual billed cost (authoritative where available)",
        "",
        "### Task-2 Seed-1 training (`strengthening-task2-sentinel`)",
        f"- H100 billed: `${t2['h100_billed_cost_usd']:.4f}` "
        f"({t2['h100_billed_seconds_implied']:.1f}s implied @ workspace rate)",
        f"- Clean-day (2026-09-01) H100 billed: `${t2['clean_training_day_2026_09_01']['h100_billed_cost_usd']:.4f}`",
        f"- Task-2 L40S billed (invalid unframed eval): `${t2['l40s_billed_cost_usd']:.4f}` — **excluded from Seed-4 projection**",
        f"- CPU/Memory billed: `${t2['cpu_billed_cost_usd'] + t2['memory_billed_cost_usd']:.4f}`",
        f"- Per-model H100 split: **unavailable** ({t2['per_model_attribution']})",
        "",
        "### Task-3.1 corrected evaluation (`strengthening-task3-1-eval`)",
        f"- L40S billed: `${t31['l40s_billed_cost_usd']:.4f}` "
        f"({t31['l40s_billed_seconds_implied']:.1f}s implied)",
        f"- Total billed (incl CPU/mem): `${t31['total_billed_cost_usd']:.4f}`",
        f"- Reported runtime cost (modeled): `${t31['reported_cost_usd']['total_l40s_cost_usd']:.4f}`",
        f"- Billing vs reported delta: `${t31['billing_vs_reported_delta_usd']:.4f}`",
        "",
        f"**Total relevant Seed-1 billed cost (excludes Task-2 L40S):** "
        f"`${actuals['total_relevant_seed1_billed_cost_usd']:.4f}`",
        "",
        "## 3. Timing evidence audit",
        "",
        f"- `6000` LM seconds field: `{timing['lm_pretrain_seconds_field']['evidence_class']}`",
        f"- `305` safety seconds field: `{timing['safety_train_seconds_field']['evidence_class']}`",
        f"- `780` persistence seconds field: `{timing['persistence_train_seconds_field']['evidence_class']}`",
        f"- Orchestrator wall clock: `{timing['orchestrator_wall_clock_seconds']['value']:.1f}s` "
        f"(`{timing['orchestrator_wall_clock_seconds']['evidence_class']}`)",
        "",
        timing["lm_pretrain_seconds_field"]["rationale"],
        "",
        "## 4. Seed-4 full-protocol projection",
        "",
        f"- LOW: `${scenarios['low']['total_usd']:.2f}` "
        f"(H100 ${scenarios['low']['h100_usd']:.2f}, L40S ${scenarios['low']['l40s_usd']:.2f}, "
        f"other ${scenarios['low']['other_incremental_usd']:.2f})",
        f"- EXPECTED: `${scenarios['expected']['total_usd']:.2f}` "
        f"(H100 ${scenarios['expected']['h100_usd']:.2f}, L40S ${scenarios['expected']['l40s_usd']:.2f}, "
        f"other ${scenarios['expected']['other_incremental_usd']:.2f})",
        f"- HIGH: `${scenarios['high']['total_usd']:.2f}` "
        f"(H100 ${scenarios['high']['h100_usd']:.2f}, L40S ${scenarios['high']['l40s_usd']:.2f}, "
        f"other ${scenarios['high']['other_incremental_usd']:.2f})",
        "",
        f"**SEED4_FULL_PROTOCOL_WITHIN_10_USD:** "
        f"`{'YES' if projection['seed4_full_protocol_within_10_usd'] else 'NO'}`",
        f"**Minimum realistic expected budget:** `${projection['minimum_realistic_expected_budget_usd']:.2f}`",
        f"**Recommended hard authorization:** `${projection['recommended_hard_authorization_usd']:.2f}`",
        "",
        "## 5. Protocol-preserving efficiencies",
        "",
    ]

    for item in shortcuts["protocol_preserving_efficiencies"]:
        lines.append(f"- `{item['name']}`: {item['notes']}")

    lines.extend(["", "## 6. Invalid shortcuts rejected", ""])
    for item in shortcuts["decisions"]:
        if item["decision"] == "NOT_ALLOWED":
            lines.append(f"- `{item['option']}`: {item['decision']}")

    lines.extend(
        [
            "",
            "## Machine artifacts",
            "",
            "- `artifacts/seed4_cost_preflight_modal_rates.json`",
            "- `artifacts/seed4_cost_preflight_billing_report.json`",
            "- `artifacts/seed4_cost_preflight_actuals.json`",
            "- `artifacts/seed4_cost_preflight_projection.json`",
            "- `artifacts/seed4_cost_preflight_shortcut_audit.json`",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    root = repo_root()
    paths = write_all_artifacts(root)
    report_path = root / "docs" / "research" / "seed4_billing_grounded_cost_preflight.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(root), encoding="utf-8")
    for name, path in paths.items():
        print(f"Wrote {path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
