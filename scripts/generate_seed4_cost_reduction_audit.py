#!/usr/bin/env python3
"""Generate Seed 4 cost-reduction forensic artifacts and research report."""

from __future__ import annotations

import json
from pathlib import Path

from ccpt.analysis.seed4_cost_reduction_audit import repo_root, write_all_artifacts


def render_report(root: Path) -> str:
    ledger = json.loads((root / "artifacts/seed4_cost_reduction_task2_object_ledger.json").read_text())
    reuse = json.loads((root / "artifacts/seed4_cost_reduction_checkpoint_reuse_audit.json").read_text())
    waste = json.loads((root / "artifacts/seed4_cost_reduction_waste_analysis.json").read_text())
    plan = json.loads((root / "artifacts/seed4_cost_reduction_execution_plan.json").read_text())
    runtime = json.loads((root / "artifacts/seed4_cost_reduction_runtime_optimization_review.json").read_text())
    projection = json.loads((root / "artifacts/seed4_cost_reduction_projection.json").read_text())

    by_class = ledger["totals"]["h100_by_classification_usd"]
    scenarios = projection["scenarios_usd"]

    lines = [
        "# Seed 4 Cost-Reduction Forensic Audit",
        "",
        f"**Audit code SHA:** `{projection['audit_code_sha']}`",
        f"**Billing-preflight base:** `{projection['base_billing_preflight_sha']}`",
        f"**Remaining Modal credits:** `${ledger['remaining_modal_credits_usd']:.2f}`",
        f"**Ledger reconciliation:** `{ledger['reconciliation']['ledger_reconciliation']}`",
        "",
        "## 1. Credits and out-of-pocket framing",
        "",
        "All Seed-4 metered spend is out-of-pocket cash:",
        "",
        "`OUT_OF_POCKET_COST_USD = METERED_COST_USD`",
        "",
        "No Starter credits, academic credits, promotions, or hypothetical refunds are assumed.",
        "",
        "## 2. Seed-1 H100 object ledger",
        "",
        f"- Full-window Task-2 H100 billed: `${ledger['totals']['full_window_h100_cost_usd']:.8f}`",
        f"- Sep-1 UTC Task-2 H100 billed: `${ledger['totals']['sep1_utc_h100_cost_usd']:.8f}`",
        "",
        "### Classification totals (H100 USD)",
        "",
    ]
    for key in sorted(by_class):
        lines.append(f"- `{key}`: `${by_class[key]:.4f}`")

    lines.extend(
        [
            "",
            "### Top objects",
            "",
        ]
    )
    for obj in ledger["objects"]:
        clf = obj["classification"]
        lines.append(
            f"- `{obj['object_id']}`: H100 `${obj['h100_cost_usd']:.4f}` "
            f"({obj['h100_seconds_implied']:.1f}s) → `{clf['primary_classification']}` "
            f"[{clf['confidence']}]"
        )

    lines.extend(
        [
            "",
            f"**$5+ avoidable hypothesis:** `{ledger['hypothesis_5plus_avoidable']['verdict']}` — "
            f"{ledger['hypothesis_5plus_avoidable']['explanation']}",
            "",
            "## 3. Seed-4 checkpoint reuse",
            "",
            f"- Seed-4 path exists: `{reuse['volume_inspection']['exists']}`",
            f"- Any valid reusable Seed-4 checkpoint: `{reuse['any_valid_reusable_seed4_checkpoint']}`",
            f"- Estimated savings from existing valid state: `${reuse['estimated_savings_from_existing_valid_state_usd']:.2f}`",
            "",
            "## 4. Proven savings (HIGH confidence only)",
            "",
            f"- Proven avoidable H100: `${projection['proven_avoidable_h100_cost_usd']:.4f}`",
            f"- Proven avoidable non-H100 (Task-2 invalid L40S): `${projection['proven_avoidable_non_h100_cost_usd']:.4f}`",
            f"- Total proven avoidable: `${projection['proven_avoidable_total_cost_usd']:.4f}`",
            f"- MEDIUM H100 excluded from expected: `${projection['medium_confidence_excluded_from_expected_h100_usd']:.4f}`",
            "",
            "## 5. Irreducible cost floor",
            "",
            f"- Scientifically irreducible H100 floor: `${projection['scientifically_irreducible_h100_cost_floor_usd']:.4f}`",
            f"- Scientifically irreducible full-protocol floor: `${projection['scientifically_irreducible_full_protocol_cost_floor_usd']:.4f}`",
            f"- Evidence quality: `{projection['irreducible_evidence_quality']}`",
            "",
            projection["irreducible_uncertainty"],
            "",
            "## 6. Revised Seed-4 projections (out-of-pocket)",
            "",
            f"- **A historical-style repeat:** `${scenarios['A_historical_repeat']['total_usd']:.2f}` "
            f"(H100 ${scenarios['A_historical_repeat']['h100_usd']:.2f}, "
            f"L40S ${scenarios['A_historical_repeat']['l40s_usd']:.2f}, "
            f"other ${scenarios['A_historical_repeat']['other_incremental_usd']:.2f})",
            f"- **B clean expected:** `${scenarios['B_clean_protocol_preserving']['total_usd']:.2f}` "
            f"(H100 ${scenarios['B_clean_protocol_preserving']['h100_usd']:.2f}, "
            f"L40S ${scenarios['B_clean_protocol_preserving']['l40s_usd']:.2f}, "
            f"other ${scenarios['B_clean_protocol_preserving']['other_incremental_usd']:.2f})",
            f"- **C conservative:** `${scenarios['C_conservative']['total_usd']:.2f}`",
            f"- **Recommended hard authorization:** `${projection['recommended_hard_authorization_usd']:.2f}`",
            "",
            "## 7. Protocol-preserving optimizations",
            "",
        ]
    )
    for item in plan["protocol_preserving_optimizations"]:
        lines.append(
            f"- `{item['name']}`: {item['expected_savings']} "
            f"(scientific semantics changed: `{item['scientific_semantics_changed']}`)"
        )

    lines.extend(
        [
            "",
            "## 8. Runtime optimization review",
            "",
            f"- CUDA Graphs: `{runtime['cuda_graphs']['classification']}`",
            f"- torch.compile: `{runtime['torch_compile']['classification']}`",
            f"- Prefetch/staging: `{runtime['prefetch_staging']['classification']}`",
            f"- Paid parity benchmark worthwhile: `{runtime['paid_parity_benchmark_worthwhile']}`",
            f"- Max proposed benchmark cost: `${runtime['maximum_proposed_benchmark_cost_usd']:.2f}` (not executed)",
            "",
            "## Machine artifacts",
            "",
            "- `artifacts/seed4_cost_reduction_task2_object_ledger.json`",
            "- `artifacts/seed4_cost_reduction_checkpoint_reuse_audit.json`",
            "- `artifacts/seed4_cost_reduction_waste_analysis.json`",
            "- `artifacts/seed4_cost_reduction_execution_plan.json`",
            "- `artifacts/seed4_cost_reduction_projection.json`",
            "- `artifacts/seed4_cost_reduction_runtime_optimization_review.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    root = repo_root()
    paths = write_all_artifacts(root)
    report_path = root / "docs" / "research" / "seed4_cost_reduction_forensic_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(root), encoding="utf-8")
    for path in paths.values():
        print(f"Wrote {path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
