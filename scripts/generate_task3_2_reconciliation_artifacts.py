#!/usr/bin/env python3
"""Generate Task 3.2 reconciliation artifact and research report (zero-GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from ccpt.analysis.strengthening_task3_2_reconciliation import (
    HORIZONS,
    LONG_HORIZON,
    PRIMARY_HORIZON,
    build_reconciliation,
    repo_root,
    write_reconciliation_artifact,
)


def _pct(rate: float) -> str:
    return f"{rate * 100:.2f}%"


def _pp(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f} pp"


def render_report(reconciliation: dict) -> str:
    hist = reconciliation["historical_seed1"]
    rerun = reconciliation["corrected_rerun_seed1"]
    cd = reconciliation["pairwise_effects"]["c_minus_d"]
    cb = reconciliation["pairwise_effects"]["c_minus_b"]
    ablation = reconciliation["model_c_ablation"]
    recon = reconciliation["reconciliation"]
    cls = reconciliation["classification"]
    join_audit = recon["task3_1_historical_join_audit"]

    lines = [
        "# CCPT Strengthening — Task 3.2 Historical Reconciliation Report",
        "",
        "**Task**: Zero-GPU evidence reconciliation for Seed-1 persistence synthesis",
        f"**Seed**: {reconciliation['seed']}",
        f"**Task-3.2 code SHA**: `{reconciliation['provenance']['task3_2_code_sha']}`",
        f"**Task-3.1 evaluation SHA**: `{reconciliation['provenance']['task3_1_evaluation_sha']}`",
        f"**Task-3.1 evidence SHA**: `{reconciliation['provenance']['task3_1_evidence_sha']}`",
        "",
        "---",
        "",
        "## 1. What was wrong in the Task-3.1 historical join",
        "",
    ]

    if join_audit.get("task3_1_historical_join_wrong"):
        lines.extend(
            [
                "Task 3.1's `strengthening_task3_1_reproducibility_summary.json` embedded a convenience "
                "historical persistence table with incorrect POST1000 refusal anchors:",
                "",
                f"- Joined Model C 1000-step retention: {_pp(join_audit['task3_1_joined_c_retention_pp'])}",
                f"- Joined Model D 1000-step retention: {_pp(join_audit['task3_1_joined_d_retention_pp'])}",
                f"- Authoritative Model C retention: {_pp(join_audit['authoritative_c_retention_pp'])}",
                f"- Authoritative Model D retention: {_pp(join_audit['authoritative_d_retention_pp'])}",
                "",
                join_audit["explanation"],
            ]
        )
    else:
        lines.append(join_audit.get("explanation", "No join discrepancy detected."))

    lines.extend(
        [
            "",
            "Task 3.1 raw evaluation evidence (`strengthening_task3_1_behavior_summary.json`, "
            "`strengthening_task3_1_summary.json`) remains unchanged. Task 3.2 supersedes only "
            "the historical comparison synthesis layer.",
            "",
            "## 2. Authoritative historical source",
            "",
            f"- Primary: `{hist['source_artifact_path']}` (SHA256: `{hist['source_artifact_sha256']}`)",
            f"- Cross-check: `{hist['crosscheck_artifact_path']}`",
            f"- Metric: {hist['metric_definition']}",
            "",
            "## 3. Historical Seed-1 C/D 1000-step result",
            "",
            "### Model C",
            f"- PRE harmful refusal: {hist['models']['model_c']['pre']['refusal_yes']}/"
            f"{hist['models']['model_c']['pre']['determinate_n']} = "
            f"{_pct(hist['models']['model_c']['pre']['rate'])}",
            f"- POST1000 harmful refusal: {hist['models']['model_c']['post1000']['refusal_yes']}/"
            f"{hist['models']['model_c']['post1000']['determinate_n']} = "
            f"{_pct(hist['models']['model_c']['post1000']['rate'])}",
            f"- Retention (POST1000 − PRE): {_pp(hist['models']['model_c']['retention_1000_pp'])}",
            "",
            "### Model D",
            f"- PRE harmful refusal: {hist['models']['model_d']['pre']['refusal_yes']}/"
            f"{hist['models']['model_d']['pre']['determinate_n']} = "
            f"{_pct(hist['models']['model_d']['pre']['rate'])}",
            f"- POST1000 harmful refusal: {hist['models']['model_d']['post1000']['refusal_yes']}/"
            f"{hist['models']['model_d']['post1000']['determinate_n']} = "
            f"{_pct(hist['models']['model_d']['post1000']['rate'])}",
            f"- Retention (POST1000 − PRE): {_pp(hist['models']['model_d']['retention_1000_pp'])}",
            "",
            f"**Historical C−D retention effect at {PRIMARY_HORIZON}**: "
            f"{_pp(hist['c_minus_d_effect_1000_pp'])}",
            "",
            "## 4. Corrected clean-rerun trajectory (Task 3.1, canonical framing)",
            "",
        ]
    )

    for model in ("model_b", "model_c", "model_d"):
        label = model.replace("model_", "Model ").upper()
        lines.append(f"### {label}")
        for step in HORIZONS:
            h = rerun["models"][model]["horizons"][str(step)]
            lines.append(
                f"- Step {step}: {h['refusal_yes']}/{h['determinate_n']} = {_pct(h['rate'])}"
            )
        for step in (250, 1000, 4000):
            ret = rerun["models"][model]["retention"][str(step)]
            lines.append(f"- Retention {step}: {_pp(ret['retention_pp'])}")
        lines.append("")

    lines.extend(
        [
            "## 5. C-vs-D effects at all horizons",
            "",
            f"- 250 steps: {_pp(cd['250_pp'])}",
            f"- 1000 steps (primary endpoint): {_pp(cd['1000_pp'])}",
            f"- 4000 steps (long-horizon stress test): {_pp(cd['4000_pp'])}",
            "",
            "## 6. C-vs-B effects at all horizons",
            "",
            f"- 250 steps: {_pp(cb['250_pp'])}",
            f"- 1000 steps: {_pp(cb['1000_pp'])}",
            f"- 4000 steps: {_pp(cb['4000_pp'])}",
            "",
            "## 7. Model-C active-vs-off gaps (0 / 1000 / 4000)",
            "",
        ]
    )

    for step in (0, 1000, 4000):
        block = ablation[str(step)]
        lines.extend(
            [
                f"### Step {step}",
                f"- Active: {_pct(block['active']['rate'])} "
                f"({block['active']['refusal_yes']}/{block['active']['determinate_n']})",
                f"- Off: {_pct(block['off']['rate'])} "
                f"({block['off']['refusal_yes']}/{block['off']['determinate_n']}, "
                f"NA={block['off']['refusal_na']})",
                f"- Gap (active − off): {_pp(block['gap_pp'])} ({block['gap_sign']})",
            ]
        )
        if block["off"]["refusal_na"] > 0:
            lines.append(
                f"- NA sensitivity: NA-as-refusal {_pct(block['off']['na_as_refusal_rate'])}, "
                f"NA-as-nonrefusal {_pct(block['off']['na_as_nonrefusal_rate'])}"
            )
        lines.append("")

    lines.extend(
        [
            "## 8. Explicit 4000-step reversals",
            "",
            "### Controller-gap reversal (Model C)",
            f"At step 4000 the active-vs-off gap is {_pp(ablation['4000']['gap_pp'])} "
            "(active refusal is lower than ablated). Controller causality is not uniformly "
            "positive through 4000 persistence steps.",
            "",
            "### C-vs-D persistence reversal",
            f"At the preregistered {PRIMARY_HORIZON}-step endpoint the corrected rerun shows "
            f"C retains better than D by {_pp(cd['1000_pp'])}. At the secondary {LONG_HORIZON}-step "
            f"stress-test horizon the advantage reverses: {_pp(cd['4000_pp'])} (D retains better). "
            "This crossover is observed for Seed 1 only and must not be generalized across seeds.",
            "",
            "## 9. Separation of scientific questions",
            "",
            "### A. Evaluation-bug diagnosis",
            "Canonical `format_eval_prompt()` framing explains the Task-2 Seed-1 inversion and "
            "collapsed refusal rates. Task 3 forensic classification "
            "`D_EVALUATION_PROTOCOL_DIVERGENCE` is confirmed; Task 3.1 corrected replay restored "
            "positive Model-C controller direction at step 0 and 1000.",
            "",
            "### B. Safety-acquisition replication",
            f"PRE harmful refusal differs from historical anchors: Model C "
            f"{_pct(rerun['models']['model_c']['horizons']['0']['rate'])} vs historical "
            f"{_pct(hist['models']['model_c']['pre']['rate'])}; Model D "
            f"{_pct(rerun['models']['model_d']['horizons']['0']['rate'])} vs historical "
            f"{_pct(hist['models']['model_d']['pre']['rate'])}. "
            f"Classification: `{cls['safety_acquisition']}`.",
            "",
            "### C. Persistence replication (primary 1000-step endpoint)",
            f"Historical C−D retention effect: {_pp(hist['c_minus_d_effect_1000_pp'])}. "
            f"Corrected rerun effect: {_pp(cd['1000_pp'])}. Direction agreement at 1000: "
            f"{recon['direction_agreement_at_1000']}. Magnitude agreement within "
            f"{5.0:.0f} pp: {recon['magnitude_agreement_at_1000']}. "
            f"Classification: `{cls['primary_1000_persistence_reproducibility']}`.",
            "",
            "## 10. Limits on interpretation",
            "",
            "- Primary endpoint remains 1000 persistence steps (32,768,000 continuation tokens).",
            "- 4000 steps (131,072,000 tokens) is a secondary long-horizon stress test.",
            "- All horizons 0 / 250 / 1000 / 4000 are reported; none are suppressed.",
            "- Seed-1-only 4000-step crossover must not be generalized.",
            "- Provisional reading: the clean Seed-1 rerun shows a modest C-vs-D persistence "
            "advantage at the preregistered 1000-step endpoint but a reversal at the 4000-step "
            "stress-test horizon.",
            "",
            "## Machine-readable classifications",
            "",
            f"- Evaluation defect: `{cls['evaluation_defect']}`",
            f"- Safety acquisition: `{cls['safety_acquisition']}`",
            f"- Controller direction step 0: `{cls['controller_direction_step0']}`",
            f"- Controller direction step 1000: `{cls['controller_direction_step1000']}`",
            f"- Controller direction step 4000: `{cls['controller_direction_step4000']}`",
            f"- Primary 1000 persistence: `{cls['primary_1000_persistence_reproducibility']}`",
            f"- Long-horizon 4000: {cls['long_horizon_4000_result']}",
            "",
            f"**Authoritative machine artifact**: `artifacts/strengthening_task3_2_reconciliation.json`",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    root = repo_root()
    artifact_path = write_reconciliation_artifact(root)
    with open(artifact_path, "r", encoding="utf-8") as handle:
        reconciliation = json.load(handle)

    report_path = root / "docs" / "research" / "strengthening_task3_2_historical_reconciliation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(reconciliation), encoding="utf-8")

    audit_path = root / "docs" / "research" / "strengthening_task3_2_onboarding_audit.md"
    audit_path.write_text(
        "\n".join(
            [
                "# Task 3.2 Onboarding Audit Note",
                "",
                "Read-only onboarding verified handoff HEAD `80de4629f1caf15dc3773429e0d58821664b818d` "
                "on branch `strengthening-task3-seed1-forensic` with clean remote parity.",
                "",
                "## Task 3 forensic conclusion",
                "- Primary root cause: `D_EVALUATION_PROTOCOL_DIVERGENCE` (missing `format_eval_prompt`, "
                "generation length 64 vs canonical 48).",
                "- Checkpoints passed structural/provenance forensic checks.",
                "",
                "## Task 3.1 correction verified",
                "- Canonical framing and `max_new_tokens=48` recorded in Task 3.1 manifests.",
                "- 10,752 framed judged records; H100 seconds = 0 in cost summary.",
                "- No checkpoint retraining.",
                "",
                "## Task 3.1 synthesis defect identified",
                "- Historical persistence join in Task 3.1 reproducibility summary used wrong POST1000 anchors.",
                "- Authoritative historical persistence resolved from `artifacts/task7_3_1a_forensic_summary.json` "
                "with Task 7.4 cross-check.",
                "",
                "This note is superseded for numerical claims by "
                "`artifacts/strengthening_task3_2_reconciliation.json`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {artifact_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
