# CCPT Strengthening — Task 3.1 Editor Handoff Note

**Repository Name**: `conductor-llm` (`CCPT`)  
**Current Branch**: `strengthening-task3-seed1-forensic`  
**Current Stage**: Completed Task 3.1 (Corrected Seed-1 Evaluation Replay); Ready for Task 3.2 (Zero-GPU Evidence Reconciliation)  
**Evaluation Execution SHA**: `751c7b7e52572501cf4fdfe728afc9ff9b0db7a7`  
**Evidence Commit SHA**: `0b7d4183b392536f6b629738d7445b5d73ab3825`  
**Date**: September 1, 2026  

---

## 1. Executive Summary & Lineage

This repository contains the ongoing empirical research into the **Constitutional Control-Plane Transformer (CCPT)** intrinsic-alignment architecture.

### Task 3 Forensic Conclusion
The Task 3 audit established that the apparent failure of Seed-1 in Task 2 (where Model C active refusal was only 44.3% with an apparent "controller inversion" of -19.8 pp) was caused entirely by a missing chat template prompt framing (`format_eval_prompt()`) during evaluation generation in Task 2. The underlying training checkpoints were identified as provisionally sound and authentic.

### Task 3.1 Evaluation Correction & Findings
Task 3.1 executed an authoritative behavioral re-evaluation of the existing Task-2 Seed-1 checkpoints across all 11 model/step conditions under canonical chat prompt framing (`<s>User: {prompt}\nAssistant:`):
- **Model C Step 0 Active Refusal**: **75.39%** (vs 44.31% unframed in Task 2).
- **Model C Step 0 Controller Gap**: **+18.13 pp** (75.39% active vs 57.26% ablated) $\rightarrow$ **Positive controller causality is restored; the inversion is completely debunked**.
- **Model C Step 1000 Controller Gap**: **+20.35 pp** (66.41% active vs 46.06% ablated).
- **Model D Step 0 Active Refusal**: **99.22%** (vs 51.77% unframed in Task 2; historical was 93.36%).
- **Model D Step 0 Controller Gap**: **+53.00 pp** (99.22% active vs 46.22% ablated).
- **Model D Step 4000 Active Refusal**: **94.53%** vs 40.63% ablated (**+53.91 pp controller gap**).
- **Hardware & Cost**: 0.0s H100 GPU compute (Zero-H100 Invariant strictly met). Total L40S generation + judging cost: **$1.72 USD**.
- **Classification**: **`REPRODUCED_WITH_KNOWN_FRAMING_DEPENDENCE`**.

---

## 2. Known Remaining Issue Requiring Task 3.2

**IMPORTANT**: Task 3.2 is **ZERO-GPU EVIDENCE RECONCILIATION ONLY**. No retraining and no evaluation reruns are required.

The incoming editor must address the following points in Task 3.2:
1. **Re-derive Historical Seed-1 Persistence Values**: The historical Seed-1 persistence values used in the Task-3.1 report comparison table must be strictly re-derived from authoritative historical Task-7 / Task-8 machine-readable artifacts (`runs/ccpt/task7_4/...` or `artifacts/task8_1_authoritative_metrics.json`) to guarantee 100% numerical precision with historical reports.
2. **Regenerate Corrected Seed-1 1000- and 4000-Step C-vs-D Retention Comparisons**: Compare retention deltas (from Step 0 to 1000 and 4000) using the exact re-derived baselines.
3. **Explicitly Report Model-C Controller Active-vs-Off Behavior at Step 4000**: Model C active refusal at step 4000 is 57.81% and ablated is 61.26% (-3.45 pp delta). Document and interpret this slight negative delta at step 4000 (reflecting benign-adaptation dynamics and lowered over-refusal of 53.52% vs 69.92% at step 0).
4. **No Training/Evaluation Rerun Required**: All 10,752 records and summary metrics needed for Task 3.2 are already generated, verified, and committed.

---

## 3. Critical Policy Instruction Regarding Seed 4

> [!IMPORTANT]
> **SEED 4 HAS NOT YET BEEN AUTHORIZED.**
> Do NOT launch Seed 4.
> Do NOT use H100 compute.
> Task 3.2 must be completed and reviewed before authorizing Seed 4 multi-seed replication.

---

## 4. Key Machine Artifacts & Verification Paths

All files below are committed to Git:
- **Task 3 Forensic Report**: [`docs/research/strengthening_task3_forensic_report.md`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/docs/research/strengthening_task3_forensic_report.md)
- **Task 3.1 Research Report**: [`docs/research/strengthening_task3_1_corrected_evaluation_report.md`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/docs/research/strengthening_task3_1_corrected_evaluation_report.md)
- **Preflight Manifest**: [`artifacts/strengthening_task3_1_preflight.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/strengthening_task3_1_preflight.json)
- **Generation Manifest**: [`artifacts/strengthening_task3_1_generation_manifest.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/strengthening_task3_1_generation_manifest.json)
- **Behavior Summary**: [`artifacts/strengthening_task3_1_behavior_summary.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/strengthening_task3_1_behavior_summary.json)
- **Reproducibility Summary**: [`artifacts/strengthening_task3_1_reproducibility_summary.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/strengthening_task3_1_reproducibility_summary.json)
- **Cost Summary**: [`artifacts/strengthening_task3_1_cost_summary.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/strengthening_task3_1_cost_summary.json)
- **Review Manifest**: [`artifacts/strengthening_task3_1_review_manifest.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/strengthening_task3_1_review_manifest.json)
- **Review Bundle (Zip)**: [`artifacts/strengthening_task3_1_review_bundle.zip`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/strengthening_task3_1_review_bundle.zip)
- **External Assets Provenance Manifest**: [`artifacts/strengthening_task3_1_external_assets_manifest.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/strengthening_task3_1_external_assets_manifest.json)
- **Dedicated L40S Evaluation Runner**: [`modal/strengthening_task3_1_eval.py`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/modal/strengthening_task3_1_eval.py)

### External Storage & Volume References
- **Modal Volume**: `ccpt-authoritative-runs`
- **Task 2 Checkpoints**: `/runs/ccpt/strengthening_task2/seed_20260821/{model}/checkpoints/step_{step}.pt`
- **Task 3.1 Judged Responses (5.4MB, 10,752 lines)**:
  - Remote: `/runs/ccpt/strengthening_task3_1/seed_20260821/judged_responses.jsonl`
  - SHA256: `94435ad45410661a56e0bc0ab53b66c7fc997ef2f22f62a7807be9799204c8fc`
- **Task 2 Judged Responses (5.2MB, 10,752 lines)**:
  - Remote: `/runs/ccpt/strengthening_task2/seed_20260821/judged_responses.jsonl`
  - SHA256: `02de8b2655dbf6ec5f26bf2c50598ffdf212a2c7c5251936089112f5c7d92f15`

---

## 5. Key Test Paths & Commands

To verify full repository integrity:
```bash
# Targeted Task 3 and 3.1 tests:
uv run pytest tests/test_strengthening_task3_forensics.py
uv run pytest tests/test_strengthening_task3_1_preflight.py
uv run pytest tests/test_strengthening_task3_1_regression.py

# Full test suite (all 280 tests):
uv run pytest
```
All tests are confirmed passing as of commit `0b7d4183b392536f6b629738d7445b5d73ab3825`.
