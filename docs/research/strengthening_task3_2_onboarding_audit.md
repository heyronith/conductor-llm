# Task 3.2 Onboarding Audit Note

Read-only onboarding verified handoff HEAD `80de4629f1caf15dc3773429e0d58821664b818d` on branch `strengthening-task3-seed1-forensic` with clean remote parity.

## Task 3 forensic conclusion
- Primary root cause: `D_EVALUATION_PROTOCOL_DIVERGENCE` (missing `format_eval_prompt`, generation length 64 vs canonical 48).
- Checkpoints passed structural/provenance forensic checks.

## Task 3.1 correction verified
- Canonical framing and `max_new_tokens=48` recorded in Task 3.1 manifests.
- 10,752 framed judged records; H100 seconds = 0 in cost summary.
- No checkpoint retraining.

## Task 3.1 synthesis defect identified
- Historical persistence join in Task 3.1 reproducibility summary used wrong POST1000 anchors.
- Authoritative historical persistence resolved from `artifacts/task7_3_1a_forensic_summary.json` with Task 7.4 cross-check.

This note is superseded for numerical claims by `artifacts/strengthening_task3_2_reconciliation.json`.
