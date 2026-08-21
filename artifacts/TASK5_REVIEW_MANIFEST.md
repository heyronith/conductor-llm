# Task 5 / 5.1 / 5.2 Review Manifest: Modal Micro-Overfit, Training-Harness, and Checkpoint Data Lineage Audit

## REPOSITORY STATE
- **Tasks 1–4**: Architecture contract, parameter isolation proofs, gradient firewall tests, dataset pipelines, and Modal live data preprocessing complete and locked.
- **Task 5 / 5.1 / 5.2 Accomplished**:
  - Implemented real-token micro diagnostic configurations (`get_task5_micro_dual_stream_config`, `get_task5_micro_baseline_config`).
  - Added controller ablation API (`controller_scale`) with verified causal equivalence ($\text{scale}=0.0 \equiv \text{mode}=\text{"lm"}$).
  - Built training harness: `checkpoint.py`, `metrics.py`, `engine.py`.
  - Added 10 unit tests in `tests/test_task5_training.py` (total: 79 unit tests passed locally and on Modal CPU).
  - Executed Modal CPU pipeline: verified Task 4 manifest hash, verified zero data leakage (WildGuardTest strictly untouched), generated sanitized deterministic micro subsets, verified deterministic checkpoint resume on CPU.
  - Executed Modal GPU training on NVIDIA A10G (float32, AdamW, cyclic batching):
    - **Model A**: LM loss reduced by 69.9%, Risk accuracy 100%, Safe-gen loss reduced.
    - **Model B**: LM loss reduced by 68.4%, Risk accuracy 100%, Safe-gen loss reduced, joint C/controller gradients verified.
    - **Model C (CCPT)**: LM loss reduced by 68.2%, Risk accuracy 100%, $\theta_C$ 100% frozen bit-for-bit (0 changed tensors in Phase 2 and Phase 3), $\theta_N$ updated (27 changed tensors), Safe-gen loss reduced by 33.8% ($\ge 20\%$), ablated loss penalty 51.0% ($\ge 5\%$), causal LM invariant verified (`True`), controller and deep N gradients active.
  - **Task 5.1 & 5.2 Checkpoint Data Lineage & Hygiene Audit**:
    - Sanitized `task5_subset_manifest.json` (no raw text, prompt keys, or token arrays).
    - Removed `wildguardtest.arrow` inspection from code.
    - Exported all 3 metrics JSONL logs to `artifacts/task5_metrics/` and computed explicit failure flags (Gate near-bound fractions: 0.0%, Steering saturation: 0.0%).
    - Directly opened and inspected all 9 checkpoints on Modal CPU without modifying checkpoint files.
    - Verified unanimous agreement on Task 4 manifest hash and training-time subset hash across all 9 checkpoints.

---

## DATA LINEAGE AUDIT
- **Task 4 Manifest Hash**: `2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097`
- **Training Subset Manifest Hash**: `3480afd5769b483a5b269f0bc8c87188454974bf13f5a862e7906501f933960f` (Verified directly from all 9 checkpoints)
- **Sanitized Review Manifest Hash**: `1b315015ee2e01c86da989192ea789526ec232b052a2349451611552f6935132`
- **Checkpoints Inspected**: `9` (3 per model across all 3 phases)
- **Checkpoint Task 4 Hash Agreement**: `PASS` (100% match)
- **Checkpoint Task 5 Subset Hash Agreement**: `PASS` (100% match)
- **Checkpoint Contents Modified**: `NO`
- **GPU Used for Audit**: `NO` (Modal CPU only)

---

## SANITIZED MANIFEST VERIFICATION
- **Raw Text Present**: `NO`
- **Prompt Group Key Present**: `NO`
- **Records Array Present**: `NO`
- **Complete Input IDs Present**: `NO`

---

## FILES CREATED OR MODIFIED
- `src/ccpt/training/checkpoint.py`: Added `inspect_checkpoint_metadata` and `validate_checkpoint_lineage`.
- `src/ccpt/training/__init__.py`: Exported lineage helpers.
- `tests/test_task5_training.py`: Added tests for lineage extraction, disagreement detection, and sanitized manifest validation (10 total tests).
- `modal/task5_micro.py`: Added dedicated `run_task5_lineage_audit` Modal CPU function.
- `artifacts/task5_lineage_audit.json`: Complete record of checkpoint lineage verification on Modal.
- `artifacts/task5_checkpoint_metadata.json`: Updated with explicit lineage separation and per-checkpoint details.
- `artifacts/task5_subset_manifest.json`: Sanitized review manifest with explicit lineage headers.
- `artifacts/task5_summary.json`: Summary updated with `data_lineage` section.
- `docs/research/task5_micro_training.md`: Added Checkpoint Data Lineage section.
- `artifacts/TASK5_REVIEW_MANIFEST.md`: This review manifest.

---

## MODAL CPU EXECUTION
- **App / Functions**: `ccpt-task5-micro` / `run_task5_lineage_audit`
- **Resources**: 4.0 CPU cores, 8 GiB RAM, 0 GPU
- **Test Results**: 79 passed, 0 failed in 13.53s on Modal CPU
- **Lineage Verification**: All 9 checkpoints verified bit-for-bit

---

## MODEL PERFORMANCE RESULTS (UNCHANGED)
- **Model A**:
  - Phase 1 LM: $10.3903 \to 3.1283$ ($69.9\%$ reduction $\ge 30\%$, Acc: $0.2\% \to 50.3\%$)
  - Phase 2 Risk: $0.7088 \to 0.0033$ (Acc: $100.0\% \ge 95\%$)
  - Phase 3 Safe-Gen: $10.4159 \to 1.4020$
- **Model B**:
  - Phase 1 LM: $10.3870 \to 3.2820$ ($68.4\%$ reduction $\ge 30\%$, Acc: $0.2\% \to 47.6\%$)
  - Phase 2 Risk: $0.6921 \to 0.0246$ (Acc: $100.0\% \ge 95\%$)
  - Phase 3 Safe-Gen: $12.4840 \to 1.0354$
- **Model C (CCPT)**:
  - Phase 1 LM: $10.3870 \to 3.3025$ ($68.2\%$ reduction $\ge 30\%$, Acc: $0.2\% \to 45.5\%$, $\theta_C$ updated, $N$ bypassed)
  - Phase 2 Risk: $0.6927 \to 0.0292$ (Acc: $100.0\% \ge 95\%$, $\theta_C$ changed tensors: 0)
  - Phase 3 Safe-Gen: Controlled: $7.7816$, Ablated: $11.7472$ (Ablation penalty: $51.0\% \ge 5\%$)
  - Causal LM Invariant ($\text{scale}=0.0 \equiv \text{mode}=\text{"lm"}$): `True`
  - $\theta_C$ changed tensors (Phase 3): 0 | $\theta_N$ changed tensors (Phase 3): 27

---

## FAILURE FLAGS
- **NaN / Inf Detected**: `False`
- **Gate Collapse Detected**: `False` (Max near bound fraction: 0.0%)
- **Steering Saturation Detected**: `False` (Max saturate fraction: 0.0%)
- **Dead Controller Detected**: `False` (Active gradients and nonzero steering)
- **Capability Mutation Detected**: `False` (0 changed tensors in $\theta_C$)
- **Data Leakage Detected**: `False` (WildGuardTest untouched)

---

## TESTS
- **Command**: `PYTHONPATH=src python3 -m pytest -v` (Executed on remote Modal CPU)
- **Result**: **79 passed, 0 failed** in 13.53s.

---

## GIT STATUS
```text
On branch main
Changes to be committed:
	new file:   .agents/rules/ccpt-research.md
	new file:   .cursor/rules/ccpt-research.mdc
	new file:   artifacts/TASK1_REVIEW_MANIFEST.md
	new file:   docs/research/task1_ccpt_architecture_spec.md
	new file:   docs/research/task1_design_review.md
	new file:   docs/research/task1_experiment_contract.md
	new file:   docs/research/task1_repo_inventory.md

Changes not staged for commit:
	modified:   .cursor/rules/ccpt-research.mdc
	modified:   artifacts/TASK1_REVIEW_MANIFEST.md
	modified:   docs/research/task1_ccpt_architecture_spec.md
	modified:   docs/research/task1_design_review.md
	modified:   docs/research/task1_experiment_contract.md
```

---

## DEVIATIONS
- None.
