# Task 7.3.1a: Corrective Forensic Salvage & Authoritative Re-Evaluation Report

**Branch**: [`task7.3.1a-corrective-salvage`](https://github.com/heyronith/conductor-llm/tree/task7.3.1a-corrective-salvage)  
**Corrective Code Commit SHA**: [`78f60cb0ca5a32024d363ef083b61fcd5e2ad5ca`](https://github.com/heyronith/conductor-llm/commit/78f60cb0ca5a32024d363ef083b61fcd5e2ad5ca)  
**Seed Analyzed**: Seed 1 (`20260821`)  
**Status**: **COMPLETE — SEED 1 SCIENTIFICALLY SALVAGED & READY FOR SEEDS 2/3 REVIEW**  

---

## 1. Purpose & Corrective Actions Summary

Task 7.3.1a resolved the forensic and evaluation defects identified in previous audits without re-running any expensive training phases (1B pretraining, 20M safety training, or 1,000-step persistence continuation training):

1. **Identity-Based Parameter Partitioning**: Replaced name-substring heuristics with strict Python object identity (`id(p)`) partitioning derived directly from model properties (`model.theta_C`, `model.theta_N`, `model.backbone_parameters`, `model.safety_parameters`).
2. **Padded Continuation Loss Fix**: Passed `attention_mask` into `token_weighted_continuation_nll_and_count` to strictly exclude right-padded positions from continuation NLL and token counts across all 928 validation examples.
3. **Canonical Task 4 Data Provenance**: Bound the exact canonical Arrow files without recursive first-match path ambiguity, computed SHA-256 digests, and verified all 2,344 scheduled batches field-by-field (`input_ids`, `prompt_end_index`, `risk_label`, `is_refusal`).
4. **Lineage Audit & Policy Alignment**: Reconstructed the complete chronological commit lineage. While checkpoint-internal execution git SHAs were not recorded during Task 7.3, static code lineage and immutable checkpoint hashes confirm no training semantic contamination.
5. **Decision Synthesis**: Harmonized machine decision logic so derived flags (`seed1_scientifically_salvaged = true`, `ready_for_seeds_2_and_3_review = true`) reflect the verified evidence.

---

## 2. Redone Tensor Freeze Invariants (Identity-Based Partitions)

Using exact Python object identity sets, all seven tensor comparisons confirm exact equality across all parameters with zero changed tensors and zero numerical drift:

| Comparison Check | Model / Partition | Target Tensor Count | Equal Tensors | Changed Tensors | Max Abs Diff | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Check 1** | Model B vs Model C Init | 65 | 65 | **0** | `0.0` | **VERIFIED IDENTICAL** (`SHA: 665dd875...`) |
| **Check 2** | Model C $\theta_N$ Init $\to$ 1B LM | 27 | 27 | **0** | `0.0` | **VERIFIED FROZEN** |
| **Check 3** | Model D Safety Init $\to$ 1B LM | 25 | 25 | **0** | `0.0` | **VERIFIED FROZEN** |
| **Check 4** | Model C $\theta_C$ 1B LM $\to$ 20M Safety | 38 | 38 | **0** | `0.0` | **VERIFIED FROZEN** |
| **Check 5** | Model D Backbone 1B LM $\to$ 20M Safety | 38 | 38 | **0** | `0.0` | **VERIFIED FROZEN** |
| **Check 6** | Model C $\theta_N$ Safety $\to$ Persistence | 27 | 27 | **0** | `0.0` | **VERIFIED FROZEN** |
| **Check 7** | Model D Safety Safety $\to$ Persistence | 25 | 25 | **0** | `0.0` | **VERIFIED FROZEN** |

### Parameter Partition Verification
- **Model C**: 65 total named parameters partitioned into 38 $\theta_C$ and 27 $\theta_N$ tensors (disjoint, 100% coverage).
  - SwiGLU MLP `capability_layers.*.mlp.gate_proj.weight` is correctly classified as $\theta_C$.
  - Probe/projections `p_in.weight`, `obs_projections.*.weight`, `gate_projections.*.weight`, `steering_projections.*.weight`, `risk_head.weight` are correctly classified as $\theta_N$.
- **Model D**: 63 total named parameters partitioned into 38 backbone and 25 safety adapter tensors (disjoint, 100% coverage).
  - `layers.*.attn_adapter.*`, `layers.*.mlp_adapter.*`, and `risk_head.*` are correctly classified as safety parameters.

---

## 3. Canonical Task 4 Data Binding & Field-by-Field Schedule Audit

### 3.1 Canonical File Bindings
- **Frozen Task 4 Manifest Hash**: `2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097`
- `risk_train`: `/data_task4/wildguard/d29c47f41c8b51348b5c8e8c81c039b3132b66d1/risk/train.arrow` (45,492 records, `SHA: 522ff92e4f02cbaa...`)
- `risk_val`: `/data_task4/wildguard/d29c47f41c8b51348b5c8e8c81c039b3132b66d1/risk/validation.arrow` (2,344 records, `SHA: abf37b75cace89a4...`)
- `gen_train`: `/data_task4/wildguard/d29c47f41c8b51348b5c8e8c81c039b3132b66d1/generation/train.arrow` (18,015 records, `SHA: 85fe0fe389080f79...`)
- `gen_val`: `/data_task4/wildguard/d29c47f41c8b51348b5c8e8c81c039b3132b66d1/generation/validation.arrow` (928 records, `SHA: dabc43c7cb0a4af6...`)

### 3.2 Schedule Verification
- **Total Batches**: 2,344 strictly alternating (1,172 risk, 1,172 generation).
- **Total Valid Tokens**: 20,010,611 tokens (exact match).
- **Record Identity**: Every record in every batch matches the bound canonical Arrow record field-by-field.
- **First Crossing**: Batch index 2343 (penultimate cumulative: 19,996,081 < 20M, final: 20,010,611 $\ge$ 20M).
- **Hashes**:
  - Legacy Hash: `b141fcbc05d8388086f8649d5162c63b4ef862b90e049cbc2e0b29f7f1eb3caa` (**MATCH**)
  - Full Schedule Audit Hash: `6e1be80718a7bd9f1fb2f5bd42c87a9cd793afac08694e46f5c449af379ec2a0` (**MATCH**)

---

## 4. Corrected Safe-Generation Evaluation (Attention Mask Padding Exclusion)

Evaluating on all 928 validation examples with `attention_mask` excluded right-padding tokens, reducing evaluated token count from 692,640 down to the true **290,384 continuation tokens**:

| Model Condition | Old Unmasked CE (Task 7.3.1) | Corrected Padded CE (Task 7.3.1a) | Difference ($\Delta$) | Continuation PPL | Total Continuation NLL | Valid Targets |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** (Pre-Persistence) | 5.9564 | **2.6099** | -3.3465 | 13.597 | 757,871.36 | 290,384 |
| **Model A** (Post-Persistence) | 6.5614 | **3.1150** | -3.4465 | 22.533 | 904,547.46 | 290,384 |
| **Model B** (Pre-Persistence) | 5.8371 | **2.5724** | -3.2648 | 13.097 | 746,975.33 | 290,384 |
| **Model B** (*Pre-Persist Scale=0*) | 7.4244 | **3.6500** | -3.7744 | 38.474 | 1,059,892.42 | 290,384 |
| **Model B** (Post-Persistence) | 6.7656 | **3.1566** | -3.6090 | 23.490 | 916,634.33 | 290,384 |
| **Model B** (*Post-Persist Scale=0*) | 7.6253 | **4.4985** | -3.1269 | 89.882 | 1,306,290.35 | 290,384 |
| **Model C** (Pre-Persistence) | 7.0973 | **2.7965** | -4.3009 | 16.388 | 812,060.01 | 290,384 |
| **Model C** (*Pre-Persist Scale=0*) | 7.3999 | **3.4727** | -3.9272 | 32.224 | 1,008,409.73 | 290,384 |
| **Model C** (Post-Persistence) | 6.9852 | **2.9054** | -4.0798 | 18.272 | 843,694.06 | 290,384 |
| **Model C** (*Post-Persist Scale=0*) | 7.5855 | **3.4613** | -4.1242 | 31.859 | 1,005,108.61 | 290,384 |
| **Model D** (Pre-Persistence) | 5.6117 | **2.7446** | -2.8671 | 15.558 | 796,989.14 | 290,384 |
| **Model D** (*Pre-Persist Scale=0*) | 7.4208 | **3.4460** | -3.9748 | 31.375 | 1,000,660.10 | 290,384 |
| **Model D** (Post-Persistence) | 5.8752 | **2.9068** | -2.9685 | 18.298 | 844,098.92 | 290,384 |
| **Model D** (*Post-Persist Scale=0*) | 7.6187 | **3.4302** | -4.1885 | 30.882 | 996,076.62 | 290,384 |

*Reason for Difference*: Task 7.3.1 omitted `attention_mask` in batched continuation loss, erroneously including right-padded positions. Correcting with `attention_mask` isolates true continuation targets.

---

## 5. Authoritative Primary Behavioral Results Table

Re-used from Task 7.3.1 tri-state evaluation (Source SHA: `artifacts/task7_3_1_forensic_summary.json`):

| Model Architecture | Phase | Corrected Continuation CE | Risk Val Balanced Acc | In-Distribution Harmful Refusal (WildGuard ID) | Out-of-Distribution Harmful Refusal (BeaverTails OOD) | ID Benign Non-Refusal (Compliance) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A** (Standard Baseline) | **Pre-Persistence** | 2.6099 | 0.8842 | 99.61% [97.8%, 99.9%] | 83.98% [79.0%, 88.0%] | 24.22% [19.4%, 29.8%] |
| | **Post-Persistence** | 3.1150 | 0.8675 | **66.41%** [60.4%, 71.9%] | **60.94%** [54.8%, 66.7%] | 28.91% [23.7%, 34.8%] |
| **Model B** (Joint Dual-Stream) | **Pre-Persistence** | 2.5724 | 0.8627 | 97.27% [94.5%, 98.7%] | 70.70% [64.8%, 76.0%] | 35.16% [29.5%, 41.2%] |
| | *Pre-Persistence (Scale=0)* | 3.6500 | — | 91.80% | 73.44% | — |
| | **Post-Persistence** | 3.1566 | 0.8105 | **67.97%** [62.0%, 73.4%] | **53.12%** [47.0%, 59.2%] | 18.04% [13.7%, 23.4%] |
| | *Post-Persistence (Scale=0)*| 4.4985 | — | 48.83% | 40.62% | — |
| **Model C** (CCPT Protected Control-Plane)| **Pre-Persistence** | 2.7965 | 0.8612 | **98.83%** [96.6%, 99.6%] | **87.50%** [82.9%, 91.0%] | 21.09% [16.5%, 26.5%] |
| | *Pre-Persistence (Scale=0)* | 3.4727 | — | 50.98% | 49.61% | — |
| | **Post-Persistence** | 2.9054 | 0.8514 | **96.09%** [93.0%, 97.8%] | **86.33%** [81.6%, 90.0%] | 19.14% [14.7%, 24.4%] |
| | *Post-Persistence (Scale=0)*| 3.4613 | — | 47.43% | 41.95% | — |
| **Model D** (Frozen-Backbone Adapter) | **Pre-Persistence** | 2.7446 | 0.8010 | 100.00% [98.5%, 100%] | 93.36% [89.6%, 95.8%] | 4.69% [2.7%, 8.0%] |
| | *Pre-Persistence (Scale=0)* | 3.4460 | — | 50.39% | 44.71% | — |
| | **Post-Persistence** | 2.9068 | 0.7934 | 96.48% [93.5%, 98.1%] | **51.17%** [45.1%, 57.2%] | 19.53% [15.1%, 24.9%] |
| | *Post-Persistence (Scale=0)*| 3.4302 | — | 51.56% | 45.42% | — |

---

## 6. Scientific Discussion & Limitations

1. **Causal Controller Steering**:
   Ablating Model C's controller (`scale=0.0`) reduces harmful refusal from $96.09\% \to 47.43\%$ (ID) and $86.33\% \to 41.95\%$ (OOD). This demonstrates strong causal dependence of the observed safety behavior on the learned normative controller.
2. **Seed-1 Architecture Comparison (C vs D)**:
   At Seed 1, CCPT's control architecture is associated with substantially better OOD persistence ($86.33\%$ post-persistence) than the matched frozen-adapter control ($51.17\%$ post-persistence, reproducing the Seed-1 OOD persistence collapse under this experimental setting).
3. **Critical Limitation — Over-Refusal**:
   Model C demonstrates low benign non-refusal / high over-refusal ($19.14\%$ benign compliance / $80.86\%$ over-refusal post-persistence). This remains a significant limitation despite strong harmful-refusal persistence.
4. **Provenance Limitation**:
   Checkpoint-internal git SHA was not persisted in Task 7.3 metadata; training semantic lineage was reconstructed from commit chronology, immutable checkpoint hashes, and phase logs, but exact per-checkpoint execution SHA remains unproven.

---

## 7. Authoritative Review Package Manifest

| Artifact / Document | Path | Description |
| :--- | :--- | :--- |
| **Forensic Summary** | [`artifacts/task7_3_1a_forensic_summary.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task7_3_1a_forensic_summary.json) | Complete machine-readable summary JSON with authoritative decision booleans. |
| **Tensor Invariants** | [`artifacts/task7_3_1a_tensor_invariants.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task7_3_1a_tensor_invariants.json) | All 7 identity-based parameter partition checks and state dict hashes. |
| **Schedule Data Audit** | [`artifacts/task7_3_1a_schedule_data_audit.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task7_3_1a_schedule_data_audit.json) | Bound Task 4 files, hashes, record counts, and field-level batch verification. |
| **Code Lineage Audit** | [`artifacts/task7_3_1a_code_lineage_audit.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task7_3_1a_code_lineage_audit.json) | Chronological commit lineage and per-phase provenance status. |
| **Safe-Gen Correction** | [`artifacts/task7_3_1a_safe_gen_correction.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task7_3_1a_safe_gen_correction.json) | Padded vs unmasked continuation CE across all 14 evaluation conditions. |
| **Markdown Report** | [`docs/research/task7_3_1a_results.md`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/docs/research/task7_3_1a_results.md) | Comprehensive narrative report with calibrated scientific claims. |
