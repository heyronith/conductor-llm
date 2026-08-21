# Task 7.1 Pilot-v2 Corrective Research Contract & Frozen Specification

**Date**: August 21, 2026  
**Status**: **FROZEN RESEARCH SPECIFICATION**  
**Repository Branch**: `task7.1-corrective`  
**Purpose**: Authoritative Pilot-v2 corrective execution with brand-new canonical 1B LM pretraining for A/B/C/D, exact parameter matching, Checkpoint Format V2, large-scale behavioral evaluation, pinned OOD generalization, and 1,000-step continuation persistence.

---

## 1. Primary Fail-Closed Rule & Fresh 1B Trunks

1. **Mandatory Fresh Training**: Models A, B, C, and D MUST receive brand-new canonical 1B LM pretraining from fresh random initialization.
2. **Strict Non-Reuse Invariant**: No historical Task 6 (`/runs/ccpt/task6/**`) or previous Task 7 checkpoint may be used for initialization.
3. **Dedicated Output Namespace**: Fresh trunks are saved under `/runs/ccpt/task7_1/<run_id>/model_{a,b,c,d}/lm_trunk_1b.pt`.
4. **Automated Checkpoint Hash Comparison**: After training, if ANY fresh A/B/C trunk hash equals any historical Task 6 trunk hash (`9bb8f7f2...`, `c54110a2...`, `ebad5933...`), the experiment ABORTS immediately before safety training.
5. **Initialization Tracking**: Initial state hashes for A, B, C, and D backbone are logged and verified before the first optimizer step.

---

## 2. Experimental Model Architecture & Parameter Matching

All four models operate on the same vocabulary ($V=32,000$) and sequence length ($T=1024$).

| Model Identifier | Architecture Description | LM Pretraining Trainable Params | Safety Trainable Params ($\theta_N$ / Adapters) | Frozen Params During Safety | Total Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A** | Parameter-matched Standard Baseline Transformer | 35,918,848 | 35,918,848 (All) | 0 | 35,918,848 |
| **Model B** | Joint-Training Dual-Stream Control (unprotected) | 35,920,384 | 35,920,384 (All) | 0 | 35,920,384 |
| **Model C** | Protected CCPT Dual-Stream Control | 33,165,824 ($\theta_C$) | **2,754,560 ($\theta_N$)** | **33,165,824 ($\theta_C$)** | 35,920,384 |
| **Model D** | Frozen-Backbone Houlsby Adapter Control | 33,165,824 (Backbone) | **2,757,120 (Adapters)** | **33,165,824 (Backbone)** | 35,922,944 |

### Parameter Accounting Verification
- **Backbone Matching**: Model D backbone parameters ($33,165,824$) exactly match Model C capability parameters ($\theta_C = 33,165,824$).
- **Safety Trainable Matching**: Model D safety parameters ($2,757,120$) match Model C $\theta_N$ ($2,754,560$) within **$2,560$ parameters ($0.0929\% \le 0.1\%$)**.
- **Total Model Matching**: Model D total parameters ($35,922,944$) match Model C total parameters ($35,920,384$) within **$2,560$ parameters ($0.0071\% \le 0.1\%$)**.

---

## 3. Canonical FineWeb Pilot-v2 Data Specification

All data processing strictly utilizes canonical Task 4 functions (`is_validation_document`, `normalize_lm_text`, `tokenize_lm_document`, `PackedTokenBuffer`).

- **Dataset Source**: `HuggingFaceFW/fineweb-edu` (`sample-100BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`).
- **Tokenizer**: `mistralai/Mistral-7B-v0.1` (revision `27d67f1b5f57dc0953326b2601d68371d40ea8da`).
- **Training Prefix**: Exactly $976,544$ packed blocks ($999,981,056$ tokens).
- **Validation Split**: Exactly $1,024$ packed blocks ($1,048,576$ tokens).
- **Persistence Continuation**: Exactly the next $32,000$ contiguous packed blocks ($32,768,000$ tokens) immediately following the 1B prefix.

---

## 4. Checkpoint Format V2 (`ccpt-checkpoint-v2`)

Mandatory fields in every Task 7.1 checkpoint:
- `format_version`: `"ccpt-checkpoint-v2"`
- `model_state_dict`, `optimizer_state_dict`, `scheduler_state`
- `global_step`, `tokens_seen`, `data_cursor`, `stream_identity`
- `data_manifest_hash`, `task4_manifest_hash`
- `safety_schedule_hash` (mandatory and non-empty for safety phases)
- `model_type`, `model_config`, `phase`, `training_seed`
- `torch_rng_state`, `cuda_rng_state`, `git_commit_sha`, `env_versions`

Strict loading fails loudly if any required field is missing or any cryptographic hash mismatches.

---

## 5. Safety Schedule: Full Hash & Zero Dropped Tails

- **Budget**: 20,000,000 valid input-token presentations.
- **Batch Size**: 32 samples.
- **Alternation**: Exactly 1:1 alternating batches (risk, gen, risk, gen...).
- **Deterministic Epoch Permutations**: Permutations generated via `TASK6_SEED = 20260821` and epoch index.
- **No Dropped Tails**: When $k < 32$ examples remain at epoch end, they are combined with the first $32 - k$ examples of the next epoch permutation.
- **Full Schedule SHA256**: Cryptographically hashes `batch_index`, `batch_type`, all 32 example indices, `batch_tokens`, and `cumulative_tokens`.

---

## 6. Real Behavioral & OOD Safety Evaluation

- **Prompt Framing**: Standardized to `<s>User: {PROMPT}\nAssistant:`.
- **In-Distribution Evaluation**: Evaluates $\ge 256$ harmful and $\ge 256$ benign prompts from WildGuard internal validation.
- **Out-of-Distribution (OOD) Evaluation**: Evaluates held-out split of `PKU-Alignment/BeaverTails` (revision pinned, zero training exposure).
- **Metrics**: Harmful Refusal Rate, Unsafe Compliance Rate, Benign Compliance Rate, Over-Refusal Rate.
- **Causal Ablations**: Model C (`controller_scale = 1.0` vs `0.0`), Model D (`adapter_scale = 1.0` vs `0.0`), Model B (`controller_scale = 1.0` vs `0.0`).

---

## 7. 1,000-Step Pure LM Persistence Protocol

- **Starting Checkpoint**: Fresh Task 7.1 20M safety checkpoints.
- **Training Stream**: Exact next $32,000$ canonical FineWeb continuation blocks ($32,768,000$ tokens) following the 1B prefix.
- **Optimizer Steps**: Exactly 1,000 optimizer updates ($32 \times 1024$ tokens/step) with AdamW (`lr = 1e-4`).
- **Update Semantics**:
  - Model A: Normal LM update across all parameters.
  - Model B: Controlled mode update across all parameters.
  - Model C: `mode = "lm"`, updates $\theta_C$ only; $\theta_N$ receives zero gradients and remains untouched.
  - Model D: `adapter_scale = 0.0`, updates backbone only; adapters remain untouched.
- **Evaluation**: Full multi-metric evaluation BEFORE step 1 and AFTER step 1000 to measure retention and delta.
