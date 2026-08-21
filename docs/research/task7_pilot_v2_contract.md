# Task 7 Pilot-v2 Research Contract & Specification

**Date**: August 21, 2026  
**Status**: **FROZEN RESEARCH CONTRACT**  
**Repository Branch**: `task7-pilot-v2`  
**Purpose**: Hardened experimental pipeline, canonical Task 4 data unification, Checkpoint Format V2, and inclusion of Model D (Frozen-Backbone Safety Adapter Control).

---

## 1. Scientific Objective

Task 7 addresses the primary questions raised by the independent end-to-end audit before any 10B scale decision:

1. **Architecture vs Backbone Freezing**: Does CCPT's dual-stream control plane outperform standard parameter-matched frozen-backbone residual adapters (Model D) in safe generation and capability retention?
2. **Data Pipeline Canonical Unification**: Elimination of divergent streaming/packing logic across scripts; canonical Task 4 functions serve as the single source of truth.
3. **True Resume Integrity**: Implementation and verification of Checkpoint Format V2 (`ccpt-checkpoint-v2`) with full environment, RNG, scheduler, and data cursor tracking.
4. **Behavioral Alignment**: Direct autoregressive generation evaluation (refusal rate on harmful prompts, compliance on benign prompts, over-refusal rate) beyond teacher-forced loss.
5. **Persistence of Safety Invariant**: Measurement of safety retention after 1,000 continuous steps of subsequent pure language modeling.

---

## 2. Experimental Model Suite (A / B / C / D)

All models operate over the same vocabulary ($V=32,000$) and sequence length ($T=1024$).

### Model Definitions

| Model Identifier | Topology & Paradigm | Trainable Parameters during LM Pretraining | Trainable Parameters during Safety Training | Frozen Parameters during Safety | Total Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A** | Standard Baseline Transformer + prompt risk head | 35,918,848 (All) | 35,918,848 (All) | 0 | 35,918,848 |
| **Model B** | Joint Dual-Stream Control (unprotected) | 35,920,384 (All) | 35,920,384 (All) | 0 | 35,920,384 |
| **Model C** | Protected CCPT Dual-Stream Control | 33,165,824 ($\theta_C$) | 2,754,560 ($\theta_N$ + controllers + risk) | 33,165,824 ($\theta_C$) | 35,920,384 |
| **Model D** | Frozen-Backbone Residual Adapter Control | 33,165,824 (Backbone) | 2,757,120 (Adapters + risk) | 33,165,824 (Backbone) | 35,922,944 |

### Parameter Accounting Analysis
- **Model C Safety Trainable Budget**: $2,754,560$ parameters (2 normative layers $d_N=256$, input/output projections, risk head).
- **Model D Safety Trainable Budget**: $2,757,120$ parameters (8 bottleneck adapters $d_{\text{mid}}=336$, risk head).
- **Parameter Delta**: Model D is matched to Model C $\theta_N$ within **$2,560$ parameters ($0.09\%$)**.

---

## 3. Canonical Data Pipeline & Stream Invariant

All FineWeb processing MUST strictly use canonical Task 4 functions:
- Document Filtering & Split: `ccpt.data.fineweb.is_validation_document(doc_id, val_modulo=1000)`
- Normalization: `ccpt.data.fineweb.normalize_lm_text(text)`
- Tokenization: `ccpt.data.fineweb.tokenize_lm_document(text, tokenizer)`
- Contiguous Block Packing: `ccpt.data.fineweb.PackedTokenBuffer(sequence_length=1024)`

---

## 4. Checkpoint Format V2 Specification (`ccpt-checkpoint-v2`)

All checkpoints produced in Task 7 and subsequent experiments must conform to `ccpt-checkpoint-v2` and contain:
- `format_version`: `"ccpt-checkpoint-v2"`
- `model_state_dict`: Full parameter state dict.
- `optimizer_state_dict`: Optimizer state dict.
- `scheduler_state`: Learning rate scheduler parameters and state.
- `global_step`: Optimizer update step count.
- `tokens_seen`: Exact cumulative valid token presentations.
- `data_cursor`: Logical block index in the training stream.
- `stream_identity`: String identifier of dataset stream.
- `data_manifest_hash`: SHA256 of data manifest.
- `safety_schedule_hash`: SHA256 of safety schedule.
- `task4_manifest_hash`: SHA256 of Task 4 manifest.
- `model_type`: `'model_a' | 'model_b' | 'model_c' | 'model_d'`.
- `model_config`: Serialized model configuration.
- `phase`: String phase identifier.
- `training_seed`: Integer global seed.
- `torch_rng_state`: PyTorch CPU RNG state.
- `cuda_rng_state`: CUDA RNG state(s) if applicable.
- `git_commit_sha`: Git commit hash at save time.
- `env_versions`: Python, PyTorch, Transformers, Datasets, PyArrow versions.

---

## 5. Safety Protocol & Alternating Batch Specification

- **Batch Size**: 32 samples per batch.
- **Alternation**: Exactly 1:1 alternating batches of:
  - Even batches: **Risk classification** (WildGuard `risk_train`)
  - Odd batches: **Safe generation** (WildGuard `gen_train`)
- **Epoch Wraparound**: Deterministic epoch permutations using `TASK6_SEED = 20260821` and epoch index. No tail samples dropped.
- **Budget**: 20,000,000 valid input tokens.
- **Optimizer**: AdamW (`lr=3e-4`, `betas=(0.9, 0.95)`, `weight_decay=0.1`, `clip=1.0`).
- **Scheduler**: `TokenCosineScheduler` over 40M horizon with 400k token warmup.

---

## 6. Behavioral Evaluation & Persistence Protocol

### Autoregressive Behavioral Evaluation
- **Harmful Prompts**: Measure refusal rate (presence of standardized refusal markers) and unsafe compliance rate ($1 - \text{refusal\_rate}$).
- **Benign Prompts**: Measure helpful compliance rate and over-refusal rate.
- **Causal Ablations**:
  - Model C: `controller_scale=1.0` vs `controller_scale=0.0`.
  - Model D: `adapter_scale=1.0` vs `adapter_scale=0.0`.
  - Model B: `controller_scale=1.0` vs `controller_scale=0.0`.

### Persistence Protocol (The Invariant Test)
- Take safety-trained models (A, B, C, D).
- Execute **1,000 optimizer steps of pure FineWeb language modeling** (batch size 32, 1024 tokens = 32.7M tokens).
- Re-evaluate:
  1. Harmful refusal rate degradation.
  2. Benign compliance rate.
  3. Risk balanced accuracy.
  4. Safe-generation cross-entropy.
  5. Language modeling perplexity.

---

## 7. Out-of-Distribution (OOD) Safety Evaluation

- Held-out safety evaluation prompts from an unobserved distribution (BeaverTails / HH-RLHF distribution).
- Strictly zero training or tuning on OOD data.
