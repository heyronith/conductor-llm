# Task 7.4: Seeds 2 & 3 Prelaunch Replication Protocol & Pre-Registration Document

**Repository:** `heyronith/conductor-llm`  
**Branch:** `task7.4-seeds23-replication-hardening`  
**Status:** Pre-Registration Frozen — Code Hardening Complete — Preflight Authorized  
**Authoritative Date:** August 23, 2026  

---

## 1. Executive Summary & Purpose

This document serves as the formal pre-registration protocol and execution freeze for the replication of the CCPT intrinsic-alignment pilot study across Seeds 2 and 3.

Following the forensic verification of Seed 1 (Task 7.3.1a) and the complete codebase audit freeze (Task 7.3.2), Task 7.4 hardens the execution pipeline, pins runtime dependencies, eliminates ambiguity in data/schedule resolution, and formally freezes Seeds 2 and 3 parameters *before* any GPU compute is spent.

**Critical Policy Invariant:** No GPU execution of Seeds 2 or 3 is performed in Task 7.4. All work in this task is code, configuration, testing, and preflight verification only.

---

## 2. Preregistered Replication Seeds

| Replication Stream | Random Seed | Status | Invariant Scope |
| :--- | :--- | :--- | :--- |
| **Seed 1 (Historical)** | `20260821` | **Frozen & Immutable** | Historical baseline and forensic reference |
| **Seed 2 (Replication 1)** | `20260822` | **Preregistered** | Independent end-to-end replication run |
| **Seed 3 (Replication 2)** | `20260823` | **Preregistered** | Independent end-to-end replication run |

---

## 3. Pinned Runtime Environment Specification

To guarantee bit-for-bit reproducibility across container invocations, the Modal execution environment is pinned to exact versions derived from the audited lockfile:

| Component / Dependency | Exact Pinned Version | Scope |
| :--- | :--- | :--- |
| **Base Python** | `3.11` (debian-slim) | Container runtime |
| **PyTorch (`torch`)** | `2.5.1` | CUDA 12.4 compatible build |
| **Hugging Face `transformers`** | `4.46.3` | Model and pipeline classes |
| **Hugging Face `tokenizers`** | `0.20.3` | Fast tokenizer execution |
| **Hugging Face `datasets`** | `3.1.0` | Dataset stream handling |
| **Hugging Face `huggingface_hub`** | `0.26.2` | Hub artifact retrieval |
| **Hugging Face `accelerate`** | `1.1.1` | Acceleration utilities |
| **Apache `pyarrow`** | `17.0.0` | Arrow table stream serialization |
| **`numpy`** | `2.1.3` | Array math & shard handling |
| **`sentencepiece`** | `0.2.0` | Tokenizer backend |
| **`tiktoken`** | `0.8.0` | Secondary tokenization |
| **`pytest`** | `8.3.3` | In-container verification suite |

**Runtime Container Fingerprinting:** Every Modal container execution captures an environment fingerprint including package versions, CUDA version, GPU model, and git commit SHA. Any mismatch fails closed immediately before training operations begin.

---

## 4. Architectural Invariants & Parameter Counts

The four experimental models maintain exact architectural definitions and parameter matched budgets:

| Model Identity | Description | Total Parameter Count | Parameter Ownership / Partitioning |
| :--- | :--- | :--- | :--- |
| **Model A** | Standard Baseline Transformer | **35,918,848** | 38 tensors, all trainable in LM and Persistence |
| **Model B** | Joint-Training Dual-Stream | **35,920,384** | 65 tensors, all active in controlled generation |
| **Model C** | CCPT Protected Dual-Stream | **35,920,384** | $\theta_C = 33,165,824$ (38 tensors), $\theta_N = 2,754,560$ (27 tensors) |
| **Model D** | Frozen-Backbone Adapter Baseline | **35,922,944** | Backbone $= 33,165,824$ (38 tensors), Safety Adapters $= 2,757,120$ (25 tensors) |

### Layer Configurations:
- **Model A**: 4 layers, $d_{\text{model}} = 512$, $d_{\text{ff}} = 2496$, $n_{\text{heads}} = 8$.
- **Model B / C Capability Stream ($\theta_C$)**: 4 layers, $d_C = 512$, $d_{\text{ff}} = 2048$, $n_{\text{heads}} = 8$.
- **Model B / C Normative Stream ($\theta_N$)**: 2 layers, $d_N = 256$, $d_{\text{ff}} = 1024$, $n_{\text{heads}} = 4$. Controllers at layers $[2, 4]$.
- **Model D**: 4 backbone layers ($d=512$, $d_{\text{ff}}=2048$) with bottleneck residual adapters ($d_{\text{mid}} = 336$) at layers $[1, 2, 3, 4]$.

---

## 5. Dataset & Schedule Provenance

| Artifact | Source Repository / Commit | Canonical Hash / Exact Counts |
| :--- | :--- | :--- |
| **Task 4 WildGuard Manifest** | `data/manifests/task4_manifest.json` | `2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097` |
| **WildGuard Risk Train** | `risk/train.arrow` | Exactly 45,492 records |
| **WildGuard Risk Val** | `risk/validation.arrow` | Exactly 2,344 records |
| **WildGuard Generation Train** | `generation/train.arrow` | Exactly 18,015 records |
| **WildGuard Generation Val** | `generation/validation.arrow` | Exactly 928 records |
| **FineWeb Edu 100BT** | `HuggingFaceFW/fineweb-edu@87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | Shard size 8,192 blocks, sequence length 1,024 |
| **Mistral Tokenizer** | `mistralai/Mistral-7B-v0.1@27d67f1b5f57dc0953326b2601d68371d40ea8da` | Vocab size 32,000 |
| **ID Behavioral Benchmark** | 256 Harmful + 256 Benign Prompts (Salted hash) | Selection Manifest: `bdfec7a39f5304144e55d5647b886ed9bd8c676b73131fcb414f8207232fbbc4` |
| **OOD BeaverTails Benchmark** | `PKU-Alignment/BeaverTails@8401fe609d288129cc684a9b3be6a93e41cfe678` | Selection Manifest: `f8cf3fd0f0ca7502e9b7fef37f49ae4b9fd13cb71438ed64fc093c0649d71b9e` |
| **WildGuard Judge** | `allenai/wildguard@cbba4823f3e8020e5a74a5e29bf85072def6f2ff` | Real moderation model |

---

## 6. Training & Evaluation Protocol

### 1. Phase 1: 1B Language Model Pretraining
- **Tokens:** Exactly 999,981,056 tokens (976,544 blocks $\times$ 1,024 tokens/block).
- **Batch Size:** 32 (32,768 tokens/step, 30,517 total optimizer steps).
- **Optimizer:** AdamW, $\text{lr}_{\text{max}} = 3\times 10^{-4}$, $\text{lr}_{\text{min}} = 0.0$, warmup $100\text{M}$ tokens, cosine decay over $10\text{B}$ tokens.
- **Model Invariants:**
  - Model A, B, D: Standard causal LM training.
  - Model C: $\theta_C$ trains; $\theta_N$ parameters are frozen and bypassed (`mode="lm"`).

### 2. Phase 3: 20M Safety Training
- **Tokens:** Exactly $\ge 20,000,000$ valid input tokens (2,344 balanced 1:1 alternating batches).
- **Batch Size:** 32 examples per batch.
- **Optimizer:** AdamW, $\text{lr}_{\text{max}} = 3\times 10^{-4}$, warmup 400,000 tokens, cosine horizon 40,000,000 tokens.
- **Loss:** $L_{\text{tot}} = L_{\text{risk}} + L_{\text{safe\_gen}}$. Safe-gen CE evaluated with attention mask over true continuation tokens only.
- **Parameter Invariants:**
  - Model A: Standard full-model tuning.
  - Model B: Joint training of all parameters (`mode="controlled"`).
  - Model C: $\theta_C$ frozen; $\theta_N$ trains.
  - Model D: Backbone frozen; safety adapters train.

### 3. Phase 4: Behavioral Safety & Capability Evaluation
- **Generation:** Greedy decoding (`do_sample=False`, `max_new_tokens=48`).
- **Prompt Framing:** Strict single-framing (`<s>User: {prompt}\nAssistant:`).
- **Judge:** External `allenai/wildguard` with tri-state classification (`YES`, `NO`, `NA`).
- **Metrics Reported:**
  - Determinate Safe Refusal Rate ($N_{\text{det}}$) with Wilson 95% CI.
  - Nonzero-NA bounds (lower bound = $\frac{\text{YES}}{N_{\text{total}}}$, upper bound = $\frac{\text{YES} + \text{NA}}{N_{\text{total}}}$).
  - Determinate Benign Non-Refusal Rate / Over-Refusal Rate with Wilson 95% CI.
  - FineWeb validation perplexity on 1,047,552 target tokens (1,048,576 raw tokens).

### 4. Phase 6: 1,000-Step Persistence Experiment
- **Tokens:** Exactly 32,768,000 continuation tokens (blocks $[976544, 1008544)$, ending at 1,032,749,056 total seen tokens).
- **Optimizer:** Fresh AdamW optimizer, resuming LM cosine scheduler at 999,981,056 tokens.
- **Parameter Invariants:**
  - Model A, B: Full parameter tuning on continuation LM data.
  - Model C: $\theta_C$ trains; $\theta_N$ parameters are strictly frozen.
  - Model D: Backbone trains; safety adapters are strictly frozen.

---

## 7. Preflight Execution Verification Summary

Every preflight check has been executed locally and verified:

```json
{
  "authorized_for_seeds_2_and_3_execution": true,
  "seeds_2_and_3_started": false,
  "preregistered_replication_seeds": {
    "seed_1_historical_frozen": 20260821,
    "seed_2_preregistered": 20260822,
    "seed_3_preregistered": 20260823
  },
  "all_preflight_checks_passed": true
}
```

---

## 8. Authorization Gate

Execution of Seeds 2 and 3 is gated by:
1. Pushing the Code-A freeze commit to GitHub and verifying that remote HEAD equals local HEAD.
2. Confirming that all 206 local tests and the Task 7.4 preflight runner pass.
3. Recording the preflight artifact `artifacts/task7_4_seeds23_preflight.json`.
4. Storing the final Evidence-B commit.

**Current Authorization State:** `AUTHORIZED_FOR_SEEDS_2_AND_3_EXECUTION = true`  
**Current Execution State:** `SEEDS_2_AND_3_STARTED = false`
