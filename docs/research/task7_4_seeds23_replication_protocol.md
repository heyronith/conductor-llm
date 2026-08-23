# Task 7.4.2: Seeds 2 & 3 Prelaunch Replication Protocol & Pre-Registration Document

**Repository:** `heyronith/conductor-llm`  
**Branch:** `task7.4-seeds23-replication-hardening`  
**Status:** Pre-Registration Frozen — Real Execution Pipeline Implemented — Real Modal Preflight Authorized  
**Authoritative Date:** August 23, 2026  
**Authoritative Code-A SHA:** `a435ddd2b36df2397c7fcf5a8f51b12398289928`

---

## 1. Executive Summary & Purpose

This document serves as the formal pre-registration protocol and execution freeze for the replication of the CCPT intrinsic-alignment pilot study across Seeds 2 and 3.

Following the forensic verification of Seed 1 (Task 7.3.1a), the complete codebase audit freeze (Task 7.3.2), and the Task 7.4.2 execution pipeline implementation and remote Modal preflight authorization, this document establishes the authoritative parameters for Seeds 2 and 3 *before* any GPU compute is spent.

**Critical Policy Invariants:**
- **NO GPU EXECUTION OF SEEDS 2 OR 3 IS PERFORMED IN THIS TASK.**
- **SEED 1 IS NOT RERUN.**
- **10B FULL PRETRAINING IS NOT RUN.**
- All work in Task 7.4.2 is code, configuration, testing, real Modal preflight probing, and preflight authorization only.

---

## 2. Preregistered Replication Seeds & Initialization Hashes

| Stream Identity | Random Seed | Role / Status | Smoke Model B/C Canonical State Dict SHA256 |
| :--- | :--- | :--- | :--- |
| **Seed 1** | `20260821` | **Historical Frozen** | `832a38e1c12298ad4bde00679ebb76fd6e83ea83490f31dc8595a98e9b22dcee` |
| **Seed 2** | `20260823` | **Preregistered Replication 1** | `da8cfc48a44dabbc25ee987f319b53e0b29821432bb4dbceb53bbf65bc661bf5` |
| **Seed 3** | `20260824` | **Preregistered Replication 2** | `d388c2524346f84039e237ae7ba2e59996fae904e4e7bf76e6bfe89ecad9689a` |
| **OOD Selection** | `20260822` | **BeaverTails Benchmark Seed** | *Reserved strictly for OOD dataset curation* |

**Initialization Invariants:**
- Seed 2 Model B and Model C initializations are bit-for-bit identical (`da8cfc48...`).
- Seed 3 Model B and Model C initializations are bit-for-bit identical (`d388c252...`).
- Cross-seed initialization hashes strictly differ across all seeds.

---

## 3. Pinned Runtime Environment Specification

**Environment Label:** `TASK7_4_FROZEN_REPLICATION_ENVIRONMENT`  
*Replication Limitation Note:* Seed-1 exact runtime package versions remain historically uncertain; for Seeds 2 and 3, one exact newly frozen environment is pinned and verified identically across both replications.

| Component / Dependency | Exact Pinned Version | Container Role |
| :--- | :--- | :--- |
| **Base Python** | `3.11` (debian-slim) | Container runtime |
| **PyTorch (`torch`)** | `2.5.1` | CUDA 12.4 compatible build |
| **Hugging Face `transformers`** | `4.46.3` | Model architectures & tokenization |
| **Hugging Face `tokenizers`** | `0.20.3` | Fast tokenizer execution |
| **Hugging Face `datasets`** | `3.1.0` | Stream processing |
| **Hugging Face `huggingface_hub`** | `0.26.2` | Hub artifact retrieval |
| **Hugging Face `accelerate`** | `1.1.1` | Acceleration utilities |
| **Apache `pyarrow`** | `17.0.0` | Arrow table deserialization |
| **`numpy`** | `2.1.3` | Array math & shard handling |
| **`sentencepiece`** | `0.2.0` | Tokenizer backend |
| **`tiktoken`** | `0.8.0` | Secondary tokenization |
| **`pytest`** | `8.3.3` | In-container test suite |

**Runtime Fingerprinting:** Every Modal production function calls `capture_and_verify_runtime_fingerprint(expected_code_sha=TASK7_4_CODE_SHA)` at its FIRST line of execution. Any package version mismatch, GPU mismatch (H100 vs L40S), or Git SHA mismatch raises `RuntimeError` and fails closed.

---

## 4. Architectural Invariants & Parameter Counts

| Model Identity | Description | Total Parameter Count | Parameter Ownership / Partitioning |
| :--- | :--- | :--- | :--- |
| **Model A** | Standard Baseline Transformer | **35,918,848** | 38 tensors, all trainable in LM and Persistence |
| **Model B** | Joint-Training Dual-Stream | **35,920,384** | 65 tensors, all active in controlled generation |
| **Model C** | CCPT Protected Dual-Stream | **35,920,384** | $\theta_C = 33,165,824$ (38 tensors), $\theta_N = 2,754,560$ (27 tensors) |
| **Model D** | Frozen-Backbone Adapter Baseline | **35,922,944** | Backbone $= 33,165,824$ (38 tensors), Safety Adapters $= 2,757,120$ (25 tensors) |

---

## 5. Dataset, Safety Schedule, & Benchmark Provenance

| Artifact | Source Repository / Path | Canonical SHA256 / Full Audit Hash |
| :--- | :--- | :--- |
| **WildGuard Risk Train** | `risk/train.arrow` (45,492 records) | `522ff92e4f02cbaa3ba88838516ce94dfed2434211db669e632efb1be4f3866a` |
| **WildGuard Risk Val** | `risk/validation.arrow` (2,344 records) | `abf37b75cace89a4e7afb4abf0b3f1419656d70e16bbe64346f7dd42c04d424b` |
| **WildGuard Gen Train** | `generation/train.arrow` (18,015 records) | `85fe0fe389080f790959c3ead43534d436c2336f446ec0c9f4bca0b7da918921` |
| **WildGuard Gen Val** | `generation/validation.arrow` (928 records) | `dabc43c7cb0a4af6a56fc183c529b0d04df438b4a7396f438175ad3f0737471d` |
| **Safety Schedule (Legacy Hash)** | `data/safety_schedule.json` (2,344 batches) | `b141fcbc05d8388086f8649d5162c63b4ef862b90e049cbc2e0b29f7f1eb3caa` |
| **Safety Schedule (Full Audit)** | `data/safety_schedule.json` (20,010,611 tokens) | `6e1be80718a7bd9f1fb2f5bd42c87a9cd793afac08694e46f5c449af379ec2a0` |
| **ID Benchmark Manifest** | `data/eval/id_benchmark_manifest.json` (500 items) | `bdfec7a39f5304144e55d5647b886ed9bd8c676b73131fcb414f8207232fbbc4` |
| **OOD BeaverTails Manifest** | `data/eval/ood_beavertails_manifest.json` (500 items)| `f8cf3fd0f0ca7502e9b7fef37f49ae4b9fd13cb71438ed64fc093c0649d71b9e` |
| **FineWeb-Edu 100BT Manifest** | `data/fineweb_authoritative/manifest.json` | `47c3424598d5878e54bf00dc0dd2df2af0217c10780d6c73d11a561220716055` |
| **WildGuard Judge Model** | `allenai/wildguard` | Git Revision `cbba4823f3e8020e5a74a5e29bf85072def6f2ff` |

---

## 6. Authoritative Remote Modal Preflight Results

- **L40S Remote Probe:** `PASSED` (Device: `NVIDIA L40S`, PyTorch: `2.5.1`, CUDA: `12.4`, Code SHA: `a435ddd2...`, Mounts & Schedule Verified)
- **H100 Remote Probe:** `PASSED` (Device: `NVIDIA H100 80GB HBM3`, PyTorch: `2.5.1`, CUDA: `12.4`, Code SHA: `a435ddd2...`, Tensor Math: `1856.0`)
- **Full Unit & Integration Suite:** 211 passed in 35.69s (0 failures, 0 regressions)

---

## 7. Cost Projection & Spending Gate Verification

- **H100 Training Compute:** 8 (seed, model) pipelines $\times$ 2,170s = 17,360s $\approx$ **$19.04**
- **L40S Evaluation Compute:** 8 (seed, model) pipelines $\times$ 900s = 7,200s $\approx$ **$3.90**
- **Centralized WildGuard Judging:** 2 seeds $\times$ 2,160s = 4,320s $\approx$ **$2.34**
- **Total Projected Incremental Spend:** **$25.29**
- **Hard Gate Ceiling:** **$35.00**
- **Cost Gate Decision:** `PASSED` ($25.29 \le \$35.00$)

---

## 8. Final Authoritative Status

```json
{
  "authorized_for_seeds_2_and_3_execution": true,
  "seeds_2_and_3_started": false,
  "full_10b_run_executed": false
}
```
