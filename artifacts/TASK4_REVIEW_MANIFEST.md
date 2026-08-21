# Task 4 / 4.2 Review Manifest: Remote Modal Data Preprocessing & Reproducibility Audit

## REPOSITORY STATE AFTER TASK 4.2
- **Task 1–3**: Fully verified architecture contract, dual-stream models, loss implementations, and gradient firewall tests (`69 passed`).
- **Task 4.2 Reproducibility Audit Accomplished**:
  - **Issue 1 (5-Split Determinism)**: Verified 100% bit-for-bit logical hash determinism across Pass 1 and Pass 2 on Modal CPU for all five splits: `risk_train`, `risk_val`, `gen_train`, `gen_val`, and `eval_records` (WildGuardTest).
  - **Issue 2 (Real Arrow File Round-Trip)**: Directly reloaded all 5 real persisted `.arrow` IPC files from the Modal Volume and verified $H(\text{before save}) == H(\text{after reload})$ bit-for-bit.
  - **Issue 3 (Evaluation Lock in Manifest)**: Locked `raw_eval_row_count = 1725`, `usable_eval_record_count = 1699`, and `eval_logical_hash = 94c8c5322ff5fd013099e2d50727b50c8806e3c4e02216e1d35e8d75b8e175d4` into `data/manifests/task4_manifest.json`.
  - **Issue 4 (Frozen Modal Dependencies)**: Pinned exact versions in `modal/task4_data.py`: `torch==2.13.0`, `transformers==5.15.1`, `tokenizers==0.22.2`, `datasets==5.0.1`, `huggingface_hub==1.28.0`, `pyarrow==25.0.1`, `numpy==2.4.6`.
  - **Minor Fix**: Corrected FineWeb stream loop condition so `max_docs=100` processes exactly `100` documents.

---

## PARTIAL TASK 4 FILES PRESERVED
- `src/ccpt/data/config.py`: Immutable data parameters and pinned repository identifiers.
- `src/ccpt/data/tokenizer.py`: Mistral-7B-v0.1 tokenizer loading and asset verification.
- `src/ccpt/data/collators.py`: Right-padding collators for risk and safe generation batches.
- `src/ccpt/data/ordering.py`: Deterministic epoch permutation generator.
- `src/ccpt/data/hashing.py`: Canonical SHA256 hashing utilities.
- `src/ccpt/training/losses.py`: Safe generation masked cross-entropy loss.

---

## FILES MODIFIED
- `src/ccpt/data/wildguard.py`: Added PyArrow IPC schema and serialization (`save_wildguard_records_arrow`, `load_wildguard_records_arrow`), dual support in `save_wildguard_records` / `load_wildguard_records`, and comprehensive length/truncation percentile statistics.
- `src/ccpt/data/fineweb.py`: Fixed `process_lm_document_stream` loop condition and exact-budget block limiting ($9,765,625$ blocks for $10^{10}$ tokens).
- `src/ccpt/data/manifests.py`: Added evaluation lock fields (`raw_eval_row_count`, `usable_eval_record_count`, `eval_logical_hash`) and execution environment metadata fields.
- `src/ccpt/data/__init__.py`: Exported new Arrow IPC schemas and streaming processors.
- `scripts/prepare_wildguard.py`: Updated to persist Arrow IPC datasets in structured subdirectories (`risk/`, `generation/`).
- `scripts/task4_data_smoke.py`: Updated for memory-safe local testing and Arrow IPC roundtrip validation.
- `tests/test_data_wildguard.py`: Added middle-token mutation hash regression tests, Arrow serialization roundtrip tests, and percentile statistics tests.
- `tests/test_data_fineweb.py`: Added exact budget limiter tests with small analogues.
- `tests/test_manifests.py`: Added unit test for manifest evaluation lock fields.
- `modal/task4_data.py`: Added frozen exact dependencies, 5-split determinism verification, real Arrow roundtrip assertions, and evaluation lock.
- `docs/research/task4_data_pipeline.md`: Added Section 10 ("Remote Data Preparation") and updated volume layout.
- `pyproject.toml`: Added `modal` optional dependency group.
- `data/manifests/task4_manifest.json`: Updated with live production dataset statistics, evaluation lock, and live validation status.

---

## LOCAL ENVIRONMENT
- **OS**: macOS (Darwin 23.6.0, x86_64/arm64)
- **Python**: `3.9.6`
- **PyTorch**: `2.8.0`
- **transformers**: `4.57.6`
- **tokenizers**: `0.22.2`
- **datasets**: `4.5.0`
- **huggingface_hub**: `0.36.2`
- **pyarrow**: `21.0.0`
- **numpy**: `2.0.2`
- **modal**: `1.2.6`
- **pytest**: `8.4.2`

---

## MODAL ENVIRONMENT (FROZEN EXACT VERSIONS)
- **Python**: `3.11.12`
- **PyTorch**: `torch==2.13.0`
- **transformers**: `transformers==5.15.1`
- **tokenizers**: `tokenizers==0.22.2`
- **datasets**: `datasets==5.0.1`
- **huggingface_hub**: `huggingface_hub==1.28.0`
- **pyarrow**: `pyarrow==25.0.1`
- **numpy**: `numpy==2.4.6`
- **modal SDK**: `1.2.6`

---

## MODAL RESOURCES
- **App Name**: `ccpt-task4-data`
- **CPU**: `8.0` cores
- **RAM**: `32,768 MiB` (32 GiB)
- **Disk**: Standard ephemeral container disk
- **GPU**: None (CPU-only data processing)

---

## MODAL STORAGE & PERSISTED ARROW ARTIFACTS
- **Volume Name**: `ccpt-data`
- **Mount Path**: `/data/ccpt`
- **Container Environment Root**: `CCPT_DATA_ROOT=/data/ccpt`
- **Cache Path**: `HF_HOME=/data/ccpt/cache/huggingface`
- **Persisted Dataset Artifacts (All Verified Bit-for-Bit)**:
  - `/data/ccpt/fineweb/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/smoke/train/smoke_tokens.bin` (`869f2e7e1cc89c2aa907b98532d6054973fe5ae2d00d458200af30c932673307`)
  - `/data/ccpt/wildguard/d29c47f41c8b51348b5c8e8c81c039b3132b66d1/risk/train.arrow`
  - `/data/ccpt/wildguard/d29c47f41c8b51348b5c8e8c81c039b3132b66d1/risk/validation.arrow`
  - `/data/ccpt/wildguard/d29c47f41c8b51348b5c8e8c81c039b3132b66d1/generation/train.arrow`
  - `/data/ccpt/wildguard/d29c47f41c8b51348b5c8e8c81c039b3132b66d1/generation/validation.arrow`
  - `/data/ccpt/wildguard/d29c47f41c8b51348b5c8e8c81c039b3132b66d1/evaluation/wildguardtest.arrow`

---

## SOURCE LOCKS
- **Tokenizer**: `mistralai/Mistral-7B-v0.1` @ `27d67f1b5f57dc0953326b2601d68371d40ea8da`
- **Language Model**: `HuggingFaceFW/fineweb-edu` (`sample-100BT`) @ `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- **Normative Dataset**: `allenai/wildguardmix` (`wildguardtrain` / `wildguardtest`) @ `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`

---

## TOKENIZER VERIFICATION
- **Vocab Size**: Exactly `32,000`
- **Special Tokens**: `unk_token_id=0`, `bos_token_id=1`, `eos_token_id=2`, `pad_token_id=2`
- **File Asset Hashes**:
  - `tokenizer.json`: `11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720`
  - `tokenizer_config.json`: `ddb008229511e51607002ffe28925001c4a9ca4177dc4de3a655d085cc610b99`
  - `special_tokens_map.json`: `6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0`
  - `tokenizer.model`: `dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055`

---

## FINEWEB REMOTE SMOKE
- **Docs Streamed**: Exactly `100`
- **Train Docs**: `100`
- **Tokens Generated**: `113,664`
- **1024-Token Blocks Packed**: `111`
- **Smoke Shard Hash**: `869f2e7e1cc89c2aa907b98532d6054973fe5ae2d00d458200af30c932673307`
- **Exact Budget Limit**: Verified mathematically ($10^{10}$ tokens = $9,765,625$ blocks) and unit-tested against overshoot.

---

## WILDGUARD LIVE PRODUCTION VALIDATION
- **Live Validated**: `true`
- **Statistics Source**: `"pinned_live_dataset"`
- **Total Raw Rows (wildguardtrain)**: `86,759`
- **Total Raw Rows (wildguardtest)**: `1,725`
- **Usable Prompt Groups**: `47,836`
- **Conflicting Excluded Prompt Groups**: `15`
- **Risk Records**:
  - `train`: `45,492`
  - `validation`: `2,344`
- **Safe Generation Records**:
  - `train`: `18,015`
  - `validation`: `928`
- **Evaluation Records (WildGuardTest)**: `1,699`
- **Eligible Categories**:
  - `harmful_refusal`: `10,651`
  - `benign_compliance`: `8,292`
- **Excluded Categories**:
  - `benign_refusal` (over-refusals): `8,329`
  - `missing_response` (prompt-only for risk): `48,781`
  - `harmful_response`: `8,368`
  - `harmful_compliance`: `2,294`

---

## TRUNCATION & LENGTH PERCENTILES (LIVE PRODUCTION DATASET)
- **Risk Prompt Token Lengths**:
  - `p50`: `75.0`
  - `p90`: `277.0`
  - `p95`: `351.0`
  - `p99`: `514.65`
  - `max`: `5,256.0`
  - `truncated_count`: `47` (0.098%)
- **Safe-Generation Prompt Token Lengths**:
  - `p50`: `59.0`
  - `p90`: `243.0`
  - `p95`: `308.0`
  - `p99`: `426.58`
  - `max`: `1,031.0`
- **Safe-Generation Response Token Lengths**:
  - `p50`: `229.0`
  - `p90`: `719.0`
  - `p95`: `879.90`
  - `p99`: `1,347.74`
  - `max`: `2,666.0`
- **Safe-Generation Combined Lengths**:
  - `p50`: `349.0`
  - `p90`: `870.0`
  - `p95`: `1,047.0`
  - `p99`: `1,547.16`
  - `max`: `2,686.0`
- **Truncation Policy Application**:
  - `total_eligible_candidates`: `18,943`
  - `truncated_count`: `1,044` (5.51%)
  - `prompt_left_truncated_count`: `1,043`
  - `response_right_truncated_count`: `547`
  - `exhausted_response_count`: `0` (Zero responses squeezed out; safety continuation invariant maintained)

---

## PRODUCTION LOGICAL HASHES & EVALUATION LOCK
- **Risk Train Logical Hash**: `aa7aa36243f43f2779a3914371464fb07df1eda103ec3c24e529eb50ac85523b`
- **Risk Val Logical Hash**: `f47f5fed050a798357fecde8eb595e42f2b60c1ad4723ab8b6a34c7af49cd89d`
- **Gen Train Logical Hash**: `b3d4705f8cb3d8150a2605af03ad7456a33403a29293919ae2ab1c9fc7a54102`
- **Gen Val Logical Hash**: `f7bb470f000c8b4e3254a2e62c3318fbf6fda9fbdbff1b483b7e9578b4855321`
- **Eval (WildGuardTest) Logical Hash**: `94c8c5322ff5fd013099e2d50727b50c8806e3c4e02216e1d35e8d75b8e175d4`

---

## REAL ARROW FILE ROUND-TRIP VERIFICATION ($H(\text{before}) == H(\text{after})$)
- `risk/train.arrow`: `aa7aa36243f43f2779a3914371464fb07df1eda103ec3c24e529eb50ac85523b` $\to$ **MATCH** (45,492 records)
- `risk/validation.arrow`: `f47f5fed050a798357fecde8eb595e42f2b60c1ad4723ab8b6a34c7af49cd89d` $\to$ **MATCH** (2,344 records)
- `generation/train.arrow`: `b3d4705f8cb3d8150a2605af03ad7456a33403a29293919ae2ab1c9fc7a54102` $\to$ **MATCH** (18,015 records)
- `generation/validation.arrow`: `f7bb470f000c8b4e3254a2e62c3318fbf6fda9fbdbff1b483b7e9578b4855321` $\to$ **MATCH** (928 records)
- `evaluation/wildguardtest.arrow`: `94c8c5322ff5fd013099e2d50727b50c8806e3c4e02216e1d35e8d75b8e175d4` $\to$ **MATCH** (1,699 records)

---

## 5-SPLIT DETERMINISM RUN 1 VS RUN 2
- Pass 1 vs Pass 2 `risk_train`: `aa7aa36243f43f2779a3914371464fb07df1eda103ec3c24e529eb50ac85523b` == `aa7aa36243f43f2779a3914371464fb07df1eda103ec3c24e529eb50ac85523b` (**MATCH**)
- Pass 1 vs Pass 2 `risk_val`: `f47f5fed050a798357fecde8eb595e42f2b60c1ad4723ab8b6a34c7af49cd89d` == `f47f5fed050a798357fecde8eb595e42f2b60c1ad4723ab8b6a34c7af49cd89d` (**MATCH**)
- Pass 1 vs Pass 2 `gen_train`: `b3d4705f8cb3d8150a2605af03ad7456a33403a29293919ae2ab1c9fc7a54102` == `b3d4705f8cb3d8150a2605af03ad7456a33403a29293919ae2ab1c9fc7a54102` (**MATCH**)
- Pass 1 vs Pass 2 `gen_val`: `f7bb470f000c8b4e3254a2e62c3318fbf6fda9fbdbff1b483b7e9578b4855321` == `f7bb470f000c8b4e3254a2e62c3318fbf6fda9fbdbff1b483b7e9578b4855321` (**MATCH**)
- Pass 1 vs Pass 2 `eval`: `94c8c5322ff5fd013099e2d50727b50c8806e3c4e02216e1d35e8d75b8e175d4` == `94c8c5322ff5fd013099e2d50727b50c8806e3c4e02216e1d35e8d75b8e175d4` (**MATCH**)
- **5-Split Determinism Verified**: `True` (100% bit-for-bit identical).

---

## LOCAL PYTEST RESULT
- **Command**: `PYTHONPATH=src python3 -m pytest -v`
- **Result**: `69 passed, 0 failed` in `7.85s`.

---

## MODAL RESULT
- App `ccpt-task4-data` initialized on Modal CPU with frozen exact dependency image.
- Tokenizer verified and committed to Volume.
- FineWeb 100-doc smoke sample streamed, packed, and committed to Volume.
- WildGuard live dataset authenticated, loaded (86,759 train rows, 1,725 test rows), preprocessed, validated, persisted as PyArrow IPC `.arrow` datasets to Volume `/data/ccpt`.
- Round-trip reopened and verified all 5 persisted Arrow files.
- Dual-pass determinism verified across all 5 splits.
- Production manifest committed to Volume `/data/ccpt/manifests/task4_manifest.json` and synchronized locally.
- Elapsed time: `117.9s`.

---

## DEVIATIONS / BLOCKERS
- **Zero Blockers**: All four Task 4.2 audit items and reporting items are completely resolved and verified.
