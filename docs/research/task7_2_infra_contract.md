# Task 7.2 Infrastructure Hardening Research Contract & Specification

**Date**: August 22, 2026  
**Status**: **FROZEN RESEARCH SPECIFICATION**  
**Repository Branch**: `task7.2-infra-hardening`  
**Purpose**: Rigorous infrastructure hardening, canonical data lineage validation, strict checkpoint V2 enforcement, external safety judging, and real production-path resume proofs BEFORE any full 1B GPU rerun.

---

## 1. Project Historical Lineage

To ensure absolute scientific integrity and experimental transparency, the repository history is explicitly categorized:

- **Task 6**: Exploratory Pilot-v1.
- **Task 7**: Incomplete attempt.
- **Task 7.1**: Fresh model training occurred (fresh A/B/C/D models, Model D parameter-matched, theta_C frozen in Model C, backbone frozen in Model D), but data streaming, OOD, persistence continuation, resume proofs, and evaluation infrastructure were not authoritative.
- **Task 7.2**: Infrastructure hardening only. No scientific model comparison claims are made. No full 1B GPU rerun is executed in Task 7.2.

---

## 2. Canonical FineWeb Source & Data Stream Contract

1. **Pinned Dataset Source**:
   - Repository: `HuggingFaceFW/fineweb-edu`
   - Config: `sample-100BT`
   - Revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
2. **Pinned Tokenizer**:
   - Repository: `mistralai/Mistral-7B-v0.1`
   - Revision: `27d67f1b5f57dc0953326b2601d68371d40ea8da`
   - Special Tokens: `bos_token_id = 1`, `eos_token_id = 2`, `unk_token_id = 0`
3. **Canonical Functions**:
   - Exclusively uses canonical Task 4 functions: `is_validation_document`, `normalize_lm_text`, `tokenize_lm_document`, `PackedTokenBuffer`, `write_token_shard`, `load_token_shard`.
   - Never reads or imports shards, manifests, or metadata from `/data/task6`.
4. **Logical Stream & Continuation Semantics**:
   - **Pretrain Prefix**: Blocks `[0, 976544)` ($976,544$ blocks = $999,981,056$ tokens).
   - **Persistence Continuation**: Blocks `[976544, 1008544)` (exactly $32,000$ blocks = $32,768,000$ tokens).
   - **Unbroken Packer Invariant**: The token packing buffer is NEVER reset between the 1B prefix and persistence continuation. The continuation blocks are the exact literal next packed tokens from the same continuous stream.
   - **Validation Split**: Blocks `[0, 1024)` ($1,024$ blocks = $1,048,576$ tokens) produced by independent validation stream.
5. **Cryptographic Manifest**:
   - Manifest schema: `ccpt-task7.2-v1` with explicit block ranges, per-shard metadata (ID, path, logical range, block count, token count, byte count, SHA256), ordered prefix/continuation/validation digests, and top-level `manifest_hash`.

---

## 3. Model D Identity-Preserving Initialization Contract

1. **Output Projection Zero-Init**:
   - In `ResidualBottleneckAdapter`, `up_proj.weight` is initialized strictly to zeros (`nn.init.zeros_`).
   - `FrozenBackboneAdapterModel._init_backbone_weights()` initializes the backbone layers, norms, and risk head separately without passing adapter up-projections through a generic reinitialization loop.
2. **Fresh Initialization Invariant**:
   - Before safety training, `logits(adapter_scale = 1.0)` and `logits(adapter_scale = 0.0)` must be numerically identical ($\max |diff| = 0.0$).
3. **Safety Optimization Dynamics**:
   - Adapter parameters receive non-zero gradients under safety loss backward, while backbone parameters remain strictly frozen (`grad is None` or `grad == 0`).
   - After a safety optimizer step, adapter-enabled forward passes diverge from adapter-disabled passes ($\Delta > 0$).

---

## 4. Checkpoint Format V2 Strict Production Schema Contract

1. **Mandatory Base Fields**:
   - `format_version`: `"ccpt-checkpoint-v2"`
   - `model_state_dict`, `optimizer_state_dict`, `scheduler_state`
   - `global_step`, `tokens_seen`, `data_cursor`, `stream_identity`
   - `data_manifest_hash`, `task4_manifest_hash`
   - `model_type`, `model_config`, `phase`, `training_seed`
   - `torch_rng_state`, `cuda_rng_state`, `git_commit_sha`, `env_versions`
2. **Production LM Phase Invariant**:
   - For phases `phase1_pretrain_1b`, `phase1_lm`, `lm_pretrain`, `persistence_continuation`, `persistence_1000`:
     - `optimizer_state_dict` must be non-null.
     - `scheduler_state` must be non-null.
     - `data_manifest_hash` and `task4_manifest_hash` must be non-empty.
     - `stream_identity` and `model_config` must be non-empty.
3. **Production Safety Phase Invariant**:
   - For `phase3_safety_20m` and `safety_20m`, all above requirements hold, and `safety_schedule_hash` must be non-empty.
4. **Validation Failures**:
   - Strict loader raises `ValueError` on any missing mandatory field, hash mismatch, incompatible `model_config`, or mismatched `stream_identity`.

---

## 5. Behavioral Evaluation & External Judge Contract

1. **Prompt Extraction & Anti-Double-Framing**:
   - `extract_raw_prompt` removes any existing `<s>`, `User: `, and `\nAssistant:` artifacts.
   - `format_eval_prompt` applies canonical formatting `<s>User: {raw_prompt}\nAssistant:` with exactly one `User:` and one `Assistant:`.
2. **Primary Behavioral Evaluator**:
   - Pinned Model: `allenai/wildguard`
   - Pinned Revision: `16260a95f9c4501a3556094595e1e7f6e80bb3b3`
   - Classification Categories:
     - Harmful Prompts: `SAFE_REFUSAL` vs `UNSAFE_COMPLIANCE`
     - Benign Prompts: `BENIGN_COMPLIANCE` vs `OVER_REFUSAL`
3. **Secondary Diagnostic**:
   - Substring refusal matching is preserved and reported solely as `heuristic_refusal_rate`, never as the authoritative behavioral metric.

---

## 6. Out-of-Distribution (OOD) BeaverTails Contract

1. **Source**:
   - Repository: `PKU-Alignment/BeaverTails`
   - Revision: `c8306df1cb6c813589b2184d0938ffdf90cb2b00`
   - Split: `30k_test`
2. **Filtering & Deterministic Sampling**:
   - Harmful: `is_safe == False`
   - Benign: `is_safe == True`
   - Deterministic sampling with fixed seed (`20260822`) and cryptographic manifest.
3. **Strict Isolation**:
   - BeaverTails data is strictly quarantined from training, tuning, or hyperparameter selection.

---

## 7. Persistence Stream & Evaluation Contract

1. **Sequential Continuation Iterator**:
   - Consumes blocks `[start_block, start_block + count)` in sequential chunks of batch size 32.
   - Authoritative Pilot-v2 parameters: `start_block = 976544`, `count = 32000`, $1,000$ batches.
   - Zero modulo wraparound or repeated subsets.
2. **Evaluation Metrics**:
   - Evaluates BEFORE and AFTER persistence on Capability, WildGuard, Safe Generation, Behavioral ID, and OOD BeaverTails.
   - Computes absolute DELTA and mathematically well-defined RETENTION ($\text{post}/\text{pre}$ for $\text{pre} > 0$).

---

## 8. Logging & Cost Accounting Contract

1. **JSONL Persistence**:
   - `LiveProgressReporter` writes JSONL lines with Chicago timestamp, UTC timestamp, step, tokens seen, loss, LR, grad norm, throughput, GPU type, VRAM, and measured GPU seconds.
2. **Measured Cost Accounting**:
   - Costs are computed strictly from observed GPU wall seconds: $\text{Cost} = (\text{seconds} / 3600) \times \text{hourly\_rate}$.
   - Hardcoded evaluation cost constants are strictly forbidden.
