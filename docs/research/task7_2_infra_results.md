# Task 7.2 Infrastructure Hardening Results Report

**Date**: August 22, 2026  
**Status**: **TASK 7.2 COMPLETE — READY FOR INDEPENDENT REVIEW**  
**Repository Branch**: `task7.2-infra-hardening`  
**Execution Mode**: Bounded real-world proofs & local CPU unit/integration tests (Compute rule enforced: NO full 1B run executed).

---

## 1. Summary of Infrastructure Findings and Fixes

| Audit Finding | Status in Task 7.1 | Resolution in Task 7.2 |
| :--- | :--- | :--- |
| **FineWeb Materializer** | Reused `/data/task6` shards | Built canonical materializer streaming from pinned source `HuggingFaceFW/fineweb-edu@87f09149...` with zero Task 6 references. |
| **Persistence Continuation** | Reused Shard 0 with `(step*32)%3000` | Created sequential iterator consuming continuous blocks `[976544, 1008544)` with unbroken packer. |
| **Model D Zero-Init** | Overwritten by generic `_init_backbone_weights` | Modular initialization guarantees `up_proj.weight == 0.0` and `logits(scale=1.0) == logits(scale=0.0)`. |
| **BeaverTails OOD** | 8 hand-written prompts repeated 32 times | Implemented real loader from `PKU-Alignment/BeaverTails@c8306df1...` with deterministic sampling. |
| **Behavioral Prompt Framing** | Double-framed `<s>User: User: ...` | Implemented `extract_raw_prompt` preventing duplicate framing. |
| **Primary Safety Judge** | Substring refusal matching | Pinned `allenai/wildguard@16260a95...` as primary judge; substring heuristic demoted to secondary diagnostic. |
| **Resume Proof** | Synthetic random data assertion | Real production path with canonical FineWeb, AdamW, TokenCosineScheduler, and Checkpoint V2 proving bitwise equivalence. |
| **Checkpoint V2 Strictness** | Incomplete enforcement | Enforced mandatory base fields, production phase non-null optimizer/scheduler/hashes, config and stream matching. |
| **Cost Accounting** | Partially hardcoded ($0.35 eval) | Implemented strictly measured wall-second cost calculations. |
| **JSONL Logging** | Not persisted to disk | Enforced real file writing in `LiveProgressReporter`. |

---

## 2. Bounded Canonical Materialization Proof

- **Source Dataset**: `HuggingFaceFW/fineweb-edu` (`sample-100BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`)
- **Tokenizer**: `mistralai/Mistral-7B-v0.1` (`27d67f1b5f57dc0953326b2601d68371d40ea8da`)
- **Prefix Range**: Blocks `[0, 50)`
- **Continuation Range**: Blocks `[50, 70)` (starts at block 50, exactly after prefix)
- **Validation Range**: Blocks `[0, 10)`
- **Prefix Hash**: `2504d20aca45cea1b22ce6ae8762d279ff81aadd49902926136aec04e35c8343`
- **Continuation Hash**: `ac6e0c1d1f3c9914fea3cf0bd5e86ad29d75c96a0ba31fc5cf691ca0e8050b12`
- **Validation Hash**: `cb3b43ece82135a189edc02ed84bfe9a4c151dd746a190edba1b481b2629c1ce`
- **Manifest Hash**: `c48027df91ea331d8cb271c9cb64ebbc68e1a36935f1d35f204d97c6518fd6e6`
- **Byte-for-Byte Continuation Proven**: `True`

---

## 3. Model D Identity-Preserving Initialization Proof

- **Total Parameters**: $35,922,944$
- **Backbone Parameters**: $33,165,824$ (Matches Model C $\theta_C$)
- **Trainable Safety Parameters**: $2,757,120$ (Matches Model C $\theta_N$ within $0.0929\% \le 0.1\%$)
- **Adapter Up-Projection Weights**: Strictly zero (`torch.equal(w, 0) == True`)
- **Fresh Forward Difference**: $\max | \text{logits}(\text{scale}=1.0) - \text{logits}(\text{scale}=0.0) | = \mathbf{0.0}$
- **Safety Step Activation**: One safety optimizer step causes scale 1.0 logits to diverge ($\Delta > 10^{-4}$), while backbone parameters remain strictly frozen with zero gradients.

---

## 4. Checkpoint V2 Production Resume Proof

- **Data Source**: Real canonical FineWeb materialization
- **Total Steps**: 8 (Batch Size 4, Sequence Length 64)
- **Interrupt Step**: 4
- **State Restored**: Model parameters, AdamW optimizer state, TokenCosineScheduler tokens/LR, PyTorch RNG, data cursor ($16$).
- **Pre-Step 5 Proof**:
  - `logical_blocks_identical`: `True`
  - `batch_sha_identical`: `True`
  - `lr_identical`: `True`
  - `tokens_seen_identical`: `True`
  - `optimizer_state_loaded`: `True`
  - `scheduler_state_loaded`: `True`
- **Post-Step 8 Comparison**:
  - `max_model_param_diff`: $\mathbf{0.0}$
  - `LOGICAL_RESUME_EQUIVALENT`: `True`
  - `BITWISE_RESUME_EQUIVALENT`: `True`

---

## 5. Behavioral Safety & OOD Evaluation Proofs

- **Behavioral Prompt Formatting**:
  - `format_eval_prompt("Explain photosynthesis")` -> `"<s>User: Explain photosynthesis\nAssistant:"`
  - `format_eval_prompt("User: Explain photosynthesis\nAssistant:")` -> `"<s>User: Explain photosynthesis\nAssistant:"`
  - Exactly one `User:` and one `Assistant:` in all cases (Zero double-framing).
- **External Safety Judge**:
  - Pinned Model: `allenai/wildguard@16260a95f9c4501a3556094595e1e7f6e80bb3b3`
  - Classifications: Harmful (Safe Refusal vs Unsafe Compliance), Benign (Helpful Compliance vs Over-Refusal).
- **BeaverTails OOD Loader**:
  - Pinned Source: `PKU-Alignment/BeaverTails@c8306df1cb6c813589b2184d0938ffdf90cb2b00` (Split: `30k_test`)
  - Deterministic Sample Manifest Hash: `ca655166cca7d06e607fbe46d84190a421e18ef18b4c3c843e9cdb78f336eaf9`

---

## 6. Full Test Suite Verification

- Total Tests Executed: **146**
- Total Tests Passed: **146 (100%)**
- Total Tests Failed: **0**
- Execution Duration: **25.49s**

---

## 7. Compute Rule & Cost Accounting

- **Full 1B Rerun Executed**: `False`
- **Actual Measured CPU Runtime**: ~3.39 seconds (Proofs) + ~25.5 seconds (Full test suite)
- **Actual Measured GPU Runtime**: 0.0 seconds
- **Accrued GPU Cost**: $0.00
