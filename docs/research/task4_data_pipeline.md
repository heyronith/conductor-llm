# Task 4 Data Pipeline and Tokenization Specification

## 1. Executive Summary and Dataset Architecture
Task 4 establishes the immutable, deterministic data and tokenization infrastructure for the CCPT research project. The pipeline ensures that all three experimental models (Model A Baseline, Model B Joint Control, and Model C CCPT) receive bit-for-bit identical token sequences, supervision labels, prompt boundaries, loss masks, and epoch permutations.

---

## 2. Pinned Remote Source Revisions

The data pipeline pins every external dependency to an exact commit SHA to guarantee reproducibility:

| Component | Hugging Face Repository | Pinned Commit SHA | Config / Split |
| :--- | :--- | :--- | :--- |
| **Tokenizer** | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | Base Tokenizer |
| **LM Pre-Training** | `HuggingFaceFW/fineweb-edu` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | `sample-100BT` (`train`) |
| **Normative Training** | `allenai/wildguardmix` | `d29c47f41c8b51348b5c8e8c81c039b3132b66d1` | `wildguardtrain` |
| **Normative Benchmark** | `allenai/wildguardmix` | `d29c47f41c8b51348b5c8e8c81c039b3132b66d1` | `wildguardtest` (Eval-Only) |

---

## 3. Rationale for WildGuard over HH-RLHF / BeaverTails

The pilot dataset was formally amended to `allenai/wildguardmix` (`wildguardtrain`) for three scientific reasons:
1. **Multi-Aspect Independent Labels**: WildGuard decomposes each interaction into `prompt_harm_label`, `response_harm_label`, and `response_refusal_label`. This enables exact extraction of:
   - **Harmful Prompt $\to$ Safe Refusal** (Risk = 1, Refusal = True)
   - **Benign Prompt $\to$ Safe Compliance** (Risk = 0, Refusal = False)
2. **Prevention of Trivial Over-Refusal**: Datasets like HH-RLHF often confound safety with refusal style. Explicitly including safe compliance on benign prompts prevents a degenerate controller that refuses all user inputs from appearing "safe".
3. **Contradiction Resolution**: Grouping rows by canonical prompt key allows identifying and excluding prompts with contradictory harmfulness labels, preventing noisy gradient conflict.

---

## 4. Frozen Tokenizer Invariants

- **Base Model Tokenizer**: `mistralai/Mistral-7B-v0.1` (Fast Byte-Pair Encoding).
- **Vocabulary Size**: Exactly `32,000`.
- **Special Token IDs**:
  - `unk_token_id`: `0`
  - `bos_token_id`: `1` (`<s>`)
  - `eos_token_id`: `2` (`</s>`)
  - `pad_token_id`: `2` (Assigned to `eos_token_id` purely as a batching collator convention; no new vocabulary token is added).
- **Asset SHA256 Hashes**:
  - `tokenizer.json`: `11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720`
  - `tokenizer_config.json`: `ddb008229511e51607002ffe28925001c4a9ca4177dc4de3a655d085cc610b99`
  - `special_tokens_map.json`: `6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0`
  - `tokenizer.model`: `dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055`

---

## 5. FineWeb-Edu Language Model Pipeline

1. **Text Normalization**:
   - Converts `\r\n` and `\r` to standard `\n`.
   - Rejects empty or whitespace-only documents.
   - Preserves all casing, internal whitespace, punctuation, and formatting unchanged.
2. **Token Stream Construction**:
   - Encodes text with `add_special_tokens=False`.
   - Appends exactly one `[EOS]` token after each document as an explicit document boundary.
   - Conceptually: `doc_1 + [EOS] + doc_2 + [EOS] + ...`
3. **Fixed Sequence Packing**:
   - Packed into contiguous blocks of `sequence_length = 1024` without padding.
   - Every token (including `[EOS]` separators) counts toward the $10^{10}$ pre-training budget.
4. **Disjoint Validation Partition**:
   - Partitioned by document ID hash: `SHA256(f"fineweb_split_v1:{doc_id}") % 1000 == 0`.
   - Assigns ~0.1% of documents to validation, strictly disjoint from training.
5. **Storage Layout**:
   - Serialized as contiguous `uint16` binary files (`tokens-00000.bin`) alongside JSON metadata manifests.

---

## 6. WildGuard Safety Pipeline

1. **Canonical Prompt Grouping**:
   - Canonical key: `unicodedata.normalize('NFC', prompt).replace('\r\n', '\n').strip()`.
   - Excludes prompt groups with conflicting `prompt_harm_label` annotations across variants.
2. **Risk Dataset (Prompt Level)**:
   - Evaluates prompt-level risk: `1` for harmful, `0` for unharmful/benign.
   - Framing: `<s>User: {PROMPT}\nAssistant:`
   - `prompt_end_index`: Token index of the trailing colon `:` in `\nAssistant:`.
3. **Safe-Generation Dataset (Response Level)**:
   - Target filtering:
     - `harmful_refusal`: `prompt_harm == 'harmful'` $\land$ `resp_harm == 'unharmful'` $\land$ `refusal == 'refusal'`.
     - `benign_compliance`: `prompt_harm == 'unharmful'` $\land$ `resp_harm == 'unharmful'` $\land$ `refusal == 'compliance'`.
   - Target continuation: ` {RESPONSE}</s>`.
4. **Deterministic 95/5 Prompt-Group Splitting**:
   - `SHA256(f"wildguard_group_split_v1:{canonical_key}") % 100 < 5` $\to$ validation (5%), else train (95%).
   - Guarantees all response variants of a prompt reside in the same split.
5. **Truncation Rules (Max Length 1024)**:
   - If prompt + response exceeds 1024 tokens:
     - Step 1: Prompt content is truncated from the **left** (retaining latest prompt instructions and `\nAssistant:`).
     - Step 2: If still overflowing, response is truncated from the **right** (preserving trailing `</s>`).
     - Safety invariant: at least 1 response token must follow `prompt_end_index`.

---

## 7. Right-Padding and Masked Loss Computation

Safety sequences are right-padded to the batch maximum length using `pad_token_id = 2` (`eos_token_id`).
- `DataCollatorForRiskTraining` and `DataCollatorForSafeGenerationTraining` produce `input_ids [B, T]` and `attention_mask [B, T]`.
- `safe_generation_loss(logits, input_ids, prompt_end_indices, attention_mask)` computes masked cross-entropy:
  $$\text{mask}_{b, p} = (p \ge \text{prompt\_end\_indices}[b]) \land (\text{attention\_mask}[b, p+1] == 1)$$
- Right-padded tokens are mathematically excluded from gradient contributions.

---

## 8. Deterministic Epoch Ordering

Epoch ordering is governed by `DATA_ORDER_SEED = 20260820`:
$$\text{sort\_key}(\text{example\_id}) = \text{SHA256}(f"{20260820}:\{\text{epoch}\}:\{\text{example\_id}\}")$$
This ensures identical, reproducible example orders across all hardware and platforms without reliance on Python's random state.

---

## 9. Modal Volume Storage Layout
 
 All data pipelines respect `$CCPT_DATA_ROOT` (default `data/processed` locally, `/data/ccpt` on Modal):
 
 ```text
 /data/ccpt/
 ├── tokenizer/
 │   └── 27d67f1b5f57dc0953326b2601d68371d40ea8da/
 │       ├── tokenizer.json
 │       ├── tokenizer_config.json
 │       ├── special_tokens_map.json
 │       └── tokenizer.model
 ├── fineweb/
 │   └── 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/
 │       ├── smoke/
 │       │   ├── train/
 │       │   │   └── smoke_tokens.bin
 │       │   ├── validation/
 │       │   └── manifest.json
 │       └── production/
 │           ├── train/
 │           │   └── tokens-00000.bin
 │           ├── validation/
 │           └── manifest.json
 ├── wildguard/
 │   └── d29c47f41c8b51348b5c8e8c81c039b3132b66d1/
 │       ├── risk/
 │       │   ├── train.arrow
 │       │   └── validation.arrow
 │       ├── generation/
 │       │   ├── train.arrow
 │       │   └── validation.arrow
 │       ├── evaluation/
 │       │   └── wildguardtest.arrow
 │       └── manifest.json
 └── manifests/
     ├── source_lock.json
     └── task4_manifest.json
 ```
 
 ---
 
 ## 10. Remote Data Preparation (Modal CPU Architecture)
 
 To eliminate local machine RAM bottlenecks, heavy dataset streaming and tokenized persistence are offloaded to Modal CPU cloud infrastructure:
 
 ### Local Responsibility:
 - Source code development and repository invariants.
 - Offline unit tests and synthetic test fixtures.
 - Modal invocation orchestration (`modal run modal/task4_data.py`).
 - Manifest and review bundle generation.
 
 ### Modal CPU Cloud Responsibility:
 - **Compute Resources**: 8.0 CPU cores, 32,768 MiB (32 GiB) RAM, 0 GPU.
 - **App Name**: `ccpt-task4-data`.
 - **Persistent Storage**: Modal Volume `ccpt-data` mounted at `/data/ccpt`.
 - **Authentication**: Modal Secret `ccpt-huggingface` (containing authorized `HF_TOKEN`).
 - **Tokenizer Live Verification**: Pinned Mistral-7B-v0.1 asset hash matching.
 - **FineWeb Live Streaming**: Streaming sample packing into uint16 contiguous binary shards with strict exact-budget block limiting ($9,765,625$ blocks for $10^{10}$ tokens).
 - **WildGuard Full Preprocessing**: Canonical grouping, contradiction filtering, left-truncation prompt framing, right-truncation response target continuations, and PyArrow IPC variable-length serialization.
 - **WildGuardTest Evaluation**: Pinned test split schema verification and Arrow artifact generation.
 - **Full Logical Hashing**: Bit-for-bit SHA256 digests computed over entire token sequences for all splits.
 - **Second-Pass Determinism**: Automated re-run validating 100% hash equality across repeated passes.

