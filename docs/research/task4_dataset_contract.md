# Task 4 Dataset Contract

## 1. Research Dataset Overview & Formally Amended Sources

This contract supersedes the initial exploratory dataset references in Task 1, formally establishing the immutable datasets, splits, and token budgets for the CCPT research pilot.

### Pilot LM Source
- **Repository**: `HuggingFaceFW/fineweb-edu`
- **Config / Subset**: `sample-100BT`
- **Pinned Source Revision**: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- **Token Budget**: **10,000,000,000** training token presentations measured strictly using the frozen `mistralai/Mistral-7B-v0.1` tokenizer.
- **EOS / Document Boundary Counting**: The 10B token budget counts every token presented to the causal language model, including the mandatory `[EOS]` token appended as a document separator after each FineWeb document.
- **Local Materialization Policy**: Task 4 does **not** download or materialize 10B tokens locally. A bounded streaming smoke slice is used for verification; full materialization will occur directly onto a Modal Volume in Task 6.

### Pilot Normative / Safety Source
- **Repository**: `allenai/wildguardmix`
- **Pinned Source Revision**: `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`
- **Training Source**: `wildguardtrain` (split deterministically into 95% training and 5% internal validation).
- **Evaluation-Only Source**: `wildguardtest` (held out strictly for downstream benchmark evaluation).

### Explicit Removal of HH-RLHF & BeaverTails
- **HH-RLHF** and **BeaverTails** are **NOT** part of the pilot training or validation datasets for Models A, B, or C.
- They are reserved exclusively for potential future out-of-distribution transfer comparisons in later tasks.

### WildGuardTest Non-Contamination Rule
- `wildguardtest` must **NEVER** be used for:
  1. Model training or fine-tuning;
  2. Selecting checkpoints or early stopping;
  3. Tuning preprocessing, prompt formatting, or truncation parameters;
  4. Calibrating safety classification thresholds or hyperparameter search.

---

## 2. Safety Supervision Rules and Filtering

### Risk Classification Supervision
- **Target Objective**: Binary cross-entropy on prompt-boundary representation ($r \in \{0, 1\}$).
- **Mapping**:
  - `prompt_harm_label == "harmful"` $\to 1$
  - `prompt_harm_label == "unharmful"` $\to 0$
- **Prompt-Level Grouping**: Deduplicated at the canonical prompt level.
- **Contradictory Label Exclusion**: If duplicate rows with the same canonical prompt have conflicting `prompt_harm_label` values, that prompt group is excluded from risk training.

### Safe-Generation Supervision
- Operates on response-containing rows only.
- **Harmful Prompt Target (Safe Refusal)**:
  - `prompt_harm_label == "harmful"`
  - `response_harm_label == "unharmful"`
  - `response_refusal_label == "refusal"`
- **Benign Prompt Target (Safe Compliance)**:
  - `prompt_harm_label == "unharmful"`
  - `response_harm_label == "unharmful"`
  - `response_refusal_label == "compliance"`
- **Excluded Categories**:
  - Missing or empty responses;
  - Missing labels;
  - Harmful responses (`response_harm_label == "harmful"`);
  - Harmful prompt compliance;
  - Benign prompt refusal (prevents incentivizing trivial over-refusal).

---

## 3. Formatting, Boundaries, and Token Budgets

### Safety Serialization
- **Prompt Framing**: `<s>User: {PROMPT}\nAssistant:`
- **Safe-Generation Continuation**: ` {RESPONSE}</s>`
- **Prompt Boundary**: `prompt_end_index` marks the exact token index of the trailing colon in `\nAssistant:`.

### Presentation Budget Interpretation
- **10B LM Tokens**: $10^{10}$ token presentations during capability pre-training.
- **100M Safety Tokens**: $10^8$ token presentations during joint normative fine-tuning. Because unique WildGuard training tokens are fewer than 100M, training cycles deterministically through the training split with frozen order per epoch to reach the presentation budget.
