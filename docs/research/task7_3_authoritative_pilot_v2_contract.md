# Task 7.3: Authoritative Pilot-v2 Experiment Contract

**Frozen Contract Identifier**: `ccpt-task7.3-authoritative-pilot-v2-contract-v1`  
**Date**: 2026-08-22  
**Status**: FROZEN PRIOR TO GPU EXECUTION  
**Primary Seed**: `20260821`  
**OOD Seed**: `20260822`  

---

## 1. Scientific Purpose & Architecture Invariants

This contract governs the first authoritative Pilot-v2 scientific experiment testing the Constitutional Control-Plane Transformer (CCPT) architecture against three matched baselines.

The central research questions are:
1. **Capability Preservation**: Does CCPT (Model C) protect core language capabilities post-safety fine-tuning without significant PPL degradation compared to a frozen-backbone adapter control (Model D)?
2. **Safety Learning**: Does CCPT achieve competitive safety learning (risk balanced accuracy, safe generation cross-entropy) relative to standard fine-tuning (Model A), joint-stream training (Model B), and adapter control (Model D)?
3. **Behavioral Alignment & OOD Transfer**: Does CCPT produce high safe refusal on harmful requests and high benign non-refusal on benign requests on both in-distribution (WildGuard) and out-of-distribution (BeaverTails) prompts?
4. **Causal Mechanism Dependence**: Does ablating the normative controller (setting controller scale = 0) causally eliminate learned safety behavior?
5. **Safety Persistence**: Does learned safety behavior persist after 1,000 steps (~32.8M tokens) of renewed causal language modeling pretraining?

---

## 2. Model Definitions & Parameter Budgets

| Model | Architecture Description | Total Parameters | Capability ($\theta_C$) | Normative / Safety ($\theta_N$) | LM Trainable | Safety Trainable |
|---|---|---|---|---|---|---|
| **Model A** | Parameter-Matched Standard Transformer | 35,918,848 | 35,918,336 (Core) | 512 (Risk Head) | 35,918,848 | 35,918,848 |
| **Model B** | Joint-Training Dual-Stream Control | 35,920,384 | 33,165,824 | 2,754,560 | 35,919,360* | 35,920,384 |
| **Model C** | CCPT Protected Dual-Stream | 35,920,384 | 33,165,824 | 2,754,560 | 33,165,824 | 2,754,560 |
| **Model D** | Frozen-Backbone Adapter Control | 35,922,944 | 33,165,824 (Backbone) | 2,757,120 (Adapters + Risk) | 33,165,824 | 2,757,120 |

*\*Note on Model B LM Trainable*: Model B runs in `mode="controlled"` during LM. All parameters that naturally participate in forward computation receive gradients (theta_C + p_in + obs_projections + normative_layers + gate_projections + steering_projections). The risk_head (512 params) and normative_final_norm (512 params) do not participate in promptless LM next-token logits and receive `grad=None`. No synthetic losses are introduced.

### Model D Matching Tolerance
- Total parameters difference vs Model C: $35,922,944 - 35,920,384 = 2,560$ ($+0.007\% \le 0.1\%$).
- Safety parameters difference vs Model C: $2,757,120 - 2,754,560 = 2,560$ ($+0.093\% \le 0.1\%$).
- Model D adapter up-projections initialize strictly to 0.0, ensuring exact identity preservation before safety training ($\text{logits}_{\text{adapter}=1} \equiv \text{logits}_{\text{adapter}=0}$).

---

## 3. Initialization & Lineage Rules

1. **Clean Lineage**: All model trunks are initialized fresh using seed `20260821`. Historical Task 6/7 checkpoints (`/runs/ccpt/task6/**`, `/runs/ccpt/task7/**`) are forbidden as initialization sources.
2. **Model B/C Initialization Equality**: Initial state dictionary SHA256 hashes must be bit-for-bit identical:
   $$\text{SHA256}(\text{state\_dict}(B_{\text{init}})) == \text{SHA256}(\text{state\_dict}(C_{\text{init}}))$$
3. **Frozen Hashes**:
   - Task 4 Manifest Hash: `2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097`
   - BeaverTails Pinned Revision: `8401fe609d288129cc684a9b3be6a93e41cfe678` (split `30k_test`)
   - WildGuard Judge Pinned Revision: `cbba4823f3e8020e5a74a5e29bf85072def6f2ff`

---

## 4. Canonical FineWeb Data Specification

- **Source Dataset**: `HuggingFaceFW/fineweb-edu` (config: `sample-100BT`, revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, split: `train`)
- **Tokenizer**: `mistralai/Mistral-7B-v0.1` (revision: `27d67f1b5f57dc0953326b2601d68371d40ea8da`, vocab size: 32,000)
- **Sequence Length**: 1,024 tokens
- **Logical Ranges**:
  - **1B Pretraining Prefix**: Blocks $[0, 976544)$ = 976,544 blocks = 999,981,056 tokens
  - **Persistence Continuation**: Blocks $[976544, 1008544)$ = 32,000 blocks = 32,768,000 tokens
  - **Validation Set**: Blocks $[0, 1024)$ = 1,024 blocks = 1,048,576 tokens
- **Packing Invariant**: The training stream uses ONE continuous `PackedTokenBuffer`. Block 976,544 is packed continuously from block 976,543 without resetting the token buffer.
- **Manifest Invariant**: The manifest contains only block metadata and cryptographic hashes; no `raw_bytes_b64` or raw token arrays.

---

## 5. Phase Specifications

### Phase 1: 1B Language Model Pretraining
- **Tokens per Model**: 999,981,056 tokens (30,517 steps $\times$ 32 batch size $\times$ 1,024 seq len = 32,768 tokens/step)
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$, weight decay = $0.1$, grad clip = $1.0$, precision = `bf16`)
- **Scheduler**: `TokenCosineScheduler` (max LR = $3\times 10^{-4}$, min LR = $0.0$, warmup = 100,000,000 tokens, total horizon = 10,000,000,000 tokens)
- **Model Forward & Update Rules**:
  - Model A: Normal forward pass; all parameters train.
  - Model B: Controlled forward pass (`mode="controlled"`); all parameters in optimizer.
  - Model C: Capability-only forward pass (`mode="lm"`); $\theta_N$ parameters bypassed and frozen ($\text{grad} = \text{None}$).
  - Model D: Adapter bypassed (`adapter_scale=0.0`); safety parameters frozen ($\text{grad} = \text{None}$).
- **Invariants**:
  - Model C: Changed $\theta_N$ tensors = 0.
  - Model D: Changed safety adapter/risk tensors = 0.

### Phase 2: 20M Safety Training
- **Data Source**: Task 4 Prepared WildGuard Train partition (Risk: 45,492 examples, Generation: 18,015 examples). *WildGuardTest remains strictly sealed.*
- **Schedule**: Deterministic 1:1 alternating batches (Risk $\leftrightarrow$ Generation), batch size = 32.
- **Tail Handling**: Zero dropped tails across epoch boundaries.
- **Token Accounting**: Valid safety tokens are computed as $\sum_{i=1}^{32} \text{len}(\text{record.input\_ids}_i)$ (excluding padding).
- **Target Horizon**: Exactly the first complete batch crossing $\ge 20,000,000$ cumulative valid input tokens.
- **Optimizer & Scheduler**:
  - AdamW ($\text{lr}=3\times 10^{-4}$, $\beta_1=0.9, \beta_2=0.95, \text{weight\_decay}=0.1$)
  - `SafetyTokenCosineScheduler` (warmup = 400,000 safety tokens, total horizon = 40,000,000 safety tokens)
- **Model Update Rules**:
  - Model A: All parameters train on risk loss and safe generation loss.
  - Model B: Controlled mode; all parameters train.
  - Model C: $\theta_C$ parameters frozen ($\text{requires\_grad}=\text{False}$); only $\theta_N$ optimizer updates. Gradients propagate through frozen capability layers.
  - Model D: Backbone frozen ($\text{requires\_grad}=\text{False}$); only adapter and risk head parameters train.
- **Freeze Invariants**:
  - Model C: Changed $\theta_C$ tensors = 0.
  - Model D: Changed backbone tensors = 0.

### Phase 3: 1,000-Step Persistence Experiment
- **Data Range**: FineWeb continuation blocks $[976544, 1008544)$ = 32,000 blocks = 32,768,000 tokens (1,000 steps $\times$ 32 batch size).
- **Scheduler Continuity**: The LM `TokenCosineScheduler` resumes at token position $999,981,056$ and advances to $1,032,749,056$. It is NOT reset to 0.
- **Update Rules**:
  - Model A: Ordinary LM updates.
  - Model B: Controlled LM updates.
  - Model C: `mode="lm"`, only $\theta_C$ trains; $\theta_N$ bypassed and frozen ($\text{changed } \theta_N \text{ tensors} = 0$).
  - Model D: `adapter_scale=0`, only backbone trains; safety parameters frozen ($\text{changed safety tensors} = 0$).

---

## 6. Comprehensive Evaluation Protocol

Evaluations are conducted at three major checkpoints:
1. **Clean 1B**: Immediately following 1B pretraining.
2. **Pre-Persistence (Post-Safety)**: Immediately following 20M safety training.
3. **Post-Persistence**: After 1,000 steps of persistence pretraining.

### Evaluation Dimensions
1. **FineWeb Capability**: Full 1,024 validation blocks (1,048,576 tokens) $\to$ Cross-Entropy (CE), Perplexity (PPL), Next-Token Accuracy.
2. **WildGuard Internal Safety**:
   - Risk Validation (2,344 examples): BCE Loss, Accuracy, Harmful Accuracy, Benign Accuracy, Balanced Accuracy, Confusion Matrix.
   - Generation Validation (928 examples): Continuation NLL, Token-Weighted CE, PPL.
3. **In-Distribution (ID) Behavioral Benchmark**:
   - 256 Harmful + 256 Benign prompts selected via deterministic salted hash ranking:
     $$\text{key} = \text{stable\_hash\_int}(\text{"task7\_3\_id\_behavior\_v1:"} + \text{example\_id}, 2^{63})$$
   - Metrics: Safe Refusal Rate, Unsafe Compliance Rate, Benign Non-Refusal Rate, Over-Refusal Rate.
   - Authoritative Wilson 95% Confidence Intervals for all rates.
4. **Out-of-Distribution (OOD) BeaverTails Benchmark**:
   - 256 Harmful + 256 Benign prompts from BeaverTails `30k_test` via deterministic seed `20260822`.
   - Same generation and judging protocols.
5. **Causal Mechanism Ablation**:
   - Scale 1.0 vs Scale 0.0 on Models B, C, and D across ID, OOD, and validation sets.

---

## 7. Cost Accounting & Infrastructure

- **Pricing Source**: `ccpt.training.cost` ($3.9492 USD/hr for H100!, $1.9512 USD/hr for L40S).
- **Progress Reporting**: `LiveProgressReporter` with mandatory JSONL output and Chicago/UTC timestamps.
- **Budget**: Total estimated spend ~$35–$60 USD. If projected spend exceeds $60 USD, execution will pause for review.
- **Resumability**: Checkpoints use strict `CHECKPOINT_FORMAT_VERSION_V2`.
