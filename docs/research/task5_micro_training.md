# Task 5 Research & Engineering Documentation: Micro-Overfit, Training Harness, and Failure Tests

## 1. Purpose and Scope

Task 5 is the first phase of the CCPT research project involving active model training. Its purpose is **strictly diagnostic and verification-oriented**: to prove that the complete training system works end-to-end on tiny deterministic data subsets before compute is spent in Task 6.

### Core Scientific Questions Answered in Task 5:
1. **Language Learning**: Can Models A, B, and C learn next-token causal language modeling using the implemented architecture and native PyTorch training loop?
2. **Safety Risk Classification**: Can the auxiliary risk heads (Model A) and normative pathways (Models B & C) overfit a balanced, deterministic safety risk subset to high accuracy ($\ge 95\%$)?
3. **Strict Parameter Isolation (Firewall)**: Does the CCPT parameter freeze invariant hold during real gradient descent? Are all capability parameters $\theta_C$ strictly bit-for-bit unchanged ($0$ modified tensors) while normative parameters $\theta_N$ train?
4. **Causal Steering of Frozen Capability**: Can the CCPT controller reduce safe-generation loss on a frozen capability pathway?
5. **Ablation Penalty**: Does disabling the controller ($\text{controller\_scale}=0.0$) remove steering influence, worsening safe-generation loss by $\ge 5\%$?
6. **Causal Equivalence Invariant**: Does $\text{mode}=\text{"controlled"}$ with $\text{controller\_scale}=0.0$ match $\text{mode}=\text{"lm"}$ within strict floating-point tolerance?
7. **Numerical & Checkpoint Health**: Are gradients, gates, steering vectors, losses, and checkpoints finite, non-saturating, and bit-reproducible across saves and resumes?

> **CRITICAL SCIENTIFIC NOTE**: Task 5 results represent micro-overfit diagnostic testing on tiny datasets (16–32 sequences) and **do not constitute empirical research claims** regarding the general safety, alignment, or superiority of CCPT.

---

## 2. Model Architecture & Diagnostic Micro Configurations

To consume real Mistral token IDs ($\text{vocab\_size}=32,000$) while preserving lightweight diagnostic scale, Task 5 defines a parameter-matched diagnostic scale ($\sim 2.24\text{M}$ parameters):

### 2.1 Dual-Stream Configuration (Models B and C)
- $\text{vocab\_size} = 32,000$
- $\text{max\_seq\_len} = 128$
- **Capability Stream ($C$)**: $n_{\text{layers}, C}=4$, $d_C=64$, $n_{\text{heads}, C}=4$, $d_{\text{ff}, C}=128$
- **Normative Stream ($N$)**: $n_{\text{layers}, N}=2$, $d_N=32$, $n_{\text{heads}, N}=4$, $d_{\text{ff}, N}=64$
- **Controlled Layers**: $[2, 4]$
- **Controller Bounds**: $\alpha=0.1$ (multiplicative gate $g_l \in [0.9, 1.1]$), $\beta=1.0$ (additive steering $\|s_l\|_\infty \le 1.0$)
- **Total Parameters**: Exactly **2,243,392** parameters.

### 2.2 Parameter-Matched Baseline (Model A)
- $\text{vocab\_size} = 32,000$
- $\text{max\_seq\_len} = 128$
- $n_{\text{layers}}=4$, $d_{\text{model}}=64$, $n_{\text{heads}}=4$, $d_{\text{ff}}=168$
- Tied embedding / unembedding, auxiliary risk head on final layer representation.
- **Total Parameters**: Exactly **2,243,200** parameters (Difference: $192$ parameters, $0.0086\%$).

### 2.3 Identical Model B and Model C Initialization
Before training begins, Model B and Model C are instantiated from the exact same initial parameter tensor dictionary under global seed `TASK5_SEED = 20260821`. Every initial tensor is bit-for-bit identical (`torch.equal(p_B, p_C) == True`).

---

## 3. Deterministic Micro Subsets & Data Lock

All training data is derived from the verified Task 4 production datasets stored on Modal Volume `ccpt-data` (`/data/ccpt`):
- **Task 4 Manifest Hash**: `2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097`
- **Zero Leakage**: Strict assertions confirm that $0$ examples in the training subsets originated from internal validation splits or `WildGuardTest`.

### 3.1 LM Micro Subset
- **Source**: `smoke_tokens.bin` (FineWeb smoke training shard)
- **Selection**: 16 contiguous slices of 128 tokens ($2,048$ tokens total).
- **Logical SHA256 Hash**: `26b54d87981de438fa0a7d60460456eb7288749a485ac38f3118a96731dcb1a3`

### 3.2 Risk Micro Subset
- **Source**: `risk/train.arrow`
- **Selection**: 16 harmful ($\text{label}=1$) and 16 benign ($\text{label}=0$) examples with sequence length $\le 128$, sorted deterministically by SHA256 key ($32$ examples total).
- **Logical SHA256 Hash**: `162dafe6760e6c4359430a2995c156e8b3e4fc7c728272690de608a14a51fd63`

### 3.3 Safe-Generation Micro Subset
- **Source**: `generation/train.arrow`
- **Selection**: 16 harmful refusals ($\text{risk\_label}=1$) and 16 benign compliances ($\text{risk\_label}=0$) with sequence length $\le 128$, sorted deterministically by SHA256 key ($32$ examples total).
- **Logical SHA256 Hash**: `b10d945e1c5e51542f5fe468d2575df13b4be3a26410560a09bb580fd507ee17`
- **Subset Manifest Hash**: `3480afd5769b483a5b269f0bc8c87188454974bf13f5a862e7906501f933960f` (verified identical across 2 remote passes).

---

## 4. Modal Infrastructure & Execution Settings

- **CPU Validation Function**: 4.0 CPU cores, 8 GiB RAM, 0 GPU. Executed remote pytest suite (76 tests passed), verified manifest lock, created deterministic subsets, and ran CPU deterministic resume test.
- **GPU Micro-Training Function**: 1 $\times$ NVIDIA A10G (24 GiB VRAM), 8.0 CPU cores, 16 GiB RAM.
- **Precision**: `torch.float32` (No mixed precision, bf16, or fp16 during Task 5 diagnostic runs).
- **Optimizer**: AdamW ($\text{lr}=10^{-3}$, $\beta=(0.9, 0.999)$, $\epsilon=10^{-8}$, $\text{weight\_decay}=0.0$, $\text{gradient\_clip\_norm}=1.0$).
- **Batching**: Cyclic deterministic batching (LM batch size 4, Risk batch size 8, Generation batch size 4).

---

## 5. Micro-Overfit Training Phases & Observed Results

### Phase 1: LM Micro-Overfit (200 Steps)
Each model was trained on the 16 FineWeb sequences ($2,048$ tokens).
- **Model A**: $\text{Loss}: 10.3903 \to 3.1283$ ($69.9\%$ reduction), $\text{Acc}: 0.2\% \to 50.3\%$
- **Model B**: $\text{Loss}: 10.3870 \to 3.2820$ ($68.4\%$ reduction), $\text{Acc}: 0.2\% \to 47.6\%$ (Joint gradients verified)
- **Model C (CCPT)**: $\text{Loss}: 10.3870 \to 3.3025$ ($68.2\%$ reduction), $\text{Acc}: 0.2\% \to 45.5\%$ ($\theta_C$ trained, $N$ bypassed)
- **Status**: **PASS** (All models achieved $\ge 30\%$ loss reduction and increased token accuracy).

### Phase 2: Risk Micro-Overfit (150 Steps)
Continued from Phase 1 checkpoints on 32 balanced risk examples.
- **Model A**: $\text{Loss}: 0.7088 \to 0.0033$, $\text{Accuracy}: 50.0\% \to 100.0\%$
- **Model B**: $\text{Loss}: 0.6921 \to 0.0246$, $\text{Accuracy}: 50.0\% \to 100.0\%$
- **Model C (CCPT)**: $\text{Loss}: 0.6927 \to 0.0292$, $\text{Accuracy}: 50.0\% \to 100.0\%$
- **CCPT Capability Freeze**: $0$ $\theta_C$ parameter tensors changed (`assert_parameters_equal` passed).
- **Status**: **PASS** (All models reached $\ge 95\%$ training accuracy and reduced BCE loss).

### Phase 3: Safe-Generation Steering (300 Steps)
Continued from Phase 2 checkpoints on 32 generation examples ($L_{\text{norm}} = L_{\text{risk}} + 1.0 \times L_{\text{safe\_gen}}$).
- **Model A**: Pre $\text{Loss}: 10.4159 \to 1.4020$ (Loss reduced)
- **Model B**: Pre $\text{Loss}: 12.4840 \to 1.0354$ (Loss reduced)
- **Model C (CCPT)**:
  - Controlled Safe-Gen Loss ($\text{scale}=1.0$): `7.7816` ($33.8\%$ reduction $\ge 20\%$ threshold)
  - Ablated Safe-Gen Loss ($\text{scale}=0.0$): `11.7472`
  - Relative Ablation Penalty: $\frac{11.7472 - 7.7816}{7.7816} = 51.0\%$ ($\ge 5\%$ threshold)
  - Causal LM Invariant ($\text{scale}=0.0 \equiv \text{mode}=\text{"lm"}$): `True`
  - $\theta_C$ Changed Tensors: **0** (100% frozen bit-for-bit)
  - $\theta_N$ Changed Tensors: **27** ($> 0$, normative weights updated)
  - **Steering L2 Metrics**: Maximum observed mean steering $L_2$ across logged steps was **$1.0870$**; final mean steering $L_2$ was **$1.0740$**.
- **Status**: **PASS** (All steering and ablation acceptance criteria met).

---

## 6. Diagnostic Health and Failure Flag Checks

| Check | Criterion | Observed Value | Status |
|---|---|---|---|
| Non-finite Gradients / Loss | No NaN or Inf | No NaN/Inf anywhere | **PASS** |
| Gate Collapse | Max fraction near bounds $> 95\%$ | Max near-bound fraction: $0.0\%$ (mean: $0.9976$, min: $0.9652$, max: $1.0390$) | **PASS** |
| Steering Saturation | Max fraction saturated $> 95\%$ | Max saturate fraction: $0.0\%$ (max observed mean $L_2$: $1.0870$) | **PASS** |
| Controller Gradient | Nonzero gradient during Phase 3 | Max controller grad: $0.9954 > 0$ | **PASS** |
| Deep $N$ Gradient | Nonzero gradient after controller movement | Max deep $N$ grad: $0.9998 > 0$ | **PASS** |
| CCPT Capability Freeze | $\theta_C$ unchanged during Phases 2 & 3 | 0 changed tensors (bit-for-bit identical) | **PASS** |
| Causal Invariant | $\text{scale}=0.0 \equiv \text{mode}=\text{"lm"}$ | `torch.allclose == True` | **PASS** |
| Checkpoint Reload & Resume | Bit-identical state and parameter reproduction | Bit-for-bit identical | **PASS** |
| Data Leakage | Zero eval IDs in training | 0 eval IDs accessed / WildGuardTest untouched | **PASS** |

---

## 7. Checkpoint Data Lineage

During Task 5, training and review artifacts were created in stages:
1. **Training-Time Subset Manifest**: Generated during initial dataset preparation (`task5_subset_hash: 3480afd5769b483a5b269f0bc8c87188454974bf13f5a862e7906501f933960f`). This hash was embedded into all 9 checkpoint dictionaries at saving time.
2. **Sanitized Review Manifest**: Generated during Task 5.1 cleanup (`sanitized_review_manifest_hash: 1b315015ee2e01c86da989192ea789526ec232b052a2349451611552f6935132`). To ensure no copyrighted or benchmark text is packaged, raw fields (such as token lists and records) were stripped, leaving only metadata, counts, slices, and hashes.
3. **Lineage Invariant**:
   - Sanitization changes the canonical serialization and SHA256 of the manifest document.
   - Therefore, `training_subset_manifest_hash` and `sanitized_review_manifest_hash` are intentionally distinct.
   - In Task 5.2, all 9 saved checkpoints on Modal Volume `ccpt-runs` were directly opened and inspected on Modal CPU without modification.
   - All 9 checkpoints unanimously contain `task4_manifest_hash: 2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097` and `task5_subset_hash: 3480afd5769b483a5b269f0bc8c87188454974bf13f5a862e7906501f933960f`.
   - Checkpoint contents were not modified or rewritten.

---

## 8. Conclusion

All acceptance criteria for Task 5 have been verified on Modal CPU and GPU infrastructure. The training harness, optimization isolation, gradient firewall, checkpointing, controller steering mechanisms, and checkpoint data lineage are fully functional, verified, and audited.

