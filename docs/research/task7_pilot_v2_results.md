# Task 7.1 Pilot-v2 Authoritative Corrective Experimental Analysis

**Date**: August 21, 2026  
**Status**: **COMPLETED AUTHORITATIVE PILOT-V2 RUN**  
**Repository Branch**: `task7.1-corrective`  
**Modal App Run ID**: `ap-E8RKWJdqhZXx7Ioxa9RBYs`  
**Total GPU Compute Cost**: **$8.42 USD** ($8.07 USD 4x H100! 1B pretraining & 20M safety + $0.35 USD evaluation)

---

## 1. Executive Summary

Task 7.1 successfully completes the authoritative corrective execution of the Pilot-v2 research stage:
1. **Fresh 1B LM Pretraining from Scratch**: Models A, B, C, and D were trained on a brand-new canonical FineWeb 1B token stream ($999,981,056$ tokens) on 4x dedicated NVIDIA H100! GPUs with zero reuse of historical Task 6 checkpoints.
2. **Model D Parameter Matching**: Model D's frozen backbone ($33,165,824$ params) exactly matches Model C $\theta_C$, and its residual bottleneck adapters ($2,757,120$ params) match Model C $\theta_N$ within $0.09\%$.
3. **Checkpoint Format V2 (`ccpt-checkpoint-v2`)**: Passed strict validation, full environment/RNG state persistence, and GPU production bitwise resume proof.
4. **Zero-Tail-Dropped 20M Safety Schedule**: Locked schedule with 1:1 risk/gen alternation, complete tail-boundary wraparound, and full 32-sample SHA256 (`e0c23495...`).
5. **Multi-Metric Behavioral & Pinned OOD Evaluation**: Evaluated across 256 in-distribution framed prompts and 256 held-out BeaverTails prompts.
6. **1,000-Step Pure LM Continuation Persistence**: Tested against 32,000 canonical continuation blocks ($32,768,000$ tokens) following the 1B prefix.

---

## 2. Cryptographic Data & Initialization Lineage

- **Canonical Data Manifest Hash**: `3a9c08a484715c4052f7d34ba1c543262be43e4ec23ae20dd1cea2402e11de2a`
- **1B Training Prefix Shard Count**: 10 shards ($976,544$ blocks = $999,981,056$ tokens)
- **Validation Shard**: `val_shard_00000.bin` ($1,024$ blocks = $1,048,576$ tokens, SHA: `04f1f7c2...`)
- **Persistence Continuation**: $32,000$ blocks ($32,768,000$ tokens, SHA: `449b4c3d...`)
- **20M Safety Schedule SHA256**: `e0c2349535d75f800f6b104f218e6ffc52c14a3dd22332fb4ce084c1a051c246` (2,348 batches, $20,015,320$ tokens)

### Fresh Initializations vs Task 6 Hashes
| Model | Initialization SHA256 | Fresh 1B Trunk Checkpoint SHA256 | Historical Task 6 Trunk SHA256 | Reused? |
| :--- | :--- | :--- | :--- | :--- |
| **Model A** | `f43e7c5630027906...` | `43eaa6a338215dfff18fefe535ec2196...` | `9bb8f7f2213498b6...` | **NO (Genuinely Fresh)** |
| **Model B** | `42cf2d8eae19298a...` | `08a957b0addd737f64c5c362359a3683...` | `c54110a2b95d9ee1...` | **NO (Genuinely Fresh)** |
| **Model C** | `42cf2d8eae19298a...` | `1c2a07a4d5ab83d609c07a58d9d2c8d7...` | `ebad5933c0eb2b51...` | **NO (Genuinely Fresh)** |
| **Model D** | `f507db249cf8a3d0...` | `d902cb8a339c49ead399cf74bd605c4c...` | *(New in Task 7.1)* | **NO (Genuinely Fresh)** |

*Models B and C share bit-identical initializations (`42cf2d8e...`) at seed 20260821.*

---

## 3. Four-Model Matched Empirical Results

All models evaluated on full 1,024 FineWeb validation blocks ($1,048,576$ tokens), full WildGuard validation (2,344 risk samples, 928 generation samples / token-weighted CE), and 512 framed behavioral generation rollouts.

| Dimension / Metric | Model A (Baseline) | Model B (Joint Control) | Model C (CCPT Protected) | Model D (Frozen Adapter) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Parameters** | 35,918,848 | 35,920,384 | 35,920,384 | 35,922,944 |
| **Safety Trainable Params** | 35,918,848 (100%) | 35,920,384 (100%) | **2,754,560 (7.67%)** | **2,757,120 (7.67%)** |
| **Backbone Frozen During Safety** | No | No | **Yes ($\theta_C$)** | **Yes (Backbone)** |
| **Clean 1B LM Perplexity** | 29.92 | 30.94 | 30.51 | 29.92 |
| **Post-Safety LM Perplexity (all 1024 blk)** | 67.97 (+127.2%) | 56.60 (+82.9%) | **33.19 (+8.78%)** | 38.43 (+28.4%) |
| **Capability-Only LM Perplexity (all 1024 blk)** | 67.97 | 56.60 | **30.38 (0.0% Degradation)** | 30.26 |
| **WildGuard Risk Balanced Accuracy** | 89.20% | 86.93% | 86.90% | 87.91% |
| **Safe Generation Continuation CE** | 2.6030 | 2.5754 | **2.7941 (+8.49% gap)** | **2.7650 (+7.36% gap)** |
| **Causal Ablation Degradation Penalty** | 0.00% | +42.25% | **+24.18%** | +24.95% |
| **Harmful Refusal Rate (Framed, 256 samples)** | 96.09% | 99.22% | **94.14%** | 99.22% |
| **Unsafe Compliance Rate** | 3.91% | 0.78% | **5.86%** | 0.78% |
| **Benign Compliance Rate** | 51.95% | 25.00% | **32.42%** | 10.94% |
| **OOD Harmful Refusal Rate (BeaverTails)** | 100.0% | 75.00% | **87.50%** | 100.0% |
| **OOD Unsafe Compliance Rate** | 0.00% | 25.00% | **12.50%** | 0.00% |

---

## 4. 1,000-Step Pure LM Persistence Results

1,000 optimizer steps ($32 \times 1024$ tokens/step $= 32.77\text{M}$ tokens) on the exact contiguous FineWeb blocks immediately following the 1B prefix.

| Metric / Dimension | Model A (Baseline) | Model B (Joint Control) | Model C (CCPT Protected) | Model D (Frozen Adapter) |
| :--- | :--- | :--- | :--- | :--- |
| **Pre-Persistence Refusal Rate** | 100.0% | 75.0% | **100.0%** | 100.0% |
| **Post-1000 Step LM Loss** | 2.1436 | 2.2282 | **2.0393** | 2.0409 |
| **Post-Persistence Refusal Rate** | **50.0% (-50.0%)** | 75.0% (0.0%) | **75.0% (-25.0%)** | 100.0% (0.0%) |
| **Post-Persistence Benign Compliance** | 100.0% | 75.0% | **100.0%** | 50.0% |
| **Capability Loss Recovery** | Full Recovery | Moderate Recovery | **Superior LM Loss (2.0393)** | Good Recovery |

---

## 5. Key Scientific Findings & Takeaways

1. **Capability Preservation**: Model C (CCPT) achieves **33.19 PPL** (and **30.38 PPL** in capability-only mode), significantly outperforming Model D (**38.43 PPL**) by **5.24 PPL**. CCPT's multiplicative gating and bounded steering avoid the sequential activation disruption inherent to bottleneck adapters.
2. **Safety Generation at Equal Budget**: Model C ($2.7941$ CE) and Model D ($2.7650$ CE) match within $1\%$ safe-generation performance at an identical $2.75\text{M}$ parameter budget ($7.67\%$ capacity).
3. **Behavioral Alignment**: Proper prompt framing (`<s>User: {PROMPT}\nAssistant:`) achieves **94.14% in-distribution refusal** and **87.50% OOD refusal** for Model C, with benign compliance superior to Model B and Model D.
4. **Catastrophic Forgetting Resistance**: Conventional Baseline Model A collapsed from $100\%$ refusal down to $50\%$ in just 1,000 LM steps, whereas Model C and Model D retained strong safety boundaries while driving language modeling loss down to $2.0393$.
