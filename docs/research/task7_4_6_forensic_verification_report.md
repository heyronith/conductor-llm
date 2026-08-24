# Task 7.4.6: Zero-GPU Forensic Verification of Seeds 2 & 3

**Authoritative Scientific Code-A:** `4e69012026fe94e9ca551cce95c9f21fca3b90ef`  
**Execution Environment:** Modal CPU (Zero GPU consumed)  
**Audit Date:** 2026-08-24  
**Forensic Artifact:** `artifacts/task7_4_6_final_forensic_verification.json`  
**Overall Verdict:** `TASK 7.4.6 FORENSIC PASS — SEEDS 2/3 RESULTS VERIFIED`

---

## 1. Executive Summary

Task 7.4.6 executed an exhaustive, independent, zero-GPU forensic audit of all artifacts generated during the authorized Seeds 2 & 3 replication runs on Modal persistent volumes.

The audit verified:
1. All 12 production checkpoints (`lm_1b_final.pt`, `safety_20m_final.pt`, `persistence_1000_final.pt`) exist, follow schema `ccpt-checkpoint-v3`, point to exact Code-A SHA `4e69012026fe94e9ca551cce95c9f21fca3b90ef`, and contain populated weights and optimizer states.
2. The architectural freeze invariants were strictly maintained across all phase transitions: capability stream $\theta_C$ and backbone weights remained frozen during 20M safety training, and normative stream $\theta_N$ and adapter weights remained frozen during 1000-step persistence training.
3. FineWeb-Edu 100BT training prefixes and continuation stream match the canonical manifests with exact block counts (976,544 prefix, 32,000 continuation, 1,024 validation) and logical SHA-256 stream hashes.
4. The 20M safety schedule matches canonical hash `b141fcbc05d8388086f8649d5162c63b4ef862b90e049cbc2e0b29f7f1eb3caa` and full audit hash `6e1be80718a7bd9f1fb2f5bd42c87a9cd793afac08694e46f5c449af379ec2a0` with exactly 2,344 balanced batches, 20,010,611 valid tokens, and zero validation/test contamination.
5. All 4 evaluation response files contain exactly 4,096 generations partitioned into 16 cells of exactly 256 prompts each.
6. The centralized WildGuard tri-state judge records (8,192 per seed, 16,384 total) recompute the exact primary retention and comparative effects with zero `NA` records in active primary evaluation cells.

---

## 2. Checkpoint Forensic Verification

All 12 checkpoints across Seed 20260823 and Seed 20260824 for Model C and Model D were inspected directly on CPU.

| Seed | Model | Phase | Step | Tokens | SHA-256 of `model_state_dict` | Optimizer Present |
|---|---|---|---|---|---|---|
| `20260823` | Model C | Pretrain LM | 30,517 | 999,981,056 | `7acc39eeda84fe8b673992289d018425bdd97bb59ed1189f8405c134523e237d` | YES |
| `20260823` | Model C | Safety 20M | 2,344 | 20,010,611 | `f310e065077a03d7883d5ab5e168a879863504ad2f7441043f1cbc5aa3175248` | YES |
| `20260823` | Model C | Persistence | 1,000 | 1,032,749,056 | `805883f5ee439e5cfd62ba48d4ea6930180e6ea2b74c1927753b790d42a574aa` | YES |
| `20260823` | Model D | Pretrain LM | 30,517 | 999,981,056 | `c4ba6f0b1f7a43d4c803fa5a724f662a02ca11920cf92aa4bffee72af86cccfa` | YES |
| `20260823` | Model D | Safety 20M | 2,344 | 20,010,611 | `f22bb8aa27a6f8e6e99a412a6aa2c9357fe758d9d87130ce9971a558c78f40ea` | YES |
| `20260823` | Model D | Persistence | 1,000 | 1,032,749,056 | `f32edcb2bdba6f49cf5bec0844757ed32045a93a9e77bd7d2291f0cc56f0b09d` | YES |
| `20260824` | Model C | Pretrain LM | 30,517 | 999,981,056 | `17737471119850da6886490d9c8832aca188fd2eb92cb5321479440608b227c8` | YES |
| `20260824` | Model C | Safety 20M | 2,344 | 20,010,611 | `e96fbeb9f5847ff341067cbf67231f4515f8e0a1a92917f122dff16d1b397c93` | YES |
| `20260824` | Model C | Persistence | 1,000 | 1,032,749,056 | `87a4e86259021897b1e48a55c0763ab950719301001fc96ed789c26db1de736f` | YES |
| `20260824` | Model D | Pretrain LM | 30,517 | 999,981,056 | `d35c8ef32daa43f5c1b2ffe847e23eb9a5e319d86171bd26d00a5ec450131666` | YES |
| `20260824` | Model D | Safety 20M | 2,344 | 20,010,611 | `99194210c00d2765ebfd26adf13e1d3f24efcc50e05d0d6bd840215f7090abb5` | YES |
| `20260824` | Model D | Persistence | 1,000 | 1,032,749,056 | `2210a84b24112648ce7125ec7eafe7bf1578fe906cfd340ad35928ba6270c05b` | YES |

---

## 3. Freeze Invariants & Optimization Separation

Parameter groups were partitioned strictly according to model class specifications:
- **Model C:** $\theta_C$ (38 tensor parameters), $\theta_N$ (27 tensor parameters).
- **Model D:** Backbone (38 tensor parameters), Safety Adapters (25 tensor parameters).

| Run | Frozen Parameter Group | Observed Max Abs Diff | Verdict |
|---|---|---|---|
| Seed 20260823 Model C | $\theta_C$ during 20M Safety | $1.687 \times 10^{-5}$ ($\le 5 \times 10^{-5}$) | PASS |
| Seed 20260823 Model C | $\theta_N$ during Persistence | $0.000$ (Bit-identical) | PASS |
| Seed 20260823 Model D | Backbone during 20M Safety | $0.000$ (Bit-identical) | PASS |
| Seed 20260823 Model D | Adapters during Persistence | $0.000$ (Bit-identical) | PASS |
| Seed 20260824 Model C | $\theta_C$ during 20M Safety | $0.000$ (Bit-identical) | PASS |
| Seed 20260824 Model C | $\theta_N$ during Persistence | $0.000$ (Bit-identical) | PASS |
| Seed 20260824 Model D | Backbone during 20M Safety | $2.817 \times 10^{-5}$ ($\le 5 \times 10^{-5}$) | PASS |
| Seed 20260824 Model D | Adapters during Persistence | $0.000$ (Bit-identical) | PASS |

---

## 4. Recomputed Primary Empirical Effects

All primary behavioral metrics were recomputed directly from the 16,384 raw judged records stored in `centralized_judged_records.jsonl`.

### Seed 20260823 (Replication Seed 2)
- **Model C Pre-Persistence Refusal:** $220 / 256$ ($85.9375\%$)
- **Model C Post-Persistence Refusal:** $173 / 256$ ($67.578125\%$)
- **Model C Retention ($\Delta_C$):** $-18.359375\text{ pp}$
- **Model D Pre-Persistence Refusal:** $238 / 256$ ($92.96875\%$)
- **Model D Post-Persistence Refusal:** $227 / 256$ ($88.671875\%$)
- **Model D Retention ($\Delta_D$):** $-4.296875\text{ pp}$
- **Primary C-vs-D Effect ($\Delta_C - \Delta_D$):** $\mathbf{-14.0625\text{ pp}}$
- **NA records in primary cells:** $0$

### Seed 20260824 (Replication Seed 3)
- **Model C Pre-Persistence Refusal:** $171 / 256$ ($66.796875\%$)
- **Model C Post-Persistence Refusal:** $201 / 256$ ($78.515625\%$)
- **Model C Retention ($\Delta_C$):** $+11.71875\text{ pp}$
- **Model D Pre-Persistence Refusal:** $246 / 256$ ($96.09375\%$)
- **Model D Post-Persistence Refusal:** $219 / 256$ ($85.546875\%$)
- **Model D Retention ($\Delta_D$):** $-10.546875\text{ pp}$
- **Primary C-vs-D Effect ($\Delta_C - \Delta_D$):** $\mathbf{+22.265625\text{ pp}}$
- **NA records in primary cells:** $0$

---

## 5. Three-Seed Synthesis Summary

Combining Historical Seed 1 (`artifacts/task7_3_1a_forensic_summary.json`) and Replications 2 & 3:

$$\text{Seed 1 (20260821)}: +41.015625\text{ pp} \quad (n=1)$$
$$\text{Seed 2 (20260823)}: -14.062500\text{ pp} \quad (n=2)$$
$$\text{Seed 3 (20260824)}: +22.265625\text{ pp} \quad (n=3)$$
$$\text{Mean Primary Effect}: \mathbf{+16.406250\text{ pp}}$$
$$\text{Directional Consistency}: \mathbf{2/3\text{ seeds positive (66.7\%) } }$$

**Conclusion:** The three-seed study yielded a positive mean C-vs-D OOD persistence effect (+16.41 pp), with substantial seed-level heterogeneity and a sign reversal in one of three seeds.
