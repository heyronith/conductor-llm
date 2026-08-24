# TASK 7.4.5R — Three-Seed CCPT Replication Report (Model C vs Model D)

**Status:** COMPLETE & AUTHORITATIVE  
**Scientific Code-A SHA:** `4e69012026fe94e9ca551cce95c9f21fca3b90ef`  
**Prelaunch Evidence-B SHA:** `ba78fac4eea51ef2b66d62b2a7d8c8ab4c9bc697`  
**Scope:** CCPT Model C vs Matched Frozen-Adapter Model D across 3 Independent Initialization Seeds (`20260821`, `20260823`, `20260824`)  
**Incremental GPU Spend:** **$9.42** (Hard spending ceiling: $\le \$35.00$)  

---

## 1. Executive Summary & Three-Seed Synthesis

The three-seed study yielded a positive mean C-vs-D Out-of-Distribution (OOD) persistence effect, with substantial seed-level heterogeneity and a sign reversal in one of three seeds.

### Primary Replicated Estimator (OOD Safety Retention)
The pre-registered primary endpoint is the difference in safety retention on Out-of-Distribution harmful prompts (BeaverTails 30k OOD Harmful, $n=256$ prompts per evaluation cell) following unconstrained continuation pretraining (1,000 LM steps corresponding to 32,768,000 token presentations, or ~32.8M token presentations on FineWeb-Edu):

$$\text{Retention}_C = \text{Refusal}_{\text{post}} - \text{Refusal}_{\text{pre}}$$
$$\text{Retention}_D = \text{Refusal}_{\text{post}} - \text{Refusal}_{\text{pre}}$$
$$\text{PRIMARY\_EFFECT} = \text{Retention}_C - \text{Retention}_D$$

All rates use the exact tri-state WildGuard determinate refusal definition $\frac{\text{YES}}{\text{YES} + \text{NO}}$.

| Seed | Model C Pre | Model C Post | Model C Retention | Model D Pre | Model D Post | Model D Retention | Primary Effect ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20260821 (Seed 1)** | 87.500000% | 86.328125% | **-1.171875 pp** | 93.359375% | 51.171875% | **-42.187500 pp** | **+41.015625 pp** |
| **20260823 (Seed 2)** | 85.937500% | 67.578125% | **-18.359375 pp** | 92.968750% | 88.671875% | **-4.296875 pp** | **-14.062500 pp** |
| **20260824 (Seed 3)** | 66.796875% | 78.515625% | **+11.718750 pp** | 96.093750% | 85.546875% | **-10.546875 pp** | **+22.265625 pp** |
| **Mean ($n=3$)** | **80.078125%** | **77.473958%** | **-2.604167 pp** | **94.140625%** | **75.130208%** | **-19.010417 pp** | **+16.406250 pp** |
| **Sample SD** | 11.536766% | 9.429074% | 15.091054 pp | 1.699318% | 20.803732% | 20.314059 pp | **28.000720 pp** |
| **Min / Max** | 66.80% / 87.50% | 67.58% / 86.33% | -18.36 pp / +11.72 pp | 92.97% / 96.09% | 51.17% / 88.67% | -42.19 pp / -4.30 pp | **-14.06 pp / +41.02 pp** |
| **Direction Consistency**| — | — | — | — | — | — | **2 / 3 (66.7%)** |

*Note on Seed 1 Source:* Seed 1 values are sourced directly from the authoritative forensic tri-state summary ([`artifacts/task7_3_1a_forensic_summary.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task7_3_1a_forensic_summary.json)), maintaining mathematical parity with Seeds 2 & 3.

---

## 2. Scientific Synthesis & Nuanced Findings

### 1. Heterogeneity and Sign Inversion in Seed 2
The primary effect is positive in 2 of 3 seeds (+41.02 pp in Seed 1, +22.27 pp in Seed 3), but reverses in Seed 2 (-14.06 pp). In Seed 2, Model D suffered very little persistence degradation (-4.30 pp drop), whereas Model C exhibited a moderate drop (-18.36 pp). Consequently, the overall direction consistency is **2/3**, with a large sample standard deviation of **28.00 pp** across the $n=3$ independent replicates.

### 2. Initialization Sensitivity and Pre-Persistence Variance
There is substantial initialization sensitivity in Model C's pre-persistence OOD refusal rate:
- **Model C Pre-Persistence OOD Refusal:** 87.50% (Seed 1), 85.94% (Seed 2), and **66.80%** (Seed 3).
- **Model D Pre-Persistence OOD Refusal:** 93.36% (Seed 1), 92.97% (Seed 2), and 96.09% (Seed 3).

Model D's pre-persistence refusal is markedly more stable across seeds ($\text{SD} = 1.70\%$), whereas Model C shows greater optimization variance ($\text{SD} = 11.54\%$). Interestingly, in Seed 3, Model C's refusal actually increased during continuation pretraining (+11.72 pp), reflecting variance in how unconstrained FineWeb tokens interact with an under-converged pre-persistence state.

### 3. Causal Steering Dependence
Controller ablation produces a large reduction in refusal behavior, demonstrating strong causal dependence of the observed behavior on the normative control mechanism:
- **Seed 2 Model C Pre:** Active ($\alpha=1$): **85.94%** $\rightarrow$ Ablated ($\alpha=0$): **42.86%** ($\Delta = -43.08\text{ pp}$)
- **Seed 2 Model C Post:** Active ($\alpha=1$): **67.58%** $\rightarrow$ Ablated ($\alpha=0$): **44.03%** ($\Delta = -23.55\text{ pp}$)
- **Seed 3 Model C Pre:** Active ($\alpha=1$): **66.80%** $\rightarrow$ Ablated ($\alpha=0$): **54.33%** ($\Delta = -12.47\text{ pp}$)
- **Seed 3 Model C Post:** Active ($\alpha=1$): **78.52%** $\rightarrow$ Ablated ($\alpha=0$): **32.91%** ($\Delta = -45.61\text{ pp}$)

When the normative stream is ablated, refusal drops to the baseline base-LM range (~33–54%), confirming that active safety interventions depend causally on the normative pathway.

---

## 3. In-Distribution (ID) Safety Metrics

Evaluated on 256 ID WildGuard Harmful test prompts per model/phase:

| Seed | Model C Pre | Model C Post | Model C ID $\Delta$ | Model D Pre | Model D Post | Model D ID $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 1 (20260821)** | 98.83% [96.61%, 99.60%] | 96.09% [92.96%, 97.86%] | -2.73 pp | 100.00% [98.52%, 100.0%] | 96.48% [93.45%, 98.14%] | -3.52 pp |
| **Seed 2 (20260823)** | 98.44% [96.05%, 99.39%] | 93.75% [90.09%, 96.12%] | -4.69 pp | 99.61% [97.82%, 99.93%] | 97.66% [94.98%, 98.92%] | -1.95 pp |
| **Seed 3 (20260824)** | 98.83% [96.61%, 99.60%] | 98.44% [96.05%, 99.39%] | -0.39 pp | 99.61% [97.82%, 99.93%] | 98.83% [96.61%, 99.60%] | -0.78 pp |
| **Mean ($n=3$)** | **98.70%** | **96.09%** | **-2.60 pp** | **99.74%** | **97.66%** | **-2.08 pp** |

Both architectures maintain high in-distribution safety (>93% in all cells).

---

## 4. Spend & Resource Accounting

| Workload | Hardware | Runtime / Units | Rate | Total Cost (USD) |
| :--- | :--- | :--- | :--- | :--- |
| **Historical & Interrupted Early Steps** | H100 SXM5 | Partial step batches | $4.49/hr | $1.50 |
| **Phase 1: Seed 2 Training (C & D)** | H100 SXM5 | 2 pipelines (LM + Safety + Persistence) | $4.49/hr | $3.12 |
| **Phase 1: Seed 3 Training (C & D)** | H100 SXM5 | 2 pipelines (LM + Safety + Persistence) | $4.49/hr | $3.12 |
| **Phase 2: L40S Evaluation Workers** | L40S | 4 workers $\times$ 4,096 resp (16,384 total) | $1.95/hr | $1.41 |
| **Phase 3: Centralized WildGuard Judge**| L40S | 2 workers $\times$ 8,192 records (16,384 total)| $1.95/hr | $0.27 |
| **TOTAL GPU SPEND** | — | — | — | **$9.42** |
| **Budget Ceiling** | — | — | — | **$35.00** |
| **Remaining Margin** | — | — | — | **+$25.58** |

---

## 5. Explicit Scientific Limitations & Open Questions

1. **Small Sample Size ($n=3$):** With three independent initialization seeds, sample variance estimates are wide.
2. **Direction Consistency (2/3):** The primary persistence advantage is observed in 2 of 3 seeds; Seed 2 displays a negative effect due to high Model D stability.
3. **Large Effect Variance:** The primary effect range spans -14.06 pp to +41.02 pp ($\text{SD} = 28.00\text{ pp}$).
4. **Historical Provenance of Seed 1:** Seed 1 was completed prior to the Task 7.4 multi-seed orchestration harness, though its forensic artifacts have been rigorously validated.
5. **Benign Over-Refusal:** Model C exhibits substantial over-refusal on benign prompts (~70–80% in both ID and OOD), indicating that normative threshold tuning remains an active engineering challenge.
6. **Scale Generalization:** These observations apply to the 1B-scale micro-architecture; they do not establish how dual-stream dynamics scale to frontier LLM regimes.
7. **Readiness for 10B:** `READY_FOR_10B = false`. No 10B training is claimed or authorized.

---

## 6. Authoritative Artifact References

- Consolidated Summary: [`artifacts/task7_4_multiseed_replication_summary.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task7_4_multiseed_replication_summary.json)
- Forensic Reference (Seed 1): [`artifacts/task7_3_1a_forensic_summary.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task7_3_1a_forensic_summary.json)
- Launch Manifest: [`artifacts/task7_4_launch_manifest.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task7_4_launch_manifest.json)
- Research Report: [`docs/research/task7_4_5r_multiseed_replication_report.md`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/docs/research/task7_4_5r_multiseed_replication_report.md)
