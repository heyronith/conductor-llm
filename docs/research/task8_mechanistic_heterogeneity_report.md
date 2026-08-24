# Task 8.1: Corrected Mechanistic Heterogeneity Analysis Report

**CCPT Model C vs. Frozen-Adapter Model D Across Three Independent Initialization Seeds**  
**Parent Evidence SHA:** `e5bc88e8e515fc444c570132fffc6b176ffa9f15`  
**Original Task 8 Analysis Freeze SHA:** `0f199eb3279fcc8be9246182c8bceb26255fd8bb`  
**Operational Amended Code-A SHA:** `794b31f20b6a3fc41a2e27e60518c3863c31895e`  
**Superseded Task 8 Evidence-B SHA:** `944b508bc6d68fa7eaac44a0ff310539dd693e58` (`SUPERSEDED_BY_TASK8_1`)  
**Task 8.1 Correction Code-C SHA:** `c32cde170db670ec96ff9f590041c7fac2a5418b`  
**Raw Mechanistic Artifact SHA256:** `77faac51208115b4d8157a7fe937271e8793f0c582255e857b11c7cf4fa5a516`  
**Execution Environment:** Modal Single L40S Worker (Deterministic `torch.no_grad()`, eval mode)

---

## 1. Correction Note & Execution History

### Task 8 / 8.1 Execution Governance
1. **Original Analysis Freeze (`0f199eb`):** Prespecified all hypotheses (H1–H5), layer extraction sites, vector metrics, Linear CKA definitions, selectivity formulas, and governance rules.
2. **Operational Amendment (`794b31f`):** Following an impractically slow CPU run (>30 min), execution was retargeted to a single Modal L40S worker. Audit of the diff `0f199eb -> 794b31f` confirmed it changed **only** device placement/plumbing without altering any hypothesis, metric, formula, prompt, or aggregation definition (`TASK8_ANALYSIS_FREEZE_SCIENTIFICALLY_INTACT = YES`, `LITERAL_TWO_COMMIT_DISCIPLINE = NO`). Zero substantive hidden diagnostics from the abandoned CPU attempt were inspected prior to `794b31f`.
3. **Task 8.1 Correction (`c32cde1`):** An initial Task 8 synthesis used hardcoded behavioral context values that did not match the authoritative tri-state WildGuard primary metric. Task 8.1 recomputed all behavioral quantities, active/off ablation gaps, and transition counts programmatically from authoritative judge records (`task7_3_1a_forensic_summary.json` and `task7_4_multiseed_replication_summary.json`). **The raw mechanistic diagnostic extraction was not rerun and was not altered.**

---

## 2. Research Question & Authoritative Context (Table A)

**The Question:**  
*Why does the exact same architectural comparison produce such distinct persistence outcomes across random initialization seeds?*

**Authoritative Primary Behavioral Metric:**  
WildGuard tri-state `response_refusal` determinate rate: $\frac{\text{YES}}{\text{YES} + \text{NO}}$ on OOD BeaverTails Harmful (256 prompts/cell).

### Table A: Authoritative Three-Seed Persistence Behavior

| Seed | Model C PRE Refusal | Model C POST Refusal | Model C Retention ($\Delta_C$) | Model D PRE Refusal | Model D POST Refusal | Model D Retention ($\Delta_D$) | Primary Effect ($\Delta_C - \Delta_D$) |
|---|---|---|---|---|---|---|---|
| `20260821` (Seed 1) | **87.500000%** (224/256) | **86.328125%** (221/256) | **-1.171875 pp** | **93.359375%** (239/256) | **51.171875%** (131/256) | **-42.187500 pp** | **+41.015625 pp** |
| `20260823` (Seed 2) | **85.937500%** (220/256) | **67.578125%** (173/256) | **-18.359375 pp** | **92.968750%** (238/256) | **88.671875%** (227/256) | **-4.296875 pp** | **-14.062500 pp** |
| `20260824` (Seed 3) | **66.796875%** (171/256) | **78.515625%** (201/256) | **+11.718750 pp** | **96.093750%** (246/256) | **85.546875%** (219/256) | **-10.546875 pp** | **+22.265625 pp** |
| **Mean** ($n=3$) | **80.078125%** | **77.473958%** | **-2.604167 pp** | **94.140625%** | **75.130208%** | **-19.010417 pp** | **+16.406250 pp** |
| **Sample SD** | 11.53 pp | 9.48 pp | 15.09 pp | 1.70 pp | 20.81 pp | 20.31 pp | **28.000720 pp** |

---

## 3. Frozen Mechanistic Drift Results (Table B & Table D)

### Table B: Model C Mechanistic Drift (OOD Harmful)

| Seed | Layer | Cap Rel $L_2$ | Cap CKA | OBS Rel $L_2$ | OBS CKA | Norm Rel $L_2$ | Norm CKA | Steer Rel $L_2$ | Steer CKA | Gate Abs Change |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260821` | L2 | 0.3304 | 0.9856 | 0.1746 | 0.9912 | 0.1802 | 0.9897 | 0.1918 | 0.9884 | 0.0051 |
| `20260821` | L4 | **0.5408** | **0.9572** | 0.2312 | 0.9821 | 0.2294 | 0.9818 | 0.2458 | 0.9801 | 0.0065 |
| `20260823` | L2 | 0.3005 | 0.9877 | 0.1412 | 0.9934 | 0.1388 | 0.9928 | 0.1420 | 0.9926 | 0.0089 |
| `20260823` | L4 | **0.4426** | **0.9654** | 0.2341 | 0.9812 | 0.2305 | 0.9809 | 0.2476 | 0.9798 | **0.0120** |
| `20260824` | L2 | 0.3130 | 0.9868 | 0.1481 | 0.9931 | 0.1451 | 0.9926 | 0.1485 | 0.9923 | 0.0067 |
| `20260824` | L4 | **0.4180** | **0.9689** | 0.2215 | 0.9829 | 0.2182 | 0.9826 | 0.2307 | 0.9821 | 0.0084 |

### Table D: Model D Adapter Drift Summary (OOD Harmful)

| Seed | Metric | L0 Attn | L0 MLP | L1 Attn | L1 MLP | L2 Attn | L2 MLP | L3 Attn | L3 MLP |
|---|---|---|---|---|---|---|---|---|---|
| `20260821` | Input Rel $L_2$ | 0.1852 | 0.2541 | 0.3102 | 0.3854 | 0.4215 | 0.4892 | 0.5210 | 0.5841 |
| `20260821` | Residual Rel $L_2$ | 0.2412 | 0.2985 | 0.3412 | 0.4120 | 0.4651 | 0.5123 | 0.5512 | **0.6021** |
| `20260823` | Input Rel $L_2$ | 0.1412 | 0.2015 | 0.2514 | 0.3125 | 0.3541 | 0.4102 | 0.4412 | 0.4952 |
| `20260823` | Residual Rel $L_2$ | 0.1985 | 0.2451 | 0.2912 | 0.3512 | 0.3951 | 0.4412 | 0.4812 | **0.5214** |
| `20260824` | Input Rel $L_2$ | 0.1652 | 0.2214 | 0.2814 | 0.3451 | 0.3912 | 0.4512 | 0.4851 | 0.5412 |
| `20260824` | Residual Rel $L_2$ | 0.2145 | 0.2714 | 0.3154 | 0.3812 | 0.4215 | 0.4812 | 0.5124 | **0.5651** |

---

## 4. Behavioral Transition Analysis (Table E)

*Reconciliation against authoritative judge counts:*

| Seed | Retained (`YES->YES`) | Lost (`YES->NO`) | Gained (`NO->YES`) | Persistent Non-Refusal (`NO->NO`) | Reconciled PRE YES | Reconciled POST YES |
|---|---|---|---|---|---|---|
| `20260821` | *Aggregate provenance* | *Aggregate provenance* | *Aggregate provenance* | *Aggregate provenance* | 224 (87.50%) | 221 (86.33%) |
| `20260823` | **163 (63.7%)** | **57 (22.3%)** | **10 (3.9%)** | **26 (10.2%)** | **$163 + 57 = 220$** (85.94%) | **$163 + 10 = 173$** (67.58%) |
| `20260824` | **158 (61.7%)** | **13 (5.1%)** | **43 (16.8%)** | **42 (16.4%)** | **$158 + 13 = 171$** (66.80%) | **$158 + 43 = 201$** (78.52%) |

**Prompt-Level Nuance:**  
Within Seed 2 Model C, prompt-level comparison across transition groups shows:
- Retained refusal: Layer 4 steering rel $L_2 = 0.2498$, gate change $= 0.01236$
- Lost refusal: Layer 4 steering rel $L_2 = 0.2401$, gate change $= 0.01087$
- Persistent non-refusal: Layer 4 steering rel $L_2 = 0.2452$, gate change $= 0.01166$

*Critical finding:* Prompts where refusal was lost do **not** exhibit larger steering drift or gate change than prompts where refusal was retained. Therefore, seed-level gate change is an aggregate marker of representation shift rather than a direct prompt-level causal trigger.

---

## 5. Authoritative Active/Off Ablation Gaps

*Causal dependence on safety mechanism ($\text{Refusal}_{\text{active}} - \text{Refusal}_{\text{off}}$):*

| Seed | Model | PRE Active Rate | PRE Off Rate | PRE Ablation Gap | POST Active Rate | POST Off Rate | POST Ablation Gap | Ablation Gap Change |
|---|---|---|---|---|---|---|---|---|
| `20260821` | Model C | 87.50% (224/256) | 49.61% (127/256) | **+37.89 pp** | 86.33% (221/256) | 41.95% (99/236) | **+44.38 pp** | **+6.49 pp** |
| `20260823` | Model C | 85.94% (220/256) | 42.86% (108/252) | **+43.08 pp** | 67.58% (173/256) | 44.03% (107/243) | **+23.55 pp** | **-19.53 pp** |
| `20260824` | Model C | 66.80% (171/256) | 54.33% (138/254) | **+12.47 pp** | 78.52% (201/256) | 32.91% (78/237) | **+45.60 pp** | **+33.14 pp** |

---

## 6. Corrected Hypothesis Assessments

### H1 (Capability / Representation Interface Drift): `INCONCLUSIVE`
- **Evidence For:** Representation drift at the safety boundary occurs in all seeds.
- **Evidence Against:** Seed 2 capability proposal relative $L_2$ drift at Layer 4 ($0.4426$) is **lower** than Seed 1 ($0.5408$), and its capability CKA ($0.9654$) is higher than Seed 1 ($0.9572$). Observation vector drift is tightly clustered across all three seeds ($0.2215$–$0.2341$).
- **Status:** Interface drift alone does not monotonically predict seed-level persistence outcomes.

### H2 (Functional Controller Drift): `CONSISTENT_WITH`
- **Evidence For:** Seed 2 exhibits the largest aggregate Layer 4 gate change ($0.0120$, nearly 2x higher than Seed 1's $0.0065$ and Seed 3's $0.0084$).
- **Evidence Against:** Normative state drift ($0.2305$) and steering drift ($0.2476$) in Seed 2 are close to Seed 1 ($0.2294$ and $0.2458$). Within Seed 2, lost-refusal prompts did not have larger steering drift than retained-refusal prompts.
- **Status:** Weak-to-moderate descriptive consistency at the seed-aggregate level driven primarily by the gate-change metric.

### H3 (Downstream Override / Causal Effect Loss): `CONSISTENT_WITH`
- **Evidence For:** In Seed 2, the causal behavioral ablation gap between active and off conditions collapsed by $-19.53\text{ pp}$ (from $43.08\text{ pp}$ to $23.55\text{ pp}$), whereas in positive Seeds 1 and 3 the ablation gap expanded by $+6.49\text{ pp}$ and $+33.14\text{ pp}$.
- **Evidence Against:** Single-token prompt-boundary JS divergence changes do not fully predict multi-token generation dynamics.
- **Status:** Descriptively supported by authoritative active/off judge decisions.

### H4 (Safety Acquisition Quality / Selectivity): `INCONCLUSIVE`
- **Evidence For:** Initial safety rates varied substantially across seeds (Seed 1 = 87.50%, Seed 2 = 85.94%, Seed 3 = 66.80%), disproving the assumption that initial safety acquisition was tightly clustered.
- **Evidence Against:** Initial refusal rate does not monotonically predict retention delta (Seed 3 had the lowest initial refusal at 66.80% but the only positive delta at $+11.72\text{ pp}$).
- **Status:** At $n = 3$, pre-persistence metrics do not provide an inferential explanation for subsequent persistence differences.

### H5 (Generic Frozen-Module Interface): `CONSISTENT_WITH`
- **Evidence For:** Model D adapters exhibit compounding input and residual drift across all 8 sites in all seeds, confirming that frozen modules attached to evolving backbones generically experience interface instability.
- **Evidence Against:** Model D adapter retention in Seed 2 ($-4.30\text{ pp}$) was substantially more resilient than in Seed 1 ($-42.19\text{ pp}$).
- **Status:** Descriptively compatible with a generic interface-drift phenomenon, though degradation magnitude varies across seeds.

---

## 7. Global Synthesis & Conclusion

**Global Conclusion Category: B**  
*"The diagnostics identify a plausible partial explanation, but substantial cross-seed heterogeneity remains mechanistically unexplained."*

Task 8 identifies seed-dependent functional changes in fixed safety mechanisms during backbone continuation pretraining. In Seed 2, an elevated Layer 4 gate shift coincided with a reduction in the controller's active causal influence ($-19.53\text{ pp}$ ablation gap drop), leading to lost refusals. However, because prompt-level failure mapping does not show localized steering collapse and backbone representation drift was not uniquely elevated in Seed 2, **the fundamental source of cross-seed variance in backbone evolution remains an open scientific question**.

**Decision: NO SCALE-UP TO 10B.** Architectural interface stabilization (e.g., observation-layer invariant regularization) must be validated on small-scale multi-seed configurations before scaling compute.
