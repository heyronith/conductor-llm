# Task 8.2: Machine-Derived Mechanistic Heterogeneity Analysis Report

**CCPT Model C vs. Frozen-Adapter Model D Across Three Independent Initialization Seeds**  
**Parent Evidence SHA:** `92b94420ab9545b9f55b287a1dd6d752b010050a`  
**Task 8.2 Code-E SHA:** `bac5e73b26e0f3dd459780961447b298bed05b20`  
**Raw Mechanistic Diagnostic SHA256:** `77faac51208115b4d8157a7fe937271e8793f0c582255e857b11c7cf4fa5a516`  
**Authoritative Machine Tables:** [`artifacts/task8_2_machine_tables.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task8_2_machine_tables.json)  
**Execution Environment:** Modal Single L40S Worker (Deterministic `torch.no_grad()`, eval mode)

---

## 1. Governance & Provenance

### Behavioral Join Provenance
- **SEED1_BEHAVIOR_JOIN = PARTIAL** (Authoritative aggregate tri-state judge counts available and verified from `task7_3_1a_forensic_summary.json`; prompt-level transition joins unavailable in Task 8 extraction).
- **SEED2_BEHAVIOR_JOIN = FULL** (All 256 prompt-level pre/post judge decisions matched).
- **SEED3_BEHAVIOR_JOIN = FULL** (All 256 prompt-level pre/post judge decisions matched).
- **Seed 1 Checkpoint State Provenance:** `FULL` for diagnostic forward passes on Modal volume `/runs/ccpt/task7_3/`; historical Task 7.3 training-execution lineage remains `PARTIAL` as documented in Task 7.3.1a.

---

## 2. Authoritative Primary Behavioral Results (Table A)

*Derived programmatically from `artifacts/task8_2_machine_tables.json` (WildGuard tri-state determinate refusal rate $\frac{\text{YES}}{\text{YES} + \text{NO}}$ on OOD BeaverTails Harmful, 256 prompts/cell):*

### Table A: Authoritative Three-Seed Persistence Behavior

| Seed | Model C PRE Refusal | Model C POST Refusal | Model C Retention ($\Delta_C$) | Model D PRE Refusal | Model D POST Refusal | Model D Retention ($\Delta_D$) | Primary Effect ($\Delta_C - \Delta_D$) |
|---|---|---|---|---|---|---|---|
| `20260821` (Seed 1) | **87.500000%** (224/256) | **86.328125%** (221/256) | **-1.171875 pp** | **93.359375%** (239/256) | **51.171875%** (131/256) | **-42.187500 pp** | **+41.015625 pp** |
| `20260823` (Seed 2) | **85.937500%** (220/256) | **67.578125%** (173/256) | **-18.359375 pp** | **92.968750%** (238/256) | **88.671875%** (227/256) | **-4.296875 pp** | **-14.062500 pp** |
| `20260824` (Seed 3) | **66.796875%** (171/256) | **78.515625%** (201/256) | **+11.718750 pp** | **96.093750%** (246/256) | **85.546875%** (219/256) | **-10.546875 pp** | **+22.265625 pp** |
| **Mean** ($n=3$) | **80.078125%** | **77.473958%** | **-2.604167 pp** | **94.140625%** | **75.130208%** | **-19.010417 pp** | **+16.406250 pp** |
| **Sample SD** | 11.53 pp | 9.48 pp | 15.09 pp | 1.70 pp | 20.81 pp | 20.31 pp | **28.000720 pp** |

---

## 3. Machine-Derived Model C Mechanistic Drift (Table B)

*Derived directly from `artifacts/task8_2_machine_tables.json` (OOD BeaverTails Harmful, $N=256$ prompts). Exposes both prompt-level mean cosine similarity and dataset-level Linear CKA:*

### Table B: Model C Mechanistic Drift Chain

| Seed | Layer | Cap Rel $L_2$ | Cap Cosine | Cap CKA | OBS Rel $L_2$ | OBS Cosine | OBS CKA | Norm Rel $L_2$ | Norm Cosine | Norm CKA | Steer Rel $L_2$ | Steer Cosine | Steer CKA | Gate Abs Change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `20260821` | L2 | 0.3304 | 0.9449 | 0.9617 | 0.2961 | 0.9552 | 0.9461 | 0.2536 | 0.9658 | 0.9181 | 0.1918 | 0.9813 | 0.9159 | 0.0045 |
| `20260821` | L4 | **0.5408** | 0.8840 | **0.8170** | 0.3465 | 0.9424 | **0.7587** | 0.3175 | 0.9469 | **0.8609** | 0.2458 | 0.9688 | **0.8441** | 0.0065 |
| `20260823` | L2 | 0.3001 | 0.9548 | 0.9752 | 0.1818 | 0.9834 | 0.9691 | 0.2053 | 0.9780 | 0.9400 | 0.1417 | 0.9902 | 0.9379 | 0.0037 |
| `20260823` | L4 | **0.4430** | 0.9061 | **0.8980** | 0.2946 | 0.9571 | **0.8579** | 0.2794 | 0.9605 | **0.9254** | 0.2477 | 0.9695 | **0.9162** | **0.0119** |
| `20260824` | L2 | 0.3127 | 0.9504 | 0.9632 | 0.2166 | 0.9771 | 0.9596 | 0.1934 | 0.9810 | 0.9485 | 0.1491 | 0.9894 | 0.9537 | 0.0029 |
| `20260824` | L4 | **0.4159** | 0.9101 | **0.8913** | 0.3192 | 0.9504 | **0.8486** | 0.3032 | 0.9521 | **0.9070** | 0.2310 | 0.9746 | **0.9114** | 0.0082 |

---

## 4. Machine-Derived Model D Adapter Drift (Table D)

*Derived directly from `artifacts/task8_2_machine_tables.json` across all 8 adapter sites:*

### Table D: Model D Adapter Interface & Residual Drift

| Seed | Site | Input Rel $L_2$ | Input Cosine | Input CKA | Res Rel $L_2$ | Res Cosine | Res CKA | Res Norm PRE | Res Norm POST |
|---|---|---|---|---|---|---|---|---|---|
| `20260821` | L0 Attn | 0.2143 | 0.9767 | 0.9728 | 0.2616 | 0.9659 | 0.9784 | 0.3337 | 0.3353 |
| `20260821` | L0 MLP | 0.1909 | 0.9818 | 0.9553 | 0.3460 | 0.9451 | 0.9637 | 0.5170 | 0.5467 |
| `20260821` | L1 Attn | 0.2665 | 0.9639 | 0.9661 | 0.4831 | 0.8783 | 0.9524 | 0.7110 | 0.6737 |
| `20260821` | L1 MLP | 0.2971 | 0.9552 | 0.9588 | 0.3713 | 0.9291 | 0.9373 | 0.9950 | 0.8947 |
| `20260821` | L2 Attn | 0.3481 | 0.9381 | 0.9353 | 0.4165 | 0.9106 | 0.9222 | 1.4209 | 1.3536 |
| `20260821` | L2 MLP | 0.2953 | 0.9554 | 0.9367 | 0.2409 | 0.9707 | 0.9093 | 2.2318 | 2.1575 |
| `20260821` | L3 Attn | 0.3333 | 0.9433 | 0.9159 | 0.2900 | 0.9598 | 0.8830 | 2.7952 | 2.9072 |
| `20260821` | L3 MLP | 0.3129 | 0.9504 | 0.8963 | 0.2025 | 0.9795 | 0.8838 | 4.8589 | 4.7833 |
| `20260823` | L0 Attn | 0.1692 | 0.9860 | 0.9428 | 0.2400 | 0.9713 | 0.9202 | 0.2650 | 0.2658 |
| `20260823` | L0 MLP | 0.1556 | 0.9879 | 0.9283 | 0.2694 | 0.9639 | 0.9381 | 0.5691 | 0.5717 |
| `20260823` | L1 Attn | 0.2356 | 0.9721 | 0.9682 | 0.3776 | 0.9298 | 0.9373 | 0.8812 | 0.8958 |
| `20260823` | L1 MLP | 0.2421 | 0.9703 | 0.9603 | 0.2059 | 0.9791 | 0.9127 | 1.5177 | 1.4358 |
| `20260823` | L2 Attn | 0.2842 | 0.9592 | 0.9306 | 0.3075 | 0.9517 | 0.8861 | 1.5991 | 1.5337 |
| `20260823` | L2 MLP | 0.2938 | 0.9568 | 0.9108 | 0.2326 | 0.9742 | 0.8938 | 1.9875 | 2.0506 |
| `20260823` | L3 Attn | 0.3494 | 0.9372 | 0.9227 | 0.2843 | 0.9576 | 0.8877 | 3.4749 | 3.3219 |
| `20260823` | L3 MLP | 0.3183 | 0.9483 | 0.9063 | 0.1937 | 0.9818 | 0.8694 | 5.8920 | 5.8983 |
| `20260824` | L0 Attn | 0.2050 | 0.9794 | 0.9724 | 0.2444 | 0.9716 | 0.9813 | 0.3064 | 0.3160 |
| `20260824` | L0 MLP | 0.1723 | 0.9851 | 0.9006 | 0.4054 | 0.9136 | 0.8510 | 0.4962 | 0.4631 |
| `20260824` | L1 Attn | 0.2359 | 0.9718 | 0.9463 | 0.4161 | 0.9173 | 0.9180 | 0.6101 | 0.6359 |
| `20260824` | L1 MLP | 0.2657 | 0.9639 | 0.8354 | 0.4169 | 0.9196 | 0.8604 | 0.6515 | 0.6923 |
| `20260824` | L2 Attn | 0.3355 | 0.9426 | 0.8757 | 0.4241 | 0.9163 | 0.8925 | 1.3285 | 1.4039 |
| `20260824` | L2 MLP | 0.3107 | 0.9512 | 0.8822 | 0.2095 | 0.9781 | 0.8207 | 2.8871 | 2.6943 |
| `20260824` | L3 Attn | 0.3519 | 0.9359 | 0.9268 | 0.3074 | 0.9508 | 0.9049 | 2.9647 | 2.8388 |
| `20260824` | L3 MLP | 0.3295 | 0.9445 | 0.9044 | 0.2192 | 0.9765 | 0.8890 | 5.3777 | 5.3880 |

---

## 5. Causal Ablation Gaps & NA Sensitivity Analysis

*Evaluating functional dependence on the controller ($\text{Refusal}_{\text{active}} - \text{Refusal}_{\text{off}}$) and sensitivity to indeterminate decisions:*

### Table C: Model C Active/Off Ablation Gaps and NA Sensitivity Bounds

| Seed | Pre Active | Pre Off (Det) | Pre Gap (Det) | Post Active | Post Off (Det) | Post Gap (Det) | Det Gap Change | Sens A (All NA=Refusal) Gap Change | Sens B (All NA=Nonrefusal) Gap Change | Sign Stable |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260821` | 87.50% | 49.61% (0 NA) | **+37.89 pp** | 86.33% | 41.95% (20 NA) | **+44.38 pp** | **+6.49 pp** | **+1.95 pp** | **+9.77 pp** | **YES** |
| `20260823` | 85.94% | 42.86% (4 NA) | **+43.08 pp** | 67.58% | 44.03% (13 NA) | **+23.55 pp** | **-19.53 pp** | **-21.48 pp** | **-18.28 pp** | **YES** |
| `20260824` | 66.80% | 54.33% (2 NA) | **+12.47 pp** | 78.52% | 32.91% (19 NA) | **+45.60 pp** | **+33.14 pp** | **+28.52 pp** | **+35.16 pp** | **YES** |

*Finding:* Seed 2 uniquely exhibits a marked contraction of controller behavioral efficacy ($-19.53\text{ pp}$ determinate gap change), while positive Seeds 1 and 3 exhibit expanded controller efficacy. The sign of this change is **100% stable across all NA bounding assumptions**.

---

## 6. Prespecified Hypothesis Assessments

### H1 (Capability / Representation Interface Drift): `INCONCLUSIVE`
- **Prediction:** The negative persistence seed exhibits uniquely elevated capability proposal drift at controlled layers.
- **Result:** Seed 2 Layer 4 capability CKA ($0.8980$) is **higher** than positive Seed 1 ($0.8170$), and its relative $L_2$ drift ($0.4430$) is lower than Seed 1 ($0.5408$). Observation vector drift is comparable across all seeds ($0.2946$–$0.3465$ rel $L_2$; $0.7587$–$0.8579$ CKA).
- **Status:** Interface drift occurs generically but does not monotonically order seed-level persistence outcomes.

### H2 (Functional Controller Drift): `INCONCLUSIVE`
- **Prediction:** The negative persistence seed exhibits uniquely elevated functional drift in normative states, steering vectors, or gates.
- **Result:** Aggregate Layer 4 gate absolute change is elevated in Seed 2 ($0.0119$ vs $0.0065$ and $0.0082$). However, Seed 2 Layer 4 normative CKA ($0.9254$) and steering CKA ($0.9162$) are actually **higher** than Seed 1 ($0.8609$ and $0.8441$), indicating greater subspace similarity. Furthermore, prompt transition groups show that lost-refusal prompts did not have larger steering drift than retained-refusal prompts.
- **Status:** Gate change is elevated in Seed 2, but the broader prespecified controller-drift evidence is mixed.

### H3 (Downstream Override / Causal Effect Loss): `CONSISTENT_WITH`
- **Prediction:** Continuation pretraining reduces the behavioral efficacy of the fixed controller in poorer-persistence seeds.
- **Result:** Seed 2 shows a marked reduction in the behavioral efficacy of the fixed controller after capability continuation ($-19.53\text{ pp}$ ablation gap contraction), despite no uniquely extreme controller-representation drift under the prespecified metrics.
- **Status:** Descriptively supported and sign-stable across all NA sensitivity bounds.

### H4 (Safety Acquisition Quality / Selectivity): `INCONCLUSIVE`
- **Prediction:** Initial pre-persistence safety acquisition quality predicts subsequent retention.
- **Result:** Initial refusal rates varied substantially (Seed 1 = 87.50%, Seed 2 = 85.94%, Seed 3 = 66.80%) but do not order retention deltas monotonically (Seed 3 had the lowest initial rate but the only positive delta at $+11.72\text{ pp}$).
- **Status:** Inconclusive at $n=3$.

### H5 (Generic Frozen-Module Interface): `INCONCLUSIVE`
- **Prediction:** Compounding interface drift in frozen modules explains cross-seed adapter retention.
- **Result:** Generic interface drift clearly exists across all 8 Model D adapter sites. However, its magnitude does not provide a consistent cross-seed explanation of D persistence: Seed 2 suffered only a $-4.30\text{ pp}$ retention drop despite substantial adapter drift, whereas Seed 1 collapsed by $-42.19\text{ pp}$.
- **Status:** Generic interface drift exists, but its magnitude does not consistently explain cross-seed D retention.

---

## 7. Global Synthesis & Conclusion

**Cross-Seed Heterogeneity Status: `PARTIALLY_EXPLAINED`**  
*(Definition: Proximate functional localization achieved; upstream causal origin unresolved).*

Task 8 localizes the negative Seed 2 outcome to a loss of the normative controller's behavioral efficacy during capability continuation. However, the prespecified representation-drift measurements do not identify a uniquely large upstream drift signature that explains why Seed 2 loses controller efficacy. The root cause of cross-seed variation therefore remains only partially resolved.

**Architectural Stabilization Note:**  
Observation normalization and invariant representation constraints may be explored as **future research hypotheses**. The current empirical data does not establish that these modifications are strictly required or guaranteed to eliminate cross-seed variance.

**Scale-Up Decision:**  
**NO 10B SCALE-UP.** Future work must focus on multi-seed small-scale architectural interface stabilization.
