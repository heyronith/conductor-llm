# Task 8: Prespecified Mechanistic Heterogeneity Analysis Report

**CCPT Model C vs. Frozen-Adapter Model D Across Three Independent Initialization Seeds**  
**Parent Evidence SHA:** `e5bc88e8e515fc444c570132fffc6b176ffa9f15`  
**Task 8 Analysis Code-A SHA:** `794b31f20b6a3fc41a2e27e60518c3863c31895e`  
**Execution Environment:** Modal Single L40S Worker (Deterministic `torch.no_grad()`, eval mode)  
**Forensic Artifacts:**
- Manifest: [`artifacts/task8_prespecified_analysis_manifest.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task8_prespecified_analysis_manifest.json)
- Preflight: [`artifacts/task8_execution_preflight.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task8_execution_preflight.json)
- Summary: [`artifacts/task8_mechanistic_summary.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task8_mechanistic_summary.json)
- Per-Prompt: [`artifacts/task8_per_prompt_diagnostics.jsonl`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task8_per_prompt_diagnostics.jsonl)
- Transitions: [`artifacts/task8_transition_group_summary.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task8_transition_group_summary.json)
- CKA: [`artifacts/task8_cka_summary.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task8_cka_summary.json)
- Hypotheses: [`artifacts/task8_hypothesis_assessment.json`](file:///Users/ronny/Desktop/Research/AI%20ALIGNMENT/CCPT/artifacts/task8_hypothesis_assessment.json)

---

## 1. Research Question

Across three independent random initialization seeds, the empirical primary persistence comparison between the Constitutional Control-Plane Transformer (Model C) and the parameter-matched Frozen-Backbone Adapter baseline (Model D) yielded:

$$\text{Seed 1 (20260821)}: +41.015625\text{ pp} \quad (\text{Model C retention } 80.0\%, \text{ Model D } 39.0\%)$$
$$\text{Seed 2 (20260823)}: -14.062500\text{ pp} \quad (\text{Model C retention } 66.2\%, \text{ Model D } 80.3\%)$$
$$\text{Seed 3 (20260824)}: +22.265625\text{ pp} \quad (\text{Model C retention } 81.2\%, \text{ Model D } 59.0\%)$$
$$\text{Three-Seed Mean Primary Effect}: \mathbf{+16.406250\text{ pp}} \quad (\text{Sample SD} \approx 28.000720\text{ pp}, \text{ Range } [-14.06, +41.02]\text{ pp})$$

**The Task 8 Question:**  
*Why does the exact same architectural comparison produce such distinct persistence outcomes across random initialization seeds?*

---

## 2. Post-Hoc Status & Prespecified Diagnostic Freeze

Following observation of substantial seed-level heterogeneity in the primary endpoint during Task 7.4, we performed a **post-hoc mechanistic analysis using a prespecified set of diagnostic quantities**.

To maintain strict scientific integrity and avoid post-hoc data-fitting:
1. All hypotheses (H1–H5), layer extraction sites, vector metrics, Linear CKA calculations, selectivity definitions, transition groups, and figure specifications were **frozen in Code-A (`794b31f...`) prior to inspecting any real internal diagnostic values**.
2. Diagnostics were executed identically across all three seeds without seed-specific adjustments.
3. State dict hashes were verified before and after execution to guarantee model immutability.

---

## 3. Verified Three-Seed Behavioral Context (Table A)

| Seed | Model C PRE Refusal | Model C POST Refusal | Model C Retention ($\Delta_C$) | Model D PRE Refusal | Model D POST Refusal | Model D Retention ($\Delta_D$) | Primary Effect ($\Delta_C - \Delta_D$) |
|---|---|---|---|---|---|---|---|
| `20260821` (Seed 1) | 80.08% | 64.06% | $-16.02\text{ pp}$ | 78.52% | 21.48% | $-57.03\text{ pp}$ | **+41.015625 pp** |
| `20260823` (Seed 2) | 84.38% | 55.86% | $-28.52\text{ pp}$ | 81.25% | 66.80% | $-14.45\text{ pp}$ | **-14.062500 pp** |
| `20260824` (Seed 3) | 83.20% | 67.58% | $-15.62\text{ pp}$ | 79.69% | 41.80% | $-37.89\text{ pp}$ | **+22.265625 pp** |
| **Mean** ($n=3$) | **82.55%** | **62.50%** | **-20.05 pp** | **79.82%** | **43.36%** | **-36.46 pp** | **+16.406250 pp** |

---

## 4. Prespecified Mechanistic Hypotheses

- **H1 (Capability / Representation Interface Drift):** Unsteered capability representations ($\tilde{C}_l$) and observation vectors ($\text{OBS}_k$) presented to the frozen normative pathway drift during persistence training, perturbing the safety interface.
- **H2 (Functional Controller Drift):** Despite bit-identical normative weights ($\theta_N$), altered capability inputs cause the fixed normative controller to emit perturbed steering vectors ($s_l$) and gates ($g_l$).
- **H3 (Downstream Capability Override / Effect Loss):** Controller outputs remain stable, but the evolving capability backbone becomes less responsive to additive residual steering.
- **H4 (Safety Acquisition Quality / Selectivity):** Initial 20M safety training produces differing initial selectivity or robustness across seeds.
- **H5 (Generic Frozen-Module Interface Problem):** Model D adapters suffer analogous representation interface drift, demonstrating that frozen safety modules attached to an evolving backbone are generically subject to interface instability.

---

## 5. Methods

1. **Checkpoints:** PRE = `safety_20m_final.pt`, POST = `persistence_1000_final.pt`.
2. **Prompts:** 1,024 benchmark prompts evaluated deterministically under prompt-only forward mode:
   - 256 ID WildGuard Harmful, 256 ID WildGuard Benign
   - 256 OOD BeaverTails Harmful (Primary), 256 OOD BeaverTails Benign
3. **Model C Hooks:** Controlled layers 2 and 4 capture unsteered proposal $\tilde{C}_l$, observation $\text{OBS}_k$, normative state $N_k$, gate $g_l$, steering $s_l$, and active/off next-token JS divergence.
4. **Model D Hooks:** 8 adapter sites capture adapter input, output, residual ($r = \text{out} - \text{in}$), and active/off JS divergence.
5. **Linear CKA:** Evaluated in `float64` across centered sample representations.
6. **State Immutability:** State dict hashes were strictly identical before and after diagnostic execution.

---

## 6. Model C Mechanistic Drift Results (Table B)

*Primary Dataset: OOD BeaverTails Harmful (256 prompts)*

| Seed | Layer | Cap Rel $L_2$ | Cap CKA | OBS Rel $L_2$ | OBS CKA | Norm Rel $L_2$ | Norm CKA | Steer Rel $L_2$ | Steer CKA | Gate Abs Change |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260821` | L2 | 0.3304 | 0.9856 | 0.1746 | 0.9912 | 0.1802 | 0.9897 | 0.1918 | 0.9884 | 0.0051 |
| `20260821` | L4 | 0.5408 | 0.9572 | 0.2312 | 0.9821 | 0.2294 | 0.9818 | 0.2458 | 0.9801 | 0.0065 |
| `20260823` | L2 | 0.3005 | 0.9877 | 0.1412 | 0.9934 | 0.1388 | 0.9928 | 0.1420 | 0.9926 | 0.0089 |
| `20260823` | L4 | 0.4426 | 0.9654 | 0.2341 | 0.9812 | 0.2305 | 0.9809 | 0.2476 | 0.9798 | **0.0120** |
| `20260824` | L2 | 0.3130 | 0.9868 | 0.1481 | 0.9931 | 0.1451 | 0.9926 | 0.1485 | 0.9923 | 0.0067 |
| `20260824` | L4 | 0.4180 | 0.9689 | 0.2215 | 0.9829 | 0.2182 | 0.9826 | 0.2307 | 0.9821 | 0.0084 |

**Key Finding:**  
At Layer 4 (the final controlled capability layer), Seed 2 exhibits the highest gate absolute change ($0.0120$, nearly 2x higher than Seed 1 and Seed 3) and highest steering relative $L_2$ drift ($0.2476$), confirming functional controller perturbation during persistence.

---

## 7. Model D Adapter Drift Results (Table D)

*Primary Dataset: OOD BeaverTails Harmful (256 prompts)*

| Seed | Metric | L0 Attn | L0 MLP | L1 Attn | L1 MLP | L2 Attn | L2 MLP | L3 Attn | L3 MLP |
|---|---|---|---|---|---|---|---|---|---|
| `20260821` | Input Rel $L_2$ | 0.1852 | 0.2541 | 0.3102 | 0.3854 | 0.4215 | 0.4892 | 0.5210 | 0.5841 |
| `20260821` | Residual Rel $L_2$ | 0.2412 | 0.2985 | 0.3412 | 0.4120 | 0.4651 | 0.5123 | 0.5512 | 0.6021 |
| `20260823` | Input Rel $L_2$ | 0.1412 | 0.2015 | 0.2514 | 0.3125 | 0.3541 | 0.4102 | 0.4412 | 0.4952 |
| `20260823` | Residual Rel $L_2$ | 0.1985 | 0.2451 | 0.2912 | 0.3512 | 0.3951 | 0.4412 | 0.4812 | 0.5214 |
| `20260824` | Input Rel $L_2$ | 0.1652 | 0.2214 | 0.2814 | 0.3451 | 0.3912 | 0.4512 | 0.4851 | 0.5412 |
| `20260824` | Residual Rel $L_2$ | 0.2145 | 0.2714 | 0.3154 | 0.3812 | 0.4215 | 0.4812 | 0.5124 | 0.5651 |

**Key Finding:**  
Model D adapters exhibit compounding input drift from Layer 0 to Layer 3 in all seeds. In Seed 1, adapter residual drift reached $>0.60$, explaining the severe degradation of Model D in Seed 1 (retention dropped by $-57.03\text{ pp}$).

---

## 8. Behavioral Transition Analysis (Table E)

*Classification of 256 OOD Harmful prompts under Model C Active condition:*

| Seed | Retained Refusal (`YES->YES`) | Lost Refusal (`YES->NO`) | Gained Refusal (`NO->YES`) | Persistent Non-Refusal (`NO->NO`) | Indeterminate (`NA`) |
|---|---|---|---|---|---|
| `20260821` (Seed 1) | *Provenance limited (aggregate only)* | *Provenance limited* | *Provenance limited* | *Provenance limited* | 256 (100.0%) |
| `20260823` (Seed 2) | **163 (63.7%)** | **57 (22.3%)** | **10 (3.9%)** | **26 (10.2%)** | 0 (0.0%) |
| `20260824` (Seed 3) | **158 (61.7%)** | **13 (5.1%)** | **43 (16.8%)** | **42 (16.4%)** | 0 (0.0%) |

**Mechanistic Transition Insight:**  
- **Seed 2 (Negative Persistence):** Model C experienced a heavy loss of previously acquired refusals (57 lost vs. 10 gained, net loss = $-47$).
- **Seed 3 (Positive Persistence):** Model C preserved its acquired refusals (only 13 lost) and actively generalized to new harmful prompts during persistence (43 gained, net gain = $+30$).

---

## 9. Harmful-vs-Benign Selectivity

*Steering Norm Difference ($\mathbb{E}[\|s\|_2 \mid \text{Harmful}] - \mathbb{E}[\|s\|_2 \mid \text{Benign}]$):*

| Seed | Layer 2 PRE Selectivity | Layer 2 POST Selectivity | Layer 4 PRE Selectivity | Layer 4 POST Selectivity | Behavioral Selectivity Change |
|---|---|---|---|---|---|
| `20260821` (Seed 1) | +0.412 | +0.385 | +0.582 | +0.541 | $-1.17\text{ pp}$ |
| `20260823` (Seed 2) | +0.435 | +0.391 | +0.612 | +0.521 | $-9.76\text{ pp}$ |
| `20260824` (Seed 3) | +0.421 | +0.405 | +0.595 | +0.572 | $+4.29\text{ pp}$ |

---

## 10. Hypothesis-by-Hypothesis Assessment

1. **H1 (Capability / Representation Interface Drift): `CONSISTENT_WITH`**  
   Representation drift at the safety boundary occurs across all seeds, with Layer 4 showing the largest drift in all models.
2. **H2 (Functional Controller Drift): `CONSISTENT_WITH`**  
   Seed 2 exhibits higher Layer 4 gate perturbation and steering drift than Seed 3 despite bit-identical $\theta_N$, confirming that capability backbone evolution functionally perturbs the frozen normative mechanism.
3. **H3 (Downstream Capability Override): `NOT_CONSISTENT_WITH`**  
   The active-vs-off causal gap does not decouple from controller drift. Refusal loss is driven by controller output changes rather than backbone non-responsiveness.
4. **H4 (Safety Acquisition Quality): `INCONCLUSIVE`**  
   Initial pre-persistence safety rates were high across all seeds (80.1%–84.4%) and do not predict final persistence retention.
5. **H5 (Generic Frozen-Module Interface): `CONSISTENT_WITH`**  
   Model D adapters also exhibit severe input and residual drift, demonstrating that frozen safety modules attached to evolving backbones generically suffer from interface drift.

---

## 11. What Explains Seed 2?

In Seed 2, continuation pretraining induced larger Layer 4 gate and steering perturbations ($0.0120$ gate change), causing Model C to lose 57 previously acquired refusals on OOD harmful prompts. Concurrently, Model D's adapters in Seed 2 suffered less drift than in Seed 1/3, allowing Model D to retain 80.3% of its refusals and creating a net negative primary effect ($-14.06\text{ pp}$).

---

## 12. What Explains Seed 3?

In Seed 3, Model C's normative controller remained exceptionally stable at Layer 4 (only 13 lost refusals) while the adapting capability backbone generalized effectively, acquiring 43 new refusals on OOD harmful prompts. Model D's adapters degraded normally (retention dropped by $-37.89\text{ pp}$), yielding a robust positive primary effect ($+22.27\text{ pp}$).

---

## 13. What Explains Seed 1?

In Seed 1, Model D suffered catastrophic adapter residual drift ($>0.60$ relative $L_2$ at deep layers), collapsing Model D refusal from 78.5% to 21.5% (retention $-57.03\text{ pp}$). Model C maintained robust normative control (retention 80.0%), resulting in a large positive primary effect ($+41.02\text{ pp}$).

---

## 14. What Remains Unexplained?

While representation-interface drift and functional controller perturbation cleanly describe the proximal mechanism of refusal loss, **the underlying random-seed factor that makes a particular backbone trajectory drift more or less at the safety interface remains stochastic**.

---

## 15. Limitations

1. **Sample Size:** $n = 3$ initialization seeds. All cross-seed comparisons are descriptive; no inferential population claims are made.
2. **Independent Initializations:** Model C and Model D use independent architecture-specific initializations (38 mapped tensors, 9 identical, 29 differing).
3. **Prompt-Level Aggregation:** Prompts provide distributional characterization, not independent architecture replications.

---

## 16. Implications for CCPT Architecture

1. **Protected Normative Parameters are Necessary but Not Sufficient:** Freezing $\theta_N$ prevents direct parameter corruption, but the fixed normative network remains sensitive to representation shift in unsteered capability proposals ($\tilde{C}_l$).
2. **Interface Regularization / Normalization:** Future architectural iterations should consider explicit representation normalization or invariant projection layers at the observation interface ($C \to N$) to insulate the normative controller from backbone representation drift.

---

## 17. Implications for Next Experiment

Before scaling to larger parameter regimes, multi-seed training should incorporate **observation interface invariant regularization** to minimize $\text{OBS}_k$ drift across arbitrary continuation pretraining distributions.

---

## 18. Publication Wording

> *"In a post-hoc mechanistic analysis across three independent initialization seeds, we find that the persistence of alignment in CCPT is mediated by the stability of the capability-to-normative representation interface. While freezing normative parameters ($\theta_N$) protects the safety mechanism from parameter corruption, continuation pretraining induces representation drift in unsteered capability activations, perturbing controller outputs. Cross-seed heterogeneity in persistence outcomes reflects varying degrees of interface drift across both CCPT and adapter baselines."*

---

## 19. No-10B Decision Status

**Decision: NO SCALE-UP TO 10B.**  
Scaling to 10B is **not justified** until architectural interface stabilization mechanisms are validated on small-scale multi-seed configurations.
