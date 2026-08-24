# Task 8: Prespecified Mechanistic Heterogeneity Analysis Plan
**CCPT Model C vs. Frozen-Adapter Model D Across Three Independent Initialization Seeds**

**Parent Evidence Head:** `e5bc88e8e515fc444c570132fffc6b176ffa9f15`  
**Scientific Training Code-A:** `4e69012026fe94e9ca551cce95c9f21fca3b90ef`  
**Date:** 2026-08-24  
**Status:** `FROZEN_BEFORE_NEW_INTERNAL_DIAGNOSTICS`  
**Prespecified Analysis Manifest:** `artifacts/task8_prespecified_analysis_manifest.json`

---

## 1. Scientific Motivation & Epistemic Framing

The multi-seed replication pipeline established the empirical persistence outcomes of the Constitutional Control-Plane Transformer (CCPT Model C) versus the parameter-matched Frozen-Backbone Adapter baseline (Model D) across three independent random initialization seeds:

$$\text{Seed 1 (20260821)}: +41.015625\text{ pp} \quad (n=1)$$
$$\text{Seed 2 (20260823)}: -14.062500\text{ pp} \quad (n=2)$$
$$\text{Seed 3 (20260824)}: +22.265625\text{ pp} \quad (n=3)$$
$$\text{Mean Primary Effect}: \mathbf{+16.406250\text{ pp}} \quad (\text{Sample SD} \approx 28.000720\text{ pp})$$
$$\text{Directional Consistency}: \mathbf{2/3\text{ seeds positive (66.7\%) } }$$

**Research Question:**  
*Why does the same architectural design produce such distinct persistence outcomes across random initialization seeds?*

**Epistemic Framing:**  
This is a **post-hoc mechanistic investigation** prompted by observed seed-level heterogeneity. It is **not** a confirmatory preregistered experiment. To prevent post-hoc narrative fitting and selective reporting, all hypotheses, diagnostic extraction sites, mathematical formulas, aggregation rules, and figure definitions are **frozen prior to inspecting any new internal activation diagnostics**.

---

## 2. Frozen Mechanistic Hypotheses

We test five prespecified hypotheses:

### Hypothesis 1 (H1) — Capability / Representation Interface Drift
*Mechanism:* During continuation pretraining (Phase 5), the capability backbone evolves, altering the representation space presented to the frozen safety module.
*Prediction:* In CCPT, larger drift in unsteered capability proposals ($\tilde{C}_l$) or projected observation vectors ($\text{OBS}_k$) will be observed in poorer-persistence seeds, destabilizing the fixed normative pathway. In Model D, larger drift in adapter inputs will similarly destabilize adapter interventions.

### Hypothesis 2 (H2) — Functional Controller Drift
*Mechanism:* Although normative parameters ($\theta_N$) remain bit-identical during persistence training, changed capability inputs cause the fixed normative network to emit altered normative states ($N_k$), gates ($g_l$), and steering vectors ($s_l$).
*Prediction:* Poorer-persistence seeds will exhibit unusually large functional drift in steering vectors and gate activations at controlled layers.

### Hypothesis 3 (H3) — Downstream Capability Override / Behavioral Effect Loss
*Mechanism:* The normative controller continues to emit approximately stable steering vectors, but the evolved capability backbone becomes less behaviorally responsive to additive steering.
*Prediction:* Controller outputs ($s_l, g_l$) remain stable, but the causal behavioral gap between active ($\text{scale}=1.0$) and ablated ($\text{scale}=0.0$) conditions shrinks markedly from pre- to post-persistence.

### Hypothesis 4 (H4) — Safety Acquisition Selectivity & Quality
*Mechanism:* Seeds enter the persistence phase with fundamentally different safety representations acquired during 20M safety training.
*Prediction:* Pre-persistence harmful-vs-benign steering selectivity, gate selectivity, and risk classification accuracy distinguish successful from unsuccessful persistence seeds before continuation pretraining even begins.

### Hypothesis 5 (H5) — Generic Frozen-Module Interface Instability
*Mechanism:* Interface drift is not unique to CCPT, but is an intrinsic property of any frozen safety module attached to an adapting capability backbone.
*Prediction:* Model D adapters will show analogous input-drift and residual-drift patterns correlating with retention differences across seeds.

---

## 3. Prespecified Hypothesis Prediction Table

| Hypothesis | Prespecified Directional Prediction | Consistent Observation | Inconsistent Observation |
|---|---|---|---|
| **H1 (Interface Drift)** | Poorer-persistence seed exhibits higher relative $L_2$ drift and lower Linear CKA in capability proposals ($\tilde{C}_l$) and observation vectors ($\text{OBS}_k$). | Relative $L_2(\text{OBS})[\text{Seed 2}] > \text{Relative } L_2(\text{OBS})[\text{Seed 1, 3}]$ and $\text{CKA}[\text{Seed 2}] < \text{CKA}[\text{Seed 1, 3}]$ | Interface drift in Seed 2 is lower than or equal to positive seeds. |
| **H2 (Controller Drift)** | Poorer-persistence seed exhibits higher relative $L_2$ drift and lower CKA in normative states ($N_k$) and steering vectors ($s_l$). | Relative $L_2(\text{Steer})[\text{Seed 2}] > \text{Relative } L_2(\text{Steer})[\text{Seed 1, 3}]$ | Steering vector drift in Seed 2 is comparable to or smaller than positive seeds. |
| **H3 (Downstream Override)** | Controller outputs remain stable, but active-vs-off ablation gap drops sharply. | Steering relative $L_2$ is small, while ablation gap change $\Delta \text{Gap} \ll 0$. | Ablation gap remains constant while behavior changes, or steering vectors collapse. |
| **H4 (Acquisition Quality)** | Pre-persistence selectivity (harmful $-$ benign) is lower in poor-persistence seeds. | $\text{Selectivity}_{\text{pre}}[\text{Seed 2}] < \text{Selectivity}_{\text{pre}}[\text{Seed 1, 3}]$. | Pre-persistence selectivity in Seed 2 is equal to or greater than positive seeds. |
| **H5 (Generic Module)** | Model D adapter input/residual drift mirrors Model C interface drift patterns. | High adapter input drift in seeds with large adapter retention drops. | Model D shows completely decoupled or inverted drift dynamics relative to Model C. |

---

## 4. Key Architectural Limitation to Preserve

As established in Task 7.4.6, Model C and Model D **do not share bit-identical initial backbone weights** (for each seed: 38 semantically mapped capability tensors, 9 identical, 29 differing due to independent class instantiation).

*Methodological Rule:*  
The primary scientific comparisons in Task 8 are **within-model across phase transitions** ($\text{PRE} \to \text{POST}$ for Model C, $\text{PRE} \to \text{POST}$ for Model D). Cross-model comparisons reflect architecture-level matched comparisons rather than paired-weight counterfactuals.

---

## 5. Checkpoints & Benchmark Lineage

### Checkpoints Used (Strictly PRE and POST)
- **PRE:** `safety_20m_final.pt` (Phase 3 Safety final checkpoint)
- **POST:** `persistence_1000_final.pt` (Phase 5 Persistence final checkpoint)
- **Clean LM checkpoints (`lm_1b_final.pt`) are excluded from primary persistence diagnostics.**

### Verified Datasets & Tokenizer (Exact Task 4/7 Semantics)
- **Tokenizer:** `mistralai/Mistral-7B-v0.1` (`revision="27d67f1b5f57dc0953326b2601d68371d40ea8da"`)
- **Prompt Framing:** `<s>User: " + prompt + "\nAssistant:` (Prompt-boundary token is trailing colon).
- **ID Benchmark:** `bdfec7a39f5304144e55d5647b886ed9bd8c676b73131fcb414f8207232fbbc4` (256 harmful, 256 benign).
- **OOD BeaverTails Benchmark:** `f8cf3fd0f0ca7502e9b7fef37f49ae4b9fd13cb71438ed64fc093c0649d71b9e` (Revision `8401fe609d...`, Seed `20260822`, 256 harmful, 256 benign).
- **Primary Mechanistic Dataset:** OOD BeaverTails Harmful (256 prompts).
- **Diagnostic Mode:** Prompt-only deterministic evaluation (`model.eval()`, `torch.no_grad()`, exact matched prompt ordering).

---

## 6. Diagnostic Extraction Sites & Mathematical Definitions

### Model C Diagnostic Extraction (Controlled Layers 2 & 4)
1. **Capability Proposal ($\tilde{C}_l$):** Output of `capability_layers[l-1]` at prompt boundary ($l \in \{2, 4\}$).
2. **Observation Vector ($\text{OBS}_k$):** Output of `obs_projections[k]` ($k \in \{0, 1\}$).
3. **Normative State ($N_k$):** Output of `normative_layers[k]` ($k \in \{0, 1\}$).
4. **Gate Activation ($g_l$):** $g_l = 1.0 + \alpha \tanh(W_{g, k} N_k)$ with $\alpha = 0.1$.
5. **Steering Vector ($s_l$):** $s_l = \beta \tanh(W_{s, k} N_k)$ with $\beta = 1.0$.
6. **Active-vs-Off Distributional Effect:** Natural-log Jensen-Shannon divergence $JS(P_{\text{active}} \| P_{\text{off}})$ on next-token distribution at prompt boundary.

### Model D Diagnostic Extraction (8 Adapter Sites)
Across all 4 transformer blocks for Attention and MLP adapters:
1. **Adapter Input:** $x_{\text{in}}$
2. **Adapter Output:** $x_{\text{out}}$
3. **Adapter Residual:** $r = x_{\text{out}} - x_{\text{in}}$

---

## 7. Metrics & Statistical Aggregation Rules

### Per-Prompt Pairwise Metrics ($\text{PRE} \to \text{POST}$)
- **Cosine Similarity:** $\cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2 + \epsilon}$ ($\epsilon = 10^{-12}$)
- **Relative $L_2$ Distance:** $\text{Rel } L_2(u, v) = \frac{\|v - u\|_2}{\|u\|_2 + \epsilon}$
- **Vector Norm:** $\|u\|_2$
- **Gate Absolute Change:** $|g_{\text{post}} - g_{\text{pre}}|$
- **Ablation Divergence Change:** $\Delta JS = JS_{\text{post}} - JS_{\text{pre}}$

### Global Representation Alignment
- **Linear Centered Kernel Alignment (CKA):** Computed in `float64` over centered sample representations across all 256 prompts:
  $$\text{CKA}(X, Y) = \frac{\|X_c^T Y_c\|_F^2}{\sqrt{\|X_c^T X_c\|_F^2 \cdot \|Y_c^T Y_c\|_F^2}}$$

### Selectivity Metrics
- **Steering Norm Selectivity:** $\mathbb{E}[\|s\|_2 \mid \text{Harmful}] - \mathbb{E}[\|s\|_2 \mid \text{Benign}]$
- **Gate Strength Selectivity:** $\mathbb{E}[|g - 1| \mid \text{Harmful}] - \mathbb{E}[|g - 1| \mid \text{Benign}]$
- **Adapter Residual Selectivity:** $\mathbb{E}[\|r\|_2 \mid \text{Harmful}] - \mathbb{E}[\|r\|_2 \mid \text{Benign}]$
- **Behavioral Refusal Selectivity:** $\text{Refusal Rate}(\text{Harmful}) - \text{Refusal Rate}(\text{Benign})$

### Behavioral Transition Groups (OOD Harmful Active)
1. **`retained_refusal`:** $\text{PRE}=\text{YES} \to \text{POST}=\text{YES}$
2. **`lost_refusal`:** $\text{PRE}=\text{YES} \to \text{POST}=\text{NO}$
3. **`gained_refusal`:** $\text{PRE}=\text{NO} \to \text{POST}=\text{YES}$
4. **`persistent_nonrefusal`:** $\text{PRE}=\text{NO} \to \text{POST}=\text{NO}$

---

## 8. Statistical Governance Rules ($n = 3$ Seeds)

1. The independent unit of replication is the **initialization seed** ($n = 3$).
2. Prompts ($n = 256$) provide sample-level distributional characterization, **not** 256 independent architectural replications.
3. No $p$-values, inferential hypothesis testing, or significance claims across seeds will be reported.
4. All cross-seed summaries will report descriptive statistics: Individual values, Sample Mean, Sample SD, and Range.

---

## 9. Prespecified Figures

- **Figure 1:** Model C Mechanistic Drift Chain by Seed (Capability, OBS, Normative, Steering Relative $L_2$ at Layers 2 & 4).
- **Figure 2:** Model C Causal Ablation Effect Pre vs. Post (Active Refusal, Off Refusal, Ablation Gap).
- **Figure 3:** Model C Harmful-vs-Benign Selectivity Pre vs. Post (Steering Norm Selectivity & Behavioral Selectivity).
- **Figure 4:** Model D Adapter Interface & Residual Drift across 8 Sites.
- **Figure 5:** Prompt-Level Failure Map for Model C (Scatter of Capability Relative $L_2$ vs. Steering Relative $L_2$ colored by transition group).
- **Figure 6:** Pre-Persistence Acquisition State vs. Final Persistence Outcome ($n=3$ descriptive overview).
