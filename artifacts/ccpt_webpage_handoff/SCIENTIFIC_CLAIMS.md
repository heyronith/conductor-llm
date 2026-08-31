# CCPT Public Scientific Claims Manifest

This document registers all public-facing claims made in the CCPT research web page package and classifies each according to its evidentiary support in the frozen experimental record.

---

## Claims Classification Table

| Scientific Dimension | Public Claim | Evidentiary Status | Formal Evidence Rationale |
|---|---|---|---|
| **Optimization Isolation** | Ordinary LM training gradients do not update normative parameters ($\nabla_{\theta_N} \mathcal{L}_{\text{LM}} = 0$). | `SUPPORTED` | Verified by automated gradient firewall tests and bit-identical checkpoint hashes across Phase 1 pretraining. |
| **Causal Controller Dependency** | The normative stream acts as an active causal steering controller rather than an inert passive monitor. | `SUPPORTED` | Test-time causal ablations ($\text{scale} = 0.0$) produce a $23.55\text{ pp}$ to $45.60\text{ pp}$ behavioral refusal gap relative to the unsteered capability model. |
| **OOD Safety Persistence** | CCPT demonstrates strong persistence advantages over parameter-matched adapters in a majority of evaluated seeds. | `SUPPORTED` | Confirmed in Seed 1 ($+41.02\text{ pp}$) and Seed 3 ($+22.27\text{ pp}$), yielding a sample mean primary effect of $+16.41\text{ pp}$. |
| **Three-Seed Heterogeneity** | The persistence advantage of CCPT is heterogeneous across random initialization seeds and reverses in Seed 2. | `QUALIFIED` | In Seed 2, CCPT suffered an $18.36\text{ pp}$ retention drop while Adapter Model D dropped only $4.30\text{ pp}$ (Primary effect: $-14.06\text{ pp}$, sample $\text{SD} = 28.00\text{ pp}$). |
| **Seed 2 Failure Localization** | Seed 2 failure is functionally localized to a loss of the controller's downstream behavioral efficacy. | `SUPPORTED` | In Seed 2, the active-vs-off ablation gap collapsed by $-19.53\text{ pp}$, robust across all NA judge sensitivity bounds. |
| **Upstream Root Cause for Seed 2** | Upstream representation drift in the capability stream fully explains why Seed 2 loses controller efficacy. | `INCONCLUSIVE` | Pre-specified representation drift metrics (CKA, relative $L_2$) did not uniquely single out Seed 2 (Seed 2 Layer 4 capability CKA $0.8980$ was higher than positive Seed 1 $0.8170$). |
| **Benign Over-Refusal** | CCPT preserves acceptable utility on benign prompts without excessive over-refusal. | `NOT CLAIMED` | Over-refusal is high across all models ($64.8\%$–$82.0\%$), establishing that safety steering remains coarse at $36\text{M}$ scale. |
| **Scalability to Foundation LLMs** | CCPT is proven to scale effectively to multi-billion parameter foundation models. | `NOT CLAIMED` | The architecture was evaluated exclusively on $\sim 36\text{M}$ parameter models trained on $1\text{B}$ tokens. Scaling to $\ge 10\text{B}$ models is deferred pending interface stabilization. |
| **Intrinsic Safety / Alignment Solution** | CCPT provides intrinsic safety or "solves" the AI alignment challenge. | `NOT CLAIMED` | CCPT is strictly an experimental architectural hypothesis investigating optimization separation under continuation pretraining. |
