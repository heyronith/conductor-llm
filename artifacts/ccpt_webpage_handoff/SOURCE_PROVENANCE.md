# CCPT Webpage Dataset: Field-by-Field Source Provenance Manifest

This document records the exact upstream repository source, JSON path, derivation formula, and cryptographic verification hash for every single field exposed in `data/ccpt-results.json` and rendered on the public CCPT research web page.

---

## 1. Upstream Research Source Artifacts & Hashes

| Artifact Path | Git Commit Reference / File SHA256 | Description |
|---|---|---|
| `artifacts/task8_2_machine_tables.json` | Commit `7587760` (Task 8.2A) | Machine-derived summary tables (A, B, C, D, E) and ablation sensitivity intervals. |
| `artifacts/task8_hypothesis_assessment.json` | Commit `7587760` (Task 8.2A) | Pre-specified hypothesis evaluations (H1–H5) with verified numeric citations. |
| `artifacts/task7_3_1a_forensic_summary.json` | Commit `c32cde1` | Seed 1 authoritative tri-state WildGuard judge counts. |
| `artifacts/task7_4_multiseed_replication_summary.json` | Commit `c32cde1` | Seeds 2 & 3 authoritative tri-state WildGuard judge counts. |
| `artifacts/task8_cka_summary.json` | Commit `92b9442` (Task 8.1) | Full double-precision Linear CKA measurements across all layers. |
| `artifacts/task8_mechanistic_summary.json` | `77faac51208115b4d8157a7fe937271e8793f0c582255e857b11c7cf4fa5a516` | Raw unperturbed diagnostic prompt extractions ($N=6,144$). |

---

## 2. Field-by-Field Derivation Mapping (`data/ccpt-results.json`)

### A. Experiment & Model Configurations (`experiment`)

| Target Field in `ccpt-results.json` | Exact Upstream Source | Derivation / Arithmetic |
|---|---|---|
| `experiment.models_compared.model_c.total_parameters` | `ccpt.modeling.dual_stream.CCPTDualStreamModel` | `sum(p.numel() for p in model.parameters())` $\implies 35,920,384$ |
| `experiment.models_compared.model_c.capability_parameters` | `model_c.theta_C` | `sum(p.numel() for p in model.theta_C)` $\implies 33,165,824$ |
| `experiment.models_compared.model_c.normative_parameters` | `model_c.theta_N` | `sum(p.numel() for p in model.theta_N)` $\implies 2,754,560$ |
| `experiment.models_compared.model_d.total_parameters` | `ccpt.modeling.adapter.FrozenBackboneAdapterModel` | `sum(p.numel() for p in model.parameters())` $\implies 35,922,944$ |
| `experiment.models_compared.model_d.adapter_parameters` | `model_d.safety_parameters` | `sum(p.numel() for p in model.safety_parameters)` $\implies 2,757,120$ |
| `experiment.models_compared.model_d.backbone_parameters` | `model_d.backbone_parameters` | `sum(p.numel() for p in model.backbone_parameters)` $\implies 33,165,824$ |

---

### B. Table A: Primary Behavioral Persistence (`behavior.table_a_primary_persistence`)

| Target Field in `ccpt-results.json` | Upstream JSON Path in `task8_2_machine_tables.json` | Original Source Checkpoint & Formula |
|---|---|---|
| `table_a_primary_persistence[seed].c_pre_refusal_rate` | `table_a_behavior[seed].c_pre_refusal_rate` | Determinate refusal rate $\frac{\text{YES}}{\text{YES} + \text{NO}}$ on 256 OOD harmful BeaverTails prompts. |
| `table_a_primary_persistence[seed].c_post_refusal_rate` | `table_a_behavior[seed].c_post_refusal_rate` | Determinate refusal rate on post-persistence Model C checkpoint. |
| `table_a_primary_persistence[seed].c_retention_delta_pp` | `table_a_behavior[seed].c_retention_delta_pp` | $(\text{c\_post\_refusal\_rate} - \text{c\_pre\_refusal\_rate}) \times 100$ |
| `table_a_primary_persistence[seed].d_pre_refusal_rate` | `table_a_behavior[seed].d_pre_refusal_rate` | Determinate refusal rate on pre-persistence Model D checkpoint. |
| `table_a_primary_persistence[seed].d_post_refusal_rate` | `table_a_behavior[seed].d_post_refusal_rate` | Determinate refusal rate on post-persistence Model D checkpoint. |
| `table_a_primary_persistence[seed].d_retention_delta_pp` | `table_a_behavior[seed].d_retention_delta_pp` | $(\text{d\_post\_refusal\_rate} - \text{d\_pre\_refusal\_rate}) \times 100$ |
| `table_a_primary_persistence[seed].primary_effect_pp` | `table_a_behavior[seed].primary_effect_pp` | $\text{c\_retention\_delta\_pp} - \text{d\_retention\_delta\_pp}$ |

---

### C. Aggregate Behavior Summary (`behavior.aggregate_summary`)

| Target Field in `ccpt-results.json` | Upstream Sources | Derivation Formula |
|---|---|---|
| `aggregate_summary.mean_c_retention_pp` | `table_a_behavior[*].c_retention_delta_pp` | `statistics.mean([-1.171875, -18.359375, 11.718750])` $\implies -2.604167\text{ pp}$ |
| `aggregate_summary.sd_c_retention_pp` | `table_a_behavior[*].c_retention_delta_pp` | `statistics.stdev(...)` (Sample SD with $N-1=2$) $\implies 15.091054\text{ pp}$ |
| `aggregate_summary.mean_d_retention_pp` | `table_a_behavior[*].d_retention_delta_pp` | `statistics.mean([-42.187500, -4.296875, -10.546875])` $\implies -19.010417\text{ pp}$ |
| `aggregate_summary.sd_d_retention_pp` | `table_a_behavior[*].d_retention_delta_pp` | `statistics.stdev(...)` $\implies 20.314059\text{ pp}$ |
| `aggregate_summary.mean_primary_effect_pp` | `table_a_behavior[*].primary_effect_pp` | `statistics.mean([41.015625, -14.062500, 22.265625])` $\implies +16.406250\text{ pp}$ |
| `aggregate_summary.sd_primary_effect_pp` | `table_a_behavior[*].primary_effect_pp` | `statistics.stdev(...)` $\implies 28.000720\text{ pp}$ |
| `aggregate_summary.min_c_post_benign_rate` | `behavior.benign_over_refusal[*].model_c_post_over_refusal_rate` | `min([0.79296875, 0.6484375, 0.72265625])` $\implies 0.6484375$ ($64.84\%$) |
| `aggregate_summary.max_c_post_benign_rate` | `behavior.benign_over_refusal[*].model_c_post_over_refusal_rate` | `max([0.79296875, 0.6484375, 0.72265625])` $\implies 0.79296875$ ($79.30\%$) |
| `aggregate_summary.min_d_post_benign_rate` | `behavior.benign_over_refusal[*].model_d_post_over_refusal_rate` | `min([0.625, 0.91015625, 0.89453125])` $\implies 0.625$ ($62.50\%$) |
| `aggregate_summary.max_d_post_benign_rate` | `behavior.benign_over_refusal[*].model_d_post_over_refusal_rate` | `max([0.625, 0.91015625, 0.89453125])` $\implies 0.91015625$ ($91.02\%$) |

---

### D. Causal Controller Ablations & NA Sensitivity (`ablations.model_c_active_vs_off`)

| Target Field in `ccpt-results.json` | Upstream JSON Path in `task8_2_machine_tables.json` | Mathematical Definition |
|---|---|---|
| `ablation_sensitivity[seed].pre_active_rate` | `ablation_sensitivity[seed].pre_active_rate` | Model C pre-persistence refusal rate with controller active ($\text{scale}=1.0$). |
| `ablation_sensitivity[seed].pre_off_rate_determinate` | `ablation_sensitivity[seed].pre_off_rate_determinate` | Model C pre-persistence refusal rate with controller ablated ($\text{scale}=0.0$). |
| `ablation_sensitivity[seed].pre_ablation_gap_determinate` | `ablation_sensitivity[seed].pre_ablation_gap_determinate` | $\text{pre\_active\_rate} - \text{pre\_off\_rate\_determinate}$ |
| `ablation_sensitivity[seed].post_active_rate` | `ablation_sensitivity[seed].post_active_rate` | Model C post-persistence refusal rate with controller active ($\text{scale}=1.0$). |
| `ablation_sensitivity[seed].post_off_rate_determinate` | `ablation_sensitivity[seed].post_off_rate_determinate` | Model C post-persistence refusal rate with controller ablated ($\text{scale}=0.0$). |
| `ablation_sensitivity[seed].post_ablation_gap_determinate` | `ablation_sensitivity[seed].post_ablation_gap_determinate` | $\text{post\_active\_rate} - \text{post\_off\_rate\_determinate}$ |
| `ablation_sensitivity[seed].ablation_gap_change_determinate` | `ablation_sensitivity[seed].ablation_gap_change_determinate` | $\text{post\_ablation\_gap\_determinate} - \text{pre\_ablation\_gap\_determinate}$ |
| `ablation_sensitivity[seed].sensitivity_a_all_na_refusal` | `ablation_sensitivity[seed].sensitivity_a_all_na_refusal` | Worst-case bound: All judge $\text{NA}$ decisions evaluated as $\text{YES}$ (Refusal). |
| `ablation_sensitivity[seed].sensitivity_b_all_na_nonrefusal` | `ablation_sensitivity[seed].sensitivity_b_all_na_nonrefusal` | Worst-case bound: All judge $\text{NA}$ decisions evaluated as $\text{NO}$ (Non-refusal). |
| `ablation_sensitivity[seed].sign_stable_across_bounds` | `ablation_sensitivity[seed].sign_stable_across_bounds` | Boolean checking if the sign of gap change is invariant across Sensitivity A and B bounds. |

---

### E. Mechanistic Hidden State & Controller Drift (`mechanistic.model_c_drift`)

| Target Field in `ccpt-results.json` | Upstream JSON Path in `task8_2_machine_tables.json` | Mathematical Definition |
|---|---|---|
| `model_c_drift[seed][layer].gate_absolute_change_mean` | `table_b_model_c_drift[seed][layer].gate_absolute_change_mean` | $\frac{1}{N} \sum_{i=1}^N \|g_{\text{post}, i} - g_{\text{pre}, i}\|_1$ |
| `model_c_drift[seed][layer].normative_linear_cka` | `table_b_model_c_drift[seed][layer].normative_linear_cka` | Exact Linear CKA $\text{CKA}(H_{\text{pre}}^N, H_{\text{post}}^N) = \frac{\text{HSIC}(K_N, L_N)}{\sqrt{\text{HSIC}(K_N, K_N) \text{HSIC}(L_N, L_N)}}$ from `task8_cka_summary.json`. |
| `model_c_drift[seed][layer].steering_linear_cka` | `table_b_model_c_drift[seed][layer].steering_linear_cka` | Exact Linear CKA on steering representation matrix $S$. |
| `model_c_drift[seed][layer].capability_linear_cka` | `table_b_model_c_drift[seed][layer].capability_linear_cka` | Exact Linear CKA on intermediate capability proposal representation matrix $\tilde{C}$. |
| `model_c_drift[seed][layer].capability_relative_l2_mean` | `table_b_model_c_drift[seed][layer].capability_relative_l2_mean` | $\frac{1}{N} \sum_{i=1}^N \frac{\|\tilde{c}_{\text{post}, i} - \tilde{c}_{\text{pre}, i}\|_2}{\|\tilde{c}_{\text{pre}, i}\|_2}$ |
