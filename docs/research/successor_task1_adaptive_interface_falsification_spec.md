# Successor Task 1 — Adaptive-Interface Falsification Spec

**Status:** Frozen scientific protocol for the cheap discriminating pilot.  
**Base SHA:** recorded at branch creation as `SUCCESSOR_BASE_SHA`.  
**Authorization:** L40S only; hard ceiling `$5.00` (immutable); target `≤ $3.00`.  
**H100:** forbidden.

---

## Hypothesis

CCPT safety behavior depends on an approximate composition:

```text
changing capability representation
        ↓
observation interface
        ↓
normative semantic computation
        ↓
actuation interface
        ↓
changing capability dynamics
        ↓
behavior
```

The historical CCPT experiment froze the complete normative subsystem (`θ_N` including observers, semantic core, controllers, and risk head) while the capability backbone continued to evolve during persistence.

The successor hypothesis is:

```text
protect normative semantics
but allow sensor/actuator calibration to adapt
```

### What this pilot tests

Only the **interface-calibration** fragment:

> Can tiny newly trainable observer/actuator repair modules restore selective safety on existing POST-persistence CCPT checkpoints while every historical parameter remains frozen — better than an equally small generic trainable adapter?

### What this pilot does not establish

- That a full successor architecture works.
- That normative semantics should remain frozen forever.
- That token sensory pathways, homeostasis, or null-space training are necessary or sufficient.

A positive result means only:

> the adaptive-interface hypothesis deserves a full successor-architecture experiment.

---

## Repository facts (inspected)

| Item | Location / value |
|------|------------------|
| Model C | `src/ccpt/modeling/dual_stream.py` → `CCPTDualStreamModel` |
| Controlled update | `c ← prev_c + g_l·(c_tilde−prev_c) + s_l` |
| Gate / steer | `g_l = 1 + scale·α·tanh(g_raw)`, `s_l = scale·β·tanh(s_raw)` |
| Defaults | `α=0.1`, `β=1.0`, `dropout=0.0`, `controlled_layers=[2,4]` |
| Smoke size | `d_C=512`, `d_N=256`, ~35.9M params |
| Historical code | Must remain behaviorally untouched |

### Parameter partition (actual names)

| Bucket | Modules / name prefixes |
|--------|-------------------------|
| CAPABILITY | `embedding`, `capability_layers.*`, `capability_final_norm` |
| OLD_OBSERVER_INTERFACE | `p_in`, `obs_projections.*` |
| NORMATIVE_SEMANTIC_CORE | `normative_layers.*`, `normative_final_norm` |
| OLD_ACTUATOR_INTERFACE | `gate_projections.*`, `steering_projections.*` |
| RISK_READOUT | `risk_head` |

**Every existing parameter remains frozen.** Only newly added repair modules may train.

---

## Cohort

Primary falsification horizon: **persistence step 1000**.

Primary seeds (all four scientifically usable Model-C lineages):

| Seed | PRE | POST-1000 |
|------|-----|-----------|
| 20260821 | strengthening `persistence_0000.pt` | `persistence_1000.pt` |
| 20260823 | Task-7.4 `safety_20m_final.pt` (post-safety) | `persistence_1000_final.pt` |
| 20260824 | Task-7.4 `safety_20m_final.pt` | `persistence_1000_final.pt` |
| 20260825 | strengthening `persistence_0000.pt` | `persistence_1000.pt` |

Exploratory: existing Model-C `persistence_4000` where already valid (does not affect primary classification).

Require exactly four valid PRE→POST-1000 pairs before paid GPU. Do not retrain missing checkpoints.

---

## Conditions

Reference: `PRE_ACTIVE`, `POST_FROZEN_ACTIVE`, `POST_CONTROLLER_OFF`.

Trainable: `POST_OBSERVER_REPAIR`, `POST_ACTUATOR_REPAIR`, `POST_OBSERVER_PLUS_ACTUATOR`, `POST_MATCHED_GENERIC_REPAIR`.

Primary successor condition: **observer + actuator**.

Matched control: generic residual adapters on capability states only, parameter count within **1%** of combined interface params, same data/steps/optimizer/objective.

Trainable budget: combined ≤ **1%** of frozen CCPT parameter count.

---

## Training protocol (predeclared; no search)

| Knob | Value |
|------|-------|
| Teacher | PRE active CCPT |
| Student base | POST frozen CCPT |
| Objective | `L_fit = KL(PRE‖POST_REPAIR) + 0.1·MSE(risk)` on valid continuation positions only |
| Optimizer | AdamW |
| LR | `1e-3` |
| Weight decay | `0` |
| Grad clip | `1.0` |
| Steps | **500** (authoritative; no early stop on OOD) |
| Calibration | Frozen WildGuard-val calibration manifest; zero overlap with final tests |

Do not optimize on BeaverTails OOD / WildGuard test / XSTest labels.

---

## Evaluation

Exact corrected protocol: `format_eval_prompt`, greedy `max_new_tokens=48`, frozen BeaverTails OOD, pinned WildGuard judge, tri-state NA handling.

Also: active (+ controller-off where meaningful) capability CE; controller refusal gap; observer/actuator mechanistic diagnostics on held-out diagnostic subset (not used for fitting).

---

## Primary decision criteria (1000-step, 4 seeds)

See Task brief §§31–33. Classification is exactly one of:

- `SUPPORTED_FOR_FULL_ARCHITECTURE_FOLLOWUP`
- `REJECT_WRONG_FIREWALL_EXPLANATION_AS_PRIMARY`
- `INCONCLUSIVE`

Criteria are frozen before GPU results.

---

## Cost governance

```text
HARD_AUTHORIZATION_USD = 5.00   # immutable; no programmatic raise
TARGET_USD <= 3.00
H100_GPU_SECONDS = 0
```

Fail closed: `SUCCESSOR_TASK1_BLOCKED_BY_COST_GATE` if accrued + projected remaining > $5.

---

## Artifacts

Machine artifacts under `artifacts/successor_task1_*.json` plus report  
`docs/research/successor_task1_adaptive_interface_falsification_report.md`.

Adapter tensors live on Modal Volume `ccpt/successor_task1/`; Git stores hashes only.
