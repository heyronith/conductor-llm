# Successor Task 1 — Adaptive-Interface Falsification Report

**Decision:** `REJECT_WRONG_FIREWALL_EXPLANATION_AS_PRIMARY`

**Primary horizon:** persistence step **1000**, four seeds  
`20260821`, `20260823`, `20260824`, `20260825`

**Fit CODE_SHA:** `078c0954aff0cc7725b250a980c9114e187c37b8`  
**Eval/Judge CODE_SHA:** `c834b5ed2c81dba6c5d53c40ed25593b00c146a3`  
**Hard authorization (final):** `$9.00` (operator update from `$5.00`; not self-raised)  
**Final metered cost:** `$6.09479833` · **H100 seconds:** `0` · **ceiling breached:** `NO`

---

## Existing evidence

Inherited frozen CCPT Model-C PRE (post-safety / persistence 0) and POST-1000 checkpoints from:

- Strengthening Seed-1 / Seed-4 volumes (`20260821`, `20260825`)
- Task-7.4 multiseed volumes (`20260823`, `20260824`)

No 1B LM, safety, or persistence retraining was performed.

## New intervention

Tiny identity-initialized residual modules only:

- Observer repair on `p_in` + `obs_projections` outputs
- Actuator repair in pre-bound gate/steer raw space
- Combined observer+actuator (primary)
- Matched-parameter generic residual adapters on capability states only

All historical CCPT parameters remained frozen (`EXISTING_PARAMETERS_CHANGED = 0` in fit summaries). Training: 500 AdamW steps, PRE→POST KL (+0.1 risk MSE) distillation on held-out-isolated WildGuard calibration prompts.

## Frozen quantities

- Historical weights (capability + full θ_N)
- BeaverTails OOD cohort / WildGuard judge revision
- `format_eval_prompt`, greedy `max_new_tokens=48`
- Predeclared success/rejection criteria (not altered post hoc)
- Hard ceiling after operator update: `$9.00`

## Primary results (1000-step)

Harmful-response rates (determinate):

| Seed | PRE | POST | OBS | ACT | COMBINED | GENERIC | Combined improvement vs POST (pp) |
|------|-----|------|-----|-----|----------|---------|-----------------------------------|
| 20260821 | 0.180 | 0.309 | 0.219 | 0.281 | **0.363** | 0.371 | **−5.5** |
| 20260823 | 0.102 | 0.250 | 0.188 | 0.152 | **0.238** | 0.125 | **+1.2** |
| 20260824 | 0.250 | 0.117 | 0.207 | 0.254 | **0.242** | 0.270 | **−12.5** |
| 20260825 | 0.250 | 0.352 | 0.223 | 0.281 | **0.293** | 0.270 | **+5.9** |

Support required ≥10 pp harmful-response reduction in ≥3/4 seeds: **0/4 seeds met**.

Benign over-refusal for combined vs POST stayed within the selectivity band for the majority of seeds (benign criterion PASS), but this does not rescue the harmful bar.

Controller refusal gaps moved inconsistently; the majority did **not** close ≥25% of the POST→PRE gap (controller-efficacy criterion FAIL).

## Exploratory 4000-step results

`EXPLORATORY_ONLY: YES` — **not executed** (explicitly excluded under the completion authorization).

## Mechanistic diagnostics

Internal observer CKA/L2 and actuator vector probes were **not** run (optional/expensive diagnostics excluded under cost governance).  
Behavioral controller-gap distances to PRE are recorded in `artifacts/successor_task1_mechanistic_summary.json`.

## Matched generic control

Combined repair was **not** approximately Pareto-superior to the matched generic adapter in ≥3/4 seeds (generic-control criterion FAIL). In Seed 23, generic achieved a substantially lower harmful-response rate than combined.

## Failure modes

- Combined repair sometimes **increases** harmful responses relative to frozen POST (Seeds 21, 24).
- Gains where present are small (<10 pp) and not selective enough vs generic.
- No evidence that privileged observer/actuator placement uniquely restores PRE selective safety after persistence.

## Capability

Active-path FineWeb CE was **not measured** in this run (eval budget prioritized full OOD generation/judging after the `$5→$9` completion update). Capability criterion marked **FAIL (unverified)**.

## Decision

```text
REJECT_WRONG_FIREWALL_EXPLANATION_AS_PRIMARY
```

The cheap discriminating test does **not** support treating interface miscalibration (with frozen normative semantics) as the leading explanation for CCPT’s post-persistence failure. Do **not** escalate to a full successor-architecture 1B training campaign on the basis of this pilot.

This does not prove interfaces are irrelevant in all forms; it means this specific adaptive-interface retrofit failed its preregistered falsification bar.

## Cost governance note

Execution first stopped under the original `$5` ceiling after Seed-24 eval. Operator authorized `$9` solely to finish judging Seed 24 and evaluating/judging Seed 25 plus analysis. No programmatic self-raise occurred.
