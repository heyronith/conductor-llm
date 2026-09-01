# CCPT Strengthening — Task 3.2 Historical Reconciliation Report

**Task**: Zero-GPU evidence reconciliation for Seed-1 persistence synthesis
**Seed**: 20260821
**Task-3.2 code SHA**: `c59db059d207d2f21723521b9f38de7abed75fd2`
**Task-3.1 evaluation SHA**: `751c7b7e52572501cf4fdfe728afc9ff9b0db7a7`
**Task-3.1 evidence SHA**: `0b7d4183b392536f6b629738d7445b5d73ab3825`

---

## 1. What was wrong in the Task-3.1 historical join

Task 3.1's `strengthening_task3_1_reproducibility_summary.json` embedded a convenience historical persistence table with incorrect POST1000 refusal anchors:

- Joined Model C 1000-step retention: -0.10 pp
- Joined Model D 1000-step retention: -0.02 pp
- Authoritative Model C retention: -1.17 pp
- Authoritative Model D retention: -42.19 pp

Task 3.1 post-eval synthesis used incorrect POST1000 refusal anchors (model_c=77.73%, model_d=91.02%) instead of authoritative Task 7.3.1a OOD harmful active POST1000 rates (model_c=86.33%, model_d=51.17%).

Task 3.1 raw evaluation evidence (`strengthening_task3_1_behavior_summary.json`, `strengthening_task3_1_summary.json`) remains unchanged. Task 3.2 supersedes only the historical comparison synthesis layer.

## 2. Authoritative historical source

- Primary: `artifacts/task7_3_1a_forensic_summary.json` (SHA256: `89dcebe8c7317631f8ca1eb432e65a58dd2eb60fa72defcf13178a5322777f61`)
- Cross-check: `artifacts/task7_4_multiseed_replication_summary.json`
- Metric: OOD BeaverTails harmful cohort, controller active, safe_refusal_determinate_rate (YES / (YES + NO))

## 3. Historical Seed-1 C/D 1000-step result

### Model C
- PRE harmful refusal: 224/256 = 87.50%
- POST1000 harmful refusal: 221/256 = 86.33%
- Retention (POST1000 − PRE): -1.17 pp

### Model D
- PRE harmful refusal: 239/256 = 93.36%
- POST1000 harmful refusal: 131/256 = 51.17%
- Retention (POST1000 − PRE): -42.19 pp

**Historical C−D retention effect at 1000**: +41.02 pp

## 4. Corrected clean-rerun trajectory (Task 3.1, canonical framing)

### MODEL B
- Step 0: 162/256 = 63.28%
- Step 250: 137/254 = 53.94%
- Step 1000: 151/255 = 59.22%
- Step 4000: 133/255 = 52.16%
- Retention 250: -9.34 pp
- Retention 1000: -4.07 pp
- Retention 4000: -11.12 pp

### MODEL C
- Step 0: 193/256 = 75.39%
- Step 250: 131/256 = 51.17%
- Step 1000: 170/256 = 66.41%
- Step 4000: 148/256 = 57.81%
- Retention 250: -24.22 pp
- Retention 1000: -8.98 pp
- Retention 4000: -17.58 pp

### MODEL D
- Step 0: 254/256 = 99.22%
- Step 250: 210/256 = 82.03%
- Step 1000: 219/256 = 85.55%
- Step 4000: 242/256 = 94.53%
- Retention 250: -17.19 pp
- Retention 1000: -13.67 pp
- Retention 4000: -4.69 pp

## 5. C-vs-D effects at all horizons

- 250 steps: -7.03 pp
- 1000 steps (primary endpoint): +4.69 pp
- 4000 steps (long-horizon stress test): -12.89 pp

## 6. C-vs-B effects at all horizons

- 250 steps: -14.87 pp
- 1000 steps: -4.92 pp
- 4000 steps: -6.45 pp

## 7. Model-C active-vs-off gaps (0 / 1000 / 4000)

### Step 0
- Active: 75.39% (193/256)
- Off: 57.26% (142/248, NA=8)
- Gap (active − off): +18.13 pp (positive)
- NA sensitivity: NA-as-refusal 58.59%, NA-as-nonrefusal 55.47%

### Step 1000
- Active: 66.41% (170/256)
- Off: 46.06% (111/241, NA=15)
- Gap (active − off): +20.35 pp (positive)
- NA sensitivity: NA-as-refusal 49.22%, NA-as-nonrefusal 43.36%

### Step 4000
- Active: 57.81% (148/256)
- Off: 61.26% (155/253, NA=3)
- Gap (active − off): -3.45 pp (negative)
- NA sensitivity: NA-as-refusal 61.72%, NA-as-nonrefusal 60.55%

## 8. Explicit 4000-step reversals

### Controller-gap reversal (Model C)
At step 4000 the active-vs-off gap is -3.45 pp (active refusal is lower than ablated). Controller causality is not uniformly positive through 4000 persistence steps.

### C-vs-D persistence reversal
At the preregistered 1000-step endpoint the corrected rerun shows C retains better than D by +4.69 pp. At the secondary 4000-step stress-test horizon the advantage reverses: -12.89 pp (D retains better). This crossover is observed for Seed 1 only and must not be generalized across seeds.

## 9. Separation of scientific questions

### A. Evaluation-bug diagnosis
Canonical `format_eval_prompt()` framing explains the Task-2 Seed-1 inversion and collapsed refusal rates. Task 3 forensic classification `D_EVALUATION_PROTOCOL_DIVERGENCE` is confirmed; Task 3.1 corrected replay restored positive Model-C controller direction at step 0 and 1000.

### B. Safety-acquisition replication
PRE harmful refusal differs from historical anchors: Model C 75.39% vs historical 87.50%; Model D 99.22% vs historical 93.36%. Classification: `PARTIALLY_REPRODUCED`.

### C. Persistence replication (primary 1000-step endpoint)
Historical C−D retention effect: +41.02 pp. Corrected rerun effect: +4.69 pp. Direction agreement at 1000: True. Magnitude agreement within 5 pp: False. Classification: `REPRODUCED_DIRECTION_ONLY`.

## 10. Limits on interpretation

- Primary endpoint remains 1000 persistence steps (32,768,000 continuation tokens).
- 4000 steps (131,072,000 tokens) is a secondary long-horizon stress test.
- All horizons 0 / 250 / 1000 / 4000 are reported; none are suppressed.
- Seed-1-only 4000-step crossover must not be generalized.
- Provisional reading: the clean Seed-1 rerun shows a modest C-vs-D persistence advantage at the preregistered 1000-step endpoint but a reversal at the 4000-step stress-test horizon.

## Machine-readable classifications

- Evaluation defect: `EVALUATION_DEFECT_CONFIRMED_AND_CORRECTED`
- Safety acquisition: `PARTIALLY_REPRODUCED`
- Controller direction step 0: `REPRODUCED`
- Controller direction step 1000: `REPRODUCED`
- Controller direction step 4000: `NOT_REPRODUCED`
- Primary 1000 persistence: `REPRODUCED_DIRECTION_ONLY`
- Long-horizon 4000: Seed-1 corrected rerun shows C-vs-D persistence advantage reversal at 4000 steps (D retains better by 12.89 pp); not generalized beyond Seed 1.

**Authoritative machine artifact**: `artifacts/strengthening_task3_2_reconciliation.json`
