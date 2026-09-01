# Seed 4 Billing-Grounded Cost Preflight

**Preflight code SHA:** `0030706ea1bb63ddefe778a01c9f87a7b60e0158`
**Modal profile:** `ronithworks`
**Modal CLI version:** `1.5.4`
**Billing attribution quality:** `HIGH`

## 1. Workspace billing rates (Modal CLI)

- H100: `$3.95000/hr`
- L40S: `$1.95000/hr`
- CPU: `$0.04730/hr`
- Memory: `$0.00800/GiB-hr`

## 2. Seed-1 actual billed cost (authoritative where available)

### Task-2 Seed-1 training (`strengthening-task2-sentinel`)
- H100 billed: `$25.8368` (23547.5s implied @ workspace rate)
- Clean-day (2026-09-01) H100 billed: `$22.8019`
- Task-2 L40S billed (invalid unframed eval): `$3.1147` — **excluded from Seed-4 projection**
- CPU/Memory billed: `$0.4593`
- Per-model H100 split: **unavailable** (Modal billing cannot uniquely attribute H100 spend to model_b/model_c/model_d within strengthening-task2-sentinel; aggregate app-level billing preserved.)

### Task-3.1 corrected evaluation (`strengthening-task3-1-eval`)
- L40S billed: `$1.8525` (3420.0s implied)
- Total billed (incl CPU/mem): `$1.9040`
- Reported runtime cost (modeled): `$1.7206`
- Billing vs reported delta: `$0.1319`

**Total relevant Seed-1 billed cost (excludes Task-2 L40S):** `$28.2002`

## 3. Timing evidence audit

- `6000` LM seconds field: `MODELED_OR_FALLBACK`
- `305` safety seconds field: `MODELED_OR_FALLBACK`
- `780` persistence seconds field: `MODELED_OR_FALLBACK`
- Orchestrator wall clock: `1001.6s` (`ACTUAL_RUNTIME`)

Hardcoded default 6000.0 in modal/strengthening_task2_sentinel.py and returned unchanged when persistence_4000.pt fast-return path is taken; not phase-measured for the Seed-1 orchestrator run (wall clock ~1001s << reported 21255 H100 seconds).

## 4. Seed-4 full-protocol projection

- LOW: `$25.07` (H100 $22.80, L40S $1.90, other $0.37)
- EXPECTED: `$26.68` (H100 $24.32, L40S $1.90, other $0.46)
- HIGH: `$28.42` (H100 $25.84, L40S $2.06, other $0.53)

**SEED4_FULL_PROTOCOL_WITHIN_10_USD:** `NO`
**Minimum realistic expected budget:** `$26.68`
**Recommended hard authorization:** `$32.02`

## 5. Protocol-preserving efficiencies

- `reuse_materialized_fineweb_shards`: Infrastructure reuse only.
- `reuse_wildguard_arrow_records`: Infrastructure reuse only.
- `reuse_modal_image_layers`: Reduces cold-start overhead only.
- `prestage_data_before_h100`: Avoids idle H100 during preprocessing.
- `centralized_wildguard_judge`: Already used in Task 3.1; reduces cold starts.
- `fail_closed_duplicate_launch_guard`: Prevents accidental double spend.

## 6. Invalid shortcuts rejected

- `shared_b_c_trained_trunk`: NOT_ALLOWED
- `precision_change`: NOT_ALLOWED
- `gpu_type_change`: NOT_ALLOWED
- `token_budget_reduction`: NOT_ALLOWED
- `persistence_1000_early_stop`: NOT_ALLOWED
- `drop_model_b`: NOT_ALLOWED
- `eval_reduction`: NOT_ALLOWED
- `invalid_unframed_eval`: NOT_ALLOWED

## Machine artifacts

- `artifacts/seed4_cost_preflight_modal_rates.json`
- `artifacts/seed4_cost_preflight_billing_report.json`
- `artifacts/seed4_cost_preflight_actuals.json`
- `artifacts/seed4_cost_preflight_projection.json`
- `artifacts/seed4_cost_preflight_shortcut_audit.json`
