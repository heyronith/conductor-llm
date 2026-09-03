# Seed 4 Cost-Reduction Forensic Audit

**Audit code SHA:** `2b05b6b25beb293751fc62b4542e488a8a79439e`
**Billing-preflight base:** `feca09c00438f323c3903d1439dfee91ef728cd1`
**Remaining Modal credits:** `$0.00`
**Ledger reconciliation:** `PASS`

## 1. Credits and out-of-pocket framing

All Seed-4 metered spend is out-of-pocket cash:

`OUT_OF_POCKET_COST_USD = METERED_COST_USD`

No Starter credits, academic credits, promotions, or hypothetical refunds are assumed.

## 2. Seed-1 H100 object ledger

- Full-window Task-2 H100 billed: `$25.83683913`
- Sep-1 UTC Task-2 H100 billed: `$22.80192289`

### Classification totals (H100 USD)

- `DUPLICATE_EXECUTION`: `$0.6123`
- `FAILED_OR_ABORTED_EXECUTION`: `$3.9961`
- `GPU_IDLE_OR_SETUP`: `$0.1843`
- `REQUIRED_SUCCESSFUL_SCIENTIFIC_WORK`: `$20.7391`
- `VALID_BUT_NOT_SEED4_REQUIRED`: `$0.3050`

### Top objects

- `ap-TaUUJJEc7NPvKK0oya8ClI`: H100 `$20.7391` (18901.5s) → `REQUIRED_SUCCESSFUL_SCIENTIFIC_WORK` [HIGH]
- `ap-7BwZvJXfXhNf34YmxJZQ9T`: H100 `$2.9362` (2676.0s) → `FAILED_OR_ABORTED_EXECUTION` [MEDIUM]
- `ap-JLcljKGGok1GtmlCRigRdw`: H100 `$1.0599` (966.0s) → `FAILED_OR_ABORTED_EXECUTION` [MEDIUM]
- `ap-ELJZiCTkakrIWg1bI0qkFM`: H100 `$0.3160` (288.0s) → `DUPLICATE_EXECUTION` [HIGH]
- `ap-cKSKLaV2osBZ75RHNo1zb0`: H100 `$0.3050` (278.0s) → `VALID_BUT_NOT_SEED4_REQUIRED` [HIGH]
- `ap-5nKMP2j3evWrTZGWNFgdqd`: H100 `$0.2963` (270.0s) → `DUPLICATE_EXECUTION` [HIGH]
- `ap-kjZzGjjT5GmvBCeG6ShnSs`: H100 `$0.1843` (168.0s) → `GPU_IDLE_OR_SETUP` [HIGH]

**$5+ avoidable hypothesis:** `PARTIALLY_SUPPORTED` — HIGH-confidence avoidable H100 among non-TaUU objects is ~$1.10 (duplicate/idle residual on mixed eval apps). Remaining ~$4.00 (ap-7Bw + ap-JL) is MEDIUM-confidence failed/aborted and is NOT subtracted from the Seed-4 expected budget.

## 3. Seed-4 checkpoint reuse

- Seed-4 path exists: `False`
- Any valid reusable Seed-4 checkpoint: `False`
- Estimated savings from existing valid state: `$0.00`

## 4. Proven savings (HIGH confidence only)

- Proven avoidable H100: `$1.1016`
- Proven avoidable non-H100 (Task-2 invalid L40S): `$3.1147`
- Total proven avoidable: `$4.2163`
- MEDIUM H100 excluded from expected: `$3.9961`

## 5. Irreducible cost floor

- Scientifically irreducible H100 floor: `$20.7391`
- Scientifically irreducible full-protocol floor: `$23.0118`
- Evidence quality: `HIGH_AGGREGATE_MEDIUM_PER_MODEL`

Per-model H100 split unavailable from Modal billing; floor is aggregate successful app object ap-TaUUJJEc7NPvKK0oya8ClI plus corrected eval. Going materially below this would require changing the experiment or a cheaper external rate.

## 6. Revised Seed-4 projections (out-of-pocket)

- **A historical-style repeat:** `$28.20` (H100 $25.84, L40S $1.90, other $0.46)
- **B clean expected:** `$23.01` (H100 $20.74, L40S $1.90, other $0.37)
- **C conservative:** `$24.88`
- **Recommended hard authorization:** `$27.37`

## 7. Protocol-preserving optimizations

- `cpu_preflight_before_h100`: reduces GPU_IDLE_OR_SETUP; not separately dollarized beyond proven residual H100 (scientific semantics changed: `False`)
- `single_controlled_deployment`: avoids MEDIUM failed/aborted launches if discipline holds (scientific semantics changed: `False`)
- `no_h100_spawn_when_persistence_4000_exists`: $1.10 HIGH proven H100 class (scientific semantics changed: `False`)
- `corrected_framed_eval_only`: avoid repeating Task-2 L40S $3.11; pay corrected ~$1.90 instead (scientific semantics changed: `False`)
- `sequential_model_pipelines_with_cash_gate`: financial risk control; may increase wall clock, not GPU-seconds (scientific semantics changed: `False`)
- `reuse_materialized_data_and_image_layers`: cold-start/setup only (scientific semantics changed: `False`)

## 8. Runtime optimization review

- CUDA Graphs: `REJECT_FOR_SEED4`
- torch.compile: `SAFE_CANDIDATE_FOR_SEPARATE_PARITY_BENCHMARK`
- Prefetch/staging: `SAFE_CANDIDATE_FOR_SEPARATE_PARITY_BENCHMARK`
- Paid parity benchmark worthwhile: `True`
- Max proposed benchmark cost: `$0.30` (not executed)

## Machine artifacts

- `artifacts/seed4_cost_reduction_task2_object_ledger.json`
- `artifacts/seed4_cost_reduction_checkpoint_reuse_audit.json`
- `artifacts/seed4_cost_reduction_waste_analysis.json`
- `artifacts/seed4_cost_reduction_execution_plan.json`
- `artifacts/seed4_cost_reduction_projection.json`
- `artifacts/seed4_cost_reduction_runtime_optimization_review.json`
