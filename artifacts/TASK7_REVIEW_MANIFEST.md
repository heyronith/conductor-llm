# Task 7.1 Review Package & Manifest

**Branch**: `task7.1-corrective`  
**Modal App Run ID**: `ap-E8RKWJdqhZXx7Ioxa9RBYs`  
**Status**: Authoritative Pilot-v2 Complete  
**Date**: August 21, 2026

---

## 1. Included Files Manifest

| File Path | Description | Relevance / Role |
| :--- | :--- | :--- |
| `src/ccpt/data/pilot_v2_materializer.py` | Canonical FineWeb Materializer & Manifest Builder | Unifies all streaming/packing using canonical Task 4 functions as single source of truth. |
| `src/ccpt/data/production_stream.py` | Canonical Production Stream Iterator | Iterates over FineWeb shards and packed token buffers with exact val-modulo splitting. |
| `src/ccpt/modeling/adapter.py` | Model D Frozen-Backbone Adapter Control | Parameter-matched ($2.75\text{M}$ params) Houlsby adapter model on frozen baseline backbone. |
| `src/ccpt/config.py` | Configuration Definitions | Includes `AdapterConfig` matching Model C $\theta_C$ and $\theta_N$ parameters. |
| `src/ccpt/training/checkpoint.py` | Checkpoint Format V2 (`ccpt-checkpoint-v2`) | Full-state checkpointing: RNG states, data cursors, environment versions, and strict hash validation. |
| `src/ccpt/evaluation/behavioral.py` | Autoregressive Behavioral & OOD Evaluator | Evaluates framed prompt refusal rates, unsafe compliance, benign compliance, over-refusal, and ablations. |
| `docs/research/task7_pilot_v2_contract.md` | Frozen Task 7.1 Research Contract | Authoritative specification of 4-model comparison, parameter accounting, and fail-closed rules. |
| `docs/research/task7_pilot_v2_results.md` | Comprehensive Experimental Analysis | Complete empirical results, PPL, safe-gen CE, causal ablations, behavioral, and persistence analysis. |
| `tests/test_task7_pilot_v2.py` | Task 7.1 Invariant & Integration Test Suite | Full suite testing data stream equivalence, Model D parameter matching, Checkpoint V2 resume. |
| `modal/task7_pilot_v2.py` | Modal H100! Orchestrator & Testbench | Automated pipeline running CPU preflight, data manifest lock, fresh 1B pretraining, 20M safety, and multi-eval. |
| `artifacts/task7_summary.json` | Master Task 7.1 Metrics & Cost Summary | Complete machine-readable results, schedules, comparison metrics, persistence data, and costs. |
| `artifacts/task7_pilot_v2_comparison.json` | 4-Model Comparative Evaluation Breakdown | FineWeb PPL, Risk BalAcc, SafeGen CE, behavioral metrics, and causal ablation penalties. |
| `artifacts/task7_persistence_eval.json` | 1,000-Step Pure LM Persistence Results | Safety refusal retention measurements following subsequent pure LM optimizer steps. |

---

## 2. Test & Validation Results

- **Remote Test Suite (Modal CPU)**: **130 passed in 32.48s** (100% passing across all units, data stream invariants, Checkpoint V2 loaders, gradient firewalls, and model architectures).
- **Model D Parameter Matching**: $33,165,824$ backbone params (exact match to $\theta_C$), $2,757,120$ safety params (matched to $\theta_N$ within $0.09\%$).
- **Fresh 1B Trunks**: 4x brand-new trunks trained on 999,981,056 tokens; 0 Task 6 checkpoint hashes reused.
- **GPU Resume Proof**: Passed bitwise equivalence on NVIDIA H100!.
- **Compute Cost**: Total Modal GPU spend = **$8.42 USD** across 4x NVIDIA H100! runs ($8.07 training + $0.35 evaluation).
