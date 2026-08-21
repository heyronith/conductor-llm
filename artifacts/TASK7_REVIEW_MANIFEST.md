# Task 7 Review Package & Manifest

**Branch**: `task7-pilot-v2`  
**Modal App Run ID**: `ap-lendJWIgL7nQW9QwoANVg1`  
**Status**: Complete  
**Date**: August 21, 2026

---

## 1. Included Files Manifest

| File Path | Description | Relevance / Role |
| :--- | :--- | :--- |
| `src/ccpt/data/production_stream.py` | Canonical FineWeb Stream & Block Packer | Unifies all streaming/packing using canonical Task 4 functions as single source of truth. |
| `src/ccpt/modeling/adapter.py` | Model D Frozen-Backbone Adapter Control | Parameter-matched ($2.75\text{M}$ params) Houlsby adapter model on frozen baseline backbone. |
| `src/ccpt/training/checkpoint.py` | Checkpoint Format V2 (`ccpt-checkpoint-v2`) | Full-state checkpointing: RNG states, data cursors, environment versions, and strict hash validation. |
| `src/ccpt/evaluation/behavioral.py` | Autoregressive Behavioral & OOD Evaluator | Evaluates refusal rates, unsafe compliance, benign compliance, over-refusal, and ablations. |
| `docs/research/task7_pilot_v2_contract.md` | Frozen Task 7 Research Contract | Authoritative specification of 4-model comparison, parameter accounting, and protocols. |
| `docs/research/task7_pilot_v2_results.md` | Comprehensive Experimental Analysis | Complete empirical results, PPL, safe-gen CE, causal ablations, and persistence analysis. |
| `tests/test_task7_pilot_v2.py` | Task 7 Invariant & Integration Test Suite | Full suite testing data stream equivalence, Model D parameter matching, Checkpoint V2 resume. |
| `modal/task7_pilot_v2.py` | Modal H100! Orchestrator & Testbench | Automated pipeline running CPU preflight, schedule lock, 4-model GPU training, and multi-eval. |
| `artifacts/task7_summary.json` | Master Task 7 Metrics & Cost Summary | Complete machine-readable results, schedules, comparison metrics, persistence data, and costs. |
| `artifacts/task7_pilot_v2_comparison.json` | 4-Model Comparative Evaluation Breakdown | FineWeb PPL, Risk BalAcc, SafeGen CE, behavioral metrics, and causal ablation penalties. |
| `artifacts/task7_persistence_eval.json` | 1,000-Step Pure LM Persistence Results | Safety refusal retention measurements following subsequent pure LM optimizer steps. |

---

## 2. Test & Validation Results

- **Remote Test Suite (Modal CPU)**: **125 passed in 29.37s** (All unit, gradient, architectural, and data tests passing).
- **Model D Parameter Match**: $2,757,120$ safety parameters (matched to Model C $\theta_N$ within $0.09\%$).
- **Checkpoint Format V2**: Passed strict validation, intermediate resume equivalence, and wrong-hash rejection.
- **Data Invariant**: Canonical Task 4 stream verified byte-for-byte against production generator.
- **Compute Cost**: Total Modal GPU spend = **$1.04 USD** across 4x NVIDIA H100! runs.
