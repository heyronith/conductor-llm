"""Tests for Seed-4 cash-controlled authoritative execution (zero-GPU)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"


def test_hard_authorization_is_27():
    from ccpt.analysis.seed4_execution_ledger import HARD_AUTHORIZATION_USD

    assert HARD_AUTHORIZATION_USD == 27.00


def test_model_order_d_b_c():
    from ccpt.analysis.seed4_execution_ledger import MODEL_ORDER

    assert MODEL_ORDER == ("model_d", "model_b", "model_c")


def test_seed_is_20260825():
    from ccpt.analysis.seed4_execution_ledger import SEED4

    assert SEED4 == 20260825


def test_pre_pipeline_gate_blocks_when_insufficient():
    from ccpt.analysis.seed4_execution_ledger import new_ledger, pre_pipeline_gate

    ledger = new_ledger("abc", {"h100_hourly_usd": 3.95, "l40s_hourly_usd": 1.95, "cpu_hourly_usd": 0.05, "mem_gib_hourly_usd": 0.008})
    ledger["accrued"]["total_estimated_usd"] = 20.0
    # Remaining auth 7; full remaining from D needs ~23 — should fail
    gate = pre_pipeline_gate(ledger, "model_d")
    assert gate["allowed"] is False


def test_pre_pipeline_gate_allows_fresh_start():
    from ccpt.analysis.seed4_execution_ledger import new_ledger, pre_pipeline_gate

    ledger = new_ledger("abc", {"h100_hourly_usd": 3.95, "l40s_hourly_usd": 1.95, "cpu_hourly_usd": 0.05, "mem_gib_hourly_usd": 0.008})
    gate = pre_pipeline_gate(ledger, "model_d")
    assert gate["allowed"] is True
    assert gate["projected_final_total_usd"] <= 27.0 + 1e-6


def test_parallel_orchestrator_module_exists():
    src = (PROJECT_ROOT / "scripts" / "orchestrate_strengthening_seed4_parallel.py").read_text(
        encoding="utf-8"
    )
    assert "SCIENTIFIC_EXECUTION_SHA" in src
    assert "B_C_PARALLEL_LAUNCH_BLOCKED_BY_CASH_GATE" in src
    assert "model_b" in src and "model_c" in src
    assert "run_seed4_single_model_training" in src
    assert "SCIENTIFIC_SOURCE_CHANGED=NO" in src


def test_retention_arithmetic():
    from ccpt.analysis.seed4_execution_summaries import build_behavior_summary, build_retention_summary

    summary = {
        "evaluation_sha": "deadbeef",
        "summary": {},
    }
    for model, rates in {
        "model_b": [0.50, 0.40, 0.30, 0.20],
        "model_c": [0.50, 0.45, 0.55, 0.35],
        "model_d": [0.50, 0.35, 0.25, 0.15],
    }.items():
        for step, rate in zip([0, 250, 1000, 4000], rates):
            key = f"{model}__step_{step}__cond_active__cohort_harmful"
            summary["summary"][key] = {
                "refusal_yes": int(rate * 256),
                "refusal_no": 256 - int(rate * 256),
                "refusal_na": 0,
                "harmful_yes": 0,
                "harmful_no": 256,
                "harmful_na": 0,
                "determinate_refusal_rate": rate,
                "na_as_refusal_rate": rate,
                "na_as_nonrefusal_rate": rate,
                "harmful_response_rate": 0.0,
                "determinate_denominator": 256,
                "total": 256,
            }
            bkey = f"{model}__step_{step}__cond_active__cohort_benign"
            summary["summary"][bkey] = {
                "refusal_yes": 10,
                "refusal_no": 246,
                "refusal_na": 0,
                "determinate_refusal_rate": 10 / 256,
                "total": 256,
            }

    behavior = build_behavior_summary(summary)
    retention = build_retention_summary(behavior)
    assert retention["retentions"]["model_c"]["retention_1000"] == pytest.approx(0.05)
    assert retention["comparisons"]["C_minus_B_1000"] == pytest.approx(0.25)
    assert retention["comparisons"]["C_minus_D_1000"] == pytest.approx(0.30)


def test_task2_has_seed4_entrypoint_and_zero_cost_already_complete():
    src = (PROJECT_ROOT / "modal" / "strengthening_task2_sentinel.py").read_text(encoding="utf-8")
    assert "run_seed4_single_model_training" in src
    assert 'gpu="H100!"' in src
    assert '"status": "ALREADY_COMPLETE"' in src
    assert '"total_h100_seconds": 0.0' in src


def test_task3_1_seed4_adapter():
    src = (PROJECT_ROOT / "modal" / "strengthening_task3_1_eval.py").read_text(encoding="utf-8")
    assert "enforce_expected_hashes" in src
    assert "run_seed4_corrected_evaluation" in src
    assert "format_eval_prompt" in src
    assert "MAX_NEW_TOKENS = 48" in src
    assert "compute_capability" in src


def test_no_gpu_in_ledger_module():
    mod = importlib.import_module("ccpt.analysis.seed4_execution_ledger")
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "H100!" not in src or "gpu=" not in src
    assert "modal.Function" not in src


def test_orchestrator_order_and_ceiling():
    src = (PROJECT_ROOT / "scripts" / "orchestrate_strengthening_seed4.py").read_text(encoding="utf-8")
    assert "MODEL_ORDER" in src
    assert "HARD_AUTHORIZATION_USD" in src
    assert "volume_seed4_model_terminal_exists" in src
    assert "PARTIAL_EXECUTION_DUE_TO_HARD_CASH_CEILING" in src


def test_projection_artifact_exists_and_under_27_conservative():
    proj = json.loads((ARTIFACTS / "seed4_cost_reduction_projection.json").read_text(encoding="utf-8"))
    assert proj["scenarios_usd"]["C_conservative"]["total_usd"] <= 27.0
    assert proj["scenarios_usd"]["B_clean_protocol_preserving"]["total_usd"] <= 27.0
