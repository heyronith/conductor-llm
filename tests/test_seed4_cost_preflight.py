"""Tests for Seed 4 billing-grounded cost preflight."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"


@pytest.fixture(scope="module")
def artifacts():
    actuals_path = ARTIFACTS / "seed4_cost_preflight_actuals.json"
    if not actuals_path.exists():
        from ccpt.analysis.seed4_cost_preflight import write_all_artifacts

        write_all_artifacts(PROJECT_ROOT)
    return {
        "rates": json.loads((ARTIFACTS / "seed4_cost_preflight_modal_rates.json").read_text()),
        "actuals": json.loads((ARTIFACTS / "seed4_cost_preflight_actuals.json").read_text()),
        "projection": json.loads((ARTIFACTS / "seed4_cost_preflight_projection.json").read_text()),
        "shortcuts": json.loads((ARTIFACTS / "seed4_cost_preflight_shortcut_audit.json").read_text()),
    }


def test_rates_artifact_schema(artifacts):
    rates = artifacts["rates"]
    assert "workspace_h100_hourly_usd" in rates
    assert "workspace_l40s_hourly_usd" in rates
    assert rates["evidence_class"] == "ACTUAL_BILLED"
    assert rates["workspace_h100_hourly_usd"] > 0
    assert rates["workspace_l40s_hourly_usd"] > 0


def test_actual_vs_modeled_classification(artifacts):
    timing = artifacts["actuals"]["timing_field_audit"]
    assert timing["lm_pretrain_seconds_field"]["evidence_class"] == "MODELED_OR_FALLBACK"
    assert artifacts["actuals"]["task2_seed1_training"]["evidence_class"] == "ACTUAL_BILLED"
    assert (
        artifacts["actuals"]["task3_1_corrected_evaluation"]["reported_cost_usd"]["evidence_class"]
        == "MODELED_OR_FALLBACK"
    )
    assert (
        artifacts["actuals"]["task3_1_corrected_evaluation"]["reported_runtime_seconds"]["evidence_class"]
        == "ACTUAL_RUNTIME"
    )


def test_projection_arithmetic(artifacts):
    projection = artifacts["projection"]
    for name in ("low", "expected", "high"):
        scenario = projection["scenarios_usd"][name]
        expected_total = round(
            scenario["h100_usd"] + scenario["l40s_usd"] + scenario["other_incremental_usd"], 4
        )
        assert scenario["total_usd"] == expected_total


def test_projection_includes_full_protocol_scope(artifacts):
    scope = artifacts["projection"]["protocol_scope"]
    assert set(scope["models"]) == {"model_b", "model_c", "model_d"}
    assert scope["lm_pretrain_1b"] is True
    assert scope["safety_tokens"] == 20_010_611
    assert scope["persistence_steps"] == [0, 250, 1000, 4000]
    assert "format_eval_prompt" in scope["evaluation"]


def test_ten_dollar_feasibility_derived_not_hardcoded(artifacts):
    projection = artifacts["projection"]
    expected_total = projection["scenarios_usd"]["expected"]["total_usd"]
    assert projection["seed4_full_protocol_within_10_usd"] == (expected_total <= 10.0)
    assert projection["seed4_full_protocol_within_10_usd"] is False

    mod = importlib.import_module("ccpt.analysis.seed4_cost_preflight")
    source = inspect.getsource(mod.build_projection)
    assert "within_10 = False" not in source
    assert "seed4_full_protocol_within_10_usd'] = False" not in source


def test_no_gpu_calls_in_preflight_module():
    mod = importlib.import_module("ccpt.analysis.seed4_cost_preflight")
    source = inspect.getsource(mod)
    assert "gpu=" not in source.lower()
    assert "cuda" not in source.lower()
    assert "torch" not in source.lower()
    assert mod.H100_GPU_SECONDS_AUTHORIZED == 0
    assert mod.L40S_GPU_SECONDS_AUTHORIZED == 0


def test_invalid_shortcuts_rejected(artifacts):
    decisions = {row["option"]: row for row in artifacts["shortcuts"]["decisions"]}
    for key in (
        "shared_b_c_trained_trunk",
        "precision_change",
        "gpu_type_change",
        "token_budget_reduction",
        "persistence_1000_early_stop",
        "drop_model_b",
        "eval_reduction",
        "invalid_unframed_eval",
    ):
        assert decisions[key]["decision"] == "NOT_ALLOWED"
        assert decisions[key]["scientific_semantics_changed"] == "YES"


def test_report_numbers_match_projection_artifact(artifacts):
    report_path = PROJECT_ROOT / "docs" / "research" / "seed4_billing_grounded_cost_preflight.md"
    if not report_path.exists():
        from scripts.generate_seed4_cost_preflight_artifacts import main

        main()
    text = report_path.read_text(encoding="utf-8")
    expected = artifacts["projection"]["scenarios_usd"]["expected"]["total_usd"]
    assert f"${expected:.2f}" in text
    assert "WITHIN_10_USD" in text
