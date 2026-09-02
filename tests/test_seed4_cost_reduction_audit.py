"""Tests for Seed 4 cost-reduction forensic audit."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
TASK32_RECON = ARTIFACTS / "strengthening_task3_2_reconciliation.json"


@pytest.fixture(scope="module")
def artifacts():
    ledger_path = ARTIFACTS / "seed4_cost_reduction_task2_object_ledger.json"
    if not ledger_path.exists():
        from ccpt.analysis.seed4_cost_reduction_audit import write_all_artifacts

        write_all_artifacts(PROJECT_ROOT)
    return {
        "ledger": json.loads(ledger_path.read_text()),
        "reuse": json.loads((ARTIFACTS / "seed4_cost_reduction_checkpoint_reuse_audit.json").read_text()),
        "waste": json.loads((ARTIFACTS / "seed4_cost_reduction_waste_analysis.json").read_text()),
        "plan": json.loads((ARTIFACTS / "seed4_cost_reduction_execution_plan.json").read_text()),
        "projection": json.loads((ARTIFACTS / "seed4_cost_reduction_projection.json").read_text()),
        "runtime": json.loads(
            (ARTIFACTS / "seed4_cost_reduction_runtime_optimization_review.json").read_text()
        ),
    }


def test_h100_ledger_reconciles_modal_billing(artifacts):
    recon = artifacts["ledger"]["reconciliation"]
    assert recon["ledger_reconciliation"] == "PASS"
    assert recon["full_window_match"] is True
    assert recon["sep1_match"] is True
    assert abs(recon["observed_full_window_h100_usd"] - 25.83683913) < 1e-8
    assert abs(recon["observed_sep1_h100_usd"] - 22.80192289) < 1e-8


def test_every_h100_object_classified(artifacts):
    classes = {
        "REQUIRED_SUCCESSFUL_SCIENTIFIC_WORK",
        "RETRY_AFTER_INFRA_FAILURE",
        "FAILED_OR_ABORTED_EXECUTION",
        "DUPLICATE_EXECUTION",
        "GPU_SMOKE_OR_PREFLIGHT",
        "GPU_IDLE_OR_SETUP",
        "VALID_BUT_NOT_SEED4_REQUIRED",
        "UNRESOLVED",
    }
    h100_objects = [o for o in artifacts["ledger"]["objects"] if o["h100_cost_usd"] > 0]
    assert h100_objects
    for obj in h100_objects:
        assert obj["classification"]["primary_classification"] in classes


def test_no_unresolved_h100_dollars(artifacts):
    assert artifacts["ledger"]["reconciliation"]["unresolved_h100_usd"] == 0.0
    by_class = artifacts["ledger"]["totals"]["h100_by_classification_usd"]
    summed = round(sum(by_class.values()), 8)
    assert abs(summed - artifacts["ledger"]["totals"]["full_window_h100_cost_usd"]) < 1e-8


def test_only_high_confidence_avoidable_in_proven_savings(artifacts):
    waste = artifacts["waste"]
    proven_h100 = waste["guaranteed_high_confidence_savings_usd"]["h100"]
    medium = waste["medium_confidence_not_in_expected_budget_usd"]["h100"]
    assert medium > 0
    # Expected budget uses TaUU only — not historical full H100 minus medium
    clean_h100 = artifacts["projection"]["scenarios_usd"]["B_clean_protocol_preserving"]["h100_usd"]
    tauu = next(
        o["h100_cost_usd"]
        for o in artifacts["ledger"]["objects"]
        if o["object_id"] == "ap-TaUUJJEc7NPvKK0oya8ClI"
    )
    assert clean_h100 == tauu
    assert proven_h100 < medium  # HIGH savings smaller than MEDIUM pool


def test_speculative_savings_excluded_from_expected(artifacts):
    runtime = artifacts["runtime"]
    assert runtime["cuda_graphs"]["classification"] == "REJECT_FOR_SEED4"
    assert runtime["maximum_proposed_benchmark_cost_usd"] <= 0.30
    # Clean expected must not embed compile/graph dollar claims
    assumptions = " ".join(
        artifacts["projection"]["scenarios_usd"]["B_clean_protocol_preserving"]["assumptions"]
    )
    assert "Speculative" in assumptions or "excluded" in assumptions.lower()


def test_scientific_invariants_intact(artifacts):
    inv = artifacts["plan"]["scientific_invariants"]
    assert inv["models_required"] == ["model_b", "model_c", "model_d"]
    assert inv["lm_1b_required"] is True
    assert inv["safety_tokens_required"] == 20_010_611
    assert inv["persistence_steps_required"] == [0, 250, 1000, 4000]
    assert inv["h100_required"] is True
    assert inv["precision_unchanged"] is True
    assert inv["corrected_framed_evaluation_required"] is True
    assert inv["scientific_semantics_changed"] is False


def test_remaining_modal_credits_zero(artifacts):
    assert artifacts["ledger"]["remaining_modal_credits_usd"] == 0.0
    assert artifacts["projection"]["remaining_modal_credits_usd"] == 0.0


def test_preflight_allocates_zero_gpu():
    mod = importlib.import_module("ccpt.analysis.seed4_cost_reduction_audit")
    assert mod.H100_GPU_SECONDS_AUTHORIZED == 0
    assert mod.L40S_GPU_SECONDS_AUTHORIZED == 0
    source = inspect.getsource(mod)
    assert 'gpu="' not in source
    assert "gpu='H100" not in source
    assert "torch.cuda" not in source
    assert "modal.gpu" not in source
    assert "import torch" not in source


def test_task32_evidence_unchanged():
    before = hashlib.sha256(TASK32_RECON.read_bytes()).hexdigest()
    from ccpt.analysis.seed4_cost_reduction_audit import write_all_artifacts

    write_all_artifacts(PROJECT_ROOT)
    after = hashlib.sha256(TASK32_RECON.read_bytes()).hexdigest()
    assert before == after


def test_report_derives_from_projection(artifacts):
    report = PROJECT_ROOT / "docs" / "research" / "seed4_cost_reduction_forensic_audit.md"
    if not report.exists():
        import importlib.util

        script = PROJECT_ROOT / "scripts" / "generate_seed4_cost_reduction_audit.py"
        spec = importlib.util.spec_from_file_location("gen_seed4_cost_reduction", script)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.main()
    text = report.read_text(encoding="utf-8")
    expected = artifacts["projection"]["new_expected_out_of_pocket_usd"]
    assert f"${expected:.2f}" in text
    assert artifacts["ledger"]["reconciliation"]["ledger_reconciliation"] == "PASS"
