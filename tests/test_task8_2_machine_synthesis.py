"""Task 8.2: Automated tests for machine-derived mechanistic synthesis parity, CKA consistency, and NA sensitivity."""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_MECHANISTIC_SHA256 = "77faac51208115b4d8157a7fe937271e8793f0c582255e857b11c7cf4fa5a516"


def test_raw_mechanistic_artifact_immutability():
    """Ensure raw diagnostic artifacts have not been modified or re-extracted."""
    summary_p = ARTIFACTS_DIR / "task8_mechanistic_summary.json"
    assert summary_p.exists()

    with open(summary_p, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha == RAW_MECHANISTIC_SHA256, f"Raw mechanistic artifact changed! {actual_sha} != {RAW_MECHANISTIC_SHA256}"


def test_cka_exact_parity():
    """Verify that all CKA fields in task8_2_machine_tables.json match task8_cka_summary.json exactly."""
    tables_p = ARTIFACTS_DIR / "task8_2_machine_tables.json"
    cka_p = ARTIFACTS_DIR / "task8_cka_summary.json"

    assert tables_p.exists() and cka_p.exists()

    with open(tables_p, "r", encoding="utf-8") as f:
        tables = json.load(f)

    with open(cka_p, "r", encoding="utf-8") as f:
        cka = json.load(f)

    # Check Model C
    seeds = ["20260821", "20260823", "20260824"]
    for s in seeds:
        for l in [2, 4]:
            t_b = tables["table_b_model_c_drift"][s][f"layer_{l}"]
            assert np.isclose(t_b["capability_linear_cka"], cka[f"seed_{s}_model_c_ood_beavertails_harmful_c_tilde_{l}"])
            assert np.isclose(t_b["obs_linear_cka"], cka[f"seed_{s}_model_c_ood_beavertails_harmful_obs_{l}"])
            assert np.isclose(t_b["normative_linear_cka"], cka[f"seed_{s}_model_c_ood_beavertails_harmful_norm_{l}"])
            assert np.isclose(t_b["steering_linear_cka"], cka[f"seed_{s}_model_c_ood_beavertails_harmful_steer_{l}"])

    # Check Model D (all 8 sites, in and res)
    for s in seeds:
        for l_idx in range(4):
            for a_type in ["attn", "mlp"]:
                site_name = f"layer_{l_idx}_{a_type}_adapter"
                t_d = tables["table_d_model_d_adapter_drift"][s][site_name]
                assert np.isclose(t_d["input_linear_cka"], cka[f"seed_{s}_model_d_ood_beavertails_harmful_{site_name}_in"])
                assert np.isclose(t_d["residual_linear_cka"], cka[f"seed_{s}_model_d_ood_beavertails_harmful_{site_name}_res"])


def test_ablation_gap_na_sensitivity_sign_stability():
    """Verify that NA bounds do not flip the sign of the active/off ablation gap change for H3."""
    tables_p = ARTIFACTS_DIR / "task8_2_machine_tables.json"
    with open(tables_p, "r", encoding="utf-8") as f:
        tables = json.load(f)

    sens = tables["ablation_sensitivity"]

    # Seed 2 must remain strictly negative across all NA bounds
    s2 = sens["20260823"]
    assert s2["ablation_gap_change_determinate"] < 0
    assert s2["sensitivity_a_all_na_refusal"]["gap_change"] < 0
    assert s2["sensitivity_b_all_na_nonrefusal"]["gap_change"] < 0
    assert s2["sign_stable_across_bounds"] is True

    # Seed 1 and Seed 3 must remain strictly positive across all NA bounds
    s1 = sens["20260821"]
    assert s1["ablation_gap_change_determinate"] > 0
    assert s1["sensitivity_a_all_na_refusal"]["gap_change"] > 0
    assert s1["sensitivity_b_all_na_nonrefusal"]["gap_change"] > 0
    assert s1["sign_stable_across_bounds"] is True

    s3 = sens["20260824"]
    assert s3["ablation_gap_change_determinate"] > 0
    assert s3["sensitivity_a_all_na_refusal"]["gap_change"] > 0
    assert s3["sensitivity_b_all_na_nonrefusal"]["gap_change"] > 0
    assert s3["sign_stable_across_bounds"] is True


def test_seed1_behavior_join_partial_qualification():
    """Ensure Seed 1 behavior join is explicitly acknowledged as partial due to lack of per-prompt records."""
    report_p = PROJECT_ROOT / "docs" / "research" / "task8_mechanistic_heterogeneity_report.md"
    assert report_p.exists()
    with open(report_p, "r", encoding="utf-8") as f:
        text = f.read()

    assert "SEED1_BEHAVIOR_JOIN = PARTIAL" in text or "Seed 1 behavior join: PARTIAL" in text or "Aggregate provenance" in text


def test_hypothesis_assessment_numerical_values_match_machine_tables():
    """Verify that all numerical values quoted in task8_hypothesis_assessment.json match task8_2_machine_tables.json."""
    hyp_p = ARTIFACTS_DIR / "task8_hypothesis_assessment.json"
    tables_p = ARTIFACTS_DIR / "task8_2_machine_tables.json"

    assert hyp_p.exists() and tables_p.exists()

    with open(hyp_p, "r", encoding="utf-8") as f:
        hyp = json.load(f)

    with open(tables_p, "r", encoding="utf-8") as f:
        tables = json.load(f)

    tb = tables["table_b_model_c_drift"]
    ta = tables["table_a_behavior"]
    sens = tables["ablation_sensitivity"]

    # Verify H1 statuses and quoted values
    h1_text = hyp["H1_capability_interface_drift"]["evidence_against"]
    assert f"{tb['20260823']['layer_4']['capability_linear_cka']:.4f}" in h1_text
    assert f"{tb['20260821']['layer_4']['capability_linear_cka']:.4f}" in h1_text
    assert f"{tb['20260824']['layer_4']['capability_linear_cka']:.4f}" in h1_text
    assert f"{tb['20260823']['layer_4']['capability_relative_l2_mean']:.4f}" in h1_text
    assert f"{tb['20260821']['layer_4']['capability_relative_l2_mean']:.4f}" in h1_text
    assert f"{tb['20260823']['layer_4']['obs_linear_cka']:.4f}" in h1_text
    assert f"{tb['20260823']['layer_4']['obs_relative_l2_mean']:.4f}" in h1_text

    # Verify H2 statuses and quoted values
    h2_for = hyp["H2_functional_controller_drift"]["evidence_for"]
    h2_against = hyp["H2_functional_controller_drift"]["evidence_against"]
    assert f"{tb['20260823']['layer_4']['gate_absolute_change_mean']:.4f}" in h2_for
    assert f"{tb['20260821']['layer_4']['gate_absolute_change_mean']:.4f}" in h2_for
    assert f"{tb['20260824']['layer_4']['gate_absolute_change_mean']:.4f}" in h2_for
    assert f"{tb['20260823']['layer_4']['normative_linear_cka']:.4f}" in h2_against
    assert f"{tb['20260823']['layer_4']['steering_linear_cka']:.4f}" in h2_against

    # Verify H3 statuses and quoted values
    h3_for = hyp["H3_downstream_override_effect_loss"]["evidence_for"]
    assert f"{sens['20260823']['ablation_gap_change_determinate'] * 100.0:+.2f}" in h3_for
    assert f"{sens['20260824']['ablation_gap_change_determinate'] * 100.0:+.2f}" in h3_for
    assert f"{sens['20260821']['ablation_gap_change_determinate'] * 100.0:+.2f}" in h3_for

    # Verify H4 statuses and quoted values
    h4_for = hyp["H4_safety_acquisition_quality_selectivity"]["evidence_for"]
    h4_against = hyp["H4_safety_acquisition_quality_selectivity"]["evidence_against"]
    assert f"{ta['20260821']['c_pre_refusal_rate'] * 100.0:.2f}" in h4_for
    assert f"{ta['20260823']['c_pre_refusal_rate'] * 100.0:.2f}" in h4_for
    assert f"{ta['20260824']['c_pre_refusal_rate'] * 100.0:.2f}" in h4_for
    assert f"{ta['20260824']['c_retention_delta_pp']:+.2f}" in h4_against

    # Verify H5 statuses and quoted values
    h5_against = hyp["H5_generic_frozen_module_interface"]["evidence_against"]
    assert f"{ta['20260823']['d_retention_delta_pp']:+.2f}" in h5_against
    assert f"{ta['20260821']['d_retention_delta_pp']:+.2f}" in h5_against
    assert f"{ta['20260824']['d_retention_delta_pp']:+.2f}" in h5_against

    # Verify all hypothesis statuses
    assert hyp["H1_capability_interface_drift"]["status"] == "INCONCLUSIVE"
    assert hyp["H2_functional_controller_drift"]["status"] == "INCONCLUSIVE"
    assert hyp["H3_downstream_override_effect_loss"]["status"] == "CONSISTENT_WITH"
    assert hyp["H4_safety_acquisition_quality_selectivity"]["status"] == "INCONCLUSIVE"
    assert hyp["H5_generic_frozen_module_interface"]["status"] == "INCONCLUSIVE"
