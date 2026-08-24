"""Hard failure tests for Task 8.1 behavioral join, transitions, and synthesis corrections."""

import sys
import json
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_task8_figures_and_tables import load_authoritative_behavioral_data
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def test_authoritative_primary_values():
    """Verify exact numerical primary values across all three seeds."""
    behavior = load_authoritative_behavioral_data()

    # Seed 1
    c_pre_s1 = behavior["20260821"]["model_c_pre_persistence_on_harmful"]["rate"]
    c_post_s1 = behavior["20260821"]["model_c_post_persistence_on_harmful"]["rate"]
    d_pre_s1 = behavior["20260821"]["model_d_pre_persistence_on_harmful"]["rate"]
    d_post_s1 = behavior["20260821"]["model_d_post_persistence_on_harmful"]["rate"]

    assert np.isclose(c_pre_s1, 0.87500000)
    assert np.isclose(c_post_s1, 0.86328125)
    assert np.isclose(d_pre_s1, 0.93359375)
    assert np.isclose(d_post_s1, 0.51171875)

    delta_c_s1 = c_post_s1 - c_pre_s1
    delta_d_s1 = d_post_s1 - d_pre_s1
    assert np.isclose(delta_c_s1, -0.01171875)
    assert np.isclose(delta_d_s1, -0.42187500)
    assert np.isclose(delta_c_s1 - delta_d_s1, 0.41015625)

    # Seed 2
    c_pre_s2 = behavior["20260823"]["model_c_pre_persistence_on_harmful"]["rate"]
    c_post_s2 = behavior["20260823"]["model_c_post_persistence_on_harmful"]["rate"]
    d_pre_s2 = behavior["20260823"]["model_d_pre_persistence_on_harmful"]["rate"]
    d_post_s2 = behavior["20260823"]["model_d_post_persistence_on_harmful"]["rate"]

    assert np.isclose(c_pre_s2, 220 / 256)
    assert np.isclose(c_post_s2, 173 / 256)
    assert np.isclose(d_pre_s2, 238 / 256)
    assert np.isclose(d_post_s2, 227 / 256)

    delta_c_s2 = c_post_s2 - c_pre_s2
    delta_d_s2 = d_post_s2 - d_pre_s2
    assert np.isclose(delta_c_s2, -0.18359375)
    assert np.isclose(delta_d_s2, -0.04296875)
    assert np.isclose(delta_c_s2 - delta_d_s2, -0.14062500)

    # Seed 3
    c_pre_s3 = behavior["20260824"]["model_c_pre_persistence_on_harmful"]["rate"]
    c_post_s3 = behavior["20260824"]["model_c_post_persistence_on_harmful"]["rate"]
    d_pre_s3 = behavior["20260824"]["model_d_pre_persistence_on_harmful"]["rate"]
    d_post_s3 = behavior["20260824"]["model_d_post_persistence_on_harmful"]["rate"]

    assert np.isclose(c_pre_s3, 171 / 256)
    assert np.isclose(c_post_s3, 201 / 256)
    assert np.isclose(d_pre_s3, 246 / 256)
    assert np.isclose(d_post_s3, 219 / 256)

    delta_c_s3 = c_post_s3 - c_pre_s3
    delta_d_s3 = d_post_s3 - d_pre_s3
    assert np.isclose(delta_c_s3, 0.11718750)
    assert np.isclose(delta_d_s3, -0.10546875)
    assert np.isclose(delta_c_s3 - delta_d_s3, 0.22265625)


def test_transition_count_mathematical_reconciliation():
    """Verify transition counts reconcile exactly with pre and post yes counts."""
    trans_p = ARTIFACTS_DIR / "task8_transition_group_summary.json"
    assert trans_p.exists()

    with open(trans_p, "r", encoding="utf-8") as f:
        t_data = json.load(f)

    # Seed 2 Model C
    s2 = t_data["20260823"]
    assert s2["retained_refusal"]["count"] == 163
    assert s2["lost_refusal"]["count"] == 57
    assert s2["gained_refusal"]["count"] == 10
    assert s2["persistent_nonrefusal"]["count"] == 26

    assert s2["retained_refusal"]["count"] + s2["lost_refusal"]["count"] == 220
    assert s2["retained_refusal"]["count"] + s2["gained_refusal"]["count"] == 173

    # Seed 3 Model C
    s3 = t_data["20260824"]
    assert s3["retained_refusal"]["count"] == 158
    assert s3["lost_refusal"]["count"] == 13
    assert s3["gained_refusal"]["count"] == 43
    assert s3["persistent_nonrefusal"]["count"] == 42

    assert s3["retained_refusal"]["count"] + s3["lost_refusal"]["count"] == 171
    assert s3["retained_refusal"]["count"] + s3["gained_refusal"]["count"] == 201


def test_no_hardcoded_behavioral_arrays_in_script():
    """Static inspection ensuring no hardcoded scientific refusal rate arrays exist in synthesis script."""
    script_p = PROJECT_ROOT / "scripts" / "generate_task8_figures_and_tables.py"
    with open(script_p, "r", encoding="utf-8") as f:
        code = f.read()

    forbidden_patterns = [
        "active_pre = [",
        "active_post = [",
        "off_pre = [0.0039",
        "pre_refusal = [80.08",
        "primary_effect = [41.02",
        "retention_c = [80.00",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in code, f"Forbidden hardcoded behavioral pattern found: '{pattern}'"


def test_h1_guard():
    """Verify that H1 assessment does not falsely claim Seed 2 capability drift > Seed 1."""
    hyp_p = ARTIFACTS_DIR / "task8_hypothesis_assessment.json"
    assert hyp_p.exists()

    with open(hyp_p, "r", encoding="utf-8") as f:
        h_data = json.load(f)

    h1 = h_data["H1_capability_interface_drift"]
    assert h1["status"] in ("INCONCLUSIVE", "NOT_CONSISTENT_WITH")
    assert "Seed 2 (the negative persistence seed) does NOT exhibit greater capability proposal drift" in h1["evidence_against"]
