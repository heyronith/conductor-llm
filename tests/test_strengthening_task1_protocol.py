"""Targeted tests asserting all protocol freeze invariants for CCPT Strengthening Round Task 1."""

import json
import hashlib
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs" / "research"
PROTOCOL_PATH = ARTIFACTS_DIR / "strengthening_task1_protocol.json"
PREFLIGHT_PATH = ARTIFACTS_DIR / "strengthening_task1_preflight.json"


@pytest.fixture(scope="module")
def protocol_data():
    """Load the machine-readable protocol specification."""
    assert PROTOCOL_PATH.exists(), f"Missing {PROTOCOL_PATH}"
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def preflight_data():
    """Load the preflight execution report."""
    assert PREFLIGHT_PATH.exists(), f"Missing {PREFLIGHT_PATH}"
    with open(PREFLIGHT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_seed_identities_and_reservation(protocol_data):
    """Assert the exact six primary seeds and verify 20260822 is strictly reserved."""
    seeds = protocol_data["seeds"]["primary_six_seed_cohort"]
    assert seeds == [20260821, 20260823, 20260824, 20260825, 20260826, 20260827]
    assert len(seeds) == 6
    assert len(set(seeds)) == 6

    # Reserved seed invariant
    reserved = protocol_data["seeds"]["reserved_seeds"]
    assert 20260822 in reserved
    assert 20260822 not in seeds, "CRITICAL ERROR: Reserved seed 20260822 cannot be in primary training cohort!"


def test_sentinel_seed_and_model_design(protocol_data):
    """Assert sentinel design contains exactly Seed 1 + Seed 4 for Models B, C, D."""
    sentinel_seeds = protocol_data["seeds"]["sentinel_seeds"]
    assert sentinel_seeds == [20260821, 20260825]
    assert len(sentinel_seeds) == 2

    # Verify models B, C, D exist in protocol
    models = protocol_data["models"]["architectures"]
    assert set(models.keys()) == {"model_b", "model_c", "model_d"}
    assert protocol_data["compute_budget"]["allocations"]["task2_sentinel"]["expected_models"] == 6


def test_model_class_resolution_and_parameter_parity(protocol_data):
    """Assert model classes resolve and parameter counts match authoritative numbers."""
    from ccpt.modeling.dual_stream import JointTrainingDualStreamModel, CCPTDualStreamModel
    from ccpt.modeling.adapter import FrozenBackboneAdapterModel
    from ccpt.config import get_smoke_dual_stream_config, get_smoke_adapter_config

    cfg_dual = get_smoke_dual_stream_config()
    cfg_ad = get_smoke_adapter_config()

    m_b = JointTrainingDualStreamModel(cfg_dual)
    m_c = CCPTDualStreamModel(cfg_dual)
    m_d = FrozenBackboneAdapterModel(cfg_ad)

    tot_b = sum(p.numel() for p in m_b.parameters())
    tot_c = sum(p.numel() for p in m_c.parameters())
    tot_d = sum(p.numel() for p in m_d.parameters())

    assert tot_b == 35_920_384
    assert tot_c == 35_920_384
    assert tot_d == 35_922_944

    proto_models = protocol_data["models"]["architectures"]
    assert proto_models["model_b"]["total_parameters"] == tot_b
    assert proto_models["model_c"]["total_parameters"] == tot_c
    assert proto_models["model_d"]["total_parameters"] == tot_d


def test_historical_task8_artifacts_byte_identical():
    """Assert that raw historical Task 7 & 8 artifacts are completely unchanged."""
    hashes = {
        "artifacts/task8_2_machine_tables.json": "1d91cc491ad17320d9be180aeda9954ae77b9243ddb92d901bb3dbde1486412e",
        "artifacts/task8_hypothesis_assessment.json": "29c0b2e16735630432b6b827426c4b9c02cd7ac74fe78214aaee42a1196bf47e",
        "artifacts/task7_3_1a_forensic_summary.json": "89dcebe8c7317631f8ca1eb432e65a58dd2eb60fa72defcf13178a5322777f61",
        "artifacts/task7_4_multiseed_replication_summary.json": "5a40b33a93b4334cae7e4037f637d3c88cbb865679b46072825cbf3f2ee2f377",
        "artifacts/task8_cka_summary.json": "e9200db454fed4a1640c48ffd0d818dca34d7f62c766b51a5c4d6047afd4ff17",
        "artifacts/task8_mechanistic_summary.json": "77faac51208115b4d8157a7fe937271e8793f0c582255e857b11c7cf4fa5a516",
    }
    for rel_p, exp_h in hashes.items():
        p = PROJECT_ROOT / rel_p
        assert p.exists(), f"Missing {rel_p}"
        with open(p, "rb") as f:
            actual_h = hashlib.sha256(f.read()).hexdigest()
        assert actual_h == exp_h, f"Hash mismatch for {rel_p}!"


def test_primary_endpoints_and_extended_persistence_curve(protocol_data):
    """Assert primary safety and persistence endpoints remain fixed and curve contains required steps."""
    prim = protocol_data["primary_experiment_specification"]
    assert prim["safety_training"]["primary_endpoint_tokens"] == 20_000_000
    assert prim["persistence_continuation"]["primary_endpoint_steps"] == 1000

    curve = prim["persistence_continuation"]["extended_curve_steps"]
    assert curve == [0, 250, 1000, 4000]


def test_hardware_specification_and_gpu_safety(protocol_data):
    """Assert training hardware is strictly 'Modal H100!' and eval is 'L40S'."""
    hw = protocol_data["hardware_matrix"]
    assert hw["training_and_persistence"] == "Modal H100!"
    assert hw["evaluation_and_judging"] == "L40S"
    assert hw["protocol_and_preflight"] == "CPU"


def test_compute_budget_limits(protocol_data):
    """Assert compute ceiling of $40.00 and sentinel gate of $14.00."""
    cb = protocol_data["compute_budget"]
    assert cb["hard_authorization_ceiling_usd"] == 40.0
    assert cb["allocations"]["task2_sentinel"]["hard_stop_gate_usd"] == 14.0
    assert cb["allocations"]["task2_sentinel"]["target_max_usd"] <= 12.0


def test_calibration_split_isolation(protocol_data):
    """Assert calibrated operating-point analysis cannot use final ID/OOD test splits."""
    op = protocol_data["operating_point_experiment"]
    assert op["status"] == "SECONDARY_SENSITIVITY_ANALYSIS"
    forbidden = op["forbidden_calibration_datasets"]
    assert any("WildGuard test" in f for f in forbidden)
    assert any("BeaverTails 30k OOD" in f for f in forbidden)
    assert any("XSTest" in f for f in forbidden)
    assert "WildGuard validation" in op["calibration_dataset"]


def test_preflight_artifact_validity(preflight_data):
    """Assert preflight artifact status is PASSED and zero GPU was used."""
    assert preflight_data["status"] == "PASSED"
    assert preflight_data["checks"]["hardware_safeguards"]["preflight_gpu_seconds_used"] == 0
    assert preflight_data["checks"]["seeds"]["reserved_seed_safeguard"] == "PASSED"
