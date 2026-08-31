"""Targeted tests asserting all protocol freeze invariants for CCPT Strengthening Round Task 1 & 1.1."""

import json
import hashlib
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs" / "research"
PROTOCOL_PATH = ARTIFACTS_DIR / "strengthening_task1_protocol.json"
PREFLIGHT_PATH = ARTIFACTS_DIR / "strengthening_task1_preflight.json"
CALIBRATION_MANIFEST_PATH = ARTIFACTS_DIR / "strengthening_calibration_prompt_manifest.json"


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


@pytest.fixture(scope="module")
def calibration_data():
    """Load the calibration prompt manifest."""
    assert CALIBRATION_MANIFEST_PATH.exists(), f"Missing {CALIBRATION_MANIFEST_PATH}"
    with open(CALIBRATION_MANIFEST_PATH, "r", encoding="utf-8") as f:
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


def test_persistence_token_parity(protocol_data):
    """Assert exact token count: 1000 steps = 32,768,000 tokens, 4000 steps = 131,072,000 tokens."""
    from ccpt.data.canonical_materializer import TARGET_PERSISTENCE_BLOCKS

    tokens_1000 = TARGET_PERSISTENCE_BLOCKS * 1024
    assert tokens_1000 == 32_768_000, f"Expected 32,768,000 tokens, got {tokens_1000}"

    tokens_4000 = 4000 * 32 * 1024
    assert tokens_4000 == 131_072_000, f"Expected 131,072,000 tokens, got {tokens_4000}"

    prim = protocol_data["primary_experiment_specification"]
    assert prim["persistence_continuation"]["primary_endpoint_tokens"] == 32_768_000
    assert prim["persistence_continuation"]["extended_curve_tokens"] == [0, 8192000, 32768000, 131072000]


def test_authoritative_task7_4_parity():
    """Assert agreement with authoritative Task-7.4 constants, manifests, and invariants."""
    from ccpt.data.canonical_materializer import TARGET_TRAIN_PREFIX_BLOCKS, TARGET_PERSISTENCE_BLOCKS
    from ccpt.data.wildguard import CANONICAL_TASK4_MANIFEST_HASH
    from ccpt.training.engine import create_identical_dual_stream_models
    from ccpt.config import get_smoke_dual_stream_config

    assert TARGET_TRAIN_PREFIX_BLOCKS == 976_544
    assert TARGET_TRAIN_PREFIX_BLOCKS * 1024 == 999_981_056
    assert TARGET_PERSISTENCE_BLOCKS == 32_000
    assert CANONICAL_TASK4_MANIFEST_HASH == "2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097"

    cfg = get_smoke_dual_stream_config()
    mb, mc = create_identical_dual_stream_models(cfg, seed=20260821)
    for (kb, pb), (kc, pc) in zip(mb.state_dict().items(), mc.state_dict().items()):
        assert kb == kc and (pb == pc).all(), f"Model B/C initialization divergence on {kb}"


def test_protocol_markdown_parity(protocol_data):
    """Assert that markdown document exactly matches machine-readable protocol values."""
    doc_path = DOCS_DIR / "strengthening_task1_protocol.md"
    assert doc_path.exists()
    text = doc_path.read_text(encoding="utf-8")

    # Seeds
    for s in protocol_data["seeds"]["primary_six_seed_cohort"]:
        assert str(s) in text
    assert str(protocol_data["seeds"]["reserved_seeds"][0]) in text

    # Model parameters & names
    assert "JointTrainingDualStreamModel" in text
    assert "CCPTDualStreamModel" in text
    assert "FrozenBackboneAdapterModel" in text
    assert "35,920,384" in text
    assert "35,922,944" in text

    # Exact token counts
    assert "32,768,000" in text
    assert "131,072,000" in text
    assert "~2.0M tokens" not in text

    # Hardware & Budget
    assert "Modal H100!" in text
    assert "L40S" in text
    assert "$40.00" in text
    assert "$14.00" in text


def test_calibration_manifest_integrity_and_isolation(calibration_data):
    """Assert calibration prompt set has exact record count, logical hash, and 0 test overlap."""
    from ccpt.data.hashing import sha256_json

    records = calibration_data["records"]
    assert len(records) == 2335
    assert calibration_data["deduplication_and_filtering_policy"]["harmful_records_count"] == 1189
    assert calibration_data["deduplication_and_filtering_policy"]["benign_records_count"] == 1146

    computed_hash = sha256_json(records)
    assert computed_hash == "e39be5aed40e698d12b5132980c208ff68ad7208501fcd918ceae1011491ef7d"

    audit = calibration_data["test_isolation_audit"]
    assert audit["wildguard_test_overlap_count"] == 0
    assert audit["beavertails_30k_test_overlap_count"] == 0
    assert audit["xstest_overlap_count"] == 0
    assert audit["isolation_status"] == "PASSED_ZERO_OVERLAP"


def test_pinned_xstest_composition_and_source(protocol_data):
    """Assert pinned XSTest dataset composition (250 safe, 200 unsafe, 450 total) and immutable source."""
    xs = protocol_data["evaluation_and_benchmarks"]["over_refusal_benchmark"]
    assert xs["benchmark"] == "XSTest"
    assert xs["total_prompts"] == 450
    assert xs["safe_prompts"] == 250
    assert xs["contrast_unsafe_prompts"] == 200

    pinned = xs["pinned_dataset"]
    assert pinned["repo"] == "natolambert/xstest-v2-copy"
    assert pinned["revision"] == "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5"
    assert pinned["file"] == "data/prompts-00000-of-00001.parquet"
    assert pinned["sha256"] == "322d4e89df9fb419c296d5b360067f3265845d40a561a37d9be77a078d219522"


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


def test_preflight_artifact_validity(preflight_data):
    """Assert preflight artifact status is PASSED and zero GPU was used."""
    assert preflight_data["status"] == "PASSED"
    assert preflight_data["checks"]["hardware_safeguards"]["preflight_gpu_seconds_used"] == 0
    assert preflight_data["checks"]["seeds"]["reserved_seed_safeguard"] == "PASSED"
