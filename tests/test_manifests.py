"""Tests for manifest validation, live vs synthetic flags, and environment metadata."""

from ccpt.data.config import DataConfig
from ccpt.data.hashing import sha256_json
from ccpt.data.manifests import build_task4_manifest


def test_synthetic_manifest_flag_invariants():
    """Verify that when wildguard is not live validated, manifest explicitly marks synthetic source."""
    config = DataConfig()
    fineweb_stats = {
        "smoke_document_count": 50,
        "smoke_token_count": 65000,
        "smoke_block_count": 63,
        "smoke_hash": "dummy_hash",
    }
    wildguard_stats = {
        "wildguard_live_validated": False,
        "statistics_source": "synthetic_fixture",
        "total_raw_rows": 7,
        "risk_train_count": 4,
        "risk_val_count": 0,
        "gen_train_count": 2,
        "gen_val_count": 0,
    }

    manifest = build_task4_manifest(config, fineweb_stats, wildguard_stats)

    assert manifest["wildguard_live_validated"] is False
    assert manifest["statistics_source"] == "synthetic_fixture"


def test_live_production_manifest_flag_invariants():
    """Verify that when wildguard is live validated, manifest explicitly marks pinned_live_dataset."""
    config = DataConfig()
    fineweb_stats = {
        "smoke_document_count": 100,
        "smoke_token_count": 130000,
        "smoke_block_count": 126,
        "smoke_hash": "dummy_hash",
    }
    wildguard_stats = {
        "wildguard_live_validated": True,
        "statistics_source": "pinned_live_dataset",
        "total_raw_rows": 1000,
        "risk_train_count": 800,
        "risk_val_count": 40,
        "gen_train_count": 700,
        "gen_val_count": 35,
    }
    env_meta = {
        "execution_environment": "modal",
        "modal_volume": "ccpt-data",
        "modal_volume_root": "/data/ccpt",
    }

    manifest = build_task4_manifest(config, fineweb_stats, wildguard_stats, execution_env=env_meta)

    assert manifest["wildguard_live_validated"] is True
    assert manifest["statistics_source"] == "pinned_live_dataset"
    assert manifest["execution_environment"] == "modal"
    assert manifest["modal_volume"] == "ccpt-data"
    assert manifest["modal_volume_root"] == "/data/ccpt"
    assert manifest["fineweb"]["production_target_blocks"] == 9_765_625


def test_evaluation_lock_in_manifest():
    """Verify evaluation split row count and logical hash are properly stored in manifest."""
    config = DataConfig()
    wildguard_stats = {
        "wildguard_live_validated": True,
        "statistics_source": "pinned_live_dataset",
        "total_raw_rows": 86759,
        "raw_eval_rows_count": 1725,
        "eval_records_count": 1699,
        "eval_hash": "94c8c5abc123",
    }
    manifest = build_task4_manifest(config, {}, wildguard_stats)
    assert manifest["wildguard"]["raw_eval_row_count"] == 1725
    assert manifest["wildguard"]["usable_eval_record_count"] == 1699
    assert manifest["wildguard"]["eval_logical_hash"] == "94c8c5abc123"

