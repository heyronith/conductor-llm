"""Targeted tests for CCPT Strengthening Round Task 2 (Sentinel Execution)."""

import json
from pathlib import Path
import pytest
import torch

from ccpt.config import (
    get_micro_adapter_config,
    get_micro_dual_stream_config,
    get_smoke_adapter_config,
    get_smoke_dual_stream_config,
)
from ccpt.data.strengthening_materializer import (
    AUTHORITATIVE_FIRST_32K_SHARDS,
    CANONICAL_FINEWEB_CONT_HASH,
    CANONICAL_FINEWEB_PREFIX_HASH,
)
from ccpt.evaluation.forensics import compute_canonical_state_dict_hash
from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
from ccpt.training.engine import create_identical_dual_stream_models
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_strengthening_task2_preflight import (
    ALLOWED_SEEDS,
    RESERVED_SEED,
    run_preflight,
)


def test_fineweb_continuation_manifest_integrity():
    """Verify the materialized strengthening FineWeb continuation manifest."""
    manifest_p = Path("artifacts/strengthening_task2_extended_fineweb_manifest.json")
    assert manifest_p.exists(), f"Missing {manifest_p}"
    with open(manifest_p, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Prefix
    prefix = meta["capability_prefix"]
    assert prefix["logical_prefix_hash"] == CANONICAL_FINEWEB_PREFIX_HASH
    assert prefix["target_blocks"] == 976544
    assert prefix["target_tokens"] == 999981056

    # First 32k parity
    orig = meta["original_persistence_continuation"]
    assert orig["target_blocks"] == 32000
    assert orig["target_tokens"] == 32768000
    assert orig["logical_continuation_hash"] == CANONICAL_FINEWEB_CONT_HASH
    assert orig["first_32k_parity"] == "BIT_IDENTICAL"

    # Extended 128k continuation
    ext = meta["persistence_continuation"]
    assert ext["target_blocks"] == 128000
    assert ext["target_tokens"] == 131072000
    assert ext["first_32k_parity"] == "BIT_IDENTICAL"
    assert len(ext["shards"]) == 16

    # Shard sequence continuity
    cur_block = 976544
    for idx, s in enumerate(ext["shards"]):
        assert s["logical_first_block"] == cur_block
        assert s["logical_last_block_exclusive"] == cur_block + s["num_blocks"]
        cur_block += s["num_blocks"]
    assert cur_block == 1104544


def test_seed_safety_constraints():
    """Verify that reserved evaluation seed 20260822 cannot be trained."""
    assert RESERVED_SEED == 20260822
    assert RESERVED_SEED not in ALLOWED_SEEDS
    assert set(ALLOWED_SEEDS) == {20260821, 20260825}


def test_model_b_c_initialization_parity_both_seeds():
    """Verify that Model B and Model C start bit-identical for both sentinel seeds."""
    cfg = get_smoke_dual_stream_config()
    for seed in ALLOWED_SEEDS:
        mb, mc = create_identical_dual_stream_models(cfg, seed=seed)
        hash_b = compute_canonical_state_dict_hash(mb.state_dict())
        hash_c = compute_canonical_state_dict_hash(mc.state_dict())
        assert hash_b == hash_c, f"Seed {seed} initialization mismatch: {hash_b} != {hash_c}"

        # Test parameter values directly
        for pb, pc in zip(mb.parameters(), mc.parameters()):
            assert torch.equal(pb, pc)


def test_model_parameter_ownership_partitions():
    """Verify strict disjoint parameter ownership partitions for Model C and Model D."""
    dual_cfg = get_micro_dual_stream_config()
    mc = CCPTDualStreamModel(dual_cfg)

    # Model C: theta_C and theta_N
    c_set = set(mc.theta_C)
    n_set = set(mc.theta_N)
    all_c = set(mc.parameters())
    assert len(c_set.intersection(n_set)) == 0
    assert c_set.union(n_set) == all_c

    # Model D: backbone and safety
    adapter_cfg = get_micro_adapter_config()
    md = FrozenBackboneAdapterModel(adapter_cfg)
    b_set = set(md.backbone_parameters)
    s_set = set(md.safety_parameters)
    all_d = set(md.parameters())
    assert len(b_set.intersection(s_set)) == 0
    assert b_set.union(s_set) == all_d


def test_task2_preflight_passes():
    """Verify that the Task 2 fail-closed preflight executes and passes cleanly."""
    res = run_preflight()
    assert res["overall_status"] == "PASSED"
    for check_name, check_data in res["checks"].items():
        assert check_data["status"] == "PASSED", f"Check {check_name} failed: {check_data}"


def test_modal_sentinel_gpu_spec_integrity():
    """Verify that Modal training jobs specify H100! and evaluation specifies L40S."""
    src_p = Path("modal/strengthening_task2_sentinel.py")
    assert src_p.exists()
    content = src_p.read_text(encoding="utf-8")
    assert 'gpu="H100!"' in content
    assert 'gpu="L40S"' in content
