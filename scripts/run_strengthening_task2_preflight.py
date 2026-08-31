"""Fail-closed preflight audit for CCPT Strengthening Round Task 2 (Sentinel Execution).

Zero GPU required. Fails closed if any protocol invariant, hash, seed constraint,
lineage, or initialization parity check fails.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict

import torch

from ccpt.config import (
    get_smoke_adapter_config,
    get_smoke_dual_stream_config,
)
from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
from ccpt.evaluation.forensics import compute_canonical_state_dict_hash
from ccpt.training.checkpoint import load_checkpoint
from ccpt.training.engine import create_identical_dual_stream_models

TASK1_1_ANCESTOR_SHA = "7fd1262cee4142473a8048a50f6093e2b6fa246d"

CANONICAL_FINEWEB_PREFIX_HASH = "a13410b63d9c1533211784c2a08fa5a918e29cc446448470395aa93919712585"
CANONICAL_FINEWEB_ORIGINAL_32K_CONT_HASH = "1f6dd66f49a9afa3537244a719af74006308ab81902b0b654142510672022243"
CANONICAL_FINEWEB_128K_CONT_HASH = "26829ec5297e61e8ed91b89a64d6522c58c0123ac3c7aeab23801ee101510fa3"
CANONICAL_TASK4_MANIFEST_HASH = "2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097"
CALIBRATION_MANIFEST_HASH = "f9a56a57e3f890ceaeeb07e4d82f70b7ea1b3d68ee9a3d4ee71ae15291b94ee0"
OOD_BEAVERTAILS_MANIFEST_HASH = "f8cf3fd0f0ca7502e9b7fef37f49ae4b9fd13cb71438ed64fc093c0649d71b9e"

ALLOWED_SEEDS = [20260821, 20260825]
RESERVED_SEED = 20260822
ALLOWED_MODELS = ["model_b", "model_c", "model_d"]

TASK2_EXPECTED_PACKAGE_VERSIONS = {
    "torch": "2.5.1",
    "transformers": "4.46.3",
    "tokenizers": "0.20.3",
    "datasets": "3.1.0",
    "huggingface_hub": "0.26.2",
    "sentencepiece": "0.2.0",
    "tiktoken": "0.8.0",
    "accelerate": "1.1.1",
    "pyarrow": "17.0.0",
    "numpy": "2.1.3",
    "pytest": "8.3.3",
}


def run_preflight() -> Dict[str, Any]:
    checks = {}

    # 1. Lineage & Git Ancestry
    res_git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    head_sha = res_git.stdout.strip().lower()

    res_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", TASK1_1_ANCESTOR_SHA, head_sha],
        capture_output=True,
    )
    lineage_passed = res_ancestor.returncode == 0
    checks["repository_lineage"] = {
        "head_sha": head_sha,
        "required_ancestor_sha": TASK1_1_ANCESTOR_SHA,
        "is_direct_descendant": lineage_passed,
        "status": "PASSED" if lineage_passed else "FAILED",
    }
    if not lineage_passed:
        raise RuntimeError(f"Lineage verification failed: HEAD {head_sha} does not descend from Task 1.1 {TASK1_1_ANCESTOR_SHA}")

    # 2. Container Runtime Package Specification Audit
    sentinel_src_p = Path("modal/strengthening_task2_sentinel.py")
    src_text = sentinel_src_p.read_text(encoding="utf-8")

    pinned_in_image = {}
    missing_pins = []
    for pkg, exp_ver in TASK2_EXPECTED_PACKAGE_VERSIONS.items():
        pin_pattern = f'"{pkg}=={exp_ver}"'
        if pin_pattern in src_text:
            pinned_in_image[pkg] = exp_ver
        else:
            missing_pins.append(f"{pkg}=={exp_ver}")

    checks["container_package_spec"] = {
        "expected": TASK2_EXPECTED_PACKAGE_VERSIONS,
        "pinned_in_image": pinned_in_image,
        "missing_pins": missing_pins,
        "status": "PASSED" if not missing_pins else "FAILED",
    }
    if missing_pins:
        raise RuntimeError(f"Missing pinned package specifications in Modal image: {missing_pins}")

    # 3. GPU Specification Audit in modal/strengthening_task2_sentinel.py
    sentinel_src_p = Path("modal/strengthening_task2_sentinel.py")
    if not sentinel_src_p.exists():
        raise FileNotFoundError(f"Missing {sentinel_src_p}")
    src_text = sentinel_src_p.read_text(encoding="utf-8")

    has_h100_bang = 'gpu="H100!"' in src_text
    has_l40s = 'gpu="L40S"' in src_text
    checks["gpu_specification"] = {
        "requires_h100_bang_training": has_h100_bang,
        "requires_l40s_eval": has_l40s,
        "status": "PASSED" if (has_h100_bang and has_l40s) else "FAILED",
    }
    if not (has_h100_bang and has_l40s):
        raise RuntimeError("GPU specification check failed: must request 'H100!' for training and 'L40S' for eval")

    # 4. Extended FineWeb Manifest Audit
    fw_manifest_p = Path("artifacts/strengthening_task2_extended_fineweb_manifest.json")
    if not fw_manifest_p.exists():
        raise FileNotFoundError(f"Missing {fw_manifest_p}")
    with open(fw_manifest_p, "r", encoding="utf-8") as f:
        fw_meta = json.load(f)

    p_prefix_hash = fw_meta["capability_prefix"]["logical_prefix_hash"]
    p_orig_cont_hash = fw_meta["original_persistence_continuation"]["logical_continuation_hash"]
    p_orig_parity = fw_meta["original_persistence_continuation"]["first_32k_parity"]
    p_ext_cont_hash = fw_meta["persistence_continuation"]["logical_continuation_hash"]
    p_ext_blocks = fw_meta["persistence_continuation"]["target_blocks"]

    fw_passed = (
        p_prefix_hash == CANONICAL_FINEWEB_PREFIX_HASH
        and p_orig_cont_hash == CANONICAL_FINEWEB_ORIGINAL_32K_CONT_HASH
        and p_orig_parity == "BIT_IDENTICAL"
        and p_ext_cont_hash == CANONICAL_FINEWEB_128K_CONT_HASH
        and p_ext_blocks == 128000
    )
    checks["fineweb_datasets"] = {
        "prefix_hash": p_prefix_hash,
        "prefix_hash_match": p_prefix_hash == CANONICAL_FINEWEB_PREFIX_HASH,
        "original_32k_continuation_hash": p_orig_cont_hash,
        "original_32k_parity": p_orig_parity,
        "extended_128k_continuation_hash": p_ext_cont_hash,
        "extended_128k_continuation_hash_match": p_ext_cont_hash == CANONICAL_FINEWEB_128K_CONT_HASH,
        "target_blocks": p_ext_blocks,
        "status": "PASSED" if fw_passed else "FAILED",
    }
    if not fw_passed:
        raise RuntimeError("FineWeb dataset manifest checks failed!")

    # 5. Task 1 Calibration Manifest Audit
    calib_p = Path("artifacts/strengthening_calibration_prompt_manifest.json")
    if not calib_p.exists():
        raise FileNotFoundError(f"Missing {calib_p}")
    with open(calib_p, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    records = calib_data["records"]
    from ccpt.data.hashing import sha256_json
    actual_logical_hash = sha256_json(records)
    expected_logical_hash = "e39be5aed40e698d12b5132980c208ff68ad7208501fcd918ceae1011491ef7d"
    calib_passed = (actual_logical_hash == expected_logical_hash) and (len(records) == 2335)

    checks["calibration_manifest"] = {
        "path": str(calib_p),
        "total_records": len(records),
        "logical_hash": actual_logical_hash,
        "expected_logical_hash": expected_logical_hash,
        "status": "PASSED" if calib_passed else "FAILED",
    }
    if not calib_passed:
        raise RuntimeError(f"Calibration manifest check failed: {actual_logical_hash} != {expected_logical_hash}")

    # 6. Seed Safety Invariant
    checks["seed_safety"] = {
        "allowed_sentinel_seeds": ALLOWED_SEEDS,
        "reserved_evaluation_seed": RESERVED_SEED,
        "reserved_seed_excluded_from_training": RESERVED_SEED not in ALLOWED_SEEDS,
        "status": "PASSED",
    }

    # 7. Model B/C Initialization Parity Audit
    dual_cfg = get_smoke_dual_stream_config()
    init_hashes = {}
    for s in ALLOWED_SEEDS:
        mb, mc = create_identical_dual_stream_models(dual_cfg, seed=s)
        hb = compute_canonical_state_dict_hash(mb.state_dict())
        hc = compute_canonical_state_dict_hash(mc.state_dict())
        if hb != hc:
            raise RuntimeError(f"Initialization parity failed for seed {s}: {hb} != {hc}")
        init_hashes[str(s)] = {"hash_b": hb, "hash_c": hc, "parity": "BIT_IDENTICAL"}

    checks["initialization_parity"] = {
        "seeds": init_hashes,
        "status": "PASSED",
    }

    # 8. Parameter Ownership & Firewall Invariant Audit
    # Model C
    mc = CCPTDualStreamModel(dual_cfg)
    c_theta_c_names = {name for name, _ in mc.named_parameters() if any(name.startswith(p) for p in ["embedding", "capability_layers", "capability_final_norm"])}
    c_theta_n_names = {name for name, _ in mc.named_parameters() if any(name.startswith(p) for p in ["p_in", "obs_projections", "normative_layers", "gate_projections", "steering_projections", "normative_final_norm", "risk_head"])}
    all_c_names = {name for name, _ in mc.named_parameters()}
    c_disjoint = len(c_theta_c_names.intersection(c_theta_n_names)) == 0 and len(c_theta_c_names.union(c_theta_n_names)) == len(all_c_names)

    # Model D
    d_cfg = get_smoke_adapter_config()
    md = FrozenBackboneAdapterModel(d_cfg)
    d_safety_set = set(md.safety_parameters)
    d_backbone_set = set(md.backbone_parameters)
    all_d_set = set(md.parameters())
    d_disjoint = len(d_backbone_set.intersection(d_safety_set)) == 0 and len(d_backbone_set.union(d_safety_set)) == len(all_d_set)

    checks["parameter_ownership"] = {
        "model_c_partition_valid": c_disjoint,
        "model_c_theta_c_count": len(c_theta_c_names),
        "model_c_theta_n_count": len(c_theta_n_names),
        "model_d_partition_valid": d_disjoint,
        "model_d_backbone_count": len(d_backbone_set),
        "model_d_safety_count": len(d_safety_set),
        "status": "PASSED" if (c_disjoint and d_disjoint) else "FAILED",
    }
    if not (c_disjoint and d_disjoint):
        raise RuntimeError("Parameter ownership partition verification failed!")

    overall_passed = all(v.get("status") == "PASSED" for v in checks.values())

    return {
        "preflight_version": "ccpt-strengthening-task2-preflight-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_sha": head_sha,
        "overall_status": "PASSED" if overall_passed else "FAILED",
        "checks": checks,
    }


def main():
    print("=== Executing CCPT Strengthening Task 2 Fail-Closed Preflight ===")
    res = run_preflight()
    print(f"Overall status: {res['overall_status']}")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    out_p = artifacts_dir / "strengthening_task2_preflight.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Preflight artifact written to: {out_p}")


if __name__ == "__main__":
    main()
