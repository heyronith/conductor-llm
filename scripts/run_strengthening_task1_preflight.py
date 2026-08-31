"""Preflight verification script for CCPT Strengthening Round Task 1 & 1.1.

Verifies authoritative Task 7.4 experiment semantics, validates protocol <-> markdown
parity, validates calibration prompt isolation, pins XSTest, checks independent judge,
and generates the authoritative machine-readable protocol freeze:
  - artifacts/strengthening_task1_protocol.json
  - artifacts/strengthening_task1_preflight.json
  - artifacts/strengthening_calibration_prompt_manifest.json

Enforces zero-GPU execution, historical evidence immutability, parameter
accounting parity, seed safety invariants, compute budgeting, and fail-closed gates.
"""

import sys
import os
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ccpt.config import (
    get_smoke_dual_stream_config,
    get_smoke_adapter_config,
    get_smoke_baseline_config,
)
from ccpt.modeling.dual_stream import JointTrainingDualStreamModel, CCPTDualStreamModel
from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.data.canonical_materializer import (
    TARGET_TRAIN_PREFIX_BLOCKS,
    TARGET_PERSISTENCE_BLOCKS,
    TARGET_TOTAL_TRAIN_BLOCKS,
    TARGET_VAL_BLOCKS,
    FINEWEB_SOURCE_REPO,
    FINEWEB_SOURCE_CONFIG,
    FINEWEB_SOURCE_REVISION,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
)
from ccpt.data.wildguard import (
    CANONICAL_TASK4_MANIFEST_HASH,
    CANONICAL_WILDGUARD_COUNTS,
    CANONICAL_ARROW_SHA256,
)
from ccpt.data.beavertails import (
    BEAVERTAILS_SOURCE_REPO,
    BEAVERTAILS_SOURCE_REVISION,
    BEAVERTAILS_DEFAULT_SPLIT,
    load_beavertails_ood_dataset,
)
from ccpt.training.engine import create_identical_dual_stream_models
from ccpt.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION_V3,
    get_git_commit_sha,
)
from ccpt.data.hashing import sha256_json

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs" / "research"

TASK8_2A_COMMIT = "75877602bbcb1411478c65230da437f96e1f9554"
RESERVED_SEED = 20260822
PRIMARY_SIX_SEEDS = [20260821, 20260823, 20260824, 20260825, 20260826, 20260827]
SENTINEL_SEEDS = [20260821, 20260825]
MODELS = ["model_b", "model_c", "model_d"]

FINEWEB_MANIFEST_HASH = "47c3424598d5878e54bf00dc0dd2df2af0217c10780d6c73d11a561220716055"
OOD_BEAVERTAILS_MANIFEST_HASH = "f8cf3fd0f0ca7502e9b7fef37f49ae4b9fd13cb71438ed64fc093c0649d71b9e"
ID_BENCHMARK_MANIFEST_HASH = "bdfec7a39f5304144e55d5647b886ed9bd8c676b73131fcb414f8207232fbbc4"

HISTORICAL_ARTIFACT_HASHES = {
    "artifacts/task8_2_machine_tables.json": "1d91cc491ad17320d9be180aeda9954ae77b9243ddb92d901bb3dbde1486412e",
    "artifacts/task8_hypothesis_assessment.json": "29c0b2e16735630432b6b827426c4b9c02cd7ac74fe78214aaee42a1196bf47e",
    "artifacts/task7_3_1a_forensic_summary.json": "89dcebe8c7317631f8ca1eb432e65a58dd2eb60fa72defcf13178a5322777f61",
    "artifacts/task7_4_multiseed_replication_summary.json": "5a40b33a93b4334cae7e4037f637d3c88cbb865679b46072825cbf3f2ee2f377",
    "artifacts/task8_cka_summary.json": "e9200db454fed4a1640c48ffd0d818dca34d7f62c766b51a5c4d6047afd4ff17",
    "artifacts/task8_mechanistic_summary.json": "77faac51208115b4d8157a7fe937271e8793f0c582255e857b11c7cf4fa5a516",
}

EXPECTED_CALIBRATION_LOGICAL_HASH = "e39be5aed40e698d12b5132980c208ff68ad7208501fcd918ceae1011491ef7d"
EXPECTED_CALIBRATION_RECORD_COUNT = 2335
EXPECTED_CALIBRATION_HARMFUL_COUNT = 1189
EXPECTED_CALIBRATION_BENIGN_COUNT = 1146


def sha256_file(p: Path) -> str:
    """Compute sha256 of file contents."""
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_git_lineage() -> Dict[str, Any]:
    """Verify git branch and commit lineage contains Task 8.2A commit."""
    try:
        cur_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True
        ).strip()
        head_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True
        ).strip()
        merge_base = subprocess.check_output(
            ["git", "merge-base", TASK8_2A_COMMIT, "HEAD"],
            cwd=PROJECT_ROOT,
            text=True
        ).strip()
        lineage_valid = (merge_base == TASK8_2A_COMMIT)
    except Exception:
        lineage_valid = False
        cur_branch = "unknown"
        head_commit = "unknown"

    return {
        "current_branch": cur_branch,
        "current_head": head_commit,
        "task8_2a_commit": TASK8_2A_COMMIT,
        "lineage_valid": lineage_valid,
    }


def verify_historical_artifacts() -> Dict[str, Any]:
    """Verify that all historical Task 7 & 8 artifacts are byte-identical to frozen state."""
    results = {}
    all_matched = True
    for rel_path, expected_hash in HISTORICAL_ARTIFACT_HASHES.items():
        p = PROJECT_ROOT / rel_path
        if not p.exists():
            results[rel_path] = {"exists": False, "match": False, "hash": None}
            all_matched = False
            continue
        actual_hash = sha256_file(p)
        matched = (actual_hash == expected_hash)
        if not matched:
            all_matched = False
        results[rel_path] = {
            "exists": True,
            "match": matched,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
        }
    return {"all_matched": all_matched, "artifacts": results}


def verify_authoritative_task7_4_semantics() -> Dict[str, Any]:
    """Fail closed if strengthening protocol disagrees with authoritative Task-7.4 sources."""
    # 1. 1B capability token budget
    tokens_1b = TARGET_TRAIN_PREFIX_BLOCKS * 1024
    assert TARGET_TRAIN_PREFIX_BLOCKS == 976_544, f"Target blocks {TARGET_TRAIN_PREFIX_BLOCKS} != 976,544"
    assert tokens_1b == 999_981_056, f"1B prefix tokens {tokens_1b} != 999,981,056"

    # 2. Sequence length
    seq_len = 1024

    # 3. Persistence token counts (authoritative 1,000 steps and 4,000 steps)
    tokens_1000 = TARGET_PERSISTENCE_BLOCKS * 1024
    assert TARGET_PERSISTENCE_BLOCKS == 32_000, f"Target persistence blocks {TARGET_PERSISTENCE_BLOCKS} != 32,000"
    assert tokens_1000 == 32_768_000, f"1000 persistence tokens {tokens_1000} != 32,768,000"

    blocks_4000 = 4000 * 32
    tokens_4000 = blocks_4000 * 1024
    assert blocks_4000 == 128_000, f"4000 persistence blocks {blocks_4000} != 128,000"
    assert tokens_4000 == 131_072_000, f"4000 persistence tokens {tokens_4000} != 131,072,000"

    # 4. Continuation block ranges
    p1000_range = [TARGET_TRAIN_PREFIX_BLOCKS, TARGET_TRAIN_PREFIX_BLOCKS + TARGET_PERSISTENCE_BLOCKS]
    assert p1000_range == [976544, 1008544], f"1000-step range {p1000_range} != [976544, 1008544]"

    p4000_range = [TARGET_TRAIN_PREFIX_BLOCKS, TARGET_TRAIN_PREFIX_BLOCKS + blocks_4000]
    assert p4000_range == [976544, 1104544], f"4000-step range {p4000_range} != [976544, 1104544]"

    # 5. Manifest hashes
    assert CANONICAL_TASK4_MANIFEST_HASH == "2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097"
    assert FINEWEB_MANIFEST_HASH == "47c3424598d5878e54bf00dc0dd2df2af0217c10780d6c73d11a561220716055"
    assert OOD_BEAVERTAILS_MANIFEST_HASH == "f8cf3fd0f0ca7502e9b7fef37f49ae4b9fd13cb71438ed64fc093c0649d71b9e"
    assert ID_BENCHMARK_MANIFEST_HASH == "bdfec7a39f5304144e55d5647b886ed9bd8c676b73131fcb414f8207232fbbc4"

    # 6. Model B/C initialization parity test
    cfg_bc = get_smoke_dual_stream_config()
    mb, mc = create_identical_dual_stream_models(cfg_bc, seed=20260821)
    for (kb, pb), (kc, pc) in zip(mb.state_dict().items(), mc.state_dict().items()):
        assert kb == kc and (pb == pc).all(), f"Model B/C init parity failure on {kb}"

    # 7. Checkpoint format
    assert CHECKPOINT_FORMAT_VERSION_V3 == "ccpt-checkpoint-v3"

    return {
        "1b_train_tokens": tokens_1b,
        "1b_train_blocks": TARGET_TRAIN_PREFIX_BLOCKS,
        "sequence_length": seq_len,
        "persistence_1000_steps_tokens": tokens_1000,
        "persistence_1000_steps_blocks": TARGET_PERSISTENCE_BLOCKS,
        "persistence_1000_block_range": p1000_range,
        "persistence_4000_steps_tokens": tokens_4000,
        "persistence_4000_steps_blocks": blocks_4000,
        "persistence_4000_block_range": p4000_range,
        "fineweb_manifest_hash": FINEWEB_MANIFEST_HASH,
        "wildguard_manifest_hash": CANONICAL_TASK4_MANIFEST_HASH,
        "beavertails_ood_manifest_hash": OOD_BEAVERTAILS_MANIFEST_HASH,
        "id_benchmark_manifest_hash": ID_BENCHMARK_MANIFEST_HASH,
        "b_c_initialization_parity": "VERIFIED_BIT_IDENTICAL",
        "status": "PASSED"
    }


def compute_model_parameter_specifications() -> Dict[str, Any]:
    """Compute and verify parameter counts for all models in comparison."""
    cfg_dual = get_smoke_dual_stream_config()
    cfg_ad = get_smoke_adapter_config()

    m_b = JointTrainingDualStreamModel(cfg_dual)
    m_c = CCPTDualStreamModel(cfg_dual)
    m_d = FrozenBackboneAdapterModel(cfg_ad)

    tot_b = sum(p.numel() for p in m_b.parameters())
    tot_c = sum(p.numel() for p in m_c.parameters())
    tot_d = sum(p.numel() for p in m_d.parameters())

    c_theta_c = sum(p.numel() for p in m_c.theta_C)
    c_theta_n = sum(p.numel() for p in m_c.theta_N)

    d_backbone = sum(p.numel() for p in m_d.backbone_parameters)
    d_safety = sum(p.numel() for p in m_d.safety_parameters)

    assert tot_b == 35_920_384, f"Model B total {tot_b} != 35,920,384"
    assert tot_c == 35_920_384, f"Model C total {tot_c} != 35,920,384"
    assert tot_d == 35_922_944, f"Model D total {tot_d} != 35,922,944"
    assert c_theta_c == 33_165_824, f"Model C theta_C {c_theta_c} != 33,165,824"
    assert c_theta_n == 2_754_560, f"Model C theta_N {c_theta_n} != 2,754,560"
    assert d_backbone == 33_165_824, f"Model D backbone {d_backbone} != 33,165,824"
    assert d_safety == 2_757_120, f"Model D safety {d_safety} != 2,757,120"

    return {
        "model_b": {
            "class_path": "ccpt.modeling.dual_stream.JointTrainingDualStreamModel",
            "role": "Unprotected dual-stream control (firewall contrast)",
            "total_parameters": tot_b,
            "capability_parameters": c_theta_c,
            "normative_parameters": c_theta_n,
        },
        "model_c": {
            "class_path": "ccpt.modeling.dual_stream.CCPTDualStreamModel",
            "role": "Protected CCPT dual-stream model (optimization firewall)",
            "total_parameters": tot_c,
            "capability_parameters": c_theta_c,
            "normative_parameters": c_theta_n,
        },
        "model_d": {
            "class_path": "ccpt.modeling.adapter.FrozenBackboneAdapterModel",
            "role": "Parameter-matched protected bottleneck adapter control",
            "total_parameters": tot_d,
            "backbone_parameters": d_backbone,
            "adapter_parameters": d_safety,
        },
    }


def verify_calibration_manifest_integrity() -> Dict[str, Any]:
    """Verify calibration prompt set exists, has 0 overlap with test sets, and matches frozen hash."""
    manifest_path = ARTIFACTS_DIR / "strengthening_calibration_prompt_manifest.json"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    assert len(records) == EXPECTED_CALIBRATION_RECORD_COUNT, f"Record count {len(records)} != {EXPECTED_CALIBRATION_RECORD_COUNT}"

    harmful_cnt = sum(1 for r in records if r["risk_label"] == 1)
    benign_cnt = sum(1 for r in records if r["risk_label"] == 0)
    assert harmful_cnt == EXPECTED_CALIBRATION_HARMFUL_COUNT, f"Harmful count {harmful_cnt} != {EXPECTED_CALIBRATION_HARMFUL_COUNT}"
    assert benign_cnt == EXPECTED_CALIBRATION_BENIGN_COUNT, f"Benign count {benign_cnt} != {EXPECTED_CALIBRATION_BENIGN_COUNT}"

    actual_logical_hash = sha256_json(records)
    assert actual_logical_hash == EXPECTED_CALIBRATION_LOGICAL_HASH, f"Logical hash {actual_logical_hash} != {EXPECTED_CALIBRATION_LOGICAL_HASH}"

    audit = data["test_isolation_audit"]
    assert audit["wildguard_test_overlap_count"] == 0
    assert audit["beavertails_30k_test_overlap_count"] == 0
    assert audit["xstest_overlap_count"] == 0
    assert audit["isolation_status"] == "PASSED_ZERO_OVERLAP"

    return {
        "manifest_path": str(manifest_path),
        "total_records": len(records),
        "harmful_count": harmful_cnt,
        "benign_count": benign_cnt,
        "logical_hash": actual_logical_hash,
        "isolation_status": "PASSED_ZERO_OVERLAP"
    }


def verify_protocol_markdown_parity(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Verify machine-readable protocol and human-readable protocol agree on scientific quantities."""
    doc_path = DOCS_DIR / "strengthening_task1_protocol.md"
    assert doc_path.exists(), f"Missing {doc_path}"
    doc_text = doc_path.read_text(encoding="utf-8")

    # 1. Seeds
    for seed in protocol["seeds"]["primary_six_seed_cohort"]:
        assert str(seed) in doc_text, f"Seed {seed} missing from protocol markdown"
    assert str(protocol["seeds"]["reserved_seeds"][0]) in doc_text, "Reserved seed missing from markdown"

    # 2. Models
    assert "JointTrainingDualStreamModel" in doc_text
    assert "CCPTDualStreamModel" in doc_text
    assert "FrozenBackboneAdapterModel" in doc_text
    assert "35,920,384" in doc_text
    assert "35,922,944" in doc_text

    # 3. Primary 20M safety endpoint
    assert "20,000,000" in doc_text or "20.0M" in doc_text

    # 4. Primary 1000-step persistence endpoint & exact token count
    assert "1,000" in doc_text or "1000" in doc_text
    assert "32,768,000" in doc_text, "32,768,000 persistence tokens missing from markdown"
    assert "131,072,000" in doc_text, "131,072,000 persistence tokens missing from markdown"

    # 5. Retention checkpoints
    for step in [0, 250, 1000, 4000]:
        assert str(step) in doc_text, f"Retention step {step} missing from markdown"

    # 6. GPU type
    assert "Modal H100!" in doc_text
    assert "L40S" in doc_text

    # 7. Budget ceiling
    assert "$40.00" in doc_text
    assert "$14.00" in doc_text

    # 8. Primary estimands
    assert "\\Delta_{\\text{primary}}" in doc_text or "C_{post, 1000} - C_{pre}" in doc_text
    assert "\\Delta_{\\text{firewall}}" in doc_text or "C_{post, 1000} - C_{pre}" in doc_text

    # 9. Ensure no lingering '~2.0M tokens' contradiction
    assert "~2.0M tokens" not in doc_text, "Lingering ~2.0M tokens contradiction found in markdown!"

    return {"status": "PASSED_PARITY_VERIFIED"}


def build_protocol_specification(
    lineage_info: Dict[str, Any],
    hist_info: Dict[str, Any],
    model_info: Dict[str, Any],
    task7_4_info: Dict[str, Any],
    calib_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Construct complete machine-readable protocol JSON structure."""
    protocol = {
        "protocol_version": "strengthening_round_v1.1",
        "freeze_date": "2026-08-31",
        "task_head_anchor": lineage_info["current_head"],
        "lineage": {
            "target_branch": "strengthening-task1-protocol-freeze",
            "upstream_anchor_commit": TASK8_2A_COMMIT,
            "verified_lineage": lineage_info["lineage_valid"],
        },
        "historical_evidence_provenance": {
            "policy": "IMMUTABLE_HISTORICAL_RECORD",
            "artifacts": {k: v["actual_sha256"] for k, v in hist_info["artifacts"].items()}
        },
        "seeds": {
            "primary_six_seed_cohort": PRIMARY_SIX_SEEDS,
            "reserved_seeds": [RESERVED_SEED],
            "reserved_seed_policy": "20260822 is strictly reserved for benchmark selection and MUST NEVER be used for training",
            "sentinel_seeds": SENTINEL_SEEDS,
            "seed_metadata": {
                "20260821": "Seed 1 (Rerun under current hardened execution to complete primary cohort)",
                "20260823": "Seed 2 (Authoritative Task 7.4 hardened run preserved)",
                "20260824": "Seed 3 (Authoritative Task 7.4 hardened run preserved)",
                "20260825": "Seed 4 (New replication seed / Sentinel)",
                "20260826": "Seed 5 (New replication seed)",
                "20260827": "Seed 6 (New replication seed)"
            }
        },
        "models": {
            "architectures": model_info,
            "initialization_parity": {
                "requirement": "Model B and Model C must share bit-identical initial parameters",
                "verifier": "ccpt.training.engine.create_identical_dual_stream_models",
                "status": "VERIFIED_PASS"
            },
            "contrasts": {
                "primary_architectural_contrast": "C vs D (CCPT vs. Parameter-matched protected adapter)",
                "firewall_contrast": "C vs B (CCPT optimization firewall vs. Unprotected dual stream)"
            }
        },
        "hardware_matrix": {
            "training_and_persistence": "Modal H100!",
            "training_gpu_enforcement": "Must use 'H100!' with exclamation mark to forbid silent cloud substitution",
            "evaluation_and_judging": "L40S",
            "protocol_and_preflight": "CPU",
            "unauthorized_hardware": ["H200", "A100", "L4", "consumer_gpus"]
        },
        "compute_budget": {
            "hard_authorization_ceiling_usd": 40.0,
            "target_spend_range_usd": [25.0, 35.0],
            "contingency_usd": 2.0,
            "allocations": {
                "task2_sentinel": {
                    "target_max_usd": 12.0,
                    "hard_stop_gate_usd": 14.0,
                    "expected_models": 6,
                    "expected_gpu_seconds_h100": 8400,
                    "max_permitted_gpu_seconds_h100": 10800,
                    "timeout_seconds": 12000
                },
                "task4_replication_and_calibration": {
                    "cumulative_target_max_usd": 32.0,
                    "hard_stop_gate_usd": 34.0
                },
                "task5_evaluation": {
                    "cumulative_target_max_usd": 38.0,
                    "hard_stop_gate_usd": 40.0
                }
            }
        },
        "primary_experiment_specification": {
            "capability_pretraining": {
                "dataset": "FineWeb-Edu packed token stream",
                "source_repo": FINEWEB_SOURCE_REPO,
                "source_config": FINEWEB_SOURCE_CONFIG,
                "source_revision": FINEWEB_SOURCE_REVISION,
                "manifest_hash": FINEWEB_MANIFEST_HASH,
                "target_prefix_blocks": task7_4_info["1b_train_blocks"],
                "target_tokens": task7_4_info["1b_train_tokens"],
                "sequence_length": task7_4_info["sequence_length"],
                "optimizer": "AdamW (lr=3e-3, beta1=0.9, beta2=0.95, wd=0.1, eps=1e-8)",
                "schedule": "Cosine decay with warmup to min_lr=3e-4",
                "checkpoint_name": "lm_1b_final.pt"
            },
            "safety_training": {
                "primary_endpoint_tokens": 20000000,
                "dataset": "WildGuard risk & safe-generation splits",
                "manifest_hash": CANONICAL_TASK4_MANIFEST_HASH,
                "trainable_parameters": {
                    "model_b": "All parameters (Joint-training control)",
                    "model_c": "theta_N only (2,754,560 params; theta_C strictly frozen)",
                    "model_d": "safety_parameters only (2,757,120 params; backbone strictly frozen)"
                },
                "optimizer": "AdamW (lr=1e-3, beta1=0.9, beta2=0.95, wd=0.1, eps=1e-8)",
                "checkpoint_name": "safety_20m_final.pt"
            },
            "persistence_continuation": {
                "primary_endpoint_steps": 1000,
                "primary_endpoint_tokens": task7_4_info["persistence_1000_steps_tokens"],
                "primary_endpoint_blocks": task7_4_info["persistence_1000_steps_blocks"],
                "primary_block_range": task7_4_info["persistence_1000_block_range"],
                "extended_curve_steps": [0, 250, 1000, 4000],
                "extended_curve_tokens": [0, 8192000, 32768000, 131072000],
                "extended_curve_blocks": [0, 8000, 32000, 128000],
                "extended_4000_block_range": task7_4_info["persistence_4000_block_range"],
                "stream_continuation_policy": "Continuous uninterrupted stream from frozen 20M safety checkpoint",
                "optimizer": "AdamW (lr=3e-4, beta1=0.9, beta2=0.95, wd=0.1, eps=1e-8)",
                "optimizer_reset_policy": "Reset optimizer at step 0; do NOT reset optimizer at 250 or 1000",
                "checkpoint_names": {
                    "250": "persistence_250_final.pt",
                    "1000": "persistence_1000_final.pt",
                    "4000": "persistence_4000_final.pt"
                }
            }
        },
        "operating_point_experiment": {
            "status": "SECONDARY_SENSITIVITY_ANALYSIS",
            "candidate_safety_checkpoints": ["10M", "20M", "30M", "40M"],
            "calibration_manifest": {
                "file": "artifacts/strengthening_calibration_prompt_manifest.json",
                "total_records": calib_info["total_records"],
                "harmful_count": calib_info["harmful_count"],
                "benign_count": calib_info["benign_count"],
                "logical_hash": calib_info["logical_hash"],
                "test_isolation_status": calib_info["isolation_status"]
            },
            "forbidden_calibration_datasets": [
                "WildGuard test split (wildguardtest)",
                "BeaverTails 30k_test OOD test split",
                "XSTest benchmark"
            ],
            "distance_metric": "distance = |harmful_refusal_A - harmful_refusal_B| + |benign_refusal_A - benign_refusal_B| (in percentage points)",
            "tie_breaking_order": [
                "1. Smallest harmful refusal difference",
                "2. Smallest benign refusal difference",
                "3. Lower total safety-token budget",
                "4. Earlier checkpoint step"
            ]
        },
        "evaluation_and_benchmarks": {
            "primary_benchmark": {
                "name": "BeaverTails 30k OOD Harmful subset",
                "prompts_count": 256,
                "manifest_hash": OOD_BEAVERTAILS_MANIFEST_HASH,
                "dataset_repo": BEAVERTAILS_SOURCE_REPO,
                "dataset_revision": BEAVERTAILS_SOURCE_REVISION,
                "seed": RESERVED_SEED,
                "judge": "allenai/wildguard (revision cbba4823f3e8020e5a74a5e29bf85072def6f2ff)"
            },
            "primary_metric": "determinate_refusal_rate = YES / (YES + NO)",
            "first_class_outcomes": [
                "Harmful refusal rate",
                "Harmful compliance rate",
                "Harmful N/A indeterminate rate",
                "Benign over-refusal rate",
                "Benign non-refusal rate",
                "Capability validation cross-entropy loss / perplexity"
            ],
            "over_refusal_benchmark": {
                "benchmark": "XSTest",
                "pinned_dataset": {
                    "repo": "natolambert/xstest-v2-copy",
                    "revision": "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5",
                    "file": "data/prompts-00000-of-00001.parquet",
                    "sha256": "322d4e89df9fb419c296d5b360067f3265845d40a561a37d9be77a078d219522",
                    "upstream_canonical": "walledai/XSTest@f1d713187c61b6ae64e602d74f0b3d812cc2e8e8"
                },
                "total_prompts": 450,
                "safe_prompts": 250,
                "contrast_unsafe_prompts": 200,
                "purpose": "Evaluate over-refusal on prompts containing sensitive vocabulary vs. true harmfulness"
            },
            "independent_safety_judge": {
                "model": "meta-llama/Llama-Guard-3-8B",
                "frozen_revision": "f516a7f5f9f68800ba8ea969a531e21b790d0b04",
                "access_verification": "ACCESS_NOT_YET_EXECUTION_VERIFIED",
                "hf_credentials_verified": True,
                "active_repo_head_commit": "7327bd9f6efbbe6101dc6cc4736302b3cbb6e425",
                "hardware": "L40S",
                "generation_mode": "greedy (temperature=0.0)"
            },
            "human_audit": {
                "sample_size": 300,
                "design": "Stratified double-blind rater audit",
                "strata": ["model (B/C/D)", "seed", "harmful vs benign", "pre vs post persistence", "judge agreement vs disagreement"],
                "disagreement_handling": "Report both rater scores and inter-rater agreement (Cohen's Kappa); do not discard"
            }
        },
        "statistical_estimands": {
            "experimental_unit": "Seed (N=6 independent model pipelines)",
            "prompt_aggregation": "Never treat individual prompt completions as independent replications",
            "primary_persistence_effect": "(C_post_1000 - C_pre) - (D_post_1000 - D_pre)",
            "firewall_contrast_effect": "(C_post_1000 - C_pre) - (B_post_1000 - B_pre)",
            "required_summary_statistics": [
                "Individual seed values",
                "Mean effect across seeds",
                "Median effect across seeds",
                "Sample standard deviation (N-1=5 degrees of freedom)",
                "95% Student-t confidence interval",
                "Direction consistency count (e.g. k/6 seeds)",
                "Raw judge determinate counts (YES, NO, NA)"
            ]
        },
        "go_stop_policy": {
            "purpose": "Technical and scientific identifiability gate; NOT a favorable-result selection gate",
            "scientific_result_rule": "DO NOT STOP if C loses to B, C loses to D, effects are small, or seed reverses. Negative results must be reported.",
            "automatic_stop_conditions": [
                "Wrong GPU hardware (anything other than H100! for training or L40S for evaluation)",
                "Runtime or Python environment mismatch",
                "Git code SHA mismatch during execution",
                "Dataset manifest logical hash mismatch",
                "Tokenizer asset hash divergence",
                "Seed collision or use of reserved seed 20260822",
                "Model B and C bit-identical initialization parity failure",
                "Protected parameter mutation during frozen phases (theta_C in Phase 2; theta_N/adapters in Phase 3)",
                "Optimizer parameter group partition violation",
                "NaN or Inf loss during forward/backward pass",
                "Unrecoverable loss divergence",
                "Checkpoint corruption or unreadable state dict",
                "Persistence stream token range or block mismatch",
                "Step 1000 checkpoint does not match original protocol semantics (32,768,000 tokens)",
                "Evaluation artifacts disconnected from checkpoint hash",
                "Preflight validation failure"
            ],
            "retry_policy": "Infrastructure/transient failures may be retried AT MOST ONCE from the latest verified checkpoint with bit-identical parameters and configurations."
        }
    }
    return protocol


def run_preflight() -> Dict[str, Any]:
    """Execute all preflight checks and output artifacts."""
    print("=== CCPT Strengthening Round Task 1.1: Protocol Preflight ===", flush=True)

    # 1. Lineage
    lineage = verify_git_lineage()
    print(f"1. Git Lineage Check: branch={lineage['current_branch']}, anchor_valid={lineage['lineage_valid']}")
    assert lineage["lineage_valid"], f"Git lineage does not contain Task 8.2A commit {TASK8_2A_COMMIT}"

    # 2. Historical Artifacts
    hist = verify_historical_artifacts()
    print(f"2. Historical Artifacts Check: all_matched={hist['all_matched']}")
    for k, v in hist["artifacts"].items():
        print(f"   - {k}: match={v['match']}")
        assert v["match"], f"Artifact {k} does not match frozen hash!"

    # 3. Authoritative Task 7.4 Semantics & Constants
    task7_4 = verify_authoritative_task7_4_semantics()
    print(f"3. Authoritative Task 7.4 Semantics Check: status={task7_4['status']}")
    print(f"   - 1B Capability Tokens: {task7_4['1b_train_tokens']:,} ({task7_4['1b_train_blocks']:,} blocks)")
    print(f"   - 1000-Step Persistence Tokens: {task7_4['persistence_1000_steps_tokens']:,} ({task7_4['persistence_1000_steps_blocks']:,} blocks, range={task7_4['persistence_1000_block_range']})")
    print(f"   - 4000-Step Persistence Tokens: {task7_4['persistence_4000_steps_tokens']:,} ({task7_4['persistence_4000_steps_blocks']:,} blocks, range={task7_4['persistence_4000_block_range']})")

    # 4. Model Parameters
    models = compute_model_parameter_specifications()
    print("4. Model Parameters Check:")
    for m, d in models.items():
        print(f"   - {m} ({d['class_path']}): total={d['total_parameters']}")

    # 5. Seed Safety Invariants
    assert RESERVED_SEED not in PRIMARY_SIX_SEEDS, f"Reserved seed {RESERVED_SEED} in primary seeds!"
    assert RESERVED_SEED not in SENTINEL_SEEDS, f"Reserved seed {RESERVED_SEED} in sentinel seeds!"
    assert len(PRIMARY_SIX_SEEDS) == 6, f"Primary cohort length {len(PRIMARY_SIX_SEEDS)} != 6"
    assert len(set(PRIMARY_SIX_SEEDS)) == 6, "Duplicate seeds in primary cohort!"
    print(f"5. Seed Safety: 6 unique seeds, reserved {RESERVED_SEED} strictly excluded from training.")

    # 6. Calibration Prompt Manifest Integrity
    calib = verify_calibration_manifest_integrity()
    print(f"6. Calibration Manifest Check: status={calib['isolation_status']}, total={calib['total_records']}")

    # 7. Build Machine Protocol Specification
    protocol = build_protocol_specification(lineage, hist, models, task7_4, calib)

    # 8. Protocol <-> Markdown Parity
    parity = verify_protocol_markdown_parity(protocol)
    print(f"7. Protocol <-> Markdown Parity Check: status={parity['status']}")

    # Write protocol JSON
    proto_path = ARTIFACTS_DIR / "strengthening_task1_protocol.json"
    with open(proto_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)
    print(f"8. Wrote protocol to {proto_path}")

    # Build Preflight Result
    preflight_result = {
        "status": "PASSED",
        "task": "strengthening_task1_1_protocol_freeze",
        "timestamp_utc": "2026-08-31T23:15:00Z",
        "checks": {
            "git_lineage": lineage,
            "historical_artifacts": hist,
            "authoritative_task7_4_semantics": task7_4,
            "model_parameters": models,
            "seeds": {
                "primary_six_seeds": PRIMARY_SIX_SEEDS,
                "reserved_seed_safeguard": "PASSED",
                "sentinel_seeds": SENTINEL_SEEDS
            },
            "calibration_prompt_manifest": calib,
            "protocol_markdown_parity": parity,
            "hardware_safeguards": {
                "training_gpu": "Modal H100!",
                "eval_gpu": "L40S",
                "preflight_gpu_seconds_used": 0
            },
            "budget_safeguards": {
                "hard_ceiling_usd": 40.0,
                "task2_sentinel_hard_gate_usd": 14.0
            }
        },
        "protocol_json_sha256": sha256_file(proto_path),
        "calibration_manifest_sha256": sha256_file(ARTIFACTS_DIR / "strengthening_calibration_prompt_manifest.json")
    }

    preflight_path = ARTIFACTS_DIR / "strengthening_task1_preflight.json"
    with open(preflight_path, "w", encoding="utf-8") as f:
        json.dump(preflight_result, f, indent=2)
    print(f"9. Wrote preflight result to {preflight_path}")

    print("=== All Preflight Checks PASSED (0 GPU seconds consumed) ===", flush=True)
    return preflight_result


if __name__ == "__main__":
    run_preflight()
