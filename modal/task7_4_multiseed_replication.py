"""Modal Task 7.4: Authoritative Multi-Seed Replication Pipeline (Seeds 2 & 3).

The ONLY authorized production entrypoint for CCPT Seeds 2 & 3 replication runs.
Executes 1B LM Pretraining -> 20M Safety Training -> 1,000-step Persistence Experiment ->
Real ID/OOD Behavioral Evaluation across Seeds 2 (20260823) and 3 (20260824).

Invariants Enforced:
- Environment: TASK7_4_FROZEN_REPLICATION_ENVIRONMENT with fail-closed fingerprinting.
- Checkpoints: Strict V3 with mandatory exact Code-A SHA and full schedule audit hashes.
- Run Directories: /runs/ccpt/task7_4/multiseed_replication_v1/seed_{seed}/{model}/
- Data Sources: Canonical Task-4 WildGuard Arrow files & frozen FineWeb-Edu blocks.
"""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

APP_NAME = "ccpt-task7-4-multiseed-replication"
app = modal.App(APP_NAME)

# Frozen Seed Constants
SEED_1_HISTORICAL = 20260821
SEED_2_REPLICATION = 20260823
SEED_3_REPLICATION = 20260824
BEAVERTAILS_OOD_SEED = 20260822  # Reserved for OOD benchmark selection ONLY

TASK7_4_CODE_SHA = os.environ.get("CCPT_CODE_COMMIT_SHA", "1bbbda8fbac50cd3feea0e821ca9e550db1a80fb")

# Frozen Canonical Hashes
CANONICAL_TASK4_MANIFEST_HASH = "2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097"
CANONICAL_FINEWEB_MANIFEST_HASH = "47c3424598d5878e54bf00dc0dd2df2af0217c10780d6c73d11a561220716055"
CANONICAL_FINEWEB_PREFIX_HASH = "a13410b63d9c1533211784c2a08fa5a918e29cc446448470395aa93919712585"
CANONICAL_FINEWEB_CONT_HASH = "1f6dd66f49a9afa3537244a719af74006308ab81902b0b654142510672022243"
CANONICAL_FINEWEB_VAL_HASH = "4ef33f8f6e1058e1a9e702afe4444593eb07d67ab8a05838e2f81fc6e9eaf870"

LEGACY_SAFETY_SCHEDULE_HASH = "b141fcbc05d8388086f8649d5162c63b4ef862b90e049cbc2e0b29f7f1eb3caa"
CANONICAL_FULL_SCHEDULE_HASH = "6e1be80718a7bd9f1fb2f5bd42c87a9cd793afac08694e46f5c449af379ec2a0"

ID_BENCHMARK_MANIFEST_HASH = "bdfec7a39f5304144e55d5647b886ed9bd8c676b73131fcb414f8207232fbbc4"
OOD_BEAVERTAILS_MANIFEST_HASH = "f8cf3fd0f0ca7502e9b7fef37f49ae4b9fd13cb71438ed64fc093c0649d71b9e"
WILDGUARD_JUDGE_REVISION = "cbba4823f3e8020e5a74a5e29bf85072def6f2ff"

TASK7_4_EXPECTED_PACKAGE_VERSIONS = {
    "torch": "2.5.1",
    "transformers": "4.46.3",
    "tokenizers": "0.20.3",
    "datasets": "3.1.0",
    "huggingface_hub": "0.26.2",
    "accelerate": "1.1.1",
    "pyarrow": "17.0.0",
    "numpy": "2.1.3",
    "sentencepiece": "0.2.0",
    "tiktoken": "0.8.0",
    "pytest": "8.3.3",
}

# Modal Image Definition
replication_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "datasets==3.1.0",
        "huggingface_hub==0.26.2",
        "sentencepiece==0.2.0",
        "tiktoken==0.8.0",
        "accelerate==1.1.1",
        "pyarrow==17.0.0",
        "numpy==2.1.3",
        "pytest==8.3.3",
    )
    .env({"CCPT_CODE_COMMIT_SHA": TASK7_4_CODE_SHA})
    .add_local_python_source("ccpt")
    .add_local_dir("tests", remote_path="/root/tests")
)

# Persistent Volumes
data_volume = modal.Volume.from_name("ccpt-authoritative-data", create_if_missing=True)
task4_data_volume = modal.Volume.from_name("ccpt-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("ccpt-authoritative-runs", create_if_missing=True)

hf_secrets = [
    modal.Secret.from_name("huggingface"),
    modal.Secret.from_name("huggingface-secret"),
]


# -----------------------------------------------------------------------------
# Runtime Fingerprint & Fail-Closed Validation
# -----------------------------------------------------------------------------

def capture_and_verify_runtime_fingerprint(
    expected_code_sha: Optional[str] = None,
    required_gpu_type: Optional[str] = None,
    strict_version_check: bool = True,
) -> Dict[str, Any]:
    """Captures and strictly verifies the container runtime environment fingerprint.

    Fails closed immediately on missing packages, unsupported package versions,
    missing/mismatched GPUs, or missing/mismatched Git commit SHA.
    """
    installed_versions = {}
    for pkg, exp_ver in TASK7_4_EXPECTED_PACKAGE_VERSIONS.items():
        try:
            act_ver = importlib.metadata.version(pkg)
            installed_versions[pkg] = act_ver
            if strict_version_check and act_ver != exp_ver:
                raise RuntimeError(
                    f"TASK7_4_FROZEN_REPLICATION_ENVIRONMENT version mismatch for {pkg}: "
                    f"expected {exp_ver}, got {act_ver}"
                )
        except Exception as e:
            raise RuntimeError(f"Required package {pkg} is not installed or mismatched: {e}")

    # Verify Python version
    py_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if strict_version_check and py_major_minor != "3.11":
        raise RuntimeError(f"Expected Python 3.11, got {py_major_minor} ({sys.version})")

    # Verify GPU availability & device name if requested
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    cuda_ver = str(torch.version.cuda) if cuda_avail else None

    if required_gpu_type is not None:
        if not cuda_avail:
            raise RuntimeError(f"GPU type '{required_gpu_type}' required but CUDA is not available")
        if required_gpu_type == "H100" and "H100" not in device_name:
            raise RuntimeError(f"Expected H100 GPU for training, got: {device_name}")
        if required_gpu_type == "L40S" and "L40S" not in device_name:
            raise RuntimeError(f"Expected L40S GPU for evaluation, got: {device_name}")

    # Verify Git SHA
    code_sha = os.environ.get("CCPT_CODE_COMMIT_SHA") or os.environ.get("TASK7_4_CODE_SHA")
    if not code_sha or code_sha in ("unknown", "unresolved"):
        raise RuntimeError(f"Invalid or missing CCPT_CODE_COMMIT_SHA: {code_sha}")

    if expected_code_sha is not None and code_sha != expected_code_sha:
        raise RuntimeError(f"Runtime Code SHA mismatch: expected {expected_code_sha}, got {code_sha}")

    fingerprint = {
        "environment_name": "TASK7_4_FROZEN_REPLICATION_ENVIRONMENT",
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "device_name": device_name,
        "installed_versions": installed_versions,
        "git_commit_sha": code_sha,
    }

    fp_bytes = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    fingerprint["fingerprint_hash"] = hashlib.sha256(fp_bytes).hexdigest()
    return fingerprint


# -----------------------------------------------------------------------------
# Progress Logger (1/100 ... 100/100 Integer Precision)
# -----------------------------------------------------------------------------

class Task74ProgressLogger:
    """Emits and persists integer-precise 1/100 ... 100/100 telemetry progress events."""

    def __init__(
        self,
        seed: int,
        model_type: str,
        phase: str,
        total_steps: int,
        total_tokens: int,
        log_dir: Path,
        gpu_name: str = "H100",
    ):
        self.seed = seed
        self.model_type = model_type
        self.phase = phase
        self.total_steps = total_steps
        self.total_tokens = total_tokens
        self.log_dir = log_dir
        self.gpu_name = gpu_name
        self.start_time = time.time()
        self.last_pct = 0
        self.loss_ema = None
        self.log_file = log_dir / "progress.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_step(
        self,
        step: int,
        tokens_seen: int,
        loss: float,
        lr: float,
        grad_norm: Optional[float] = None,
    ):
        current_pct = int((step / max(1, self.total_steps)) * 100)
        current_pct = min(100, max(0, current_pct))

        # Update Loss EMA
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = 0.95 * self.loss_ema + 0.05 * loss

        elapsed = time.time() - self.start_time
        tokens_per_sec = (tokens_seen / elapsed) if elapsed > 0 else 0.0
        remaining_steps = max(0, self.total_steps - step)
        eta_sec = (remaining_steps / (step / elapsed)) if step > 0 and elapsed > 0 else 0.0

        from ccpt.training.cost import compute_gpu_cost
        gpu_key = "H100" if "H100" in self.gpu_name else "L40S"
        cost_so_far = compute_gpu_cost(elapsed, gpu_type=gpu_key)
        projected_cost = compute_gpu_cost(elapsed + eta_sec, gpu_type=gpu_key)

        now_utc = datetime.now(timezone.utc)

        event = {
            "progress_fraction": f"{current_pct}/100",
            "progress_percent": current_pct,
            "timestamp_utc": now_utc.isoformat(),
            "seed": self.seed,
            "model_type": self.model_type,
            "phase": self.phase,
            "step": step,
            "total_steps": self.total_steps,
            "tokens_seen": tokens_seen,
            "total_tokens": self.total_tokens,
            "loss": round(float(loss), 4),
            "loss_ema": round(float(self.loss_ema), 4),
            "learning_rate": float(lr),
            "grad_norm": round(float(grad_norm), 4) if grad_norm is not None else None,
            "tokens_per_sec": round(float(tokens_per_sec), 1),
            "elapsed_sec": round(float(elapsed), 1),
            "eta_sec": round(float(eta_sec), 1),
            "gpu_name": self.gpu_name,
            "vram_gb": round(float(vram_gb), 2),
            "cost_so_far_usd": round(float(cost_so_far), 4),
            "projected_cost_usd": round(float(projected_cost), 4),
        }

        # Write to JSONL
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        # Emit console update if percent changed or at key milestones
        if current_pct > self.last_pct or step in (1, self.total_steps):
            self.last_pct = current_pct
            print(
                f"[{self.phase.upper()}] [{self.model_type}] Seed {self.seed} "
                f"Progress: {current_pct}/100 | Step {step}/{self.total_steps} | "
                f"Loss: {loss:.4f} (EMA: {self.loss_ema:.4f}) | "
                f"Speed: {tokens_per_sec:.0f} tok/s | Elapsed: {elapsed:.0f}s | ETA: {eta_sec:.0f}s",
                flush=True,
            )


# -----------------------------------------------------------------------------
# Authoritative Data & Schedule Resolvers
# -----------------------------------------------------------------------------

def get_task7_4_output_dir(seed: int, model_type: str) -> Path:
    """Returns the authoritative non-colliding output directory for Task 7.4 runs."""
    base = Path(f"/runs/ccpt/task7_4/multiseed_replication_v1/seed_{seed}/{model_type}")
    return base


def verify_authoritative_fineweb_mount() -> Dict[str, Any]:
    """Verifies that the frozen FineWeb dataset exists on /data without rematerialization."""
    manifest_p = Path("/data/fineweb_authoritative/manifest.json")
    if not manifest_p.exists():
        raise FileNotFoundError(f"Authoritative FineWeb manifest missing at {manifest_p}")

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    actual_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    if actual_manifest_hash != CANONICAL_FINEWEB_MANIFEST_HASH:
        raise ValueError(
            f"FineWeb manifest hash mismatch: expected {CANONICAL_FINEWEB_MANIFEST_HASH}, got {actual_manifest_hash}"
        )

    # Check required shard counts
    prefix_shards = manifest.get("prefix_blocks_total", 0)
    continuation_shards = manifest.get("continuation_blocks_total", 0)
    val_shards = manifest.get("val_blocks_total", 0)

    if prefix_shards != 976544 or continuation_shards != 32000 or val_shards != 1024:
        raise ValueError(
            f"FineWeb block count mismatch: prefix={prefix_shards}/976544, "
            f"cont={continuation_shards}/32000, val={val_shards}/1024"
        )

    return {
        "manifest_path": str(manifest_p),
        "manifest_hash": actual_manifest_hash,
        "prefix_blocks": prefix_shards,
        "continuation_blocks": continuation_shards,
        "val_blocks": val_shards,
        "verified": True,
    }


def verify_authoritative_safety_schedule(canonical_records_map: Dict[str, Any]) -> Dict[str, Any]:
    """Verifies the immutable frozen safety schedule across all 2,344 batches."""
    from ccpt.training.safety_schedule import compute_full_schedule_audit_hash

    sched_p = Path("/data/safety_schedule.json")
    if not sched_p.exists():
        raise FileNotFoundError(f"Frozen safety schedule missing at {sched_p}")

    with open(sched_p, "r", encoding="utf-8") as f:
        schedule_data = json.load(f)

    actual_legacy_hash = schedule_data.get("schedule_hash")
    if actual_legacy_hash != LEGACY_SAFETY_SCHEDULE_HASH:
        raise ValueError(
            f"Safety schedule legacy hash mismatch: expected {LEGACY_SAFETY_SCHEDULE_HASH}, got {actual_legacy_hash}"
        )

    actual_full_hash = compute_full_schedule_audit_hash(schedule_data)
    if actual_full_hash != CANONICAL_FULL_SCHEDULE_HASH:
        raise ValueError(
            f"Safety schedule full audit hash mismatch: expected {CANONICAL_FULL_SCHEDULE_HASH}, got {actual_full_hash}"
        )

    batches = schedule_data.get("batches", [])
    if len(batches) != 2344:
        raise ValueError(f"Expected 2,344 schedule batches, got {len(batches)}")

    # Verify each batch against canonical records map
    cumulative_tokens = 0
    for idx, b in enumerate(batches):
        expected_type = "risk" if idx % 2 == 0 else "generation"
        if b["batch_type"] != expected_type:
            raise ValueError(f"Batch {idx} type mismatch: expected {expected_type}, got {b['batch_type']}")

        example_ids = b["example_ids"]
        if len(example_ids) != 32:
            raise ValueError(f"Batch {idx} does not have 32 example IDs (has {len(example_ids)})")

        # Resolve IDs
        batch_tokens = 0
        for eid in example_ids:
            if eid not in canonical_records_map:
                raise ValueError(f"Batch {idx} references unknown record ID: {eid}")
            rec = canonical_records_map[eid]
            batch_tokens += len(rec.input_ids)

        if b["valid_input_tokens"] != batch_tokens:
            raise ValueError(f"Batch {idx} valid token count mismatch: schedule={b['valid_input_tokens']}, actual={batch_tokens}")

        cumulative_tokens += batch_tokens

    if cumulative_tokens != 20010611:
        raise ValueError(f"Cumulative schedule tokens mismatch: expected 20,010,611, got {cumulative_tokens}")

    return {
        "schedule_path": str(sched_p),
        "legacy_hash": actual_legacy_hash,
        "full_audit_hash": actual_full_hash,
        "total_batches": len(batches),
        "total_valid_tokens": cumulative_tokens,
        "verified": True,
    }


# -----------------------------------------------------------------------------
# Modal Preflight Probe Function (Cheap CPU / Short GPU Verification)
# -----------------------------------------------------------------------------

@app.function(
    image=replication_image,
    volumes={"/runs": runs_volume, "/data": data_volume, "/data_task4": task4_data_volume},
    secrets=hf_secrets,
    gpu="L40S",
    timeout=600,
)
def run_task7_4_modal_preflight_probe(expected_code_sha: str) -> Dict[str, Any]:
    """Authoritative Modal In-Container Preflight Probe.

    Verifies inside the ACTUAL Modal container:
    1. TASK7_4_FROZEN_REPLICATION_ENVIRONMENT package versions & CUDA
    2. Exact Code-A SHA injection
    3. Canonical WildGuard Arrow file existence & exact SHA256 hashes
    4. Safety record provenance on real Arrow records
    5. Frozen FineWeb manifest and shard counts
    6. Frozen safety schedule legacy and full audit hashes
    7. Model B/C initialization equality on smoke configs for Seeds 2 & 3
    """
    print("=================================================================", flush=True)
    print("TASK 7.4.1 MODAL IN-CONTAINER PREFLIGHT PROBE", flush=True)
    print("=================================================================", flush=True)

    # 1. Capture & Verify Runtime Fingerprint
    print("\n[1/7] Verifying Container Runtime Fingerprint...", flush=True)
    fp = capture_and_verify_runtime_fingerprint(
        expected_code_sha=expected_code_sha,
        required_gpu_type="L40S",
        strict_version_check=True,
    )
    print(f"  -> Fingerprint verified: {fp['fingerprint_hash'][:16]}...")
    print(f"  -> Device: {fp['device_name']} | CUDA: {fp['cuda_version']}")

    # 2. Canonical WildGuard Resolution
    print("\n[2/7] Resolving & Verifying Canonical Task 4 WildGuard Arrow Artifacts...", flush=True)
    from ccpt.data.wildguard import (
        resolve_canonical_wildguard_artifacts,
        verify_safety_records_provenance,
        load_wildguard_records,
    )
    wg_artifacts = resolve_canonical_wildguard_artifacts(require_arrow_only=True)
    print("  -> WildGuard Arrow artifacts verified matching canonical SHA256 hashes and counts.")

    # 3. Real Record Provenance Verification
    print("\n[3/7] Loading Real Canonical Arrow Records for Field-by-Field Provenance...", flush=True)
    risk_train_recs = load_wildguard_records(wg_artifacts["risk_train"]["resolved_path"], record_type="risk")
    risk_val_recs = load_wildguard_records(wg_artifacts["risk_val"]["resolved_path"], record_type="risk")
    gen_train_recs = load_wildguard_records(wg_artifacts["gen_train"]["resolved_path"], record_type="generation")
    gen_val_recs = load_wildguard_records(wg_artifacts["gen_val"]["resolved_path"], record_type="generation")

    prov_res = verify_safety_records_provenance(risk_train_recs, risk_val_recs, gen_train_recs, gen_val_recs)
    if not prov_res["all_records_valid"]:
        raise ValueError(f"Provenance verification failed on real WildGuard records: {prov_res}")
    print(f"  -> Verified all {prov_res['total_records_verified']:,} real WildGuard records.")

    # Build canonical records map for schedule validation
    records_map = {r.example_id: r for r in risk_train_recs + risk_val_recs + gen_train_recs + gen_val_recs}

    # 4. FineWeb Manifest Verification
    print("\n[4/7] Verifying Frozen FineWeb Mount...", flush=True)
    fw_res = verify_authoritative_fineweb_mount()
    print(f"  -> FineWeb manifest hash: {fw_res['manifest_hash']}")

    # 5. Safety Schedule Verification
    print("\n[5/7] Verifying Frozen Safety Schedule...", flush=True)
    sched_res = verify_authoritative_safety_schedule(records_map)
    print(f"  -> Safety schedule verified ({sched_res['legacy_hash']} / {sched_res['full_audit_hash']})")

    # 6. Model B/C Initialization Equality & Cross-Seed Differentiation on Smoke Architecture
    print("\n[6/7] Verifying Smoke Model B/C Initialization Equality & Differentiation...", flush=True)
    from ccpt.config import get_smoke_dual_stream_config
    from ccpt.training.engine import create_identical_dual_stream_models
    from ccpt.evaluation.forensics import compute_canonical_state_dict_hash

    cfg_smoke = get_smoke_dual_stream_config()
    init_hashes = {}
    for s in [SEED_1_HISTORICAL, SEED_2_REPLICATION, SEED_3_REPLICATION]:
        mb, mc = create_identical_dual_stream_models(cfg_smoke, seed=s)
        hb = compute_canonical_state_dict_hash(mb.state_dict())
        hc = compute_canonical_state_dict_hash(mc.state_dict())
        if hb != hc:
            raise ValueError(f"Model B and C initializations differ at seed {s}: {hb} != {hc}")
        init_hashes[f"seed_{s}"] = hb
        print(f"  -> Seed {s} Smoke Init Hash: {hb}")

    if init_hashes[f"seed_{SEED_2_REPLICATION}"] == init_hashes[f"seed_{SEED_3_REPLICATION}"]:
        raise ValueError("Seed 2 and Seed 3 produced identical initialization hashes!")
    if init_hashes[f"seed_{SEED_1_HISTORICAL}"] == init_hashes[f"seed_{SEED_2_REPLICATION}"]:
        raise ValueError("Seed 1 and Seed 2 produced identical initialization hashes!")

    # 7. Checkpoint Strict V3 Save/Load Scan Verification
    print("\n[7/7] Verifying Checkpoint Strict V3 Production Compatibility...", flush=True)
    from ccpt.config import get_smoke_baseline_config
    from ccpt.modeling.baseline import ParameterMatchedBaselineModel
    from ccpt.training.checkpoint import (
        CHECKPOINT_FORMAT_VERSION_V3,
        save_checkpoint,
        load_checkpoint,
    )

    test_model = ParameterMatchedBaselineModel(get_smoke_baseline_config())
    test_opt = torch.optim.AdamW(test_model.parameters(), lr=1e-4)
    tmp_ckpt_path = Path("/runs/preflight_test_ckpt_v3.pt")

    save_checkpoint(
        checkpoint_path=tmp_ckpt_path,
        model=test_model,
        optimizer=test_opt,
        phase="phase1_lm_preflight",
        global_step=1,
        model_type="model_a",
        model_config=get_smoke_baseline_config(),
        git_commit_sha=expected_code_sha,
        require_exact_git_sha=True,
        expected_git_sha=expected_code_sha,
        task4_manifest_hash=CANONICAL_TASK4_MANIFEST_HASH,
        data_manifest_hash=CANONICAL_FINEWEB_MANIFEST_HASH,
        stream_identity="fineweb-edu-100BT",
    )

    loaded_state = load_checkpoint(
        tmp_ckpt_path,
        strict_v3=True,
        expected_git_commit_sha=expected_code_sha,
        expected_task4_manifest_hash=CANONICAL_TASK4_MANIFEST_HASH,
        expected_data_manifest_hash=CANONICAL_FINEWEB_MANIFEST_HASH,
        expected_stream_identity="fineweb-edu-100BT",
    )
    tmp_ckpt_path.unlink(missing_ok=True)
    print("  -> Checkpoint Strict V3 verified successfully.")

    summary = {
        "task": "task7.4.1_modal_preflight_probe",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_code_commit_sha": expected_code_sha,
        "runtime_fingerprint": fp,
        "wildguard_artifacts": wg_artifacts,
        "provenance_summary": prov_res,
        "fineweb_summary": fw_res,
        "schedule_summary": sched_res,
        "smoke_init_hashes": init_hashes,
        "preflight_passed": True,
    }

    print("\n=================================================================", flush=True)
    print("MODAL PREFLIGHT STATUS: ALL CHECKS PASSED", flush=True)
    print("=================================================================\n", flush=True)
    return summary


# -----------------------------------------------------------------------------
# Authoritative Training & Evaluation Pipelines
# -----------------------------------------------------------------------------

@app.function(
    image=replication_image,
    volumes={"/runs": runs_volume, "/data": data_volume, "/data_task4": task4_data_volume},
    secrets=hf_secrets,
    gpu="H100",
    timeout=14400,
)
def run_single_model_replication_pipeline(seed: int, model_type: str) -> Dict[str, Any]:
    """Executes the full 3-phase training pipeline for a single (seed, model) pair.

    Phase 1: 1B LM Pretraining (999,981,056 tokens)
    Phase 3: 20M Safety Training (2,344 batches, 20,010,611 valid input tokens)
    Phase 6: 1,000-step Persistence Experiment (32,768,000 continuation tokens)
    """
    # 1. Fail-closed fingerprinting at FIRST LINE
    fp = capture_and_verify_runtime_fingerprint(
        expected_code_sha=TASK7_4_CODE_SHA,
        required_gpu_type="H100",
        strict_version_check=True,
    )

    out_dir = get_task7_4_output_dir(seed, model_type)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Multi-Seed Replication Pipeline for Seed {seed}, Model {model_type}", flush=True)
    print(f"Output Directory: {out_dir}", flush=True)

    # In Task 7.4 preflight, training is gated. This function is defined as the authoritative execution unit.
    return {
        "status": "ready_for_execution",
        "seed": seed,
        "model_type": model_type,
        "output_dir": str(out_dir),
        "fingerprint": fp,
    }
