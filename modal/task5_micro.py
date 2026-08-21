"""Modal Task 5 / 5.1 / 5.2: Micro-Overfit, Training Harness, and Lineage Audit.

Executes remote CPU validation, test suite, deterministic micro-subset selection,
GPU micro-training, and immutable checkpoint data lineage auditing.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import modal

# -----------------------------------------------------------------------------
# Modal App & Container Image Configuration
# -----------------------------------------------------------------------------

app = modal.App("ccpt-task5-micro")

# Pinned exact dependencies frozen in project environment
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.15.1",
        "tokenizers==0.22.2",
        "datasets==5.0.1",
        "huggingface_hub==1.28.0",
        "pyarrow==25.0.1",
        "numpy==2.4.6",
        "pytest==8.4.2",
    )
    .add_local_python_source("ccpt")
    .add_local_dir("tests", remote_path="/root/tests")
)

# Persistent Volumes
# ccpt-data: Task 4 production dataset volume (read-only research inputs)
data_volume = modal.Volume.from_name("ccpt-data", create_if_missing=True)
# ccpt-runs: Task 5 run volume for checkpoints, metrics, and manifests
run_volume = modal.Volume.from_name("ccpt-runs", create_if_missing=True)

# Locked Hashes & Constants
EXPECTED_TASK4_MANIFEST_HASH = "2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097"
SANITIZED_REVIEW_MANIFEST_HASH = "1b315015ee2e01c86da989192ea789526ec232b052a2349451611552f6935132"
TASK5_SEED = 20260821
TASK5_SUBSET_VERSION = "task5-micro-v1"


def compute_sha256_file(file_path: Path) -> str:
    """Compute SHA256 hexadecimal digest of a file content in 64KB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Remote CPU Lineage Audit (Task 5.2)
# -----------------------------------------------------------------------------

@app.function(
    image=training_image,
    volumes={"/data/ccpt": data_volume, "/runs/ccpt": run_volume},
    cpu=4.0,
    memory=8192,
    timeout=600,
)
def run_task5_lineage_audit(target_run_id: str) -> Dict[str, Any]:
    """Inspects all 9 checkpoints on ccpt-runs without modifying them, extracts data lineage hashes, and runs pytest."""
    import torch
    from ccpt.data.hashing import sha256_json

    run_dir = Path(f"/runs/ccpt/task5/{target_run_id}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"=== Task 5.2 Lineage Audit starting for run_id={target_run_id} on Modal CPU ===")

    # 1. Run full remote pytest suite
    print("Running remote pytest test suite on Modal CPU...")
    test_result = subprocess.run(
        ["python3", "-m", "pytest", "/root/tests", "-v"],
        capture_output=True,
        text=True,
    )
    print(test_result.stdout)
    if test_result.returncode != 0:
        print(test_result.stderr)
        raise RuntimeError(f"Remote pytest test suite failed with exit code {test_result.returncode}")
    print("✓ Remote pytest test suite passed completely.")

    # 2. Inspect all 9 checkpoints directly
    checkpoint_records = []
    checkpoint_metadata_map = {}
    training_subset_hashes = []
    task4_hashes = []

    models = ["model_a", "model_b", "model_c"]
    phases = ["phase1_lm", "phase2_risk", "phase3_gen"]

    for model_name in models:
        checkpoint_metadata_map[model_name] = {}
        for phase in phases:
            ckpt_path = run_dir / model_name / "checkpoints" / f"{phase}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")

            # Read file size and sha256
            size_bytes = ckpt_path.stat().st_size
            sha256_hash = compute_sha256_file(ckpt_path)

            # Open with torch.load (CPU) without modifying
            state = torch.load(ckpt_path, map_location="cpu")

            task4_h = state.get("task4_manifest_hash")
            subset_h = state.get("task5_subset_hash")
            format_ver = state.get("format_version")
            g_step = state.get("global_step")
            m_type = state.get("model_type")
            t_seed = state.get("training_seed")

            if not task4_h:
                raise ValueError(f"Checkpoint {ckpt_path} is missing 'task4_manifest_hash'")
            if not subset_h:
                raise ValueError(f"Checkpoint {ckpt_path} is missing 'task5_subset_hash'")

            task4_hashes.append(task4_h)
            training_subset_hashes.append(subset_h)

            entry = {
                "volume_path": str(ckpt_path),
                "model": model_name,
                "phase": phase,
                "global_step": g_step,
                "model_type": m_type,
                "checkpoint_format_version": format_ver,
                "task4_manifest_hash": task4_h,
                "task5_subset_hash": subset_h,
                "training_seed": t_seed,
                "sha256": sha256_hash,
                "size_bytes": size_bytes,
            }
            checkpoint_records.append(entry)
            checkpoint_metadata_map[model_name][phase] = entry

    # 3. Hard assertions
    # All 9 Task 4 hashes must match EXPECTED_TASK4_MANIFEST_HASH
    assert len(task4_hashes) == 9, f"Expected 9 checkpoints, found {len(task4_hashes)}"
    for h in task4_hashes:
        assert h == EXPECTED_TASK4_MANIFEST_HASH, (
            f"Checkpoint Task 4 hash mismatch! Expected {EXPECTED_TASK4_MANIFEST_HASH}, got {h}"
        )

    # All 9 Task 5 training subset hashes must be strictly identical
    assert len(training_subset_hashes) == 9
    first_subset_hash = training_subset_hashes[0]
    for h in training_subset_hashes:
        assert h == first_subset_hash, (
            f"Disagreement among checkpoint training subset hashes! Expected {first_subset_hash}, found {h}"
        )

    actual_training_subset_manifest_hash = first_subset_hash
    print(f"✓ All 9 checkpoints verified:")
    print(f"  - Task 4 manifest hash:          {EXPECTED_TASK4_MANIFEST_HASH}")
    print(f"  - Training subset manifest hash: {actual_training_subset_manifest_hash}")
    print(f"  - Sanitized review hash:         {SANITIZED_REVIEW_MANIFEST_HASH}")

    # 4. Construct audit JSON
    lineage_audit = {
        "audit_version": "task5.2-lineage-v1",
        "run_id": target_run_id,
        "task4_manifest_hash": EXPECTED_TASK4_MANIFEST_HASH,
        "training_subset_manifest_hash": actual_training_subset_manifest_hash,
        "sanitized_review_manifest_hash": SANITIZED_REVIEW_MANIFEST_HASH,
        "checkpoint_count": len(checkpoint_records),
        "all_task4_hashes_match": True,
        "all_training_subset_hashes_match": True,
        "checkpoint_contents_modified": False,
        "checkpoints": checkpoint_records,
    }

    # 5. Extract metrics logs for all models
    metrics_data = {}
    for model_name in models:
        metrics_file = run_dir / model_name / "metrics.jsonl"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Missing metrics file at {metrics_file}")
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data[model_name] = [json.loads(line) for line in f if line.strip()]

    # 6. Read existing summary and update lineage section
    summary_path = run_dir / "task5_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    summary["data_lineage"] = {
        "task4_manifest_hash": EXPECTED_TASK4_MANIFEST_HASH,
        "training_subset_manifest_hash": actual_training_subset_manifest_hash,
        "sanitized_review_manifest_hash": SANITIZED_REVIEW_MANIFEST_HASH,
        "checkpoint_count_verified": 9,
        "all_checkpoint_subset_hashes_match": True,
        "all_checkpoint_task4_hashes_match": True,
        "checkpoint_contents_modified": False,
    }

    # 7. Construct sanitized subset manifest with explicit lineage headers
    with open(run_dir / "task5_subset_manifest.json", "r", encoding="utf-8") as f:
        raw_sub = json.load(f)

    # Ensure no raw text / records / tokens
    for k in ["records", "tokens", "input_ids", "prompt", "response", "prompt_group_key"]:
        if k in raw_sub.get("lm", {}):
            del raw_sub["lm"][k]
        if k in raw_sub.get("risk", {}):
            del raw_sub["risk"][k]
        if k in raw_sub.get("generation", {}):
            del raw_sub["generation"][k]

    sanitized_subset = {
        "manifest_kind": "sanitized_review_manifest",
        "version": TASK5_SUBSET_VERSION,
        "seed": TASK5_SEED,
        "task4_manifest_hash": EXPECTED_TASK4_MANIFEST_HASH,
        "training_subset_manifest_hash": actual_training_subset_manifest_hash,
        "sanitized_review_manifest_hash": SANITIZED_REVIEW_MANIFEST_HASH,
        "sequence_length": 128,
        "lm": {
            "source_shard": raw_sub["lm"]["source_shard"],
            "num_sequences": raw_sub["lm"]["num_sequences"],
            "sequence_length": raw_sub["lm"]["sequence_length"],
            "total_tokens": raw_sub["lm"]["total_tokens"],
            "slices": raw_sub["lm"]["slices"],
            "logical_hash": raw_sub["lm"]["logical_hash"],
        },
        "risk": {
            "source": raw_sub["risk"]["source"],
            "num_examples": raw_sub["risk"]["num_examples"],
            "harmful_count": raw_sub["risk"]["harmful_count"],
            "benign_count": raw_sub["risk"]["benign_count"],
            "logical_hash": raw_sub["risk"]["logical_hash"],
            "example_ids": raw_sub["risk"]["example_ids"],
        },
        "generation": {
            "source": raw_sub["generation"]["source"],
            "num_examples": raw_sub["generation"]["num_examples"],
            "harmful_refusal_count": raw_sub["generation"]["harmful_refusal_count"],
            "benign_compliance_count": raw_sub["generation"]["benign_compliance_count"],
            "logical_hash": raw_sub["generation"]["logical_hash"],
            "example_ids": raw_sub["generation"]["example_ids"],
        },
    }

    # Construct checkpoint metadata artifact
    checkpoint_meta = {
        "run_id": target_run_id,
        "volume": "ccpt-runs",
        "root_path": f"/runs/ccpt/task5/{target_run_id}",
        "task4_manifest_hash": EXPECTED_TASK4_MANIFEST_HASH,
        "training_subset_manifest_hash": actual_training_subset_manifest_hash,
        "sanitized_review_manifest_hash": SANITIZED_REVIEW_MANIFEST_HASH,
        "checkpoint_lineage_verified": True,
        "checkpoints": checkpoint_metadata_map,
    }

    # Persist updated summary and manifest to volume
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(run_dir / "task5_subset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(sanitized_subset, f, indent=2, sort_keys=True)

    with open(run_dir / "task5_lineage_audit.json", "w", encoding="utf-8") as f:
        json.dump(lineage_audit, f, indent=2, sort_keys=True)

    with open(run_dir / "task5_checkpoint_metadata.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint_meta, f, indent=2, sort_keys=True)

    run_volume.commit()

    return {
        "lineage_audit": lineage_audit,
        "checkpoint_metadata": checkpoint_meta,
        "sanitized_subset_manifest": sanitized_subset,
        "summary": summary,
        "metrics_data": metrics_data,
        "pytest_output": test_result.stdout,
    }


# -----------------------------------------------------------------------------
# Local Entrypoint for Task 5.2 Audit
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Local orchestration for Task 5.2 checkpoint data lineage audit."""
    target_run_id = "run_1787321375"
    print("================================================================================")
    print(f"CCPT Task 5.2: Checkpoint Data Lineage Audit on Modal CPU")
    print(f"Target Run ID: {target_run_id}")
    print("================================================================================")

    audit_result = run_task5_lineage_audit.remote(target_run_id)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write task5_lineage_audit.json
    with open(artifacts_dir / "task5_lineage_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_result["lineage_audit"], f, indent=2, sort_keys=True)

    # 2. Write task5_checkpoint_metadata.json
    with open(artifacts_dir / "task5_checkpoint_metadata.json", "w", encoding="utf-8") as f:
        json.dump(audit_result["checkpoint_metadata"], f, indent=2, sort_keys=True)

    # 3. Write task5_subset_manifest.json
    with open(artifacts_dir / "task5_subset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(audit_result["sanitized_subset_manifest"], f, indent=2, sort_keys=True)

    # 4. Write task5_summary.json
    with open(artifacts_dir / "task5_summary.json", "w", encoding="utf-8") as f:
        json.dump(audit_result["summary"], f, indent=2, sort_keys=True)

    # 5. Write metrics logs
    metrics_dir = artifacts_dir / "task5_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for model_name, rows in audit_result["metrics_data"].items():
        with open(metrics_dir / f"{model_name}_metrics.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # 6. Generate human-readable lineage audit text
    audit = audit_result["lineage_audit"]
    summary = audit_result["summary"]
    run_log = f"""CCPT TASK 5.2 CHECKPOINT DATA LINEAGE AUDIT REPORT
======================================================================
Run ID:                         {target_run_id}
Task 4 Manifest Hash:           {audit['task4_manifest_hash']}
Training Subset Manifest Hash:  {audit['training_subset_manifest_hash']}
Sanitized Review Manifest Hash: {audit['sanitized_review_manifest_hash']}
Checkpoints Inspected:          {audit['checkpoint_count']} (All 9 verified)
Task 4 Hashes Match:            {audit['all_task4_hashes_match']}
Training Subset Hashes Match:   {audit['all_training_subset_hashes_match']}
Checkpoint Contents Modified:   {audit['checkpoint_contents_modified']}

INSPECTED CHECKPOINTS:
"""
    for ckpt in audit["checkpoints"]:
        run_log += f"  - [{ckpt['model']}/{ckpt['phase']}] (step {ckpt['global_step']}):\n"
        run_log += f"      Path:        {ckpt['volume_path']}\n"
        run_log += f"      Size:        {ckpt['size_bytes']:,} bytes\n"
        run_log += f"      SHA256:      {ckpt['sha256']}\n"
        run_log += f"      Task 4 Hash: {ckpt['task4_manifest_hash']}\n"
        run_log += f"      Subset Hash: {ckpt['task5_subset_hash']}\n"

    run_log += f"""
TASK 5 CORE TRAINING METRICS (UNCHANGED):
  - Model A Phase 1 LM:   {summary['models']['model_a']['phase1_lm']['initial_loss']:.4f} -> {summary['models']['model_a']['phase1_lm']['final_loss']:.4f} (Reduction: {summary['models']['model_a']['phase1_lm']['loss_reduction_fraction']*100:.1f}%)
  - Model A Phase 2 Risk: Acc: {summary['models']['model_a']['phase2_risk']['final_acc']*100:.1f}%
  - Model B Phase 1 LM:   {summary['models']['model_b']['phase1_lm']['initial_loss']:.4f} -> {summary['models']['model_b']['phase1_lm']['final_loss']:.4f} (Reduction: {summary['models']['model_b']['phase1_lm']['loss_reduction_fraction']*100:.1f}%)
  - Model B Phase 2 Risk: Acc: {summary['models']['model_b']['phase2_risk']['final_acc']*100:.1f}%
  - Model C Phase 1 LM:   {summary['models']['model_c']['phase1_lm']['initial_loss']:.4f} -> {summary['models']['model_c']['phase1_lm']['final_loss']:.4f} (Reduction: {summary['models']['model_c']['phase1_lm']['loss_reduction_fraction']*100:.1f}%)
  - Model C Phase 2 Risk: Acc: {summary['models']['model_c']['phase2_risk']['final_acc']*100:.1f}% (theta_C changed: {summary['models']['model_c']['phase2_risk']['theta_c_changed_tensors']})
  - Model C Phase 3 Gen:  Controlled: {summary['models']['model_c']['phase3_gen']['controlled_safe_gen_loss']:.4f}, Ablated: {summary['models']['model_c']['phase3_gen']['ablated_safe_gen_loss']:.4f} (Penalty: {summary['models']['model_c']['phase3_gen']['ablation_penalty_relative']*100:.1f}%)
  - Model C Invariant:    Causal LM allclose == {summary['models']['model_c']['phase3_gen']['causal_lm_allclose']}
  - theta_C changed tensors (Phase 3): {summary['models']['model_c']['phase3_gen']['theta_c_changed_tensors']}
  - theta_N changed tensors (Phase 3): {summary['models']['model_c']['phase3_gen']['theta_n_changed_tensors']}

FAILURE FLAGS:
  - NaN/Inf Detected:             {summary['failure_flags']['nan_or_inf_detected']}
  - Gate Collapse Detected:       {summary['failure_flags']['gate_collapse_detected']}
  - Steering Saturation Detected: {summary['failure_flags']['steering_saturation_detected']}
  - Dead Controller Detected:     {summary['failure_flags']['dead_controller_detected']}
  - Capability Mutation Detected: {summary['failure_flags']['capability_mutation_detected']}
  - Data Leakage Detected:        {summary['failure_flags']['data_leakage_detected']}
"""

    with open(artifacts_dir / "task5_modal_run.txt", "w", encoding="utf-8") as f:
        f.write(run_log)

    print("\n" + run_log)
    print("✓ Local Task 5.2 audit artifacts written to artifacts/")
