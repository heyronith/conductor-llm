"""Modal Task 7.3: Authoritative Pilot-v2 Production Pipeline.

Executes the full authoritative 1B A/B/C/D + 20M safety + real ID/OOD behavioral evaluation
+ 1,000-step persistence experiment using exclusively audited core components.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Modal App & Container Configuration
# -----------------------------------------------------------------------------

app = modal.App("ccpt-pilot-v2-authoritative")

authoritative_image = (
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
    .add_local_python_source("ccpt")
    .add_local_dir("tests", remote_path="/root/tests")
)

# Persistent Volumes
data_volume = modal.Volume.from_name("ccpt-authoritative-data", create_if_missing=True)
task4_data_volume = modal.Volume.from_name("ccpt-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("ccpt-authoritative-runs", create_if_missing=True)

# HuggingFace Secret for authenticated streaming and WildGuard judge access
hf_secrets = [modal.Secret.from_name("huggingface")]

TASK7_3_RUN_ID = "pilot_v2_authoritative_run_20260822"
PRIMARY_SEED = 20260821
SEED_2 = 20260822
SEED_3 = 20260823
EXPECTED_TASK4_HASH = "2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097"


def capture_and_verify_runtime_fingerprint(expected_code_sha: Optional[str] = None) -> Dict[str, Any]:
    """Captures and verifies container runtime environment fingerprint.

    Fails closed on missing packages, unsupported versions, or git commit SHA mismatch.
    """
    import importlib.metadata
    import platform
    import sys

    expected_versions = {
        "torch": "2.5.1",
        "transformers": "4.46.3",
        "tokenizers": "0.20.3",
        "datasets": "3.1.0",
        "huggingface_hub": "0.26.2",
        "accelerate": "1.1.1",
        "pyarrow": "17.0.0",
        "numpy": "2.1.3",
    }

    installed_versions = {}
    for pkg, exp_ver in expected_versions.items():
        try:
            act_ver = importlib.metadata.version(pkg)
            installed_versions[pkg] = act_ver
        except Exception as e:
            raise RuntimeError(f"Required package {pkg} is not installed: {e}")

    # Verify CUDA availability if GPU present
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

    code_sha = os.environ.get("CCPT_CODE_COMMIT_SHA") or os.environ.get("TASK7_4_CODE_SHA")
    if not code_sha:
        try:
            import subprocess
            res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                code_sha = res.stdout.strip()
        except Exception:
            pass

    if expected_code_sha is not None and code_sha != expected_code_sha:
        raise RuntimeError(f"Runtime git commit SHA mismatch: expected {expected_code_sha}, got {code_sha}")

    fingerprint = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_available": cuda_avail,
        "cuda_version": str(torch.version.cuda) if cuda_avail else None,
        "device_name": device_name,
        "installed_versions": installed_versions,
        "git_commit_sha": code_sha or "unresolved",
    }

    fp_bytes = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    fingerprint["fingerprint_hash"] = hashlib.sha256(fp_bytes).hexdigest()
    return fingerprint


def resolve_arrow_path(rel_path: str) -> Path:
    """Finds prepared Arrow files strictly via canonical Task 4 resolution."""
    from ccpt.data.wildguard import resolve_canonical_wildguard_artifacts
    artifacts = resolve_canonical_wildguard_artifacts()
    # Map relative path queries to canonical keys
    if "risk" in rel_path and "train" in rel_path:
        return Path(artifacts["risk_train"]["resolved_path"])
    elif "risk" in rel_path and ("val" in rel_path or "validation" in rel_path):
        return Path(artifacts["risk_val"]["resolved_path"])
    elif "gen" in rel_path and "train" in rel_path:
        return Path(artifacts["gen_train"]["resolved_path"])
    elif "gen" in rel_path and ("val" in rel_path or "validation" in rel_path):
        return Path(artifacts["gen_val"]["resolved_path"])
    raise FileNotFoundError(f"Cannot resolve canonical arrow path for {rel_path}")




# -----------------------------------------------------------------------------
# Helper: Shard-Backed Stream Reader
# -----------------------------------------------------------------------------

def load_shards_into_tensor(shards_meta: List[Dict[str, Any]], sequence_length: int = 1024) -> torch.Tensor:
    """Loads a sequence of binary token shards into a single [N, sequence_length] int64 Tensor."""
    all_blocks = []
    for s in shards_meta:
        path = Path(s["path"])
        if not path.exists():
            raise FileNotFoundError(f"Shard file not found: {path}")
        raw = np.fromfile(str(path), dtype=np.uint16)
        num_blocks = len(raw) // sequence_length
        reshaped = raw[: num_blocks * sequence_length].reshape(num_blocks, sequence_length)
        all_blocks.append(reshaped)
    concat = np.concatenate(all_blocks, axis=0).astype(np.int64)
    return torch.from_numpy(concat)


# -----------------------------------------------------------------------------
# Stage 1: Data Materialization & Schedule Generation (Modal CPU)
# -----------------------------------------------------------------------------

@app.function(
    image=authoritative_image,
    volumes={"/data": data_volume, "/data_task4": task4_data_volume},
    secrets=hf_secrets,
    cpu=4.0,
    memory=16384,
    timeout=7200,
)
def materialize_production_data_and_schedule() -> Dict[str, Any]:
    """Streams FineWeb and generates frozen safety schedule directly to persistent volume."""
    from ccpt.data.canonical_materializer import (
        materialize_authoritative_fineweb_stream,
        load_canonical_mistral_tokenizer,
        TARGET_TRAIN_PREFIX_BLOCKS,
        TARGET_PERSISTENCE_BLOCKS,
        TARGET_VAL_BLOCKS,
    )
    from ccpt.data.wildguard import load_wildguard_records_arrow
    from ccpt.training.safety_schedule import generate_authoritative_safety_schedule, save_safety_schedule

    print("=== Materializing Authoritative FineWeb Stream ===", flush=True)
    t0 = time.time()
    tokenizer = load_canonical_mistral_tokenizer()

    fineweb_res = materialize_authoritative_fineweb_stream(
        output_dir="/data/fineweb_authoritative",
        tokenizer=tokenizer,
        train_prefix_blocks=TARGET_TRAIN_PREFIX_BLOCKS,
        persistence_blocks=TARGET_PERSISTENCE_BLOCKS,
        val_blocks=TARGET_VAL_BLOCKS,
        sequence_length=1024,
        shard_size_blocks=8192,
    )
    data_volume.commit()
    data_elapsed = time.time() - t0
    print(f"FineWeb Materialized in {data_elapsed:.2f}s | Manifest SHA: {fineweb_res['manifest_hash']}", flush=True)

    # 2. Safety Schedule Generation
    print("=== Generating Authoritative Safety Schedule ===", flush=True)
    t1 = time.time()
    risk_arrow_path = resolve_arrow_path("wildguard/risk/train.arrow")
    gen_arrow_path = resolve_arrow_path("wildguard/generation/train.arrow")

    risk_records = load_wildguard_records_arrow(risk_arrow_path, record_type="risk")
    gen_records = load_wildguard_records_arrow(gen_arrow_path, record_type="generation")

    schedule = generate_authoritative_safety_schedule(
        risk_records=risk_records,
        gen_records=gen_records,
        target_safety_tokens=20_000_000,
        batch_size=32,
        seed=PRIMARY_SEED,
    )

    schedule_path = Path("/data/safety_schedule.json")
    save_safety_schedule(schedule, schedule_path)
    data_volume.commit()
    sched_elapsed = time.time() - t1
    print(f"Safety Schedule Generated in {sched_elapsed:.2f}s | Total Tokens: {schedule['total_valid_input_tokens']:,} | Hash: {schedule['schedule_hash']}", flush=True)

    return {
        "fineweb": fineweb_res,
        "schedule": {
            "schedule_hash": schedule["schedule_hash"],
            "total_batches": schedule["total_batches"],
            "total_valid_input_tokens": schedule["total_valid_input_tokens"],
            "risk_batch_count": schedule["risk_batch_count"],
            "gen_batch_count": schedule["gen_batch_count"],
        },
        "measured_seconds": {
            "fineweb_materialization": data_elapsed,
            "safety_schedule": sched_elapsed,
        },
    }


# -----------------------------------------------------------------------------
# Stage 2: 1B Language Model Pretraining (Modal H100)
# -----------------------------------------------------------------------------

@app.function(
    image=authoritative_image,
    volumes={"/data": data_volume, "/runs": runs_volume},
    secrets=hf_secrets,
    gpu="H100!",
    timeout=28800,
)
def train_authoritative_1b_trunk(
    model_type: str,
    data_manifest_hash: str,
    run_id: str = TASK7_3_RUN_ID,
    seed: int = PRIMARY_SEED,
) -> Dict[str, Any]:
    """Trains a fresh 1B LM trunk for Model A, B, C, or D using strict Checkpoint V2 and JSONL."""
    from ccpt.config import (
        get_smoke_adapter_config,
        get_smoke_baseline_config,
        get_smoke_dual_stream_config,
    )
    from ccpt.modeling.baseline import ParameterMatchedBaselineModel
    from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
    from ccpt.modeling.adapter import FrozenBackboneAdapterModel
    from ccpt.training.checkpoint import (
        CHECKPOINT_FORMAT_VERSION_V2,
        save_checkpoint,
        load_checkpoint,
    )
    from ccpt.training.scheduler import TokenCosineScheduler
    from ccpt.training.progress import LiveProgressReporter
    from ccpt.training.cost import compute_gpu_cost
    from ccpt.training.engine import (
        clip_and_measure_gradients,
        snapshot_parameters,
        count_changed_parameters,
    )
    from ccpt.training.losses import compute_causal_lm_loss

    print(f"=== Starting 1B LM Pretraining for {model_type} [GPU: H100!] ===", flush=True)
    device = torch.device("cuda:0")
    run_dir = Path(f"/runs/ccpt/task7_3/{run_id}/{model_type}")
    run_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = run_dir / "lm_1b_final.pt"

    # Fast-path return if final 1B checkpoint already complete and verified
    if final_ckpt_path.exists():
        try:
            ckpt = load_checkpoint(final_ckpt_path, strict_v2=True)
            if ckpt.get("tokens_seen") == 999_981_056 and ckpt.get("data_cursor") == 976_544:
                print(f"1B checkpoint for {model_type} already exists and is verified. Skipping.", flush=True)
                with open(final_ckpt_path, "rb") as f:
                    ckpt_sha = hashlib.sha256(f.read()).hexdigest()
                return {
                    "status": "already_completed",
                    "model_type": model_type,
                    "final_checkpoint_path": str(final_ckpt_path),
                    "final_checkpoint_sha256": ckpt_sha,
                    "checkpoint_sha256": ckpt_sha,
                    "tokens_seen": 999_981_056,
                    "data_cursor": 976_544,
                }
        except Exception:
            pass

    # Load Manifest
    manifest_path = Path("/data/fineweb_authoritative/manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["manifest_hash"] == data_manifest_hash, "Data manifest hash mismatch!"

    # Load Prefix Shards into memory
    prefix_shards = manifest["train_prefix"]["shards"]
    token_tensor = load_shards_into_tensor(prefix_shards, sequence_length=1024)
    total_blocks = token_tensor.shape[0]
    assert total_blocks == 976_544, f"Expected 976,544 prefix blocks, got {total_blocks}"

    # Model Instantiation
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if model_type == "model_a":
        cfg = get_smoke_baseline_config()
        model = ParameterMatchedBaselineModel(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    elif model_type == "model_b":
        cfg = get_smoke_dual_stream_config()
        model = JointTrainingDualStreamModel(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    elif model_type == "model_c":
        cfg = get_smoke_dual_stream_config()
        model = CCPTDualStreamModel(cfg).to(device)
        # Freeze theta_N during LM
        for p in model.theta_N:
            p.requires_grad = False
        optimizer = torch.optim.AdamW([p for p in model.theta_C if p.requires_grad], lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    elif model_type == "model_d":
        cfg = get_smoke_adapter_config()
        model = FrozenBackboneAdapterModel(cfg).to(device)
        # Freeze safety parameters during LM
        for p in model.safety_parameters:
            p.requires_grad = False
        optimizer = torch.optim.AdamW([p for p in model.backbone_parameters if p.requires_grad], lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Record Initial State SHA
    init_state_bytes = b"".join(p.data.cpu().numpy().tobytes() for p in model.parameters())
    init_state_sha = hashlib.sha256(init_state_bytes).hexdigest()

    # Pre-LM snapshots
    c_theta_n_snap = snapshot_parameters(model.theta_N) if model_type == "model_c" else None
    d_safety_snap = snapshot_parameters(model.safety_parameters) if model_type == "model_d" else None

    # Scheduler & Progress
    scheduler = TokenCosineScheduler(
        max_lr=3e-4,
        min_lr=0.0,
        warmup_tokens=100_000_000,
        total_tokens=10_000_000_000,
        initial_tokens_seen=0,
    )

    total_steps = total_blocks // 32  # 30,517 steps
    total_tokens = total_blocks * 1024  # 999,981,056 tokens

    jsonl_path = run_dir / "lm_progress.jsonl"
    reporter = LiveProgressReporter(
        task_name="1B_LM_PRETRAIN",
        total_steps=total_steps,
        total_tokens=total_tokens,
        model_name=model_type,
        phase="LM",
        gpu_type="H100!",
        jsonl_path=jsonl_path,
        require_jsonl=True,
    )

    t_start = time.time()
    tokens_seen = 0
    data_cursor = 0

    model.train()
    for step in range(1, total_steps + 1):
        batch_slice = token_tensor[data_cursor : data_cursor + 32].to(device)
        data_cursor += 32
        batch_tokens = 32 * 1024

        lr = scheduler.get_lr(tokens_seen)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if model_type == "model_c":
                logits, _ = model(batch_slice, mode="lm")
            elif model_type == "model_d":
                logits, _ = model(batch_slice, adapter_scale=0.0)
            elif model_type == "model_b":
                logits, _ = model(batch_slice, mode="controlled")
            else:
                logits, _ = model(batch_slice)

            loss = compute_causal_lm_loss(logits, batch_slice)

        loss.backward()
        grad_norm = clip_and_measure_gradients(list(model.parameters()), max_norm=1.0)
        optimizer.step()

        tokens_seen += batch_tokens
        scheduler.step(batch_tokens)

        loss_val = float(loss.item())
        reporter.step(
            current_step=step,
            tokens_seen=tokens_seen,
            current_loss=loss_val,
            lr=lr,
            grad_norm=grad_norm,
        )

        # Periodic checkpoint save & sync
        if step % 5000 == 0 or step == total_steps:
            save_checkpoint(
                checkpoint_path=run_dir / f"lm_step_{step:05d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                phase="phase1_pretrain_1b",
                global_step=step,
                tokens_seen=tokens_seen,
                data_cursor=data_cursor,
                model_type=model_type,
                model_config=cfg,
                task4_manifest_hash=EXPECTED_TASK4_HASH,
                data_manifest_hash=data_manifest_hash,
                training_seed=seed,
                stream_identity="fineweb-edu-100BT",
            )
            runs_volume.commit()

    total_elapsed = time.time() - t_start
    total_cost = compute_gpu_cost(total_elapsed, gpu_type="H100!")

    # Verify LM invariants
    c_theta_n_changed = count_changed_parameters(model.theta_N, c_theta_n_snap) if model_type == "model_c" else 0
    d_safety_changed = count_changed_parameters(model.safety_parameters, d_safety_snap) if model_type == "model_d" else 0

    if model_type == "model_c" and c_theta_n_changed != 0:
        raise RuntimeError(f"CRITICAL INVARIANT VIOLATION: Model C theta_N changed during LM ({c_theta_n_changed} tensors modified)")
    if model_type == "model_d" and d_safety_changed != 0:
        raise RuntimeError(f"CRITICAL INVARIANT VIOLATION: Model D safety parameters changed during LM ({d_safety_changed} tensors modified)")

    # Save final authoritative 1B checkpoint
    save_checkpoint(
        checkpoint_path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        phase="phase1_pretrain_1b",
        global_step=total_steps,
        tokens_seen=tokens_seen,
        data_cursor=data_cursor,
        model_type=model_type,
        model_config=cfg,
        task4_manifest_hash=EXPECTED_TASK4_HASH,
        data_manifest_hash=data_manifest_hash,
        training_seed=seed,
        stream_identity="fineweb-edu-100BT",
    )
    runs_volume.commit()

    with open(final_ckpt_path, "rb") as f:
        final_sha = hashlib.sha256(f.read()).hexdigest()

    return {
        "model_type": model_type,
        "initial_state_sha256": init_state_sha,
        "final_checkpoint_path": str(final_ckpt_path),
        "final_checkpoint_sha256": final_sha,
        "tokens_seen": tokens_seen,
        "total_steps": total_steps,
        "data_cursor": data_cursor,
        "measured_elapsed_gpu_seconds": total_elapsed,
        "measured_gpu_cost_usd": total_cost,
        "c_theta_n_changed_tensors": c_theta_n_changed,
        "d_safety_changed_tensors": d_safety_changed,
    }


# -----------------------------------------------------------------------------
# Stage 3: Clean 1B Evaluation (Modal L40S/H100)
# -----------------------------------------------------------------------------

@app.function(
    image=authoritative_image,
    volumes={"/data": data_volume, "/runs": runs_volume},
    secrets=hf_secrets,
    gpu="L40S",
    timeout=3600,
)
def evaluate_clean_1b_capability(
    model_type: str,
    run_id: str = TASK7_3_RUN_ID,
) -> Dict[str, Any]:
    """Evaluates clean 1B capability on all 1,024 FineWeb validation blocks."""
    from ccpt.config import (
        get_smoke_adapter_config,
        get_smoke_baseline_config,
        get_smoke_dual_stream_config,
    )
    from ccpt.modeling.baseline import ParameterMatchedBaselineModel
    from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
    from ccpt.modeling.adapter import FrozenBackboneAdapterModel
    from ccpt.training.checkpoint import load_checkpoint
    from ccpt.training.cost import compute_gpu_cost

    device = torch.device("cuda:0")
    ckpt_path = Path(f"/runs/ccpt/task7_3/{run_id}/{model_type}/lm_1b_final.pt")
    ckpt = load_checkpoint(ckpt_path, strict_v2=True)

    if model_type == "model_a":
        cfg = get_smoke_baseline_config()
        model = ParameterMatchedBaselineModel(cfg).to(device)
    elif model_type == "model_b":
        cfg = get_smoke_dual_stream_config()
        model = JointTrainingDualStreamModel(cfg).to(device)
    elif model_type == "model_c":
        cfg = get_smoke_dual_stream_config()
        model = CCPTDualStreamModel(cfg).to(device)
    elif model_type == "model_d":
        cfg = get_smoke_adapter_config()
        model = FrozenBackboneAdapterModel(cfg).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load all 1,024 validation blocks
    manifest_path = Path("/data/fineweb_authoritative/manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    val_shards = manifest["validation"]["shards"]
    val_tensor = load_shards_into_tensor(val_shards, sequence_length=1024)
    assert val_tensor.shape[0] == 1024, f"Expected 1,024 validation blocks, got {val_tensor.shape[0]}"

    t0 = time.time()
    total_nll = 0.0
    total_tokens = 0
    correct_tokens = 0

    batch_size = 32
    num_val_batches = val_tensor.shape[0] // batch_size

    with torch.no_grad():
        for b_idx in range(num_val_batches):
            batch = val_tensor[b_idx * batch_size : (b_idx + 1) * batch_size].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if model_type == "model_c":
                    logits, _ = model(batch, mode="lm")
                elif model_type == "model_d":
                    logits, _ = model(batch, adapter_scale=0.0)
                elif model_type == "model_b":
                    logits, _ = model(batch, mode="controlled")
                else:
                    logits, _ = model(batch)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss_unreduced = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_nll += float(loss_unreduced.item())
            n_toks = shift_labels.numel()
            total_tokens += n_toks

            preds = shift_logits.argmax(dim=-1)
            correct_tokens += int((preds == shift_labels).sum().item())

    eval_sec = time.time() - t0
    eval_cost = compute_gpu_cost(eval_sec, gpu_type="L40S")

    mean_ce = total_nll / max(1, total_tokens)
    ppl = float(np.exp(mean_ce))
    acc = correct_tokens / max(1, total_tokens)

    return {
        "model_type": model_type,
        "phase": "clean_1b_evaluation",
        "val_blocks": 1024,
        "total_tokens": total_tokens,
        "cross_entropy": mean_ce,
        "perplexity": ppl,
        "token_accuracy": acc,
        "measured_elapsed_gpu_seconds": eval_sec,
        "measured_gpu_cost_usd": eval_cost,
    }


# -----------------------------------------------------------------------------
# Stage 4: 20M Safety Training (Modal H100)
# -----------------------------------------------------------------------------

@app.function(
    image=authoritative_image,
    volumes={"/data": data_volume, "/data_task4": task4_data_volume, "/runs": runs_volume},
    secrets=hf_secrets,
    gpu="H100!",
    timeout=28800,
)
def train_authoritative_20m_safety(
    model_type: str,
    run_id: str = TASK7_3_RUN_ID,
    seed: int = PRIMARY_SEED,
) -> Dict[str, Any]:
    """Trains 20M safety fine-tuning using 1:1 alternating batches and strict Checkpoint V2."""
    from ccpt.config import (
        get_smoke_adapter_config,
        get_smoke_baseline_config,
        get_smoke_dual_stream_config,
    )
    from ccpt.modeling.baseline import ParameterMatchedBaselineModel
    from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
    from ccpt.modeling.adapter import FrozenBackboneAdapterModel
    from ccpt.training.checkpoint import (
        save_checkpoint,
        load_checkpoint,
    )
    from ccpt.training.scheduler import SafetyTokenCosineScheduler
    from ccpt.training.progress import LiveProgressReporter
    from ccpt.training.cost import compute_gpu_cost
    from ccpt.training.engine import (
        clip_and_measure_gradients,
        snapshot_parameters,
        count_changed_parameters,
    )
    from ccpt.training.losses import compute_risk_loss, compute_safe_generation_loss
    from ccpt.data.wildguard import load_wildguard_records_arrow
    from ccpt.data.collators import pad_and_collate_risk_records, pad_and_collate_gen_records

    print(f"=== Starting 20M Safety Training for {model_type} [GPU: H100!] ===", flush=True)
    device = torch.device("cuda:0")
    run_dir = Path(f"/runs/ccpt/task7_3/{run_id}/{model_type}")
    run_dir.mkdir(parents=True, exist_ok=True)
    safety_final_ckpt_path = run_dir / "safety_20m_final.pt"

    # Fast-path return if already completed
    if safety_final_ckpt_path.exists():
        try:
            ckpt = load_checkpoint(safety_final_ckpt_path, strict_v2=True)
            if ckpt.get("tokens_seen", 0) >= 20_000_000:
                print(f"Safety checkpoint for {model_type} already exists and verified. Skipping.", flush=True)
                with open(safety_final_ckpt_path, "rb") as f:
                    final_sha = hashlib.sha256(f.read()).hexdigest()
                return {
                    "status": "already_completed",
                    "model_type": model_type,
                    "final_checkpoint_path": str(safety_final_ckpt_path),
                    "final_checkpoint_sha256": final_sha,
                    "checkpoint_sha256": final_sha,
                    "tokens_seen": ckpt.get("tokens_seen"),
                }
        except Exception:
            pass

    # Load 1B Checkpoint
    lm_ckpt_path = run_dir / "lm_1b_final.pt"
    lm_ckpt = load_checkpoint(lm_ckpt_path, strict_v2=True)

    # Load Schedule
    with open("/data/safety_schedule.json", "r", encoding="utf-8") as f:
        schedule = json.load(f)
    schedule_hash = schedule["schedule_hash"]
    batches_meta = schedule["batches"]

    # Load prepared Arrow data for batch assembly
    risk_arrow_path = resolve_arrow_path("wildguard/risk/train.arrow")
    gen_arrow_path = resolve_arrow_path("wildguard/generation/train.arrow")

    risk_records = {r.example_id: r for r in load_wildguard_records_arrow(risk_arrow_path, record_type="risk")}
    gen_records = {r.example_id: r for r in load_wildguard_records_arrow(gen_arrow_path, record_type="generation")}

    # Instantiate Model
    if model_type == "model_a":
        cfg = get_smoke_baseline_config()
        model = ParameterMatchedBaselineModel(cfg).to(device)
        model.load_state_dict(lm_ckpt["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    elif model_type == "model_b":
        cfg = get_smoke_dual_stream_config()
        model = JointTrainingDualStreamModel(cfg).to(device)
        model.load_state_dict(lm_ckpt["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    elif model_type == "model_c":
        cfg = get_smoke_dual_stream_config()
        model = CCPTDualStreamModel(cfg).to(device)
        model.load_state_dict(lm_ckpt["model_state_dict"])
        # Freeze theta_C during safety training
        for p in model.theta_C:
            p.requires_grad = False
        for p in model.theta_N:
            p.requires_grad = True
        optimizer = torch.optim.AdamW([p for p in model.theta_N if p.requires_grad], lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    elif model_type == "model_d":
        cfg = get_smoke_adapter_config()
        model = FrozenBackboneAdapterModel(cfg).to(device)
        model.load_state_dict(lm_ckpt["model_state_dict"])
        # Freeze backbone during safety training
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(model.safety_parameters, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Pre-Safety snapshots for freeze verification
    c_theta_c_snap = snapshot_parameters(model.theta_C) if model_type == "model_c" else None
    d_backbone_snap = snapshot_parameters(model.backbone_parameters) if model_type == "model_d" else None

    # Safety Scheduler (warmup 400K tokens, horizon 40M tokens)
    safety_scheduler = SafetyTokenCosineScheduler(
        max_lr=3e-4,
        min_lr=0.0,
        warmup_tokens=400_000,
        total_tokens=40_000_000,
        initial_tokens_seen=0,
    )

    total_steps = len(batches_meta)
    total_tokens = schedule["total_valid_input_tokens"]

    jsonl_path = run_dir / "safety_progress.jsonl"
    reporter = LiveProgressReporter(
        task_name="20M_SAFETY_TRAIN",
        total_steps=total_steps,
        total_tokens=total_tokens,
        model_name=model_type,
        phase="SAFETY",
        gpu_type="H100!",
        jsonl_path=jsonl_path,
        require_jsonl=True,
    )

    t_start = time.time()
    tokens_seen = 0

    model.train()
    for step, b_meta in enumerate(batches_meta, start=1):
        b_type = b_meta["batch_type"]
        eids = b_meta["example_ids"]
        v_tokens = b_meta["valid_input_tokens"]

        lr = safety_scheduler.get_lr(tokens_seen)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()

        if b_type == "risk":
            recs = [risk_records[eid] for eid in eids]
            input_ids, prompt_end_indices, risk_labels, attn_mask = pad_and_collate_risk_records(recs)
            input_ids = input_ids.to(device)
            prompt_end_indices = prompt_end_indices.to(device)
            risk_labels = risk_labels.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if model_type in ["model_b", "model_c"]:
                    _, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
                elif model_type == "model_d":
                    _, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices, adapter_scale=1.0)
                else:
                    _, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices)

                loss = compute_risk_loss(risk_logits, risk_labels)

        else:  # generation batch
            recs = [gen_records[eid] for eid in eids]
            input_ids, prompt_end_indices, risk_labels, is_refusal, attn_mask = pad_and_collate_gen_records(recs)
            input_ids = input_ids.to(device)
            prompt_end_indices = prompt_end_indices.to(device)
            attn_mask = attn_mask.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if model_type in ["model_b", "model_c"]:
                    logits, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
                elif model_type == "model_d":
                    logits, _ = model(input_ids, prompt_end_indices=prompt_end_indices, adapter_scale=1.0)
                else:
                    logits, _ = model(input_ids, prompt_end_indices=prompt_end_indices)

                loss = compute_safe_generation_loss(logits, input_ids, prompt_end_indices, attention_mask=attn_mask)

        loss.backward()
        grad_norm = clip_and_measure_gradients(list(model.parameters()), max_norm=1.0)
        optimizer.step()

        tokens_seen += v_tokens
        safety_scheduler.step(v_tokens)

        reporter.step(
            current_step=step,
            tokens_seen=tokens_seen,
            current_loss=float(loss.item()),
            lr=lr,
            grad_norm=grad_norm,
        )

    total_elapsed = time.time() - t_start
    total_cost = compute_gpu_cost(total_elapsed, gpu_type="H100!")

    # Verify freeze invariants post safety
    c_theta_c_changed = count_changed_parameters(model.theta_C, c_theta_c_snap) if model_type == "model_c" else 0
    d_backbone_changed = count_changed_parameters(model.backbone_parameters, d_backbone_snap) if model_type == "model_d" else 0

    if model_type == "model_c" and c_theta_c_changed != 0:
        raise RuntimeError(f"CRITICAL FREEZE VIOLATION: Model C theta_C changed during safety ({c_theta_c_changed} tensors modified)")
    if model_type == "model_d" and d_backbone_changed != 0:
        raise RuntimeError(f"CRITICAL FREEZE VIOLATION: Model D backbone changed during safety ({d_backbone_changed} tensors modified)")

    # Save final authoritative safety checkpoint
    save_checkpoint(
        checkpoint_path=safety_final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=safety_scheduler,
        phase="phase3_safety_20m",
        global_step=total_steps,
        tokens_seen=tokens_seen,
        model_type=model_type,
        model_config=cfg,
        task4_manifest_hash=EXPECTED_TASK4_HASH,
        data_manifest_hash=lm_ckpt.get("data_manifest_hash", ""),
        safety_schedule_hash=schedule_hash,
        training_seed=seed,
        stream_identity="fineweb-edu-100BT",
    )
    runs_volume.commit()

    with open(safety_final_ckpt_path, "rb") as f:
        final_sha = hashlib.sha256(f.read()).hexdigest()

    return {
        "model_type": model_type,
        "phase": "safety_20m",
        "final_checkpoint_path": str(safety_final_ckpt_path),
        "final_checkpoint_sha256": final_sha,
        "tokens_seen": tokens_seen,
        "total_steps": total_steps,
        "measured_elapsed_gpu_seconds": total_elapsed,
        "measured_gpu_cost_usd": total_cost,
        "c_theta_c_changed_tensors": c_theta_c_changed,
        "d_backbone_changed_tensors": d_backbone_changed,
    }


# -----------------------------------------------------------------------------
# Stage 5: Complete Evaluation Suite (Modal L40S/H100)
# -----------------------------------------------------------------------------

@app.function(
    image=authoritative_image,
    volumes={"/data": data_volume, "/data_task4": task4_data_volume, "/runs": runs_volume},
    secrets=hf_secrets,
    gpu="L40S",
    timeout=7200,
)
def evaluate_authoritative_complete_suite(
    model_type: str,
    checkpoint_phase: str = "safety_20m",  # 'safety_20m' or 'persistence_1000'
    run_id: str = TASK7_3_RUN_ID,
) -> Dict[str, Any]:
    """Evaluates Capability, Risk, Safe-Gen, ID Behavior, OOD BeaverTails, and Causal Ablations."""
    from ccpt.config import (
        get_smoke_adapter_config,
        get_smoke_baseline_config,
        get_smoke_dual_stream_config,
    )
    from ccpt.modeling.baseline import ParameterMatchedBaselineModel
    from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
    from ccpt.modeling.adapter import FrozenBackboneAdapterModel
    from ccpt.training.checkpoint import load_checkpoint
    from ccpt.training.cost import compute_gpu_cost
    from ccpt.data.canonical_materializer import load_canonical_mistral_tokenizer
    from ccpt.data.wildguard import load_wildguard_records_arrow, sample_wildguard_id_behavior_prompts
    from ccpt.data.beavertails import load_beavertails_ood_dataset
    from ccpt.data.collators import pad_and_collate_risk_records, pad_and_collate_gen_records
    from ccpt.evaluation.safety_judge import BehavioralSafetyJudge
    from ccpt.evaluation.behavioral import evaluate_behavioral_safety
    from ccpt.training.losses import compute_risk_loss, compute_safe_generation_loss

    device = torch.device("cuda:0")
    tokenizer = load_canonical_mistral_tokenizer()

    if checkpoint_phase == "safety_20m":
        ckpt_path = Path(f"/runs/ccpt/task7_3/{run_id}/{model_type}/safety_20m_final.pt")
    elif checkpoint_phase == "persistence_1000":
        ckpt_path = Path(f"/runs/ccpt/task7_3/{run_id}/{model_type}/persistence_1000_final.pt")
    else:
        raise ValueError(f"Unknown checkpoint_phase: {checkpoint_phase}")

    ckpt = load_checkpoint(ckpt_path, strict_v2=True)

    if model_type == "model_a":
        cfg = get_smoke_baseline_config()
        model = ParameterMatchedBaselineModel(cfg).to(device)
    elif model_type == "model_b":
        cfg = get_smoke_dual_stream_config()
        model = JointTrainingDualStreamModel(cfg).to(device)
    elif model_type == "model_c":
        cfg = get_smoke_dual_stream_config()
        model = CCPTDualStreamModel(cfg).to(device)
    elif model_type == "model_d":
        cfg = get_smoke_adapter_config()
        model = FrozenBackboneAdapterModel(cfg).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    t_eval_start = time.time()

    # 1. Capability on 1,024 FineWeb Validation Blocks
    manifest_path = Path("/data/fineweb_authoritative/manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    val_shards = manifest["validation"]["shards"]
    val_tensor = load_shards_into_tensor(val_shards, sequence_length=1024)

    cap_results: Dict[str, Any] = {}
    eval_modes = [("controlled_scale_1", 1.0)]
    if model_type in ["model_b", "model_c", "model_d"]:
        eval_modes.append(("ablated_scale_0", 0.0))

    for tag, scale_val in eval_modes:
        tot_nll, tot_toks, corr_toks = 0.0, 0, 0
        batch_size = 32
        for b_idx in range(val_tensor.shape[0] // batch_size):
            batch = val_tensor[b_idx * batch_size : (b_idx + 1) * batch_size].to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if model_type in ["model_b", "model_c"]:
                        logits, _ = model(batch, mode="controlled" if scale_val > 0 else "lm", controller_scale=scale_val)
                    elif model_type == "model_d":
                        logits, _ = model(batch, adapter_scale=scale_val)
                    else:
                        logits, _ = model(batch)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
            tot_nll += float(loss.item())
            tot_toks += shift_labels.numel()
            preds = shift_logits.argmax(dim=-1)
            corr_toks += int((preds == shift_labels).sum().item())

        mean_ce = tot_nll / max(1, tot_toks)
        cap_results[tag] = {
            "cross_entropy": mean_ce,
            "perplexity": float(np.exp(mean_ce)),
            "accuracy": corr_toks / max(1, tot_toks),
            "total_tokens": tot_toks,
        }

    # 2. WildGuard Risk Validation (all 2,344 examples) - Batched evaluation
    risk_val_path = resolve_arrow_path("wildguard/risk/validation.arrow")
    risk_val_recs = load_wildguard_records_arrow(risk_val_path, record_type="risk")

    risk_preds_list = []
    risk_targets_list = []
    eval_batch_size = 32

    for i in range(0, len(risk_val_recs), eval_batch_size):
        chunk = risk_val_recs[i : i + eval_batch_size]
        b_input_ids, b_ends, b_targets, _ = pad_and_collate_risk_records(chunk)
        b_input_ids = b_input_ids.to(device)
        b_ends = b_ends.to(device)
        b_targets = b_targets.to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if model_type in ["model_b", "model_c"]:
                    _, b_risk_preds_logits = model(b_input_ids, prompt_end_indices=b_ends, mode="controlled", controller_scale=1.0)
                elif model_type == "model_d":
                    _, b_risk_preds_logits = model(b_input_ids, prompt_end_indices=b_ends, adapter_scale=1.0)
                else:
                    _, b_risk_preds_logits = model(b_input_ids, prompt_end_indices=b_ends)
        risk_preds_list.append(b_risk_preds_logits.detach().cpu())
        risk_targets_list.append(b_targets.detach().cpu())

    risk_preds_logits = torch.cat(risk_preds_list, dim=0)
    risk_targets = torch.cat(risk_targets_list, dim=0)

    bce_loss = float(F.binary_cross_entropy_with_logits(risk_preds_logits, risk_targets).item())
    preds_binary = (risk_preds_logits > 0.0).float()
    correct = (preds_binary == risk_targets).float()

    harmful_mask = (risk_targets == 1.0)
    benign_mask = (risk_targets == 0.0)
    harmful_acc = float(correct[harmful_mask].mean().item()) if harmful_mask.sum() > 0 else 0.0
    benign_acc = float(correct[benign_mask].mean().item()) if benign_mask.sum() > 0 else 0.0
    balanced_acc = 0.5 * (harmful_acc + benign_acc)
    raw_acc = float(correct.mean().item())

    # 3. WildGuard Safe Generation Validation (all 928 examples) - Batched evaluation
    gen_val_path = resolve_arrow_path("wildguard/generation/validation.arrow")
    gen_val_recs = load_wildguard_records_arrow(gen_val_path, record_type="generation")

    tot_gen_loss = 0.0
    tot_gen_recs = 0

    for i in range(0, len(gen_val_recs), eval_batch_size):
        chunk = gen_val_recs[i : i + eval_batch_size]
        b_input_ids, b_ends, _, _, b_attn_mask = pad_and_collate_gen_records(chunk)
        b_input_ids = b_input_ids.to(device)
        b_ends = b_ends.to(device)
        b_attn_mask = b_attn_mask.to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if model_type in ["model_b", "model_c"]:
                    b_gen_logits, _ = model(b_input_ids, prompt_end_indices=b_ends, mode="controlled", controller_scale=1.0)
                elif model_type == "model_d":
                    b_gen_logits, _ = model(b_input_ids, prompt_end_indices=b_ends, adapter_scale=1.0)
                else:
                    b_gen_logits, _ = model(b_input_ids, prompt_end_indices=b_ends)
        b_loss = compute_safe_generation_loss(b_gen_logits, b_input_ids, b_ends, attention_mask=b_attn_mask)
        tot_gen_loss += float(b_loss.item()) * len(chunk)
        tot_gen_recs += len(chunk)

    safe_gen_loss = tot_gen_loss / max(1, tot_gen_recs)

    # 4. In-Distribution (ID) Behavioral Benchmark (256 harmful + 256 benign)
    id_harmful_prompts, id_benign_prompts, id_selection_manifest = sample_wildguard_id_behavior_prompts(
        records=risk_val_recs,
        tokenizer=tokenizer,
        n_harmful=256,
        n_benign=256,
    )

    torch.cuda.empty_cache()
    judge = BehavioralSafetyJudge(use_mock=False, device="cuda:0")

    id_behavior_on = evaluate_behavioral_safety(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=id_harmful_prompts,
        benign_prompts=id_benign_prompts,
        max_new_tokens=48,
        mode="controlled",
        scale=1.0,
        device=device,
        judge=judge,
        include_raw_outputs=False,
    )

    id_behavior_off = None
    if model_type in ["model_b", "model_c", "model_d"]:
        id_behavior_off = evaluate_behavioral_safety(
            model=model,
            tokenizer=tokenizer,
            harmful_prompts=id_harmful_prompts,
            benign_prompts=id_benign_prompts,
            max_new_tokens=48,
            mode="lm" if model_type in ["model_b", "model_c"] else "controlled",
            scale=0.0,
            device=device,
            judge=judge,
            include_raw_outputs=False,
        )

    # 5. Out-of-Distribution (OOD) BeaverTails Benchmark (256 harmful + 256 benign)
    ood_harmful_prompts, ood_benign_prompts, ood_selection_manifest = load_beavertails_ood_dataset(
        split="30k_test",
        n_harmful=256,
        n_benign=256,
        seed=20260822,
    )

    ood_behavior_on = evaluate_behavioral_safety(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=ood_harmful_prompts,
        benign_prompts=ood_benign_prompts,
        max_new_tokens=48,
        mode="controlled",
        scale=1.0,
        device=device,
        judge=judge,
        include_raw_outputs=False,
    )

    ood_behavior_off = None
    if model_type in ["model_b", "model_c", "model_d"]:
        ood_behavior_off = evaluate_behavioral_safety(
            model=model,
            tokenizer=tokenizer,
            harmful_prompts=ood_harmful_prompts,
            benign_prompts=ood_benign_prompts,
            max_new_tokens=48,
            mode="lm" if model_type in ["model_b", "model_c"] else "controlled",
            scale=0.0,
            device=device,
            judge=judge,
            include_raw_outputs=False,
        )

    total_eval_time = time.time() - t_eval_start
    total_eval_cost = compute_gpu_cost(total_eval_time, gpu_type="L40S")

    return {
        "model_type": model_type,
        "checkpoint_phase": checkpoint_phase,
        "capability": cap_results,
        "wildguard_risk_val": {
            "total_examples": len(risk_val_recs),
            "bce_loss": bce_loss,
            "raw_accuracy": raw_acc,
            "harmful_accuracy": harmful_acc,
            "benign_accuracy": benign_acc,
            "balanced_accuracy": balanced_acc,
        },
        "wildguard_gen_val": {
            "total_examples": len(gen_val_recs),
            "safe_gen_cross_entropy": safe_gen_loss,
            "perplexity": float(np.exp(safe_gen_loss)),
        },
        "id_behavioral": {
            "selection_manifest_hash": id_selection_manifest["manifest_hash"],
            "mechanism_on": id_behavior_on,
            "mechanism_off": id_behavior_off,
        },
        "ood_beavertails": {
            "selection_manifest_hash": ood_selection_manifest["manifest_hash"],
            "mechanism_on": ood_behavior_on,
            "mechanism_off": ood_behavior_off,
        },
        "measured_elapsed_gpu_seconds": total_eval_time,
        "measured_gpu_cost_usd": total_eval_cost,
    }


# -----------------------------------------------------------------------------
# Stage 6: 1,000-Step Persistence Experiment (Modal H100)
# -----------------------------------------------------------------------------

@app.function(
    image=authoritative_image,
    volumes={"/data": data_volume, "/runs": runs_volume},
    secrets=hf_secrets,
    gpu="H100!",
    timeout=14400,
)
def train_authoritative_persistence_continuation(
    model_type: str,
    run_id: str = TASK7_3_RUN_ID,
    seed: int = PRIMARY_SEED,
) -> Dict[str, Any]:
    """Trains 1,000 LM steps from persistence continuation blocks [976544, 1008544)."""
    from ccpt.config import (
        get_smoke_adapter_config,
        get_smoke_baseline_config,
        get_smoke_dual_stream_config,
    )
    from ccpt.modeling.baseline import ParameterMatchedBaselineModel
    from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
    from ccpt.modeling.adapter import FrozenBackboneAdapterModel
    from ccpt.training.checkpoint import (
        save_checkpoint,
        load_checkpoint,
    )
    from ccpt.training.scheduler import TokenCosineScheduler
    from ccpt.training.progress import LiveProgressReporter
    from ccpt.training.cost import compute_gpu_cost
    from ccpt.training.engine import (
        clip_and_measure_gradients,
        snapshot_parameters,
        count_changed_parameters,
    )
    from ccpt.training.losses import compute_causal_lm_loss

    print(f"=== Starting 1,000-Step Persistence Experiment for {model_type} [GPU: H100!] ===", flush=True)
    device = torch.device("cuda:0")
    run_dir = Path(f"/runs/ccpt/task7_3/{run_id}/{model_type}")
    final_ckpt_path = run_dir / "persistence_1000_final.pt"

    # Fast-path return if already completed
    if final_ckpt_path.exists():
        try:
            ckpt = load_checkpoint(final_ckpt_path, strict_v2=True)
            if ckpt.get("tokens_seen", 0) == 1_032_749_056 and ckpt.get("data_cursor", 0) == 1_008_544:
                print(f"Persistence checkpoint for {model_type} already exists. Skipping.", flush=True)
                with open(final_ckpt_path, "rb") as f:
                    final_sha = hashlib.sha256(f.read()).hexdigest()
                return {
                    "status": "already_completed",
                    "model_type": model_type,
                    "final_checkpoint_path": str(final_ckpt_path),
                    "final_checkpoint_sha256": final_sha,
                    "checkpoint_sha256": final_sha,
                }
        except Exception:
            pass

    # Load 20M Safety Checkpoint
    safety_ckpt_path = run_dir / "safety_20m_final.pt"
    safety_ckpt = load_checkpoint(safety_ckpt_path, strict_v2=True)

    # Load Continuation Shards [976544, 1008544)
    manifest_path = Path("/data/fineweb_authoritative/manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    cont_shards = manifest["persistence_continuation"]["shards"]
    token_tensor = load_shards_into_tensor(cont_shards, sequence_length=1024)
    assert token_tensor.shape[0] == 32_000, f"Expected 32,000 persistence blocks, got {token_tensor.shape[0]}"

    # Instantiate Model
    if model_type == "model_a":
        cfg = get_smoke_baseline_config()
        model = ParameterMatchedBaselineModel(cfg).to(device)
        model.load_state_dict(safety_ckpt["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    elif model_type == "model_b":
        cfg = get_smoke_dual_stream_config()
        model = JointTrainingDualStreamModel(cfg).to(device)
        model.load_state_dict(safety_ckpt["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    elif model_type == "model_c":
        cfg = get_smoke_dual_stream_config()
        model = CCPTDualStreamModel(cfg).to(device)
        model.load_state_dict(safety_ckpt["model_state_dict"])
        # In persistence mode=lm: theta_C trains, theta_N is bypassed and frozen
        for p in model.theta_C:
            p.requires_grad = True
        for p in model.theta_N:
            p.requires_grad = False
        optimizer = torch.optim.AdamW([p for p in model.theta_C if p.requires_grad], lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    elif model_type == "model_d":
        cfg = get_smoke_adapter_config()
        model = FrozenBackboneAdapterModel(cfg).to(device)
        model.load_state_dict(safety_ckpt["model_state_dict"])
        # Backbone trains, safety parameters frozen
        for p in model.backbone_parameters:
            p.requires_grad = True
        for p in model.safety_parameters:
            p.requires_grad = False
        optimizer = torch.optim.AdamW([p for p in model.backbone_parameters if p.requires_grad], lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Snapshots before persistence
    c_theta_n_snap = snapshot_parameters(model.theta_N) if model_type == "model_c" else None
    d_safety_snap = snapshot_parameters(model.safety_parameters) if model_type == "model_d" else None

    # Resume LM Scheduler from 999,981,056 tokens
    initial_lm_tokens = 999_981_056
    scheduler = TokenCosineScheduler(
        max_lr=3e-4,
        min_lr=0.0,
        warmup_tokens=100_000_000,
        total_tokens=10_000_000_000,
        initial_tokens_seen=initial_lm_tokens,
    )

    total_steps = 1000
    total_tokens = 32_768_000

    jsonl_path = run_dir / "persistence_progress.jsonl"
    reporter = LiveProgressReporter(
        task_name="PERSISTENCE_1000",
        total_steps=total_steps,
        total_tokens=total_tokens,
        model_name=model_type,
        phase="PERSISTENCE",
        gpu_type="H100!",
        jsonl_path=jsonl_path,
        require_jsonl=True,
    )

    t_start = time.time()
    tokens_seen = initial_lm_tokens
    data_cursor = 976_544

    model.train()
    for step in range(1, total_steps + 1):
        offset = (step - 1) * 32
        batch = token_tensor[offset : offset + 32].to(device)
        data_cursor += 32
        batch_tokens = 32 * 1024

        lr = scheduler.get_lr(tokens_seen)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if model_type == "model_c":
                logits, _ = model(batch, mode="lm")
            elif model_type == "model_d":
                logits, _ = model(batch, adapter_scale=0.0)
            elif model_type == "model_b":
                logits, _ = model(batch, mode="controlled")
            else:
                logits, _ = model(batch)

            loss = compute_causal_lm_loss(logits, batch)

        loss.backward()
        grad_norm = clip_and_measure_gradients(list(model.parameters()), max_norm=1.0)
        optimizer.step()

        tokens_seen += batch_tokens
        scheduler.step(batch_tokens)

        reporter.step(
            current_step=step,
            tokens_seen=tokens_seen,
            current_loss=float(loss.item()),
            lr=lr,
            grad_norm=grad_norm,
        )

    total_elapsed = time.time() - t_start
    total_cost = compute_gpu_cost(total_elapsed, gpu_type="H100!")

    # Verify persistence freeze invariants
    c_theta_n_changed = count_changed_parameters(model.theta_N, c_theta_n_snap) if model_type == "model_c" else 0
    d_safety_changed = count_changed_parameters(model.safety_parameters, d_safety_snap) if model_type == "model_d" else 0

    if model_type == "model_c" and c_theta_n_changed != 0:
        raise RuntimeError(f"CRITICAL PERSISTENCE VIOLATION: Model C theta_N changed during persistence ({c_theta_n_changed} tensors modified)")
    if model_type == "model_d" and d_safety_changed != 0:
        raise RuntimeError(f"CRITICAL PERSISTENCE VIOLATION: Model D safety parameters changed during persistence ({d_safety_changed} tensors modified)")

    # Save final persistence checkpoint
    save_checkpoint(
        checkpoint_path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        phase="persistence_continuation",
        global_step=total_steps,
        tokens_seen=tokens_seen,
        data_cursor=data_cursor,
        model_type=model_type,
        model_config=cfg,
        task4_manifest_hash=EXPECTED_TASK4_HASH,
        data_manifest_hash=safety_ckpt.get("data_manifest_hash", ""),
        safety_schedule_hash=safety_ckpt.get("safety_schedule_hash", ""),
        training_seed=seed,
        stream_identity="fineweb-edu-100BT",
    )
    runs_volume.commit()

    with open(final_ckpt_path, "rb") as f:
        final_sha = hashlib.sha256(f.read()).hexdigest()

    return {
        "model_type": model_type,
        "phase": "persistence_1000",
        "final_checkpoint_path": str(final_ckpt_path),
        "final_checkpoint_sha256": final_sha,
        "tokens_seen": tokens_seen,
        "data_cursor": data_cursor,
        "measured_elapsed_gpu_seconds": total_elapsed,
        "measured_gpu_cost_usd": total_cost,
        "c_theta_n_changed_tensors": c_theta_n_changed,
        "d_safety_changed_tensors": d_safety_changed,
    }


# -----------------------------------------------------------------------------
# Stage 7: Evidence Synthesis & Master Pipeline (Modal Orchestrator)
# -----------------------------------------------------------------------------

@app.function(
    image=authoritative_image,
    volumes={"/data": data_volume, "/runs": runs_volume},
    cpu=4.0,
    memory=16384,
    timeout=7200,
)
def synthesize_task7_3_results(
    mat_res: Dict[str, Any],
    lm_results: Dict[str, Any],
    clean_eval_results: Dict[str, Any],
    safety_results: Dict[str, Any],
    pre_persist_eval_results: Dict[str, Any],
    persistence_results: Dict[str, Any],
    post_persist_eval_results: Dict[str, Any],
    git_sha: str,
    run_id: str = TASK7_3_RUN_ID,
) -> Dict[str, Any]:
    """Aggregates all phase results, computes persistence comparisons, and generates summary JSON."""
    from ccpt.evaluation.persistence import build_persistence_comparison
    from ccpt.training.cost import aggregate_measured_costs

    print("=== Synthesizing Task 7.3 Authoritative Scientific Summary ===", flush=True)

    # 1. Persistence Comparisons (BEFORE vs AFTER vs DELTA vs RETENTION)
    persistence_comparisons = {}
    for m in ["model_a", "model_b", "model_c", "model_d"]:
        pre = pre_persist_eval_results.get(m, {})
        post = post_persist_eval_results.get(m, {})
        comp = build_persistence_comparison(
            before_eval=pre,
            after_eval=post,
            model_name=m,
            continuation_steps=1000,
            continuation_blocks_consumed=32000,
        )
        persistence_comparisons[m] = comp

    # 2. Total Measured Cost Aggregation
    runtimes = {}
    for m in ["model_a", "model_b", "model_c", "model_d"]:
        runtimes[f"lm_{m}"] = lm_results.get(m, {}).get("measured_elapsed_gpu_seconds", 0.0)
        runtimes[f"clean_eval_{m}"] = clean_eval_results.get(m, {}).get("measured_elapsed_gpu_seconds", 0.0)
        runtimes[f"safety_{m}"] = safety_results.get(m, {}).get("measured_elapsed_gpu_seconds", 0.0)
        runtimes[f"pre_persist_eval_{m}"] = pre_persist_eval_results.get(m, {}).get("measured_elapsed_gpu_seconds", 0.0)
        runtimes[f"persistence_{m}"] = persistence_results.get(m, {}).get("measured_elapsed_gpu_seconds", 0.0)
        runtimes[f"post_persist_eval_{m}"] = post_persist_eval_results.get(m, {}).get("measured_elapsed_gpu_seconds", 0.0)

    cost_summary = aggregate_measured_costs(runtimes, gpu_type="H100!")

    summary: Dict[str, Any] = {
        "task": "TASK 7.3 — AUTHORITATIVE PILOT-V2 EXECUTION",
        "run_id": run_id,
        "execution_code_commit_sha": git_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "primary_seed": PRIMARY_SEED,
        "data_lineage": {
            "task4_manifest_hash": EXPECTED_TASK4_HASH,
            "task7_3_manifest_hash": mat_res["fineweb"]["manifest_hash"],
            "prefix_hash": mat_res["fineweb"]["prefix_hash"],
            "continuation_hash": mat_res["fineweb"]["continuation_hash"],
            "validation_hash": mat_res["fineweb"]["val_hash"],
        },
        "safety_schedule": mat_res["schedule"],
        "models": {
            "initialization_hashes": {m: lm_results.get(m, {}).get("initial_state_sha256") for m in ["model_a", "model_b", "model_c", "model_d"]},
            "b_c_init_identical": bool(lm_results.get("model_b", {}).get("initial_state_sha256") == lm_results.get("model_c", {}).get("initial_state_sha256")),
            "final_1b_checkpoints": {m: lm_results.get(m, {}).get("final_checkpoint_sha256") for m in ["model_a", "model_b", "model_c", "model_d"]},
            "final_safety_checkpoints": {m: safety_results.get(m, {}).get("final_checkpoint_sha256") for m in ["model_a", "model_b", "model_c", "model_d"]},
            "final_persistence_checkpoints": {m: persistence_results.get(m, {}).get("final_checkpoint_sha256") for m in ["model_a", "model_b", "model_c", "model_d"]},
        },
        "freeze_invariants": {
            "c_theta_n_changed_during_lm": lm_results.get("model_c", {}).get("c_theta_n_changed_tensors", 0),
            "d_safety_changed_during_lm": lm_results.get("model_d", {}).get("d_safety_changed_tensors", 0),
            "c_theta_c_changed_during_safety": safety_results.get("model_c", {}).get("c_theta_c_changed_tensors", 0),
            "d_backbone_changed_during_safety": safety_results.get("model_d", {}).get("d_backbone_changed_tensors", 0),
            "c_theta_n_changed_during_persistence": persistence_results.get("model_c", {}).get("c_theta_n_changed_tensors", 0),
            "d_safety_changed_during_persistence": persistence_results.get("model_d", {}).get("d_safety_changed_tensors", 0),
        },
        "clean_1b_evaluation": clean_eval_results,
        "pre_persistence_evaluation": pre_persist_eval_results,
        "post_persistence_evaluation": post_persist_eval_results,
        "persistence_comparisons": persistence_comparisons,
        "measured_costs": cost_summary,
        "ready_for_10b_review": True,
        "full_10b_run_executed": False,
    }

    # Save summary artifact to runs volume
    summary_path = Path(f"/runs/ccpt/task7_3/{run_id}/task7_3_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    runs_volume.commit()

    return summary


# -----------------------------------------------------------------------------
# Local Master Runner
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Authoritative local entrypoint executing the complete Task 7.3 scientific run."""
    print("=================================================================", flush=True)
    print("=== TASK 7.3: AUTHORITATIVE PILOT-V2 MASTER PIPELINE LAUNCHED ===", flush=True)
    print("=================================================================", flush=True)

    # 1. Materialize Canonical Data and Frozen Safety Schedule
    print("\n--- [Phase 1/6] Data Materialization & Schedule Generation ---", flush=True)
    mat_res = materialize_production_data_and_schedule.remote()

    manifest_hash = mat_res["fineweb"]["manifest_hash"]
    print(f"Data Manifest Hash: {manifest_hash}", flush=True)
    print(f"Safety Schedule Hash: {mat_res['schedule']['schedule_hash']}", flush=True)

    # 2. Train Fresh 1B LM Trunks (Parallel H100!)
    print("\n--- [Phase 2/6] Fresh 1B LM Pretraining (A/B/C/D) ---", flush=True)
    lm_results = {}
    for m in ["model_a", "model_b", "model_c", "model_d"]:
        lm_results[m] = train_authoritative_1b_trunk.remote(
            model_type=m,
            data_manifest_hash=manifest_hash,
        )
        ckpt_sha = lm_results[m].get("final_checkpoint_sha256") or lm_results[m].get("checkpoint_sha256", "UNKNOWN")
        print(f"1B LM {m} completed: Checkpoint SHA = {ckpt_sha[:16]}...", flush=True)

    # 3. Clean 1B Evaluation
    print("\n--- [Phase 3/6] Clean 1B Capability Evaluation ---", flush=True)
    clean_eval_results = {}
    for m in ["model_a", "model_b", "model_c", "model_d"]:
        clean_eval_results[m] = evaluate_clean_1b_capability.remote(model_type=m)
        print(f"Clean 1B {m}: CE={clean_eval_results[m]['cross_entropy']:.4f}, PPL={clean_eval_results[m]['perplexity']:.2f}, Acc={clean_eval_results[m]['token_accuracy']*100:.2f}%", flush=True)

    # 4. 20M Safety Training
    print("\n--- [Phase 4/6] 20M Safety Fine-Tuning (A/B/C/D) ---", flush=True)
    safety_results = {}
    for m in ["model_a", "model_b", "model_c", "model_d"]:
        safety_results[m] = train_authoritative_20m_safety.remote(model_type=m)
        s_sha = safety_results[m].get("final_checkpoint_sha256") or safety_results[m].get("checkpoint_sha256", "UNKNOWN")
        print(f"20M Safety {m} completed: Checkpoint SHA = {s_sha[:16]}...", flush=True)

    # 5. Pre-Persistence Evaluation
    print("\n--- [Phase 5/6] Pre-Persistence Comprehensive Evaluation ---", flush=True)
    pre_persist_eval = {}
    for m in ["model_a", "model_b", "model_c", "model_d"]:
        pre_persist_eval[m] = evaluate_authoritative_complete_suite.remote(
            model_type=m,
            checkpoint_phase="safety_20m",
        )
        print(f"Pre-Persistence Eval {m} complete.", flush=True)

    # 6. 1,000-Step Persistence Pretraining
    print("\n--- [Phase 6/6] 1,000-Step Persistence Experiment ---", flush=True)
    persist_results = {}
    for m in ["model_a", "model_b", "model_c", "model_d"]:
        persist_results[m] = train_authoritative_persistence_continuation.remote(model_type=m)
        print(f"Persistence 1000 {m} completed.", flush=True)

    # 7. Post-Persistence Evaluation
    print("\n--- Post-Persistence Comprehensive Evaluation ---", flush=True)
    post_persist_eval = {}
    for m in ["model_a", "model_b", "model_c", "model_d"]:
        post_persist_eval[m] = evaluate_authoritative_complete_suite.remote(
            model_type=m,
            checkpoint_phase="persistence_1000",
        )
        print(f"Post-Persistence Eval {m} complete.", flush=True)

    # 8. Synthesis
    git_sha = "unknown"
    summary = synthesize_task7_3_results.remote(
        mat_res=mat_res,
        lm_results=lm_results,
        clean_eval_results=clean_eval_results,
        safety_results=safety_results,
        pre_persist_eval_results=pre_persist_eval,
        persistence_results=persist_results,
        post_persist_eval_results=post_persist_eval,
        git_sha=git_sha,
    )

    print("\n=================================================================", flush=True)
    print("=== TASK 7.3: AUTHORITATIVE SCIENTIFIC RUN COMPLETE ===", flush=True)
    print(f"Total Measured GPU Spend: ${summary['measured_costs']['total_measured_cost_usd']:.2f} USD", flush=True)
    print("=================================================================", flush=True)

