"""Real Production-Path Resume Proof for CCPT / Task 7.2.1.

Demonstrates exact continuation and state restoration using:
- Actual canonical FineWeb stream (HuggingFaceFW/fineweb-edu@87f09149...)
- Real Mistral tokenizer (mistralai/Mistral-7B-v0.1@27d67f1b...)
- Production model architecture
- AdamW optimizer
- TokenCosineScheduler with full state serialization
- Strict Checkpoint V2 format
- Exact data cursor and tokens_seen semantics
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccpt.config import get_smoke_baseline_config
from ccpt.data.canonical_materializer import (
    FINEWEB_SOURCE_REPO,
    FINEWEB_SOURCE_REVISION,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    build_task7_2_data_manifest,
    load_canonical_mistral_tokenizer,
    materialize_bounded_canonical_fineweb_proof,
)
from ccpt.data.hashing import sha256_bytes, sha256_file, sha256_json
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION_V2,
    load_checkpoint,
    save_checkpoint,
)
from ccpt.training.scheduler import TokenCosineScheduler


def run_production_path_resume_proof(
    output_dir: Union[str, Path] = "artifacts/resume_proof",
    total_steps: int = 8,
    interrupt_step: int = 4,
    batch_size: int = 4,
    seq_len: int = 64,
    device: Optional[torch.device] = None,
    materialized_manifest: Optional[Dict[str, Any]] = None,
    document_iterable: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Executes a real production-path resume proof comparing uninterrupted vs interrupted training."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load or materialize canonical FineWeb data blocks
    tok = load_canonical_mistral_tokenizer()

    if materialized_manifest is None:
        data_dir = out_path / "canonical_data"
        mat_res = materialize_bounded_canonical_fineweb_proof(
            tokenizer=tok,
            document_iterable=document_iterable,
            output_dir=data_dir,
            prefix_blocks_target=total_steps * batch_size,
            continuation_blocks_target=16,
            val_blocks_target=8,
            sequence_length=seq_len,
            val_modulo=10,
        )
        manifest = mat_res["manifest"]
    else:
        manifest = materialized_manifest

    data_manifest_hash = manifest["manifest_hash"]
    task4_manifest_hash = "task4_canonical_lineage_v1_proof_hash"

    # Load materialized prefix shard
    prefix_shard_path = Path(manifest["train_prefix"]["shards"][0]["path"])
    raw_token_data = np.fromfile(prefix_shard_path, dtype=np.uint16)
    all_blocks = raw_token_data.reshape(-1, seq_len).astype(np.int64)
    total_blocks_available = all_blocks.shape[0]
    assert total_blocks_available >= total_steps * batch_size, f"Available blocks {total_blocks_available} < required {total_steps * batch_size}"

    cfg = get_smoke_baseline_config()
    cfg.vocab_size = 32000
    cfg.max_seq_len = seq_len

    # ==========================================
    # Branch 1: Uninterrupted Execution (Steps 0 -> total_steps)
    # ==========================================
    torch.manual_seed(20260822)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260822)

    model_uninterrupted = ParameterMatchedBaselineModel(cfg).to(device=device)
    opt_uninterrupted = torch.optim.AdamW(model_uninterrupted.parameters(), lr=1e-3, weight_decay=0.01)
    sched_uninterrupted = TokenCosineScheduler(
        max_lr=1e-3,
        min_lr=1e-5,
        warmup_tokens=2 * batch_size * seq_len,
        total_tokens=total_steps * batch_size * seq_len,
    )

    uninterrupted_step_records = []

    for step in range(total_steps):
        b_start = step * batch_size
        b_end = b_start + batch_size
        batch_blocks = all_blocks[b_start:b_end]
        batch_bytes = batch_blocks.tobytes()
        batch_sha = hashlib.sha256(batch_bytes).hexdigest()

        input_tensor = torch.from_numpy(batch_blocks).to(device=device)

        opt_uninterrupted.zero_grad()
        current_lr = sched_uninterrupted.get_lr()
        for param_group in opt_uninterrupted.param_groups:
            param_group["lr"] = current_lr

        logits, _ = model_uninterrupted(input_tensor)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), input_tensor[:, 1:].reshape(-1))
        loss.backward()
        opt_uninterrupted.step()

        tokens_in_batch = batch_size * seq_len
        sched_uninterrupted.step(tokens_in_batch)

        uninterrupted_step_records.append({
            "step": step,
            "b_start": b_start,
            "b_end": b_end,
            "batch_sha": batch_sha,
            "lr": current_lr,
            "loss": float(loss.item()),
            "tokens_seen": sched_uninterrupted.tokens_seen,
        })

    # ==========================================
    # Branch 2: Interrupted & Resumed Execution (Steps 0 -> interrupt_step, then interrupt_step -> total_steps)
    # ==========================================
    torch.manual_seed(20260822)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260822)

    model_resumed = ParameterMatchedBaselineModel(cfg).to(device=device)
    opt_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-3, weight_decay=0.01)
    sched_resumed = TokenCosineScheduler(
        max_lr=1e-3,
        min_lr=1e-5,
        warmup_tokens=2 * batch_size * seq_len,
        total_tokens=total_steps * batch_size * seq_len,
    )

    resumed_step_records = []

    # Run steps 0 -> interrupt_step
    for step in range(interrupt_step):
        b_start = step * batch_size
        b_end = b_start + batch_size
        batch_blocks = all_blocks[b_start:b_end]
        batch_bytes = batch_blocks.tobytes()
        batch_sha = hashlib.sha256(batch_bytes).hexdigest()

        input_tensor = torch.from_numpy(batch_blocks).to(device=device)

        opt_resumed.zero_grad()
        current_lr = sched_resumed.get_lr()
        for param_group in opt_resumed.param_groups:
            param_group["lr"] = current_lr

        logits, _ = model_resumed(input_tensor)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), input_tensor[:, 1:].reshape(-1))
        loss.backward()
        opt_resumed.step()

        tokens_in_batch = batch_size * seq_len
        sched_resumed.step(tokens_in_batch)

        resumed_step_records.append({
            "step": step,
            "b_start": b_start,
            "b_end": b_end,
            "batch_sha": batch_sha,
            "lr": current_lr,
            "loss": float(loss.item()),
            "tokens_seen": sched_resumed.tokens_seen,
        })

    # Save checkpoint at interrupt_step
    ckpt_path = out_path / "resume_test_checkpoint_v2.pt"
    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model_resumed,
        optimizer=opt_resumed,
        scheduler=sched_resumed,
        phase="phase1_pretrain_1b",
        global_step=interrupt_step,
        model_type="model_a",
        model_config=cfg,
        task4_manifest_hash=task4_manifest_hash,
        data_manifest_hash=data_manifest_hash,
        stream_identity="fineweb-edu-100BT",
        tokens_seen=sched_resumed.tokens_seen,
        data_cursor=interrupt_step * batch_size,
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )

    # Completely destroy model, optimizer, scheduler objects
    del model_resumed
    del opt_resumed
    del sched_resumed

    # Reload from checkpoint
    loaded_state = load_checkpoint(
        checkpoint_path=ckpt_path,
        expected_task4_manifest_hash=task4_manifest_hash,
        expected_data_manifest_hash=data_manifest_hash,
        expected_model_type="model_a",
        expected_stream_identity="fineweb-edu-100BT",
        expected_phase="phase1_pretrain_1b",
        expected_model_config=cfg,
        strict_v2=True,
    )

    # Reconstruct fresh model, optimizer, scheduler and restore state
    model_reconstructed = ParameterMatchedBaselineModel(cfg).to(device=device)
    model_reconstructed.load_state_dict(loaded_state["model_state_dict"])

    opt_reconstructed = torch.optim.AdamW(model_reconstructed.parameters(), lr=1e-3, weight_decay=0.01)
    opt_reconstructed.load_state_dict(loaded_state["optimizer_state_dict"])

    sched_reconstructed = TokenCosineScheduler(
        max_lr=1e-3,
        min_lr=1e-5,
        warmup_tokens=2 * batch_size * seq_len,
        total_tokens=total_steps * batch_size * seq_len,
    )
    sched_reconstructed.load_state_dict(loaded_state["scheduler_state"])

    # Restore RNG states
    torch.set_rng_state(loaded_state["torch_rng_state"])
    if torch.cuda.is_available() and loaded_state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(loaded_state["cuda_rng_state"])

    cursor = loaded_state["data_cursor"]
    restored_tokens_seen = loaded_state["tokens_seen"]
    start_step = cursor // batch_size

    # Check state before resumed step (step == interrupt_step, i.e., step 4)
    next_batch_start = cursor
    next_batch_end = next_batch_start + batch_size
    next_batch_blocks = all_blocks[next_batch_start:next_batch_end]
    next_batch_sha = hashlib.sha256(next_batch_blocks.tobytes()).hexdigest()

    expected_uninterrupted_record = uninterrupted_step_records[interrupt_step]
    before_step_match = {
        "logical_blocks_identical": (next_batch_start == expected_uninterrupted_record["b_start"] and next_batch_end == expected_uninterrupted_record["b_end"]),
        "batch_sha_identical": (next_batch_sha == expected_uninterrupted_record["batch_sha"]),
        "lr_identical": (sched_reconstructed.get_lr() == expected_uninterrupted_record["lr"]),
        "tokens_seen_identical": (restored_tokens_seen == uninterrupted_step_records[interrupt_step - 1]["tokens_seen"]),
        "optimizer_state_loaded": (len(opt_reconstructed.state) > 0),
        "scheduler_state_loaded": (sched_reconstructed.tokens_seen == restored_tokens_seen),
    }

    # Continue execution from interrupt_step to total_steps
    for step in range(start_step, total_steps):
        b_start = step * batch_size
        b_end = b_start + batch_size
        batch_blocks = all_blocks[b_start:b_end]
        batch_bytes = batch_blocks.tobytes()
        batch_sha = hashlib.sha256(batch_bytes).hexdigest()

        input_tensor = torch.from_numpy(batch_blocks).to(device=device)

        opt_reconstructed.zero_grad()
        current_lr = sched_reconstructed.get_lr()
        for param_group in opt_reconstructed.param_groups:
            param_group["lr"] = current_lr

        logits, _ = model_reconstructed(input_tensor)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), input_tensor[:, 1:].reshape(-1))
        loss.backward()
        opt_reconstructed.step()

        tokens_in_batch = batch_size * seq_len
        sched_reconstructed.step(tokens_in_batch)

        resumed_step_records.append({
            "step": step,
            "b_start": b_start,
            "b_end": b_end,
            "batch_sha": batch_sha,
            "lr": current_lr,
            "loss": float(loss.item()),
            "tokens_seen": sched_reconstructed.tokens_seen,
        })

    # Compare final states between uninterrupted and resumed runs
    max_model_param_diff = 0.0
    for (n1, p1), (n2, p2) in zip(model_uninterrupted.named_parameters(), model_reconstructed.named_parameters()):
        diff = (p1 - p2).abs().max().item()
        if diff > max_model_param_diff:
            max_model_param_diff = diff

    bitwise_equivalent = (max_model_param_diff == 0.0)
    logical_equivalent = (
        all(before_step_match.values())
        and (sched_uninterrupted.tokens_seen == sched_reconstructed.tokens_seen)
        and (len(uninterrupted_step_records) == len(resumed_step_records))
        and all(u["batch_sha"] == r["batch_sha"] for u, r in zip(uninterrupted_step_records, resumed_step_records))
        and all(abs(u["lr"] - r["lr"]) < 1e-12 for u, r in zip(uninterrupted_step_records, resumed_step_records))
    )

    return {
        "checkpoint_step": interrupt_step,
        "total_steps": total_steps,
        "data_source": f"{FINEWEB_SOURCE_REPO}@{FINEWEB_SOURCE_REVISION}",
        "tokenizer_source": f"{TOKENIZER_REPO}@{TOKENIZER_REVISION}",
        "data_manifest_hash": data_manifest_hash,
        "before_step_proof": before_step_match,
        "uninterrupted_step_records": uninterrupted_step_records,
        "resumed_step_records": resumed_step_records,
        "final_tokens_uninterrupted": sched_uninterrupted.tokens_seen,
        "final_tokens_resumed": sched_reconstructed.tokens_seen,
        "final_lr_uninterrupted": sched_uninterrupted.get_lr(),
        "final_lr_resumed": sched_reconstructed.get_lr(),
        "next_batch_uninterrupted_hash": uninterrupted_step_records[interrupt_step]["batch_sha"],
        "next_batch_resumed_hash": next_batch_sha,
        "max_model_param_diff": max_model_param_diff,
        "LOGICAL_RESUME_EQUIVALENT": logical_equivalent,
        "BITWISE_RESUME_EQUIVALENT": bitwise_equivalent,
    }
