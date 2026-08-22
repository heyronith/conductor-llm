"""Checkpointing system for CCPT models, training state, and data hash verification.

Supports ccpt-checkpoint-v1 and full-lineage ccpt-checkpoint-v2 with complete
environment, scheduler, data cursor, and RNG persistence.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from ccpt.config import AdapterConfig, BaselineConfig, DualStreamConfig


CHECKPOINT_FORMAT_VERSION_V1 = "ccpt-checkpoint-v1"
CHECKPOINT_FORMAT_VERSION_V2 = "ccpt-checkpoint-v2"
CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION_V2

MANDATORY_V2_BASE_FIELDS = [
    "format_version",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state",
    "global_step",
    "tokens_seen",
    "data_cursor",
    "stream_identity",
    "data_manifest_hash",
    "task4_manifest_hash",
    "model_type",
    "model_config",
    "phase",
    "training_seed",
    "torch_rng_state",
    "cuda_rng_state",
    "git_commit_sha",
    "env_versions",
]

PRODUCTION_LM_PHASES = {
    "phase1_pretrain_1b",
    "phase1_lm",
    "lm_pretrain",
    "persistence_continuation",
    "persistence_1000",
}

PRODUCTION_SAFETY_PHASES = {
    "phase3_safety_20m",
    "safety_20m",
}


def validate_checkpoint_lineage(
    checkpoint_paths: Sequence[Union[str, Path]],
    expected_task4_hash: Optional[str] = None,
    expected_training_subset_hash: Optional[str] = None,
    expected_data_manifest_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Validates unanimous manifest lineage across a collection of checkpoint files."""
    task4_hashes = set()
    subset_hashes = set()
    data_hashes = set()

    for p in checkpoint_paths:
        meta = inspect_checkpoint_metadata(p)
        t4 = meta.get("task4_manifest_hash")
        t5 = meta.get("task5_subset_hash")
        dh = meta.get("data_manifest_hash")
        if t4:
            task4_hashes.add(t4)
        if t5:
            subset_hashes.add(t5)
        if dh:
            data_hashes.add(dh)

    if len(task4_hashes) > 1:
        raise ValueError(f"Task 4 manifest hash disagreement across checkpoints: {task4_hashes}")
    if len(subset_hashes) > 1:
        raise ValueError(f"Task 5 subset hash disagreement across checkpoints: {subset_hashes}")
    if len(data_hashes) > 1:
        raise ValueError(f"Data manifest hash disagreement across checkpoints: {data_hashes}")

    if expected_task4_hash is not None and task4_hashes and next(iter(task4_hashes)) != expected_task4_hash:
        raise ValueError(f"Task 4 hash mismatch: expected {expected_task4_hash}, got {task4_hashes}")
    if expected_training_subset_hash is not None and subset_hashes and next(iter(subset_hashes)) != expected_training_subset_hash:
        raise ValueError(f"Subset hash mismatch: expected {expected_training_subset_hash}, got {subset_hashes}")
    if expected_data_manifest_hash is not None and data_hashes and next(iter(data_hashes)) != expected_data_manifest_hash:
        raise ValueError(f"Data hash mismatch: expected {expected_data_manifest_hash}, got {data_hashes}")

    return {
        "all_checkpoint_task4_hashes_match": len(task4_hashes) <= 1,
        "all_checkpoint_subset_hashes_match": len(subset_hashes) <= 1,
        "checkpoint_count_verified": len(checkpoint_paths),
    }


def get_git_commit_sha() -> str:
    """Attempts to retrieve the current git commit SHA, falling back to 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_environment_versions() -> Dict[str, str]:
    """Captures Python, PyTorch, platform, and core package versions."""
    deps = {}
    for pkg in ["torch", "transformers", "tokenizers", "datasets", "pyarrow", "numpy", "modal"]:
        try:
            deps[pkg] = importlib.metadata.version(pkg)
        except Exception:
            deps[pkg] = "not_installed"

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        **deps,
    }


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    phase: str,
    global_step: int,
    model_type: str,
    model_config: Union[BaselineConfig, DualStreamConfig, AdapterConfig, Dict[str, Any]],
    task4_manifest_hash: str = "",
    task5_subset_hash: str = "",
    training_seed: int = 20260821,
    metrics_so_far: Optional[Dict[str, Any]] = None,
    format_version: str = CHECKPOINT_FORMAT_VERSION_V2,
    tokens_seen: int = 0,
    data_cursor: int = 0,
    data_manifest_hash: str = "",
    safety_schedule_hash: str = "",
    stream_identity: str = "fineweb-edu-100BT",
    scheduler: Optional[Any] = None,
) -> Path:
    """Saves model state, optimizer state, scheduler, RNG states, and locked hashes atomically.

    Supports both V1 and full-lineage V2 formats.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = model_config if isinstance(model_config, dict) else model_config.__dict__

    scheduler_state = None
    if scheduler is not None:
        if hasattr(scheduler, "state_dict"):
            scheduler_state = scheduler.state_dict()
        else:
            scheduler_state = {
                "max_lr": getattr(scheduler, "max_lr", None),
                "min_lr": getattr(scheduler, "min_lr", None),
                "warmup_tokens": getattr(scheduler, "warmup_tokens", None),
                "total_tokens": getattr(scheduler, "total_tokens", None),
                "tokens_seen": getattr(scheduler, "tokens_seen", tokens_seen),
            }

    state = {
        "format_version": format_version,
        "phase": phase,
        "global_step": global_step,
        "model_type": model_type,
        "model_config": config_dict,
        "task4_manifest_hash": task4_manifest_hash,
        "task5_subset_hash": task5_subset_hash,
        "training_seed": training_seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler_state,
        "tokens_seen": tokens_seen,
        "data_cursor": data_cursor,
        "stream_identity": stream_identity,
        "data_manifest_hash": data_manifest_hash,
        "safety_schedule_hash": safety_schedule_hash,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "git_commit_sha": get_git_commit_sha(),
        "env_versions": get_environment_versions(),
        "metrics_so_far": metrics_so_far or {},
    }

    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

    return path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    expected_task4_manifest_hash: Optional[str] = None,
    expected_task5_subset_hash: Optional[str] = None,
    expected_data_manifest_hash: Optional[str] = None,
    expected_safety_schedule_hash: Optional[str] = None,
    expected_model_type: Optional[str] = None,
    expected_model_config: Optional[Union[BaselineConfig, DualStreamConfig, AdapterConfig, Dict[str, Any]]] = None,
    expected_stream_identity: Optional[str] = None,
    expected_phase: Optional[str] = None,
    strict_v2: bool = False,
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Dict[str, Any]:
    """Loads a checkpoint and strictly validates locked data hashes and format integrity.

    Raises ValueError loudly on any hash mismatch, missing required V2 field, or model type mismatch.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=map_location, weights_only=False)

    fmt = state.get("format_version", CHECKPOINT_FORMAT_VERSION_V1)
    if strict_v2 and fmt != CHECKPOINT_FORMAT_VERSION_V2:
        raise ValueError(f"Strict V2 loading requested, but checkpoint has format '{fmt}'")

    if strict_v2:
        for field in MANDATORY_V2_BASE_FIELDS:
            if field not in state:
                raise ValueError(f"Checkpoint V2 missing mandatory base field: '{field}'")

        ckpt_phase = state.get("phase", "")
        # Production LM Checkpoint rules
        if ckpt_phase in PRODUCTION_LM_PHASES:
            if state.get("optimizer_state_dict") is None:
                raise ValueError(f"Production LM checkpoint ({ckpt_phase}) requires non-null 'optimizer_state_dict'")
            if state.get("scheduler_state") is None:
                raise ValueError(f"Production LM checkpoint ({ckpt_phase}) requires non-null 'scheduler_state'")
            if not state.get("data_manifest_hash"):
                raise ValueError(f"Production LM checkpoint ({ckpt_phase}) requires non-empty 'data_manifest_hash'")
            if not state.get("task4_manifest_hash"):
                raise ValueError(f"Production LM checkpoint ({ckpt_phase}) requires non-empty 'task4_manifest_hash'")
            if not state.get("stream_identity"):
                raise ValueError(f"Production LM checkpoint ({ckpt_phase}) requires non-empty 'stream_identity'")
            if not state.get("model_config"):
                raise ValueError(f"Production LM checkpoint ({ckpt_phase}) requires non-empty 'model_config'")

        # Production Safety Checkpoint rules
        if ckpt_phase in PRODUCTION_SAFETY_PHASES:
            if state.get("optimizer_state_dict") is None:
                raise ValueError(f"Production Safety checkpoint ({ckpt_phase}) requires non-null 'optimizer_state_dict'")
            if state.get("scheduler_state") is None:
                raise ValueError(f"Production Safety checkpoint ({ckpt_phase}) requires non-null 'scheduler_state'")
            if not state.get("safety_schedule_hash"):
                raise ValueError(f"Production Safety checkpoint ({ckpt_phase}) requires non-empty 'safety_schedule_hash'")
            if not state.get("model_config"):
                raise ValueError(f"Production Safety checkpoint ({ckpt_phase}) requires non-empty 'model_config'")

    if expected_task4_manifest_hash is not None:
        saved_hash = state.get("task4_manifest_hash")
        if saved_hash != expected_task4_manifest_hash:
            raise ValueError(
                f"Checkpoint Task 4 manifest hash mismatch! Saved: {saved_hash}, Expected: {expected_task4_manifest_hash}"
            )

    if expected_task5_subset_hash is not None:
        saved_hash = state.get("task5_subset_hash")
        if saved_hash != expected_task5_subset_hash:
            raise ValueError(
                f"Checkpoint Task 5 subset hash mismatch! Saved: {saved_hash}, Expected: {expected_task5_subset_hash}"
            )

    if expected_data_manifest_hash is not None:
        saved_hash = state.get("data_manifest_hash")
        if saved_hash != expected_data_manifest_hash:
            raise ValueError(
                f"Checkpoint data manifest hash mismatch! Saved: {saved_hash}, Expected: {expected_data_manifest_hash}"
            )

    if expected_safety_schedule_hash is not None:
        saved_hash = state.get("safety_schedule_hash")
        if saved_hash != expected_safety_schedule_hash:
            raise ValueError(
                f"Checkpoint safety schedule hash mismatch! Saved: {saved_hash}, Expected: {expected_safety_schedule_hash}"
            )

    if expected_model_type is not None:
        saved_type = state.get("model_type")
        if saved_type != expected_model_type:
            raise ValueError(
                f"Checkpoint model_type mismatch! Saved: {saved_type}, Expected: {expected_model_type}"
            )

    if expected_stream_identity is not None:
        saved_stream = state.get("stream_identity")
        if saved_stream != expected_stream_identity:
            raise ValueError(
                f"Checkpoint stream_identity mismatch! Saved: {saved_stream}, Expected: {expected_stream_identity}"
            )

    if expected_phase is not None:
        saved_phase = state.get("phase")
        if saved_phase != expected_phase:
            raise ValueError(
                f"Checkpoint phase mismatch! Saved: {saved_phase}, Expected: {expected_phase}"
            )

    if expected_model_config is not None:
        expected_dict = expected_model_config if isinstance(expected_model_config, dict) else expected_model_config.__dict__
        saved_dict = state.get("model_config", {})
        for key in ["d_model", "n_layers", "n_heads", "vocab_size", "max_seq_len"]:
            if key in expected_dict and key in saved_dict:
                if expected_dict[key] != saved_dict[key]:
                    raise ValueError(
                        f"Checkpoint model_config mismatch on '{key}'! Saved: {saved_dict[key]}, Expected: {expected_dict[key]}"
                    )

    return state


def inspect_checkpoint_metadata(
    checkpoint_path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Dict[str, Any]:
    """Inspects checkpoint header metadata without retaining large weight tensors in memory."""
    state = load_checkpoint(checkpoint_path, map_location=map_location)
    model_state = state.get("model_state_dict", {})

    total_params = sum(t.numel() for t in model_state.values())
    total_tensors = len(model_state)

    meta = {
        "format_version": state.get("format_version", CHECKPOINT_FORMAT_VERSION_V1),
        "phase": state.get("phase"),
        "global_step": state.get("global_step"),
        "model_type": state.get("model_type"),
        "task4_manifest_hash": state.get("task4_manifest_hash"),
        "task5_subset_hash": state.get("task5_subset_hash"),
        "training_seed": state.get("training_seed"),
        "total_parameters": total_params,
        "total_tensors": total_tensors,
        "has_optimizer_state": state.get("optimizer_state_dict") is not None,
        "has_scheduler_state": state.get("scheduler_state") is not None,
        "has_rng_state": state.get("torch_rng_state") is not None,
        "metrics_so_far": state.get("metrics_so_far", {}),
    }

    if meta["format_version"] == CHECKPOINT_FORMAT_VERSION_V2:
        meta.update({
            "tokens_seen": state.get("tokens_seen", 0),
            "data_cursor": state.get("data_cursor", 0),
            "stream_identity": state.get("stream_identity", ""),
            "data_manifest_hash": state.get("data_manifest_hash", ""),
            "safety_schedule_hash": state.get("safety_schedule_hash", ""),
            "git_commit_sha": state.get("git_commit_sha", ""),
            "env_versions": state.get("env_versions", {}),
        })

    return meta
