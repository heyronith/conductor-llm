"""Checkpointing system for CCPT models, training state, and data hash verification."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


import torch
import torch.nn as nn
from ccpt.config import BaselineConfig, DualStreamConfig


CHECKPOINT_FORMAT_VERSION = "ccpt-checkpoint-v1"


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    phase: str,
    global_step: int,
    model_type: str,
    model_config: Union[BaselineConfig, DualStreamConfig, Dict[str, Any]],
    task4_manifest_hash: str,
    task5_subset_hash: str,
    training_seed: int,
    metrics_so_far: Optional[Dict[str, Any]] = None,
) -> Path:
    """Saves model state, optimizer state, RNG states, and locked data hashes atomically.

    Args:
        checkpoint_path: Destination path for the checkpoint file.
        model: PyTorch model to save.
        optimizer: PyTorch optimizer or None.
        phase: Current training phase name (e.g. 'phase1_lm', 'phase2_risk', 'phase3_gen').
        global_step: Global optimizer step count.
        model_type: Identifier string ('model_a', 'model_b', or 'model_c').
        model_config: Model configuration object or dictionary.
        task4_manifest_hash: SHA256 hash of the Task 4 data manifest.
        task5_subset_hash: SHA256 hash of the Task 5 subset manifest.
        training_seed: Global training seed.
        metrics_so_far: Optional dictionary of aggregated metrics.

    Returns:
        Path to the saved checkpoint file.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = model_config if isinstance(model_config, dict) else model_config.__dict__

    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "phase": phase,
        "global_step": global_step,
        "model_type": model_type,
        "model_config": config_dict,
        "task4_manifest_hash": task4_manifest_hash,
        "task5_subset_hash": task5_subset_hash,
        "training_seed": training_seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "metrics_so_far": metrics_so_far or {},
    }

    # Atomic write pattern: save to temporary file then rename
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

    return path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    expected_task4_manifest_hash: Optional[str] = None,
    expected_task5_subset_hash: Optional[str] = None,
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Dict[str, Any]:
    """Loads a checkpoint and strictly validates locked data hashes.

    Args:
        checkpoint_path: Path to checkpoint file.
        expected_task4_manifest_hash: Expected Task 4 manifest hash. If provided, raises ValueError on mismatch.
        expected_task5_subset_hash: Expected Task 5 subset hash. If provided, raises ValueError on mismatch.
        map_location: PyTorch device mapping location.

    Returns:
        Loaded state dictionary.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=map_location)

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

    return state


def inspect_checkpoint_metadata(

    checkpoint_path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Dict[str, Any]:
    """Inspects and extracts metadata fields from a saved checkpoint file without modifying it.

    Returns:
        Dictionary containing extracted metadata fields (task4_manifest_hash, task5_subset_hash,
        phase, global_step, model_type, format_version, etc.).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=map_location)
    return {
        "format_version": state.get("format_version"),
        "phase": state.get("phase"),
        "global_step": state.get("global_step"),
        "model_type": state.get("model_type"),
        "task4_manifest_hash": state.get("task4_manifest_hash"),
        "task5_subset_hash": state.get("task5_subset_hash"),
        "training_seed": state.get("training_seed"),
        "metrics_so_far": state.get("metrics_so_far", {}),
    }


def validate_checkpoint_lineage(
    checkpoint_paths: Sequence[Union[str, Path]],
    expected_task4_hash: str,
    expected_training_subset_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Validates that a group of checkpoints strictly agree on data lineage hashes.

    Args:
        checkpoint_paths: List of paths to checkpoint files.
        expected_task4_hash: Expected Task 4 manifest hash.
        expected_training_subset_hash: Optional expected training-time subset hash. If None,
            the subset hash from the first checkpoint is used and checked for unanimous agreement.

    Returns:
        Summary dictionary with verified lineage hashes and per-checkpoint metadata.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for lineage validation")

    inspected = []
    first_subset_hash = None

    for p in checkpoint_paths:
        meta = inspect_checkpoint_metadata(p)
        task4_h = meta.get("task4_manifest_hash")
        subset_h = meta.get("task5_subset_hash")

        if not task4_h:
            raise ValueError(f"Checkpoint {p} is missing 'task4_manifest_hash'")
        if not subset_h:
            raise ValueError(f"Checkpoint {p} is missing 'task5_subset_hash'")

        if task4_h != expected_task4_hash:
            raise ValueError(
                f"Checkpoint {p} Task 4 hash mismatch! Found: {task4_h}, Expected: {expected_task4_hash}"
            )

        if expected_training_subset_hash is not None:
            if subset_h != expected_training_subset_hash:
                raise ValueError(
                    f"Checkpoint {p} Task 5 subset hash mismatch! Found: {subset_h}, Expected: {expected_training_subset_hash}"
                )
        else:
            if first_subset_hash is None:
                first_subset_hash = subset_h
            elif subset_h != first_subset_hash:
                raise ValueError(
                    f"Checkpoint {p} Task 5 subset hash disagreement! Found: {subset_h}, First checkpoint had: {first_subset_hash}"
                )

        inspected.append({
            "path": str(p),
            "model_type": meta.get("model_type"),
            "phase": meta.get("phase"),
            "global_step": meta.get("global_step"),
            "task4_manifest_hash": task4_h,
            "task5_subset_hash": subset_h,
        })

    verified_subset_hash = expected_training_subset_hash or first_subset_hash

    return {
        "task4_manifest_hash": expected_task4_hash,
        "training_subset_manifest_hash": verified_subset_hash,
        "checkpoint_count_verified": len(checkpoint_paths),
        "all_checkpoint_task4_hashes_match": True,
        "all_checkpoint_subset_hashes_match": True,
        "checkpoints": inspected,
    }

