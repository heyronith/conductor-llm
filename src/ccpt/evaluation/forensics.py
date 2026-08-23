import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from ccpt.config import BaselineConfig, DualStreamConfig, AdapterConfig
from ccpt.modeling import (
    ParameterMatchedBaselineModel,
    JointTrainingDualStreamModel,
    CCPTDualStreamModel,
    FrozenBackboneAdapterModel,
)
from ccpt.training.cost import (
    GPU_HOURLY_PRICES,
    compute_gpu_cost,
    aggregate_measured_costs,
)


def compute_canonical_state_dict_hash(state_dict: Dict[str, torch.Tensor]) -> str:
    """Computes a deterministic cryptographic hash over sorted named state_dict tensors.
    
    Includes key name, dtype string, shape tuple, and contiguous raw tensor bytes.
    """
    hasher = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name].detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def compare_named_tensors(
    left_dict: Dict[str, torch.Tensor],
    right_dict: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Forensically compares two named state dictionaries tensor-by-tensor.
    
    Requires identical key sets, then performs exact torch.equal checks and computes max_abs_diff.
    Never defaults missing values to success.
    """
    left_keys = set(left_dict.keys())
    right_keys = set(right_dict.keys())

    keys_match = (left_keys == right_keys)
    missing_in_right = sorted(list(left_keys - right_keys))
    missing_in_left = sorted(list(right_keys - left_keys))

    common_keys = sorted(list(left_keys & right_keys))
    equal_count = 0
    changed_count = 0
    changed_names = []
    max_abs_diff = 0.0

    for k in common_keys:
        t_left = left_dict[k].detach().cpu().float()
        t_right = right_dict[k].detach().cpu().float()

        if t_left.shape != t_right.shape:
            changed_count += 1
            changed_names.append(f"{k} (shape mismatch: {tuple(t_left.shape)} vs {tuple(t_right.shape)})")
            continue

        if torch.equal(t_left, t_right):
            equal_count += 1
        else:
            changed_count += 1
            diff = float((t_left - t_right).abs().max().item())
            max_abs_diff = max(max_abs_diff, diff)
            changed_names.append(f"{k} (diff={diff:.6e})")

    exact_equal = (keys_match and changed_count == 0)

    return {
        "keys_match": keys_match,
        "total_named_tensors": len(left_keys),
        "equal_named_tensors": equal_count,
        "changed_named_tensors": changed_count,
        "changed_names": changed_names,
        "max_abs_diff": max_abs_diff,
        "exact_equal": exact_equal,
        "missing_in_right": missing_in_right,
        "missing_in_left": missing_in_left,
    }


def compute_full_schedule_audit_hash(schedule_data: Dict[str, Any]) -> str:
    """Computes the authoritative Task 7.3.1 cryptographic hash over the safety schedule.
    
    Includes batch_index, batch_type, example_ids, epoch_indices, valid_input_tokens,
    and cumulative_valid_input_tokens for EVERY batch.
    """
    batches = schedule_data.get("batches", [])
    canonical_batches = []

    for b in batches:
        canonical_batches.append({
            "batch_index": int(b["batch_index"]),
            "batch_type": str(b["batch_type"]),
            "example_ids": [str(eid) for eid in b["example_ids"]],
            "epoch_indices": [int(ep) for ep in b.get("epoch_indices", [])],
            "valid_input_tokens": int(b["valid_input_tokens"]),
            "cumulative_valid_input_tokens": int(b["cumulative_valid_input_tokens"]),
        })

    canonical_obj = {
        "total_batches": len(canonical_batches),
        "total_valid_input_tokens": int(schedule_data.get("total_valid_input_tokens", 0)),
        "batches": canonical_batches,
    }

    serialized = json.dumps(canonical_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def reconstruct_model_initialization(
    model_type: str,
    seed: int = 20260821,
) -> Tuple[nn.Module, str, Dict[str, torch.Tensor]]:
    """Reconstructs the deterministic initial model state on CPU at the given seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if model_type == "model_a":
        cfg = BaselineConfig(
            vocab_size=32000,
            n_layers=4,
            d_model=512,
            n_heads=8,
            d_ff=2496,
            max_seq_len=1024,
        )
        model = ParameterMatchedBaselineModel(cfg)
    elif model_type == "model_b":
        cfg = DualStreamConfig(
            vocab_size=32000,
            n_layers_C=4,
            d_C=512,
            n_heads_C=8,
            d_ff_C=2048,
            n_layers_N=2,
            d_N=256,
            n_heads_N=4,
            d_ff_N=1024,
            controlled_layers=[2, 4],
            max_seq_len=1024,
        )
        model = JointTrainingDualStreamModel(cfg)
    elif model_type == "model_c":
        cfg = DualStreamConfig(
            vocab_size=32000,
            n_layers_C=4,
            d_C=512,
            n_heads_C=8,
            d_ff_C=2048,
            n_layers_N=2,
            d_N=256,
            n_heads_N=4,
            d_ff_N=1024,
            controlled_layers=[2, 4],
            max_seq_len=1024,
        )
        model = CCPTDualStreamModel(cfg)
    elif model_type == "model_d":
        cfg = AdapterConfig(
            vocab_size=32000,
            n_layers=4,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            d_mid=336,
            max_seq_len=1024,
        )
        model = FrozenBackboneAdapterModel(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    init_sha = compute_canonical_state_dict_hash(state_dict)
    return model, init_sha, state_dict


def infer_identity(left_hash: Optional[str], right_hash: Optional[str]) -> Optional[bool]:
    """Infers hash identity. Returns None (UNPROVEN) if either hash is missing or None."""
    if left_hash is None or right_hash is None:
        return None
    return (left_hash == right_hash)


def infer_freeze_status(changed_count: Optional[int]) -> Optional[bool]:
    """Infers freeze invariant. Returns None (UNPROVEN) if changed_count is None."""
    if changed_count is None:
        return None
    return (changed_count == 0)

