"""Authoritative Safety Schedule Generator for Task 7.3.

Generates deterministic, frozen 1:1 alternating safety batches (risk vs generation)
of size 32 with zero dropped tails across epochs. Tracks valid unpadded input token
presentations and halts at the first complete batch crossing >= 20,000,000 safety tokens.
Computes a comprehensive cryptographic schedule hash over the entire batch sequence.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ccpt.data.hashing import sha256_json
from ccpt.data.wildguard import RiskRecord, SafeGenerationRecord


TARGET_SAFETY_TOKENS = 20_000_000
SAFETY_BATCH_SIZE = 32
SAFETY_SCHEDULE_SEED = 20260821


@dataclass(frozen=True)
class ScheduledSafetyBatch:
    """Immutable record for a single scheduled safety batch."""

    batch_index: int
    batch_type: str  # 'risk' or 'generation'
    example_ids: List[str]
    valid_input_tokens: int
    cumulative_valid_input_tokens: int
    epoch_indices: List[int]


def generate_authoritative_safety_schedule(
    risk_records: Sequence[RiskRecord],
    gen_records: Sequence[SafeGenerationRecord],
    target_safety_tokens: int = TARGET_SAFETY_TOKENS,
    batch_size: int = SAFETY_BATCH_SIZE,
    seed: int = SAFETY_SCHEDULE_SEED,
) -> Dict[str, Any]:
    """Generates the full frozen Task 7.3 safety schedule with 1:1 alternation.

    Args:
        risk_records: Sequence of prepared RiskRecords (train split, 45,492).
        gen_records: Sequence of prepared SafeGenerationRecords (train split, 18,015).
        target_safety_tokens: Minimum valid input token presentations (default: 20,000,000).
        batch_size: Number of examples per batch (default: 32).
        seed: Random seed for deterministic epoch shuffling.

    Returns:
        Dictionary containing the full schedule, batches, token counts, and cryptographic schedule hash.
    """
    assert len(risk_records) > 0, "risk_records cannot be empty"
    assert len(gen_records) > 0, "gen_records cannot be empty"
    assert batch_size > 0, "batch_size must be positive"
    assert target_safety_tokens > 0, "target_safety_tokens must be positive"

    # Index maps for quick lookup of valid input token lengths
    risk_token_counts = {r.example_id: len(r.input_ids) for r in risk_records}
    gen_token_counts = {r.example_id: len(r.input_ids) for r in gen_records}

    risk_id_list = [r.example_id for r in risk_records]
    gen_id_list = [r.example_id for r in gen_records]

    # Deterministic epoch permutation generators
    rng_risk = random.Random(seed)
    rng_gen = random.Random(seed + 1)

    risk_epoch = 0
    risk_pool: List[str] = []
    risk_epoch_pool: List[int] = []

    def _refill_risk():
        nonlocal risk_epoch, risk_pool, risk_epoch_pool
        perm = list(risk_id_list)
        rng_risk.shuffle(perm)
        risk_pool.extend(perm)
        risk_epoch_pool.extend([risk_epoch] * len(perm))
        risk_epoch += 1

    gen_epoch = 0
    gen_pool: List[str] = []
    gen_epoch_pool: List[int] = []

    def _refill_gen():
        nonlocal gen_epoch, gen_pool, gen_epoch_pool
        perm = list(gen_id_list)
        rng_gen.shuffle(perm)
        gen_pool.extend(perm)
        gen_epoch_pool.extend([gen_epoch] * len(perm))
        gen_epoch += 1

    batches: List[Dict[str, Any]] = []
    cumulative_tokens = 0
    batch_idx = 0

    while cumulative_tokens < target_safety_tokens:
        # 1. Risk Batch
        while len(risk_pool) < batch_size:
            _refill_risk()

        risk_batch_ids = risk_pool[:batch_size]
        risk_batch_epochs = risk_epoch_pool[:batch_size]
        risk_pool = risk_pool[batch_size:]
        risk_epoch_pool = risk_epoch_pool[batch_size:]

        risk_batch_tokens = sum(risk_token_counts[eid] for eid in risk_batch_ids)
        cumulative_tokens += risk_batch_tokens

        batches.append({
            "batch_index": batch_idx,
            "batch_type": "risk",
            "example_ids": risk_batch_ids,
            "valid_input_tokens": risk_batch_tokens,
            "cumulative_valid_input_tokens": cumulative_tokens,
            "epoch_indices": risk_batch_epochs,
        })
        batch_idx += 1

        # Check if crossed target after risk batch
        if cumulative_tokens >= target_safety_tokens:
            break

        # 2. Generation Batch
        while len(gen_pool) < batch_size:
            _refill_gen()

        gen_batch_ids = gen_pool[:batch_size]
        gen_batch_epochs = gen_epoch_pool[:batch_size]
        gen_pool = gen_pool[batch_size:]
        gen_epoch_pool = gen_epoch_pool[batch_size:]

        gen_batch_tokens = sum(gen_token_counts[eid] for eid in gen_batch_ids)
        cumulative_tokens += gen_batch_tokens

        batches.append({
            "batch_index": batch_idx,
            "batch_type": "generation",
            "example_ids": gen_batch_ids,
            "valid_input_tokens": gen_batch_tokens,
            "cumulative_valid_input_tokens": cumulative_tokens,
            "epoch_indices": gen_batch_epochs,
        })
        batch_idx += 1

    # Compute full schedule SHA256 digest
    schedule_digest_entries = []
    full_audit_digest_entries = []
    for b in batches:
        entry_legacy = (
            f"{b['batch_index']}:{b['batch_type']}:{b['valid_input_tokens']}:"
            f"{b['cumulative_valid_input_tokens']}:{','.join(b['example_ids'])}"
        )
        schedule_digest_entries.append(entry_legacy)
        epochs_str = ",".join(str(e) for e in b.get("epoch_indices", []))

    schedule_bytes = "\n".join(schedule_digest_entries).encode("utf-8")
    schedule_hash = hashlib.sha256(schedule_bytes).hexdigest()

    risk_batch_count = sum(1 for b in batches if b["batch_type"] == "risk")
    gen_batch_count = sum(1 for b in batches if b["batch_type"] == "generation")
    total_valid_tokens = cumulative_tokens

    summary: Dict[str, Any] = {
        "schedule_version": "task7_4_safety_schedule_v2",
        "seed": seed,
        "batch_size": batch_size,
        "target_safety_tokens": target_safety_tokens,
        "total_batches": len(batches),
        "risk_batch_count": risk_batch_count,
        "gen_batch_count": gen_batch_count,
        "total_valid_input_tokens": total_valid_tokens,
        "total_risk_records_source": len(risk_records),
        "total_gen_records_source": len(gen_records),
        "risk_epochs_consumed": risk_epoch,
        "gen_epochs_consumed": gen_epoch,
        "no_tails_dropped": True,
        "schedule_hash": schedule_hash,
        "batches": batches,
    }

    summary["full_schedule_audit_hash"] = compute_full_schedule_audit_hash(summary)
    return summary


def compute_full_schedule_audit_hash(schedule_data: Dict[str, Any]) -> str:
    """Computes the authoritative canonical Task 7.3.1 cryptographic hash over the safety schedule.
    
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


def save_safety_schedule(schedule: Dict[str, Any], output_path: Union[str, Path]) -> str:
    """Saves safety schedule JSON to disk and returns file SHA256."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2)
    return sha256_json(schedule)
