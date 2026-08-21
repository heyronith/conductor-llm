"""Deterministic epoch example ordering utilities using SHA256 sorting keys."""

from typing import Any, Callable, List, Sequence, TypeVar

from ccpt.data.config import DATA_ORDER_SEED
from ccpt.data.hashing import sha256_text


T = TypeVar("T")


def get_epoch_example_order(
    example_ids: Sequence[str],
    seed: int = DATA_ORDER_SEED,
    epoch: int = 0,
) -> List[str]:
    """Deterministically order example IDs for a specific epoch.

    Avoids Python RNG or platform-specific pseudorandom differences.
    """
    sorted_ids = sorted(
        example_ids,
        key=lambda eid: sha256_text(f"{seed}:{epoch}:{eid}"),
    )
    return sorted_ids


def order_records_for_epoch(
    records: Sequence[T],
    id_getter: Callable[[T], str],
    seed: int = DATA_ORDER_SEED,
    epoch: int = 0,
) -> List[T]:
    """Deterministically order generic dataset records for a specific training epoch."""
    return sorted(
        records,
        key=lambda rec: sha256_text(f"{seed}:{epoch}:{id_getter(rec)}"),
    )
