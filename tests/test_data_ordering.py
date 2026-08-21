"""Tests for deterministic epoch example ordering."""

from ccpt.data.config import DATA_ORDER_SEED
from ccpt.data.ordering import get_epoch_example_order


def test_epoch_ordering_determinism():
    """Verify that example ordering is completely deterministic and stable across calls."""
    example_ids = [f"example_{i:04d}" for i in range(100)]

    order_ep0_run1 = get_epoch_example_order(example_ids, seed=DATA_ORDER_SEED, epoch=0)
    order_ep0_run2 = get_epoch_example_order(example_ids, seed=DATA_ORDER_SEED, epoch=0)

    assert order_ep0_run1 == order_ep0_run2, "Ordering was not identical between runs!"
    assert set(order_ep0_run1) == set(example_ids), "Permutation dropped or added elements!"


def test_distinct_ordering_across_epochs():
    """Verify that different epochs produce different deterministic orderings."""
    example_ids = [f"example_{i:04d}" for i in range(100)]

    order_ep0 = get_epoch_example_order(example_ids, seed=DATA_ORDER_SEED, epoch=0)
    order_ep1 = get_epoch_example_order(example_ids, seed=DATA_ORDER_SEED, epoch=1)

    assert order_ep0 != order_ep1, "Epoch 0 and Epoch 1 should have distinct permutations"
    assert set(order_ep0) == set(order_ep1)
