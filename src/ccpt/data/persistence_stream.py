"""Sequential FineWeb Persistence Stream Iterator for CCPT / Task 7.2.

Consumes strictly sequential logical blocks from the canonical continuation range
[start_block, start_block + count) without modulo wraparound or repetition.
"""

from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch


FUTURE_PERSISTENCE_START_BLOCK = 976_544
FUTURE_PERSISTENCE_COUNT = 32_000
FUTURE_PERSISTENCE_END_EXCLUSIVE = 1_008_544
FUTURE_PERSISTENCE_BATCH_SIZE = 32
FUTURE_PERSISTENCE_TOTAL_BATCHES = 1_000


class PersistenceBlockIterator:
    """Sequential iterator over logical FineWeb blocks for persistence experiments."""

    def __init__(
        self,
        start_block: int = FUTURE_PERSISTENCE_START_BLOCK,
        num_blocks: int = FUTURE_PERSISTENCE_COUNT,
        batch_size: int = FUTURE_PERSISTENCE_BATCH_SIZE,
        blocks_tensor_or_shards: Optional[Union[torch.Tensor, np.ndarray, Sequence[np.ndarray]]] = None,
    ) -> None:
        assert num_blocks > 0, "num_blocks must be positive"
        assert batch_size > 0, "batch_size must be positive"
        assert num_blocks % batch_size == 0, f"num_blocks ({num_blocks}) must be divisible by batch_size ({batch_size})"

        self.start_block = start_block
        self.num_blocks = num_blocks
        self.end_block_exclusive = start_block + num_blocks
        self.batch_size = batch_size
        self.total_batches = num_blocks // batch_size
        self.data_source = blocks_tensor_or_shards
        self.current_batch_index = 0
        self.current_block_index = start_block

    def get_logical_block_ids(self) -> List[int]:
        """Returns the complete list of unique sequential logical block IDs."""
        return list(range(self.start_block, self.end_block_exclusive))

    def iter_batch_indices(self) -> Generator[Tuple[int, List[int]], None, None]:
        """Yields (batch_idx, list_of_logical_block_ids) for each batch."""
        for b_idx in range(self.total_batches):
            b_start = self.start_block + b_idx * self.batch_size
            b_end = b_start + self.batch_size
            yield b_idx, list(range(b_start, b_end))

    def iter_batches(
        self,
        device: Optional[torch.device] = None,
    ) -> Generator[Tuple[int, List[int], torch.Tensor], None, None]:
        """Yields (batch_idx, block_ids, batch_tensor)."""
        if self.data_source is None:
            raise ValueError("Cannot yield batch tensors when data_source is None")

        for b_idx, block_ids in self.iter_batch_indices():
            offset = b_idx * self.batch_size
            if isinstance(self.data_source, torch.Tensor):
                batch_tensor = self.data_source[offset : offset + self.batch_size]
            elif isinstance(self.data_source, np.ndarray):
                batch_arr = self.data_source[offset : offset + self.batch_size]
                batch_tensor = torch.from_numpy(batch_arr.astype(np.int64))
            else:
                batch_arr = np.stack(self.data_source[offset : offset + self.batch_size], axis=0)
                batch_tensor = torch.from_numpy(batch_arr.astype(np.int64))

            if device is not None:
                batch_tensor = batch_tensor.to(device=device)

            self.current_batch_index = b_idx + 1
            self.current_block_index = block_ids[-1] + 1
            yield b_idx, block_ids, batch_tensor
