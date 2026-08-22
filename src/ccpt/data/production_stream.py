"""Canonical production FineWeb stream and packed block generator for CCPT / Task 7.2.

Uses canonical Task 4 text normalization, document splitting, tokenization,
and token packing as the single source of truth.
"""

from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
from transformers import PreTrainedTokenizerFast

from ccpt.data.fineweb import (
    PackedTokenBuffer,
    is_validation_document,
    normalize_lm_text,
    tokenize_lm_document,
)


class CanonicalFineWebStream:
    """Streams and packs FineWeb-Edu documents into contiguous fixed-size token blocks.

    Guarantees bit-for-bit equivalence with the canonical Task 4 data specification:
    - Normalization: normalize_lm_text
    - Split: is_validation_document(doc_id, val_modulo=1000)
    - Tokenization: tokenize_lm_document
    - Packing: PackedTokenBuffer (continuous, never reset between prefix and continuation)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        sequence_length: int = 1024,
        split: str = "train",  # "train" or "validation"
        val_modulo: int = 1000,
        initial_block_index: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.split = split
        self.val_modulo = val_modulo
        self.packer = PackedTokenBuffer(sequence_length=sequence_length)
        self.total_docs_processed = 0
        self.total_docs_accepted = 0
        self.total_tokens_seen = 0
        self.current_block_index = initial_block_index

    def process_document(self, text: str, doc_id: str) -> List[np.ndarray]:
        """Processes a single raw text document and yields completed blocks if buffer fills."""
        self.total_docs_processed += 1

        is_val = is_validation_document(doc_id, val_modulo=self.val_modulo)
        if (self.split == "validation" and not is_val) or (self.split == "train" and is_val):
            return []

        norm_text = normalize_lm_text(text)
        if norm_text is None:
            return []

        self.total_docs_accepted += 1
        tokens = tokenize_lm_document(norm_text, self.tokenizer)
        self.total_tokens_seen += len(tokens)

        blocks = self.packer.add_tokens(tokens)
        self.current_block_index += len(blocks)
        return blocks

    def iter_blocks(self, dataset_iterable: Iterator[Dict[str, Any]]) -> Generator[np.ndarray, None, None]:
        """Iterates over raw HuggingFace dataset items and yields packed numpy blocks."""
        doc_count = 0
        for item in dataset_iterable:
            doc_count += 1
            doc_id = str(item.get("id", doc_count))
            text = item.get("text", "")
            blocks = self.process_document(text, doc_id)
            for blk in blocks:
                yield blk

    def state_dict(self) -> Dict[str, Any]:
        """Captures stream and packer state for deterministic resumption."""
        return {
            "sequence_length": self.sequence_length,
            "split": self.split,
            "val_modulo": self.val_modulo,
            "total_docs_processed": self.total_docs_processed,
            "total_docs_accepted": self.total_docs_accepted,
            "total_tokens_seen": self.total_tokens_seen,
            "current_block_index": self.current_block_index,
            "packer_buffer": list(self.packer.buffer),
            "packer_tokens_yielded": self.packer.total_tokens_yielded,
            "packer_blocks_yielded": self.packer.total_blocks_yielded,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores stream and packer state."""
        self.sequence_length = state_dict["sequence_length"]
        self.split = state_dict["split"]
        self.val_modulo = state_dict["val_modulo"]
        self.total_docs_processed = state_dict["total_docs_processed"]
        self.total_docs_accepted = state_dict["total_docs_accepted"]
        self.total_tokens_seen = state_dict["total_tokens_seen"]
        self.current_block_index = state_dict["current_block_index"]
        self.packer.sequence_length = self.sequence_length
        self.packer.buffer = list(state_dict["packer_buffer"])
        self.packer.total_tokens_yielded = state_dict["packer_tokens_yielded"]
        self.packer.total_blocks_yielded = state_dict["packer_blocks_yielded"]
