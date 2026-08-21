"""FineWeb-Edu causal language modeling data pipeline and shard serializer."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
from transformers import PreTrainedTokenizerFast

from ccpt.data.config import DataConfig
from ccpt.data.hashing import sha256_bytes, sha256_file, sha256_text, stable_hash_int


def normalize_lm_text(text: str) -> Optional[str]:
    """Apply minimal text normalization to LM documents.

    Replaces CRLF and CR with LF. Rejects empty or whitespace-only documents.
    Preserves all casing, internal whitespace, and punctuation.
    """
    if not text:
        return None
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return None
    return normalized


def is_validation_document(doc_id: str, val_modulo: int = 1000) -> bool:
    """Determine if a document belongs to the validation split using deterministic hashing.

    Assigns ~0.1% of documents to validation (when val_modulo=1000).
    """
    return stable_hash_int(f"fineweb_split_v1:{doc_id}", modulo=val_modulo) == 0


def tokenize_lm_document(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
) -> List[int]:
    """Tokenize a single LM document without special tokens and append exactly one EOS token."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids.append(tokenizer.eos_token_id)
    return token_ids


class PackedTokenBuffer:
    """Accumulates token IDs and yields fixed contiguous blocks of sequence_length."""

    def __init__(self, sequence_length: int = 1024) -> None:
        self.sequence_length = sequence_length
        self.buffer: List[int] = []
        self.total_tokens_yielded = 0
        self.total_blocks_yielded = 0

    def add_tokens(self, tokens: List[int]) -> List[np.ndarray]:
        """Add tokens to buffer and return any completed fixed blocks."""
        self.buffer.extend(tokens)
        blocks = []
        while len(self.buffer) >= self.sequence_length:
            block = np.array(self.buffer[: self.sequence_length], dtype=np.uint16)
            self.buffer = self.buffer[self.sequence_length :]
            blocks.append(block)
            self.total_tokens_yielded += self.sequence_length
            self.total_blocks_yielded += 1
        return blocks

    def remaining_tokens(self) -> int:
        return len(self.buffer)


def write_token_shard(
    blocks: List[np.ndarray],
    output_path: Path,
) -> Dict[str, Any]:
    """Write a list of [sequence_length] uint16 blocks into a contiguous binary shard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if blocks:
        arr = np.concatenate(blocks, axis=0).astype(np.uint16)
    else:
        arr = np.empty((0,), dtype=np.uint16)

    arr.tofile(str(output_path))
    file_hash = sha256_file(output_path)

    metadata = {
        "file_name": output_path.name,
        "token_count": int(arr.size),
        "block_count": len(blocks),
        "sequence_length": 1024 if not blocks else int(blocks[0].size),
        "dtype": "uint16",
        "sha256": file_hash,
    }
    return metadata


def load_token_shard(shard_path: Path) -> np.ndarray:
    """Load a binary token shard as a 1D uint16 numpy array."""
    return np.fromfile(str(shard_path), dtype=np.uint16)


def process_lm_document_stream(
    doc_iterator: Iterator[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    target_train_tokens: int,
    sequence_length: int = 1024,
    shard_size_blocks: int = 48828,
    train_dir: Optional[Path] = None,
    val_dir: Optional[Path] = None,
    max_docs: Optional[int] = None,
) -> Dict[str, Any]:
    """Process a stream of document dicts into packed token blocks with strict exact-budget enforcement.

    Invariant: total_train_blocks <= target_train_tokens // sequence_length
    Total train tokens will never exceed target_train_tokens.
    """
    target_train_blocks = target_train_tokens // sequence_length

    train_buffer = PackedTokenBuffer(sequence_length=sequence_length)
    val_buffer = PackedTokenBuffer(sequence_length=sequence_length)

    train_shard_blocks: List[np.ndarray] = []
    val_shard_blocks: List[np.ndarray] = []
    all_train_blocks: List[np.ndarray] = []
    all_val_blocks: List[np.ndarray] = []

    train_shard_idx = 0
    val_shard_idx = 0
    train_manifests: List[Dict[str, Any]] = []
    val_manifests: List[Dict[str, Any]] = []

    docs_read = 0
    train_docs = 0
    val_docs = 0
    total_train_blocks = 0
    total_val_blocks = 0

    for item in doc_iterator:
        if max_docs is not None and docs_read >= max_docs:
            break
        docs_read += 1


        raw_text = item.get("text", "")
        clean_text = normalize_lm_text(raw_text)
        if clean_text is None:
            continue

        doc_id = str(item.get("id", ""))
        if not doc_id:
            doc_id = sha256_text(clean_text)

        tokens = tokenize_lm_document(clean_text, tokenizer)

        if is_validation_document(doc_id):
            val_docs += 1
            blocks = val_buffer.add_tokens(tokens)
            for b in blocks:
                val_shard_blocks.append(b)
                all_val_blocks.append(b)
                total_val_blocks += 1
                if val_dir is not None and len(val_shard_blocks) >= shard_size_blocks:
                    shard_file = val_dir / f"tokens-{val_shard_idx:05d}.bin"
                    meta = write_token_shard(val_shard_blocks, shard_file)
                    val_manifests.append(meta)
                    val_shard_idx += 1
                    val_shard_blocks = []
        else:
            blocks = train_buffer.add_tokens(tokens)
            if blocks:
                blocks_needed = target_train_blocks - total_train_blocks
                if blocks_needed <= 0:
                    break
                if len(blocks) > blocks_needed:
                    blocks = blocks[:blocks_needed]

                train_docs += 1
                for b in blocks:
                    train_shard_blocks.append(b)
                    all_train_blocks.append(b)
                    total_train_blocks += 1
                    if train_dir is not None and len(train_shard_blocks) >= shard_size_blocks:
                        shard_file = train_dir / f"tokens-{train_shard_idx:05d}.bin"
                        meta = write_token_shard(train_shard_blocks, shard_file)
                        train_manifests.append(meta)
                        train_shard_idx += 1
                        train_shard_blocks = []

                if total_train_blocks >= target_train_blocks:
                    break
            else:
                train_docs += 1

    # Flush remaining accumulated shard blocks
    if train_dir is not None and train_shard_blocks:
        shard_file = train_dir / f"tokens-{train_shard_idx:05d}.bin"
        meta = write_token_shard(train_shard_blocks, shard_file)
        train_manifests.append(meta)

    if val_dir is not None and val_shard_blocks:
        shard_file = val_dir / f"tokens-{val_shard_idx:05d}.bin"
        meta = write_token_shard(val_shard_blocks, shard_file)
        val_manifests.append(meta)

    total_train_tokens = total_train_blocks * sequence_length
    total_val_tokens = total_val_blocks * sequence_length

    return {
        "docs_read": docs_read,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "total_train_blocks": total_train_blocks,
        "total_val_blocks": total_val_blocks,
        "total_train_tokens": total_train_tokens,
        "total_val_tokens": total_val_tokens,
        "train_blocks": all_train_blocks,
        "val_blocks": all_val_blocks,
        "train_manifests": train_manifests,
        "val_manifests": val_manifests,
    }

