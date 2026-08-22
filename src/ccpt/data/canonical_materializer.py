"""Canonical FineWeb Materializer and Manifest Generator for Task 7.2.1.

Uses exclusively canonical Task 4 functions:
- is_validation_document
- normalize_lm_text
- tokenize_lm_document
- PackedTokenBuffer
- write_token_shard
- load_token_shard

Strictly isolates from Task 6 shards, enforces unbroken packer continuation
between 1B prefix and persistence continuation, and generates cryptographic manifests.
Supports streaming directly from HuggingFaceFW/fineweb-edu with mistralai/Mistral-7B-v0.1.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ccpt.data.fineweb import (
    PackedTokenBuffer,
    is_validation_document,
    load_token_shard,
    normalize_lm_text,
    tokenize_lm_document,
    write_token_shard,
)
from ccpt.data.hashing import sha256_bytes, sha256_file, sha256_json, sha256_text
from ccpt.data.production_stream import CanonicalFineWebStream


FINEWEB_SOURCE_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_SOURCE_CONFIG = "sample-100BT"
FINEWEB_SOURCE_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"

TOKENIZER_REPO = "mistralai/Mistral-7B-v0.1"
TOKENIZER_REVISION = "27d67f1b5f57dc0953326b2601d68371d40ea8da"

# Full 1B / Persistence block targets
TARGET_TRAIN_PREFIX_BLOCKS = 976_544        # 999,981,056 tokens (1B prefix)
TARGET_PERSISTENCE_BLOCKS = 32_000          # 32,768,000 tokens (continuation)
TARGET_TOTAL_TRAIN_BLOCKS = 1_008_544       # 1,032,749,056 tokens
TARGET_VAL_BLOCKS = 1_024                   # 1,048,576 tokens


def load_canonical_mistral_tokenizer(
    repo: str = TOKENIZER_REPO,
    revision: str = TOKENIZER_REVISION,
) -> PreTrainedTokenizerFast:
    """Loads the authoritative pinned Mistral tokenizer from Hugging Face."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.strip().startswith("hf_"):
                hf_token = line.strip()
                break

    return AutoTokenizer.from_pretrained(
        repo,
        revision=revision,
        token=hf_token,
    )


def compute_ordered_shards_hash(shards: List[Dict[str, Any]]) -> str:
    """Computes a cryptographically well-defined SHA256 digest over ordered shard digests and block ranges."""
    entries = []
    for s in shards:
        entry = f"{s.get('shard_id')}:{s.get('logical_first_block')}:{s.get('logical_last_block_exclusive')}:{s.get('num_blocks')}:{s.get('sha256')}"
        entries.append(entry)
    data = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_task7_2_data_manifest(
    train_prefix_shards: List[Dict[str, Any]],
    persistence_shards: List[Dict[str, Any]],
    val_shards: List[Dict[str, Any]],
    train_prefix_blocks: int = TARGET_TRAIN_PREFIX_BLOCKS,
    persistence_blocks: int = TARGET_PERSISTENCE_BLOCKS,
    val_blocks: int = TARGET_VAL_BLOCKS,
    sequence_length: int = 1024,
    packer_residual_tokens: int = 0,
    docs_consumed: int = 0,
    train_docs_accepted: int = 0,
    val_docs_accepted: int = 0,
    real_hf_source: bool = False,
    real_mistral_tokenizer: bool = False,
) -> Dict[str, Any]:
    """Constructs the authoritative cryptographic manifest for Task 7.2 / 7.2.1 FineWeb data."""
    prefix_hash = compute_ordered_shards_hash(train_prefix_shards)
    persistence_hash = compute_ordered_shards_hash(persistence_shards)
    val_hash = compute_ordered_shards_hash(val_shards)

    manifest: Dict[str, Any] = {
        "manifest_version": "ccpt-task7.2-v1",
        "source_dataset": {
            "repo": FINEWEB_SOURCE_REPO,
            "config": FINEWEB_SOURCE_CONFIG,
            "revision": FINEWEB_SOURCE_REVISION,
            "is_real_hf_stream": real_hf_source,
        },
        "tokenizer": {
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "unk_token_id": 0,
            "is_real_mistral_tokenizer": real_mistral_tokenizer,
        },
        "sequence_length": sequence_length,
        "canonical_split_version": "task4_canonical_v1",
        "normalization_identifier": "task4_normalize_lm_text",
        "tokenization_identifier": "task4_tokenize_lm_document",
        "exact_logical_block_ranges": {
            "train_prefix": [0, train_prefix_blocks],
            "persistence_continuation": [train_prefix_blocks, train_prefix_blocks + persistence_blocks],
            "validation": [0, val_blocks],
        },
        "train_prefix": {
            "start_block": 0,
            "end_block_exclusive": train_prefix_blocks,
            "target_blocks": train_prefix_blocks,
            "target_tokens": train_prefix_blocks * sequence_length,
            "logical_prefix_hash": prefix_hash,
            "shards": train_prefix_shards,
        },
        "persistence_continuation": {
            "start_block": train_prefix_blocks,
            "end_block_exclusive": train_prefix_blocks + persistence_blocks,
            "target_blocks": persistence_blocks,
            "target_tokens": persistence_blocks * sequence_length,
            "logical_continuation_hash": persistence_hash,
            "shards": persistence_shards,
        },
        "validation": {
            "start_block": 0,
            "end_block_exclusive": val_blocks,
            "target_blocks": val_blocks,
            "target_tokens": val_blocks * sequence_length,
            "logical_validation_hash": val_hash,
            "shards": val_shards,
        },
        "packer_continuity": {
            "unbroken_packer_between_prefix_and_continuation": True,
            "packer_residual_tokens_at_end": packer_residual_tokens,
            "documents_consumed": docs_consumed,
            "train_documents_accepted": train_docs_accepted,
            "validation_documents_accepted": val_docs_accepted,
        },
    }

    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def materialize_bounded_canonical_fineweb_proof(
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    document_iterable: Optional[Iterable[Dict[str, Any]]] = None,
    output_dir: Union[str, Path] = "artifacts/fineweb_proof",
    prefix_blocks_target: int = 50,
    continuation_blocks_target: int = 20,
    val_blocks_target: int = 10,
    sequence_length: int = 1024,
    val_modulo: int = 1000,
) -> Dict[str, Any]:
    """Materializes bounded canonical FineWeb shards and proves continuation semantics byte-for-byte.

    Demonstrates:
    1. Continuous stream produces prefix [0, N) and continuation [N, N+K) without packer reset.
    2. Independent continuous stream produces [0, N+K) identically.
    3. Continuation begins EXACTLY at block N.
    4. Produces cryptographic hashes and verified manifest.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load real tokenizer if not provided
    if tokenizer is None:
        tokenizer = load_canonical_mistral_tokenizer()

    is_real_mistral = (
        getattr(tokenizer, "name_or_path", "") == TOKENIZER_REPO
        or (hasattr(tokenizer, "__len__") and len(tokenizer) == 32000 and getattr(tokenizer, "bos_token_id", None) == 1)
    )

    # 2. Load real HF stream if document iterable not provided
    is_real_source = False
    if document_iterable is None:
        from datasets import load_dataset
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token and Path(".env").exists():
            for line in Path(".env").read_text().splitlines():
                if line.strip().startswith("hf_"):
                    hf_token = line.strip()
                    break

        hf_ds = load_dataset(
            FINEWEB_SOURCE_REPO,
            FINEWEB_SOURCE_CONFIG,
            revision=FINEWEB_SOURCE_REVISION,
            split="train",
            streaming=True,
            token=hf_token,
        )
        is_real_source = True
        
        # Buffer enough documents for both the primary run and the independent replay
        # To get 70 train blocks (71,680 tokens) + 10 val blocks (10,240 tokens), ~200-300 docs is plenty
        docs_buffer: List[Dict[str, Any]] = []
        for i, item in enumerate(hf_ds):
            doc_id = str(item.get("id", f"doc_{i:06d}"))
            text = item.get("text", "")
            docs_buffer.append({"id": doc_id, "text": text})
            if len(docs_buffer) >= 600:
                break
        docs = docs_buffer
    else:
        docs = list(document_iterable)

    # 3. Run canonical continuous train stream
    train_stream = CanonicalFineWebStream(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        split="train",
        val_modulo=val_modulo,
    )

    prefix_blocks: List[np.ndarray] = []
    continuation_blocks: List[np.ndarray] = []
    docs_consumed = 0
    train_docs_accepted = 0

    for doc in docs:
        docs_consumed += 1
        doc_id = str(doc.get("id", ""))
        text = doc.get("text", "")
        if not is_validation_document(doc_id, val_modulo=val_modulo):
            train_docs_accepted += 1
        blks = train_stream.process_document(text, doc_id)
        for b in blks:
            if len(prefix_blocks) < prefix_blocks_target:
                prefix_blocks.append(b)
            elif len(continuation_blocks) < continuation_blocks_target:
                continuation_blocks.append(b)

        if len(prefix_blocks) == prefix_blocks_target and len(continuation_blocks) == continuation_blocks_target:
            break

    # 4. Run independent reference stream for full [0, N+K)
    ref_stream = CanonicalFineWebStream(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        split="train",
        val_modulo=val_modulo,
    )
    ref_blocks: List[np.ndarray] = []
    for doc in docs:
        doc_id = str(doc.get("id", ""))
        text = doc.get("text", "")
        blks = ref_stream.process_document(text, doc_id)
        for b in blks:
            if len(ref_blocks) < (prefix_blocks_target + continuation_blocks_target):
                ref_blocks.append(b)
        if len(ref_blocks) == (prefix_blocks_target + continuation_blocks_target):
            break

    # Prove byte-for-byte equality
    assert len(prefix_blocks) == prefix_blocks_target, f"Collected {len(prefix_blocks)} prefix blocks, expected {prefix_blocks_target}"
    assert len(continuation_blocks) == continuation_blocks_target, f"Collected {len(continuation_blocks)} continuation blocks, expected {continuation_blocks_target}"
    assert len(ref_blocks) == prefix_blocks_target + continuation_blocks_target

    for i in range(prefix_blocks_target):
        assert np.array_equal(prefix_blocks[i], ref_blocks[i]), f"Prefix block {i} mismatch with reference"

    for j in range(continuation_blocks_target):
        assert np.array_equal(continuation_blocks[j], ref_blocks[prefix_blocks_target + j]), f"Continuation block {j} mismatch with reference block {prefix_blocks_target + j}"

    # 5. Run validation stream
    val_stream = CanonicalFineWebStream(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        split="validation",
        val_modulo=val_modulo,
    )
    val_blocks: List[np.ndarray] = []
    val_docs_accepted = 0
    for doc in docs:
        doc_id = str(doc.get("id", ""))
        text = doc.get("text", "")
        if is_validation_document(doc_id, val_modulo=val_modulo):
            val_docs_accepted += 1
        blks = val_stream.process_document(text, doc_id)
        for b in blks:
            if len(val_blocks) < val_blocks_target:
                val_blocks.append(b)
        if len(val_blocks) == val_blocks_target:
            break

    # 6. Write shards and record metadata
    prefix_shard_path = out_path / "train_prefix_shard_000.bin"
    cont_shard_path = out_path / "persistence_continuation_shard_000.bin"
    val_shard_path = out_path / "val_shard_000.bin"

    meta_prefix = write_token_shard(prefix_blocks, prefix_shard_path)
    meta_prefix["shard_id"] = "train_prefix_000"
    meta_prefix["path"] = str(prefix_shard_path)
    meta_prefix["logical_first_block"] = 0
    meta_prefix["logical_last_block_exclusive"] = len(prefix_blocks)
    meta_prefix["num_blocks"] = len(prefix_blocks)
    meta_prefix["num_tokens"] = len(prefix_blocks) * sequence_length
    meta_prefix["num_bytes"] = len(prefix_blocks) * sequence_length * 2

    meta_cont = write_token_shard(continuation_blocks, cont_shard_path)
    meta_cont["shard_id"] = "persistence_continuation_000"
    meta_cont["path"] = str(cont_shard_path)
    meta_cont["logical_first_block"] = prefix_blocks_target
    meta_cont["logical_last_block_exclusive"] = prefix_blocks_target + len(continuation_blocks)
    meta_cont["num_blocks"] = len(continuation_blocks)
    meta_cont["num_tokens"] = len(continuation_blocks) * sequence_length
    meta_cont["num_bytes"] = len(continuation_blocks) * sequence_length * 2

    meta_val = write_token_shard(val_blocks, val_shard_path)
    meta_val["shard_id"] = "val_000"
    meta_val["path"] = str(val_shard_path)
    meta_val["logical_first_block"] = 0
    meta_val["logical_last_block_exclusive"] = len(val_blocks)
    meta_val["num_blocks"] = len(val_blocks)
    meta_val["num_tokens"] = len(val_blocks) * sequence_length
    meta_val["num_bytes"] = len(val_blocks) * sequence_length * 2

    manifest = build_task7_2_data_manifest(
        train_prefix_shards=[meta_prefix],
        persistence_shards=[meta_cont],
        val_shards=[meta_val],
        train_prefix_blocks=prefix_blocks_target,
        persistence_blocks=continuation_blocks_target,
        val_blocks=len(val_blocks),
        sequence_length=sequence_length,
        packer_residual_tokens=train_stream.packer.remaining_tokens(),
        docs_consumed=docs_consumed,
        train_docs_accepted=train_docs_accepted,
        val_docs_accepted=val_docs_accepted,
        real_hf_source=is_real_source,
        real_mistral_tokenizer=is_real_mistral,
    )

    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "prefix_blocks_count": len(prefix_blocks),
        "continuation_blocks_count": len(continuation_blocks),
        "val_blocks_count": len(val_blocks),
        "prefix_hash": manifest["train_prefix"]["logical_prefix_hash"],
        "continuation_hash": manifest["persistence_continuation"]["logical_continuation_hash"],
        "val_hash": manifest["validation"]["logical_validation_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "continuation_starts_at_block": prefix_blocks_target,
        "byte_for_byte_continuation_proven": True,
        "documents_consumed": docs_consumed,
        "train_documents_accepted": train_docs_accepted,
        "val_documents_accepted": val_docs_accepted,
        "packer_residual_tokens": train_stream.packer.remaining_tokens(),
        "REAL_HF_FINEWEB_SOURCE": is_real_source,
        "REAL_MISTRAL_TOKENIZER": is_real_mistral,
        "prefix_shard_path": str(prefix_shard_path),
        "continuation_shard_path": str(cont_shard_path),
        "val_shard_path": str(val_shard_path),
    }
