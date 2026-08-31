"""Deterministic materializer and verification for CCPT Strengthening Round FineWeb continuation.

Extends the authoritative 32k continuation (1000 persistence steps) to 128k blocks (4000 persistence steps)
in the logical range [976544, 1104544) while guaranteeing bit-for-bit cryptographic identity on the first 32k blocks.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from transformers import PreTrainedTokenizerFast

from ccpt.data.canonical_materializer import (
    compute_ordered_shards_hash,
    load_canonical_mistral_tokenizer,
    FINEWEB_SOURCE_REPO,
    FINEWEB_SOURCE_CONFIG,
    FINEWEB_SOURCE_REVISION,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
)

CANONICAL_FINEWEB_PREFIX_HASH = "a13410b63d9c1533211784c2a08fa5a918e29cc446448470395aa93919712585"
CANONICAL_FINEWEB_CONT_HASH = "1f6dd66f49a9afa3537244a719af74006308ab81902b0b654142510672022243"
CANONICAL_FINEWEB_VAL_HASH = "4ef33f8f6e1058e1a9e702afe4444593eb07d67ab8a05838e2f81fc6e9eaf870"
from ccpt.data.fineweb import (
    PackedTokenBuffer,
    is_validation_document,
    normalize_lm_text,
    tokenize_lm_document,
    write_token_shard,
)
from ccpt.data.hashing import sha256_file, sha256_json

# Authoritative first-32k continuation shard hashes
AUTHORITATIVE_FIRST_32K_SHARDS = [
    {
        "shard_id": "persistence_continuation_0000",
        "file_name": "persistence_continuation_0000.bin",
        "logical_first_block": 976544,
        "logical_last_block_exclusive": 984736,
        "num_blocks": 8192,
        "num_tokens": 8388608,
        "num_bytes": 16777216,
        "sha256": "91edf8dc1a9047c8a90bc62f98235bc562558bfee6e4604feb4d38acc0454b40",
    },
    {
        "shard_id": "persistence_continuation_0001",
        "file_name": "persistence_continuation_0001.bin",
        "logical_first_block": 984736,
        "logical_last_block_exclusive": 992928,
        "num_blocks": 8192,
        "num_tokens": 8388608,
        "num_bytes": 16777216,
        "sha256": "41fbf91197fef03002dcd908eee735487573fcccf43ae59795106a154344f942",
    },
    {
        "shard_id": "persistence_continuation_0002",
        "file_name": "persistence_continuation_0002.bin",
        "logical_first_block": 992928,
        "logical_last_block_exclusive": 1001120,
        "num_blocks": 8192,
        "num_tokens": 8388608,
        "num_bytes": 16777216,
        "sha256": "15c43d1316c335053eace4d5a454f361ad9dd0f317ffe0ff0167f3b9f8377c47",
    },
    {
        "shard_id": "persistence_continuation_0003",
        "file_name": "persistence_continuation_0003.bin",
        "logical_first_block": 1001120,
        "logical_last_block_exclusive": 1008544,
        "num_blocks": 7424,
        "num_tokens": 7602176,
        "num_bytes": 15204352,
        "sha256": "531a41afa03a2946e427c3ae157bbeca6f12769e0b0f03d482d1043935743c59",
    },
]

TARGET_STRENGTHENING_PERSISTENCE_BLOCKS = 128000
TARGET_STRENGTHENING_PERSISTENCE_TOKENS = 131072000
STRENGTHENING_PERSISTENCE_BLOCK_RANGE = [976544, 1104544]


def materialize_strengthening_fineweb_continuation(
    authoritative_dir: Union[str, Path] = "/data/fineweb_authoritative",
    output_dir: Union[str, Path] = "/data/fineweb_strengthening",
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    sequence_length: int = 1024,
    shard_size_blocks: int = 8192,
    code_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Extends the canonical FineWeb persistence continuation to 128,000 blocks append-only."""
    auth_path = Path(authoritative_dir)
    out_path = Path(output_dir)
    shards_dir = out_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / "manifest.json"

    # Fast-path return if manifest already exists and is completely valid
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            cont_shards = manifest.get("persistence_continuation", {}).get("shards", [])
            if len(cont_shards) == 16 and all(Path(s["path"]).exists() for s in cont_shards):
                # Verify first 4 shards
                for i in range(4):
                    assert cont_shards[i]["sha256"] == AUTHORITATIVE_FIRST_32K_SHARDS[i]["sha256"]
                return {
                    "status": "ALREADY_MATERIALIZED",
                    "manifest": manifest,
                    "manifest_path": str(manifest_path),
                    "first_32k_parity": "BIT_IDENTICAL",
                    "extended_continuation_hash": manifest["persistence_continuation"]["logical_continuation_hash"],
                    "total_continuation_blocks": manifest["persistence_continuation"]["target_blocks"],
                }
        except Exception as e:
            print(f"Existing manifest check failed: {e}. Rematerializing...")

    if tokenizer is None:
        tokenizer = load_canonical_mistral_tokenizer()

    # Step 1: Copy / link first 4 authoritative shards and verify bit-identity
    print("--- Verifying and linking first 4 authoritative persistence shards ---", flush=True)
    persistence_shards: List[Dict[str, Any]] = []
    auth_shards_dir = auth_path / "shards"

    for exp in AUTHORITATIVE_FIRST_32K_SHARDS:
        src_p = auth_shards_dir / exp["file_name"]
        dst_p = shards_dir / exp["file_name"]
        if not src_p.exists():
            raise FileNotFoundError(f"Missing authoritative shard {src_p}")

        # Compute source hash
        actual_hash = sha256_file(src_p)
        if actual_hash != exp["sha256"]:
            raise ValueError(f"Authoritative shard hash mismatch on {src_p}: expected {exp['sha256']}, got {actual_hash}")

        if not dst_p.exists():
            try:
                os.link(src_p, dst_p)
            except Exception:
                shutil.copy2(src_p, dst_p)

        meta = dict(exp)
        meta["path"] = str(dst_p)
        persistence_shards.append(meta)

    # First 32k parity check
    first_32k_hash = compute_ordered_shards_hash(persistence_shards[:4])
    if first_32k_hash != CANONICAL_FINEWEB_CONT_HASH:
        raise RuntimeError(f"First 32k continuation hash mismatch: expected {CANONICAL_FINEWEB_CONT_HASH}, got {first_32k_hash}")
    print(f"First 32k parity PASSED: {first_32k_hash} (BIT_IDENTICAL)", flush=True)

    # Step 2: Download HuggingFace FineWeb parquet shard 1 containing doc 902449
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    print("--- Downloading FineWeb-Edu sample/100BT/000_00001.parquet ---", flush=True)
    parquet_path = hf_hub_download(
        repo_id=FINEWEB_SOURCE_REPO,
        filename="sample/100BT/000_00001.parquet",
        repo_type="dataset",
        revision=FINEWEB_SOURCE_REVISION,
    )

    # Step 3: Stream and pack remaining 96,000 blocks starting at doc 902449 (offset 446)
    print("--- Materializing remaining 96,000 persistence blocks ---", flush=True)
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows

    packer = PackedTokenBuffer(sequence_length=sequence_length)

    current_shard_blocks: List[np.ndarray] = []
    current_shard_idx = 4
    current_shard_start_block = 1008544
    total_extended_blocks_emitted = 0
    target_extended_blocks = 96000  # 128,000 - 32,000

    def _flush_shard(first_blk: int) -> Dict[str, Any]:
        nonlocal current_shard_blocks, current_shard_idx
        num_blks = len(current_shard_blocks)
        shard_id = f"persistence_continuation_{current_shard_idx:04d}"
        shard_file = shards_dir / f"{shard_id}.bin"

        meta = write_token_shard(current_shard_blocks, shard_file)
        meta["shard_id"] = shard_id
        meta["path"] = str(shard_file)
        meta["logical_first_block"] = first_blk
        meta["logical_last_block_exclusive"] = first_blk + num_blks
        meta["num_blocks"] = num_blks
        meta["num_tokens"] = num_blks * sequence_length
        meta["num_bytes"] = num_blks * sequence_length * 2
        current_shard_blocks = []
        current_shard_idx += 1
        return meta

    # Row 176448 is doc 902449
    start_row = 176448
    cum_row = 0
    docs_consumed_extended = 0

    for rg_idx in range(pf.num_row_groups):
        rg_meta = pf.metadata.row_group(rg_idx)
        num_rows = rg_meta.num_rows
        if cum_row + num_rows <= start_row:
            cum_row += num_rows
            continue

        rg = pf.read_row_group(rg_idx, columns=["id", "text"])
        ids = rg["id"].to_pylist()
        texts = rg["text"].to_pylist()

        for local_idx in range(num_rows):
            global_row = cum_row + local_idx
            if global_row < start_row:
                continue

            doc_id = ids[local_idx]
            raw_text = texts[local_idx]
            docs_consumed_extended += 1

            if is_validation_document(doc_id, val_modulo=1000):
                continue

            norm_text = normalize_lm_text(raw_text)
            if norm_text is None:
                continue

            tokens = tokenize_lm_document(norm_text, tokenizer)

            # For doc 902449, slice from offset 446
            if global_row == start_row:
                tokens = tokens[446:]

            blks = packer.add_tokens(tokens)
            for b in blks:
                current_shard_blocks.append(b)
                total_extended_blocks_emitted += 1

                if len(current_shard_blocks) == shard_size_blocks:
                    meta = _flush_shard(first_blk=current_shard_start_block)
                    persistence_shards.append(meta)
                    current_shard_start_block = 1008544 + total_extended_blocks_emitted

                if total_extended_blocks_emitted == target_extended_blocks:
                    break

            if total_extended_blocks_emitted >= target_extended_blocks:
                break

        cum_row += num_rows
        if total_extended_blocks_emitted >= target_extended_blocks:
            break

    # Flush final tail shard (5,888 blocks)
    if current_shard_blocks:
        meta = _flush_shard(first_blk=current_shard_start_block)
        persistence_shards.append(meta)

    if total_extended_blocks_emitted != target_extended_blocks:
        raise RuntimeError(
            f"Expected {target_extended_blocks} extended blocks, emitted {total_extended_blocks_emitted}"
        )

    # Step 4: Build Extended Manifest
    total_blocks = sum(s["num_blocks"] for s in persistence_shards)
    assert total_blocks == TARGET_STRENGTHENING_PERSISTENCE_BLOCKS == 128000
    assert len(persistence_shards) == 16

    extended_continuation_hash = compute_ordered_shards_hash(persistence_shards)

    # Read original prefix shards from authoritative manifest
    auth_manifest_path = auth_path / "manifest.json"
    with open(auth_manifest_path, "r", encoding="utf-8") as f:
        auth_manifest = json.load(f)

    manifest: Dict[str, Any] = {
        "manifest_version": "ccpt-strengthening-task2-v1",
        "created_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha": code_sha or "UNSPECIFIED",
        "source_dataset": {
            "repo": FINEWEB_SOURCE_REPO,
            "config": FINEWEB_SOURCE_CONFIG,
            "revision": FINEWEB_SOURCE_REVISION,
        },
        "tokenizer": {
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "unk_token_id": 0,
        },
        "sequence_length": sequence_length,
        "capability_prefix": {
            "logical_block_range": [0, 976544],
            "target_blocks": 976544,
            "target_tokens": 999981056,
            "logical_prefix_hash": CANONICAL_FINEWEB_PREFIX_HASH,
            "source_manifest_path": str(auth_manifest_path),
        },
        "original_persistence_continuation": {
            "logical_block_range": [976544, 1008544],
            "target_blocks": 32000,
            "target_tokens": 32768000,
            "logical_continuation_hash": CANONICAL_FINEWEB_CONT_HASH,
            "first_32k_parity": "BIT_IDENTICAL",
            "shards": persistence_shards[:4],
        },
        "persistence_continuation": {
            "logical_block_range": STRENGTHENING_PERSISTENCE_BLOCK_RANGE,
            "target_blocks": TARGET_STRENGTHENING_PERSISTENCE_BLOCKS,
            "target_tokens": TARGET_STRENGTHENING_PERSISTENCE_TOKENS,
            "logical_continuation_hash": extended_continuation_hash,
            "first_32k_parity": "BIT_IDENTICAL",
            "shards": persistence_shards,
        },
        "materialization_method": {
            "method": "deterministic_append_only_unbroken_packer",
            "first_32k_source": "authoritative_relink",
            "extended_continuation_start_doc": 902449,
            "doc_902449_offset": 446,
            "extended_docs_consumed": docs_consumed_extended,
            "packer_residual_tokens_at_end": packer.remaining_tokens(),
        },
    }

    manifest["manifest_hash"] = sha256_json(manifest)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Materialization complete! Extended 128k hash: {extended_continuation_hash}", flush=True)

    return {
        "status": "SUCCESS",
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "first_32k_parity": "BIT_IDENTICAL",
        "extended_continuation_hash": extended_continuation_hash,
        "total_continuation_blocks": total_blocks,
    }
