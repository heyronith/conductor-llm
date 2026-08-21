"""Preparation script for FineWeb-Edu tokenization and shard creation."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from ccpt.data.config import DataConfig
from ccpt.data.fineweb import (
    PackedTokenBuffer,
    is_validation_document,
    normalize_lm_text,
    tokenize_lm_document,
    write_token_shard,
)
from ccpt.data.hashing import sha256_text
from ccpt.data.tokenizer import load_ccpt_tokenizer


def prepare_fineweb(
    data_root: str,
    token_budget: int = 10_000_000_000,
    max_doc_limit: Optional[int] = None,
    shard_size_blocks: int = 48828,  # ~50M tokens per shard (48828 * 1024)
) -> dict:
    """Stream and tokenize FineWeb-Edu into uint16 binary shards up to the exact token budget."""
    config = DataConfig(data_root=data_root, lm_target_tokens=token_budget)
    tokenizer = load_ccpt_tokenizer(config)

    output_dir = config.get_data_path("fineweb", config.fineweb_revision)
    train_dir = output_dir / "train"
    val_dir = output_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    target_train_blocks = token_budget // config.max_seq_len

    print(f"Streaming from {config.fineweb_repo}@{config.fineweb_revision} ({config.fineweb_config})...")
    print(f"Target train token budget: {token_budget} tokens ({target_train_blocks} contiguous {config.max_seq_len}-token blocks).")

    dataset_stream = load_dataset(
        config.fineweb_repo,
        name=config.fineweb_config,
        revision=config.fineweb_revision,
        split="train",
        streaming=True,
    )

    train_buffer = PackedTokenBuffer(sequence_length=config.max_seq_len)
    val_buffer = PackedTokenBuffer(sequence_length=config.max_seq_len)

    train_shard_blocks = []
    val_shard_blocks = []
    train_shard_idx = 0
    val_shard_idx = 0

    train_manifests = []
    val_manifests = []

    doc_count = 0
    train_doc_count = 0
    val_doc_count = 0
    total_train_blocks = 0
    total_val_blocks = 0

    for item in dataset_stream:
        doc_count += 1
        if max_doc_limit is not None and doc_count > max_doc_limit:
            break

        raw_text = item.get("text", "")
        clean_text = normalize_lm_text(raw_text)
        if clean_text is None:
            continue

        doc_id = str(item.get("id", ""))
        if not doc_id:
            doc_id = sha256_text(clean_text)

        tokens = tokenize_lm_document(clean_text, tokenizer)

        if is_validation_document(doc_id):
            val_doc_count += 1
            blocks = val_buffer.add_tokens(tokens)
            for b in blocks:
                val_shard_blocks.append(b)
                total_val_blocks += 1
                if len(val_shard_blocks) >= shard_size_blocks:
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

                train_doc_count += 1
                for b in blocks:
                    train_shard_blocks.append(b)
                    total_train_blocks += 1
                    if len(train_shard_blocks) >= shard_size_blocks:
                        shard_file = train_dir / f"tokens-{train_shard_idx:05d}.bin"
                        meta = write_token_shard(train_shard_blocks, shard_file)
                        train_manifests.append(meta)
                        train_shard_idx += 1
                        train_shard_blocks = []

                if total_train_blocks >= target_train_blocks:
                    print(f"Reached exact train token budget: {total_train_blocks * config.max_seq_len} tokens.")
                    break
            else:
                train_doc_count += 1

    # Flush remaining accumulated shard blocks
    if train_shard_blocks:
        shard_file = train_dir / f"tokens-{train_shard_idx:05d}.bin"
        meta = write_token_shard(train_shard_blocks, shard_file)
        train_manifests.append(meta)

    if val_shard_blocks:
        shard_file = val_dir / f"tokens-{val_shard_idx:05d}.bin"
        meta = write_token_shard(val_shard_blocks, shard_file)
        val_manifests.append(meta)

    total_train_tokens = total_train_blocks * config.max_seq_len
    total_val_tokens = total_val_blocks * config.max_seq_len

    manifest_data = {
        "dataset_repo": config.fineweb_repo,
        "dataset_revision": config.fineweb_revision,
        "config": config.fineweb_config,
        "tokenizer_repo": config.tokenizer_repo,
        "tokenizer_revision": config.tokenizer_revision,
        "sequence_length": config.max_seq_len,
        "docs_processed": doc_count,
        "train_docs": train_doc_count,
        "val_docs": val_doc_count,
        "train_tokens": total_train_tokens,
        "val_tokens": total_val_tokens,
        "train_blocks": total_train_blocks,
        "val_blocks": total_val_blocks,
        "train_shards": train_manifests,
        "val_shards": val_manifests,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"FineWeb preparation finished: {train_doc_count} train docs ({total_train_tokens} tokens), {val_doc_count} val docs.")
    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu tokenized shards.")
    parser.add_argument("--data-root", type=str, default="data/processed")
    parser.add_argument("--token-budget", type=int, default=10_000_000_000)
    parser.add_argument("--max-doc-limit", type=int, default=None)
    args = parser.parse_args()

    prepare_fineweb(
        data_root=args.data_root,
        token_budget=args.token_budget,
        max_doc_limit=args.max_doc_limit,
    )
