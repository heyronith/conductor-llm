"""Canonical FineWeb Pilot-v2 materialization and manifest generation for Task 7.1.

Uses exclusively canonical Task 4 functions:
- is_validation_document
- normalize_lm_text
- tokenize_lm_document
- PackedTokenBuffer
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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
from ccpt.data.hashing import sha256_file, sha256_json


FINEWEB_SOURCE_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_SOURCE_CONFIG = "sample-100BT"
FINEWEB_SOURCE_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"

TOKENIZER_REPO = "mistralai/Mistral-7B-v0.1"
TOKENIZER_REVISION = "27d67f1b5f57dc0953326b2601d68371d40ea8da"

TARGET_TRAIN_PREFIX_BLOCKS = 976_544        # 999,981,056 tokens (1B prefix)
TARGET_PERSISTENCE_BLOCKS = 32_000          # 32,768,000 tokens (continuation)
TARGET_TOTAL_TRAIN_BLOCKS = 1_008_544       # 1,032,749,056 tokens
TARGET_VAL_BLOCKS = 1_024                   # 1,048,576 tokens


def build_task7_1_data_manifest(
    train_shards: List[Dict[str, Any]],
    val_shards: List[Dict[str, Any]],
    persistence_shards: List[Dict[str, Any]],
    train_prefix_blocks: int = TARGET_TRAIN_PREFIX_BLOCKS,
    persistence_blocks: int = TARGET_PERSISTENCE_BLOCKS,
    val_blocks: int = TARGET_VAL_BLOCKS,
) -> Dict[str, Any]:
    """Constructs the authoritative cryptographic manifest for Task 7.1 FineWeb data."""
    manifest: Dict[str, Any] = {
        "manifest_version": "ccpt-task7.1-v1",
        "source_dataset": {
            "repo": FINEWEB_SOURCE_REPO,
            "config": FINEWEB_SOURCE_CONFIG,
            "revision": FINEWEB_SOURCE_REVISION,
        },
        "tokenizer": {
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
        },
        "sequence_length": 1024,
        "train_prefix": {
            "target_blocks": train_prefix_blocks,
            "target_tokens": train_prefix_blocks * 1024,
            "shards": train_shards,
        },
        "persistence_continuation": {
            "target_blocks": persistence_blocks,
            "target_tokens": persistence_blocks * 1024,
            "shards": persistence_shards,
        },
        "validation": {
            "target_blocks": val_blocks,
            "target_tokens": val_blocks * 1024,
            "shards": val_shards,
        },
    }

    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest
