"""Deterministic canonical SHA256 hashing utilities for datasets and metadata."""

import hashlib
import json
from pathlib import Path
from typing import Any, List, Union


def sha256_text(text: str) -> str:
    """Compute SHA256 hexadecimal digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: Union[bytes, bytearray, memoryview]) -> str:
    """Compute SHA256 hexadecimal digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(file_path: Union[str, Path]) -> str:
    """Compute SHA256 hexadecimal digest of a file content in 64KB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Compute deterministic SHA256 digest of a JSON-serializable object with sorted keys."""
    canonical_str = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(canonical_str)


def sha256_records(records: List[Any]) -> str:
    """Compute deterministic SHA256 digest over an ordered list of records."""
    return sha256_json(records)


def stable_hash_int(key: str, modulo: int = 10000) -> int:
    """Map a string key deterministically to [0, modulo) using SHA256 integer prefix."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Use first 16 hex digits (64-bit uint)
    val = int(digest[:16], 16)
    return val % modulo
