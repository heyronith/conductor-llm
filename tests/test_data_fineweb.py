"""Tests for FineWeb-Edu text normalization, token stream packing, and shard storage."""

import tempfile
from pathlib import Path

import numpy as np

from ccpt.data.config import DataConfig
from ccpt.data.fineweb import (
    PackedTokenBuffer,
    is_validation_document,
    load_token_shard,
    normalize_lm_text,
    tokenize_lm_document,
    write_token_shard,
)
from ccpt.data.tokenizer import load_ccpt_tokenizer


def test_fineweb_text_normalization():
    """Verify CRLF conversion and empty text rejection."""
    assert normalize_lm_text("Hello\r\nWorld\r!") == "Hello\nWorld\n!"
    assert normalize_lm_text("   \r\n \n \t  ") is None
    assert normalize_lm_text("") is None
    assert normalize_lm_text("Valid text without alterations.") == "Valid text without alterations."


def test_fineweb_tokenization_and_eos_appending():
    """Verify document tokenization appends single EOS without BOS."""
    config = DataConfig()
    tokenizer = load_ccpt_tokenizer(config)

    text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenize_lm_document(text, tokenizer)

    assert tokens[-1] == 2, "Last token must be EOS (2)"
    assert tokens[0] != 1 or text.startswith("<s>"), "Should not auto-inject BOS"


def test_packed_token_buffer_fixed_blocks():
    """Verify that buffer strictly packs tokens into 1024-token contiguous blocks without padding."""
    buffer = PackedTokenBuffer(sequence_length=1024)

    doc1 = list(range(600))
    doc2 = list(range(700))
    doc3 = list(range(2000))

    blocks1 = buffer.add_tokens(doc1)
    assert len(blocks1) == 0
    assert buffer.remaining_tokens() == 600

    blocks2 = buffer.add_tokens(doc2)
    assert len(blocks2) == 1
    assert blocks2[0].shape == (1024,)
    assert blocks2[0].dtype == np.uint16
    assert buffer.remaining_tokens() == (600 + 700 - 1024)

    blocks3 = buffer.add_tokens(doc3)
    assert len(blocks3) == 2
    assert buffer.total_blocks_yielded == 3
    assert buffer.total_tokens_yielded == 3 * 1024


def test_fineweb_validation_split_disjoint():
    """Verify that document split is deterministic and partition is disjoint."""
    doc_ids = [f"doc_{i:06d}" for i in range(5000)]

    val_docs = [did for did in doc_ids if is_validation_document(did)]
    train_docs = [did for did in doc_ids if not is_validation_document(did)]

    assert len(set(val_docs).intersection(set(train_docs))) == 0
    # Expected ~0.1% val docs (around 5 out of 5000)
    assert 0 < len(val_docs) < 50


def test_token_shard_binary_roundtrip():
    """Verify binary shard serialization and memory mapping round-trip."""
    blocks = [np.arange(1024, dtype=np.uint16) + (i * 10) for i in range(5)]

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = Path(tmpdir) / "test_shard.bin"
        meta = write_token_shard(blocks, shard_path)

        assert meta["token_count"] == 5 * 1024
        assert meta["block_count"] == 5
        assert meta["dtype"] == "uint16"

        loaded = load_token_shard(shard_path)
        assert loaded.shape == (5 * 1024,)
        assert loaded.dtype == np.uint16
        for i in range(5):
            np.testing.assert_array_equal(loaded[i * 1024 : (i + 1) * 1024], blocks[i])


def test_exact_budget_limiter_cannot_overshoot():
    """Verify that process_lm_document_stream strictly caps output at target_tokens and never overshoots."""
    from ccpt.data.fineweb import process_lm_document_stream

    config = DataConfig()
    tokenizer = load_ccpt_tokenizer(config)

    # Synthetic document generator with plenty of text to overshoot if unchecked
    raw_docs = [
        {"id": f"train_doc_{i}", "text": "This is a sentence that produces several tokens per document. " * 5}
        for i in range(50)
    ]

    # Target: 5 blocks of 4 tokens = 20 tokens
    seq_len = 4
    target_tokens = 20
    target_blocks = 5

    res = process_lm_document_stream(
        doc_iterator=iter(raw_docs),
        tokenizer=tokenizer,
        target_train_tokens=target_tokens,
        sequence_length=seq_len,
    )

    assert res["total_train_blocks"] == target_blocks, f"Expected exactly {target_blocks} blocks, got {res['total_train_blocks']}"
    assert res["total_train_tokens"] == target_tokens, f"Expected exactly {target_tokens} tokens, got {res['total_train_tokens']}"
    assert len(res["train_blocks"]) == target_blocks
    for b in res["train_blocks"]:
        assert len(b) == seq_len


def test_production_scale_exact_budget_calculation():
    """Verify production target invariants for 10B tokens at 1024 sequence length."""
    production_target_tokens = 10_000_000_000
    seq_len = 1024
    expected_blocks = 9_765_625

    assert production_target_tokens // seq_len == expected_blocks
    assert expected_blocks * seq_len == production_target_tokens

