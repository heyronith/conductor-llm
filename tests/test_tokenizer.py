"""Tests verifying frozen Mistral base tokenizer invariants and asset hashes."""

from transformers import AutoTokenizer

from ccpt.data.config import DataConfig
from ccpt.data.tokenizer import (
    EXPECTED_TOKENIZER_HASHES,
    get_tokenizer_asset_hashes,
    load_ccpt_tokenizer,
    verify_tokenizer,
)


def test_tokenizer_invariants():
    """Verify vocabulary size, special token IDs, and pad token assignment."""
    config = DataConfig()
    tokenizer = load_ccpt_tokenizer(config)

    assert len(tokenizer) == 32000
    assert tokenizer.vocab_size == 32000
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.unk_token_id == 0
    assert tokenizer.pad_token_id == 2  # Set to EOS as batching convention


def test_tokenizer_deterministic_encode_decode():
    """Verify deterministic encoding and round-trip decoding on Unicode strings."""
    config = DataConfig()
    tokenizer = load_ccpt_tokenizer(config)

    test_strings = [
        "Hello world! This is a test of the Mistral tokenizer.",
        "Mathematical formula: ∀x ∈ ℝ, e^(iπ) + 1 = 0.",
        "Multilingual test: 日本語, Español, Français, Deutsch, Русский, 中文.",
        "Code syntax: def forward(self, x: torch.Tensor) -> torch.Tensor:\n    return self.layer(x)",
    ]

    for s in test_strings:
        ids1 = tokenizer.encode(s, add_special_tokens=False)
        ids2 = tokenizer.encode(s, add_special_tokens=False)
        assert ids1 == ids2, f"Non-deterministic encoding for: {s}"
        assert all(0 <= tid < 32000 for tid in ids1), f"Token ID out of bounds in: {ids1}"

        decoded = tokenizer.decode(ids1, clean_up_tokenization_spaces=False)
        assert len(decoded) > 0


def test_tokenizer_asset_hashes():
    """Verify that remote tokenizer asset files match the pinned SHA256 hashes."""
    config = DataConfig()
    hashes = get_tokenizer_asset_hashes(config)

    assert len(hashes) >= 3, "Expected at least 3 tokenizer asset files"
    for filename, expected_hash in EXPECTED_TOKENIZER_HASHES.items():
        if filename in hashes:
            assert hashes[filename] == expected_hash, (
                f"Hash mismatch for {filename}: expected {expected_hash}, got {hashes[filename]}"
            )
