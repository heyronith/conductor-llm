"""Frozen Mistral base tokenizer loader, asset verifier, and formatting helpers."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

from ccpt.data.config import DataConfig
from ccpt.data.hashing import sha256_file


TOKENIZER_ASSET_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
]

EXPECTED_TOKENIZER_HASHES = {
    "tokenizer.json": "11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720",
    "tokenizer_config.json": "ddb008229511e51607002ffe28925001c4a9ca4177dc4de3a655d085cc610b99",
    "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
    "tokenizer.model": "dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055",
}


def load_ccpt_tokenizer(
    config: Optional[DataConfig] = None,
    local_files_only: bool = False,
) -> PreTrainedTokenizerFast:
    """Load and verify the frozen Mistral base tokenizer.

    Guarantees:
    - vocab_size == 32000
    - bos_token_id == 1
    - eos_token_id == 2
    - unk_token_id == 0
    - pad_token_id == 2 (batching convention only; no new token is added)

    Args:
        config: DataConfig containing repo and revision. Defaults to default DataConfig.
        local_files_only: If True, do not attempt network connections.

    Returns:
        Verified PreTrainedTokenizerFast instance.
    """
    if config is None:
        config = DataConfig()

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_repo,
        revision=config.tokenizer_revision,
        use_fast=True,
        local_files_only=local_files_only,
    )

    verify_tokenizer(tokenizer)

    # Assign pad_token_id to eos_token_id as a batching convention only
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Ensure assigning pad token did not modify vocab size
    assert len(tokenizer) == 32000, f"Unexpected vocab size after setting pad_token_id: {len(tokenizer)}"

    return tokenizer


def verify_tokenizer(tokenizer: PreTrainedTokenizerFast) -> None:
    """Assert all tokenizer structural invariants."""
    vocab_len = len(tokenizer)
    assert vocab_len == 32000, f"Expected vocab size 32000, got {vocab_len}"
    assert tokenizer.bos_token_id == 1, f"Expected BOS 1, got {tokenizer.bos_token_id}"
    assert tokenizer.eos_token_id == 2, f"Expected EOS 2, got {tokenizer.eos_token_id}"
    assert tokenizer.unk_token_id == 0, f"Expected UNK 0, got {tokenizer.unk_token_id}"


def get_tokenizer_asset_hashes(config: Optional[DataConfig] = None) -> Dict[str, str]:
    """Download/locate tokenizer assets and compute their SHA256 hashes."""
    if config is None:
        config = DataConfig()

    hashes: Dict[str, str] = {}
    for filename in TOKENIZER_ASSET_FILES:
        try:
            local_path = hf_hub_download(
                repo_id=config.tokenizer_repo,
                filename=filename,
                revision=config.tokenizer_revision,
            )
            hashes[filename] = sha256_file(local_path)
        except Exception:
            pass
    return hashes
