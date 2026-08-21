"""CCPT data processing, tokenization, and collation package."""

from ccpt.data.collators import (
    DataCollatorForRiskTraining,
    DataCollatorForSafeGenerationTraining,
)
from ccpt.data.config import (
    DATA_ORDER_SEED,
    DEFAULT_DATA_ROOT,
    FORMAT_VERSION,
    DataConfig,
)
from ccpt.data.fineweb import (
    PackedTokenBuffer,
    is_validation_document,
    load_token_shard,
    normalize_lm_text,
    process_lm_document_stream,
    tokenize_lm_document,
    write_token_shard,
)
from ccpt.data.hashing import (
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_records,
    sha256_text,
    stable_hash_int,
)
from ccpt.data.manifests import (
    build_task4_manifest,
    compute_records_logical_hash,
    record_to_dict,
)
from ccpt.data.ordering import (
    get_epoch_example_order,
    order_records_for_epoch,
)
from ccpt.data.tokenizer import (
    EXPECTED_TOKENIZER_HASHES,
    TOKENIZER_ASSET_FILES,
    get_tokenizer_asset_hashes,
    load_ccpt_tokenizer,
    verify_tokenizer,
)
from ccpt.data.wildguard import (
    RISK_ARROW_SCHEMA,
    SAFE_GEN_ARROW_SCHEMA,
    RiskRecord,
    SafeGenerationRecord,
    canonicalize_prompt,
    compute_length_percentiles,
    format_safety_prefix,
    format_safety_response,
    is_eligible_safe_generation_row,
    is_validation_prompt_group,
    load_wildguard_records,
    load_wildguard_records_arrow,
    process_wildguard_raw_dataset,
    save_wildguard_records,
    save_wildguard_records_arrow,
    tokenize_risk_example,
    tokenize_safe_generation_example,
)

__all__ = [
    "DataConfig",
    "DEFAULT_DATA_ROOT",
    "DATA_ORDER_SEED",
    "FORMAT_VERSION",
    "load_ccpt_tokenizer",
    "verify_tokenizer",
    "get_tokenizer_asset_hashes",
    "TOKENIZER_ASSET_FILES",
    "EXPECTED_TOKENIZER_HASHES",
    "sha256_text",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_records",
    "stable_hash_int",
    "normalize_lm_text",
    "is_validation_document",
    "tokenize_lm_document",
    "PackedTokenBuffer",
    "write_token_shard",
    "load_token_shard",
    "process_lm_document_stream",
    "canonicalize_prompt",
    "is_validation_prompt_group",
    "format_safety_prefix",
    "format_safety_response",
    "tokenize_risk_example",
    "tokenize_safe_generation_example",
    "is_eligible_safe_generation_row",
    "process_wildguard_raw_dataset",
    "compute_length_percentiles",
    "save_wildguard_records",
    "load_wildguard_records",
    "save_wildguard_records_arrow",
    "load_wildguard_records_arrow",
    "RISK_ARROW_SCHEMA",
    "SAFE_GEN_ARROW_SCHEMA",
    "RiskRecord",
    "SafeGenerationRecord",
    "DataCollatorForRiskTraining",
    "DataCollatorForSafeGenerationTraining",
    "get_epoch_example_order",
    "order_records_for_epoch",
    "compute_records_logical_hash",
    "record_to_dict",
    "build_task4_manifest",
]

