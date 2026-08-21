"""Manifest generation and logical dataset hashing utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ccpt.data.config import DataConfig
from ccpt.data.hashing import sha256_json, sha256_text
from ccpt.data.tokenizer import EXPECTED_TOKENIZER_HASHES, get_tokenizer_asset_hashes


def record_to_dict(rec: Any) -> Dict[str, Any]:
    """Convert a record object or dictionary into a normalized dictionary for canonical hashing."""
    if hasattr(rec, "__dict__"):
        d = dict(rec.__dict__)
    elif isinstance(rec, dict):
        d = dict(rec)
    else:
        raise ValueError(f"Unsupported record type: {type(rec)}")

    if "input_ids" in d:
        if hasattr(d["input_ids"], "tolist"):
            d["input_ids"] = d["input_ids"].tolist()
        else:
            d["input_ids"] = [int(x) for x in d["input_ids"]]
    return d


def compute_records_logical_hash(records: List[Any]) -> str:
    """Compute deterministic full-content logical hash across entire tokenized records."""
    normalized_records = [record_to_dict(r) for r in records]
    return sha256_json(normalized_records)


def build_task4_manifest(
    config: DataConfig,
    fineweb_stats: Dict[str, Any],
    wildguard_stats: Dict[str, Any],
    execution_env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct complete machine-readable Task 4 data manifest."""
    tok_hashes = get_tokenizer_asset_hashes(config)
    if not tok_hashes:
        tok_hashes = EXPECTED_TOKENIZER_HASHES

    is_live = wildguard_stats.get("wildguard_live_validated", False)
    stats_source = wildguard_stats.get("statistics_source", "pinned_live_dataset" if is_live else "synthetic_fixture")

    env_meta = execution_env or {}

    manifest = {
        "format_version": config.format_version,
        "execution_environment": env_meta.get("execution_environment", "local" if not is_live else "modal"),
        "modal_volume": env_meta.get("modal_volume", "ccpt-data"),
        "modal_volume_root": env_meta.get("modal_volume_root", "/data/ccpt"),
        "wildguard_live_validated": is_live,
        "statistics_source": stats_source,
        "tokenizer": {
            "repo": config.tokenizer_repo,
            "revision": config.tokenizer_revision,
            "vocab_size": 32000,
            "special_tokens": {
                "unk_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 2,
            },
            "file_hashes": tok_hashes,
        },
        "fineweb": {
            "repo": config.fineweb_repo,
            "revision": config.fineweb_revision,
            "config": config.fineweb_config,
            "split_algorithm": "SHA256(doc_id) % 1000 == 0 (0.1% validation, 99.9% train)",
            "sequence_length": config.max_seq_len,
            "eos_handling": "append exactly 1 EOS after every document; no BOS; count EOS in token budget",
            "token_dtype": "uint16",
            "smoke_document_count": fineweb_stats.get("smoke_document_count", 0),
            "smoke_token_count": fineweb_stats.get("smoke_token_count", 0),
            "smoke_block_count": fineweb_stats.get("smoke_block_count", 0),
            "smoke_hash": fineweb_stats.get("smoke_hash", ""),
            "production_target_token_budget": config.lm_target_tokens,
            "production_target_blocks": config.lm_target_tokens // config.max_seq_len,
        },
        "wildguard": {
            "repo": config.wildguard_repo,
            "revision": config.wildguard_revision,
            "train_config": config.wildguard_train_config,
            "eval_config": config.wildguard_test_config,
            "raw_train_row_count": wildguard_stats.get("total_raw_rows", 0),
            "usable_risk_train_count": wildguard_stats.get("risk_train_count", 0),
            "usable_risk_val_count": wildguard_stats.get("risk_val_count", 0),
            "eligible_safe_gen_train_count": wildguard_stats.get("gen_train_count", 0),
            "eligible_safe_gen_val_count": wildguard_stats.get("gen_val_count", 0),
            "raw_eval_row_count": wildguard_stats.get("raw_eval_rows_count", 0),
            "usable_eval_record_count": wildguard_stats.get("eval_records_count", 0),
            "harmful_refusal_count": wildguard_stats.get("eligible_harmful_refusal", 0),
            "benign_compliance_count": wildguard_stats.get("eligible_benign_compliance", 0),
            "conflicting_prompt_groups_excluded": wildguard_stats.get("conflicting_groups", 0),
            "risk_train_logical_hash": wildguard_stats.get("risk_train_hash", ""),
            "risk_val_logical_hash": wildguard_stats.get("risk_val_hash", ""),
            "gen_train_logical_hash": wildguard_stats.get("gen_train_hash", ""),
            "gen_val_logical_hash": wildguard_stats.get("gen_val_hash", ""),
            "eval_logical_hash": wildguard_stats.get("eval_hash", ""),
            "length_and_truncation_stats": wildguard_stats.get("length_and_truncation_stats", {}),
            "excluded_reasons": wildguard_stats.get("excluded_stats", {}),
        },
        "ordering": {
            "data_seed": config.data_seed,
            "algorithm": "SHA256(f'{seed}:{epoch}:{example_id}') sort key",
        },
    }
    return manifest

