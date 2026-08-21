"""Preparation script for WildGuard dataset downloading, processing, and record persistence."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from ccpt.data.config import DataConfig
from ccpt.data.manifests import compute_records_logical_hash
from ccpt.data.tokenizer import load_ccpt_tokenizer
from ccpt.data.wildguard import (
    load_wildguard_records_arrow,
    process_wildguard_raw_dataset,
    save_wildguard_records_arrow,
)


def prepare_wildguard(data_root: str) -> dict:
    """Download, process, and persist wildguardmix records and manifests."""
    config = DataConfig(data_root=data_root)
    tokenizer = load_ccpt_tokenizer(config)

    output_dir = config.get_data_path("wildguard", config.wildguard_revision)
    risk_dir = output_dir / "risk"
    gen_dir = output_dir / "generation"
    risk_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {config.wildguard_repo}@{config.wildguard_revision} ({config.wildguard_train_config})...")
    raw_dataset = load_dataset(
        config.wildguard_repo,
        name=config.wildguard_train_config,
        revision=config.wildguard_revision,
        split="train",
    )
    raw_rows = [dict(r) for r in raw_dataset]
    print(f"Loaded {len(raw_rows)} raw rows.")

    processed = process_wildguard_raw_dataset(
        raw_rows,
        tokenizer=tokenizer,
        dataset_revision=config.wildguard_revision,
        max_seq_len=config.max_seq_len,
    )

    risk_train = processed["risk_train_records"]
    risk_val = processed["risk_val_records"]
    gen_train = processed["gen_train_records"]
    gen_val = processed["gen_val_records"]

    # 1. Persist tokenized records to disk as PyArrow IPC for training consumption
    save_wildguard_records_arrow(risk_train, risk_dir / "train.arrow", record_type="risk")
    save_wildguard_records_arrow(risk_val, risk_dir / "validation.arrow", record_type="risk")
    save_wildguard_records_arrow(gen_train, gen_dir / "train.arrow", record_type="generation")
    save_wildguard_records_arrow(gen_val, gen_dir / "validation.arrow", record_type="generation")

    # 2. Compute full-record logical hashes
    risk_train_hash = compute_records_logical_hash(risk_train)
    risk_val_hash = compute_records_logical_hash(risk_val)
    gen_train_hash = compute_records_logical_hash(gen_train)
    gen_val_hash = compute_records_logical_hash(gen_val)

    stats = {
        "dataset_repo": config.wildguard_repo,
        "dataset_revision": config.wildguard_revision,
        "train_config": config.wildguard_train_config,
        "wildguard_live_validated": True,
        "statistics_source": "pinned_live_dataset",
        "total_raw_rows": processed["total_raw_rows"],
        "prompt_only_rows_count": processed["prompt_only_rows_count"],
        "response_containing_rows_count": processed["response_containing_rows_count"],
        "total_prompt_groups": processed["total_prompt_groups"],
        "usable_prompt_groups": processed["usable_prompt_groups"],
        "conflicting_groups": processed["conflicting_groups"],
        "harmful_risk_count": processed["harmful_risk_count"],
        "benign_risk_count": processed["benign_risk_count"],
        "risk_train_count": len(risk_train),
        "risk_val_count": len(risk_val),
        "gen_train_count": len(gen_train),
        "gen_val_count": len(gen_val),
        "eligible_harmful_refusal": processed["eligible_harmful_refusal"],
        "eligible_benign_compliance": processed["eligible_benign_compliance"],
        "risk_train_hash": risk_train_hash,
        "risk_val_hash": risk_val_hash,
        "gen_train_hash": gen_train_hash,
        "gen_val_hash": gen_val_hash,
        "length_and_truncation_stats": processed["length_and_truncation_stats"],
        "excluded_stats": processed["excluded_stats"],
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"WildGuard processing and persistence complete: {len(risk_train)} risk train, {len(gen_train)} gen train.")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare WildGuard dataset.")
    parser.add_argument("--data-root", type=str, default="data/processed")
    args = parser.parse_args()

    prepare_wildguard(data_root=args.data_root)

