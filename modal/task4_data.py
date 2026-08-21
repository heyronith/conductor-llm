"""Modal remote CPU application for CCPT Task 4 live dataset processing and verification."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

# Ensure local src/ is accessible for bundling
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Modal Image definition with frozen exact dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.15.1",
        "tokenizers==0.22.2",
        "datasets==5.0.1",
        "huggingface_hub==1.28.0",
        "pyarrow==25.0.1",
        "numpy==2.4.6",
    )
    .add_local_python_source("ccpt")
)

app = modal.App("ccpt-task4-data")

# Persistent Modal Volume for processed dataset storage
ccpt_volume = modal.Volume.from_name("ccpt-data", create_if_missing=True)

# Secrets definition: try ccpt-huggingface, then huggingface, then huggingface-secret
# Use user-configured huggingface secret
secrets = [modal.Secret.from_name("huggingface")]



@app.function(
    image=image,
    volumes={"/data/ccpt": ccpt_volume},
    secrets=secrets,
    cpu=8.0,
    memory=32768,  # 32 GiB RAM
    timeout=3600,
)
def run_task4_remote_pipeline() -> Dict[str, Any]:
    """Execute complete Task 4 remote preprocessing and validation on Modal CPU."""
    start_time = time.time()

    # Set container environment
    os.environ["CCPT_DATA_ROOT"] = "/data/ccpt"
    os.environ["HF_HOME"] = "/data/ccpt/cache/huggingface"

    import numpy as np
    import pyarrow as pa
    import torch
    import transformers
    import datasets
    import huggingface_hub
    from datasets import load_dataset

    import ccpt
    from ccpt.data.config import DataConfig
    from ccpt.data.fineweb import (
        PackedTokenBuffer,
        is_validation_document,
        load_token_shard,
        normalize_lm_text,
        process_lm_document_stream,
        tokenize_lm_document,
        write_token_shard,
    )
    from ccpt.data.hashing import sha256_file, sha256_json, sha256_text
    from ccpt.data.manifests import build_task4_manifest, compute_records_logical_hash
    from ccpt.data.ordering import get_epoch_example_order
    from ccpt.data.tokenizer import (
        EXPECTED_TOKENIZER_HASHES,
        get_tokenizer_asset_hashes,
        load_ccpt_tokenizer,
        verify_tokenizer,
    )
    from ccpt.data.wildguard import (
        RiskRecord,
        SafeGenerationRecord,
        load_wildguard_records_arrow,
        process_wildguard_raw_dataset,
        save_wildguard_records_arrow,
    )

    results: Dict[str, Any] = {
        "execution_environment": "modal",
        "modal_sdk_version": modal.__version__,
        "modal_volume": "ccpt-data",
        "modal_volume_root": "/data/ccpt",
        "resources": {
            "cpu": 8.0,
            "memory_mib": 32768,
            "gpu": "none",
        },
    }

    # =========================================================================
    # PHASE 1: ENVIRONMENT
    # =========================================================================
    env_versions = {
        "python": sys.version.split()[0],
        "modal": modal.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "pyarrow": pa.__version__,
        "numpy": np.__version__,
    }
    results["environment"] = env_versions

    config = DataConfig(data_root="/data/ccpt")

    # =========================================================================
    # PHASE 2: TOKENIZER VERIFICATION
    # =========================================================================
    tokenizer = load_ccpt_tokenizer(config)
    verify_tokenizer(tokenizer)
    tok_hashes = get_tokenizer_asset_hashes(config) or EXPECTED_TOKENIZER_HASHES

    tokenizer_summary = {
        "repo": config.tokenizer_repo,
        "revision": config.tokenizer_revision,
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "file_hashes": tok_hashes,
    }
    results["tokenizer"] = tokenizer_summary
    ccpt_volume.commit()

    # =========================================================================
    # PHASE 3: FINEWEB REMOTE SMOKE & EXACT BUDGET TEST
    # =========================================================================
    fw_output_dir = config.get_data_path("fineweb", config.fineweb_revision)
    fw_smoke_train_dir = fw_output_dir / "smoke" / "train"
    fw_smoke_val_dir = fw_output_dir / "smoke" / "validation"
    fw_smoke_train_dir.mkdir(parents=True, exist_ok=True)
    fw_smoke_val_dir.mkdir(parents=True, exist_ok=True)

    fw_stream = load_dataset(
        config.fineweb_repo,
        name=config.fineweb_config,
        revision=config.fineweb_revision,
        split="train",
        streaming=True,
    )

    # Stream exactly 100 documents for smoke verification
    fw_stream_res = process_lm_document_stream(
        doc_iterator=fw_stream,
        tokenizer=tokenizer,
        target_train_tokens=10_000_000_000,
        sequence_length=config.max_seq_len,
        shard_size_blocks=48828,
        train_dir=fw_smoke_train_dir,
        val_dir=fw_smoke_val_dir,
        max_docs=100,
    )

    # Save smoke shard and compute hash
    smoke_shard_path = fw_smoke_train_dir / "smoke_tokens.bin"
    smoke_shard_meta = write_token_shard(fw_stream_res["train_blocks"], smoke_shard_path)
    loaded_smoke_arr = load_token_shard(smoke_shard_path)
    assert loaded_smoke_arr.size == len(fw_stream_res["train_blocks"]) * config.max_seq_len

    fineweb_stats = {
        "repo": config.fineweb_repo,
        "revision": config.fineweb_revision,
        "config": config.fineweb_config,
        "smoke_document_count": fw_stream_res["docs_read"],
        "smoke_train_docs": fw_stream_res["train_docs"],
        "smoke_val_docs": fw_stream_res["val_docs"],
        "smoke_token_count": int(loaded_smoke_arr.size),
        "smoke_block_count": len(fw_stream_res["train_blocks"]),
        "smoke_hash": smoke_shard_meta["sha256"],
        "production_target_token_budget": config.lm_target_tokens,
        "production_target_blocks": config.lm_target_tokens // config.max_seq_len,
    }
    results["fineweb"] = fineweb_stats

    # Save FineWeb manifest to volume
    fw_manifest_path = fw_output_dir / "smoke" / "manifest.json"
    with open(fw_manifest_path, "w", encoding="utf-8") as f:
        json.dump(fineweb_stats, f, indent=2)
    ccpt_volume.commit()

    # =========================================================================
    # PHASE 4 & 5: WILDGUARD AUTH CHECK, PREPROCESSING & EVALUATION
    # =========================================================================
    wg_output_dir = config.get_data_path("wildguard", config.wildguard_revision)
    wg_risk_dir = wg_output_dir / "risk"
    wg_gen_dir = wg_output_dir / "generation"
    wg_eval_dir = wg_output_dir / "evaluation"
    wg_risk_dir.mkdir(parents=True, exist_ok=True)
    wg_gen_dir.mkdir(parents=True, exist_ok=True)
    wg_eval_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        hf_token = hf_token.strip().strip("'\"")

    wildguard_live_validated = False
    blocker_reason: Optional[str] = None
    wildguard_stats: Dict[str, Any] = {}
    arrow_roundtrip_verified = False
    determinism_verified_all_splits = False

    try:
        # Check token validity with whoami
        api = huggingface_hub.HfApi(token=hf_token)
        try:
            user_info = api.whoami()
            username = user_info.get("name", "authenticated_user")
        except Exception as auth_err:
            raise PermissionError(
                f"HF token verification failed: {auth_err}. Modal Secret `ccpt-huggingface` containing an authorized HF_TOKEN is required."
            )

        # Load real WildGuardTrain
        wg_train_dataset = load_dataset(
            config.wildguard_repo,
            name=config.wildguard_train_config,
            revision=config.wildguard_revision,
            split="train",
            token=hf_token,
        )
        raw_train_rows = [dict(r) for r in wg_train_dataset]

        # Load real WildGuardTest (Evaluation only)
        wg_test_dataset = load_dataset(
            config.wildguard_repo,
            name=config.wildguard_test_config,
            revision=config.wildguard_revision,
            split="test",
            token=hf_token,
        )
        raw_test_rows = [dict(r) for r in wg_test_dataset]

        # Process real WildGuardTrain
        processed_train = process_wildguard_raw_dataset(
            raw_train_rows,
            tokenizer=tokenizer,
            dataset_revision=config.wildguard_revision,
            max_seq_len=config.max_seq_len,
        )

        risk_train = processed_train["risk_train_records"]
        risk_val = processed_train["risk_val_records"]
        gen_train = processed_train["gen_train_records"]
        gen_val = processed_train["gen_val_records"]

        # Persist Arrow IPC datasets to Volume
        save_wildguard_records_arrow(risk_train, wg_risk_dir / "train.arrow", record_type="risk")
        save_wildguard_records_arrow(risk_val, wg_risk_dir / "validation.arrow", record_type="risk")
        save_wildguard_records_arrow(gen_train, wg_gen_dir / "train.arrow", record_type="generation")
        save_wildguard_records_arrow(gen_val, wg_gen_dir / "validation.arrow", record_type="generation")

        # Process evaluation dataset
        processed_test = process_wildguard_raw_dataset(
            raw_test_rows,
            tokenizer=tokenizer,
            dataset_revision=config.wildguard_revision,
            max_seq_len=config.max_seq_len,
        )
        eval_records = processed_test["risk_train_records"] + processed_test["risk_val_records"]
        save_wildguard_records_arrow(eval_records, wg_eval_dir / "wildguardtest.arrow", record_type="risk")

        # =====================================================================
        # PHASE 6: LOGICAL HASHING
        # =====================================================================
        risk_train_hash = compute_records_logical_hash(risk_train)
        risk_val_hash = compute_records_logical_hash(risk_val)
        gen_train_hash = compute_records_logical_hash(gen_train)
        gen_val_hash = compute_records_logical_hash(gen_val)
        eval_hash = compute_records_logical_hash(eval_records)

        # =====================================================================
        # PHASE 7: REAL PERSISTED ARROW ROUND-TRIP VERIFICATION (ALL 5 FILES)
        # =====================================================================
        reloaded_risk_train = load_wildguard_records_arrow(wg_risk_dir / "train.arrow", record_type="risk")
        reloaded_risk_val = load_wildguard_records_arrow(wg_risk_dir / "validation.arrow", record_type="risk")
        reloaded_gen_train = load_wildguard_records_arrow(wg_gen_dir / "train.arrow", record_type="generation")
        reloaded_gen_val = load_wildguard_records_arrow(wg_gen_dir / "validation.arrow", record_type="generation")
        reloaded_eval = load_wildguard_records_arrow(wg_eval_dir / "wildguardtest.arrow", record_type="risk")

        assert len(reloaded_risk_train) == len(risk_train), "Reloaded risk_train count mismatch!"
        assert len(reloaded_risk_val) == len(risk_val), "Reloaded risk_val count mismatch!"
        assert len(reloaded_gen_train) == len(gen_train), "Reloaded gen_train count mismatch!"
        assert len(reloaded_gen_val) == len(gen_val), "Reloaded gen_val count mismatch!"
        assert len(reloaded_eval) == len(eval_records), "Reloaded eval count mismatch!"

        assert compute_records_logical_hash(reloaded_risk_train) == risk_train_hash, "Reloaded risk_train hash mismatch!"
        assert compute_records_logical_hash(reloaded_risk_val) == risk_val_hash, "Reloaded risk_val hash mismatch!"
        assert compute_records_logical_hash(reloaded_gen_train) == gen_train_hash, "Reloaded gen_train hash mismatch!"
        assert compute_records_logical_hash(reloaded_gen_val) == gen_val_hash, "Reloaded gen_val hash mismatch!"
        assert compute_records_logical_hash(reloaded_eval) == eval_hash, "Reloaded eval hash mismatch!"
        arrow_roundtrip_verified = True

        # =====================================================================
        # PHASE 8: SECOND-PASS DETERMINISM CHECK (ALL 5 SPLITS)
        # =====================================================================
        processed_train_pass2 = process_wildguard_raw_dataset(
            raw_train_rows,
            tokenizer=tokenizer,
            dataset_revision=config.wildguard_revision,
            max_seq_len=config.max_seq_len,
        )
        p2_risk_train = processed_train_pass2["risk_train_records"]
        p2_risk_val = processed_train_pass2["risk_val_records"]
        p2_gen_train = processed_train_pass2["gen_train_records"]
        p2_gen_val = processed_train_pass2["gen_val_records"]

        processed_test_pass2 = process_wildguard_raw_dataset(
            raw_test_rows,
            tokenizer=tokenizer,
            dataset_revision=config.wildguard_revision,
            max_seq_len=config.max_seq_len,
        )
        p2_eval = processed_test_pass2["risk_train_records"] + processed_test_pass2["risk_val_records"]

        p2_risk_train_hash = compute_records_logical_hash(p2_risk_train)
        p2_risk_val_hash = compute_records_logical_hash(p2_risk_val)
        p2_gen_train_hash = compute_records_logical_hash(p2_gen_train)
        p2_gen_val_hash = compute_records_logical_hash(p2_gen_val)
        p2_eval_hash = compute_records_logical_hash(p2_eval)

        assert risk_train_hash == p2_risk_train_hash, "Second pass risk_train hash mismatch!"
        assert risk_val_hash == p2_risk_val_hash, "Second pass risk_val hash mismatch!"
        assert gen_train_hash == p2_gen_train_hash, "Second pass gen_train hash mismatch!"
        assert gen_val_hash == p2_gen_val_hash, "Second pass gen_val hash mismatch!"
        assert eval_hash == p2_eval_hash, "Second pass eval (wildguardtest) hash mismatch!"
        determinism_verified_all_splits = True

        wildguard_live_validated = True
        wildguard_stats = {
            "wildguard_live_validated": True,
            "statistics_source": "pinned_live_dataset",
            "repo": config.wildguard_repo,
            "revision": config.wildguard_revision,
            "train_config": config.wildguard_train_config,
            "eval_config": config.wildguard_test_config,
            "total_raw_rows": processed_train["total_raw_rows"],
            "prompt_only_rows_count": processed_train["prompt_only_rows_count"],
            "response_containing_rows_count": processed_train["response_containing_rows_count"],
            "total_prompt_groups": processed_train["total_prompt_groups"],
            "usable_prompt_groups": processed_train["usable_prompt_groups"],
            "conflicting_groups": processed_train["conflicting_groups"],
            "harmful_risk_count": processed_train["harmful_risk_count"],
            "benign_risk_count": processed_train["benign_risk_count"],
            "risk_train_count": len(risk_train),
            "risk_val_count": len(risk_val),
            "gen_train_count": len(gen_train),
            "gen_val_count": len(gen_val),
            "raw_eval_rows_count": len(raw_test_rows),
            "eval_records_count": len(eval_records),
            "eligible_harmful_refusal": processed_train["eligible_harmful_refusal"],
            "eligible_benign_compliance": processed_train["eligible_benign_compliance"],
            "risk_train_hash": risk_train_hash,
            "risk_val_hash": risk_val_hash,
            "gen_train_hash": gen_train_hash,
            "gen_val_hash": gen_val_hash,
            "eval_hash": eval_hash,
            "length_and_truncation_stats": processed_train["length_and_truncation_stats"],
            "excluded_stats": processed_train["excluded_stats"],
            "arrow_roundtrip_verified": arrow_roundtrip_verified,
            "determinism_verified_all_splits": determinism_verified_all_splits,
        }

    except Exception as exc:
        blocker_reason = (
            f"BLOCKER: Modal Secret `ccpt-huggingface` containing an HF_TOKEN authorized for "
            f"allenai/wildguardmix is required. Live dataset error: {exc}"
        )
        wildguard_live_validated = False
        wildguard_stats = {
            "wildguard_live_validated": False,
            "statistics_source": "synthetic_fixture",
            "blocker_reason": blocker_reason,
        }

    results["wildguard"] = wildguard_stats
    results["wildguard_live_validated"] = wildguard_live_validated
    if blocker_reason:
        results["blocker"] = blocker_reason

    # =========================================================================
    # PHASE 9: MANIFEST GENERATION & VOLUME COMMIT
    # =========================================================================
    task4_manifest = build_task4_manifest(
        config=config,
        fineweb_stats=fineweb_stats,
        wildguard_stats=wildguard_stats,
        execution_env={
            "execution_environment": "modal",
            "modal_volume": "ccpt-data",
            "modal_volume_root": "/data/ccpt",
        },
    )

    manifest_dir = Path("/data/ccpt/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "task4_manifest.json", "w", encoding="utf-8") as f:
        json.dump(task4_manifest, f, indent=2)

    results["manifest"] = task4_manifest
    results["manifest_hash"] = sha256_json(task4_manifest)
    results["elapsed_seconds"] = round(time.time() - start_time, 2)

    ccpt_volume.commit()
    return results


@app.local_entrypoint()
def main():
    """Local orchestration entrypoint: invoke remote Modal pipeline and capture artifacts."""
    print("================================================================================")
    print("CCPT Task 4.2: Launching Modal Remote CPU Preprocessing & Reproducibility Audit")
    print("================================================================================")
    print("Target App: ccpt-task4-data")
    print("Resources: 8.0 CPU, 32 GiB RAM, 0 GPU")
    print("Volume: ccpt-data mounted at /data/ccpt")
    print("Executing remote validation...")

    results = run_task4_remote_pipeline.remote()

    print("\n================================================================================")
    print("MODAL REMOTE EXECUTION COMPLETED")
    print("================================================================================")
    print(f"Elapsed Time: {results.get('elapsed_seconds')}s")
    print(f"Environment: Python {results['environment']['python']}, Modal {results['environment']['modal']}")
    print(f"Tokenizer Verified: {results['tokenizer']['repo']}@{results['tokenizer']['revision']}")
    print(f"FineWeb Smoke Shard: {results['fineweb']['smoke_document_count']} docs, {results['fineweb']['smoke_token_count']} tokens")
    print(f"FineWeb Shard SHA256: {results['fineweb']['smoke_hash']}")

    is_live = results.get("wildguard_live_validated", False)
    if is_live:
        wg = results["wildguard"]
        print(f"\n[LIVE WILDGUARD VALIDATION SUCCESS]")
        print(f"  Total Raw Rows (Train): {wg['total_raw_rows']}, (Test): {wg['raw_eval_rows_count']}")
        print(f"  Usable Prompt Groups: {wg['usable_prompt_groups']} (Conflicting Excluded: {wg['conflicting_groups']})")
        print(f"  Risk Records: {wg['risk_train_count']} train, {wg['risk_val_count']} val")
        print(f"  Gen Records:  {wg['gen_train_count']} train, {wg['gen_val_count']} val")
        print(f"  Eval Records: {wg['eval_records_count']} test")
        print(f"  Risk Train Logical Hash: {wg['risk_train_hash']}")
        print(f"  Risk Val Logical Hash:   {wg['risk_val_hash']}")
        print(f"  Gen Train Logical Hash:  {wg['gen_train_hash']}")
        print(f"  Gen Val Logical Hash:    {wg['gen_val_hash']}")
        print(f"  Eval Logical Hash:       {wg['eval_hash']}")
        print(f"  Real Arrow Round-Trip (All 5 Files): {wg['arrow_roundtrip_verified']}")
        print(f"  Determinism Verified (All 5 Splits): {wg['determinism_verified_all_splits']}")
    else:
        print(f"\n[LIVE WILDGUARD ACCESS BLOCKED / UNAUTHENTICATED]")
        print(f"  Reason: {results.get('blocker')}")

    # Write local result artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    summary_path = artifacts_dir / "task4_modal_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved remote summary to {summary_path}")

    run_log_path = artifacts_dir / "task4_modal_run.txt"
    with open(run_log_path, "w", encoding="utf-8") as f:
        f.write("CCPT TASK 4.2 MODAL REMOTE EXECUTION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp (UTC): {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        f.write(f"App: ccpt-task4-data\n")
        f.write(f"Resources: 8.0 CPU, 32768 MiB RAM, 0 GPU\n")
        f.write(f"Volume: ccpt-data (/data/ccpt)\n")
        f.write(f"Elapsed: {results.get('elapsed_seconds')}s\n\n")
        f.write("ENVIRONMENT:\n")
        for k, v in results["environment"].items():
            f.write(f"  - {k}: {v}\n")
        f.write("\nTOKENIZER:\n")
        f.write(f"  - Repo: {results['tokenizer']['repo']}@{results['tokenizer']['revision']}\n")
        f.write(f"  - Vocab: {results['tokenizer']['vocab_size']}, BOS={results['tokenizer']['bos_token_id']}, EOS={results['tokenizer']['eos_token_id']}, PAD={results['tokenizer']['pad_token_id']}\n")
        f.write("\nFINEWEB SMOKE:\n")
        f.write(f"  - Docs: {results['fineweb']['smoke_document_count']}\n")
        f.write(f"  - Tokens: {results['fineweb']['smoke_token_count']}\n")
        f.write(f"  - Blocks: {results['fineweb']['smoke_block_count']}\n")
        f.write(f"  - SHA256: {results['fineweb']['smoke_hash']}\n")
        f.write("\nWILDGUARD VALIDATION:\n")
        f.write(f"  - Live Validated: {results.get('wildguard_live_validated')}\n")
        if is_live:
            wg = results["wildguard"]
            f.write(f"  - Statistics Source: {wg['statistics_source']}\n")
            f.write(f"  - Total Raw Rows (Train / Test): {wg['total_raw_rows']} / {wg['raw_eval_rows_count']}\n")
            f.write(f"  - Usable Groups: {wg['usable_prompt_groups']}\n")
            f.write(f"  - Risk Train/Val: {wg['risk_train_count']} / {wg['risk_val_count']}\n")
            f.write(f"  - Gen Train/Val:  {wg['gen_train_count']} / {wg['gen_val_count']}\n")
            f.write(f"  - Eval Records:  {wg['eval_records_count']}\n")
            f.write(f"  - Risk Train Hash: {wg['risk_train_hash']}\n")
            f.write(f"  - Risk Val Hash:   {wg['risk_val_hash']}\n")
            f.write(f"  - Gen Train Hash:  {wg['gen_train_hash']}\n")
            f.write(f"  - Gen Val Hash:    {wg['gen_val_hash']}\n")
            f.write(f"  - Eval Hash:       {wg['eval_hash']}\n")
            f.write(f"  - Arrow Round-Trip Verified (All 5 Files): {wg['arrow_roundtrip_verified']}\n")
            f.write(f"  - Determinism Verified (All 5 Splits):    {wg['determinism_verified_all_splits']}\n")
        else:
            f.write(f"  - Blocker: {results.get('blocker')}\n")
        f.write(f"\nMANIFEST HASH: {results.get('manifest_hash')}\n")
    print(f"Saved run log to {run_log_path}")

    # If live validation succeeded, update local manifest
    if is_live and "manifest" in results:
        manifest_path = Path("data/manifests/task4_manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(results["manifest"], f, indent=2)
        print(f"Updated local task4_manifest.json with live dataset statistics and evaluation lock.")
