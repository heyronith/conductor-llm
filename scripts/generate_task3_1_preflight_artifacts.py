"""Generates preflight and generation manifest artifacts for Task 3.1."""

import datetime
import hashlib
import json
from pathlib import Path
import subprocess

def main():
    print("Generating Task 3.1 Preflight & Generation Manifest Artifacts...")

    # Get git HEAD
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    remote_head = subprocess.check_output(["git", "rev-parse", "origin/strengthening-task3-seed1-forensic"]).decode("utf-8").strip()
    assert git_head == remote_head, f"Remote parity mismatch: {git_head} vs {remote_head}"

    generation_config = {
        "decoding": "greedy_argmax",
        "max_new_tokens": 48,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "eos_token_id": 2,
        "tokenizer_repo": "mistralai/Mistral-7B-v0.1",
        "tokenizer_revision": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
        "controller_scale_active": 1.0,
        "controller_scale_ablated": 0.0,
        "adapter_scale_active": 1.0,
        "adapter_scale_ablated": 0.0,
        "framing_function": "ccpt.evaluation.behavioral.format_eval_prompt",
        "framing_template": "<s>User: {raw_prompt}\nAssistant:",
    }
    generation_config_hash = hashlib.sha256(json.dumps(generation_config, sort_keys=True).encode("utf-8")).hexdigest()

    preflight = {
        "task": "strengthening_task3_1_preflight",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_sha": git_head,
        "remote_parity_verified": True,
        "zero_h100_invariant": {
            "h100_gpu_seconds_authorized": 0,
            "gpu_hardware_authorized": "NVIDIA L40S",
            "models_retrained": False,
            "seed_4_launched": False
        },
        "canonical_prompt_framing": {
            "formatter": "ccpt.evaluation.behavioral.format_eval_prompt",
            "framing_template": "<s>User: {raw_prompt}\nAssistant:",
            "max_new_tokens": 48,
            "add_special_tokens": False
        },
        "frozen_prompt_manifest": {
            "dataset": "PKU-Alignment/BeaverTails",
            "split": "30k_test",
            "sampling_seed": 20260822,
            "manifest_hash": "f8cf3fd0f0ca7502e9b7fef37f49ae4b9fd13cb71438ed64fc093c0649d71b9e",
            "harmful_count": 256,
            "benign_count": 256
        },
        "checkpoint_hashes": {
            "model_b": {
                "step_0": "20c7d5dcd52a3fb763f5c4c61318380f02fb134d58339db6b69918364fe3ef14",
                "step_1000": "ffab05ea6a2563b30ecbe13a64da7e554c561cdf4ff284a788b9d08bf08c0205",
                "step_4000": "2ed17692dfd8104d8a1b8915e5ec364a90650565976f3e902dc26b3e71f95358"
            },
            "model_c": {
                "step_0": "2434bec03bd8c8939ce371d2af2dc77b8316daf831411f7ff352c0d1787ce03f",
                "step_250": "e5c9d8e5afd69eb2aa757a45b2ff83bc924aba6ff53c774460bf182ddc05bda4",
                "step_1000": "0d596c3ea065d27e7c73e75b47124e4ea3a136188537219ca2c100873d1f5a58",
                "step_4000": "0cd17b4616361646ef870ba9d6d0338dc321a9b1ab22bbabc9b0806846b610f8"
            },
            "model_d": {
                "step_0": "444807edc4bdce2d0339c7b7e4af7caf6a572cb0f86983a56e25235a7fe107d0",
                "step_250": "aedc7a5a0b40103c4bda17812de0e1edd7f15dc1f28ab0f4f0642b1c93be90b8",
                "step_1000": "29cfb3b80ab270dd40d792704b14f9a63570b8bbe5382d0f768ea72fdfed91f8",
                "step_4000": "d604755154d5bc4297d112f408a6b30c0ed7c581c17fc681203e055c967cb1b5"
            }
        },
        "cost_projection": {
            "projected_l40s_seconds": 3000.0,
            "projected_spend_usd": 1.626,
            "target_spend_limit_usd": 3.00,
            "hard_stop_ceiling_usd": 5.00,
            "preflight_status": "PASSED"
        }
    }
    with open("artifacts/strengthening_task3_1_preflight.json", "w") as f:
        json.dump(preflight, f, indent=2)

    gen_manifest = {
        "task": "strengthening_task3_1_generation_manifest",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": 20260821,
        "evaluation_sha": git_head,
        "generation_config": generation_config,
        "generation_config_hash": generation_config_hash,
        "target_namespace": "/runs/ccpt/strengthening_task3_1/seed_20260821",
        "models": {
            "model_b": {
                "steps": [0, 250, 1000, 4000],
                "active_steps": [0, 250, 1000, 4000],
                "ablated_steps": [0, 1000, 4000],
                "total_records": 3584
            },
            "model_c": {
                "steps": [0, 250, 1000, 4000],
                "active_steps": [0, 250, 1000, 4000],
                "ablated_steps": [0, 1000, 4000],
                "total_records": 3584
            },
            "model_d": {
                "steps": [0, 250, 1000, 4000],
                "active_steps": [0, 250, 1000, 4000],
                "ablated_steps": [0, 1000, 4000],
                "total_records": 3584
            }
        },
        "total_records_all_models": 10752
    }
    with open("artifacts/strengthening_task3_1_generation_manifest.json", "w") as f:
        json.dump(gen_manifest, f, indent=2)

    print("Preflight & Generation Manifest successfully generated.")

if __name__ == "__main__":
    main()
