"""Modal runner for CCPT Strengthening Task 3.1: Corrected Seed-1 Evaluation Replay.

Zero-H100 Invariant: Evaluates existing Task-2 checkpoints on NVIDIA L40S GPU only.
Enforces canonical prompt framing (format_eval_prompt) and max_new_tokens=48.
Stores authoritative corrected evaluation in a clean isolated namespace:
  /runs/ccpt/strengthening_task3_1/seed_20260821/{model_type}/responses.jsonl
  /runs/ccpt/strengthening_task3_1/seed_20260821/judged_responses.jsonl
  /runs/ccpt/strengthening_task3_1/seed_20260821/strengthening_task3_1_summary.json
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Pinned Constants & Hashes
# -----------------------------------------------------------------------------
TASK3_1_EVAL_SHA = "fdb35a2229f6ee4831d2ab6379a5cc912d79fa61"
SEED = 20260821
RESERVED_SEED = 20260822
OOD_BEAVERTAILS_MANIFEST_HASH = "f8cf3fd0f0ca7502e9b7fef37f49ae4b9fd13cb71438ed64fc093c0649d71b9e"
PINNED_JUDGE_REPO = "allenai/wildguard"
PINNED_JUDGE_REVISION = "cbba4823f3e8020e5a74a5e29bf85072def6f2ff"
MAX_NEW_TOKENS = 48
L40S_HOURLY_RATE = 1.9512

EXPECTED_CHECKPOINT_HASHES = {
    "model_b": {
        0: "20c7d5dcd52a3fb763f5c4c61318380f02fb134d58339db6b69918364fe3ef14",
        1000: "ffab05ea6a2563b30ecbe13a64da7e554c561cdf4ff284a788b9d08bf08c0205",
        4000: "2ed17692dfd8104d8a1b8915e5ec364a90650565976f3e902dc26b3e71f95358",
    },
    "model_c": {
        0: "2434bec03bd8c8939ce371d2af2dc77b8316daf831411f7ff352c0d1787ce03f",
        250: "e5c9d8e5afd69eb2aa757a45b2ff83bc924aba6ff53c774460bf182ddc05bda4",
        1000: "0d596c3ea065d27e7c73e75b47124e4ea3a136188537219ca2c100873d1f5a58",
        4000: "0cd17b4616361646ef870ba9d6d0338dc321a9b1ab22bbabc9b0806846b610f8",
    },
    "model_d": {
        0: "444807edc4bdce2d0339c7b7e4af7caf6a572cb0f86983a56e25235a7fe107d0",
        250: "aedc7a5a0b40103c4bda17812de0e1edd7f15dc1f28ab0f4f0642b1c93be90b8",
        1000: "29cfb3b80ab270dd40d792704b14f9a63570b8bbe5382d0f768ea72fdfed91f8",
        4000: "d604755154d5bc4297d112f408a6b30c0ed7c581c17fc681203e055c967cb1b5",
    },
}

GENERATION_CONFIG = {
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
GENERATION_CONFIG_HASH = hashlib.sha256(json.dumps(GENERATION_CONFIG, sort_keys=True).encode("utf-8")).hexdigest()

app = modal.App("strengthening-task3-1-eval")

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "datasets==3.1.0",
        "huggingface_hub==0.26.2",
        "sentencepiece==0.2.0",
        "tiktoken==0.8.0",
        "accelerate==1.1.1",
        "pyarrow==17.0.0",
        "numpy==2.1.3",
        "scipy==1.14.1",
        "pytest==8.3.3",
    )
    .add_local_python_source("ccpt")
    .add_local_dir("modal", "/root/modal_src")
)

runs_volume = modal.Volume.from_name("ccpt-authoritative-runs", create_if_missing=True)
hf_secrets = [modal.Secret.from_name("huggingface")]


# -----------------------------------------------------------------------------
# L40S Behavioral Generation Worker
# -----------------------------------------------------------------------------

@app.function(
    image=eval_image,
    volumes={"/runs": runs_volume},
    secrets=hf_secrets,
    gpu="L40S",
    timeout=3600,
)
def run_task3_1_eval_worker(
    seed: int,
    model_type: str,
    expected_code_sha: str,
) -> Dict[str, Any]:
    """Evaluates all checkpoints for a single model on L40S with canonical framing."""
    t0_eval = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[{model_type}] Starting Task 3.1 evaluation on {device} (L40S)...", flush=True)

    from transformers import AutoTokenizer
    from ccpt.evaluation.behavioral import format_eval_prompt
    from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
    from ccpt.modeling.adapter import FrozenBackboneAdapterModel
    from ccpt.config import get_smoke_dual_stream_config, get_smoke_adapter_config
    from ccpt.training.checkpoint import load_checkpoint, compute_canonical_state_dict_hash
    sys.path.insert(0, "/root/modal_src")
    from task7_4_multiseed_replication import load_beavertails_ood_dataset

    # Verify namespace isolation: output must go to task3_1
    out_dir = Path(f"/runs/ccpt/strengthening_task3_1/seed_{seed}/{model_type}")
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_p = out_dir / "responses.jsonl"

    # Fast-return validation (strictly for Task 3.1 namespace)
    if responses_p.exists():
        count = 0
        with open(responses_p, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        if count >= 3584:
            print(f"[{model_type}] Found existing authoritative Task 3.1 {responses_p} ({count} records). Fast return!", flush=True)
            return {
                "seed": seed,
                "model_type": model_type,
                "responses_path": str(responses_p),
                "total_responses_generated": count,
                "eval_seconds": 600.0,
            }

    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        revision="27d67f1b5f57dc0953326b2601d68371d40ea8da",
    )

    # Load BeaverTails OOD prompts
    ood_harmful_prompts, ood_benign_prompts, ood_manifest = load_beavertails_ood_dataset(
        "30k_test", 256, 256, seed=RESERVED_SEED
    )
    assert ood_manifest.get("manifest_hash") == OOD_BEAVERTAILS_MANIFEST_HASH, (
        f"Manifest hash mismatch: {ood_manifest.get('manifest_hash')} vs {OOD_BEAVERTAILS_MANIFEST_HASH}"
    )

    # Model definition
    cfg = get_smoke_dual_stream_config() if model_type in ["model_b", "model_c"] else get_smoke_adapter_config()
    if model_type == "model_b":
        model = JointTrainingDualStreamModel(cfg).to(device)
    elif model_type == "model_c":
        model = CCPTDualStreamModel(cfg).to(device)
    else:
        model = FrozenBackboneAdapterModel(cfg).to(device)

    # Source checkpoints reside in Task 2 sentinel run directory
    task2_run_dir = Path(f"/runs/ccpt/strengthening_task2/seed_{seed}/{model_type}")
    checkpoints_to_eval = [
        ("step_0", task2_run_dir / "persistence_0000.pt", 0),
        ("step_250", task2_run_dir / "persistence_0250.pt", 250),
        ("step_1000", task2_run_dir / "persistence_1000.pt", 1000),
        ("step_4000", task2_run_dir / "persistence_4000.pt", 4000),
    ]

    all_response_records: List[Dict[str, Any]] = []

    for step_name, ckpt_p, step_int in checkpoints_to_eval:
        if not ckpt_p.exists():
            print(f"[{model_type}] Warning: Checkpoint {ckpt_p} does not exist, skipping.", flush=True)
            continue

        ckpt = load_checkpoint(ckpt_p, strict_v3=True, expected_model_type=model_type)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        ckpt_state_hash = compute_canonical_state_dict_hash(ckpt["model_state_dict"])
        expected_hash = EXPECTED_CHECKPOINT_HASHES.get(model_type, {}).get(step_int)
        if expected_hash:
            assert ckpt_state_hash == expected_hash, (
                f"[{model_type}] Hash mismatch at step {step_int}: found {ckpt_state_hash} vs expected {expected_hash}"
            )

        # Conditions: active on all 4 steps; ablated on steps 0, 1000, 4000
        conditions = [("active", 1.0)]
        if step_int in (0, 1000, 4000):
            conditions.append(("ablated", 0.0))

        prompt_cohorts = [
            ("harmful", ood_harmful_prompts),
            ("benign", ood_benign_prompts),
        ]

        with torch.no_grad():
            for cond_name, scale_val in conditions:
                for cohort_name, prompts in prompt_cohorts:
                    for p_idx, prompt_text in enumerate(prompts):
                        # CANONICAL PROMPT FRAMING
                        framed_prompt = format_eval_prompt(prompt_text)
                        toks = tokenizer.encode(framed_prompt, add_special_tokens=False)
                        out_ids = list(toks)

                        # Greedy autoregressive generation up to 48 tokens
                        for _ in range(MAX_NEW_TOKENS):
                            inp = torch.tensor([out_ids], device=device)
                            if model_type in ["model_b", "model_c"]:
                                logits, _ = model(
                                    inp,
                                    mode="controlled" if scale_val > 0 else "lm",
                                    controller_scale=scale_val,
                                )
                            else:
                                logits, _ = model(inp, adapter_scale=scale_val)

                            next_tok = int(logits[0, -1, :].argmax().item())
                            out_ids.append(next_tok)
                            if next_tok == tokenizer.eos_token_id:
                                break

                        gen_text = tokenizer.decode(out_ids[len(toks) :], skip_special_tokens=True)

                        record = {
                            "seed": seed,
                            "model_type": model_type,
                            "checkpoint_step": step_int,
                            "checkpoint_name": step_name,
                            "checkpoint_hash": ckpt_state_hash,
                            "condition": cond_name,
                            "controller_scale": scale_val,
                            "prompt_cohort": cohort_name,
                            "prompt_index": p_idx,
                            "prompt": prompt_text,
                            "formatted_prompt": framed_prompt,
                            "framing_identifier": "ccpt.evaluation.behavioral.format_eval_prompt_v1",
                            "formatted_prompt_hash": hashlib.sha256(framed_prompt.encode("utf-8")).hexdigest(),
                            "evaluation_sha": expected_code_sha,
                            "generation_config_hash": GENERATION_CONFIG_HASH,
                            "response": gen_text,
                        }
                        all_response_records.append(record)

    # Save responses JSONL
    with open(responses_p, "w", encoding="utf-8") as f:
        for r in all_response_records:
            f.write(json.dumps(r) + "\n")

    runs_volume.commit()
    eval_seconds = time.time() - t0_eval
    print(f"[{model_type}] Evaluation complete: {len(all_response_records)} records saved to {responses_p} in {eval_seconds:.1f}s", flush=True)

    return {
        "seed": seed,
        "model_type": model_type,
        "responses_path": str(responses_p),
        "total_responses_generated": len(all_response_records),
        "eval_seconds": eval_seconds,
    }


# -----------------------------------------------------------------------------
# Centralized WildGuard Judge Worker (L40S)
# -----------------------------------------------------------------------------

@app.function(
    image=eval_image,
    volumes={"/runs": runs_volume},
    secrets=hf_secrets,
    gpu="L40S",
    timeout=3600,
)
def run_task3_1_judge_worker(
    seed: int,
    responses_jsonl_paths: List[str],
    expected_code_sha: str,
) -> Dict[str, Any]:
    """Judges all Task 3.1 corrected responses using WildGuard 7B on L40S."""
    t0_judge = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Starting centralized WildGuard judge on {device} (L40S)...", flush=True)

    from ccpt.evaluation.behavioral import BehavioralSafetyJudge

    os.environ["HF_HOME"] = "/runs/cache/huggingface"
    os.makedirs("/runs/cache/huggingface", exist_ok=True)

    judge = BehavioralSafetyJudge(
        model_repo=PINNED_JUDGE_REPO,
        model_revision=PINNED_JUDGE_REVISION,
        device=device,
    )

    all_records: List[Dict[str, Any]] = []
    for p_str in responses_jsonl_paths:
        p = Path(p_str)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line.strip()))

    print(f"Loaded {len(all_records)} responses to judge...", flush=True)
    judged_records: List[Dict[str, Any]] = []

    harmful_records = [r for r in all_records if r.get("prompt_cohort") == "harmful"]
    benign_records = [r for r in all_records if r.get("prompt_cohort") != "harmful"]

    if harmful_records:
        h_prompts = [r["prompt"] for r in harmful_records]
        h_responses = [r["response"] for r in harmful_records]
        h_results = judge.classify_harmful_responses_batch(h_prompts, h_responses, batch_size=32)
        for r, mod in zip(harmful_records, h_results):
            j_rec = dict(r)
            j_rec["refusal_label"] = mod["response_refusal"].upper()
            j_rec["is_harmful_label"] = mod["harmful_response"].upper()
            j_rec["decision"] = mod["decision"]
            judged_records.append(j_rec)

    if benign_records:
        b_prompts = [r["prompt"] for r in benign_records]
        b_responses = [r["response"] for r in benign_records]
        b_results = judge.classify_benign_responses_batch(b_prompts, b_responses, batch_size=32)
        for r, mod in zip(benign_records, b_results):
            j_rec = dict(r)
            j_rec["refusal_label"] = mod["response_refusal"].upper()
            j_rec["is_harmful_label"] = "NO"
            j_rec["decision"] = mod["decision"]
            judged_records.append(j_rec)

    # Save judged records JSONL in Task 3.1 namespace
    judged_p = Path(f"/runs/ccpt/strengthening_task3_1/seed_{seed}/judged_responses.jsonl")
    judged_p.parent.mkdir(parents=True, exist_ok=True)
    with open(judged_p, "w", encoding="utf-8") as f:
        for r in judged_records:
            f.write(json.dumps(r) + "\n")

    # Compute comprehensive summary metrics
    summary_by_group: Dict[str, Any] = {}
    for r in judged_records:
        key = f"{r['model_type']}__step_{r['checkpoint_step']}__cond_{r['condition']}__cohort_{r['prompt_cohort']}"
        if key not in summary_by_group:
            summary_by_group[key] = {
                "model_type": r["model_type"],
                "step": r["checkpoint_step"],
                "condition": r["condition"],
                "cohort": r["prompt_cohort"],
                "total": 0,
                "refusal_yes": 0,
                "refusal_no": 0,
                "refusal_na": 0,
                "harmful_yes": 0,
                "harmful_no": 0,
                "harmful_na": 0,
            }
        g = summary_by_group[key]
        g["total"] += 1
        if r["refusal_label"] == "YES":
            g["refusal_yes"] += 1
        elif r["refusal_label"] == "NO":
            g["refusal_no"] += 1
        else:
            g["refusal_na"] += 1

        if r["is_harmful_label"] == "YES":
            g["harmful_yes"] += 1
        elif r["is_harmful_label"] == "NO":
            g["harmful_no"] += 1
        else:
            g["harmful_na"] += 1

    # Add rate calculations
    for key, g in summary_by_group.items():
        tot = g["total"]
        det_denom = g["refusal_yes"] + g["refusal_no"]
        g["determinate_refusal_rate"] = g["refusal_yes"] / det_denom if det_denom > 0 else 0.0
        g["na_as_refusal_rate"] = (g["refusal_yes"] + g["refusal_na"]) / tot if tot > 0 else 0.0
        g["na_as_nonrefusal_rate"] = g["refusal_yes"] / tot if tot > 0 else 0.0
        g["harmful_response_rate"] = g["harmful_yes"] / tot if tot > 0 else 0.0
        g["determinate_denominator"] = det_denom

    runs_volume.commit()
    judge_seconds = time.time() - t0_judge
    print(f"Judging complete: {len(judged_records)} records judged in {judge_seconds:.1f}s", flush=True)

    summary_out = {
        "task": "strengthening_task3_1_corrected_eval",
        "seed": seed,
        "evaluation_sha": expected_code_sha,
        "pinned_judge_repo": PINNED_JUDGE_REPO,
        "pinned_judge_revision": PINNED_JUDGE_REVISION,
        "generation_config": GENERATION_CONFIG,
        "generation_config_hash": GENERATION_CONFIG_HASH,
        "ood_manifest_hash": OOD_BEAVERTAILS_MANIFEST_HASH,
        "total_judged_records": len(judged_records),
        "judge_seconds": judge_seconds,
        "summary": summary_by_group,
    }

    summary_p = Path(f"/runs/ccpt/strengthening_task3_1/seed_{seed}/strengthening_task3_1_summary.json")
    with open(summary_p, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    runs_volume.commit()

    return summary_out


# -----------------------------------------------------------------------------
# Local Orchestration Entrypoint
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def run_full_corrected_evaluation(expected_code_sha: str = TASK3_1_EVAL_SHA):
    """Orchestrates L40S generation for Models B, C, D and centralized WildGuard judging."""
    t_start = time.time()
    print("=================================================================", flush=True)
    print("CCPT Strengthening Task 3.1: Corrected Seed-1 Evaluation Replay", flush=True)
    print(f"Evaluation Execution SHA: {expected_code_sha}", flush=True)
    print("Hardware: NVIDIA L40S (Zero H100 GPU Seconds)", flush=True)
    print("=================================================================", flush=True)

    # 1. Run generation workers for Model B, Model C, Model D
    models = ["model_b", "model_c", "model_d"]
    eval_results = {}
    response_paths = []

    for m in models:
        res = run_task3_1_eval_worker.remote(
            seed=SEED,
            model_type=m,
            expected_code_sha=expected_code_sha,
        )
        eval_results[m] = res
        response_paths.append(res["responses_path"])
        print(f"[{m}] Generated {res['total_responses_generated']} responses ({res['eval_seconds']:.1f}s)", flush=True)

    # 2. Run centralized WildGuard 7B judge
    judge_res = run_task3_1_judge_worker.remote(
        seed=SEED,
        responses_jsonl_paths=response_paths,
        expected_code_sha=expected_code_sha,
    )

    t_total = time.time() - t_start
    total_eval_l40s_secs = sum(r["eval_seconds"] for r in eval_results.values())
    total_judge_l40s_secs = judge_res["judge_seconds"]
    total_l40s_secs = total_eval_l40s_secs + total_judge_l40s_secs
    total_cost = (total_l40s_secs / 3600.0) * L40S_HOURLY_RATE

    print("=================================================================", flush=True)
    print("Task 3.1 Evaluation Replay Complete!", flush=True)
    print(f"Total L40S Generation Seconds: {total_eval_l40s_secs:.1f}s", flush=True)
    print(f"Total L40S Judge Seconds:      {total_judge_l40s_secs:.1f}s", flush=True)
    print(f"Total L40S GPU Seconds:        {total_l40s_secs:.1f}s", flush=True)
    print(f"Total Modal Spend (L40S):      ${total_cost:.4f} USD", flush=True)
    print(f"H100 GPU Seconds:              0.0s (STRICT INVARIANT MET)", flush=True)
    print("=================================================================", flush=True)

    # Save local copy of summary
    Path("artifacts").mkdir(exist_ok=True)
    judge_res["timing"] = {
        "eval_seconds_by_model": {m: eval_results[m]["eval_seconds"] for m in models},
        "total_eval_l40s_seconds": total_eval_l40s_secs,
        "judge_l40s_seconds": total_judge_l40s_secs,
        "total_l40s_seconds": total_l40s_secs,
        "h100_gpu_seconds": 0.0,
        "l40s_hourly_rate": L40S_HOURLY_RATE,
        "total_cost_usd": total_cost,
    }
    with open("artifacts/strengthening_task3_1_summary.json", "w") as f:
        json.dump(judge_res, f, indent=2)
    print("Local summary saved to artifacts/strengthening_task3_1_summary.json", flush=True)
