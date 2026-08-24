"""Task 8: Modal Execution Script for Prespecified Mechanistic Heterogeneity Analysis.

Executes deterministic prompt-boundary forward-pass diagnostics on Model C and Model D
across Seeds 20260821, 20260823, 20260824 on Modal persistent volumes.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
import importlib.util

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path("/Users/ronny/Desktop/Research/AI ALIGNMENT/CCPT")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "modal"))

CODE_SHA = "4e69012026fe94e9ca551cce95c9f21fca3b90ef"
os.environ["CCPT_CODE_COMMIT_SHA"] = CODE_SHA

try:
    import task7_4_multiseed_replication as task74_mod
except ImportError:
    mod_path = PROJECT_ROOT / "modal" / "task7_4_multiseed_replication.py"
    spec = importlib.util.spec_from_file_location("task7_4_multiseed_replication", str(mod_path))
    task74_mod = importlib.util.module_from_spec(spec)
    sys.modules["task7_4_multiseed_replication"] = task74_mod
    spec.loader.exec_module(task74_mod)

app = task74_mod.app
replication_image = task74_mod.replication_image
data_volume = task74_mod.data_volume
task4_data_volume = task74_mod.task4_data_volume
runs_volume = task74_mod.runs_volume
import modal

task8_image = (
    replication_image
    .add_local_file(PROJECT_ROOT / "modal" / "task7_4_multiseed_replication.py", remote_path="/root/task7_4_multiseed_replication.py")
    .add_local_file(PROJECT_ROOT / "src" / "ccpt" / "analysis" / "task8_mechanistic.py", remote_path="/root/task8_mechanistic.py")
)


@app.function(
    image=task8_image,
    gpu=None,  # CPU execution first as specified
    volumes={"/data": data_volume, "/data_task4": task4_data_volume, "/runs": runs_volume},
    timeout=3600,
)
def run_task8_mechanistic_diagnostics(test_mode: bool = False) -> Dict[str, Any]:
    """Execute complete prespecified mechanistic diagnostics across all 3 seeds."""
    from transformers import AutoTokenizer
    from task7_4_multiseed_replication import (
        resolve_canonical_wildguard_artifacts,
        load_wildguard_records_arrow,
        sample_wildguard_id_behavior_prompts,
        load_beavertails_ood_dataset,
        extract_raw_prompt,
        format_eval_prompt,
    )
    from ccpt.config import (
        get_smoke_dual_stream_config,
        get_smoke_adapter_config,
        get_micro_dual_stream_config,
        get_micro_adapter_config,
    )
    from ccpt.modeling.dual_stream import CCPTDualStreamModel
    from ccpt.modeling.adapter import FrozenBackboneAdapterModel
    from ccpt.analysis.task8_mechanistic import (
        cosine_similarity,
        relative_l2,
        vector_norm,
        jensen_shannon_divergence,
        compute_linear_cka,
        classify_behavioral_transition,
        ModelCDiagnosticHooks,
        ModelDDiagnosticHooks,
    )

    device = torch.device("cpu")
    num_prompts = 10 if test_mode else 256

    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1", revision="27d67f1b5f57dc0953326b2601d68371d40ea8da"
    )

    # 1. Load benchmark datasets
    wg_artifacts = resolve_canonical_wildguard_artifacts(require_arrow_only=True)
    risk_val_recs = load_wildguard_records_arrow(wg_artifacts["risk_val"]["resolved_path"], record_type="risk")

    id_harmful_raw, id_benign_raw, _ = sample_wildguard_id_behavior_prompts(risk_val_recs, tokenizer, num_prompts, num_prompts)
    ood_harmful_raw, ood_benign_raw, _ = load_beavertails_ood_dataset("30k_test", num_prompts, num_prompts, seed=20260822)

    prompt_datasets = {
        "id_wildguard_harmful": [extract_raw_prompt(p) for p in id_harmful_raw],
        "id_wildguard_benign": [extract_raw_prompt(p) for p in id_benign_raw],
        "ood_beavertails_harmful": [extract_raw_prompt(p) for p in ood_harmful_raw],
        "ood_beavertails_benign": [extract_raw_prompt(p) for p in ood_benign_raw],
    }

    seeds = [20260821, 20260823, 20260824]
    models = ["model_c", "model_d"]

    results: Dict[str, Any] = {
        "audit_version": "task8_mechanistic_diagnostics_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "models": models,
        "per_prompt_records": [],
        "cka_summary": {},
        "model_c_summary": {},
        "model_d_summary": {},
        "transition_group_summary": {},
        "selectivity_summary": {},
        "causal_ablation_summary": {},
        "immutability_passed": True,
    }

    # Helper to resolve checkpoint paths across Task 7.3 and Task 7.4
    def get_ckpt_paths(seed: int, model_name: str) -> Tuple[Path, Path]:
        if seed == 20260821:
            base = Path("/runs/ccpt/task7_3/pilot_v2_authoritative_run_20260822") / model_name
        else:
            base = Path(f"/runs/ccpt/task7_4/multiseed_replication_v1/seed_{seed}/{model_name}")
        return base / "safety_20m_final.pt", base / "persistence_1000_final.pt"

    # Helper to load judged records for behavioral transitions
    judge_decisions: Dict[Tuple[int, str, str, str], Dict[str, str]] = {}
    # Load Seeds 2 & 3 judge records
    for s in [20260823, 20260824]:
        j_path = Path(f"/runs/ccpt/task7_4/multiseed_replication_v1/seed_{s}/centralized_judged_records.jsonl")
        if j_path.exists():
            with open(j_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line.strip())
                        # Key: (seed, model, condition, dataset, prompt_type, prompt_str)
                        k = (s, r["model"], r["condition"], r["dataset"], r["prompt_type"], r["prompt"])
                        if k not in judge_decisions:
                            judge_decisions[k] = {}
                        judge_decisions[k][r["phase"]] = r.get("response_refusal", "NA")

    # Iterate across seeds and models
    for s in seeds:
        for m in models:
            tag = f"seed_{s}_{m}"
            p_pre, p_post = get_ckpt_paths(s, m)
            if not p_pre.exists() or not p_post.exists():
                print(f"Skipping {tag}: Checkpoints not found ({p_pre}, {p_post})")
                continue

            # Load PRE and POST checkpoints
            ckpt_pre = torch.load(str(p_pre), map_location=device, weights_only=False)
            ckpt_post = torch.load(str(p_post), map_location=device, weights_only=False)

            # Checkpoint SHA immutability baseline
            sha_pre_before = hashlib.sha256(b"".join([t.numpy().tobytes() for t in ckpt_pre["model_state_dict"].values()])).hexdigest()
            sha_post_before = hashlib.sha256(b"".join([t.numpy().tobytes() for t in ckpt_post["model_state_dict"].values()])).hexdigest()

            if m == "model_c":
                cfg = get_smoke_dual_stream_config() if not test_mode else get_micro_dual_stream_config()
                model_pre = CCPTDualStreamModel(cfg).to(device)
                model_post = CCPTDualStreamModel(cfg).to(device)
            else:
                cfg = get_smoke_adapter_config() if not test_mode else get_micro_adapter_config()
                model_pre = FrozenBackboneAdapterModel(cfg).to(device)
                model_post = FrozenBackboneAdapterModel(cfg).to(device)

            model_pre.load_state_dict(ckpt_pre["model_state_dict"])
            model_post.load_state_dict(ckpt_post["model_state_dict"])
            model_pre.eval()
            model_post.eval()

            # Execute prompt forward passes across all 4 datasets
            for dset_name, prompt_list in prompt_datasets.items():
                is_harmful = "harmful" in dset_name
                p_type = "harmful" if is_harmful else "benign"
                dataset_category = "id_wildguard" if "id" in dset_name else "ood_beavertails"

                # Containers for CKA matrices
                c_tensors_pre: Dict[str, List[np.ndarray]] = {}
                c_tensors_post: Dict[str, List[np.ndarray]] = {}

                for p_idx, raw_p in enumerate(prompt_list):
                    example_id = f"{dset_name}_{p_idx:04d}"
                    framed = format_eval_prompt(raw_p)
                    input_ids = tokenizer.encode(framed, add_special_tokens=False)
                    in_t = torch.tensor([input_ids], dtype=torch.long, device=device)
                    prompt_end_idx = torch.tensor([len(input_ids) - 1], dtype=torch.long, device=device)

                    rec: Dict[str, Any] = {
                        "example_id": example_id,
                        "seed": s,
                        "model": m,
                        "dataset": dataset_category,
                        "prompt_type": p_type,
                        "prompt_index": p_idx,
                    }

                    # Behavioral transition lookup
                    j_key = (s, m, "on", dataset_category, p_type, raw_p)
                    pre_dec = judge_decisions.get(j_key, {}).get("pre_persistence", "NA")
                    post_dec = judge_decisions.get(j_key, {}).get("post_persistence", "NA")
                    rec["transition_group"] = classify_behavioral_transition(pre_dec, post_dec)

                    if m == "model_c":
                        with torch.no_grad(), ModelCDiagnosticHooks(model_pre) as hooks_pre:
                            logits_pre_active, _ = model_pre(in_t, prompt_end_indices=prompt_end_idx, mode="controlled", controller_scale=1.0)
                        with torch.no_grad():
                            logits_pre_off, _ = model_pre(in_t, prompt_end_indices=prompt_end_idx, mode="controlled", controller_scale=0.0)

                        with torch.no_grad(), ModelCDiagnosticHooks(model_post) as hooks_post:
                            logits_post_active, _ = model_post(in_t, prompt_end_indices=prompt_end_idx, mode="controlled", controller_scale=1.0)
                        with torch.no_grad():
                            logits_post_off, _ = model_post(in_t, prompt_end_indices=prompt_end_idx, mode="controlled", controller_scale=0.0)

                        # Compute active/off next-token JS divergence at prompt boundary
                        prob_pre_on = F.softmax(logits_pre_active[0, -1, :], dim=-1).cpu().numpy()
                        prob_pre_off = F.softmax(logits_pre_off[0, -1, :], dim=-1).cpu().numpy()
                        prob_post_on = F.softmax(logits_post_active[0, -1, :], dim=-1).cpu().numpy()
                        prob_post_off = F.softmax(logits_post_off[0, -1, :], dim=-1).cpu().numpy()

                        js_pre = jensen_shannon_divergence(prob_pre_on, prob_pre_off)
                        js_post = jensen_shannon_divergence(prob_post_on, prob_post_off)
                        rec["active_off_js_pre"] = js_pre
                        rec["active_off_js_post"] = js_post
                        rec["active_off_js_change"] = js_post - js_pre

                        # Controlled layers 2 and 4 metrics
                        for l_num in [2, 4]:
                            # Vectors at prompt boundary (index -1)
                            c_pre = hooks_pre.captured[f"c_tilde_layer_{l_num}"][0, -1, :].cpu().numpy()
                            c_post = hooks_post.captured[f"c_tilde_layer_{l_num}"][0, -1, :].cpu().numpy()
                            obs_pre = hooks_pre.captured[f"obs_layer_{l_num}"][0, -1, :].cpu().numpy()
                            obs_post = hooks_post.captured[f"obs_layer_{l_num}"][0, -1, :].cpu().numpy()
                            norm_pre = hooks_pre.captured[f"normative_layer_{l_num}"][0, -1, :].cpu().numpy()
                            norm_post = hooks_post.captured[f"normative_layer_{l_num}"][0, -1, :].cpu().numpy()
                            g_pre = hooks_pre.captured[f"gate_scaled_layer_{l_num}"][0, -1, 0].item()
                            g_post = hooks_post.captured[f"gate_scaled_layer_{l_num}"][0, -1, 0].item()
                            s_pre = hooks_pre.captured[f"steering_scaled_layer_{l_num}"][0, -1, :].cpu().numpy()
                            s_post = hooks_post.captured[f"steering_scaled_layer_{l_num}"][0, -1, :].cpu().numpy()

                            # Store for CKA calculation
                            c_tensors_pre.setdefault(f"c_tilde_{l_num}", []).append(c_pre)
                            c_tensors_post.setdefault(f"c_tilde_{l_num}", []).append(c_post)
                            c_tensors_pre.setdefault(f"obs_{l_num}", []).append(obs_pre)
                            c_tensors_post.setdefault(f"obs_{l_num}", []).append(obs_post)
                            c_tensors_pre.setdefault(f"norm_{l_num}", []).append(norm_pre)
                            c_tensors_post.setdefault(f"norm_{l_num}", []).append(norm_post)
                            c_tensors_pre.setdefault(f"steer_{l_num}", []).append(s_pre)
                            c_tensors_post.setdefault(f"steer_{l_num}", []).append(s_post)

                            # Metrics
                            rec[f"layer_{l_num}_capability_cosine"] = cosine_similarity(c_pre, c_post)
                            rec[f"layer_{l_num}_capability_relative_l2"] = relative_l2(c_pre, c_post)
                            rec[f"layer_{l_num}_obs_cosine"] = cosine_similarity(obs_pre, obs_post)
                            rec[f"layer_{l_num}_obs_relative_l2"] = relative_l2(obs_pre, obs_post)
                            rec[f"layer_{l_num}_normative_cosine"] = cosine_similarity(norm_pre, norm_post)
                            rec[f"layer_{l_num}_normative_relative_l2"] = relative_l2(norm_pre, norm_post)
                            rec[f"layer_{l_num}_gate_strength_pre"] = abs(g_pre - 1.0)
                            rec[f"layer_{l_num}_gate_strength_post"] = abs(g_post - 1.0)
                            rec[f"layer_{l_num}_gate_absolute_change"] = abs(g_post - g_pre)
                            rec[f"layer_{l_num}_steering_norm_pre"] = vector_norm(s_pre)
                            rec[f"layer_{l_num}_steering_norm_post"] = vector_norm(s_post)
                            rec[f"layer_{l_num}_steering_cosine"] = cosine_similarity(s_pre, s_post)
                            rec[f"layer_{l_num}_steering_relative_l2"] = relative_l2(s_pre, s_post)

                    else:
                        # Model D
                        with torch.no_grad(), ModelDDiagnosticHooks(model_pre) as hooks_pre:
                            logits_pre_active, _ = model_pre(in_t, adapter_scale=1.0)
                        with torch.no_grad():
                            logits_pre_off, _ = model_pre(in_t, adapter_scale=0.0)

                        with torch.no_grad(), ModelDDiagnosticHooks(model_post) as hooks_post:
                            logits_post_active, _ = model_post(in_t, adapter_scale=1.0)
                        with torch.no_grad():
                            logits_post_off, _ = model_post(in_t, adapter_scale=0.0)

                        prob_pre_on = F.softmax(logits_pre_active[0, -1, :], dim=-1).cpu().numpy()
                        prob_pre_off = F.softmax(logits_pre_off[0, -1, :], dim=-1).cpu().numpy()
                        prob_post_on = F.softmax(logits_post_active[0, -1, :], dim=-1).cpu().numpy()
                        prob_post_off = F.softmax(logits_post_off[0, -1, :], dim=-1).cpu().numpy()

                        js_pre = jensen_shannon_divergence(prob_pre_on, prob_pre_off)
                        js_post = jensen_shannon_divergence(prob_post_on, prob_post_off)
                        rec["active_off_js_pre"] = js_pre
                        rec["active_off_js_post"] = js_post
                        rec["active_off_js_change"] = js_post - js_pre

                        # 8 Adapter sites
                        for l_idx in range(4):
                            for a_type in ["attn", "mlp"]:
                                site_name = f"layer_{l_idx}_{a_type}_adapter"
                                in_pre = hooks_pre.captured[f"{site_name}_input"][0, -1, :].cpu().numpy()
                                in_post = hooks_post.captured[f"{site_name}_input"][0, -1, :].cpu().numpy()
                                res_pre = hooks_pre.captured[f"{site_name}_residual"][0, -1, :].cpu().numpy()
                                res_post = hooks_post.captured[f"{site_name}_residual"][0, -1, :].cpu().numpy()

                                c_tensors_pre.setdefault(f"{site_name}_in", []).append(in_pre)
                                c_tensors_post.setdefault(f"{site_name}_in", []).append(in_post)
                                c_tensors_pre.setdefault(f"{site_name}_res", []).append(res_pre)
                                c_tensors_post.setdefault(f"{site_name}_res", []).append(res_post)

                                rec[f"{site_name}_input_cosine"] = cosine_similarity(in_pre, in_post)
                                rec[f"{site_name}_input_relative_l2"] = relative_l2(in_pre, in_post)
                                rec[f"{site_name}_residual_norm_pre"] = vector_norm(res_pre)
                                rec[f"{site_name}_residual_norm_post"] = vector_norm(res_post)
                                rec[f"{site_name}_residual_cosine"] = cosine_similarity(res_pre, res_post)
                                rec[f"{site_name}_residual_relative_l2"] = relative_l2(res_pre, res_post)

                    results["per_prompt_records"].append(rec)

                # Compute Linear CKA for each diagnostic site across this dataset
                cka_key_prefix = f"seed_{s}_{m}_{dset_name}"
                for site_k in c_tensors_pre:
                    X = np.array(c_tensors_pre[site_k], dtype=np.float64)
                    Y = np.array(c_tensors_post[site_k], dtype=np.float64)
                    cka_val = compute_linear_cka(X, Y)
                    results["cka_summary"][f"{cka_key_prefix}_{site_k}"] = cka_val

            # Checkpoint immutability verification
            sha_pre_after = hashlib.sha256(b"".join([t.numpy().tobytes() for t in model_pre.state_dict().values()])).hexdigest()
            sha_post_after = hashlib.sha256(b"".join([t.numpy().tobytes() for t in model_post.state_dict().values()])).hexdigest()
            if sha_pre_before != sha_pre_after or sha_post_before != sha_post_after:
                results["immutability_passed"] = False
                raise RuntimeError(f"Immutability violation in {tag}!")

    return results


def main():
    print("=== TASK 8: PRESPECIFIED MECHANISTIC HETEROGENEITY ANALYSIS ===", flush=True)
    with modal.enable_output():
        with app.run():
            print(" -> Running diagnostic extraction on Modal CPU...", flush=True)
            res = run_task8_mechanistic_diagnostics.remote(test_mode=False)

            out_dir = PROJECT_ROOT / "artifacts"
            out_dir.mkdir(parents=True, exist_ok=True)

            out_summary_p = out_dir / "task8_mechanistic_summary.json"
            with open(out_summary_p, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

            print(f" -> Diagnostic results written to {out_summary_p}", flush=True)

if __name__ == "__main__":
    main()
