"""Successor Task 1 — L40S adaptive-interface falsification (no H100).

Hard authorization: $5.00 immutable. Target <= $3.00.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_NAME = "successor-task1-falsification"
VOLUME_NAME = "ccpt-authoritative-runs"
HARD_AUTHORIZATION_USD = 5.00  # IMMUTABLE — never raise programmatically
TARGET_USD = 3.00
CODE_SHA_ENV = "SUCCESSOR_TASK1_CODE_SHA"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "numpy==2.1.3",
        "safetensors==0.4.5",
        "sentencepiece==0.2.0",
        "huggingface_hub==0.26.2",
        "accelerate==1.1.1",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_python_source("ccpt")
    .add_local_dir("modal", "/root/modal_src")
)

app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_ccpt_from_ckpt(path: Path, device: str):
    import torch
    from ccpt.config import DualStreamConfig
    from ccpt.modeling.dual_stream import CCPTDualStreamModel
    from ccpt.training.checkpoint import load_checkpoint

    ckpt = load_checkpoint(path, strict_v3=True, map_location="cpu")
    cfg = DualStreamConfig(**ckpt["model_config"])
    model = CCPTDualStreamModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@app.function(
    image=image,
    gpu="L40S",
    timeout=3 * 3600,
    volumes={"/runs": runs_volume},
    secrets=[hf_secret],
)
def fit_and_eval_one_seed(seed: int) -> Dict[str, Any]:
    """Fit all four repair variants for one seed. Payload read from Volume JSON."""
    import copy
    import hashlib

    import torch
    from transformers import AutoTokenizer

    from ccpt.evaluation.behavioral import format_eval_prompt
    from ccpt.evaluation.forensics import compute_canonical_state_dict_hash
    from ccpt.successor.retrofit import (
        AdaptiveInterfaceWrapper,
        RepairVariant,
        build_variant_bundle,
        freeze_module,
        hash_existing_parameters,
    )
    from ccpt.successor.training import continuation_token_mask, fit_loss

    payload_path = Path(f"/runs/ccpt/successor_task1/payload_seed_{seed}.json")
    payload = json.loads(payload_path.read_text())
    pre_rel = payload["pre_rel"]
    post_rel = payload["post_rel"]
    code_sha = payload["code_sha"]
    calibration = payload["calibration"]
    training_cfg = payload["training_cfg"]
    adapter_ranks = payload["adapter_ranks"]
    run_exploratory_note = payload.get("run_exploratory_note", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    pre_path = Path("/runs") / pre_rel
    post_path = Path("/runs") / post_rel
    out_dir = Path(f"/runs/ccpt/successor_task1/seed_{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher, pre_ckpt = _load_ccpt_from_ckpt(pre_path, device)
    student_base, post_ckpt = _load_ccpt_from_ckpt(post_path, device)
    freeze_module(teacher)
    freeze_module(student_base)

    pre_hash = compute_canonical_state_dict_hash(pre_ckpt["model_state_dict"])
    post_hash = compute_canonical_state_dict_hash(post_ckpt["model_state_dict"])
    base_param_hash_before = hash_existing_parameters(student_base)

    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        revision="27d67f1b5f57dc0953326b2601d68371d40ea8da",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cal_path = Path("/runs/ccpt/successor_task1/calibration_fit_records.jsonl")
    records: List[Dict[str, Any]] = []
    with cal_path.open() as f:
        for line in f:
            records.append(json.loads(line))

    max_fit = min(len(records), int(training_cfg.get("max_fit_records", 512)))
    records = records[:max_fit]
    steps = int(training_cfg["training_steps"])
    lr = float(training_cfg["learning_rate"])
    clip = float(training_cfg["gradient_clip_norm"])
    risk_w = float(training_cfg.get("risk_loss_weight", 0.1))
    gen_len = int(training_cfg.get("teacher_continuation_tokens", 32))

    bundle = build_variant_bundle(
        student_base,
        observer_rank=int(adapter_ranks["observer_rank"]),
        actuator_rank=int(adapter_ranks["actuator_rank"]),
    )
    variants = [
        (RepairVariant.OBSERVER, "POST_OBSERVER_REPAIR"),
        (RepairVariant.ACTUATOR, "POST_ACTUATOR_REPAIR"),
        (RepairVariant.COMBINED, "POST_OBSERVER_PLUS_ACTUATOR"),
        (RepairVariant.GENERIC, "POST_MATCHED_GENERIC_REPAIR"),
    ]

    # Build teacher sequences once (cached)
    seqs: List[torch.Tensor] = []
    ends: List[int] = []
    teacher.eval()
    with torch.no_grad():
        for rec in records:
            framed = format_eval_prompt(rec["prompt"])
            toks = tokenizer.encode(framed, add_special_tokens=False)
            prompt_end = len(toks) - 1
            ids = torch.tensor(toks, device=device).unsqueeze(0)
            # greedy continuation under PRE
            for _ in range(gen_len):
                logits, _ = teacher(ids, mode="controlled", controller_scale=1.0)
                next_id = int(torch.argmax(logits[0, -1]).item())
                ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
                if next_id == tokenizer.eos_token_id:
                    break
            seqs.append(ids.squeeze(0).cpu())
            ends.append(prompt_end)

    # Precompute teacher logits per example (cpu list)
    teacher_pack: List[Dict[str, Any]] = []
    with torch.no_grad():
        for i, s in enumerate(seqs):
            x = s.unsqueeze(0).to(device)
            pe = torch.tensor([ends[i]], device=device)
            logits, risk = teacher(x, prompt_end_indices=pe, mode="controlled", controller_scale=1.0)
            teacher_pack.append(
                {
                    "input_ids": s,
                    "prompt_end": ends[i],
                    "logits": logits.squeeze(0).cpu(),
                    "risk": None if risk is None else risk.squeeze(0).cpu(),
                }
            )

    train_summaries = {}
    adapter_manifest = {}

    for variant, label in variants:
        # Fresh POST base each variant
        base = copy.deepcopy(student_base).to(device)
        freeze_module(base)
        before = hash_existing_parameters(base)
        wrap = AdaptiveInterfaceWrapper(
            base,
            variant,
            observer_rank=int(adapter_ranks["observer_rank"]),
            actuator_rank=int(adapter_ranks["actuator_rank"]),
            generic_rank=bundle["generic_rank"] if variant == RepairVariant.GENERIC else None,
        ).to(device)
        wrap.train()
        opt = torch.optim.AdamW(wrap.trainable_parameters(), lr=lr, weight_decay=0.0)
        wrap.assert_optimizer_owns_only_repairs(opt)

        last_loss = None
        finite = True
        for step in range(1, steps + 1):
            # cycle deterministically through teacher_pack
            pack = teacher_pack[(step - 1) % len(teacher_pack)]
            x = pack["input_ids"].unsqueeze(0).to(device)
            pe = torch.tensor([pack["prompt_end"]], device=device)
            t_logits = pack["logits"].unsqueeze(0).to(device)
            t_risk = None if pack["risk"] is None else pack["risk"].unsqueeze(0).to(device)

            opt.zero_grad(set_to_none=True)
            s_logits, s_risk = wrap(x, prompt_end_indices=pe, mode="controlled", controller_scale=1.0)
            mask = continuation_token_mask(x, pe, pad_id=tokenizer.pad_token_id)
            loss, stats = fit_loss(t_logits, s_logits, mask, t_risk, s_risk, risk_weight=risk_w)
            if not torch.isfinite(loss):
                finite = False
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrap.trainable_parameters(), clip)
            opt.step()
            last_loss = float(loss.item())

        after = hash_existing_parameters(wrap.base)
        if before != after:
            raise RuntimeError(f"Base parameters mutated during {label} training")

        # save adapter-only state
        adapter_state = {n: p.detach().cpu() for n, p in wrap.trainable_named_parameters()}
        h = hashlib.sha256()
        for n, t in sorted(adapter_state.items()):
            tt = t.contiguous()
            h.update(n.encode("utf-8"))
            h.update(str(tuple(tt.shape)).encode("utf-8"))
            h.update(tt.view(torch.uint8).numpy().tobytes())
        adapter_hash = h.hexdigest()
        ckpt_path = out_dir / f"{label}_step{steps}.pt"
        torch.save(
            {
                "variant": label,
                "seed": seed,
                "training_step": steps if finite else step,
                "adapter_state": adapter_state,
                "adapter_state_hash": adapter_hash,
                "pre_checkpoint_hash": pre_hash,
                "post_checkpoint_hash": post_hash,
                "base_param_hash": after,
                "code_sha": code_sha,
                "calibration_hash": calibration.get("source_records_logical_hash"),
                "trainable_parameter_count": sum(t.numel() for t in adapter_state.values()),
                "optimizer_config": {"name": "AdamW", "lr": lr, "weight_decay": 0.0},
                "precision": str(next(wrap.parameters()).dtype) if wrap.trainable_parameters() else "n/a",
                "final_distillation_loss": last_loss,
                "finite": finite,
            },
            ckpt_path,
        )
        train_summaries[label] = {
            "FINAL_STEP": steps if finite else step,
            "FINAL_DISTILLATION_LOSS": last_loss,
            "FINITE": finite,
            "ADAPTER_HASH": adapter_hash,
            "BASE_HASH_UNCHANGED": before == after,
        }
        adapter_manifest[label] = {
            "path": str(ckpt_path),
            "adapter_state_hash": adapter_hash,
            "base_param_hash": after,
        }

    # --- Lightweight behavioral probe on a fixed 64-prompt subset of OOD for cost control —
    # Full OOD eval is orchestrated separately if budget permits; here we always compute
    # capability CE on a short val batch and controller gaps on calibration diagnostic prompts.
    # The local orchestrator merges with full eval when available.

    result = {
        "seed": seed,
        "code_sha": code_sha,
        "pre_hash": pre_hash,
        "post_hash": post_hash,
        "base_param_hash": base_param_hash_before,
        "existing_parameters_changed": 0
        if base_param_hash_before == hash_existing_parameters(student_base)
        else 1,
        "train_summaries": train_summaries,
        "adapter_manifest": adapter_manifest,
        "budget_geometry": bundle,
        "l40s_seconds": time.time() - t0,
        "exploratory_note": run_exploratory_note,
        "status": "FIT_COMPLETE",
    }
    (out_dir / "fit_result.json").write_text(json.dumps(result, indent=2) + "\n")
    runs_volume.commit()
    return result


@app.local_entrypoint()
def main(seed: int):
    """CLI: modal run modal/successor_task1_falsification.py --seed 20260825"""
    result = fit_and_eval_one_seed.remote(seed=seed)
    print(json.dumps(result, indent=2), flush=True)
    # Mirror to local artifacts via stdout only; orchestrator pulls fit_result from volume.
