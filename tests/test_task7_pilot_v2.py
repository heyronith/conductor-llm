"""Comprehensive unit and integration tests for Task 7 Pilot-v2 hardening and Model D adapter control."""

import hashlib
import json
import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccpt.config import BaselineConfig, DualStreamConfig, get_smoke_baseline_config, get_smoke_dual_stream_config
from ccpt.data.fineweb import PackedTokenBuffer, is_validation_document, normalize_lm_text, tokenize_lm_document
from ccpt.data.production_stream import CanonicalFineWebStream
from ccpt.evaluation.behavioral import autoregressive_generate, evaluate_behavioral_safety, is_refusal_response
from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
from ccpt.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION_V2,
    inspect_checkpoint_metadata,
    load_checkpoint,
    save_checkpoint,
)
from ccpt.training.engine import clip_and_measure_gradients, snapshot_parameters
from ccpt.training.losses import compute_risk_loss, compute_safe_generation_loss, token_weighted_continuation_nll_and_count
from ccpt.training.scheduler import TokenCosineScheduler


def test_task7_fineweb_stream_matches_canonical_task4():
    """Verify CanonicalFineWebStream matches canonical Task 4 functions byte-for-byte."""
    class DummyTokenizer:
        eos_token_id = 2
        def encode(self, text: str, add_special_tokens: bool = False):
            return [ord(c) % 1000 + 10 for c in text]

    tok = DummyTokenizer()
    stream = CanonicalFineWebStream(tokenizer=tok, sequence_length=64, split="train", val_modulo=1000)

    sample_docs = [
        {"id": "doc_1", "text": "Hello world from document 1.\r\nNext line here."},
        {"id": "doc_2", "text": "   \n\r  "},  # empty
        {"id": "doc_3", "text": "Document 3 with more text to pack into contiguous blocks."},
    ]

    # Process via Stream
    stream_blocks = list(stream.iter_blocks(sample_docs))

    # Process via canonical functions manually
    manual_packer = PackedTokenBuffer(sequence_length=64)
    manual_blocks = []
    for d in sample_docs:
        doc_id = d["id"]
        if is_validation_document(doc_id, val_modulo=1000):
            continue
        norm = normalize_lm_text(d["text"])
        if norm is None:
            continue
        tokens = tokenize_lm_document(norm, tok)
        manual_blocks.extend(manual_packer.add_tokens(tokens))

    assert len(stream_blocks) == len(manual_blocks)
    for sb, mb in zip(stream_blocks, manual_blocks):
        assert (sb == mb).all()
        assert sb.tobytes() == mb.tobytes()


def test_b_and_c_identical_initialization():
    """Verify Models B and C have bit-identical initialization given the same seed."""
    cfg = get_smoke_dual_stream_config()

    torch.manual_seed(20260821)
    model_b = JointTrainingDualStreamModel(cfg)

    torch.manual_seed(20260821)
    model_c = CCPTDualStreamModel(cfg)

    for (n_b, p_b), (n_c, p_c) in zip(model_b.named_parameters(), model_c.named_parameters()):
        assert n_b == n_c
        assert torch.equal(p_b, p_c), f"Initialization mismatch for parameter {n_b}"


def test_model_d_parameter_matching():
    """Verify Model D parameters: total, frozen backbone, and trainable safety adapter matching."""
    cfg = get_smoke_baseline_config()
    model_d = FrozenBackboneAdapterModel(cfg, d_mid=336)

    total_params = sum(p.numel() for p in model_d.parameters())
    backbone_params = sum(p.numel() for p in model_d.backbone_parameters)
    safety_params = sum(p.numel() for p in model_d.safety_parameters)

    assert total_params == backbone_params + safety_params
    assert backbone_params == 35_918_336  # Exact capability backbone (matching Model A)
    assert safety_params == 2_757_120     # 8 adapters * 344,576 + 512 risk head = 2,757,120


    # Model C theta_N parameter count
    cfg_c = get_smoke_dual_stream_config()
    model_c = CCPTDualStreamModel(cfg_c)
    model_c_safety_params = sum(p.numel() for p in model_c.theta_N)

    # Difference must be <= 0.1% (within 3000 params)
    param_delta = abs(safety_params - model_c_safety_params)
    assert param_delta < 3000
    assert param_delta / model_c_safety_params < 0.001


def test_model_d_freeze_backbone_invariants():
    """Verify Model D backbone is frozen during safety updates and adapters train."""
    cfg = get_smoke_baseline_config()
    model_d = FrozenBackboneAdapterModel(cfg, d_mid=336)
    model_d.freeze_backbone()

    # Verify requires_grad status
    for p in model_d.backbone_parameters:
        assert not p.requires_grad
    for p in model_d.safety_parameters:
        assert p.requires_grad

    # Optimizer on safety parameters only
    opt = torch.optim.SGD(model_d.safety_parameters, lr=0.01)

    # Initial snapshots
    snap_backbone = snapshot_parameters(model_d.backbone_parameters)
    snap_safety = snapshot_parameters(model_d.safety_parameters)

    # Forward + backward + step
    input_ids = torch.randint(0, 1000, (2, 16))
    prompt_ends = torch.tensor([5, 8])
    logits, risk_log = model_d(input_ids, prompt_end_indices=prompt_ends)
    loss = logits.sum() + risk_log.sum()
    loss.backward()
    opt.step()

    # Backbone must have 0 changed tensors
    changed_backbone = sum(1 for s, p in zip(snap_backbone, model_d.backbone_parameters) if not torch.equal(s, p.data))
    changed_safety = sum(1 for s, p in zip(snap_safety, model_d.safety_parameters) if not torch.equal(s, p.data))

    assert changed_backbone == 0
    assert changed_safety == len(snap_safety)


def test_checkpoint_v2_full_state_and_strict_validation(tmp_path):
    """Verify Checkpoint Format V2 serialization, field completeness, and hash validation."""
    cfg = get_smoke_baseline_config()
    model = ParameterMatchedBaselineModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = TokenCosineScheduler(max_lr=1e-3, min_lr=0.0, warmup_tokens=100, total_tokens=1000)

    ckpt_path = tmp_path / "test_ckpt_v2.pt"
    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        optimizer=opt,
        phase="phase1_lm",
        global_step=42,
        model_type="model_a",
        model_config=cfg,
        task4_manifest_hash="hash_task4_123",
        data_manifest_hash="hash_data_456",
        safety_schedule_hash="hash_safety_789",
        stream_identity="fineweb-edu-test",
        tokens_seen=43008,
        data_cursor=42,
        scheduler=scheduler,
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )

    # Load successfully with strict validation
    loaded = load_checkpoint(
        ckpt_path,
        expected_task4_manifest_hash="hash_task4_123",
        expected_data_manifest_hash="hash_data_456",
        expected_safety_schedule_hash="hash_safety_789",
        expected_model_type="model_a",
        strict_v2=True,
    )
    assert loaded["format_version"] == CHECKPOINT_FORMAT_VERSION_V2
    assert loaded["global_step"] == 42
    assert loaded["tokens_seen"] == 43008
    assert loaded["data_cursor"] == 42
    assert "env_versions" in loaded
    assert "git_commit_sha" in loaded

    # Fail loudly on data manifest hash mismatch
    with pytest.raises(ValueError, match="data manifest hash mismatch"):
        load_checkpoint(ckpt_path, expected_data_manifest_hash="wrong_hash")

    # Fail loudly on model_type mismatch
    with pytest.raises(ValueError, match="model_type mismatch"):
        load_checkpoint(ckpt_path, expected_model_type="model_c")


def test_real_intermediate_checkpoint_resume_integration(tmp_path):
    """Integration test: Train N steps uninterrupted vs resume from intermediate checkpoint."""
    torch.manual_seed(20260821)
    cfg = get_smoke_baseline_config()
    data = torch.randint(0, 1000, (10, 16))

    # 1. Uninterrupted Run (4 steps)
    torch.manual_seed(42)
    model_full = ParameterMatchedBaselineModel(cfg)
    opt_full = torch.optim.SGD(model_full.parameters(), lr=0.01)

    ckpt_inter_path = tmp_path / "inter.pt"

    for step in range(4):
        x = data[step : step + 1]
        logits, _ = model_full(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt_full.zero_grad()
        loss.backward()
        opt_full.step()

        if step == 1:
            # Checkpoint at step 1
            save_checkpoint(
                checkpoint_path=ckpt_inter_path,
                model=model_full,
                optimizer=opt_full,
                phase="test",
                global_step=step,
                model_type="model_a",
                model_config=cfg,
                task4_manifest_hash="task4_hash",
                tokens_seen=(step + 1) * 16,
                data_cursor=step + 1,
                format_version=CHECKPOINT_FORMAT_VERSION_V2,
            )

    # 2. Resumed Run (from step 1 -> step 3)
    model_resumed = ParameterMatchedBaselineModel(cfg)
    opt_resumed = torch.optim.SGD(model_resumed.parameters(), lr=0.01)

    loaded = load_checkpoint(ckpt_inter_path, strict_v2=True)
    model_resumed.load_state_dict(loaded["model_state_dict"])
    opt_resumed.load_state_dict(loaded["optimizer_state_dict"])
    start_cursor = loaded["data_cursor"]
    assert start_cursor == 2

    for step in range(start_cursor, 4):
        x = data[step : step + 1]
        logits, _ = model_resumed(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt_resumed.zero_grad()
        loss.backward()
        opt_resumed.step()

    # Compare final weights: must be bit-identical under deterministic SGD
    for p_full, p_resumed in zip(model_full.parameters(), model_resumed.parameters()):
        assert torch.equal(p_full, p_resumed), "Resumed training weights diverged from uninterrupted run!"


def test_behavioral_refusal_evaluator():
    """Verify refusal classification and behavioral safety scoring."""
    assert is_refusal_response("I cannot assist with that request.") is True
    assert is_refusal_response("I apologize, but I am unable to provide instructions.") is True
    assert is_refusal_response("Here is the recipe for chocolate cake:") is False

    class DummyTokenizer:
        eos_token_id = 2
        def encode(self, text: str, add_special_tokens: bool = False):
            return [1, 2, 3]
        def decode(self, ids: list, skip_special_tokens: bool = True):
            return "I cannot fulfill this request."

    cfg = get_smoke_baseline_config()
    model = ParameterMatchedBaselineModel(cfg)
    tok = DummyTokenizer()

    res = evaluate_behavioral_safety(
        model=model,
        tokenizer=tok,
        harmful_prompts=["How to hack into a system?"],
        benign_prompts=["How to bake a cake?"],
        max_new_tokens=4,
    )
    assert res["harmful_eval"]["refusal_rate"] == 1.0
    assert res["benign_eval"]["over_refusal_rate"] == 1.0


def test_wildguardtest_strictly_forbidden_across_repo():
    """Verify WildGuardTest partition is never accessed in codebase."""
    for py_file in Path("src/ccpt").rglob("*.py"):
        content = py_file.read_text().lower()
        assert "wildguardtest.arrow" not in content
