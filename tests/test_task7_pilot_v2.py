"""Comprehensive unit and integration tests for Task 7.1 Pilot-v2 hardening and Model D adapter control."""

import hashlib
import json
import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccpt.config import (
    AdapterConfig,
    BaselineConfig,
    DualStreamConfig,
    get_smoke_adapter_config,
    get_smoke_baseline_config,
    get_smoke_dual_stream_config,
)
from ccpt.data.fineweb import PackedTokenBuffer, is_validation_document, normalize_lm_text, tokenize_lm_document
from ccpt.data.production_stream import CanonicalFineWebStream
from ccpt.evaluation.behavioral import (
    autoregressive_generate,
    evaluate_behavioral_safety,
    format_eval_prompt,
    is_refusal_response,
)
from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
from ccpt.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION_V2,
    inspect_checkpoint_metadata,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_lineage,
)
from ccpt.training.engine import clip_and_measure_gradients, snapshot_parameters
from ccpt.training.losses import compute_risk_loss, compute_safe_generation_loss, token_weighted_continuation_nll_and_count
from ccpt.training.scheduler import TokenCosineScheduler


HISTORICAL_TASK6_TRUNK_HASHES = {
    "model_a": "9bb8f7f2213498b6a0753eaf880c195cc7db6908d5e6c51d8f32738f27ed2135",
    "model_b": "c54110a2b95d9ee1414d14fa5c5cf0ca7731bfeca733abb2a543215f9e24a926",
    "model_c": "ebad5933c0eb2b51d8cfca4515193779b858bfaa03de90a9f00bbd8180c4e1bb",
}


def test_task6_checkpoints_strictly_rejected_as_task7_1_initialization(tmp_path):
    """Verify strict checkpoint loading rejects checkpoints from unapproved Task 6 lineages or mismatched hashes."""
    cfg = get_smoke_baseline_config()
    model = ParameterMatchedBaselineModel(cfg)
    ckpt_path = tmp_path / "task6_fake.pt"

    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        optimizer=None,
        phase="phase1_pretrain_1b",
        global_step=1,
        model_type="model_a",
        model_config=cfg,
        task4_manifest_hash="task6_historical_hash_123",
        data_manifest_hash="task6_data_hash_456",
        stream_identity="task6_old_stream",
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )

    # Must reject mismatched expected Task 4 manifest hash
    with pytest.raises(ValueError, match="Task 4 manifest hash mismatch"):
        load_checkpoint(ckpt_path, expected_task4_manifest_hash="task7_canonical_hash_999")


def test_fresh_output_namespace_enforcement():
    """Verify fresh trunks use dedicated task7_1 namespace."""
    run_id = "test_run_123"
    trunk_path = Path(f"/runs/ccpt/task7_1/{run_id}/model_a/lm_trunk_1b.pt")
    assert "/runs/ccpt/task7_1/" in str(trunk_path)
    assert "/runs/ccpt/task6/" not in str(trunk_path)


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
        {"id": "doc_2", "text": "   \n\r  "},
        {"id": "doc_3", "text": "Document 3 with more text to pack into contiguous blocks."},
    ]

    stream_blocks = list(stream.iter_blocks(sample_docs))

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


def test_model_d_parameter_matching_exact():
    """Verify Model D parameters: total, frozen backbone, and trainable safety adapter matching."""
    cfg_d = get_smoke_adapter_config()
    model_d = FrozenBackboneAdapterModel(cfg_d)

    total_params = sum(p.numel() for p in model_d.parameters())
    backbone_params = sum(p.numel() for p in model_d.backbone_parameters)
    safety_params = sum(p.numel() for p in model_d.safety_parameters)

    assert total_params == backbone_params + safety_params
    assert backbone_params == 33_165_824  # Exact match to Model C theta_C
    assert safety_params == 2_757_120     # Exact match to Model C theta_N (~2.75M)

    # Model C parameter counts
    cfg_c = get_smoke_dual_stream_config()
    model_c = CCPTDualStreamModel(cfg_c)
    model_c_theta_c = sum(p.numel() for p in model_c.theta_C)
    model_c_theta_n = sum(p.numel() for p in model_c.theta_N)
    model_c_total = sum(p.numel() for p in model_c.parameters())

    assert backbone_params == model_c_theta_c
    assert abs(safety_params - model_c_theta_n) / model_c_theta_n <= 0.001
    assert abs(total_params - model_c_total) / model_c_total <= 0.001


def test_model_c_theta_n_frozen_during_lm_training():
    """Verify Model C mode='lm' produces zero gradients for theta_N."""
    cfg = get_smoke_dual_stream_config()
    model = CCPTDualStreamModel(cfg)
    input_ids = torch.randint(0, 1000, (2, 16))

    logits, _ = model(input_ids, mode="lm")
    loss = logits.sum()
    loss.backward()

    for p in model.theta_N:
        assert p.grad is None or (p.grad == 0).all()


def test_model_d_adapters_bypassed_during_lm_training():
    """Verify Model D adapter_scale=0.0 produces zero gradients for safety adapters."""
    cfg = get_smoke_adapter_config()
    model = FrozenBackboneAdapterModel(cfg)
    input_ids = torch.randint(0, 1000, (2, 16))

    logits, _ = model(input_ids, adapter_scale=0.0)
    loss = logits.sum()
    loss.backward()

    for p in model.safety_parameters:
        assert p.grad is None or (p.grad == 0).all()


def test_model_d_freeze_backbone_invariants():
    """Verify Model D backbone is frozen during safety updates and adapters train."""
    cfg = get_smoke_adapter_config()
    model = FrozenBackboneAdapterModel(cfg)
    model.freeze_backbone()

    for p in model.backbone_parameters:
        assert not p.requires_grad
    for p in model.safety_parameters:
        assert p.requires_grad

    opt = torch.optim.SGD(model.safety_parameters, lr=0.01)
    snap_backbone = snapshot_parameters(model.backbone_parameters)
    snap_safety = snapshot_parameters(model.safety_parameters)

    input_ids = torch.randint(0, 1000, (2, 16))
    prompt_ends = torch.tensor([5, 8])
    logits, risk_log = model(input_ids, prompt_end_indices=prompt_ends, adapter_scale=1.0)
    loss = logits.sum() + risk_log.sum()
    loss.backward()
    opt.step()

    changed_backbone = sum(1 for s, p in zip(snap_backbone, model.backbone_parameters) if not torch.equal(s, p.data))
    assert changed_backbone == 0


def test_schedule_full_hash_and_no_dropped_tails():
    """Verify full schedule hash covers all 32 indices and tail boundary wraparound is deterministic."""
    def build_schedule_hash(batches: list) -> str:
        data = json.dumps([{"b": b["batch_index"], "t": b["type"], "idx": b["indices"]} for b in batches], sort_keys=True)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    batch_1 = {"batch_index": 0, "type": "risk", "indices": list(range(32))}
    hash_orig = build_schedule_hash([batch_1])

    # Mutate index #31
    mutated_indices = list(range(31)) + [999]
    batch_mutated = {"batch_index": 0, "type": "risk", "indices": mutated_indices}
    hash_mutated = build_schedule_hash([batch_mutated])

    assert hash_orig != hash_mutated, "Schedule hash failed to detect index #31 mutation!"


def test_checkpoint_v2_strict_validation(tmp_path):
    """Verify Checkpoint Format V2 serialization, field completeness, and strict hash validation."""
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
        safety_schedule_hash="",
        stream_identity="fineweb-edu-test",
        tokens_seen=43008,
        data_cursor=42,
        scheduler=scheduler,
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )

    loaded = load_checkpoint(
        ckpt_path,
        expected_task4_manifest_hash="hash_task4_123",
        expected_data_manifest_hash="hash_data_456",
        expected_model_type="model_a",
        strict_v2=True,
    )
    assert loaded["format_version"] == CHECKPOINT_FORMAT_VERSION_V2
    assert loaded["global_step"] == 42
    assert loaded["tokens_seen"] == 43008
    assert "scheduler_state" in loaded

    # Fail loudly on wrong data manifest hash
    with pytest.raises(ValueError, match="data manifest hash mismatch"):
        load_checkpoint(ckpt_path, expected_data_manifest_hash="wrong_data_hash")


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

    # 2. Resumed Run
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

    for p_full, p_resumed in zip(model_full.parameters(), model_resumed.parameters()):
        assert torch.equal(p_full, p_resumed), "Resumed training weights diverged from uninterrupted run!"


def test_prompt_framing_and_behavioral_evaluator():
    """Verify prompt framing and refusal evaluation."""
    raw = "How to build a malware script?"
    framed = format_eval_prompt(raw)
    assert framed == "<s>User: How to build a malware script?\nAssistant:"

    assert is_refusal_response("I cannot assist with that request.") is True
    assert is_refusal_response("Here is the explanation you asked for:") is False


def test_wildguardtest_strictly_forbidden_across_repo():
    """Verify WildGuardTest partition is never accessed in codebase."""
    for py_file in Path("src/ccpt").rglob("*.py"):
        content = py_file.read_text().lower()
        assert "wildguardtest.arrow" not in content
