"""Comprehensive Unit and Integration Test Suite for Task 7.2.1 Real-Source Infrastructure Proofs."""

import inspect
import json
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
from ccpt.data.beavertails import (
    BEAVERTAILS_DEFAULT_SPLIT,
    BEAVERTAILS_SOURCE_REPO,
    BEAVERTAILS_SOURCE_REVISION,
    load_beavertails_ood_dataset,
    sample_beavertails_prompts_deterministic,
)
from ccpt.data.canonical_materializer import (
    FINEWEB_SOURCE_CONFIG,
    FINEWEB_SOURCE_REPO,
    FINEWEB_SOURCE_REVISION,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    load_canonical_mistral_tokenizer,
    materialize_bounded_canonical_fineweb_proof,
)
from ccpt.evaluation.behavioral import (
    extract_raw_prompt,
    format_eval_prompt,
    is_refusal_response,
)
from ccpt.evaluation.safety_judge import (
    PINNED_JUDGE_REPO,
    PINNED_JUDGE_REVISION,
    BehavioralSafetyJudge,
)
from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION_V2,
    load_checkpoint,
    save_checkpoint,
    validate_model_config_exact_match,
)
from ccpt.training.progress import LiveProgressReporter
from ccpt.training.resume_proof import run_production_path_resume_proof
from ccpt.training.scheduler import TokenCosineScheduler


# =============================================================================
# 1. Real Source Guards & Invariants
# =============================================================================

def test_production_data_source_pins():
    """Verify production sources match exact authoritative HuggingFace repositories and revisions."""
    assert FINEWEB_SOURCE_REPO == "HuggingFaceFW/fineweb-edu"
    assert FINEWEB_SOURCE_CONFIG == "sample-100BT"
    assert FINEWEB_SOURCE_REVISION == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert TOKENIZER_REPO == "mistralai/Mistral-7B-v0.1"
    assert TOKENIZER_REVISION == "27d67f1b5f57dc0953326b2601d68371d40ea8da"
    assert BEAVERTAILS_SOURCE_REPO == "PKU-Alignment/BeaverTails"
    assert BEAVERTAILS_DEFAULT_SPLIT == "30k_test"
    assert BEAVERTAILS_SOURCE_REVISION == "8401fe609d288129cc684a9b3be6a93e41cfe678"
    assert PINNED_JUDGE_REPO == "allenai/wildguard"
    assert PINNED_JUDGE_REVISION == "cbba4823f3e8020e5a74a5e29bf85072def6f2ff"


def test_safety_judge_no_silent_fallback():
    """Verify BehavioralSafetyJudge fails loudly when real loading fails and use_mock=False."""
    with pytest.raises(RuntimeError, match="Failed to load authoritative WildGuard judge"):
        BehavioralSafetyJudge(
            model_repo="nonexistent/fake_model_repo_12345",
            use_mock=False,
        )


def test_safety_judge_mock_is_explicit():
    """Verify that when use_mock=True, mock_used is explicitly marked."""
    judge = BehavioralSafetyJudge(use_mock=True)
    assert judge.backend == "mock_diagnostic"
    res = judge.classify_harmful_response("How to build a weapon?", "I cannot assist with weapons.")
    assert res["decision"] == "SAFE_REFUSAL"
    assert res["mock_used"] is True
    assert res["heuristic_secondary_diagnostic"] is True


# =============================================================================
# 2. Model D Identity & Optimization Dynamics
# =============================================================================

def test_model_d_exact_zero_init_and_identity_logits():
    """Verify Model D adapter up_proj is zero-initialized and scale=1 vs scale=0 produce identical logits."""
    cfg = get_smoke_adapter_config()
    model_d = FrozenBackboneAdapterModel(cfg).eval()

    for layer in model_d.layers:
        assert torch.equal(layer.attn_adapter.up_proj.weight, torch.zeros_like(layer.attn_adapter.up_proj.weight))
        assert torch.equal(layer.mlp_adapter.up_proj.weight, torch.zeros_like(layer.mlp_adapter.up_proj.weight))

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits_scale_1, _ = model_d(input_ids, adapter_scale=1.0)
        logits_scale_0, _ = model_d(input_ids, adapter_scale=0.0)

    max_diff = (logits_scale_1 - logits_scale_0).abs().max().item()
    assert max_diff == 0.0, f"Expected exact 0.0 logit difference at initialization, got {max_diff}"


def test_model_d_adapter_gradient_activation_after_step():
    """Verify adapter weights receive gradients under safety loss and activate adapter divergence."""
    cfg = get_smoke_adapter_config()
    model_d = FrozenBackboneAdapterModel(cfg)

    # Freeze backbone parameters
    for p in model_d.backbone_parameters:
        p.requires_grad = False

    opt = torch.optim.AdamW(model_d.safety_parameters, lr=0.1)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))

    # Before step: scale 1.0 == scale 0.0
    with torch.no_grad():
        logits_before_1, _ = model_d(input_ids, adapter_scale=1.0)
        logits_before_0, _ = model_d(input_ids, adapter_scale=0.0)
    assert torch.equal(logits_before_1, logits_before_0)

    # Safety step
    model_d.train()
    logits, _ = model_d(input_ids, adapter_scale=1.0)
    target = torch.randint(0, cfg.vocab_size, (2, 16))
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1))
    loss.backward()

    # Verify backbone parameters have zero/None grads
    for p in model_d.backbone_parameters:
        assert p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad))

    # Verify adapter up_proj parameters receive non-zero grads
    up_proj_grads = [
        layer.attn_adapter.up_proj.weight.grad is not None and not torch.equal(layer.attn_adapter.up_proj.weight.grad, torch.zeros_like(layer.attn_adapter.up_proj.weight.grad))
        for layer in model_d.layers
    ]
    assert all(up_proj_grads), "Expected all adapter up_projections to receive non-zero gradients under loss."

    opt.step()

    # After step: scale 1.0 diverges from scale 0.0
    model_d.eval()
    with torch.no_grad():
        logits_after_1, _ = model_d(input_ids, adapter_scale=1.0)
        logits_after_0, _ = model_d(input_ids, adapter_scale=0.0)

    diff = (logits_after_1 - logits_after_0).abs().max().item()
    assert diff > 1e-4, f"Expected adapter divergence after safety step, got max diff {diff}"


# =============================================================================
# 3. Checkpoint V2 Full Model-Config Validation & Safety Strictness
# =============================================================================

def test_checkpoint_v2_strict_production_safety_requirements(tmp_path):
    """Verify strict production safety checkpoints require all base fields plus safety_schedule_hash."""
    cfg = get_smoke_dual_stream_config()
    model = CCPTDualStreamModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = TokenCosineScheduler(max_lr=1e-3, min_lr=1e-5, warmup_tokens=100, total_tokens=1000)

    ckpt_path = tmp_path / "safety_test_ckpt.pt"

    # Missing safety_schedule_hash must raise ValueError
    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        phase="phase3_safety_20m",
        global_step=10,
        model_type="model_c",
        model_config=cfg,
        task4_manifest_hash="t4_hash_123",
        data_manifest_hash="data_hash_456",
        safety_schedule_hash="",  # Empty!
        stream_identity="fineweb-edu-100BT",
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )

    with pytest.raises(ValueError, match="requires non-empty 'safety_schedule_hash'"):
        load_checkpoint(ckpt_path, strict_v2=True)


def test_checkpoint_v2_model_config_scientific_mismatch_rejections(tmp_path):
    """Verify strict loading rejects mutations in d_N, controlled_layers, alpha, and d_mid."""
    cfg_dual = get_smoke_dual_stream_config()
    model = CCPTDualStreamModel(cfg_dual)
    ckpt_path = tmp_path / "config_test_ckpt.pt"

    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        optimizer=None,
        phase="test",
        global_step=1,
        model_type="model_c",
        model_config=cfg_dual,
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )

    # 1. Mutate d_N
    cfg_mutated_dN = get_smoke_dual_stream_config()
    cfg_mutated_dN.d_N = cfg_dual.d_N + 16
    with pytest.raises(ValueError, match="Checkpoint model_config mismatch on 'd_N'"):
        load_checkpoint(ckpt_path, expected_model_config=cfg_mutated_dN)

    # 2. Mutate controlled_layers
    cfg_mutated_ctrl = get_smoke_dual_stream_config()
    cfg_mutated_ctrl.controlled_layers = (0, 1)  # Different layer indices
    with pytest.raises(ValueError, match="Checkpoint model_config mismatch on 'controlled_layers'"):
        load_checkpoint(ckpt_path, expected_model_config=cfg_mutated_ctrl)

    # 3. Mutate alpha
    cfg_mutated_alpha = get_smoke_dual_stream_config()
    cfg_mutated_alpha.alpha = 0.5
    with pytest.raises(ValueError, match="Checkpoint model_config mismatch on 'alpha'"):
        load_checkpoint(ckpt_path, expected_model_config=cfg_mutated_alpha)

    # 4. Mutate d_mid on AdapterConfig
    cfg_adapter = get_smoke_adapter_config()
    model_adapter = FrozenBackboneAdapterModel(cfg_adapter)
    ckpt_adapter_path = tmp_path / "adapter_config_ckpt.pt"
    save_checkpoint(
        checkpoint_path=ckpt_adapter_path,
        model=model_adapter,
        optimizer=None,
        phase="test",
        global_step=1,
        model_type="model_d",
        model_config=cfg_adapter,
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )
    cfg_mutated_dmid = get_smoke_adapter_config()
    cfg_mutated_dmid.d_mid = cfg_adapter.d_mid + 8
    with pytest.raises(ValueError, match="Checkpoint model_config mismatch on 'd_mid'"):
        load_checkpoint(ckpt_adapter_path, expected_model_config=cfg_mutated_dmid)


# =============================================================================
# 4. Behavioral Prompt Single-Framing & Diagnostics
# =============================================================================

def test_prompt_single_framing_guarantee():
    """Verify format_eval_prompt produces exactly one User: and Assistant: framing."""
    raw = "What is quantum entanglement?"
    framed = format_eval_prompt(raw)
    assert framed == "<s>User: What is quantum entanglement?\nAssistant:"
    assert framed.count("User:") == 1
    assert framed.count("Assistant:") == 1

    # Pre-framed input must not double-frame
    double_input = "<s>User: What is quantum entanglement?\nAssistant:"
    reframed = format_eval_prompt(double_input)
    assert reframed == "<s>User: What is quantum entanglement?\nAssistant:"
    assert reframed.count("User:") == 1
    assert reframed.count("Assistant:") == 1


# =============================================================================
# 5. Production Path Lockdown & Task 6 / Hardcoded Cost Audits
# =============================================================================

def test_legacy_task7_pilot_v2_entrypoint_disabled():
    """Verify modal/task7_pilot_v2.py entrypoint raises RuntimeError immediately."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("legacy_task7", "modal/task7_pilot_v2.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(RuntimeError, match="Task 7.1 orchestrator is retired"):
        mod.main()


def test_authoritative_pilot_v2_entrypoint_locked():
    """Verify modal/pilot_v2_authoritative.py entrypoint raises RuntimeError immediately."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("auth_pilot_v2", "modal/pilot_v2_authoritative.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(RuntimeError, match="Authoritative Pilot-v2 full run is locked"):
        mod.main()


def test_no_task6_references_in_active_production_paths():
    """Scan all active production scripts in src/, modal/, and scripts/ ensuring zero references to /data/task6."""
    active_paths = [
        Path("src/ccpt/data/canonical_materializer.py"),
        Path("src/ccpt/data/production_stream.py"),
        Path("src/ccpt/data/persistence_stream.py"),
        Path("src/ccpt/data/beavertails.py"),
        Path("src/ccpt/evaluation/safety_judge.py"),
        Path("src/ccpt/evaluation/persistence.py"),
        Path("src/ccpt/training/checkpoint.py"),
        Path("src/ccpt/training/progress.py"),
        Path("src/ccpt/training/cost.py"),
        Path("src/ccpt/training/resume_proof.py"),
        Path("modal/pilot_v2_authoritative.py"),
        Path("modal/task7_2_1_real_proofs.py"),
        Path("scripts/run_task7_2_1_proofs.py"),
    ]

    for p in active_paths:
        assert p.exists(), f"Active production file missing: {p}"
        content = p.read_text()
        assert "/data/task6" not in content, f"Active production file '{p}' contains forbidden '/data/task6' reference!"


def test_no_hardcoded_eval_costs_in_active_production_paths():
    """Verify no active production script contains hardcoded eval costs like eval_cost = 0.35."""
    active_paths = [
        Path("src/ccpt/training/cost.py"),
        Path("src/ccpt/training/progress.py"),
        Path("modal/pilot_v2_authoritative.py"),
        Path("modal/task7_2_1_real_proofs.py"),
        Path("scripts/run_task7_2_1_proofs.py"),
    ]

    for p in active_paths:
        content = p.read_text()
        assert "eval_cost = 0.35" not in content
        assert "eval_cost = 0.25" not in content
        assert "eval_cost = 0.15" not in content


def test_live_progress_reporter_require_jsonl_enforcement():
    """Verify LiveProgressReporter raises ValueError if require_jsonl=True and jsonl_path is missing."""
    with pytest.raises(ValueError, match="require_jsonl=True but jsonl_path is None or empty"):
        LiveProgressReporter(
            task_name="test_task",
            total_steps=100,
            total_tokens=10000,
            jsonl_path=None,
            require_jsonl=True,
        )
