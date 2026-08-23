"""Task 7.4: Comprehensive Replication Hardening & Prelaunch Verification Tests.

Covers:
1. pad_and_collate_gen_records API fix (is_refusals boolean tensor return).
2. Canonical WildGuard artifact resolution & provenance validation.
3. Safety schedule full hash sensitivity to epoch_indices mutations.
4. Checkpoint V3 schema and mandatory git commit SHA verification.
5. Token-weighted safe-generation loss on variable-length padded sequences.
6. Tri-state WildGuard classification, nonzero-NA bounds, and Wilson score intervals.
7. Architectural parameter count assertions for Models A, B, C, D.
8. Model B / Model C identical initialization from identical seeds.
9. Deterministic state dict cryptographic hashing across bfloat16/float32.
10. Evaluation prompt formatting and generation config reproducibility.
"""

import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from ccpt.config import (
    get_smoke_baseline_config,
    get_smoke_dual_stream_config,
    get_smoke_adapter_config,
)
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.data.collators import (
    DataCollatorForSafeGenerationTraining,
    pad_and_collate_gen_records,
)
from ccpt.data.wildguard import (
    CANONICAL_TASK4_MANIFEST_HASH,
    CANONICAL_WILDGUARD_COUNTS,
    RiskRecord,
    SafeGenerationRecord,
    verify_safety_records_provenance,
)
from ccpt.training.safety_schedule import (
    generate_authoritative_safety_schedule,
    compute_full_schedule_audit_hash,
)
from ccpt.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION_V2,
    CHECKPOINT_FORMAT_VERSION_V3,
    save_checkpoint,
    load_checkpoint,
    get_git_commit_sha,
)
from ccpt.training.losses import compute_safe_generation_loss
from ccpt.evaluation.behavioral import (
    wilson_score_interval,
    extract_raw_prompt,
    format_eval_prompt,
)
from ccpt.evaluation.forensics import (
    get_ccpt_named_partitions,
    get_adapter_named_partitions,
    compute_canonical_state_dict_hash,
)


def test_pad_and_collate_gen_records_returns_is_refusal_tensor():
    """Verifies pad_and_collate_gen_records returns boolean is_refusals tensor in slot 4."""
    recs = [
        SafeGenerationRecord(
            example_id="gen_001",
            prompt_group_key="grp_1",
            input_ids=[1, 10, 20, 30, 40],
            prompt_end_index=2,
            risk_label=1,
            is_refusal=True,
            is_adversarial=False,
            subcategory="harmful",
            split="train",
        ),
        SafeGenerationRecord(
            example_id="gen_002",
            prompt_group_key="grp_2",
            input_ids=[1, 10, 50, 60, 70, 80],
            prompt_end_index=1,
            risk_label=0,
            is_refusal=False,
            is_adversarial=False,
            subcategory="benign",
            split="train",
        ),
    ]
    input_ids, prompt_ends, risk_labels, is_refusals, attn_mask = pad_and_collate_gen_records(recs, pad_token_id=2)

    assert isinstance(is_refusals, torch.Tensor)
    assert is_refusals.dtype == torch.bool
    assert is_refusals.shape == (2,)
    assert is_refusals[0].item() is True
    assert is_refusals[1].item() is False

    # Also check dict collation directly
    collator = DataCollatorForSafeGenerationTraining(pad_token_id=2)
    res = collator(recs)
    assert "is_refusals" in res
    assert torch.equal(res["is_refusals"], torch.tensor([True, False]))


def test_safety_schedule_full_hash_sensitivity_to_epoch_indices():
    """Verifies that mutating epoch_indices alters full_schedule_audit_hash."""
    risk_recs = [
        RiskRecord(
            example_id=f"r_{i}",
            prompt_group_key=f"g_{i}",
            input_ids=[1, 10, 20, 30],
            prompt_end_index=2,
            risk_label=i % 2,
            is_adversarial=False,
            subcategory="none",
            split="train",
        )
        for i in range(100)
    ]
    gen_recs = [
        SafeGenerationRecord(
            example_id=f"g_{i}",
            prompt_group_key=f"g_{i}",
            input_ids=[1, 10, 20, 30],
            prompt_end_index=2,
            risk_label=i % 2,
            is_refusal=True,
            is_adversarial=False,
            subcategory="none",
            split="train",
        )
        for i in range(50)
    ]

    sched = generate_authoritative_safety_schedule(
        risk_records=risk_recs,
        gen_records=gen_recs,
        target_safety_tokens=2_000,
        batch_size=16,
        seed=20260821,
    )

    orig_hash = compute_full_schedule_audit_hash(sched)
    assert sched["full_schedule_audit_hash"] == orig_hash

    # Mutate an epoch index in a batch
    sched_mutated = copy.deepcopy(sched)
    sched_mutated["batches"][0]["epoch_indices"][0] += 1
    mutated_hash = compute_full_schedule_audit_hash(sched_mutated)

    assert orig_hash != mutated_hash, "Mutating epoch_indices MUST change full_schedule_audit_hash!"


def test_canonical_wildguard_provenance():
    """Verifies canonical WildGuard constants and provenance validation logic."""
    assert CANONICAL_TASK4_MANIFEST_HASH == "2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097"
    assert CANONICAL_WILDGUARD_COUNTS == {
        "risk_train": 45492,
        "risk_val": 2344,
        "gen_train": 18015,
        "gen_val": 928,
    }

    # Test provenance validator fails on duplicate ID
    risk_train = [RiskRecord(f"r_{i}", "g", [1, 2, 3], 1, 0, False, "none", "train") for i in range(45492)]
    risk_val = [RiskRecord(f"rv_{i}", "g", [1, 2, 3], 1, 0, False, "none", "val") for i in range(2344)]
    gen_train = [SafeGenerationRecord(f"g_{i}", "g", [1, 2, 3], 1, 0, False, False, "none", "train") for i in range(18015)]
    gen_val = [SafeGenerationRecord(f"gv_{i}", "g", [1, 2, 3], 1, 0, False, False, "none", "val") for i in range(928)]

    # Provenance check should pass
    prov_res = verify_safety_records_provenance(risk_train, risk_val, gen_train, gen_val)
    assert prov_res["all_records_valid"] is True
    assert prov_res["total_records_verified"] == 45492 + 2344 + 18015 + 928

    # Duplicate in risk dataset should fail
    risk_train_dup = copy.deepcopy(risk_train)
    risk_train_dup[1] = RiskRecord("r_0", "g", [1, 2, 3], 1, 0, False, "none", "train")
    with pytest.raises(ValueError, match="Duplicate example_id"):
        verify_safety_records_provenance(risk_train_dup, risk_val, gen_train, gen_val)


def test_checkpoint_v3_mandatory_git_sha(tmp_path):
    """Verifies checkpoint saving requires valid git SHA and rejects unknown when mandatory."""
    cfg = get_smoke_baseline_config()
    model = ParameterMatchedBaselineModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt_path = tmp_path / "test_ckpt_v3.pt"

    # Should raise if require_exact_git_sha=True and sha is unknown/missing
    with pytest.raises(RuntimeError, match="Exact git commit SHA is mandatory"):
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=opt,
            phase="phase1_lm",
            global_step=10,
            model_type="model_a",
            model_config=cfg,
            git_commit_sha="unknown",
            require_exact_git_sha=True,
        )

    from ccpt.training.scheduler import TokenCosineScheduler
    sched = TokenCosineScheduler(max_lr=1e-4, warmup_tokens=1000, total_tokens=100000)

    # Save with valid SHA
    dummy_sha = "0123456789abcdef0123456789abcdef01234567"
    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        phase="phase1_lm",
        global_step=10,
        model_type="model_a",
        model_config=cfg,
        git_commit_sha=dummy_sha,
        task4_manifest_hash="dummy_task4_hash",
        data_manifest_hash="dummy_data_hash",
        stream_identity="fineweb-edu-100BT",
    )

    # Load with strict V3
    loaded = load_checkpoint(
        ckpt_path,
        strict_v3=True,
        expected_git_commit_sha=dummy_sha,
    )
    assert loaded["git_commit_sha"] == dummy_sha
    assert "creation_timestamp" in loaded

    # Mismatched SHA load should fail
    with pytest.raises(ValueError, match="Checkpoint git commit SHA mismatch"):
        load_checkpoint(
            ckpt_path,
            strict_v3=True,
            expected_git_commit_sha="mismatched_sha",
        )


def test_token_weighted_safe_gen_loss_padded_regression():
    """Verifies compute_safe_generation_loss excludes padding tokens accurately."""
    vocab_size = 32000
    batch_size = 2
    seq_len = 10

    # Create dummy logits
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Sequence 0: length 6, prompt_end 2 -> 3 continuation tokens (indices 3, 4, 5)
    # Sequence 1: length 10, prompt_end 4 -> 5 continuation tokens (indices 5, 6, 7, 8, 9)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    prompt_ends = torch.tensor([2, 4], dtype=torch.long)
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=torch.long)

    loss = compute_safe_generation_loss(logits, input_ids, prompt_ends, attention_mask=attention_mask)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    # Calculate token-by-token expected CE manually
    ce_fn = nn.CrossEntropyLoss(reduction="sum")
    
    # Seq 0: target tokens are input_ids[0, 3:6], predicting logits are logits[0, 2:5]
    logits_0 = logits[0, 2:5]
    targets_0 = input_ids[0, 3:6]
    nll_0 = ce_fn(logits_0, targets_0).item()
    tokens_0 = 3

    # Seq 1: target tokens are input_ids[1, 5:10], predicting logits are logits[1, 4:9]
    logits_1 = logits[1, 4:9]
    targets_1 = input_ids[1, 5:10]
    nll_1 = ce_fn(logits_1, targets_1).item()
    tokens_1 = 5

    expected_loss = (nll_0 + nll_1) / (tokens_0 + tokens_1)
    assert abs(loss.item() - expected_loss) < 1e-5


def test_wilson_score_interval_bounds():
    """Verifies Wilson score confidence interval computation and edge cases."""
    low, high = wilson_score_interval(50, 100, confidence=0.95)
    assert 0.39 < low < 0.41
    assert 0.59 < high < 0.61

    # 0 successes
    low_0, high_0 = wilson_score_interval(0, 100, confidence=0.95)
    assert low_0 == 0.0
    assert 0.0 < high_0 < 0.05

    # 100% successes
    low_100, high_100 = wilson_score_interval(100, 100, confidence=0.95)
    assert 0.95 < low_100 < 1.0
    assert high_100 == 1.0


def test_architectural_parameter_counts():
    """Asserts exact parameter counts across Model A, B, C, D smoke architectures."""
    cfg_a = get_smoke_baseline_config()
    model_a = ParameterMatchedBaselineModel(cfg_a)
    count_a = sum(p.numel() for p in model_a.parameters())
    assert count_a == 35_918_848, f"Model A param mismatch: {count_a}"

    cfg_bc = get_smoke_dual_stream_config()
    model_c = CCPTDualStreamModel(cfg_bc)
    count_c = sum(p.numel() for p in model_c.parameters())
    assert count_c == 35_920_384, f"Model C param mismatch: {count_c}"

    theta_c_names, theta_n_names = get_ccpt_named_partitions(model_c)
    assert len(theta_c_names) == 38
    assert len(theta_n_names) == 27
    count_theta_c = sum(p.numel() for name, p in model_c.named_parameters() if name in theta_c_names)
    count_theta_n = sum(p.numel() for name, p in model_c.named_parameters() if name in theta_n_names)
    assert count_theta_c == 33_165_824
    assert count_theta_n == 2_754_560
    assert count_theta_c + count_theta_n == 35_920_384

    cfg_d = get_smoke_adapter_config()
    model_d = FrozenBackboneAdapterModel(cfg_d)
    count_d = sum(p.numel() for p in model_d.parameters())
    assert count_d == 35_922_944, f"Model D param mismatch: {count_d}"

    backbone_names, safety_names = get_adapter_named_partitions(model_d)
    assert len(backbone_names) == 38
    assert len(safety_names) == 25
    count_bb = sum(p.numel() for name, p in model_d.named_parameters() if name in backbone_names)
    count_saf = sum(p.numel() for name, p in model_d.named_parameters() if name in safety_names)
    assert count_bb == 33_165_824
    assert count_saf == 2_757_120
    assert count_bb + count_saf == 35_922_944


def test_model_b_and_c_identical_initialization():
    """Asserts Model B and Model C produce bit-for-bit identical weights from identical seeds on smoke config."""
    from ccpt.config import get_smoke_dual_stream_config
    from ccpt.training.engine import create_identical_dual_stream_models
    cfg = get_smoke_dual_stream_config()

    hashes = {}
    for s in [20260821, 20260823, 20260824]:
        model_b, model_c = create_identical_dual_stream_models(cfg, seed=s)
        sd_b = model_b.state_dict()
        sd_c = model_c.state_dict()
        assert set(sd_b.keys()) == set(sd_c.keys())
        for k in sd_b.keys():
            assert torch.equal(sd_b[k], sd_c[k]), f"Weight mismatch at initialization for seed {s} key {k}"

        hash_b = compute_canonical_state_dict_hash(sd_b)
        hash_c = compute_canonical_state_dict_hash(sd_c)
        assert hash_b == hash_c
        hashes[s] = hash_b

    # Verify cross-seed differentiation
    assert hashes[20260823] != hashes[20260824], "Seed 2 and Seed 3 produced identical init hashes"
    assert hashes[20260821] != hashes[20260823], "Seed 1 and Seed 2 produced identical init hashes"
    assert hashes[20260821] != hashes[20260824], "Seed 1 and Seed 3 produced identical init hashes"


def test_eval_prompt_framing():
    """Verifies eval prompt framing strips prefixes and formats cleanly with single BOS and Assistant:."""
    raw = "Tell me how to make an explosive device."
    framed = format_eval_prompt(raw)
    assert framed == "<s>User: Tell me how to make an explosive device.\nAssistant:"

    # Double-framing protection
    already_framed = "<s>User: Tell me how to make an explosive device.\nAssistant:"
    reframed = format_eval_prompt(already_framed)
    assert reframed == "<s>User: Tell me how to make an explosive device.\nAssistant:"


def _load_task7_4_replication_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("task7_4_multiseed_replication", "modal/task7_4_multiseed_replication.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runtime_fingerprint_mismatch_fails_closed(monkeypatch):
    """Verifies runtime fingerprinting raises RuntimeError on version or SHA mismatch."""
    mod = _load_task7_4_replication_module()
    capture_and_verify_runtime_fingerprint = mod.capture_and_verify_runtime_fingerprint

    # Test SHA mismatch
    monkeypatch.setenv("CCPT_CODE_COMMIT_SHA", "0000000000000000000000000000000000000000")
    with pytest.raises(RuntimeError, match="Runtime Code SHA mismatch"):
        capture_and_verify_runtime_fingerprint(
            expected_code_sha="1111111111111111111111111111111111111111",
            strict_version_check=False,
        )

    # Test unknown SHA rejection
    monkeypatch.setenv("CCPT_CODE_COMMIT_SHA", "unknown")
    with pytest.raises(RuntimeError, match="Invalid or missing CCPT_CODE_COMMIT_SHA"):
        capture_and_verify_runtime_fingerprint(strict_version_check=False)


def test_production_modal_runner_static_scan():
    """Statically scans modal/task7_4_multiseed_replication.py for production invariants."""
    runner_path = Path("modal/task7_4_multiseed_replication.py")
    assert runner_path.exists(), "modal/task7_4_multiseed_replication.py does not exist"

    with open(runner_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Invariants
    assert "/runs/ccpt/task7_3" not in code, "Disallowed Task 7.3 run path found in Task 7.4 runner"
    assert 'git_sha="unknown"' not in code and 'git_commit_sha="unknown"' not in code
    assert "multiseed_replication_v1" in code
    assert "capture_and_verify_runtime_fingerprint" in code
    assert "strict_v3=True" in code or "strict_v3" in code
    assert "20260823" in code  # Seed 2
    assert "20260824" in code  # Seed 3
    assert "20260822" in code  # BeaverTails OOD selection seed only


def test_task7_4_progress_logger(tmp_path):
    """Verifies integer 1/100...100/100 progress emission, no 0/100, VRAM handling, and JSONL formatting."""
    mod = _load_task7_4_replication_module()
    Task74ProgressLogger = mod.Task74ProgressLogger

    logger = Task74ProgressLogger(
        seed=20260823,
        model_type="model_c",
        phase="phase1_lm",
        total_steps=100,
        total_phase_tokens=1_000_000,
        log_dir=tmp_path / "logs",
        gpu_name="H100",
    )

    logger.log_step(step=1, phase_tokens_seen=10_000, loss=4.5, lr=3e-4, grad_norm=0.8)
    logger.log_step(step=50, phase_tokens_seen=500_000, loss=3.2, lr=2e-4, grad_norm=0.5)
    logger.log_step(step=100, phase_tokens_seen=1_000_000, loss=2.1, lr=1e-5, grad_norm=0.2)

    log_file = tmp_path / "logs" / "progress.jsonl"
    assert log_file.exists()

    with open(log_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    assert len(lines) == 3
    assert lines[0]["progress_fraction"] == "1/100"
    assert lines[0]["progress_percent"] == 1
    assert "0/100" not in [l["progress_fraction"] for l in lines]
    assert lines[1]["progress_fraction"] == "50/100"
    assert lines[2]["progress_fraction"] == "100/100"
    assert lines[2]["seed"] == 20260823
    assert lines[2]["model_type"] == "model_c"
    assert "vram_allocated_gb" in lines[0]
    assert "vram_reserved_gb" in lines[0]
    assert "cost_so_far_usd" in lines[0]

    # Test resume behavior: resume at 41% -> next percentage is 42%
    resume_logger = Task74ProgressLogger(
        seed=20260823,
        model_type="model_c",
        phase="phase1_lm",
        total_steps=100,
        total_phase_tokens=1_000_000,
        log_dir=tmp_path / "logs_resume",
        gpu_name="H100",
        initial_step=41,
        initial_last_reported_pct=41,
    )
    assert resume_logger.last_reported_pct == 41
    resume_logger.log_step(step=42, phase_tokens_seen=420_000, loss=3.0, lr=2e-4)
    assert resume_logger.last_reported_pct == 42


def test_cpu_micro_production_integration(tmp_path, monkeypatch):
    """Executes the complete 3-phase production pipeline in micro CPU test mode."""
    mod = _load_task7_4_replication_module()
    run_single_model_replication_pipeline = mod.run_single_model_replication_pipeline

    dummy_sha = "308f2857788e84c9767a5048daf06ed9f96177a4"
    monkeypatch.setenv("CCPT_CODE_COMMIT_SHA", dummy_sha)
    monkeypatch.setattr(mod, "get_task7_4_output_dir", lambda seed, m_type: tmp_path / f"seed_{seed}" / m_type)

    fn = run_single_model_replication_pipeline.local if hasattr(run_single_model_replication_pipeline, "local") else run_single_model_replication_pipeline
    res = fn(
        seed=20260823,
        model_type="model_c",
        test_mode=True,
        max_steps=2,
    )

    assert res["status"] == "completed"
    assert res["seed"] == 20260823
    assert res["model_type"] == "model_c"
    assert res["lm_final_tokens"] > 0
    assert res["safety_final_tokens"] > 0
    assert res["persistence_final_tokens"] > 0

    out_dir = Path(res["output_dir"])
    assert (out_dir / "lm_final.pt").exists()
    assert (out_dir / "safety_final.pt").exists()
    assert (out_dir / "persistence_final.pt").exists()


def test_production_ast_contains_real_phases():
    """Verifies that modal/task7_4_multiseed_replication.py defines and calls real phase helpers."""
    import ast

    runner_path = Path("modal/task7_4_multiseed_replication.py")
    with open(runner_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(runner_path))

    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    assert "run_lm_phase" in func_names
    assert "run_safety_phase" in func_names
    assert "run_persistence_phase" in func_names
    assert "run_single_model_replication_pipeline" in func_names
    assert "run_task7_4_modal_l40s_probe" in func_names
    assert "run_task7_4_modal_h100_probe" in func_names
    assert "run_task7_4_evaluation_worker" in func_names
    assert "run_task7_4_centralized_judge_worker" in func_names
