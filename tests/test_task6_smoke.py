"""Unit tests for Task 6A smoke experiment configuration, schedulers, progress, and invariants."""

import copy
import hashlib
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from ccpt.config import get_smoke_baseline_config, get_smoke_dual_stream_config
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.training.engine import (
    assert_parameters_equal,
    count_changed_parameters,
    create_identical_dual_stream_models,
    snapshot_parameters,
)
from ccpt.training.progress import GPU_PRICES, LiveProgressReporter
from ccpt.training.scheduler import SafetyTokenCosineScheduler, TokenCosineScheduler


def test_task6_smoke_parameter_counts():
    """Verify exact parameter counts for Model A (35,918,848) and Models B/C (35,920,384)."""
    cfg_dual = get_smoke_dual_stream_config()
    m_dual = CCPTDualStreamModel(cfg_dual)
    p_dual = sum(p.numel() for p in m_dual.parameters())
    assert p_dual == 35_920_384, f"Expected 35,920,384 params for Dual-Stream, got {p_dual}"

    cfg_base = get_smoke_baseline_config()
    m_base = ParameterMatchedBaselineModel(cfg_base)
    p_base = sum(p.numel() for p in m_base.parameters())
    assert p_base == 35_918_848, f"Expected 35,918,848 params for Model A, got {p_base}"

    diff = abs(p_dual - p_base)
    assert diff == 1536, f"Expected diff of 1,536, got {diff}"
    rel_diff = diff / p_dual
    assert rel_diff < 0.0001, f"Relative param difference {rel_diff*100:.4f}% exceeds 0.01%"


def test_model_b_and_c_identical_initialization():
    """Verify Model B and Model C initialize with bit-identical parameters."""
    cfg = get_smoke_dual_stream_config()
    model_b, model_c = create_identical_dual_stream_models(cfg, seed=20260821)

    for (name_b, p_b), (name_c, p_c) in zip(model_b.named_parameters(), model_c.named_parameters()):
        assert name_b == name_c
        assert torch.equal(p_b, p_c), f"Parameter {name_b} differs between Model B and Model C"


def test_10b_token_based_scheduler_and_resume():
    """Verify 10B token schedule, warmup linearity, nonzero 1B LR, zero 10B LR, and state resume."""
    scheduler = TokenCosineScheduler(
        max_lr=3e-4,
        min_lr=0.0,
        warmup_tokens=100_000_000,
        total_tokens=10_000_000_000,
    )

    # 1. Warmup tests
    assert scheduler.get_lr(0) == 0.0
    assert scheduler.get_lr(50_000_000) == pytest.approx(1.5e-4)
    assert scheduler.get_lr(100_000_000) == pytest.approx(3e-4)

    # 2. 1B tokens checkpoint (step 30,517 * 32,768 = 999,981,056 tokens)
    tokens_1b = 999_981_056
    lr_1b = scheduler.get_lr(tokens_1b)
    assert lr_1b > 0.0, "LR at 1B must be nonzero"
    assert lr_1b == pytest.approx(2.9392e-4, rel=1e-3), f"Expected ~2.939e-4 at 1B, got {lr_1b}"

    # 3. 10B tokens endpoint
    assert scheduler.get_lr(10_000_000_000) == 0.0
    assert scheduler.get_lr(11_000_000_000) == 0.0

    # 4. State dict roundtrip & resume
    scheduler.step(tokens_1b)
    state = scheduler.state_dict()

    resumed_scheduler = TokenCosineScheduler()
    resumed_scheduler.load_state_dict(state)
    assert resumed_scheduler.tokens_seen == tokens_1b
    assert resumed_scheduler.get_lr() == pytest.approx(lr_1b)


def test_safety_token_scheduler():
    """Verify 10M safety token scheduler with 1% warmup."""
    scheduler = SafetyTokenCosineScheduler(
        max_lr=3e-4,
        min_lr=0.0,
        warmup_tokens=100_000,
        total_tokens=10_000_000,
    )

    assert scheduler.get_lr(0) == 0.0
    assert scheduler.get_lr(50_000) == pytest.approx(1.5e-4)
    assert scheduler.get_lr(100_000) == pytest.approx(3e-4)
    assert scheduler.get_lr(5_050_000) == pytest.approx(1.5e-4, rel=1e-2)
    assert scheduler.get_lr(10_000_000) == 0.0


def test_exact_task6a_token_arithmetic():
    """Verify exact 30,517 step token arithmetic."""
    global_batch_size = 32
    seq_len = 1024
    tokens_per_step = global_batch_size * seq_len
    assert tokens_per_step == 32_768

    total_steps = 30_517
    total_tokens = total_steps * tokens_per_step
    assert total_tokens == 999_981_056, f"Expected 999,981,056 tokens, got {total_tokens}"

    training_blocks = total_steps * global_batch_size
    assert training_blocks == 976_544, f"Expected 976,544 training blocks, got {training_blocks}"


def test_fineweb_validation_split_formula():
    """Verify document validation splitting formula SHA256(doc_id) % 1000 == 0."""
    def is_val_doc(doc_id: str) -> bool:
        h = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16)
        return (h % 1000) == 0

    val_count = 0
    total_samples = 10_000
    for i in range(total_samples):
        if is_val_doc(f"doc_{i}"):
            val_count += 1

    # Roughly 1/1000 (0.1%)
    assert 1 <= val_count <= 50, f"Unexpected validation sample count {val_count} out of {total_samples}"


def test_live_progress_reporter(tmp_path):
    """Verify LiveProgressReporter generates 1..100 monotonic progress with timestamps."""
    log_file = tmp_path / "progress.jsonl"
    reporter = LiveProgressReporter(
        task_name="TEST_TASK",
        total_steps=100,
        total_tokens=100_000,
        model_name="model_c",
        phase="LM",
        gpu_type="H100!",
        jsonl_path=log_file,
    )

    reported_pcts = []
    for step in range(1, 101):
        rec = reporter.step(
            current_step=step,
            tokens_seen=step * 1000,
            current_loss=5.0 - 0.03 * step,
            lr=3e-4,
            grad_norm=0.8,
            token_acc=0.45,
        )
        if rec:
            reported_pcts.append(rec["progress_pct"])

    # All 1..100 percentages must be emitted
    assert reported_pcts == list(range(1, 101))
    assert log_file.exists()

    with open(log_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 100


def test_gpu_selection_rule():
    """Verify GPU selection rule (cheapest unless within 10% cost and >= 25% faster)."""
    # Case 1: L40S is cheapest and H100 is >10% more expensive -> choose L40S
    results_1 = {
        "L40S": {"projected_total_cost_usd": 40.0, "projected_concurrent_wall_hours": 6.8},
        "H100!": {"projected_total_cost_usd": 46.0, "projected_concurrent_wall_hours": 3.9},
        "H200": {"projected_total_cost_usd": 52.0, "projected_concurrent_wall_hours": 3.8},
    }

    def select_gpu(benchmark_data):
        cheapest_gpu = min(benchmark_data.keys(), key=lambda g: benchmark_data[g]["projected_total_cost_usd"])
        cheapest_cost = benchmark_data[cheapest_gpu]["projected_total_cost_usd"]
        cheapest_time = benchmark_data[cheapest_gpu]["projected_concurrent_wall_hours"]

        winner = cheapest_gpu
        for g, data in benchmark_data.items():
            if g == cheapest_gpu:
                continue
            cost = data["projected_total_cost_usd"]
            wall_time = data["projected_concurrent_wall_hours"]

            cost_delta_frac = (cost - cheapest_cost) / cheapest_cost
            speedup_frac = (cheapest_time - wall_time) / cheapest_time

            if cost_delta_frac <= 0.10 and speedup_frac >= 0.25:
                winner = g
                break
        return winner

    assert select_gpu(results_1) == "L40S"

    # Case 2: H100 is within 8% cost and 40% faster -> choose H100
    results_2 = {
        "L40S": {"projected_total_cost_usd": 40.0, "projected_concurrent_wall_hours": 6.8},
        "H100!": {"projected_total_cost_usd": 43.0, "projected_concurrent_wall_hours": 3.9},
        "H200": {"projected_total_cost_usd": 50.0, "projected_concurrent_wall_hours": 3.8},
    }
    assert select_gpu(results_2) == "H100!"


def test_wildguardtest_path_forbidden():
    """Verify that WildGuardTest path is not referenced in Task 6 source files."""
    import re
    src_dir = Path("src/ccpt")
    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text()
        assert "wildguardtest.arrow" not in content.lower(), f"Forbidden WildGuardTest found in {py_file}"
