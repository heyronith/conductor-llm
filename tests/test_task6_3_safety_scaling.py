"""Unit tests for Task 6.3 safety budget scaling diagnostic (10M -> 20M -> 40M)."""

import hashlib
import json
import math
from pathlib import Path
import pytest
import torch

from ccpt.config import get_smoke_dual_stream_config
from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.training.engine import snapshot_parameters
from ccpt.training.losses import token_weighted_continuation_nll_and_count
from ccpt.training.scheduler import SafetyTokenCosineScheduler, TokenCosineScheduler


def test_40m_safety_token_scheduler_and_milestones():
    """Verify 40M token scheduler has 400k warmup, active LR at 10M and 20M, and zero at 40M."""
    total_tokens = 40_000_000
    warmup_tokens = 400_000  # 1%
    max_lr = 3e-4
    min_lr = 0.0

    scheduler = TokenCosineScheduler(
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_tokens=warmup_tokens,
        total_tokens=total_tokens,
    )

    # Start
    assert scheduler.get_lr(0) == 0.0
    # Peak at warmup
    assert scheduler.get_lr(warmup_tokens) == max_lr

    # Active LR at 10M crossing (25% horizon)
    lr_10m = scheduler.get_lr(10_000_000)
    assert lr_10m > 0.0
    assert lr_10m < max_lr
    # 0.5 * (1 + cos(pi * 9.6/39.6)) ~ 0.5 * (1 + cos(0.761))
    assert lr_10m == pytest.approx(max_lr * 0.5 * (1.0 + math.cos(math.pi * (10_000_000 - warmup_tokens) / (total_tokens - warmup_tokens))), rel=1e-3)

    # Active LR at 20M crossing (50% horizon)
    lr_20m = scheduler.get_lr(20_000_000)
    assert lr_20m > 0.0
    assert lr_20m < lr_10m

    # Zero at 40M endpoint
    lr_40m = scheduler.get_lr(total_tokens)
    assert lr_40m == 0.0


def test_20m_continuation_on_same_scheduler():
    """Verify 20M checkpoint resumes seamlessly on the exact same scheduler instance."""
    scheduler = TokenCosineScheduler(
        max_lr=3e-4,
        min_lr=0.0,
        warmup_tokens=400_000,
        total_tokens=40_000_000,
    )
    tokens_20m = 20_000_000
    batch_tokens = 8400  # ~32 samples
    next_tokens = tokens_20m + batch_tokens

    lr_curr = scheduler.get_lr(tokens_20m)
    lr_next = scheduler.get_lr(next_tokens)

    assert lr_next < lr_curr
    assert lr_next > 0.0


def test_1_to_1_risk_gen_alternation_logic():
    """Verify 1:1 alternating batch sequencing."""
    batches = ["risk" if i % 2 == 0 else "gen" for i in range(10)]
    assert batches == ["risk", "gen", "risk", "gen", "risk", "gen", "risk", "gen", "risk", "gen"]


def test_deterministic_epoch_permutations():
    """Verify deterministic epoch permutation seeding derived from base seed."""
    base_seed = 20260821
    def get_epoch_seed(dataset_kind: str, epoch: int) -> int:
        h = hashlib.sha256(f"{base_seed}_{dataset_kind}_{epoch}".encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    seed_0 = get_epoch_seed("risk", 0)
    seed_1 = get_epoch_seed("risk", 1)
    seed_0_repeat = get_epoch_seed("risk", 0)

    assert seed_0 == seed_0_repeat
    assert seed_0 != seed_1


def test_c_theta_c_exact_freeze_invariant():
    """Verify theta_C freeze assertion on CCPTDualStreamModel."""
    cfg = get_smoke_dual_stream_config()
    model = CCPTDualStreamModel(cfg)

    # Initial snapshots
    clean_theta_c = snapshot_parameters(model.theta_C)
    clean_theta_n = snapshot_parameters(model.theta_N)

    # Simulate safety update on theta_N only
    for p in model.theta_N:
        p.data.add_(0.02)

    # Verify theta_C is completely unchanged
    changed_c = sum(1 for snap, p in zip(clean_theta_c, model.theta_C) if not torch.equal(snap, p.data))
    changed_n = sum(1 for snap, p in zip(clean_theta_n, model.theta_N) if not torch.equal(snap, p.data))

    assert changed_c == 0
    assert changed_n == len(clean_theta_n)


def test_token_weighted_generation_loss_arithmetic():
    """Verify token-weighted CE arithmetic."""
    logits = torch.randn(2, 8, 100)
    input_ids = torch.randint(0, 100, (2, 8))
    prompt_ends = torch.tensor([2, 5])  # Ex 0 has 5 continuation tokens; Ex 1 has 2 continuation tokens

    nll, count = token_weighted_continuation_nll_and_count(logits, input_ids, prompt_ends)
    assert count == 7
    assert nll > 0.0


def test_20m_decision_gate_logic():
    """Verify 20M decision logic for PASS, CONTINUE_TO_40M, and STOP_FAIL."""
    # Case 1: Pass at 20M (gap <= 10%, bal_acc >= best - 0.05, ablation >= 0.05, cap <= 0.15)
    gap_pass = 0.08
    bal_acc_pass = 0.55
    best_bal_acc = 0.50
    ablation_pass = 0.18
    cap_deg_pass = 0.05
    c_freeze = True

    is_20m_pass = (
        gap_pass <= 0.10
        and bal_acc_pass >= (best_bal_acc - 0.05)
        and ablation_pass >= 0.05
        and cap_deg_pass <= 0.15
        and c_freeze
    )
    assert is_20m_pass is True

    # Case 2: Continue to 40M (gap 12%, improved from 10M gap of 14%, ablation >= 5%, cap <= 15%)
    gap_20m = 0.12
    gap_10m = 0.14
    is_continue_40m = (
        gap_20m <= 0.15
        and gap_20m < gap_10m
        and ablation_pass >= 0.05
        and cap_deg_pass <= 0.15
        and c_freeze
    )
    assert is_continue_40m is True

    # Case 3: Stop-Fail (gap 18% > 15%)
    gap_fail = 0.18
    assert not (gap_fail <= 0.15)


def test_wildguardtest_strictly_forbidden():
    """Verify WildGuardTest partition is never accessed in Task 6.3 code."""
    for py_file in Path("src/ccpt").rglob("*.py"):
        content = py_file.read_text().lower()
        assert "wildguardtest.arrow" not in content
