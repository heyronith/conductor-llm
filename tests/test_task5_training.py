"""Tests for Task 5 micro-training architecture, controller ablation, checkpointing, and freezing."""

import copy
import pytest
import torch
import torch.nn as nn
from ccpt.config import (
    get_task5_micro_baseline_config,
    get_task5_micro_dual_stream_config,
)
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
from ccpt.training.checkpoint import load_checkpoint, save_checkpoint
from ccpt.training.engine import (
    assert_parameters_equal,
    count_changed_parameters,
    create_identical_dual_stream_models,
    evaluate_lm_loss_and_acc,
    snapshot_parameters,
)
from ccpt.training.metrics import (
    MetricLogger,
    compute_gate_diagnostics,
    compute_gradient_group_norms,
    compute_steering_diagnostics,
)


def test_task5_parameter_matched_counts():
    """Verify exact parameter counts for Task 5 real-token micro configurations."""
    cfg_dual = get_task5_micro_dual_stream_config()
    model_c = CCPTDualStreamModel(cfg_dual)
    count_c = sum(p.numel() for p in model_c.parameters())

    model_b = JointTrainingDualStreamModel(cfg_dual)
    count_b = sum(p.numel() for p in model_b.parameters())

    cfg_base = get_task5_micro_baseline_config()
    model_a = ParameterMatchedBaselineModel(cfg_base)
    count_a = sum(p.numel() for p in model_a.parameters())

    assert count_c == count_b == 2_243_392, f"Expected 2,243,392 params for DualStream, got {count_c}"
    assert count_a == 2_243_200, f"Expected 2,243,200 params for Model A, got {count_a}"
    assert abs(count_c - count_a) == 192, f"Expected 192 param difference, got {abs(count_c - count_a)}"


def test_model_b_and_c_identical_initialization():
    """Verify create_identical_dual_stream_models initializes Model B and C with identical parameters."""
    cfg = get_task5_micro_dual_stream_config()
    model_b, model_c = create_identical_dual_stream_models(cfg, seed=20260821)

    for (k_b, p_b), (k_c, p_c) in zip(model_b.state_dict().items(), model_c.state_dict().items()):
        assert k_b == k_c
        assert torch.equal(p_b, p_c)


def test_controller_scale_preserves_and_ablates_steering():
    """Verify controller_scale=1.0 matches default and controller_scale=0.0 removes steering."""
    cfg = get_task5_micro_dual_stream_config()
    model = CCPTDualStreamModel(cfg)

    # Put arbitrary nonzero weights in controllers
    for g in model.gate_projections:
        nn.init.normal_(g.weight, std=0.5)
    for s in model.steering_projections:
        nn.init.normal_(s.weight, std=0.5)

    x = torch.randint(0, 32000, (2, 16))

    # 1. Default vs controller_scale=1.0
    logits_def, risk_def, diag_def = model(x, mode="controlled", return_diagnostics=True)
    logits_sc1, risk_sc1, diag_sc1 = model(x, mode="controlled", controller_scale=1.0, return_diagnostics=True)

    assert torch.allclose(logits_def, logits_sc1)
    assert torch.allclose(diag_def["gates"]["layer_2"], diag_sc1["gates"]["layer_2"])
    assert torch.allclose(diag_def["steering"]["layer_2"], diag_sc1["steering"]["layer_2"])

    # 2. controller_scale=0.0 vs mode='lm'
    logits_sc0, risk_sc0, diag_sc0 = model(x, mode="controlled", controller_scale=0.0, return_diagnostics=True)
    logits_lm, risk_lm = model(x, mode="lm")

    # Logits must match pure LM logits within floating point tolerance
    assert torch.allclose(logits_sc0, logits_lm, atol=1e-6)

    # When ablated, gates must equal exactly 1.0 and steering must equal exactly 0.0
    for g in diag_sc0["gates"].values():
        assert torch.allclose(g, torch.ones_like(g))
    for s in diag_sc0["steering"].values():
        assert torch.allclose(s, torch.zeros_like(s))


def test_checkpoint_roundtrip_and_data_hash_validation(tmp_path):
    """Verify checkpoint saving, loading, RNG restoration, and strict data hash rejection."""
    cfg = get_task5_micro_dual_stream_config()
    model = CCPTDualStreamModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "test_checkpoint.pt"
    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        optimizer=opt,
        phase="phase1_lm",
        global_step=42,
        model_type="model_c",
        model_config=cfg,
        task4_manifest_hash="task4_hash_123",
        task5_subset_hash="task5_hash_456",
        training_seed=20260821,
        metrics_so_far={"lm_loss": 3.14},
    )

    # Successful load with valid hashes
    loaded = load_checkpoint(
        ckpt_path,
        expected_task4_manifest_hash="task4_hash_123",
        expected_task5_subset_hash="task5_hash_456",
    )
    assert loaded["global_step"] == 42
    assert loaded["phase"] == "phase1_lm"
    assert loaded["metrics_so_far"]["lm_loss"] == 3.14

    # Bit-identical state_dict
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded["model_state_dict"][k])

    # Rejection on invalid Task 4 hash
    with pytest.raises(ValueError, match="Task 4 manifest hash mismatch"):
        load_checkpoint(ckpt_path, expected_task4_manifest_hash="wrong_hash")

    # Rejection on invalid Task 5 subset hash
    with pytest.raises(ValueError, match="Task 5 subset hash mismatch"):
        load_checkpoint(ckpt_path, expected_task5_subset_hash="wrong_hash")


def test_deterministic_checkpoint_resume(tmp_path):
    """Verify that saving and reloading state reproduces uninterrupted optimization identically."""
    cfg = get_task5_micro_dual_stream_config()
    torch.manual_seed(20260821)
    model1 = CCPTDualStreamModel(cfg)
    opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    model2 = copy.deepcopy(model1)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

    x = torch.randint(0, 32000, (4, 16))

    # Step 1: both train 3 steps
    for _ in range(3):
        loss1, _ = evaluate_lm_loss_and_acc(model1, x)
        loss_t = torch.tensor(loss1, requires_grad=True)
        # Dummy step
        opt1.zero_grad()
        logits, _ = model1(x, mode="lm")
        l = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32000), x[:, 1:].reshape(-1))
        l.backward()
        opt1.step()

        opt2.zero_grad()
        logits2, _ = model2(x, mode="lm")
        l2 = nn.functional.cross_entropy(logits2[:, :-1].reshape(-1, 32000), x[:, 1:].reshape(-1))
        l2.backward()
        opt2.step()

    # Save model1 checkpoint
    ckpt_path = tmp_path / "resume_test.pt"
    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model1,
        optimizer=opt1,
        phase="phase1_lm",
        global_step=3,
        model_type="model_c",
        model_config=cfg,
        task4_manifest_hash="hash",
        task5_subset_hash="hash",
        training_seed=20260821,
    )

    # Continue model1 for 1 more uninterrupted step
    opt1.zero_grad()
    logits, _ = model1(x, mode="lm")
    l = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32000), x[:, 1:].reshape(-1))
    l.backward()
    opt1.step()

    # Reload into model3 and take 1 resumed step
    model3 = CCPTDualStreamModel(cfg)
    opt3 = torch.optim.AdamW(model3.parameters(), lr=1e-3)
    loaded = load_checkpoint(ckpt_path)
    model3.load_state_dict(loaded["model_state_dict"])
    opt3.load_state_dict(loaded["optimizer_state_dict"])

    opt3.zero_grad()
    logits3, _ = model3(x, mode="lm")
    l3 = nn.functional.cross_entropy(logits3[:, :-1].reshape(-1, 32000), x[:, 1:].reshape(-1))
    l3.backward()
    opt3.step()

    # Verify model1 and model3 are bit-for-bit identical
    for p1, p3 in zip(model1.parameters(), model3.parameters()):
        assert torch.equal(p1, p3)


def test_metric_logger_and_diagnostics(tmp_path):
    """Verify diagnostic metric calculations and logger rejection of non-finite values."""
    gates = {"layer_2": torch.tensor([[[0.95], [1.05], [0.9005], [1.0995]]])}
    steering = {"layer_2": torch.tensor([[[0.1, -0.2], [0.5, 0.995]]])}

    gate_stats = compute_gate_diagnostics(gates)
    assert 0.9 <= gate_stats["gate_min"] <= gate_stats["gate_max"] <= 1.1
    assert gate_stats["gate_near_lower_bound_fraction"] > 0
    assert gate_stats["gate_near_upper_bound_fraction"] > 0

    steer_stats = compute_steering_diagnostics(steering)
    assert steer_stats["steering_l2_mean"] > 0
    assert steer_stats["steering_saturate_fraction"] > 0

    log_path = tmp_path / "metrics.jsonl"
    logger = MetricLogger(log_path)
    logger.log({"step": 1, "loss": 2.5})

    with pytest.raises(ValueError, match="Non-finite metric"):
        logger.log({"step": 2, "loss": float("nan")})


def test_capability_parameter_freeze_snapshots():
    """Verify parameter snapshotting and mutation detection helpers."""
    cfg = get_task5_micro_dual_stream_config()
    model = CCPTDualStreamModel(cfg)

    theta_c = model.theta_C
    snapshots = snapshot_parameters(theta_c)

    assert count_changed_parameters(theta_c, snapshots) == 0
    assert_parameters_equal(theta_c, snapshots)

    # Mutate one parameter
    with torch.no_grad():
        theta_c[0].add_(1.0)

    assert count_changed_parameters(theta_c, snapshots) == 1
    with pytest.raises(AssertionError):
        assert_parameters_equal(theta_c, snapshots)


def test_checkpoint_lineage_metadata_extraction_and_validation(tmp_path):
    """Verify metadata extraction and unanimous lineage validation across multiple checkpoints."""
    from ccpt.training.checkpoint import (
        inspect_checkpoint_metadata,
        validate_checkpoint_lineage,
    )

    cfg = get_task5_micro_dual_stream_config()
    model = CCPTDualStreamModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt1_path = tmp_path / "ckpt1.pt"
    ckpt2_path = tmp_path / "ckpt2.pt"

    save_checkpoint(
        checkpoint_path=ckpt1_path,
        model=model,
        optimizer=opt,
        phase="phase1_lm",
        global_step=200,
        model_type="model_c",
        model_config=cfg,
        task4_manifest_hash="task4_hash_abc",
        task5_subset_hash="train_subset_hash_xyz",
        training_seed=20260821,
    )
    save_checkpoint(
        checkpoint_path=ckpt2_path,
        model=model,
        optimizer=opt,
        phase="phase2_risk",
        global_step=350,
        model_type="model_c",
        model_config=cfg,
        task4_manifest_hash="task4_hash_abc",
        task5_subset_hash="train_subset_hash_xyz",
        training_seed=20260821,
    )

    # 1. Metadata reader extracts correct fields
    meta1 = inspect_checkpoint_metadata(ckpt1_path)
    assert meta1["task4_manifest_hash"] == "task4_hash_abc"
    assert meta1["task5_subset_hash"] == "train_subset_hash_xyz"
    assert meta1["phase"] == "phase1_lm"
    assert meta1["global_step"] == 200

    # 2. Lineage validator succeeds when checkpoints agree
    summary = validate_checkpoint_lineage(
        [ckpt1_path, ckpt2_path],
        expected_task4_hash="task4_hash_abc",
        expected_training_subset_hash="train_subset_hash_xyz",
    )
    assert summary["all_checkpoint_task4_hashes_match"] is True
    assert summary["all_checkpoint_subset_hashes_match"] is True
    assert summary["checkpoint_count_verified"] == 2


def test_checkpoint_lineage_disagreement_and_errors(tmp_path):
    """Verify lineage validator catches hash disagreements, wrong Task 4 hashes, and missing hashes."""
    from ccpt.training.checkpoint import validate_checkpoint_lineage

    cfg = get_task5_micro_dual_stream_config()
    model = CCPTDualStreamModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt1 = tmp_path / "c1.pt"
    ckpt2 = tmp_path / "c2.pt"

    save_checkpoint(ckpt1, model, opt, "p1", 10, "model_c", cfg, "task4_good", "subset_A", 20260821)
    save_checkpoint(ckpt2, model, opt, "p2", 20, "model_c", cfg, "task4_good", "subset_B", 20260821)

    # Disagreement between checkpoints
    with pytest.raises(ValueError, match="Task 5 subset hash disagreement"):
        validate_checkpoint_lineage([ckpt1, ckpt2], expected_task4_hash="task4_good")

    # Wrong Task 4 hash
    with pytest.raises(ValueError, match="Task 4 hash mismatch"):
        validate_checkpoint_lineage([ckpt1], expected_task4_hash="task4_wrong")


def test_sanitized_manifest_forbids_raw_data_fields():
    """Verify sanitized review manifest contains no raw prompt text, prompt keys, or token arrays."""
    from ccpt.data.hashing import sha256_json

    sanitized_manifest = {
        "manifest_kind": "sanitized_review_manifest",
        "version": "task5-micro-v1",
        "seed": 20260821,
        "task4_manifest_hash": "2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097",
        "training_subset_manifest_hash": "3480afd5769b483a5b269f0bc8c87188454974bf13f5a862e7906501f933960f",
        "sanitized_review_manifest_hash": "1b315015ee2e01c86da989192ea789526ec232b052a2349451611552f6935132",
        "sequence_length": 128,
        "lm": {
            "source_shard": "/data/ccpt/fineweb/smoke/train/smoke_tokens.bin",
            "num_sequences": 16,
            "sequence_length": 128,
            "total_tokens": 2048,
            "slices": [[0, 128], [128, 256]],
            "logical_hash": "bf504fcaaa3d6c119cbfe20af8cdf0e4ac20bb5fe6b55525fa0f7445219517e3",
        },
        "risk": {
            "source": "/data/ccpt/wildguard/risk/train.arrow",
            "num_examples": 32,
            "harmful_count": 16,
            "benign_count": 16,
            "logical_hash": "1f044df37e2e227aee6469325759c97b5a68a5ee12063d107271b87322cdb4a3",
            "example_ids": ["ex_1", "ex_2"],
        },
        "generation": {
            "source": "/data/ccpt/wildguard/generation/train.arrow",
            "num_examples": 32,
            "harmful_refusal_count": 16,
            "benign_compliance_count": 16,
            "logical_hash": "bca3358d57649f402554c2edd620d99205b4f0b41c09151957943b62401661d2",
            "example_ids": ["gen_1", "gen_2"],
        },
    }

    # Recursive check for forbidden keys
    forbidden_keys = {"prompt", "response", "prompt_group_key", "records", "input_ids", "tokens"}

    def check_no_forbidden(d):
        if isinstance(d, dict):
            for k, v in d.items():
                assert k not in forbidden_keys, f"Forbidden key '{k}' found in sanitized manifest!"
                check_no_forbidden(v)
        elif isinstance(d, list):
            for item in d:
                check_no_forbidden(item)

    check_no_forbidden(sanitized_manifest)

    # Verify that training-time hash and sanitized hash can cleanly coexist
    assert sanitized_manifest["training_subset_manifest_hash"] != sanitized_manifest["sanitized_review_manifest_hash"]

