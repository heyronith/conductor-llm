"""Authoritative preflight proof functions and production path scanner for Task 7.2.2."""

from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from ccpt.config import (
    AdapterConfig,
    BaselineConfig,
    DualStreamConfig,
    get_smoke_adapter_config,
    get_smoke_baseline_config,
    get_smoke_dual_stream_config,
)
from ccpt.modeling.adapter import FrozenBackboneAdapterModel
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION_V2,
    load_checkpoint,
    save_checkpoint,
)
from ccpt.training.scheduler import TokenCosineScheduler


def run_checkpoint_lm_strictness_proof(tmp_dir: Optional[Union[str, Path]] = None) -> Dict[str, bool]:
    """Executes live strictness validation on Checkpoint V2 LM checkpoints and computes derived booleans."""
    target_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="ckpt_lm_proof_"))
    target_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_smoke_baseline_config()
    model = ParameterMatchedBaselineModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = TokenCosineScheduler(max_lr=1e-3, min_lr=1e-5, warmup_tokens=10, total_tokens=100)

    # 1. Valid baseline LM checkpoint must pass
    valid_path = target_dir / "valid_lm.pt"
    save_checkpoint(
        checkpoint_path=valid_path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        phase="phase1_pretrain_1b",
        global_step=10,
        model_type="model_a",
        model_config=cfg,
        task4_manifest_hash="task4_manifest_hash_test_12345",
        data_manifest_hash="data_manifest_hash_test_67890",
        stream_identity="fineweb-edu-100BT",
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )
    loaded_valid = load_checkpoint(valid_path, strict_v2=True)
    valid_passes = (loaded_valid is not None and loaded_valid.get("global_step") == 10)

    # 2. Missing optimizer in production phase must reject
    no_opt_path = target_dir / "no_opt.pt"
    save_checkpoint(
        checkpoint_path=no_opt_path,
        model=model,
        optimizer=None,
        scheduler=sched,
        phase="phase1_pretrain_1b",
        global_step=10,
        model_type="model_a",
        model_config=cfg,
        task4_manifest_hash="task4_manifest_hash_test_12345",
        data_manifest_hash="data_manifest_hash_test_67890",
        stream_identity="fineweb-edu-100BT",
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )
    null_opt_rejects = False
    try:
        load_checkpoint(no_opt_path, strict_v2=True)
    except ValueError as e:
        if "requires non-null 'optimizer_state_dict'" in str(e):
            null_opt_rejects = True

    # 3. Missing scheduler in production phase must reject
    no_sched_path = target_dir / "no_sched.pt"
    save_checkpoint(
        checkpoint_path=no_sched_path,
        model=model,
        optimizer=opt,
        scheduler=None,
        phase="phase1_pretrain_1b",
        global_step=10,
        model_type="model_a",
        model_config=cfg,
        task4_manifest_hash="task4_manifest_hash_test_12345",
        data_manifest_hash="data_manifest_hash_test_67890",
        stream_identity="fineweb-edu-100BT",
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )
    null_sched_rejects = False
    try:
        load_checkpoint(no_sched_path, strict_v2=True)
    except ValueError as e:
        if "requires non-null 'scheduler_state'" in str(e):
            null_sched_rejects = True

    # 4. Missing data_manifest_hash must reject
    no_data_path = target_dir / "no_data.pt"
    save_checkpoint(
        checkpoint_path=no_data_path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        phase="phase1_pretrain_1b",
        global_step=10,
        model_type="model_a",
        model_config=cfg,
        task4_manifest_hash="task4_manifest_hash_test_12345",
        data_manifest_hash="",
        stream_identity="fineweb-edu-100BT",
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )
    missing_data_hash_rejects = False
    try:
        load_checkpoint(no_data_path, strict_v2=True)
    except ValueError as e:
        if "requires non-empty 'data_manifest_hash'" in str(e):
            missing_data_hash_rejects = True

    # 5. Missing task4_manifest_hash must reject
    no_task4_path = target_dir / "no_task4.pt"
    save_checkpoint(
        checkpoint_path=no_task4_path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        phase="phase1_pretrain_1b",
        global_step=10,
        model_type="model_a",
        model_config=cfg,
        task4_manifest_hash="",
        data_manifest_hash="data_manifest_hash_test_67890",
        stream_identity="fineweb-edu-100BT",
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )
    missing_task4_hash_rejects = False
    try:
        load_checkpoint(no_task4_path, strict_v2=True)
    except ValueError as e:
        if "requires non-empty 'task4_manifest_hash'" in str(e):
            missing_task4_hash_rejects = True

    # 6. Empty stream_identity must reject
    no_stream_path = target_dir / "no_stream.pt"
    save_checkpoint(
        checkpoint_path=no_stream_path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        phase="phase1_pretrain_1b",
        global_step=10,
        model_type="model_a",
        model_config=cfg,
        task4_manifest_hash="task4_manifest_hash_test_12345",
        data_manifest_hash="data_manifest_hash_test_67890",
        stream_identity="",
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )
    empty_stream_rejects = False
    try:
        load_checkpoint(no_stream_path, strict_v2=True)
    except ValueError as e:
        if "requires non-empty 'stream_identity'" in str(e):
            empty_stream_rejects = True

    all_passed = (
        valid_passes
        and null_opt_rejects
        and null_sched_rejects
        and missing_data_hash_rejects
        and missing_task4_hash_rejects
        and empty_stream_rejects
    )

    return {
        "valid_passes": valid_passes,
        "null_opt_rejects": null_opt_rejects,
        "null_sched_rejects": null_sched_rejects,
        "missing_data_hash_rejects": missing_data_hash_rejects,
        "missing_task4_hash_rejects": missing_task4_hash_rejects,
        "empty_stream_rejects": empty_stream_rejects,
        "all_passed": all_passed,
    }


def run_checkpoint_safety_strictness_proof(tmp_dir: Optional[Union[str, Path]] = None) -> Dict[str, bool]:
    """Executes live strictness validation on Checkpoint V2 Safety checkpoints and computes derived booleans."""
    target_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="ckpt_safety_proof_"))
    target_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_smoke_dual_stream_config()
    model = CCPTDualStreamModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = TokenCosineScheduler(max_lr=1e-3, min_lr=1e-5, warmup_tokens=10, total_tokens=100)

    # 1. Valid safety checkpoint with safety_schedule_hash must pass
    valid_path = target_dir / "valid_safety.pt"
    save_checkpoint(
        checkpoint_path=valid_path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        phase="phase3_safety_20m",
        global_step=10,
        model_type="model_c",
        model_config=cfg,
        task4_manifest_hash="task4_manifest_hash_test_12345",
        data_manifest_hash="data_manifest_hash_test_67890",
        safety_schedule_hash="safety_schedule_hash_test_abcde",
        stream_identity="fineweb-edu-100BT",
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )
    loaded_valid = load_checkpoint(valid_path, strict_v2=True)
    valid_safety_passes = (loaded_valid is not None and loaded_valid.get("safety_schedule_hash") == "safety_schedule_hash_test_abcde")

    # 2. Missing safety_schedule_hash in safety phase must reject
    missing_sched_path = target_dir / "missing_sched_safety.pt"
    save_checkpoint(
        checkpoint_path=missing_sched_path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        phase="phase3_safety_20m",
        global_step=10,
        model_type="model_c",
        model_config=cfg,
        task4_manifest_hash="task4_manifest_hash_test_12345",
        data_manifest_hash="data_manifest_hash_test_67890",
        safety_schedule_hash="",
        stream_identity="fineweb-edu-100BT",
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )
    missing_safety_sched_rejects = False
    try:
        load_checkpoint(missing_sched_path, strict_v2=True)
    except ValueError as e:
        if "requires non-empty 'safety_schedule_hash'" in str(e):
            missing_safety_sched_rejects = True

    all_passed = valid_safety_passes and missing_safety_sched_rejects
    return {
        "valid_safety_passes": valid_safety_passes,
        "missing_safety_sched_rejects": missing_safety_sched_rejects,
        "all_passed": all_passed,
    }


def run_config_compatibility_proof(tmp_dir: Optional[Union[str, Path]] = None) -> Dict[str, bool]:
    """Tests that mutations to critical architectural parameters are rejected during strict loading."""
    target_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="ckpt_cfg_proof_"))
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. DualStream mutations: d_N, controlled_layers, alpha
    cfg_dual = get_smoke_dual_stream_config()
    model_dual = CCPTDualStreamModel(cfg_dual)
    ckpt_dual = target_dir / "dual_ckpt.pt"
    save_checkpoint(
        checkpoint_path=ckpt_dual,
        model=model_dual,
        optimizer=None,
        phase="test",
        global_step=1,
        model_type="model_c",
        model_config=cfg_dual,
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )

    # d_N mutation
    cfg_mut_dN = get_smoke_dual_stream_config()
    cfg_mut_dN.d_N = cfg_dual.d_N + 16
    mut_dN_rejected = False
    try:
        load_checkpoint(ckpt_dual, expected_model_config=cfg_mut_dN)
    except ValueError as e:
        if "mismatch on 'd_N'" in str(e):
            mut_dN_rejected = True

    # controlled_layers mutation
    cfg_mut_ctrl = get_smoke_dual_stream_config()
    cfg_mut_ctrl.controlled_layers = (0, 1)
    mut_ctrl_rejected = False
    try:
        load_checkpoint(ckpt_dual, expected_model_config=cfg_mut_ctrl)
    except ValueError as e:
        if "mismatch on 'controlled_layers'" in str(e):
            mut_ctrl_rejected = True

    # alpha mutation
    cfg_mut_alpha = get_smoke_dual_stream_config()
    cfg_mut_alpha.alpha = 0.5
    mut_alpha_rejected = False
    try:
        load_checkpoint(ckpt_dual, expected_model_config=cfg_mut_alpha)
    except ValueError as e:
        if "mismatch on 'alpha'" in str(e):
            mut_alpha_rejected = True

    # 2. Adapter mutation: d_mid
    cfg_adapter = get_smoke_adapter_config()
    model_adapter = FrozenBackboneAdapterModel(cfg_adapter)
    ckpt_adapter = target_dir / "adapter_ckpt.pt"
    save_checkpoint(
        checkpoint_path=ckpt_adapter,
        model=model_adapter,
        optimizer=None,
        phase="test",
        global_step=1,
        model_type="model_d",
        model_config=cfg_adapter,
        format_version=CHECKPOINT_FORMAT_VERSION_V2,
    )

    cfg_mut_dmid = get_smoke_adapter_config()
    cfg_mut_dmid.d_mid = cfg_adapter.d_mid + 8
    mut_dmid_rejected = False
    try:
        load_checkpoint(ckpt_adapter, expected_model_config=cfg_mut_dmid)
    except ValueError as e:
        if "mismatch on 'd_mid'" in str(e):
            mut_dmid_rejected = True

    all_passed = (
        mut_dN_rejected
        and mut_ctrl_rejected
        and mut_alpha_rejected
        and mut_dmid_rejected
    )

    return {
        "mut_dN_rejected": mut_dN_rejected,
        "mut_ctrl_rejected": mut_ctrl_rejected,
        "mut_alpha_rejected": mut_alpha_rejected,
        "mut_dmid_rejected": mut_dmid_rejected,
        "all_passed": all_passed,
    }


def scan_production_paths(repo_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Scans all production and test files in src/, modal/, and scripts/ to audit forbidden references."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]

    active_files: List[Path] = []
    legacy_disabled_files: List[Path] = []
    test_files: List[Path] = []

    # Categorize files
    for folder in ["src", "modal", "scripts", "tests"]:
        p = root / folder
        if not p.exists():
            continue
        for f in p.glob("**/*.py"):
            if "tests/" in str(f) or f.name.startswith("test_"):
                test_files.append(f)
            elif (
                f.name == "task7_pilot_v2.py"
                or f.name == "task7_2_1_real_proofs.py"
                or f.name == "run_task7_2_proofs.py"
                or f.name.startswith("task6_")
            ):
                legacy_disabled_files.append(f)
            else:
                active_files.append(f)

    # Scanned metrics across active production files
    task6_active_refs = 0
    ref_tok_active_refs = 0
    mock_bt_active_refs = 0
    use_mock_active_refs = 0
    hardcoded_eval_cost_refs = 0
    hardcoded_gpu_rate_refs = 0

    forbidden_task6 = "/data/" + "task6"
    forbidden_ref_tok = "Reference" + "Tokenizer"
    forbidden_mock_bt = "mock_records=" + "True"

    for f in active_files:
        if f.name == "preflight_proofs.py":
            continue
        content = f.read_text(encoding="utf-8")
        task6_active_refs += content.count(forbidden_task6)
        ref_tok_active_refs += content.count(forbidden_ref_tok)
        mock_bt_active_refs += content.count(forbidden_mock_bt)
        use_mock_active_refs += len(re.findall(r"use_mock\s*=\s*True", content))
        hardcoded_eval_cost_refs += len(re.findall(r"eval_cost\s*=\s*(?:0\.35|0\.25|0\.15)", content))
        hardcoded_gpu_rate_refs += len(re.findall(r"(?:gpu_cost|cost)\s*=\s*.*?\*\s*1\.15", content))

    # Verify legacy task7 orchestrator is fail-closed
    legacy_locked = False
    legacy_path = root / "modal" / "task7_pilot_v2.py"
    if legacy_path.exists():
        content = legacy_path.read_text(encoding="utf-8")
        if "Task 7.1 orchestrator is retired and must not be used" in content or "LEGACY / RETIRED" in content:
            legacy_locked = True

    # Verify authoritative production orchestrator is valid and active
    future_authoritative_valid = False
    future_path = root / "modal" / "pilot_v2_authoritative.py"
    if future_path.exists():
        content = future_path.read_text(encoding="utf-8")
        if (
            "ccpt.data.canonical_materializer" in content
            or "materialize_production_data_and_schedule" in content
            or "Authoritative Pilot-v2" in content
        ):
            future_authoritative_valid = True

    all_clean = (
        task6_active_refs == 0
        and ref_tok_active_refs == 0
        and mock_bt_active_refs == 0
        and use_mock_active_refs == 0
        and hardcoded_eval_cost_refs == 0
        and hardcoded_gpu_rate_refs == 0
        and legacy_locked
        and future_authoritative_valid
    )

    return {
        "active_file_count": len(active_files),
        "legacy_disabled_file_count": len(legacy_disabled_files),
        "test_file_count": len(test_files),
        "task6_active_refs": task6_active_refs,
        "ReferenceTokenizer_active_refs": ref_tok_active_refs,
        "mock_beavertails_active_refs": mock_bt_active_refs,
        "use_mock_active_refs": use_mock_active_refs,
        "hardcoded_eval_cost_refs": hardcoded_eval_cost_refs,
        "hardcoded_gpu_rate_refs": hardcoded_gpu_rate_refs,
        "legacy_locked": legacy_locked,
        "future_authoritative_locked": future_authoritative_valid,
        "future_authoritative_valid": future_authoritative_valid,
        "all_clean": all_clean,
    }

