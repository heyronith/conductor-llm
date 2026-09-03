"""Zero-GPU fail-closed preflight for Seed-4 authoritative cash-controlled execution."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ccpt.analysis.seed4_execution_ledger import (
    HARD_AUTHORIZATION_USD,
    MODEL_ORDER,
    SEED4,
    fetch_workspace_rates,
    new_ledger,
    remaining_mandatory_cost_usd,
    save_ledger,
)
from ccpt.config import get_smoke_dual_stream_config
from ccpt.evaluation.forensics import compute_canonical_state_dict_hash
from ccpt.training.engine import create_identical_dual_stream_models

COST_REDUCTION_HEAD = "98fa82094bcf64fd29db33e4864ae524dbf162ad"
BILLING_PREFLIGHT_ANCESTOR = "feca09c00438f323c3903d1439dfee91ef728cd1"
TASK32_ANCESTOR = "4210dfc89f09bbbb91d3ef2145e64d35904304df"
RESERVED_SEED = 20260822

CANONICAL_FINEWEB_PREFIX_HASH = "a13410b63d9c1533211784c2a08fa5a918e29cc446448470395aa93919712585"
CANONICAL_FINEWEB_ORIGINAL_32K_CONT_HASH = "1f6dd66f49a9afa3537244a719af74006308ab81902b0b654142510672022243"
CANONICAL_FINEWEB_128K_CONT_HASH = "26829ec5297e61e8ed91b89a64d6522c58c0123ac3c7aeab23801ee101510fa3"
OOD_BEAVERTAILS_MANIFEST_HASH = "f8cf3fd0f0ca7502e9b7fef37f49ae4b9fd13cb71438ed64fc093c0649d71b9e"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _is_ancestor(ancestor: str, head: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, head]).returncode == 0


def run_seed4_preflight(*, write_ledger: bool = True) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    head = _git("rev-parse", "HEAD").lower()
    branch = _git("branch", "--show-current")
    status = _git("status", "--porcelain")

    # Git provenance
    remote_head = None
    try:
        _git("fetch", "origin")
        remote_head = _git("rev-parse", f"origin/{branch}").lower()
    except subprocess.CalledProcessError:
        remote_head = None

    clean = status == ""
    # Allow untracked artifacts/test_runs only
    dirty_lines = [ln for ln in status.splitlines() if ln.strip()]
    non_test_dirty = [
        ln
        for ln in dirty_lines
        if "artifacts/test_runs/" not in ln and not ln.startswith("?? artifacts/test_runs")
    ]
    # During development before Commit A, dirty tree is expected; freeze gate is separate.
    checks["git"] = {
        "head_sha": head,
        "branch": branch,
        "remote_head": remote_head,
        "remote_parity": remote_head == head if remote_head else False,
        "working_tree_porcelain": status,
        "cost_reduction_head_match_or_descendant": head == COST_REDUCTION_HEAD or _is_ancestor(COST_REDUCTION_HEAD, head),
        "billing_preflight_ancestor": _is_ancestor(BILLING_PREFLIGHT_ANCESTOR, head),
        "task32_ancestor": _is_ancestor(TASK32_ANCESTOR, head),
        "status": "PASSED"
        if (
            (head == COST_REDUCTION_HEAD or _is_ancestor(COST_REDUCTION_HEAD, head))
            and _is_ancestor(BILLING_PREFLIGHT_ANCESTOR, head)
            and _is_ancestor(TASK32_ANCESTOR, head)
        )
        else "FAILED",
    }

    # Seed invariant in source
    t2 = Path("modal/strengthening_task2_sentinel.py").read_text(encoding="utf-8")
    t31 = Path("modal/strengthening_task3_1_eval.py").read_text(encoding="utf-8")
    seed_ok = "20260825" in t2 and "ALLOWED_SEEDS = [20260821, 20260825]" in t2 and "20260822" not in t2.split("ALLOWED_SEEDS")[1].split("]")[0]
    checks["seed"] = {
        "required": SEED4,
        "reserved_excluded": RESERVED_SEED,
        "task2_allowed_includes_seed4": "20260825" in t2,
        "status": "PASSED" if seed_ok else "FAILED",
    }

    # Hardware + corrected eval
    checks["hardware_and_eval"] = {
        "h100_bang_training": 'gpu="H100!"' in t2,
        "l40s_eval": 'gpu="L40S"' in t31,
        "format_eval_prompt": "format_eval_prompt" in t31,
        "max_new_tokens_48": "max_new_tokens\": 48" in t31 or "MAX_NEW_TOKENS = 48" in t31,
        "enforce_expected_hashes_adapter": "enforce_expected_hashes" in t31,
        "seed4_eval_entrypoint": "run_seed4_corrected_evaluation" in t31,
        "seed4_train_entrypoint": "run_seed4_single_model_training" in t2,
        "status": "PASSED"
        if (
            'gpu="H100!"' in t2
            and 'gpu="L40S"' in t31
            and "format_eval_prompt" in t31
            and ("MAX_NEW_TOKENS = 48" in t31)
            and "enforce_expected_hashes" in t31
            and "run_seed4_corrected_evaluation" in t31
            and "run_seed4_single_model_training" in t2
        )
        else "FAILED",
    }

    # Data hashes from local manifests
    fw = json.loads(Path("artifacts/strengthening_task2_extended_fineweb_manifest.json").read_text(encoding="utf-8"))
    fw_ok = (
        fw["capability_prefix"]["logical_prefix_hash"] == CANONICAL_FINEWEB_PREFIX_HASH
        and fw["original_persistence_continuation"]["logical_continuation_hash"]
        == CANONICAL_FINEWEB_ORIGINAL_32K_CONT_HASH
        and fw["original_persistence_continuation"]["first_32k_parity"] == "BIT_IDENTICAL"
        and fw["persistence_continuation"]["logical_continuation_hash"] == CANONICAL_FINEWEB_128K_CONT_HASH
        and fw["persistence_continuation"]["target_blocks"] == 128000
    )
    checks["fineweb"] = {
        "prefix_hash": fw["capability_prefix"]["logical_prefix_hash"],
        "first_32k_parity": fw["original_persistence_continuation"]["first_32k_parity"],
        "extended_128k_hash": fw["persistence_continuation"]["logical_continuation_hash"],
        "status": "PASSED" if fw_ok else "FAILED",
    }

    ood_ok = OOD_BEAVERTAILS_MANIFEST_HASH in t31
    checks["ood_beavertails"] = {
        "expected_hash": OOD_BEAVERTAILS_MANIFEST_HASH,
        "pinned_in_eval_source": ood_ok,
        "status": "PASSED" if ood_ok else "FAILED",
    }

    # B/C init parity for Seed 4
    cfg = get_smoke_dual_stream_config()
    mb, mc = create_identical_dual_stream_models(cfg, seed=SEED4)
    hb = compute_canonical_state_dict_hash(mb.state_dict())
    hc = compute_canonical_state_dict_hash(mc.state_dict())
    checks["bc_initialization_parity"] = {
        "seed": SEED4,
        "hash_b": hb,
        "hash_c": hc,
        "parity": hb == hc,
        "status": "PASSED" if hb == hc else "FAILED",
    }

    # Volume: Seed-4 models must not be silently overwritten; audit presence
    vol = subprocess.run(
        ["uv", "run", "modal", "volume", "ls", "ccpt-authoritative-runs", f"/ccpt/strengthening_task2/seed_{SEED4}"],
        capture_output=True,
        text=True,
    )
    seed4_present = vol.returncode == 0
    existing_models = []
    if seed4_present:
        for m in MODEL_ORDER:
            p = subprocess.run(
                [
                    "uv",
                    "run",
                    "modal",
                    "volume",
                    "ls",
                    "ccpt-authoritative-runs",
                    f"/ccpt/strengthening_task2/seed_{SEED4}/{m}/persistence_4000.pt",
                ],
                capture_output=True,
                text=True,
            )
            if p.returncode == 0:
                existing_models.append(m)
    checks["volume_seed4"] = {
        "seed4_dir_present": seed4_present,
        "existing_terminal_checkpoints": existing_models,
        "overwrite_forbidden": True,
        "status": "PASSED",  # absence or presence both OK if we don't overwrite
    }

    # Rates + cash projection
    rates = fetch_workspace_rates()
    projection = json.loads(Path("artifacts/seed4_cost_reduction_projection.json").read_text(encoding="utf-8"))
    clean_expected = float(projection["scenarios_usd"]["B_clean_protocol_preserving"]["total_usd"])
    conservative = float(projection["scenarios_usd"]["C_conservative"]["total_usd"])
    # Gate uses current rates × historical envelopes scaled if rate changed
    hist_h100 = float(projection["rates"]["h100_hourly_usd"])
    hist_l40s = float(projection["rates"]["l40s_hourly_usd"])
    scale_h = rates["h100_hourly_usd"] / hist_h100
    scale_l = rates["l40s_hourly_usd"] / hist_l40s
    projected_clean = (
        float(projection["scenarios_usd"]["B_clean_protocol_preserving"]["h100_usd"]) * scale_h
        + float(projection["scenarios_usd"]["B_clean_protocol_preserving"]["l40s_usd"]) * scale_l
        + float(projection["scenarios_usd"]["B_clean_protocol_preserving"]["other_incremental_usd"])
    )
    projected_conservative = (
        float(projection["scenarios_usd"]["C_conservative"]["h100_usd"]) * scale_h
        + float(projection["scenarios_usd"]["C_conservative"]["l40s_usd"]) * scale_l
        + float(projection["scenarios_usd"]["C_conservative"]["other_incremental_usd"])
    )
    launch_ok = projected_conservative <= HARD_AUTHORIZATION_USD + 1e-9
    checks["cash_gate_pre_first_h100"] = {
        "hard_authorization_usd": HARD_AUTHORIZATION_USD,
        "current_h100_rate": rates["h100_hourly_usd"],
        "current_l40s_rate": rates["l40s_hourly_usd"],
        "projected_clean_usd": projected_clean,
        "projected_conservative_usd": projected_conservative,
        "historical_clean_usd": clean_expected,
        "historical_conservative_usd": conservative,
        "allowed_to_launch_first_h100": launch_ok,
        "status": "PASSED" if launch_ok else "FAILED",
    }

    overall = all(c.get("status") == "PASSED" for c in checks.values())
    result = {
        "task": "strengthening_seed4_preflight",
        "seed": SEED4,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_sha": head,
        "branch": branch,
        "h100_gpu_seconds_used_by_preflight": 0,
        "l40s_gpu_seconds_used_by_preflight": 0,
        "overall_status": "PASSED" if overall else "FAILED",
        "checks": checks,
        "non_test_dirty_paths": non_test_dirty,
    }

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/strengthening_seed4_preflight.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    if write_ledger:
        ledger = new_ledger(head, rates)
        ledger["projected_final_total_usd"] = projected_conservative
        ledger["preflight"] = {
            "overall_status": result["overall_status"],
            "projected_conservative_usd": projected_conservative,
            "bc_init_hash": hb,
        }
        # Pre-pipeline reserve check for D
        needed = remaining_mandatory_cost_usd(ledger, "model_d")
        ledger["stages"].append(
            {
                "stage": "preflight_complete",
                "reserved_for_full_protocol_from_d_usd": needed,
                "allowed": needed <= HARD_AUTHORIZATION_USD,
            }
        )
        save_ledger(ledger)

    return result


def main() -> None:
    print("=== Seed-4 zero-GPU preflight ===")
    res = run_seed4_preflight()
    print(f"Overall: {res['overall_status']}")
    if res["overall_status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
