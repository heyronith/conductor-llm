"""Seed 4 cost-reduction forensic audit: attribute Seed-1 H100 spend and plan clean execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

REMAINING_MODAL_CREDITS_USD = 0.0
H100_GPU_SECONDS_AUTHORIZED = 0
L40S_GPU_SECONDS_AUTHORIZED = 0
SEED4 = 20260825
SEED1 = 20260821
TASK2_APP = "strengthening-task2-sentinel"

CLASSIFICATIONS = (
    "REQUIRED_SUCCESSFUL_SCIENTIFIC_WORK",
    "RETRY_AFTER_INFRA_FAILURE",
    "FAILED_OR_ABORTED_EXECUTION",
    "DUPLICATE_EXECUTION",
    "GPU_SMOKE_OR_PREFLIGHT",
    "GPU_IDLE_OR_SETUP",
    "VALID_BUT_NOT_SEED4_REQUIRED",
    "UNRESOLVED",
)


def repo_root(start: Path | None = None) -> Path:
    return start or Path(__file__).resolve().parents[3]


def git_head_sha(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _d(value: Any) -> Decimal:
    return Decimal(str(value))


def _f(value: Decimal | float) -> float:
    return float(value)


def load_rates(root: Path) -> dict[str, Any]:
    path = root / "artifacts" / "seed4_cost_preflight_modal_rates.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_billing_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def implied_seconds(cost_usd: Decimal, hourly_rate: Decimal) -> float:
    if hourly_rate <= 0:
        return 0.0
    return float(cost_usd / hourly_rate * Decimal("3600"))


def build_task2_object_ledger(root: Path) -> dict[str, Any]:
    rates = load_rates(root)
    h100_rate = _d(rates["workspace_h100_hourly_usd"])
    window_path = root / "artifacts" / "seed4_cost_preflight" / "relevant_workspace_window_raw.json"
    sep1_path = root / "artifacts" / "seed4_cost_preflight" / "task2_seed1_billing_raw.json"
    rows = load_billing_rows(window_path)
    sep1_rows = load_billing_rows(sep1_path)

    objects: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("description") != TASK2_APP:
            continue
        oid = row["object_id"]
        if oid not in objects:
            objects[oid] = {
                "object_id": oid,
                "app_description": TASK2_APP,
                "environment": row.get("environment"),
                "h100_cost_usd": Decimal("0"),
                "l40s_cost_usd": Decimal("0"),
                "cpu_cost_usd": Decimal("0"),
                "memory_cost_usd": Decimal("0"),
                "intervals": set(),
                "resources_seen": set(),
            }
        cost = _d(row["cost"])
        res = row["resource"]
        objects[oid]["resources_seen"].add(res)
        objects[oid]["intervals"].add(row["interval_start"])
        key = {
            "H100": "h100_cost_usd",
            "L40S": "l40s_cost_usd",
            "CPU": "cpu_cost_usd",
            "Memory": "memory_cost_usd",
        }.get(res)
        if key:
            objects[oid][key] += cost

    sep1_h100_by_oid: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for row in sep1_rows:
        if row.get("description") != TASK2_APP or row["resource"] != "H100":
            continue
        sep1_h100_by_oid[row["object_id"]] += _d(row["cost"])

    # Checkpoint timestamps (volume listing evidence captured in this audit)
    checkpoint_timeline = {
        "timezone": "America/Chicago (CDT as reported by modal volume ls)",
        "source": "uv run modal volume ls --json ccpt-authoritative-runs /ccpt/strengthening_task2/seed_20260821/{model}/",
        "models": {
            "model_b": {
                "lm_1b_final.pt": "2026-08-31 20:37 CDT",
                "safety_20m_final.pt": "2026-08-31 20:46 CDT",
                "persistence_4000.pt": "2026-08-31 21:02 CDT",
                "responses.jsonl": "2026-08-31 21:37 CDT",
            },
            "model_c": {
                "lm_1b_final.pt": "2026-08-31 20:25 CDT",
                "safety_20m_final.pt": "2026-08-31 20:45 CDT",
                "persistence_4000.pt": "2026-08-31 21:01 CDT",
                "responses.jsonl": "2026-08-31 21:43 CDT",
            },
            "model_d": {
                "lm_1b_final.pt": "2026-08-31 20:25 CDT",
                "safety_20m_final.pt": "2026-08-31 20:45 CDT",
                "persistence_4000.pt": "2026-08-31 21:01 CDT",
                "responses.jsonl": "2026-08-31 21:46 CDT",
            },
        },
        "authoritative_training_window_cdt": "approx 2026-08-31 18:45–21:02 CDT",
        "authoritative_training_window_utc": "approx 2026-08-31 23:45 UTC – 2026-09-01 02:02 UTC",
        "final_orchestrator_utc": "2026-09-01T03:51:20+00:00 (sentinel summary; post-checkpoint fast-return)",
    }

    classifications: dict[str, dict[str, Any]] = {
        "ap-TaUUJJEc7NPvKK0oya8ClI": {
            "primary_classification": "REQUIRED_SUCCESSFUL_SCIENTIFIC_WORK",
            "confidence": "HIGH",
            "avoidable_in_seed4": False,
            "avoided_cost_usd_if_avoidable": 0.0,
            "model_identity": "aggregate_B_C_D_concurrent_unknown_split",
            "evidence": [
                "Largest H100 object ($20.74); spans Aug31–Sep1 UTC.",
                "Surviving Seed-1 lm/safety/persistence checkpoints were written 20:25–21:02 CDT Aug 31, which maps into Sep1 UTC and the overnight edge of Aug31 UTC.",
                "Aug31 UTC H100 on this object ($3.0349 ≈ 2766s) matches ~15 minutes of 3×H100 concurrent work before 00:00 UTC Sep1 — consistent with successful LM start crossing midnight, not a discarded failed app.",
                "No Seed-4 path exists; this spend produced Seed-1 authoritative state only.",
            ],
            "notes": "Treat full object as irreducible successful concurrent B/C/D training envelope.",
        },
        "ap-7BwZvJXfXhNf34YmxJZQ9T": {
            "primary_classification": "FAILED_OR_ABORTED_EXECUTION",
            "confidence": "MEDIUM",
            "avoidable_in_seed4": True,
            "avoided_cost_usd_if_avoidable": None,  # filled below from ledger
            "model_identity": "unknown",
            "evidence": [
                "H100-only object ($2.936) on Sep1 UTC with no L40S.",
                "No surviving checkpoint timestamps uniquely bind to this object ID.",
                "Authoritative checkpoints already explained by ap-TaUU concurrent envelope.",
                "Consistent with a separate modal run / aborted training launch.",
            ],
            "notes": "MEDIUM confidence — excluded from HIGH proven savings.",
        },
        "ap-JLcljKGGok1GtmlCRigRdw": {
            "primary_classification": "FAILED_OR_ABORTED_EXECUTION",
            "confidence": "MEDIUM",
            "avoidable_in_seed4": True,
            "avoided_cost_usd_if_avoidable": None,
            "model_identity": "unknown",
            "evidence": [
                "H100-only object ($1.060) on Sep1 UTC.",
                "No checkpoint/app-log binding; excess beyond TaUU successful envelope.",
            ],
            "notes": "MEDIUM confidence — excluded from HIGH proven savings.",
        },
        "ap-ELJZiCTkakrIWg1bI0qkFM": {
            "primary_classification": "DUPLICATE_EXECUTION",
            "confidence": "HIGH",
            "avoidable_in_seed4": True,
            "avoided_cost_usd_if_avoidable": None,
            "model_identity": "unknown_mixed_h100_l40s",
            "evidence": [
                "Mixed H100 ($0.316) + L40S ($0.204) on Sep1 UTC after authoritative checkpoints existed.",
                "Final sentinel orchestrator fast-returned training when persistence_4000.pt existed; residual H100 allocation is duplicate relative to completed scientific state.",
            ],
            "notes": "H100 portion avoidable; L40S portion part of invalid Task-2 eval spend.",
        },
        "ap-cKSKLaV2osBZ75RHNo1zb0": {
            "primary_classification": "VALID_BUT_NOT_SEED4_REQUIRED",
            "confidence": "HIGH",
            "avoidable_in_seed4": True,
            "avoided_cost_usd_if_avoidable": None,
            "model_identity": "n/a_evaluation",
            "evidence": [
                "Dominated by L40S ($2.363) with small H100 ($0.305).",
                "Present in post-checkpoint orchestrator hourly window 03:00–05:00 UTC Sep1.",
                "Task-2 behavioral evaluation used unframed prompts (Task-3 root cause); Seed-4 must use corrected framed eval instead.",
            ],
            "notes": "L40S is valid Task-2 work but not required for Seed-4 corrected protocol.",
        },
        "ap-5nKMP2j3evWrTZGWNFgdqd": {
            "primary_classification": "DUPLICATE_EXECUTION",
            "confidence": "HIGH",
            "avoidable_in_seed4": True,
            "avoided_cost_usd_if_avoidable": None,
            "model_identity": "unknown_mixed_h100_l40s",
            "evidence": [
                "Hourly billing 03:00 UTC Sep1 shows H100+L40S during final sentinel orchestrator after checkpoints completed.",
                "Training path would fast-return; H100 seconds are duplicate/idle relative to scientific need.",
            ],
            "notes": "H100 portion HIGH avoidable; L40S tied to invalid Task-2 eval.",
        },
        "ap-kjZzGjjT5GmvBCeG6ShnSs": {
            "primary_classification": "GPU_IDLE_OR_SETUP",
            "confidence": "HIGH",
            "avoidable_in_seed4": True,
            "avoided_cost_usd_if_avoidable": None,
            "model_identity": "unknown",
            "evidence": [
                "Small H100 ($0.184) with tiny L40S on Sep1 UTC; no unique checkpoint binding.",
                "Scale consistent with short-lived GPU allocation / setup rather than a full phase.",
            ],
            "notes": "HIGH avoidable for Seed-4 clean launch discipline.",
        },
    }

    ledger_objects = []
    total_h100 = Decimal("0")
    total_sep1_h100 = Decimal("0")
    by_class_h100: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    proven_avoidable_h100 = Decimal("0")
    medium_avoidable_h100 = Decimal("0")
    proven_avoidable_l40s = Decimal("0")

    for oid, raw in sorted(objects.items(), key=lambda item: -item[1]["h100_cost_usd"]):
        intervals = sorted(raw["intervals"])
        h100 = raw["h100_cost_usd"]
        if h100 == 0 and raw["l40s_cost_usd"] == 0 and raw["cpu_cost_usd"] == 0:
            continue
        total_h100 += h100
        sep1_h100 = sep1_h100_by_oid.get(oid, Decimal("0"))
        total_sep1_h100 += sep1_h100
        clf = classifications.get(
            oid,
            {
                "primary_classification": "UNRESOLVED",
                "confidence": "LOW",
                "avoidable_in_seed4": False,
                "avoided_cost_usd_if_avoidable": 0.0,
                "model_identity": "unknown",
                "evidence": ["Object not covered by specialized attribution rules."],
                "notes": "",
            },
        )
        # Fill avoided cost: for HIGH avoidable, count H100; for VALID_BUT_NOT_SEED4_REQUIRED count L40S+H100 carefully
        avoidable_h100 = Decimal("0")
        avoidable_l40s = Decimal("0")
        if clf["avoidable_in_seed4"] and clf["confidence"] == "HIGH":
            if clf["primary_classification"] == "VALID_BUT_NOT_SEED4_REQUIRED":
                avoidable_l40s = raw["l40s_cost_usd"]
                avoidable_h100 = h100  # residual H100 on eval app
            else:
                avoidable_h100 = h100
                # L40S on duplicate/idle mixed apps is also Task-2 eval waste
                if raw["l40s_cost_usd"] > 0:
                    avoidable_l40s = raw["l40s_cost_usd"]
            proven_avoidable_h100 += avoidable_h100
            proven_avoidable_l40s += avoidable_l40s
        elif clf["avoidable_in_seed4"] and clf["confidence"] == "MEDIUM":
            medium_avoidable_h100 += h100

        clf_out = dict(clf)
        clf_out["avoided_cost_usd_if_avoidable"] = _f(avoidable_h100 + avoidable_l40s)

        by_class_h100[clf["primary_classification"]] += h100
        entry = {
            "object_id": oid,
            "app_description": TASK2_APP,
            "environment": raw["environment"],
            "billing_intervals": intervals,
            "first_timestamp": intervals[0] if intervals else None,
            "last_timestamp": intervals[-1] if intervals else None,
            "h100_cost_usd": _f(h100),
            "h100_seconds_implied": implied_seconds(h100, h100_rate),
            "sep1_utc_h100_cost_usd": _f(sep1_h100),
            "l40s_cost_usd": _f(raw["l40s_cost_usd"]),
            "cpu_cost_usd": _f(raw["cpu_cost_usd"]),
            "memory_cost_usd": _f(raw["memory_cost_usd"]),
            "total_cost_usd": _f(
                h100 + raw["l40s_cost_usd"] + raw["cpu_cost_usd"] + raw["memory_cost_usd"]
            ),
            "classification": clf_out,
        }
        ledger_objects.append(entry)

    expected_full = _d("25.83683913")
    expected_sep1 = _d("22.80192289")
    # Recompute exact from machine rows for reconciliation
    recon_full_ok = abs(total_h100 - expected_full) < Decimal("0.0000001")
    recon_sep1_ok = abs(total_sep1_h100 - expected_sep1) < Decimal("0.0000001")

    return {
        "task": "seed4_cost_reduction_task2_object_ledger",
        "remaining_modal_credits_usd": REMAINING_MODAL_CREDITS_USD,
        "workspace_h100_hourly_usd": _f(h100_rate),
        "source_billing_files": [
            str(window_path.relative_to(root)),
            str(sep1_path.relative_to(root)),
        ],
        "checkpoint_timeline": checkpoint_timeline,
        "objects": ledger_objects,
        "totals": {
            "full_window_h100_cost_usd": _f(total_h100),
            "sep1_utc_h100_cost_usd": _f(total_sep1_h100),
            "h100_by_classification_usd": {k: _f(v) for k, v in sorted(by_class_h100.items())},
            "proven_high_confidence_avoidable_h100_usd": _f(proven_avoidable_h100),
            "medium_confidence_avoidable_h100_usd": _f(medium_avoidable_h100),
            "proven_high_confidence_avoidable_l40s_usd": _f(proven_avoidable_l40s),
        },
        "reconciliation": {
            "expected_full_window_h100_usd": _f(expected_full),
            "observed_full_window_h100_usd": _f(total_h100),
            "full_window_match": recon_full_ok,
            "expected_sep1_h100_usd": _f(expected_sep1),
            "observed_sep1_h100_usd": _f(total_sep1_h100),
            "sep1_match": recon_sep1_ok,
            "ledger_reconciliation": "PASS" if (recon_full_ok and recon_sep1_ok) else "FAIL",
            "every_h100_object_classified": all(
                obj["classification"]["primary_classification"] in CLASSIFICATIONS for obj in ledger_objects
            ),
            "unresolved_h100_usd": _f(by_class_h100.get("UNRESOLVED", Decimal("0"))),
        },
        "hypothesis_5plus_avoidable": {
            "statement": "Non-TaUU H100 (~$5.10) is avoidable infrastructure waste",
            "non_tauu_h100_usd": _f(total_h100 - objects["ap-TaUUJJEc7NPvKK0oya8ClI"]["h100_cost_usd"]),
            "verdict": "PARTIALLY_SUPPORTED",
            "explanation": (
                "HIGH-confidence avoidable H100 among non-TaUU objects is ~$1.10 "
                "(duplicate/idle residual on mixed eval apps). "
                "Remaining ~$4.00 (ap-7Bw + ap-JL) is MEDIUM-confidence failed/aborted "
                "and is NOT subtracted from the Seed-4 expected budget."
            ),
        },
    }


def build_checkpoint_reuse_audit(root: Path) -> dict[str, Any]:
    """CPU-only Modal volume inspection results for Seed 4."""
    # Seed-4 directory absence was verified via: modal volume ls .../seed_20260825 → No such file
    seed4_listing = {
        "volume": "ccpt-authoritative-runs",
        "seed4_path": f"/ccpt/strengthening_task2/seed_{SEED4}",
        "exists": False,
        "listing_error": "No such file or directory",
        "inspection_method": "uv run modal volume ls (CPU/CLI only; no GPU allocated)",
    }
    models = {}
    for model in ("model_b", "model_c", "model_d"):
        models[model] = {
            "path": f"/ccpt/strengthening_task2/seed_{SEED4}/{model}",
            "status": "NOT_PRESENT",
            "classification": "NOT_PRESENT",
            "reusable": False,
            "estimated_savings_usd": 0.0,
        }
    return {
        "task": "seed4_cost_reduction_checkpoint_reuse_audit",
        "seed": SEED4,
        "volume_inspection": seed4_listing,
        "models": models,
        "any_valid_reusable_seed4_checkpoint": False,
        "estimated_savings_from_existing_valid_state_usd": 0.0,
        "seed1_authoritative_state_present": True,
        "seed1_note": "Seed-1 B/C/D checkpoints exist and are complete; they must not be reused for Seed-4 scientific identity.",
    }


def build_waste_analysis(ledger: dict[str, Any]) -> dict[str, Any]:
    observed = [
        {
            "issue": "multiple_modal_app_deployments_for_same_task2_app_name",
            "status": "OBSERVED",
            "evidence": "Seven distinct ap-* object IDs billed under strengthening-task2-sentinel for Seed-1 window.",
            "seed4_mitigation": "Single controlled launch; fail-closed identity guard; no parallel redeploys.",
            "proven_dollar_savings": "partial — MEDIUM objects not counted in expected",
        },
        {
            "issue": "invalid_unframed_task2_evaluation_l40s",
            "status": "OBSERVED",
            "evidence": "Task-3 forensic root cause + Task-2 L40S billed $3.11 under sentinel app.",
            "seed4_mitigation": "Use corrected framed Task-3.1-style evaluation only.",
            "proven_dollar_savings": ledger["totals"]["proven_high_confidence_avoidable_l40s_usd"],
        },
        {
            "issue": "post_checkpoint_h100_allocation_during_fast_return_orchestrator",
            "status": "OBSERVED",
            "evidence": "Checkpoints completed ~02:02 UTC Sep1; orchestrator at 03:51 UTC still billed residual H100 on mixed apps.",
            "seed4_mitigation": "Do not spawn H100 training functions when authoritative persistence_4000 exists; CPU preflight first.",
            "proven_dollar_savings": "included in HIGH avoidable H100",
        },
        {
            "issue": "data_manifest_loads_inside_h100_function",
            "status": "OBSERVED",
            "evidence": "modal/strengthening_task2_sentinel.py loads FineWeb/WildGuard manifests after H100 allocation.",
            "seed4_mitigation": "CPU pre-stage/validate manifests before H100 spawn.",
            "proven_dollar_savings": "not quantified — POSSIBLE only",
        },
        {
            "issue": "synchronous_volume_commit_after_checkpoints",
            "status": "OBSERVED",
            "evidence": "runs_volume.commit() after LM/safety/persistence saves inside H100 function.",
            "seed4_mitigation": "Keep commits; consider reducing frequency only if scientifically transparent. No proven savings claimed.",
            "proven_dollar_savings": 0.0,
        },
    ]
    possible = [
        {
            "issue": "torch_compile_or_cuda_graphs",
            "status": "POSSIBLE",
            "note": "Requires separate parity benchmark; excluded from expected budget.",
        },
        {
            "issue": "async_checkpoint_writes",
            "status": "POSSIBLE",
            "note": "May alter crash-consistency; not claimed as Seed-4 savings.",
        },
    ]
    return {
        "task": "seed4_cost_reduction_waste_analysis",
        "observed_issues": observed,
        "possible_issues": possible,
        "guaranteed_high_confidence_savings_usd": {
            "h100": ledger["totals"]["proven_high_confidence_avoidable_h100_usd"],
            "l40s_task2_invalid_eval": ledger["totals"]["proven_high_confidence_avoidable_l40s_usd"],
            "total": round(
                ledger["totals"]["proven_high_confidence_avoidable_h100_usd"]
                + ledger["totals"]["proven_high_confidence_avoidable_l40s_usd"],
                4,
            ),
        },
        "medium_confidence_not_in_expected_budget_usd": {
            "h100": ledger["totals"]["medium_confidence_avoidable_h100_usd"],
        },
    }


def build_execution_plan(ledger: dict[str, Any], reuse: dict[str, Any]) -> dict[str, Any]:
    tauu = next(o for o in ledger["objects"] if o["object_id"] == "ap-TaUUJJEc7NPvKK0oya8ClI")
    return {
        "task": "seed4_cost_reduction_execution_plan",
        "seed": SEED4,
        "remaining_modal_credits_usd": REMAINING_MODAL_CREDITS_USD,
        "out_of_pocket_equals_metered": True,
        "scientific_invariants": {
            "seed": SEED4,
            "models_required": ["model_b", "model_c", "model_d"],
            "lm_1b_required": True,
            "safety_tokens_required": 20_010_611,
            "persistence_steps_required": [0, 250, 1000, 4000],
            "h100_required": True,
            "precision_unchanged": True,
            "corrected_framed_evaluation_required": True,
            "scientific_semantics_changed": False,
        },
        "checkpoint_reuse": {
            "any_valid_reusable": reuse["any_valid_reusable_seed4_checkpoint"],
            "savings_usd": 0.0,
        },
        "protocol_preserving_optimizations": [
            {
                "name": "cpu_preflight_before_h100",
                "evidence": "Manifest/tokenizer/path checks currently can occur after GPU allocation.",
                "expected_savings": "reduces GPU_IDLE_OR_SETUP; not separately dollarized beyond proven residual H100",
                "scientific_semantics_changed": False,
            },
            {
                "name": "single_controlled_deployment",
                "evidence": "Seven ap-* IDs under same app name in Seed-1 window.",
                "expected_savings": "avoids MEDIUM failed/aborted launches if discipline holds",
                "scientific_semantics_changed": False,
            },
            {
                "name": "no_h100_spawn_when_persistence_4000_exists",
                "evidence": "Fast-return path still incurred residual H100 in final orchestrator.",
                "expected_savings": f"${ledger['totals']['proven_high_confidence_avoidable_h100_usd']:.2f} HIGH proven H100 class",
                "scientific_semantics_changed": False,
            },
            {
                "name": "corrected_framed_eval_only",
                "evidence": "Task-2 L40S invalid framing billed; Task-3.1 corrected eval ~$1.90 billed.",
                "expected_savings": "avoid repeating Task-2 L40S $3.11; pay corrected ~$1.90 instead",
                "scientific_semantics_changed": False,
            },
            {
                "name": "sequential_model_pipelines_with_cash_gate",
                "evidence": "Concurrent 3×H100 makes mid-flight cash ceiling hard to enforce.",
                "expected_savings": "financial risk control; may increase wall clock, not GPU-seconds",
                "scientific_semantics_changed": False,
                "recommended_order": ["model_d", "model_b", "model_c"],
                "order_rationale": (
                    "Technical/cost only: run one full pipeline first to measure realized H100 seconds "
                    "before committing remaining cash; D first as independent architecture (no B/C init coupling)."
                ),
            },
            {
                "name": "reuse_materialized_data_and_image_layers",
                "evidence": "FineWeb/WildGuard already on volumes; prior shortcut audit ALLOWED.",
                "expected_savings": "cold-start/setup only",
                "scientific_semantics_changed": False,
            },
        ],
        "live_budget_monitor": {
            "cash_remaining_authorization": "hard_authorization_usd",
            "before_each_model_pipeline": [
                "estimate full-pipeline H100 envelope from Seed-1 TaUU aggregate / 3",
                "verify remaining authorization covers that envelope + remaining eval share",
                "otherwise do not start the pipeline",
            ],
            "during_pipeline": [
                "track H100 wall seconds",
                "accrue estimated dollars using workspace H100 rate",
                "record phase/model/checkpoint completed",
            ],
            "do_not_stop_healthy_mid_pipeline_for_estimate_miss": True,
            "fail_closed_on_hard_authorization_breach": True,
            "after_each_model": [
                "refresh Modal billing if available",
                "recompute remaining authorization",
                "decide next model launch",
            ],
            "per_model_h100_envelope_usd": round(tauu["h100_cost_usd"] / 3.0, 4),
            "irreducible_aggregate_h100_usd": tauu["h100_cost_usd"],
        },
        "forbidden_shortcuts": [
            "shared_b_c_trained_trunk",
            "precision_change",
            "gpu_type_change",
            "token_budget_reduction",
            "persistence_1000_early_stop",
            "drop_model_b",
            "eval_reduction",
            "invalid_unframed_eval",
        ],
    }


def build_runtime_optimization_review() -> dict[str, Any]:
    return {
        "task": "seed4_cost_reduction_runtime_optimization_review",
        "cuda_graphs": {
            "classification": "REJECT_FOR_SEED4",
            "rationale": (
                "Training loop uses dynamic Python control, per-step scheduler mutation, "
                "clip_grad_norm_, and model-type branches; CUDA Graphs risk changing execution "
                "order/RNG/reproducibility without a parity gate."
            ),
            "scientific_risk": "HIGH",
        },
        "torch_compile": {
            "classification": "SAFE_CANDIDATE_FOR_SEPARATE_PARITY_BENCHMARK",
            "rationale": (
                "Could be tested on a short fixed-step H100 benchmark with identical seed, "
                "batches, optimizer state, and precision; not authorized for Seed-4 without parity."
            ),
            "scientific_risk": "MEDIUM",
        },
        "prefetch_staging": {
            "classification": "SAFE_CANDIDATE_FOR_SEPARATE_PARITY_BENCHMARK",
            "rationale": (
                "CPU pre-stage of immutable manifests/shards before H100 allocation is "
                "scientifically transparent if token order and reader cursors remain identical."
            ),
            "scientific_risk": "LOW",
        },
        "paid_parity_benchmark_worthwhile": True,
        "maximum_proposed_benchmark_cost_usd": 0.30,
        "benchmark_design": {
            "execute_now": False,
            "hardware": "H100!",
            "compare": ["eager_path", "candidate_optimized_path"],
            "requirements": [
                "same seed",
                "same input batches",
                "same initial checkpoint",
                "same optimizer state",
                "same precision",
                "short fixed step count",
                "compare per-step loss, gradients where practical, model/optimizer/RNG state, next-step continuation",
            ],
        },
    }


def build_projection(
    root: Path,
    ledger: dict[str, Any],
    waste: dict[str, Any],
    rates: dict[str, Any],
) -> dict[str, Any]:
    preflight_actuals = json.loads(
        (root / "artifacts" / "seed4_cost_preflight_actuals.json").read_text(encoding="utf-8")
    )
    tauu_h100 = next(o["h100_cost_usd"] for o in ledger["objects"] if o["object_id"] == "ap-TaUUJJEc7NPvKK0oya8ClI")
    historical_h100 = ledger["totals"]["full_window_h100_cost_usd"]
    task31_eval = preflight_actuals["task3_1_corrected_evaluation"]["total_billed_cost_usd"]
    task2_cpu_mem = (
        preflight_actuals["task2_seed1_training"]["cpu_billed_cost_usd"]
        + preflight_actuals["task2_seed1_training"]["memory_billed_cost_usd"]
    )
    # Clean CPU/mem: scale roughly with successful training share
    clean_other = round(task2_cpu_mem * (tauu_h100 / historical_h100), 4)

    proven_h100 = waste["guaranteed_high_confidence_savings_usd"]["h100"]
    proven_l40s = waste["guaranteed_high_confidence_savings_usd"]["l40s_task2_invalid_eval"]
    proven_total = waste["guaranteed_high_confidence_savings_usd"]["total"]

    historical = {
        "h100_usd": historical_h100,
        "l40s_usd": task31_eval,  # Seed-4 would use corrected eval, not invalid Task-2 L40S
        "other_incremental_usd": round(task2_cpu_mem, 4),
        "assumptions": [
            "Repeat Seed-1 H100 inefficiencies including MEDIUM failed/aborted apps.",
            "Use corrected framed evaluation cost (Task-3.1 billed), not invalid Task-2 L40S.",
        ],
    }
    historical["total_usd"] = round(
        historical["h100_usd"] + historical["l40s_usd"] + historical["other_incremental_usd"], 4
    )

    clean = {
        "h100_usd": tauu_h100,
        "l40s_usd": task31_eval,
        "other_incremental_usd": clean_other,
        "assumptions": [
            "Only ap-TaUU successful concurrent B/C/D H100 envelope.",
            "Exclude HIGH proven residual/duplicate H100 and MEDIUM failed apps.",
            "Corrected Task-3.1-style L40S evaluation.",
            "Speculative CUDA Graph/compile savings excluded.",
        ],
    }
    clean["total_usd"] = round(clean["h100_usd"] + clean["l40s_usd"] + clean["other_incremental_usd"], 4)

    conservative = {
        "h100_usd": round(tauu_h100 * 1.08, 4),
        "l40s_usd": round(task31_eval * 1.08, 4),
        "other_incremental_usd": round(clean_other * 1.15, 4),
        "assumptions": [
            "8% H100 runtime variance buffer on irreducible envelope.",
            "8% L40S buffer; 15% CPU/memory buffer.",
            "Does not include a second full training rerun.",
            "Does not include MEDIUM $4 failed-app risk as expected, but hard auth covers it.",
        ],
    }
    conservative["total_usd"] = round(
        conservative["h100_usd"] + conservative["l40s_usd"] + conservative["other_incremental_usd"], 4
    )

    medium_risk = ledger["totals"]["medium_confidence_avoidable_h100_usd"]
    recommended_hard = round(max(conservative["total_usd"] * 1.1, clean["total_usd"] + medium_risk), 2)

    irreducible_h100 = tauu_h100
    irreducible_full = round(tauu_h100 + task31_eval + clean_other, 4)

    return {
        "task": "seed4_cost_reduction_projection",
        "seed4": SEED4,
        "remaining_modal_credits_usd": REMAINING_MODAL_CREDITS_USD,
        "rates": {
            "h100_hourly_usd": rates["workspace_h100_hourly_usd"],
            "l40s_hourly_usd": rates["workspace_l40s_hourly_usd"],
        },
        "scenarios_usd": {
            "A_historical_repeat": historical,
            "B_clean_protocol_preserving": clean,
            "C_conservative": conservative,
        },
        "proven_avoidable_h100_cost_usd": proven_h100,
        "proven_avoidable_non_h100_cost_usd": proven_l40s,
        "proven_avoidable_total_cost_usd": proven_total,
        "new_expected_out_of_pocket_usd": clean["total_usd"],
        "new_conservative_out_of_pocket_usd": conservative["total_usd"],
        "recommended_hard_authorization_usd": recommended_hard,
        "scientifically_irreducible_h100_cost_floor_usd": irreducible_h100,
        "scientifically_irreducible_full_protocol_cost_floor_usd": irreducible_full,
        "irreducible_evidence_quality": "HIGH_AGGREGATE_MEDIUM_PER_MODEL",
        "irreducible_uncertainty": (
            "Per-model H100 split unavailable from Modal billing; floor is aggregate successful "
            "app object ap-TaUUJJEc7NPvKK0oya8ClI plus corrected eval. Going materially below "
            "this would require changing the experiment or a cheaper external rate."
        ),
        "medium_confidence_excluded_from_expected_h100_usd": medium_risk,
    }


def write_all_artifacts(root: Path | None = None) -> dict[str, Path]:
    root = repo_root(root)
    rates = load_rates(root)
    ledger = build_task2_object_ledger(root)
    reuse = build_checkpoint_reuse_audit(root)
    waste = build_waste_analysis(ledger)
    plan = build_execution_plan(ledger, reuse)
    runtime = build_runtime_optimization_review()
    projection = build_projection(root, ledger, waste, rates)

    code_sha = git_head_sha(root)
    common = {
        "audit_code_sha": code_sha,
        "base_billing_preflight_sha": "feca09c00438f323c3903d1439dfee91ef728cd1",
        "h100_gpu_seconds_used_by_audit": H100_GPU_SECONDS_AUTHORIZED,
        "l40s_gpu_seconds_used_by_audit": L40S_GPU_SECONDS_AUTHORIZED,
        "remaining_modal_credits_usd": REMAINING_MODAL_CREDITS_USD,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    payloads = {
        "artifacts/seed4_cost_reduction_task2_object_ledger.json": {**ledger, **common},
        "artifacts/seed4_cost_reduction_checkpoint_reuse_audit.json": {**reuse, **common},
        "artifacts/seed4_cost_reduction_waste_analysis.json": {**waste, **common},
        "artifacts/seed4_cost_reduction_execution_plan.json": {**plan, **common},
        "artifacts/seed4_cost_reduction_runtime_optimization_review.json": {**runtime, **common},
        "artifacts/seed4_cost_reduction_projection.json": {**projection, **common},
    }
    out_paths = {}
    for rel, payload in payloads.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        out_paths[rel] = path
    return out_paths
