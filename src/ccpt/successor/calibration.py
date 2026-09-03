"""Calibration isolation for successor Task 1 (reuse frozen strengthening calibration)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_calibration_manifest_path() -> Path:
    return repo_root() / "artifacts" / "strengthening_calibration_prompt_manifest.json"


def load_calibration_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or default_calibration_manifest_path()
    return json.loads(p.read_text(encoding="utf-8"))


def build_successor_calibration_reference(
    path: Optional[Path] = None,
    *,
    held_out_diagnostic_count: int = 64,
) -> Dict[str, Any]:
    """Reference the frozen strengthening calibration manifest; carve a held-out diagnostic slice.

    Fitting uses records[held_out_diagnostic_count:] only.
    Diagnostics use the first held_out_diagnostic_count records (deterministic order).
    No regeneration of prompts; no evaluation-set leakage.
    """
    p = path or default_calibration_manifest_path()
    man = load_calibration_manifest(p)
    records: List[Dict[str, Any]] = list(man["records"])
    audit = man.get("test_isolation_audit", {})
    fit_records = records[held_out_diagnostic_count:]
    diag_records = records[:held_out_diagnostic_count]

    def _ids(rs: List[Dict[str, Any]]) -> List[str]:
        return [r["example_id"] for r in rs]

    fit_ids = _ids(fit_records)
    diag_ids = _ids(diag_records)
    overlap_fit_diag = sorted(set(fit_ids) & set(diag_ids))

    return {
        "task": "successor_task1_calibration_manifest",
        "reuses_existing_manifest": True,
        "source_path": str(p.relative_to(repo_root())),
        "source_manifest_version": man.get("manifest_version"),
        "source_records_logical_hash": man.get("records_logical_hash"),
        "source_revision": man.get("source_dataset", {}).get("revision"),
        "source_repo": man.get("source_dataset", {}).get("repo"),
        "source_split": man.get("source_dataset", {}).get("split"),
        "total_source_records": len(records),
        "held_out_diagnostic_count": held_out_diagnostic_count,
        "fit_record_count": len(fit_records),
        "diagnostic_record_ids_head": diag_ids[:8],
        "fit_record_ids_head": fit_ids[:8],
        "fit_diagnostic_id_overlap": overlap_fit_diag,
        "harmful_records_count_source": man.get("deduplication_and_filtering_policy", {}).get(
            "harmful_records_count"
        ),
        "benign_records_count_source": man.get("deduplication_and_filtering_policy", {}).get(
            "benign_records_count"
        ),
        "wildguard_test_overlap": int(audit.get("wildguard_test_overlap_count", -1)),
        "beavertails_ood_overlap": int(audit.get("beavertails_30k_test_overlap_count", -1)),
        "xstest_overlap": int(audit.get("xstest_overlap_count", -1)),
        "isolation_status": audit.get("isolation_status"),
        "CALIBRATION_FINAL_TEST_OVERLAP": int(audit.get("wildguard_test_overlap_count", -1))
        + int(audit.get("beavertails_30k_test_overlap_count", -1))
        + int(audit.get("xstest_overlap_count", -1)),
        "ordering": "source_manifest_record_order_stable",
        "deduplication_rule": man.get("deduplication_and_filtering_policy", {}).get(
            "deduplication_key"
        ),
        "policy": {
            "no_beavertails_ood_in_loss": True,
            "no_wildguard_test_in_loss": True,
            "no_xstest_in_loss": True,
            "teacher": "PRE_ACTIVE",
            "student_base": "POST_FROZEN",
        },
    }


def assert_zero_eval_overlap(ref: Dict[str, Any]) -> None:
    if ref.get("CALIBRATION_FINAL_TEST_OVERLAP") != 0:
        raise RuntimeError(f"Non-zero calibration/eval overlap: {ref}")
    if ref.get("fit_diagnostic_id_overlap"):
        raise RuntimeError("Fit/diagnostic calibration slice overlap")
