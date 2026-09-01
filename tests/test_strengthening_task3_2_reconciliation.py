"""Tests for CCPT Strengthening Task 3.2 historical evidence reconciliation."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@pytest.fixture(scope="module")
def reconciliation():
    from ccpt.analysis.strengthening_task3_2_reconciliation import build_reconciliation

    return build_reconciliation(PROJECT_ROOT)


@pytest.fixture(scope="module")
def reconciliation_artifact_path():
    artifact = ARTIFACTS_DIR / "strengthening_task3_2_reconciliation.json"
    if not artifact.exists():
        import importlib.util

        script_path = PROJECT_ROOT / "scripts" / "generate_task3_2_reconciliation_artifacts.py"
        spec = importlib.util.spec_from_file_location("generate_task3_2_reconciliation_artifacts", script_path)
        report_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(report_mod)
        report_mod.main()
    assert artifact.exists()
    with open(artifact, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_historical_values_from_authoritative_artifacts(reconciliation):
    hist = reconciliation["historical_seed1"]
    assert hist["source_artifact_path"] == "artifacts/task7_3_1a_forensic_summary.json"
    assert Path(PROJECT_ROOT / hist["source_artifact_path"]).exists()

    c = hist["models"]["model_c"]
    d = hist["models"]["model_d"]
    assert c["pre"]["refusal_yes"] == 224
    assert c["post1000"]["refusal_yes"] == 221
    assert pytest.approx(c["pre"]["rate"]) == 0.875
    assert pytest.approx(c["post1000"]["rate"]) == 0.86328125
    assert pytest.approx(c["retention_1000"]) == -0.01171875

    assert d["pre"]["refusal_yes"] == 239
    assert d["post1000"]["refusal_yes"] == 131
    assert pytest.approx(d["pre"]["rate"]) == 0.93359375
    assert pytest.approx(d["post1000"]["rate"]) == 0.51171875
    assert pytest.approx(d["retention_1000"]) == -0.421875

    assert pytest.approx(reconciliation["historical_seed1"]["c_minus_d_effect_1000"]) == 0.41015625


def test_no_historical_constants_in_report_generator():
    import importlib.util

    script_path = PROJECT_ROOT / "scripts" / "generate_task3_2_reconciliation_artifacts.py"
    spec = importlib.util.spec_from_file_location("generate_task3_2_reconciliation_artifacts", script_path)
    report_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(report_mod)
    source = inspect.getsource(report_mod.render_report)
    assert "0.875" not in source
    assert "0.93359" not in source
    assert "41.02" not in source
    assert "0.41015625" not in source


def test_no_historical_constants_in_reconciliation_module_beyond_schema():
    from ccpt.analysis import strengthening_task3_2_reconciliation as mod

    source = inspect.getsource(mod)
    assert "0.87500" not in source
    assert "0.77734" not in source
    assert "41.02" not in source


def test_historical_retention_arithmetic_exact(reconciliation):
    c = reconciliation["historical_seed1"]["models"]["model_c"]
    d = reconciliation["historical_seed1"]["models"]["model_d"]
    assert c["retention_1000"] == c["post1000"]["rate"] - c["pre"]["rate"]
    assert d["retention_1000"] == d["post1000"]["rate"] - d["pre"]["rate"]
    effect = c["retention_1000"] - d["retention_1000"]
    assert pytest.approx(effect) == reconciliation["historical_seed1"]["c_minus_d_effect_1000"]


def test_corrected_rerun_retention_arithmetic_exact(reconciliation):
    for model in ("model_b", "model_c", "model_d"):
        step0 = reconciliation["corrected_rerun_seed1"]["models"][model]["horizons"]["0"]["rate"]
        for step in (250, 1000, 4000):
            rate = reconciliation["corrected_rerun_seed1"]["models"][model]["horizons"][str(step)]["rate"]
            retention = reconciliation["corrected_rerun_seed1"]["models"][model]["retention"][str(step)]
            assert retention["retention"] == rate - step0


def test_c_d_1000_effect_correct(reconciliation):
    c_ret = reconciliation["corrected_rerun_seed1"]["models"]["model_c"]["retention"]["1000"]["retention"]
    d_ret = reconciliation["corrected_rerun_seed1"]["models"]["model_d"]["retention"]["1000"]["retention"]
    assert reconciliation["pairwise_effects"]["c_minus_d"]["1000"] == c_ret - d_ret
    assert pytest.approx(reconciliation["pairwise_effects"]["c_minus_d"]["1000_pp"], rel=1e-6) == 4.6875


def test_c_d_4000_effect_correct(reconciliation):
    c_ret = reconciliation["corrected_rerun_seed1"]["models"]["model_c"]["retention"]["4000"]["retention"]
    d_ret = reconciliation["corrected_rerun_seed1"]["models"]["model_d"]["retention"]["4000"]["retention"]
    assert reconciliation["pairwise_effects"]["c_minus_d"]["4000"] == c_ret - d_ret
    assert pytest.approx(reconciliation["pairwise_effects"]["c_minus_d"]["4000_pp"], rel=1e-6) == -12.890625


def test_model_c_step4000_active_off_gap_present(reconciliation):
    ablation = reconciliation["model_c_ablation"]["4000"]
    assert ablation["gap_pp"] < 0
    assert reconciliation["reconciliation"]["crossover_reversal_at_4000"] is True
    assert reconciliation["classification"]["controller_direction_step4000"] == "NOT_REPRODUCED"


def test_report_numbers_derive_from_reconciliation_artifact(reconciliation_artifact_path):
    report_path = PROJECT_ROOT / "docs" / "research" / "strengthening_task3_2_historical_reconciliation.md"
    if not report_path.exists():
        import importlib.util

        script_path = PROJECT_ROOT / "scripts" / "generate_task3_2_reconciliation_artifacts.py"
        spec = importlib.util.spec_from_file_location("generate_task3_2_reconciliation_artifacts", script_path)
        report_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(report_mod)
        report_mod.main()
    text = report_path.read_text(encoding="utf-8")

    hist_effect_pp = reconciliation_artifact_path["historical_seed1"]["c_minus_d_effect_1000_pp"]
    rerun_effect_pp = reconciliation_artifact_path["pairwise_effects"]["c_minus_d"]["1000_pp"]
    gap_4000_pp = reconciliation_artifact_path["model_c_ablation"]["4000"]["gap_pp"]

    assert f"{hist_effect_pp:+.2f} pp".replace("+", "") in text or f"{hist_effect_pp:+.2f} pp" in text
    assert f"{rerun_effect_pp:+.2f} pp".replace("+", "") in text or f"{rerun_effect_pp:+.2f} pp" in text
    assert f"{gap_4000_pp:+.2f} pp" in text


def test_task3_1_raw_evidence_unchanged():
    before = {
        "behavior": _sha256(ARTIFACTS_DIR / "strengthening_task3_1_behavior_summary.json"),
        "summary": _sha256(ARTIFACTS_DIR / "strengthening_task3_1_summary.json"),
        "repro": _sha256(ARTIFACTS_DIR / "strengthening_task3_1_reproducibility_summary.json"),
    }
    # Re-run reconciliation build; raw Task 3.1 evidence must remain bit-identical.
    from ccpt.analysis.strengthening_task3_2_reconciliation import build_reconciliation

    build_reconciliation(PROJECT_ROOT)
    after = {
        "behavior": _sha256(ARTIFACTS_DIR / "strengthening_task3_1_behavior_summary.json"),
        "summary": _sha256(ARTIFACTS_DIR / "strengthening_task3_1_summary.json"),
        "repro": _sha256(ARTIFACTS_DIR / "strengthening_task3_1_reproducibility_summary.json"),
    }
    assert before == after


def test_task3_forensic_evidence_unchanged():
    paths = [
        ARTIFACTS_DIR / "strengthening_task3_root_cause.json",
        ARTIFACTS_DIR / "strengthening_task3_behavior_reconciliation.json",
        ARTIFACTS_DIR / "strengthening_task3_checkpoint_comparison.json",
    ]
    before = {str(p): _sha256(p) for p in paths}
    from ccpt.analysis.strengthening_task3_2_reconciliation import build_reconciliation

    build_reconciliation(PROJECT_ROOT)
    after = {str(p): _sha256(p) for p in paths}
    assert before == after


def test_zero_gpu_invariant(reconciliation):
    assert reconciliation["h100_gpu_seconds"] == 0
    assert reconciliation["l40s_gpu_seconds"] == 0
    mod = importlib.import_module("ccpt.analysis.strengthening_task3_2_reconciliation")
    source = inspect.getsource(mod)
    assert "modal" not in source.lower()
    assert "cuda" not in source.lower()
    assert "torch" not in source.lower()


def test_seed4_cannot_be_launched_from_task3_2_module():
    mod = importlib.import_module("ccpt.analysis.strengthening_task3_2_reconciliation")
    source = inspect.getsource(mod)
    assert "20260825" not in source
    assert "seed_4" not in source.lower()
    assert "seed4" not in source.lower()


def test_task3_1_historical_join_was_wrong(reconciliation):
    audit = reconciliation["reconciliation"]["task3_1_historical_join_audit"]
    assert audit["task3_1_historical_join_wrong"] is True


def test_primary_classifications_explicit(reconciliation):
    cls = reconciliation["classification"]
    assert cls["evaluation_defect"] == "EVALUATION_DEFECT_CONFIRMED_AND_CORRECTED"
    assert cls["primary_1000_persistence_reproducibility"] == "REPRODUCED_DIRECTION_ONLY"
    assert cls["safety_acquisition"] == "PARTIALLY_REPRODUCED"
    assert cls["controller_direction_step0"] == "REPRODUCED"
    assert cls["controller_direction_step1000"] == "REPRODUCED"
    assert cls["controller_direction_step4000"] == "NOT_REPRODUCED"
