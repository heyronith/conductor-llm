"""Automated validation tests for the public CCPT webpage handoff package."""

import json
import re
import zipfile
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
HANDOFF_DIR = ARTIFACTS_DIR / "ccpt_webpage_handoff"
ZIP_PATH = ARTIFACTS_DIR / "ccpt_webpage_handoff.zip"


def test_package_structure_and_files():
    """Verify all required files exist in the handoff directory and ZIP archive."""
    required_files = [
        "README_INTEGRATION.md",
        "SOURCE_PROVENANCE.md",
        "SCIENTIFIC_CLAIMS.md",
        "ccpt-article.html",
        "ccpt-article.css",
        "ccpt-article.js",
        "data/ccpt-results.json",
        "assets/architecture-fallback.svg",
        "assets/persistence-fallback.svg",
        "assets/controller-fallback.svg",
        "scripts/build_public_results.py",
    ]

    for rel_path in required_files:
        full_p = HANDOFF_DIR / rel_path
        assert full_p.exists(), f"Missing required file: {rel_path}"
        assert full_p.stat().st_size > 0, f"Empty file: {rel_path}"

    assert ZIP_PATH.exists(), "Missing ccpt_webpage_handoff.zip"
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        namelist = z.namelist()
        for rel_path in required_files:
            assert f"ccpt_webpage_handoff/{rel_path}" in namelist, f"Missing in zip: {rel_path}"


def test_public_results_numerical_parity_with_machine_tables():
    """Ensure data/ccpt-results.json strictly matches task8_2_machine_tables.json."""
    public_data_p = HANDOFF_DIR / "data" / "ccpt-results.json"
    machine_tables_p = ARTIFACTS_DIR / "task8_2_machine_tables.json"

    with open(public_data_p, "r", encoding="utf-8") as f:
        p_data = json.load(f)

    with open(machine_tables_p, "r", encoding="utf-8") as f:
        m_tables = json.load(f)

    # Check Table A primary values
    for s in ["20260821", "20260823", "20260824"]:
        p_s = p_data["behavior"]["table_a_primary_persistence"][s]
        m_s = m_tables["table_a_behavior"][s]
        assert p_s["c_pre_refusal_rate"] == m_s["c_pre_refusal_rate"]
        assert p_s["c_post_refusal_rate"] == m_s["c_post_refusal_rate"]
        assert p_s["c_retention_delta_pp"] == m_s["c_retention_delta_pp"]
        assert p_s["d_pre_refusal_rate"] == m_s["d_pre_refusal_rate"]
        assert p_s["d_post_refusal_rate"] == m_s["d_post_refusal_rate"]
        assert p_s["d_retention_delta_pp"] == m_s["d_retention_delta_pp"]
        assert p_s["primary_effect_pp"] == m_s["primary_effect_pp"]

    # Verify benign post range
    min_b = p_data["behavior"]["aggregate_summary"]["min_c_post_benign_rate"]
    max_b = p_data["behavior"]["aggregate_summary"]["max_c_post_benign_rate"]
    assert pytest.approx(min_b, rel=1e-4) == 0.6484375
    assert pytest.approx(max_b, rel=1e-4) == 0.79296875


def test_scientific_guardrails_in_html_and_claims():
    """Verify that scientific guardrails (seed heterogeneity, over-refusal, no overclaiming) are honored."""
    html_p = HANDOFF_DIR / "ccpt-article.html"
    claims_p = HANDOFF_DIR / "SCIENTIFIC_CLAIMS.md"

    with open(html_p, "r", encoding="utf-8") as f:
        html_text = f.read()

    with open(claims_p, "r", encoding="utf-8") as f:
        claims_text = f.read()

    # Dynamic data-bind attributes must exist for key numbers
    assert 'data-bind="s1_primary_effect"' in html_text
    assert 'data-bind="s2_primary_effect"' in html_text
    assert 'data-bind="s3_primary_effect"' in html_text
    assert 'data-bind="mean_primary_effect"' in html_text
    assert 'data-bind="post_c_benign_range"' in html_text

    # Benign over-refusal must be disclosed
    assert "over-refusal" in html_text.lower()

    # Forbidden marketing terms must NOT appear
    forbidden_terms = ["solves alignment", "intrinsically aligned", "intrinsically safe"]
    for term in forbidden_terms:
        assert term not in html_text.lower(), f"Forbidden marketing term found: '{term}'"

    assert "NOT CLAIMED" in claims_text


def test_no_hardcoded_scientific_literals_in_js_charts():
    """Verify that JS uses variables from parsed data rather than hardcoded chart numbers."""
    js_p = HANDOFF_DIR / "ccpt-article.js"
    with open(js_p, "r", encoding="utf-8") as f:
        js_text = f.read()

    # Verify that chart rendering uses variable properties rather than hardcoded primary effect strings
    assert "s1.primary_effect_pp.toFixed" in js_text
    assert "s2.primary_effect_pp.toFixed" in js_text
    assert "s3.primary_effect_pp.toFixed" in js_text
    assert "agg.mean_primary_effect_pp.toFixed" in js_text


def test_article_prose_word_count():
    """Verify that the article prose is substantial and meets the target length."""
    html_p = HANDOFF_DIR / "ccpt-article.html"
    with open(html_p, "r", encoding="utf-8") as f:
        html_text = f.read()

    clean_text = re.sub(r"<[^>]+>", " ", html_text)
    clean_text = re.sub(r"\s+", " ", clean_text)
    words = clean_text.split()
    word_count = len(words)

    assert 1000 <= word_count <= 2500, f"Article word count out of expected range: {word_count}"
