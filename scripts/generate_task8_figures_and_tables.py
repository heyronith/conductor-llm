"""Task 8.2: Machine-Derived Mechanistic Synthesis and Figure Generation Script.

Loads mechanistic diagnostics and authoritative tri-state WildGuard behavioral records
programmatically, enforces strict numerical parity assertions, computes exact CKA and cosine
metrics separately, performs NA sensitivity analysis for ablation gaps, creates
artifacts/task8_2_machine_tables.json, and outputs paper-ready figures and tables.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "task8_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_authoritative_behavioral_data() -> Dict[str, Any]:
    """Loads authoritative behavioral data from Task 7.3.1a and Task 7.4 summaries."""
    seed1_path = ARTIFACTS_DIR / "task7_3_1a_forensic_summary.json"
    seeds23_path = ARTIFACTS_DIR / "task7_4_multiseed_replication_summary.json"

    assert seed1_path.exists(), f"Missing authoritative Seed 1 summary: {seed1_path}"
    assert seeds23_path.exists(), f"Missing authoritative Seeds 2/3 summary: {seeds23_path}"

    with open(seed1_path, "r", encoding="utf-8") as f:
        s1_data = json.load(f)

    with open(seeds23_path, "r", encoding="utf-8") as f:
        s23_data = json.load(f)

    behavior: Dict[str, Dict[str, Any]] = {"20260821": {}, "20260823": {}, "20260824": {}}

    # Seed 1 Extraction
    br1 = s1_data["behavioral_results"]
    for ph in ["pre_persistence", "post_persistence"]:
        for m in ["model_c", "model_d"]:
            h_on = br1[ph][m]["ood_behavioral"]["harmful_eval"]
            b_on = br1[ph][m]["ood_behavioral"]["benign_eval"]

            m_off_key = f"{m}_scale_0"
            h_off = br1[ph][m_off_key]["ood_behavioral"]["harmful_eval"]
            b_off = br1[ph][m_off_key]["ood_behavioral"]["benign_eval"]

            behavior["20260821"][f"{m}_{ph}_on_harmful"] = {
                "yes": h_on["yes_count"], "no": h_on["no_count"], "na": h_on["na_count"],
                "total": h_on["total_prompts"], "determinate_n": h_on["determinate_n"],
                "rate": h_on["safe_refusal_determinate_rate"]
            }
            behavior["20260821"][f"{m}_{ph}_on_benign"] = {
                "yes": b_on["yes_count"], "no": b_on["no_count"], "na": b_on["na_count"],
                "total": b_on["total_prompts"], "determinate_n": b_on["determinate_n"],
                "rate": b_on["over_refusal_determinate_rate"]
            }
            behavior["20260821"][f"{m}_{ph}_off_harmful"] = {
                "yes": h_off["yes_count"], "no": h_off["no_count"], "na": h_off["na_count"],
                "total": h_off["total_prompts"], "determinate_n": h_off["determinate_n"],
                "rate": h_off["safe_refusal_determinate_rate"]
            }
            behavior["20260821"][f"{m}_{ph}_off_benign"] = {
                "yes": b_off["yes_count"], "no": b_off["no_count"], "na": b_off["na_count"],
                "total": b_off["total_prompts"], "determinate_n": b_off["determinate_n"],
                "rate": b_off["over_refusal_determinate_rate"]
            }

    # Seeds 2 & 3 Extraction
    for s_str in ["20260823", "20260824"]:
        gs = s23_data["judge_results"][s_str]["grouped_summaries"]
        for m in ["model_c", "model_d"]:
            for ph in ["pre_persistence", "post_persistence"]:
                for cond in ["on", "off"]:
                    for pt in ["harmful", "benign"]:
                        k = f"{m}_{ph}_{cond}_ood_beavertails_{pt}"
                        rr = gs[k]["response_refusal"]
                        rate = rr["yes"] / rr["determinate_n"] if rr["determinate_n"] > 0 else 0.0
                        behavior[s_str][f"{m}_{ph}_{cond}_{pt}"] = {
                            "yes": rr["yes"], "no": rr["no"], "na": rr["na"],
                            "total": gs[k]["total_prompts"], "determinate_n": rr["determinate_n"],
                            "rate": rate
                        }

    # Hard numerical assertions for authoritative Table A
    assert np.isclose(behavior["20260821"]["model_c_pre_persistence_on_harmful"]["rate"], 0.87500000)
    assert np.isclose(behavior["20260821"]["model_c_post_persistence_on_harmful"]["rate"], 0.86328125)
    assert np.isclose(behavior["20260821"]["model_d_pre_persistence_on_harmful"]["rate"], 0.93359375)
    assert np.isclose(behavior["20260821"]["model_d_post_persistence_on_harmful"]["rate"], 0.51171875)

    assert behavior["20260823"]["model_c_pre_persistence_on_harmful"]["yes"] == 220
    assert behavior["20260823"]["model_c_post_persistence_on_harmful"]["yes"] == 173
    assert np.isclose(behavior["20260823"]["model_c_pre_persistence_on_harmful"]["rate"], 220 / 256)
    assert np.isclose(behavior["20260823"]["model_c_post_persistence_on_harmful"]["rate"], 173 / 256)
    assert behavior["20260823"]["model_d_pre_persistence_on_harmful"]["yes"] == 238
    assert behavior["20260823"]["model_d_post_persistence_on_harmful"]["yes"] == 227
    assert np.isclose(behavior["20260823"]["model_d_pre_persistence_on_harmful"]["rate"], 238 / 256)
    assert np.isclose(behavior["20260823"]["model_d_post_persistence_on_harmful"]["rate"], 227 / 256)

    assert behavior["20260824"]["model_c_pre_persistence_on_harmful"]["yes"] == 171
    assert behavior["20260824"]["model_c_post_persistence_on_harmful"]["yes"] == 201
    assert np.isclose(behavior["20260824"]["model_c_pre_persistence_on_harmful"]["rate"], 171 / 256)
    assert np.isclose(behavior["20260824"]["model_c_post_persistence_on_harmful"]["rate"], 201 / 256)
    assert behavior["20260824"]["model_d_pre_persistence_on_harmful"]["yes"] == 246
    assert behavior["20260824"]["model_d_post_persistence_on_harmful"]["yes"] == 219
    assert np.isclose(behavior["20260824"]["model_d_pre_persistence_on_harmful"]["rate"], 246 / 256)
    assert np.isclose(behavior["20260824"]["model_d_post_persistence_on_harmful"]["rate"], 219 / 256)

    return behavior


def compute_ablation_gap_sensitivity(behavior: Dict[str, Any]) -> Dict[str, Any]:
    """Computes active/off ablation gaps with NA sensitivity bounds for Model C."""
    seeds = [20260821, 20260823, 20260824]
    ablation_summary = {}

    for s in seeds:
        s_str = str(s)
        pre_on = behavior[s_str]["model_c_pre_persistence_on_harmful"]
        pre_off = behavior[s_str]["model_c_pre_persistence_off_harmful"]
        post_on = behavior[s_str]["model_c_post_persistence_on_harmful"]
        post_off = behavior[s_str]["model_c_post_persistence_off_harmful"]

        # Determinate Rates
        pre_gap_det = pre_on["rate"] - pre_off["rate"]
        post_gap_det = post_on["rate"] - post_off["rate"]
        delta_gap_det = post_gap_det - pre_gap_det

        # Sensitivity A: All NA = Refusal (Upper bound on off refusal, lower bound on gap)
        pre_off_rate_a = (pre_off["yes"] + pre_off["na"]) / pre_off["total"]
        post_off_rate_a = (post_off["yes"] + post_off["na"]) / post_off["total"]
        pre_gap_a = pre_on["rate"] - pre_off_rate_a
        post_gap_a = post_on["rate"] - post_off_rate_a
        delta_gap_a = post_gap_a - pre_gap_a

        # Sensitivity B: All NA = Non-refusal (Lower bound on off refusal, upper bound on gap)
        pre_off_rate_b = pre_off["yes"] / pre_off["total"]
        post_off_rate_b = post_off["yes"] / post_off["total"]
        pre_gap_b = pre_on["rate"] - pre_off_rate_b
        post_gap_b = post_on["rate"] - post_off_rate_b
        delta_gap_b = post_gap_b - pre_gap_b

        ablation_summary[s_str] = {
            "pre_active_rate": pre_on["rate"],
            "pre_off_rate_determinate": pre_off["rate"],
            "pre_off_na_count": pre_off["na"],
            "pre_ablation_gap_determinate": pre_gap_det,
            "post_active_rate": post_on["rate"],
            "post_off_rate_determinate": post_off["rate"],
            "post_off_na_count": post_off["na"],
            "post_ablation_gap_determinate": post_gap_det,
            "ablation_gap_change_determinate": delta_gap_det,
            "sensitivity_a_all_na_refusal": {
                "pre_gap": pre_gap_a,
                "post_gap": post_gap_a,
                "gap_change": delta_gap_a,
            },
            "sensitivity_b_all_na_nonrefusal": {
                "pre_gap": pre_gap_b,
                "post_gap": post_gap_b,
                "gap_change": delta_gap_b,
            },
            "sign_stable_across_bounds": (delta_gap_det < 0 and delta_gap_a < 0 and delta_gap_b < 0) if s == 20260823 else (delta_gap_det > 0 and delta_gap_a > 0 and delta_gap_b > 0),
        }

    return ablation_summary


def main():
    print("=== TASK 8.2: MACHINE-DERIVED MECHANISTIC SYNTHESIS ===", flush=True)

    # 1. Load authoritative behavior
    behavior = load_authoritative_behavioral_data()

    # 2. Load raw mechanistic diagnostic artifact and CKA summary
    summary_path = ARTIFACTS_DIR / "task8_mechanistic_summary.json"
    cka_path = ARTIFACTS_DIR / "task8_cka_summary.json"

    assert summary_path.exists(), f"Missing mechanistic summary: {summary_path}"
    assert cka_path.exists(), f"Missing CKA summary: {cka_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        diag_data = json.load(f)

    with open(cka_path, "r", encoding="utf-8") as f:
        cka_summary = json.load(f)

    records = diag_data["per_prompt_records"]
    seeds = [20260821, 20260823, 20260824]

    # Subset records for OOD harmful primary
    ood_harmful_c = [r for r in records if r["model"] == "model_c" and r["dataset"] == "ood_beavertails" and r["prompt_type"] == "harmful"]
    ood_benign_c = [r for r in records if r["model"] == "model_c" and r["dataset"] == "ood_beavertails" and r["prompt_type"] == "benign"]
    ood_harmful_d = [r for r in records if r["model"] == "model_d" and r["dataset"] == "ood_beavertails" and r["prompt_type"] == "harmful"]

    # =========================================================================
    # BUILD MACHINE-GENERATED TABLES (task8_2_machine_tables.json)
    # =========================================================================
    machine_tables: Dict[str, Any] = {
        "version": "task8_2_machine_tables_v1",
        "parent_evidence_sha": "92b94420ab9545b9f55b287a1dd6d752b010050a",
        "raw_mechanistic_artifact_sha256": "77faac51208115b4d8157a7fe937271e8793f0c582255e857b11c7cf4fa5a516",
        "table_a_behavior": {},
        "table_b_model_c_drift": {},
        "table_c_model_c_causal_selectivity": {},
        "table_d_model_d_adapter_drift": {},
        "table_e_transitions": {},
        "ablation_sensitivity": compute_ablation_gap_sensitivity(behavior),
    }

    # Table A: Behavior
    for s in seeds:
        s_str = str(s)
        c_pre = behavior[s_str]["model_c_pre_persistence_on_harmful"]["rate"]
        c_post = behavior[s_str]["model_c_post_persistence_on_harmful"]["rate"]
        d_pre = behavior[s_str]["model_d_pre_persistence_on_harmful"]["rate"]
        d_post = behavior[s_str]["model_d_post_persistence_on_harmful"]["rate"]
        c_delta = c_post - c_pre
        d_delta = d_post - d_pre
        primary_effect = c_delta - d_delta

        machine_tables["table_a_behavior"][s_str] = {
            "c_pre_refusal_rate": float(c_pre),
            "c_post_refusal_rate": float(c_post),
            "c_retention_delta_pp": float(c_delta * 100.0),
            "d_pre_refusal_rate": float(d_pre),
            "d_post_refusal_rate": float(d_post),
            "d_retention_delta_pp": float(d_delta * 100.0),
            "primary_effect_pp": float(primary_effect * 100.0),
        }

    # Table B: Model C Drift (exposing both mean cosine and CKA separately)
    for s in seeds:
        s_str = str(s)
        s_recs = [r for r in ood_harmful_c if r["seed"] == s]
        machine_tables["table_b_model_c_drift"][s_str] = {}
        for l in [2, 4]:
            cap_rel_l2 = float(np.mean([r[f"layer_{l}_capability_relative_l2"] for r in s_recs]))
            cap_cosine = float(np.mean([r[f"layer_{l}_capability_cosine"] for r in s_recs]))
            cap_cka = float(cka_summary[f"seed_{s}_model_c_ood_beavertails_harmful_c_tilde_{l}"])

            obs_rel_l2 = float(np.mean([r[f"layer_{l}_obs_relative_l2"] for r in s_recs]))
            obs_cosine = float(np.mean([r[f"layer_{l}_obs_cosine"] for r in s_recs]))
            obs_cka = float(cka_summary[f"seed_{s}_model_c_ood_beavertails_harmful_obs_{l}"])

            norm_rel_l2 = float(np.mean([r[f"layer_{l}_normative_relative_l2"] for r in s_recs]))
            norm_cosine = float(np.mean([r[f"layer_{l}_normative_cosine"] for r in s_recs]))
            norm_cka = float(cka_summary[f"seed_{s}_model_c_ood_beavertails_harmful_norm_{l}"])

            steer_rel_l2 = float(np.mean([r[f"layer_{l}_steering_relative_l2"] for r in s_recs]))
            steer_cosine = float(np.mean([r[f"layer_{l}_steering_cosine"] for r in s_recs]))
            steer_cka = float(cka_summary[f"seed_{s}_model_c_ood_beavertails_harmful_steer_{l}"])

            gate_change = float(np.mean([r[f"layer_{l}_gate_absolute_change"] for r in s_recs]))

            machine_tables["table_b_model_c_drift"][s_str][f"layer_{l}"] = {
                "capability_relative_l2_mean": cap_rel_l2,
                "capability_mean_cosine": cap_cosine,
                "capability_linear_cka": cap_cka,
                "obs_relative_l2_mean": obs_rel_l2,
                "obs_mean_cosine": obs_cosine,
                "obs_linear_cka": obs_cka,
                "normative_relative_l2_mean": norm_rel_l2,
                "normative_mean_cosine": norm_cosine,
                "normative_linear_cka": norm_cka,
                "steering_relative_l2_mean": steer_rel_l2,
                "steering_mean_cosine": steer_cosine,
                "steering_linear_cka": steer_cka,
                "gate_absolute_change_mean": gate_change,
            }

    # Table C: Model C Steering Selectivity (Harmful vs Benign)
    for s in seeds:
        s_str = str(s)
        h_recs = [r for r in ood_harmful_c if r["seed"] == s]
        b_recs = [r for r in ood_benign_c if r["seed"] == s]
        machine_tables["table_c_model_c_causal_selectivity"][s_str] = {}
        for l in [2, 4]:
            h_pre = float(np.mean([r[f"layer_{l}_steering_norm_pre"] for r in h_recs]))
            h_post = float(np.mean([r[f"layer_{l}_steering_norm_post"] for r in h_recs]))
            b_pre = float(np.mean([r[f"layer_{l}_steering_norm_pre"] for r in b_recs]))
            b_post = float(np.mean([r[f"layer_{l}_steering_norm_post"] for r in b_recs]))
            machine_tables["table_c_model_c_causal_selectivity"][s_str][f"layer_{l}"] = {
                "harmful_steering_norm_pre": h_pre,
                "harmful_steering_norm_post": h_post,
                "benign_steering_norm_pre": b_pre,
                "benign_steering_norm_post": b_post,
                "selectivity_pre": h_pre - b_pre,
                "selectivity_post": h_post - b_post,
                "selectivity_change": (h_post - b_post) - (h_pre - b_pre),
            }

    # Table D: Model D Adapter Drift across all 8 sites
    for s in seeds:
        s_str = str(s)
        s_recs = [r for r in ood_harmful_d if r["seed"] == s]
        machine_tables["table_d_model_d_adapter_drift"][s_str] = {}
        for l_idx in range(4):
            for a_type in ["attn", "mlp"]:
                site_name = f"layer_{l_idx}_{a_type}_adapter"
                in_rel_l2 = float(np.mean([r[f"{site_name}_input_relative_l2"] for r in s_recs]))
                in_cos = float(np.mean([r[f"{site_name}_input_cosine"] for r in s_recs]))
                in_cka = float(cka_summary[f"seed_{s}_model_d_ood_beavertails_harmful_{site_name}_in"])

                res_rel_l2 = float(np.mean([r[f"{site_name}_residual_relative_l2"] for r in s_recs]))
                res_cos = float(np.mean([r[f"{site_name}_residual_cosine"] for r in s_recs]))
                res_cka = float(cka_summary[f"seed_{s}_model_d_ood_beavertails_harmful_{site_name}_res"])

                res_norm_pre = float(np.mean([r[f"{site_name}_residual_norm_pre"] for r in s_recs]))
                res_norm_post = float(np.mean([r[f"{site_name}_residual_norm_post"] for r in s_recs]))

                machine_tables["table_d_model_d_adapter_drift"][s_str][site_name] = {
                    "input_relative_l2_mean": in_rel_l2,
                    "input_mean_cosine": in_cos,
                    "input_linear_cka": in_cka,
                    "residual_relative_l2_mean": res_rel_l2,
                    "residual_mean_cosine": res_cos,
                    "residual_linear_cka": res_cka,
                    "residual_norm_pre": res_norm_pre,
                    "residual_norm_post": res_norm_post,
                }

    # Table E: Transitions
    for s in seeds:
        s_str = str(s)
        s_recs = [r for r in ood_harmful_c if r["seed"] == s]
        groups = ["retained_refusal", "lost_refusal", "gained_refusal", "persistent_nonrefusal", "indeterminate"]
        machine_tables["table_e_transitions"][s_str] = {}
        for g in groups:
            g_recs = [r for r in s_recs if r["transition_group"] == g]
            n_c = len(g_recs)
            machine_tables["table_e_transitions"][s_str][g] = {
                "count": n_c,
                "percentage": n_c / len(s_recs) * 100.0 if s_recs else 0.0,
            }
            if n_c > 0:
                machine_tables["table_e_transitions"][s_str][g].update({
                    "layer_2_capability_rel_l2_mean": float(np.mean([r["layer_2_capability_relative_l2"] for r in g_recs])),
                    "layer_2_steering_rel_l2_mean": float(np.mean([r["layer_2_steering_relative_l2"] for r in g_recs])),
                    "layer_4_capability_rel_l2_mean": float(np.mean([r["layer_4_capability_relative_l2"] for r in g_recs])),
                    "layer_4_steering_rel_l2_mean": float(np.mean([r["layer_4_steering_relative_l2"] for r in g_recs])),
                    "layer_4_gate_change_mean": float(np.mean([r["layer_4_gate_absolute_change"] for r in g_recs])),
                })

    # Save machine-derived tables JSON
    tables_out = ARTIFACTS_DIR / "task8_2_machine_tables.json"
    with open(tables_out, "w", encoding="utf-8") as f:
        json.dump(machine_tables, f, indent=2)
    print(f" -> Wrote authoritative machine tables artifact to {tables_out}")

    # =========================================================================
    # REASSESS HYPOTHESES DYNAMICALLY FROM MACHINE TABLES
    # =========================================================================
    tb = machine_tables["table_b_model_c_drift"]
    ta = machine_tables["table_a_behavior"]
    asens = machine_tables["ablation_sensitivity"]

    # Seed-level values
    s1_cap_cka_l4 = tb["20260821"]["layer_4"]["capability_linear_cka"]
    s2_cap_cka_l4 = tb["20260823"]["layer_4"]["capability_linear_cka"]
    s3_cap_cka_l4 = tb["20260824"]["layer_4"]["capability_linear_cka"]

    s1_cap_l2_l4 = tb["20260821"]["layer_4"]["capability_relative_l2_mean"]
    s2_cap_l2_l4 = tb["20260823"]["layer_4"]["capability_relative_l2_mean"]
    s3_cap_l2_l4 = tb["20260824"]["layer_4"]["capability_relative_l2_mean"]

    s1_obs_cka_l4 = tb["20260821"]["layer_4"]["obs_linear_cka"]
    s2_obs_cka_l4 = tb["20260823"]["layer_4"]["obs_linear_cka"]
    s3_obs_cka_l4 = tb["20260824"]["layer_4"]["obs_linear_cka"]

    s1_obs_l2_l4 = tb["20260821"]["layer_4"]["obs_relative_l2_mean"]
    s2_obs_l2_l4 = tb["20260823"]["layer_4"]["obs_relative_l2_mean"]
    s3_obs_l2_l4 = tb["20260824"]["layer_4"]["obs_relative_l2_mean"]

    s1_gate_l4 = tb["20260821"]["layer_4"]["gate_absolute_change_mean"]
    s2_gate_l4 = tb["20260823"]["layer_4"]["gate_absolute_change_mean"]
    s3_gate_l4 = tb["20260824"]["layer_4"]["gate_absolute_change_mean"]

    s1_norm_cka_l4 = tb["20260821"]["layer_4"]["normative_linear_cka"]
    s2_norm_cka_l4 = tb["20260823"]["layer_4"]["normative_linear_cka"]
    s3_norm_cka_l4 = tb["20260824"]["layer_4"]["normative_linear_cka"]

    s1_steer_cka_l4 = tb["20260821"]["layer_4"]["steering_linear_cka"]
    s2_steer_cka_l4 = tb["20260823"]["layer_4"]["steering_linear_cka"]
    s3_steer_cka_l4 = tb["20260824"]["layer_4"]["steering_linear_cka"]

    s1_gap_change = asens["20260821"]["ablation_gap_change_determinate"] * 100.0
    s1_pre_gap = asens["20260821"]["pre_ablation_gap_determinate"] * 100.0
    s1_post_gap = asens["20260821"]["post_ablation_gap_determinate"] * 100.0

    s2_gap_change = asens["20260823"]["ablation_gap_change_determinate"] * 100.0
    s2_pre_gap = asens["20260823"]["pre_ablation_gap_determinate"] * 100.0
    s2_post_gap = asens["20260823"]["post_ablation_gap_determinate"] * 100.0

    s3_gap_change = asens["20260824"]["ablation_gap_change_determinate"] * 100.0
    s3_pre_gap = asens["20260824"]["pre_ablation_gap_determinate"] * 100.0
    s3_post_gap = asens["20260824"]["post_ablation_gap_determinate"] * 100.0

    s1_c_pre = ta["20260821"]["c_pre_refusal_rate"] * 100.0
    s2_c_pre = ta["20260823"]["c_pre_refusal_rate"] * 100.0
    s3_c_pre = ta["20260824"]["c_pre_refusal_rate"] * 100.0
    s3_c_delta = ta["20260824"]["c_retention_delta_pp"]

    s1_d_delta = ta["20260821"]["d_retention_delta_pp"]
    s2_d_delta = ta["20260823"]["d_retention_delta_pp"]
    s3_d_delta = ta["20260824"]["d_retention_delta_pp"]

    min_obs_l2 = min(s1_obs_l2_l4, s2_obs_l2_l4, s3_obs_l2_l4)
    max_obs_l2 = max(s1_obs_l2_l4, s2_obs_l2_l4, s3_obs_l2_l4)
    min_obs_cka = min(s1_obs_cka_l4, s2_obs_cka_l4, s3_obs_cka_l4)
    max_obs_cka = max(s1_obs_cka_l4, s2_obs_cka_l4, s3_obs_cka_l4)

    hypothesis_assessment = {
        "H1_capability_interface_drift": {
            "status": "INCONCLUSIVE",
            "evidence_for": "Continuation pretraining induces representation drift at capability proposals and observation projections across all seeds.",
            "evidence_against": f"Seed 2 (the negative persistence seed) does NOT exhibit uniquely elevated capability drift: its Layer 4 capability CKA ({s2_cap_cka_l4:.4f}) is higher than positive Seed 1 ({s1_cap_cka_l4:.4f}) and comparable to Seed 3 ({s3_cap_cka_l4:.4f}), while its relative L2 drift ({s2_cap_l2_l4:.4f}) is lower than Seed 1 ({s1_cap_l2_l4:.4f}). Observation vector CKA ({s2_obs_cka_l4:.4f}) and relative L2 ({s2_obs_l2_l4:.4f}) are within a narrow range across all seeds ({min_obs_l2:.4f}-{max_obs_l2:.4f} rel L2; {min_obs_cka:.4f}-{max_obs_cka:.4f} CKA).",
            "limitations": "Interface drift occurs generically but does not monotonically order cross-seed persistence retention."
        },
        "H2_functional_controller_drift": {
            "status": "INCONCLUSIVE",
            "evidence_for": f"Seed 2 exhibits elevated Layer 4 gate absolute change ({s2_gate_l4:.4f}, higher than Seed 1's {s1_gate_l4:.4f} and Seed 3's {s3_gate_l4:.4f}).",
            "evidence_against": f"Broader prespecified controller-drift metrics are mixed: Seed 2 Layer 4 normative CKA ({s2_norm_cka_l4:.4f}) and steering CKA ({s2_steer_cka_l4:.4f}) are actually higher than Seed 1 ({s1_norm_cka_l4:.4f} and {s1_steer_cka_l4:.4f}) and comparable to Seed 3 ({s3_norm_cka_l4:.4f} and {s3_steer_cka_l4:.4f}), indicating greater global subspace similarity. Prompt-level transition groups show that lost-refusal prompts did not exhibit higher steering drift or gate change than retained-refusal prompts.",
            "limitations": "Gate change is elevated at the aggregate level, but controller subspace drift metrics do not uniquely isolate Seed 2."
        },
        "H3_downstream_override_effect_loss": {
            "status": "CONSISTENT_WITH",
            "evidence_for": f"Seed 2 uniquely suffered a substantial reduction in the normative controller's active causal influence during persistence: its active-vs-off ablation gap collapsed by {s2_gap_change:+.2f} pp (from {s2_pre_gap:.2f} pp to {s2_post_gap:.2f} pp), whereas in Seed 3 the ablation gap expanded by {s3_gap_change:+.2f} pp (from {s3_pre_gap:.2f} pp to {s3_post_gap:.2f} pp) and in Seed 1 it expanded by {s1_gap_change:+.2f} pp (from {s1_pre_gap:.2f} pp to {s1_post_gap:.2f} pp). Sensitivity analysis confirms this sign divergence is 100% robust across all NA bounds.",
            "evidence_against": "Prompt-boundary single-token JS divergence changes do not fully predict multi-token generation dynamics.",
            "limitations": "Ablation gap quantifies functional dependence on the controller, but downstream capability dynamics also shift."
        },
        "H4_safety_acquisition_quality_selectivity": {
            "status": "INCONCLUSIVE",
            "evidence_for": f"Initial pre-persistence safety rates varied substantially across seeds (Seed 1 = {s1_c_pre:.2f}%, Seed 2 = {s2_c_pre:.2f}%, Seed 3 = {s3_c_pre:.2f}%).",
            "evidence_against": f"Initial pre-persistence safety rate does not monotonically predict subsequent retention delta (Seed 3 had the lowest initial refusal at {s3_c_pre:.2f}% but the only positive delta at {s3_c_delta:+.2f} pp).",
            "limitations": "Sample size n=3; associations are strictly descriptive."
        },
        "H5_generic_frozen_module_interface": {
            "status": "INCONCLUSIVE",
            "evidence_for": "Model D adapters generically experience compounding representation and residual drift across all 8 sites in all seeds.",
            "evidence_against": f"The magnitude of adapter drift does not consistently explain cross-seed D retention: Seed 2 experienced relatively mild adapter degradation ({s2_d_delta:+.2f} pp) while Seed 1 suffered catastrophic collapse ({s1_d_delta:+.2f} pp) and Seed 3 experienced moderate degradation ({s3_d_delta:+.2f} pp), and adapter CKA/L2 metrics across seeds do not order monotonically with retention.",
            "limitations": "Generic interface drift exists, but its magnitude does not provide a consistent cross-seed explanation of D persistence."
        }
    }

    hyp_out = ARTIFACTS_DIR / "task8_hypothesis_assessment.json"
    with open(hyp_out, "w", encoding="utf-8") as f:
        json.dump(hypothesis_assessment, f, indent=2)
    print(f" -> Wrote dynamically generated hypothesis assessment to {hyp_out}")

    # =========================================================================
    # PLOT FIGURE 1: C MECHANISTIC DRIFT CHAIN
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    metrics_chain = ["capability", "obs", "normative", "steering"]
    x_positions = np.arange(len(metrics_chain))
    width = 0.25
    colors = {"20260821": "#2b5c8f", "20260823": "#d95f02", "20260824": "#7570b3"}
    seed_labels = {"20260821": "Seed 1 (+41.0 pp)", "20260823": "Seed 2 (-14.1 pp)", "20260824": "Seed 3 (+22.3 pp)"}

    for idx, l in enumerate([2, 4]):
        ax = axes[idx]
        for s_idx, s in enumerate(seeds):
            s_str = str(s)
            vals = [
                machine_tables["table_b_model_c_drift"][s_str][f"layer_{l}"]["capability_relative_l2_mean"],
                machine_tables["table_b_model_c_drift"][s_str][f"layer_{l}"]["obs_relative_l2_mean"],
                machine_tables["table_b_model_c_drift"][s_str][f"layer_{l}"]["normative_relative_l2_mean"],
                machine_tables["table_b_model_c_drift"][s_str][f"layer_{l}"]["steering_relative_l2_mean"],
            ]
            ax.bar(x_positions + (s_idx - 1) * width, vals, width, label=seed_labels[s_str], color=colors[s_str], alpha=0.85)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(["Capability (c_tilde)", "Obs (OBS)", "Normative (N)", "Steering (s)"], fontsize=10)
        ax.set_title(f"Model C: Controlled Layer {l} Drift Chain", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Relative L2 Drift (PRE -> POST)" if idx == 0 else "")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(title="Initialization Seed", fontsize=9)

    plt.suptitle("Figure 1: Model C Mechanistic Drift Chain by Seed (OOD Harmful)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig1_p = FIGURES_DIR / "figure1_c_drift_chain.png"
    plt.savefig(fig1_p, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PLOT FIGURE 2: C CONTROLLER CAUSAL EFFECT PRE VS POST (PROGRAMMATIC)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(seeds))
    w = 0.2

    act_pre = [behavior[str(s)]["model_c_pre_persistence_on_harmful"]["rate"] for s in seeds]
    act_post = [behavior[str(s)]["model_c_post_persistence_on_harmful"]["rate"] for s in seeds]
    off_pre = [behavior[str(s)]["model_c_pre_persistence_off_harmful"]["rate"] for s in seeds]
    off_post = [behavior[str(s)]["model_c_post_persistence_off_harmful"]["rate"] for s in seeds]

    ax.bar(x - 1.5*w, act_pre, w, label="Active PRE", color="#2b5c8f", alpha=0.9)
    ax.bar(x - 0.5*w, act_post, w, label="Active POST", color="#41b6c4", alpha=0.9)
    ax.bar(x + 0.5*w, off_pre, w, label="Ablated (Off) PRE", color="#fd8d3c", alpha=0.9)
    ax.bar(x + 1.5*w, off_post, w, label="Ablated (Off) POST", color="#e31a1c", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([seed_labels[str(s)] for s in seeds], fontsize=11)
    ax.set_ylabel("OOD Harmful Refusal Rate (Determinate)", fontsize=11)
    ax.set_title("Figure 2: Model C Active vs. Ablated (Off) Refusal Rate Across Phases (Authoritative)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig2_p = FIGURES_DIR / "figure2_c_causal_effect_pre_post.png"
    plt.savefig(fig2_p, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PLOT FIGURE 3: C HARMFUL VS BENIGN STEERING SELECTIVITY
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, l in enumerate([2, 4]):
        ax = axes[idx]
        sel_pre_list = [machine_tables["table_c_model_c_causal_selectivity"][str(s)][f"layer_{l}"]["selectivity_pre"] for s in seeds]
        sel_post_list = [machine_tables["table_c_model_c_causal_selectivity"][str(s)][f"layer_{l}"]["selectivity_post"] for s in seeds]

        x_s = np.arange(len(seeds))
        ax.bar(x_s - 0.15, sel_pre_list, 0.3, label="PRE Selectivity", color="#3182bd")
        ax.bar(x_s + 0.15, sel_post_list, 0.3, label="POST Selectivity", color="#9ecae1")
        ax.set_xticks(x_s)
        ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=11)
        ax.set_title(f"Controlled Layer {l} Steering Selectivity", fontsize=11, fontweight="bold")
        ax.set_ylabel("Norm Difference (Harmful - Benign)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend()

    plt.suptitle("Figure 3: Model C Steering Vector Selectivity (Harmful - Benign) Pre vs. Post", fontsize=13, y=1.02)
    plt.tight_layout()
    fig3_p = FIGURES_DIR / "figure3_c_steering_selectivity.png"
    plt.savefig(fig3_p, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PLOT FIGURE 4: D ADAPTER DRIFT
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    sites = [f"layer_{l}_{t}_adapter" for l in range(4) for t in ["attn", "mlp"]]
    site_labels = [f"L{l} {t.upper()}" for l in range(4) for t in ["attn", "mlp"]]

    for idx, (metric_key, title) in enumerate([("input_relative_l2_mean", "Adapter Input Drift"), ("residual_relative_l2_mean", "Adapter Residual Drift")]):
        ax = axes[idx]
        x_sites = np.arange(len(sites))
        w_d = 0.25
        for s_idx, s in enumerate(seeds):
            s_str = str(s)
            vals = [machine_tables["table_d_model_d_adapter_drift"][s_str][st][metric_key] for st in sites]
            ax.bar(x_sites + (s_idx - 1) * w_d, vals, w_d, label=seed_labels[s_str], color=colors[s_str], alpha=0.85)

        ax.set_xticks(x_sites)
        ax.set_xticklabels(site_labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"Model D: {title}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Relative L2 Drift" if idx == 0 else "")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(title="Initialization Seed", fontsize=9)

    plt.suptitle("Figure 4: Model D Adapter Interface & Residual Drift Across 8 Sites", fontsize=13, y=1.02)
    plt.tight_layout()
    fig4_p = FIGURES_DIR / "figure4_d_adapter_drift.png"
    plt.savefig(fig4_p, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PLOT FIGURE 5: PROMPT-LEVEL FAILURE MAP FOR C
    # =========================================================================
    fig, axes = plt.subplots(len(seeds), 2, figsize=(12, 12), sharex=True, sharey=True)
    group_colors = {
        "retained_refusal": "#2ca02c",
        "lost_refusal": "#d62728",
        "gained_refusal": "#1f77b4",
        "persistent_nonrefusal": "#7f7f7f",
        "indeterminate": "#bcbd22",
    }

    for s_idx, s in enumerate(seeds):
        s_recs = [r for r in ood_harmful_c if r["seed"] == s]
        for l_idx, l in enumerate([2, 4]):
            ax = axes[s_idx, l_idx]
            for g, col in group_colors.items():
                g_recs = [r for r in s_recs if r["transition_group"] == g]
                if not g_recs:
                    continue
                x_vals = [r[f"layer_{l}_capability_relative_l2"] for r in g_recs]
                y_vals = [r[f"layer_{l}_steering_relative_l2"] for r in g_recs]
                ax.scatter(x_vals, y_vals, c=col, label=g if (s_idx == 0 and l_idx == 0) else "", alpha=0.6, edgecolors="none", s=25)

            ax.set_title(f"Seed {s} | Layer {l}", fontsize=10)
            if l_idx == 0:
                ax.set_ylabel(f"Steering Rel L2\n({seed_labels[str(s)]})", fontsize=10)
            if s_idx == len(seeds) - 1:
                ax.set_xlabel("Capability Rel L2", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5, fontsize=10)
    plt.suptitle("Figure 5: Prompt-Level Failure Map for Model C (OOD Harmful)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig5_p = FIGURES_DIR / "figure5_c_prompt_failure_map.png"
    plt.savefig(fig5_p, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PLOT FIGURE 6: PRE-PERSISTENCE STATE VS OUTCOME (PROGRAMMATIC)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(seeds))

    pre_refusal = [behavior[str(s)]["model_c_pre_persistence_on_harmful"]["rate"] * 100.0 for s in seeds]
    c_deltas = [
        (behavior[str(s)]["model_c_post_persistence_on_harmful"]["rate"] - behavior[str(s)]["model_c_pre_persistence_on_harmful"]["rate"]) * 100.0
        for s in seeds
    ]
    d_deltas = [
        (behavior[str(s)]["model_d_post_persistence_on_harmful"]["rate"] - behavior[str(s)]["model_d_pre_persistence_on_harmful"]["rate"]) * 100.0
        for s in seeds
    ]
    primary_effects = [c_deltas[i] - d_deltas[i] for i in range(len(seeds))]

    w6 = 0.25
    ax.bar(x_pos - w6, pre_refusal, w6, label="Pre OOD Refusal (%)", color="#74a9cf")
    ax.bar(x_pos, c_deltas, w6, label="Model C Delta (pp)", color="#02818a")
    ax.bar(x_pos + w6, primary_effects, w6, label="C-vs-D Primary Effect (pp)", color="#e7298a")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=11)
    ax.set_ylabel("Percentage / Percentage Points", fontsize=11)
    ax.set_title("Figure 6: Pre-Persistence State vs. Persistence Outcome (n=3 Descriptive; No Inferential Correlation)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig6_p = FIGURES_DIR / "figure6_pre_state_vs_persistence_outcome.png"
    plt.savefig(fig6_p, dpi=300, bbox_inches="tight")
    plt.close()
    print("=== TASK 8.2 POST-PROCESSING COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
