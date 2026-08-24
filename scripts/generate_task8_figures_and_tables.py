"""Task 8.1: Corrected Synthesis and Figure Generation Script.

Loads mechanistic diagnostics and authoritative tri-state WildGuard behavioral records
programmatically, enforces strict numerical parity assertions, reconciles transition
groups, evaluates frozen hypotheses with calibrated scientific language, and plots Figures 1-6.
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
            # On
            h_on = br1[ph][m]["ood_behavioral"]["harmful_eval"]
            b_on = br1[ph][m]["ood_behavioral"]["benign_eval"]
            # Off
            m_off_key = f"{m}_scale_0"
            h_off = br1[ph][m_off_key]["ood_behavioral"]["harmful_eval"]
            b_off = br1[ph][m_off_key]["ood_behavioral"]["benign_eval"]

            behavior["20260821"][f"{m}_{ph}_on_harmful"] = {
                "yes": h_on["yes_count"], "no": h_on["no_count"], "na": h_on["na_count"],
                "determinate_n": h_on["determinate_n"], "rate": h_on["safe_refusal_determinate_rate"]
            }
            behavior["20260821"][f"{m}_{ph}_on_benign"] = {
                "yes": b_on["yes_count"], "no": b_on["no_count"], "na": b_on["na_count"],
                "determinate_n": b_on["determinate_n"], "rate": b_on["over_refusal_determinate_rate"]
            }
            behavior["20260821"][f"{m}_{ph}_off_harmful"] = {
                "yes": h_off["yes_count"], "no": h_off["no_count"], "na": h_off["na_count"],
                "determinate_n": h_off["determinate_n"], "rate": h_off["safe_refusal_determinate_rate"]
            }
            behavior["20260821"][f"{m}_{ph}_off_benign"] = {
                "yes": b_off["yes_count"], "no": b_off["no_count"], "na": b_off["na_count"],
                "determinate_n": b_off["determinate_n"], "rate": b_off["over_refusal_determinate_rate"]
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
                            "determinate_n": rr["determinate_n"], "rate": rate
                        }

    # =========================================================================
    # HARD NUMERICAL ASSERTIONS (SECTION 5 OF TASK 8.1 PROMPT)
    # =========================================================================
    # Seed 1
    assert np.isclose(behavior["20260821"]["model_c_pre_persistence_on_harmful"]["rate"], 0.87500000)
    assert np.isclose(behavior["20260821"]["model_c_post_persistence_on_harmful"]["rate"], 0.86328125)
    assert np.isclose(behavior["20260821"]["model_d_pre_persistence_on_harmful"]["rate"], 0.93359375)
    assert np.isclose(behavior["20260821"]["model_d_post_persistence_on_harmful"]["rate"], 0.51171875)

    # Seed 2
    assert behavior["20260823"]["model_c_pre_persistence_on_harmful"]["yes"] == 220
    assert behavior["20260823"]["model_c_post_persistence_on_harmful"]["yes"] == 173
    assert np.isclose(behavior["20260823"]["model_c_pre_persistence_on_harmful"]["rate"], 220 / 256)
    assert np.isclose(behavior["20260823"]["model_c_post_persistence_on_harmful"]["rate"], 173 / 256)
    assert behavior["20260823"]["model_d_pre_persistence_on_harmful"]["yes"] == 238
    assert behavior["20260823"]["model_d_post_persistence_on_harmful"]["yes"] == 227
    assert np.isclose(behavior["20260823"]["model_d_pre_persistence_on_harmful"]["rate"], 238 / 256)
    assert np.isclose(behavior["20260823"]["model_d_post_persistence_on_harmful"]["rate"], 227 / 256)

    # Seed 3
    assert behavior["20260824"]["model_c_pre_persistence_on_harmful"]["yes"] == 171
    assert behavior["20260824"]["model_c_post_persistence_on_harmful"]["yes"] == 201
    assert np.isclose(behavior["20260824"]["model_c_pre_persistence_on_harmful"]["rate"], 171 / 256)
    assert np.isclose(behavior["20260824"]["model_c_post_persistence_on_harmful"]["rate"], 201 / 256)
    assert behavior["20260824"]["model_d_pre_persistence_on_harmful"]["yes"] == 246
    assert behavior["20260824"]["model_d_post_persistence_on_harmful"]["yes"] == 219
    assert np.isclose(behavior["20260824"]["model_d_pre_persistence_on_harmful"]["rate"], 246 / 256)
    assert np.isclose(behavior["20260824"]["model_d_post_persistence_on_harmful"]["rate"], 219 / 256)

    return behavior


def main():
    print("=== TASK 8.1: CORRECTED SYNTHESIS & FIGURE GENERATION ===", flush=True)

    # 1. Load and assert authoritative behavioral values
    behavior = load_authoritative_behavioral_data()
    print(" -> Authoritative behavioral metrics loaded and asserted successfully.")

    # 2. Load existing frozen mechanistic diagnostic summary
    summary_path = ARTIFACTS_DIR / "task8_mechanistic_summary.json"
    assert summary_path.exists(), f"Missing mechanistic diagnostic summary: {summary_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        diag_data = json.load(f)

    records = diag_data["per_prompt_records"]
    cka_summary = diag_data["cka_summary"]
    seeds = [20260821, 20260823, 20260824]

    # Privacy verification on per-prompt records
    forbidden_keys = {"prompt", "response", "input_ids", "hidden_tensors"}
    for r in records[:50]:
        assert not any(k in r for k in forbidden_keys), f"Privacy violation in per-prompt records: {r.keys()}"

    # 3. Transition Group Reconciliation
    ood_harmful_c = [r for r in records if r["model"] == "model_c" and r["dataset"] == "ood_beavertails" and r["prompt_type"] == "harmful"]
    ood_benign_c = [r for r in records if r["model"] == "model_c" and r["dataset"] == "ood_beavertails" and r["prompt_type"] == "benign"]
    ood_harmful_d = [r for r in records if r["model"] == "model_d" and r["dataset"] == "ood_beavertails" and r["prompt_type"] == "harmful"]

    transition_summary: Dict[str, Any] = {}
    for s in seeds:
        s_str = str(s)
        s_recs = [r for r in ood_harmful_c if r["seed"] == s]
        groups = ["retained_refusal", "lost_refusal", "gained_refusal", "persistent_nonrefusal", "indeterminate"]
        transition_summary[s_str] = {}
        for g in groups:
            g_recs = [r for r in s_recs if r["transition_group"] == g]
            n_c = len(g_recs)
            transition_summary[s_str][g] = {
                "count": n_c,
                "percentage": n_c / len(s_recs) * 100.0 if s_recs else 0.0,
            }
            if n_c > 0:
                transition_summary[s_str][g].update({
                    "layer_2_capability_rel_l2_mean": float(np.mean([r["layer_2_capability_relative_l2"] for r in g_recs])),
                    "layer_2_steering_rel_l2_mean": float(np.mean([r["layer_2_steering_relative_l2"] for r in g_recs])),
                    "layer_4_capability_rel_l2_mean": float(np.mean([r["layer_4_capability_relative_l2"] for r in g_recs])),
                    "layer_4_steering_rel_l2_mean": float(np.mean([r["layer_4_steering_relative_l2"] for r in g_recs])),
                    "layer_4_gate_change_mean": float(np.mean([r["layer_4_gate_absolute_change"] for r in g_recs])),
                })

    # Assert mathematical transition reconciliation for Seeds 2 & 3
    # Seed 2
    s2_t = transition_summary["20260823"]
    pre_yes_s2 = s2_t["retained_refusal"]["count"] + s2_t["lost_refusal"]["count"]
    post_yes_s2 = s2_t["retained_refusal"]["count"] + s2_t["gained_refusal"]["count"]
    assert pre_yes_s2 == behavior["20260823"]["model_c_pre_persistence_on_harmful"]["yes"]
    assert post_yes_s2 == behavior["20260823"]["model_c_post_persistence_on_harmful"]["yes"]

    # Seed 3
    s3_t = transition_summary["20260824"]
    pre_yes_s3 = s3_t["retained_refusal"]["count"] + s3_t["lost_refusal"]["count"]
    post_yes_s3 = s3_t["retained_refusal"]["count"] + s3_t["gained_refusal"]["count"]
    assert pre_yes_s3 == behavior["20260824"]["model_c_pre_persistence_on_harmful"]["yes"]
    assert post_yes_s3 == behavior["20260824"]["model_c_post_persistence_on_harmful"]["yes"]

    trans_out = ARTIFACTS_DIR / "task8_transition_group_summary.json"
    with open(trans_out, "w", encoding="utf-8") as f:
        json.dump(transition_summary, f, indent=2)
    print(f" -> Reconciled and saved transition group summary to {trans_out}")

    # 4. Build Model C & Model D Drift Tables
    c_drift_table: Dict[str, Any] = {}
    for s in seeds:
        s_str = str(s)
        s_recs = [r for r in ood_harmful_c if r["seed"] == s]
        c_drift_table[s_str] = {}
        for l in [2, 4]:
            c_drift_table[s_str][f"layer_{l}"] = {
                "capability_relative_l2_mean": float(np.mean([r[f"layer_{l}_capability_relative_l2"] for r in s_recs])),
                "capability_cka": cka_summary.get(f"seed_{s}_model_c_ood_beavertails_harmful_c_tilde_{l}", 0.0),
                "obs_relative_l2_mean": float(np.mean([r[f"layer_{l}_obs_relative_l2"] for r in s_recs])),
                "obs_cka": cka_summary.get(f"seed_{s}_model_c_ood_beavertails_harmful_obs_{l}", 0.0),
                "normative_relative_l2_mean": float(np.mean([r[f"layer_{l}_normative_relative_l2"] for r in s_recs])),
                "normative_cka": cka_summary.get(f"seed_{s}_model_c_ood_beavertails_harmful_norm_{l}", 0.0),
                "steering_relative_l2_mean": float(np.mean([r[f"layer_{l}_steering_relative_l2"] for r in s_recs])),
                "steering_cka": cka_summary.get(f"seed_{s}_model_c_ood_beavertails_harmful_steer_{l}", 0.0),
                "gate_absolute_change_mean": float(np.mean([r[f"layer_{l}_gate_absolute_change"] for r in s_recs])),
            }

    d_drift_table: Dict[str, Any] = {}
    for s in seeds:
        s_str = str(s)
        s_recs = [r for r in ood_harmful_d if r["seed"] == s]
        d_drift_table[s_str] = {}
        for l_idx in range(4):
            for a_type in ["attn", "mlp"]:
                site = f"layer_{l_idx}_{a_type}_adapter"
                d_drift_table[s_str][site] = {
                    "input_relative_l2_mean": float(np.mean([r[f"{site}_input_relative_l2"] for r in s_recs])),
                    "residual_relative_l2_mean": float(np.mean([r[f"{site}_residual_relative_l2"] for r in s_recs])),
                }

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
                c_drift_table[s_str][f"layer_{l}"]["capability_relative_l2_mean"],
                c_drift_table[s_str][f"layer_{l}"]["obs_relative_l2_mean"],
                c_drift_table[s_str][f"layer_{l}"]["normative_relative_l2_mean"],
                c_drift_table[s_str][f"layer_{l}"]["steering_relative_l2_mean"],
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
        sel_pre_list = []
        sel_post_list = []
        for s in seeds:
            h_recs = [r for r in ood_harmful_c if r["seed"] == s]
            b_recs = [r for r in ood_benign_c if r["seed"] == s]
            pre_h = np.mean([r[f"layer_{l}_steering_norm_pre"] for r in h_recs])
            pre_b = np.mean([r[f"layer_{l}_steering_norm_pre"] for r in b_recs])
            post_h = np.mean([r[f"layer_{l}_steering_norm_post"] for r in h_recs])
            post_b = np.mean([r[f"layer_{l}_steering_norm_post"] for r in b_recs])
            sel_pre_list.append(pre_h - pre_b)
            sel_post_list.append(post_h - post_b)

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
            vals = [d_drift_table[s_str][st][metric_key] for st in sites]
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

    # =========================================================================
    # REASSESS HYPOTHESES HONESTLY & CONSERVATIVELY
    # =========================================================================
    # H1 Guard: Ensure we do NOT claim Seed2 capability drift > Seed1
    assert c_drift_table["20260823"]["layer_4"]["capability_relative_l2_mean"] < c_drift_table["20260821"]["layer_4"]["capability_relative_l2_mean"]

    hypothesis_assessment = {
        "H1_capability_interface_drift": {
            "status": "INCONCLUSIVE",
            "evidence_for": "Continuation pretraining alters capability representations and projected observation vectors at controlled layers across all seeds.",
            "evidence_against": "Seed 2 (the negative persistence seed) does NOT exhibit greater capability proposal drift at Layer 4 (0.4426) than positive Seed 1 (0.5408), and its capability CKA (0.9654) is higher than Seed 1 (0.9572). OBS drift is very close across all three seeds (0.2215-0.2341).",
            "limitations": "Interface drift measures do not monotonically order seed-level persistence outcomes at n=3."
        },
        "H2_functional_controller_drift": {
            "status": "CONSISTENT_WITH",
            "evidence_for": "Seed 2 exhibits the largest Layer 4 gate absolute change (0.0120, nearly 2x higher than Seed 1's 0.0065 and Seed 3's 0.0084), indicating functional controller perturbation despite bit-identical theta_N.",
            "evidence_against": "Normative state relative L2 (0.2305) and steering relative L2 (0.2476) in Seed 2 are comparable to Seed 1 (0.2294 and 0.2458). Prompt-level transition groups show that lost-refusal prompts did not have higher gate/steering drift than retained-refusal prompts.",
            "limitations": "Consistency is weak-to-moderate and driven primarily by the gate-change metric at the aggregate level."
        },
        "H3_downstream_override_effect_loss": {
            "status": "CONSISTENT_WITH",
            "evidence_for": "In Seed 2, the causal behavioral ablation gap between active and ablated controller conditions dropped by 19.53 pp (from 43.08% to 23.55%), whereas in Seed 3 the ablation gap expanded by +33.14 pp (from 12.47% to 45.60%) and in Seed 1 it expanded by +6.49 pp (from 37.89% to 44.38%).",
            "evidence_against": "Prompt-boundary next-token JS divergence changes do not fully track multi-token behavioral ablation gaps.",
            "limitations": "Ablation gap quantifies functional dependence on the controller, but downstream capability dynamics also shift."
        },
        "H4_safety_acquisition_quality_selectivity": {
            "status": "INCONCLUSIVE",
            "evidence_for": "Initial pre-persistence safety rates varied substantially across seeds (Seed 1 = 87.50%, Seed 2 = 85.94%, Seed 3 = 66.80%), disproving the prior assumption that initial safety acquisition was tightly clustered.",
            "evidence_against": "Initial pre-persistence safety rate does not monotonically predict subsequent persistence delta (Seed 3 had the lowest initial refusal at 66.80% but the only positive delta at +11.72 pp).",
            "limitations": "Sample size n=3; associations are strictly descriptive."
        },
        "H5_generic_frozen_module_interface": {
            "status": "CONSISTENT_WITH",
            "evidence_for": "Model D adapters exhibit compounding input drift and residual drift across all 8 sites in all seeds, demonstrating that frozen safety modules attached to evolving backbones generically experience interface instability.",
            "evidence_against": "Model D adapter retention in Seed 2 (-4.30 pp) was far more resilient than in Seed 1 (-42.19 pp), indicating varying architectural sensitivity to interface drift.",
            "limitations": "Model C and Model D use fundamentally different steering mechanics."
        }
    }

    hyp_out = ARTIFACTS_DIR / "task8_hypothesis_assessment.json"
    with open(hyp_out, "w", encoding="utf-8") as f:
        json.dump(hypothesis_assessment, f, indent=2)
    print(f" -> Wrote corrected hypothesis assessment to {hyp_out}")
    print("=== TASK 8.1 POST-PROCESSING COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
