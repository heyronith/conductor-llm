"""Reproducible diagnostic gradient probe script for CCPT and control architectures."""

import torch

from ccpt.config import get_micro_dual_stream_config
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel
from ccpt.training.gradients import gradient_summary, set_requires_grad
from ccpt.training.losses import (
    causal_lm_loss,
    risk_classification_loss,
    safe_generation_loss,
)


def run_probe():
    torch.manual_seed(1234)
    config = get_micro_dual_stream_config()

    # Deterministic synthetic batch
    B, T = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    prompt_end_indices = torch.tensor([2, 4])
    risk_labels = torch.tensor([1, 0])

    experiments = []

    def get_row(name, model):
        c_norm = gradient_summary(model.theta_C)["grad_norm"]
        n_norm = gradient_summary(model.theta_N)["grad_norm"]
        pin_norm = gradient_summary(model.p_in.parameters())["grad_norm"]
        nblock_norm = gradient_summary(model.normative_layers.parameters())["grad_norm"]
        gate_norm = gradient_summary(model.gate_projections.parameters())["grad_norm"]
        steer_norm = gradient_summary(model.steering_projections.parameters())["grad_norm"]
        risk_norm = gradient_summary(model.risk_head.parameters())["grad_norm"]

        return {
            "Experiment": name,
            "C Norm": f"{c_norm:.4f}",
            "N Norm": f"{n_norm:.4f}",
            "P_in Norm": f"{pin_norm:.4f}",
            "N-Block Norm": f"{nblock_norm:.4f}",
            "Gate Norm": f"{gate_norm:.4f}",
            "Steer Norm": f"{steer_norm:.4f}",
            "Risk Norm": f"{risk_norm:.4f}",
        }

    # 1. CCPT LM loss
    model = CCPTDualStreamModel(config)
    set_requires_grad(model.theta_C, True)
    set_requires_grad(model.theta_N, True)
    logits, _ = model(input_ids, mode="lm")
    loss = causal_lm_loss(logits, input_ids)
    loss.backward()
    experiments.append(get_row("1. CCPT LM Loss (mode=lm)", model))

    # 2. CCPT risk-only loss
    model = CCPTDualStreamModel(config)
    set_requires_grad(model.theta_C, True)
    set_requires_grad(model.theta_N, True)
    _, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
    loss = risk_classification_loss(risk_logits, risk_labels)
    loss.backward()
    experiments.append(get_row("2. CCPT Risk-Only (C trainable)", model))

    # 3. CCPT safe-generation-only at zero controller init
    model = CCPTDualStreamModel(config)
    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)
    logits, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
    loss = safe_generation_loss(logits, input_ids, prompt_end_indices)
    loss.backward()
    experiments.append(get_row("3. CCPT Safe-Gen (Zero Controller Init)", model))

    # 4. CCPT safe-generation after controller perturbation
    model = CCPTDualStreamModel(config)
    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)
    with torch.no_grad():
        for steer in model.steering_projections:
            steer.weight.normal_(0.0, 0.05)
        for gate in model.gate_projections:
            gate.weight.normal_(0.0, 0.05)
    logits, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
    loss = safe_generation_loss(logits, input_ids, prompt_end_indices)
    loss.backward()
    experiments.append(get_row("4. CCPT Safe-Gen (Perturbed Controller)", model))

    # 5. CCPT combined normative loss
    model = CCPTDualStreamModel(config)
    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)
    logits, risk_logits = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
    loss = risk_classification_loss(risk_logits, risk_labels) + 1.0 * safe_generation_loss(
        logits, input_ids, prompt_end_indices
    )
    loss.backward()
    experiments.append(get_row("5. CCPT Combined Normative (C frozen)", model))

    # 6. Model B LM loss
    model_b = JointTrainingDualStreamModel(config)
    set_requires_grad(model_b.theta_C, True)
    set_requires_grad(model_b.theta_N, True)
    logits, _ = model_b(input_ids, mode="controlled")
    loss = causal_lm_loss(logits, input_ids)
    loss.backward()
    experiments.append(get_row("6. Model B Joint LM Loss", model_b))

    # Print Markdown / formatted table
    headers = [
        "Experiment",
        "C Norm",
        "N Norm",
        "P_in Norm",
        "N-Block Norm",
        "Gate Norm",
        "Steer Norm",
        "Risk Norm",
    ]
    col_widths = [max(len(row[h]) for row in experiments + [{h: h}]) + 2 for h in headers]

    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "".join("-" * (w - 2) + "  " for w in col_widths)

    print("=" * len(header_line))
    print("CCPT Task 3 Diagnostic Gradient Probe (Deterministic Micro Configuration)")
    print("=" * len(header_line))
    print(header_line)
    print(sep_line)
    for row in experiments:
        print("".join(row[h].ljust(w) for h, w in zip(headers, col_widths)))
    print("=" * len(header_line))


if __name__ == "__main__":
    run_probe()
