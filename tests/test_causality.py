"""Tests verifying autoregressive causality and absence of future-token risk leakage."""

import torch

from ccpt.config import get_micro_baseline_config, get_micro_dual_stream_config
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel


def test_causality_token_invariance():
    """Modifying token at position j must not affect logits at positions < j."""
    base_config = get_micro_baseline_config()
    dual_config = get_micro_dual_stream_config()

    model_a = ParameterMatchedBaselineModel(base_config).eval()
    model_b = JointTrainingDualStreamModel(dual_config).eval()
    model_c = CCPTDualStreamModel(dual_config).eval()

    # Also perturb controller weights away from 0 to test active steering causality
    with torch.no_grad():
        for gate in model_c.gate_projections:
            gate.weight.normal_(0.0, 0.5)
        for steer in model_c.steering_projections:
            steer.weight.normal_(0.0, 0.5)
        for gate in model_b.gate_projections:
            gate.weight.normal_(0.0, 0.5)
        for steer in model_b.steering_projections:
            steer.weight.normal_(0.0, 0.5)

    batch_size = 2
    seq_len = 16
    cutoff_j = 8

    # Create base sequence x1
    x1 = torch.randint(0, dual_config.vocab_size, (batch_size, seq_len))
    # Create modified sequence x2 identical up to cutoff_j, but different from cutoff_j onward
    x2 = x1.clone()
    x2[:, cutoff_j:] = (x2[:, cutoff_j:] + 1) % dual_config.vocab_size

    models_and_modes = [
        (model_a, {}),
        (model_b, {"mode": "controlled"}),
        (model_c, {"mode": "lm"}),
        (model_c, {"mode": "controlled"}),
    ]

    with torch.no_grad():
        for model, kwargs in models_and_modes:
            logits_1, _ = model(x1, **kwargs)
            logits_2, _ = model(x2, **kwargs)

            # Positions 0 to cutoff_j - 1 must be exactly identical
            prefix_1 = logits_1[:, :cutoff_j, :]
            prefix_2 = logits_2[:, :cutoff_j, :]

            assert torch.allclose(prefix_1, prefix_2, atol=1e-6, rtol=1e-5), (
                f"Future token change leaked into past token logits for {model.__class__.__name__} ({kwargs})"
            )


def test_risk_prediction_no_continuation_leakage():
    """Risk logit at prompt boundary p must be completely invariant to continuation tokens at positions > p."""
    base_config = get_micro_baseline_config()
    dual_config = get_micro_dual_stream_config()

    model_a = ParameterMatchedBaselineModel(base_config).eval()
    model_b = JointTrainingDualStreamModel(dual_config).eval()
    model_c = CCPTDualStreamModel(dual_config).eval()

    prompt_len = 6
    total_len = 16
    prompt_boundary = prompt_len - 1  # 0-indexed position of last prompt token

    # Sequence A: Prompt followed by continuation A
    seq_a = torch.randint(0, dual_config.vocab_size, (2, total_len))

    # Sequence B: Same prompt prefix, but completely different continuation
    seq_b = seq_a.clone()
    seq_b[:, prompt_len:] = torch.randint(0, dual_config.vocab_size, (2, total_len - prompt_len))

    prompt_end_indices = torch.tensor([prompt_boundary, prompt_boundary])

    with torch.no_grad():
        # Test Model A
        _, risk_a1 = model_a(seq_a, prompt_end_indices=prompt_end_indices)
        _, risk_a2 = model_a(seq_b, prompt_end_indices=prompt_end_indices)
        assert torch.allclose(risk_a1, risk_a2, atol=1e-6), "Model A risk head leaked continuation tokens"

        # Test Model B
        _, risk_b1 = model_b(seq_a, prompt_end_indices=prompt_end_indices)
        _, risk_b2 = model_b(seq_b, prompt_end_indices=prompt_end_indices)
        assert torch.allclose(risk_b1, risk_b2, atol=1e-6), "Model B risk head leaked continuation tokens"

        # Test Model C
        _, risk_c1 = model_c(seq_a, prompt_end_indices=prompt_end_indices, mode="controlled")
        _, risk_c2 = model_c(seq_b, prompt_end_indices=prompt_end_indices, mode="controlled")
        assert torch.allclose(risk_c1, risk_c2, atol=1e-6), "Model C risk head leaked continuation tokens"
