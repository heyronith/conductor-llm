"""Tests for tensor shapes and configuration validity across all models."""

import pytest
import torch

from ccpt.config import (
    BaselineConfig,
    DualStreamConfig,
    get_micro_baseline_config,
    get_micro_dual_stream_config,
    get_smoke_baseline_config,
    get_smoke_dual_stream_config,
)
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel


def test_config_validation():
    """Verify configuration validation assertions."""
    # Valid configs should instantiate without error
    _ = get_smoke_dual_stream_config()
    _ = get_smoke_baseline_config()
    _ = get_micro_dual_stream_config()
    _ = get_micro_baseline_config()

    # Invalid head divisibility
    with pytest.raises(AssertionError):
        DualStreamConfig(d_C=65, n_heads_C=4)

    with pytest.raises(AssertionError):
        BaselineConfig(d_model=65, n_heads=4)

    # Invalid controlled layer count
    with pytest.raises(AssertionError):
        DualStreamConfig(n_layers_N=2, controlled_layers=[2])

    # Controlled layers out of range
    with pytest.raises(AssertionError):
        DualStreamConfig(n_layers_C=4, n_layers_N=2, controlled_layers=[2, 5])

    # Unordered controlled layers
    with pytest.raises(AssertionError):
        DualStreamConfig(n_layers_N=2, controlled_layers=[4, 2])


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("seq_len", [4, 16, 32])
def test_baseline_forward_shapes(batch_size: int, seq_len: int):
    """Test output tensor shapes for Model A."""
    config = get_micro_baseline_config()
    model = ParameterMatchedBaselineModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Without prompt boundary
    logits, risk_logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert risk_logits is None

    # With prompt boundary (variable indices across batch)
    prompt_indices = torch.randint(0, seq_len, (batch_size,))
    logits, risk_logits = model(input_ids, prompt_end_indices=prompt_indices)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert risk_logits is not None
    assert risk_logits.shape == (batch_size,)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("seq_len", [4, 16, 32])
def test_dual_stream_forward_shapes(batch_size: int, seq_len: int):
    """Test output tensor shapes for Model B and Model C."""
    config = get_micro_dual_stream_config()
    model_c = CCPTDualStreamModel(config)
    model_b = JointTrainingDualStreamModel(config)
    model_c.eval()
    model_b.eval()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    prompt_indices = torch.randint(0, seq_len, (batch_size,))

    # 1. Model C in LM mode
    logits_lm, risk_lm = model_c(input_ids, mode="lm")
    assert logits_lm.shape == (batch_size, seq_len, config.vocab_size)
    assert risk_lm is None

    # 2. Model C in controlled mode with diagnostics
    logits_c, risk_c, diags_c = model_c(
        input_ids,
        prompt_end_indices=prompt_indices,
        mode="controlled",
        return_diagnostics=True,
    )
    assert logits_c.shape == (batch_size, seq_len, config.vocab_size)
    assert risk_c is not None
    assert risk_c.shape == (batch_size,)

    assert "gates" in diags_c and "steering" in diags_c and "normative_states" in diags_c
    for layer_idx in config.controlled_layers:
        gate = diags_c["gates"][f"layer_{layer_idx}"]
        steer = diags_c["steering"][f"layer_{layer_idx}"]
        assert gate.shape == (batch_size, seq_len, 1)
        assert steer.shape == (batch_size, seq_len, config.d_C)

    # 3. Model B controlled mode
    logits_b, risk_b = model_b(input_ids, prompt_end_indices=prompt_indices)
    assert logits_b.shape == (batch_size, seq_len, config.vocab_size)
    assert risk_b is not None
    assert risk_b.shape == (batch_size,)
