"""Tests asserting exact parameter count arithmetic for Smoke and Micro configurations."""

from ccpt.config import (
    get_micro_baseline_config,
    get_micro_dual_stream_config,
    get_smoke_baseline_config,
    get_smoke_dual_stream_config,
)
from ccpt.modeling.baseline import ParameterMatchedBaselineModel
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_smoke_parameter_counts():
    """Assert exact parameter counts for Smoke configurations as mathematically specified in Task 1 & Task 2."""
    smoke_dual_config = get_smoke_dual_stream_config()
    smoke_base_config = get_smoke_baseline_config()

    model_c = CCPTDualStreamModel(smoke_dual_config)
    model_b = JointTrainingDualStreamModel(smoke_dual_config)
    model_a = ParameterMatchedBaselineModel(smoke_base_config)

    # 1. Model C (CCPT) exact counts
    total_c = count_parameters(model_c)
    theta_C_count = sum(p.numel() for p in model_c.theta_C)
    theta_N_count = sum(p.numel() for p in model_c.theta_N)

    assert theta_C_count == 33_165_824, f"Expected theta_C 33,165,824, got {theta_C_count}"
    assert theta_N_count == 2_754_560, f"Expected theta_N 2,754,560, got {theta_N_count}"
    assert total_c == 35_920_384, f"Expected Model C total 35,920,384, got {total_c}"

    # 2. Model B (Joint-Training Control) exact counts
    total_b = count_parameters(model_b)
    assert total_b == 35_920_384, f"Expected Model B total 35,920,384, got {total_b}"

    # 3. Model A (Parameter-Matched Baseline) exact counts
    core_a_count = sum(p.numel() for p in model_a.core_lm_parameters())
    total_a = count_parameters(model_a)

    assert core_a_count == 35_918_336, f"Expected Model A core LM 35,918,336, got {core_a_count}"
    assert total_a == 35_918_848, f"Expected Model A total 35,918,848, got {total_a}"

    # Verify structural matching between Model B and Model C
    assert total_b == total_c


def test_micro_parameter_counts():
    """Assert parameter counts for Micro configurations."""
    micro_dual_config = get_micro_dual_stream_config()
    micro_base_config = get_micro_baseline_config()

    model_c = CCPTDualStreamModel(micro_dual_config)
    model_b = JointTrainingDualStreamModel(micro_dual_config)
    model_a = ParameterMatchedBaselineModel(micro_base_config)

    total_c = count_parameters(model_c)
    total_b = count_parameters(model_b)
    total_a = count_parameters(model_a)

    assert total_b == total_c
    assert total_c > 0
    assert total_a > 0
