"""Tests verifying strict, disjoint parameter ownership between capability and normative parameter groups."""

from ccpt.config import get_micro_dual_stream_config, get_smoke_dual_stream_config
from ccpt.modeling.dual_stream import CCPTDualStreamModel, JointTrainingDualStreamModel


def test_parameter_ownership_disjoint_and_exhaustive():
    """Verify that theta_C and theta_N partition all model parameters with zero intersection."""
    for config in [get_micro_dual_stream_config(), get_smoke_dual_stream_config()]:
        model_c = CCPTDualStreamModel(config)
        model_b = JointTrainingDualStreamModel(config)

        for model in [model_c, model_b]:
            theta_C_params = list(model.theta_C)
            theta_N_params = list(model.theta_N)
            all_params = list(model.parameters())

            theta_C_ids = {id(p) for p in theta_C_params}
            theta_N_ids = {id(p) for p in theta_N_params}
            all_param_ids = {id(p) for p in all_params}

            # 1. Zero overlap between capability and normative parameter IDs
            intersection = theta_C_ids.intersection(theta_N_ids)
            assert len(intersection) == 0, f"Found overlapping parameter IDs between theta_C and theta_N: {intersection}"

            # 2. Exhaustive partition: union must equal all model parameters exactly
            union = theta_C_ids.union(theta_N_ids)
            assert union == all_param_ids, "theta_C and theta_N do not form an exhaustive partition of model parameters"

            # 3. No duplicate parameters within theta_C or theta_N
            assert len(theta_C_params) == len(theta_C_ids), "Duplicate parameters found in theta_C"
            assert len(theta_N_params) == len(theta_N_ids), "Duplicate parameters found in theta_N"


def test_specific_component_parameter_ownership():
    """Verify that specific architectural components belong to the expected parameter group."""
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config)

    theta_C_ids = {id(p) for p in model.theta_C}
    theta_N_ids = {id(p) for p in model.theta_N}

    # Embedding and Capability blocks must be in theta_C
    assert id(model.embedding.weight) in theta_C_ids
    assert id(model.capability_final_norm.weight) in theta_C_ids
    for block in model.capability_layers:
        for p in block.parameters():
            assert id(p) in theta_C_ids
            assert id(p) not in theta_N_ids

    # P_in, Observation, Normative blocks, Controllers, and Risk head must be in theta_N
    assert id(model.p_in.weight) in theta_N_ids
    assert id(model.normative_final_norm.weight) in theta_N_ids
    assert id(model.risk_head.weight) in theta_N_ids

    for obs in model.obs_projections:
        assert id(obs.weight) in theta_N_ids
        assert id(obs.weight) not in theta_C_ids

    for block in model.normative_layers:
        for p in block.parameters():
            assert id(p) in theta_N_ids
            assert id(p) not in theta_C_ids

    for gate in model.gate_projections:
        assert id(gate.weight) in theta_N_ids
        assert id(gate.weight) not in theta_C_ids

    for steer in model.steering_projections:
        assert id(steer.weight) in theta_N_ids
        assert id(steer.weight) not in theta_C_ids
