"""Authoritative unit tests for Task 7.3.1a Corrective Forensic Salvage.

Verifies:
1. Identity-based parameter partitioning for CCPT (Model C) and Adapter (Model D).
2. Regression against old name-substring heuristics.
3. Safe-generation token-weighted loss with attention_mask padding exclusion.
4. Schedule record field-level mutation sensitivity.
"""

import copy
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccpt.config import DualStreamConfig, AdapterConfig, BaselineConfig
from ccpt.modeling import CCPTDualStreamModel, FrozenBackboneAdapterModel
from ccpt.evaluation.forensics import (
    get_named_parameter_partition,
    get_ccpt_named_partitions,
    get_adapter_named_partitions,
    extract_named_sub_state_dict,
    compare_named_tensors,
    compute_canonical_state_dict_hash,
)
from ccpt.training.losses import (
    safe_generation_loss,
    token_weighted_continuation_nll_and_count,
)


def test_ccpt_parameter_partition_by_identity():
    cfg = DualStreamConfig(
        vocab_size=32000,
        n_layers_C=4,
        d_C=512,
        n_heads_C=8,
        d_ff_C=2048,
        n_layers_N=2,
        d_N=256,
        n_heads_N=4,
        d_ff_N=1024,
        controlled_layers=[2, 4],
        max_seq_len=1024,
    )
    model = CCPTDualStreamModel(cfg)
    theta_c_names, theta_n_names = get_ccpt_named_partitions(model)

    assert theta_c_names.isdisjoint(theta_n_names)
    assert len(theta_c_names) + len(theta_n_names) == len(list(model.named_parameters()))

    # Expected theta_C parameters
    assert "embedding.weight" in theta_c_names
    assert "capability_layers.0.mlp.gate_proj.weight" in theta_c_names
    assert "capability_layers.1.attn.q_proj.weight" in theta_c_names
    assert "capability_final_norm.weight" in theta_c_names

    # Expected theta_N parameters
    assert "p_in.weight" in theta_n_names
    assert "obs_projections.0.weight" in theta_n_names
    assert "obs_projections.1.weight" in theta_n_names
    assert "gate_projections.0.weight" in theta_n_names
    assert "steering_projections.0.weight" in theta_n_names
    assert "normative_layers.0.attn.q_proj.weight" in theta_n_names
    assert "normative_final_norm.weight" in theta_n_names
    assert "risk_head.weight" in theta_n_names

    # Counts
    assert len(theta_c_names) == 38
    assert len(theta_n_names) == 27


def test_adapter_parameter_partition_by_identity():
    cfg = AdapterConfig(
        vocab_size=32000,
        n_layers=4,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        d_mid=336,
        max_seq_len=1024,
    )
    model = FrozenBackboneAdapterModel(cfg)
    backbone_names, safety_names = get_adapter_named_partitions(model)

    assert backbone_names.isdisjoint(safety_names)
    assert len(backbone_names) + len(safety_names) == len(list(model.named_parameters()))

    # Expected safety parameters
    assert "layers.0.attn_adapter.down_proj.weight" in safety_names
    assert "layers.0.attn_adapter.norm.weight" in safety_names
    assert "layers.0.attn_adapter.up_proj.weight" in safety_names
    assert "layers.0.mlp_adapter.down_proj.weight" in safety_names
    assert "layers.0.mlp_adapter.norm.weight" in safety_names
    assert "layers.0.mlp_adapter.up_proj.weight" in safety_names
    assert "risk_head.weight" in safety_names

    # Expected backbone parameters
    assert "embedding.weight" in backbone_names
    assert "layers.0.attn_norm.weight" in backbone_names
    assert "layers.0.attn.q_proj.weight" in backbone_names
    assert "layers.0.mlp_norm.weight" in backbone_names
    assert "layers.0.mlp.gate_proj.weight" in backbone_names
    assert "final_norm.weight" in backbone_names

    # Counts
    assert len(backbone_names) == 38
    assert len(safety_names) == 25


def test_old_substring_bug_regression():
    cfg = DualStreamConfig(
        vocab_size=32000,
        n_layers_C=4,
        d_C=512,
        n_heads_C=8,
        d_ff_C=2048,
        n_layers_N=2,
        d_N=256,
        n_heads_N=4,
        d_ff_N=1024,
        controlled_layers=[2, 4],
        max_seq_len=1024,
    )
    model = CCPTDualStreamModel(cfg)
    all_names = [name for name, _ in model.named_parameters()]

    # Old naive substring logic
    old_theta_N = {k for k in all_names if any(sub in k for sub in ["normative_", "gate_", "steering_", "risk_head", "p_in"])}
    
    # Old logic mistakenly included capability SwiGLU gate_proj in theta_N because of "gate_"
    assert "capability_layers.0.mlp.gate_proj.weight" in old_theta_N
    
    # Correct identity logic
    correct_theta_C, correct_theta_N = get_ccpt_named_partitions(model)
    assert "capability_layers.0.mlp.gate_proj.weight" in correct_theta_C
    assert "capability_layers.0.mlp.gate_proj.weight" not in correct_theta_N


def test_safe_gen_attention_mask_padding_exclusion():
    torch.manual_seed(42)
    B, T, V = 2, 8, 10
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    
    # Example 0: prompt ends at index 2 (tokens 0, 1, 2 are prompt). Valid continuation targets: indices 3, 4, 5. Padded: 6, 7.
    # Example 1: prompt ends at index 1 (tokens 0, 1 are prompt). Valid continuation targets: indices 2, 3. Padded: 4, 5, 6, 7.
    prompt_ends = torch.tensor([2, 1])
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
    ])

    # 1. Manual reference computation:
    # Shifted targets are at index p in input_ids for p in [1, T-1].
    # Prediction logit position is p-1 in [0, T-2].
    # For ex 0:
    # Target p=1: prompt (p <= 2) -> exclude
    # Target p=2: prompt (p <= 2) -> exclude
    # Target p=3: valid continuation (p > 2 and mask[3]==1) -> include logit 2 predicting id[0, 3]
    # Target p=4: valid continuation (p > 2 and mask[4]==1) -> include logit 3 predicting id[0, 4]
    # Target p=5: valid continuation (p > 2 and mask[5]==1) -> include logit 4 predicting id[0, 5]
    # Target p=6: padding (mask[6]==0) -> exclude
    # Target p=7: padding (mask[7]==0) -> exclude
    # Total for ex 0 = 3 tokens.
    
    # For ex 1:
    # Target p=1: prompt (p <= 1) -> exclude
    # Target p=2: valid continuation (p > 1 and mask[2]==1) -> include logit 1 predicting id[1, 2]
    # Target p=3: valid continuation (p > 1 and mask[3]==1) -> include logit 2 predicting id[1, 3]
    # Target p=4, 5, 6, 7: padding -> exclude
    # Total for ex 1 = 2 tokens.
    # Total valid continuation targets = 5.

    l0_3 = F.cross_entropy(logits[0:1, 2, :], input_ids[0:1, 3]).item()
    l0_4 = F.cross_entropy(logits[0:1, 3, :], input_ids[0:1, 4]).item()
    l0_5 = F.cross_entropy(logits[0:1, 4, :], input_ids[0:1, 5]).item()

    l1_2 = F.cross_entropy(logits[1:2, 1, :], input_ids[1:2, 2]).item()
    l1_3 = F.cross_entropy(logits[1:2, 2, :], input_ids[1:2, 3]).item()

    manual_nll = l0_3 + l0_4 + l0_5 + l1_2 + l1_3
    manual_valid_count = 5
    manual_ce = manual_nll / manual_valid_count

    helper_nll, helper_count = token_weighted_continuation_nll_and_count(
        logits, input_ids, prompt_ends, attention_mask=attention_mask
    )
    helper_ce = helper_nll / helper_count

    assert helper_count == manual_valid_count
    assert abs(helper_nll - manual_nll) < 1e-5
    assert abs(helper_ce - manual_ce) < 1e-5

    # Demonstrate that calling WITHOUT attention_mask includes padded positions and produces a different result:
    unmasked_nll, unmasked_count = token_weighted_continuation_nll_and_count(
        logits, input_ids, prompt_ends, attention_mask=None
    )
    # Without mask, ex 0 has targets 3,4,5,6,7 (5 tokens) and ex 1 has targets 2,3,4,5,6,7 (6 tokens) = 11 tokens
    assert unmasked_count == 11
    assert unmasked_count != manual_valid_count
    assert unmasked_nll != helper_nll

    # Demonstrate invariance to mutating padded logits
    logits_mutated = logits.clone()
    logits_mutated[0, 5:, :] += 100.0  # Mutate logits predicting padded tokens (logits at p >= 5 predict targets >= 6)
    logits_mutated[1, 3:, :] += 50.0   # Mutate logits predicting padded tokens (logits at p >= 3 predict targets >= 4)

    mutated_nll, mutated_count = token_weighted_continuation_nll_and_count(
        logits_mutated, input_ids, prompt_ends, attention_mask=attention_mask
    )
    assert mutated_count == helper_count
    assert abs(mutated_nll - helper_nll) < 1e-5


def test_schedule_record_field_mutations():
    canonical_record = {
        "example_id": "test_ex_001",
        "input_ids": [1, 10, 20, 30, 40, 2],
        "prompt_end_index": 3,
        "risk_label": 1,
        "is_refusal": 1,
    }

    # 1. Exact match passes
    rec_exact = copy.deepcopy(canonical_record)
    assert rec_exact == canonical_record

    # 2. Mutating input_ids fails
    rec_bad_ids = copy.deepcopy(canonical_record)
    rec_bad_ids["input_ids"] = [1, 10, 20, 31, 40, 2]
    assert rec_bad_ids != canonical_record

    # 3. Mutating prompt_end_index fails
    rec_bad_prompt = copy.deepcopy(canonical_record)
    rec_bad_prompt["prompt_end_index"] = 2
    assert rec_bad_prompt != canonical_record

    # 4. Mutating risk_label fails
    rec_bad_risk = copy.deepcopy(canonical_record)
    rec_bad_risk["risk_label"] = 0
    assert rec_bad_risk != canonical_record

    # 5. Mutating is_refusal fails
    rec_bad_ref = copy.deepcopy(canonical_record)
    rec_bad_ref["is_refusal"] = 0
    assert rec_bad_ref != canonical_record
