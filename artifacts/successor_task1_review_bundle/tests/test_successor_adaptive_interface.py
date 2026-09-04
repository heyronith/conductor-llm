"""CPU-only tests for successor Task 1 adaptive-interface falsification."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from ccpt.config import get_micro_dual_stream_config, get_smoke_dual_stream_config
from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.successor.calibration import assert_zero_eval_overlap, build_successor_calibration_reference
from ccpt.successor.cohort import PRIMARY_COHORT, resolve_cohort
from ccpt.successor.criteria import assess_hypothesis
from ccpt.successor.partition import build_parameter_partition, classify_parameter_name
from ccpt.successor.retrofit import (
    AdaptiveInterfaceWrapper,
    RepairVariant,
    build_variant_bundle,
    freeze_module,
    hash_existing_parameters,
)
from ccpt.successor.training import continuation_token_mask, fit_loss


def _micro_model() -> CCPTDualStreamModel:
    return CCPTDualStreamModel(get_micro_dual_stream_config())


def test_parameter_partition_covers_all_names():
    m = CCPTDualStreamModel(get_smoke_dual_stream_config())
    part = build_parameter_partition(m)
    names = set(dict(m.named_parameters()).keys())
    covered = set()
    for lst in part["buckets"].values():
        covered.update(lst)
    assert covered == names
    assert classify_parameter_name("p_in.weight") == "OLD_OBSERVER_INTERFACE"
    assert classify_parameter_name("gate_projections.0.weight") == "OLD_ACTUATOR_INTERFACE"
    assert classify_parameter_name("risk_head.weight") == "RISK_READOUT"


def test_cohort_resolver_structure_offline():
    cohort = resolve_cohort(check_volume=False)
    assert cohort["required_primary_pairs"] == 4
    assert cohort["primary_post_horizon"] == 1000
    assert len(PRIMARY_COHORT) == 4
    assert cohort["gpu_allowed"] is True


def test_calibration_isolation_zero_overlap():
    ref = build_successor_calibration_reference()
    assert_zero_eval_overlap(ref)
    assert ref["CALIBRATION_FINAL_TEST_OVERLAP"] == 0
    assert ref["wildguard_test_overlap"] == 0
    assert ref["beavertails_ood_overlap"] == 0
    assert ref["xstest_overlap"] == 0
    assert ref["fit_record_count"] > 0


def test_budget_and_generic_match_smoke():
    m = CCPTDualStreamModel(get_smoke_dual_stream_config())
    b = build_variant_bundle(m, observer_rank=32, actuator_rank=32)
    assert b["within_1pct_budget"]
    assert b["generic_within_1pct_match"]
    assert b["combined_percent_of_base"] < 1.0


@pytest.mark.parametrize(
    "variant",
    [
        RepairVariant.OBSERVER,
        RepairVariant.ACTUATOR,
        RepairVariant.COMBINED,
        RepairVariant.GENERIC,
    ],
)
def test_identity_init_matches_frozen_post(variant: RepairVariant):
    torch.manual_seed(0)
    base = _micro_model()
    base.eval()
    freeze_module(base)
    x = torch.randint(0, base.config.vocab_size, (2, 16))
    with torch.no_grad():
        ref_logits, _ = base(x, mode="controlled", controller_scale=1.0)

    bundle = build_variant_bundle(base, observer_rank=4, actuator_rank=4)
    wrap = AdaptiveInterfaceWrapper(
        copy.deepcopy(base),
        variant,
        observer_rank=4,
        actuator_rank=4,
        generic_rank=bundle["generic_rank"] if variant == RepairVariant.GENERIC else None,
    )
    wrap.eval()
    with torch.no_grad():
        out_logits, _ = wrap(x, mode="controlled", controller_scale=1.0)
    assert torch.allclose(out_logits, ref_logits, atol=1e-5, rtol=1e-5)


def test_controller_scale_zero_unchanged_for_interface_variants():
    torch.manual_seed(1)
    base = _micro_model()
    freeze_module(base)
    x = torch.randint(0, base.config.vocab_size, (2, 12))
    with torch.no_grad():
        off_ref, _ = base(x, mode="controlled", controller_scale=0.0)

    for variant in (RepairVariant.OBSERVER, RepairVariant.ACTUATOR, RepairVariant.COMBINED):
        wrap = AdaptiveInterfaceWrapper(
            copy.deepcopy(base), variant, observer_rank=4, actuator_rank=4
        )
        if wrap.actuator is not None:
            for p in wrap.actuator.parameters():
                p.data.add_(0.1)
        with torch.no_grad():
            off, _ = wrap(x, mode="controlled", controller_scale=0.0)
        assert torch.allclose(off, off_ref, atol=1e-5, rtol=1e-5)


def test_optimizer_groups_exclude_base():
    base = _micro_model()
    wrap = AdaptiveInterfaceWrapper(base, RepairVariant.COMBINED, observer_rank=4, actuator_rank=4)
    opt = torch.optim.AdamW(wrap.trainable_parameters(), lr=1e-3)
    wrap.assert_optimizer_owns_only_repairs(opt)
    for p in wrap.base.parameters():
        assert p.requires_grad is False


def test_synthetic_optimization_updates_adapters_only():
    torch.manual_seed(2)
    teacher = _micro_model()
    student_base = _micro_model()
    with torch.no_grad():
        student_base.embedding.weight.add_(0.01)

    wrap = AdaptiveInterfaceWrapper(
        student_base, RepairVariant.COMBINED, observer_rank=4, actuator_rank=4
    )
    for p in wrap.trainable_parameters():
        p.data.add_(0.01)

    pre_hash = hash_existing_parameters(wrap.base)
    opt = torch.optim.AdamW(wrap.trainable_parameters(), lr=1e-2)
    x = torch.randint(0, teacher.config.vocab_size, (2, 20))
    ends = torch.tensor([8, 8])

    teacher.eval()
    with torch.no_grad():
        t_logits, t_risk = teacher(
            x, prompt_end_indices=ends, mode="controlled", controller_scale=1.0
        )

    losses = []
    for _ in range(5):
        opt.zero_grad()
        s_logits, s_risk = wrap(
            x, prompt_end_indices=ends, mode="controlled", controller_scale=1.0
        )
        mask = continuation_token_mask(x, ends)
        loss, _ = fit_loss(t_logits, s_logits, mask, t_risk, s_risk)
        loss.backward()
        for p in wrap.base.parameters():
            assert p.grad is None
        for p in wrap.trainable_parameters():
            assert p.grad is not None
        opt.step()
        losses.append(float(loss.item()))

    post_hash = hash_existing_parameters(wrap.base)
    assert pre_hash == post_hash
    assert losses[-1] < losses[0]


def test_adapter_state_roundtrip():
    base = _micro_model()
    wrap = AdaptiveInterfaceWrapper(base, RepairVariant.COMBINED, observer_rank=4, actuator_rank=4)
    sd = {n: p.detach().clone() for n, p in wrap.trainable_named_parameters()}
    for p in wrap.trainable_parameters():
        p.data.zero_()
    name_to_param = dict(wrap.trainable_named_parameters())
    for n, t in sd.items():
        name_to_param[n].data.copy_(t)
    for n, t in sd.items():
        assert torch.equal(name_to_param[n].data, t)


def test_hypothesis_assessment_machine_derived():
    seeds = []
    for seed in [1, 2, 3, 4]:
        seeds.append(
            {
                "seed": seed,
                "H_POST": 0.40,
                "H_COMBINED": 0.20,
                "H_GENERIC": 0.35,
                "B_POST": 0.50,
                "B_COMBINED": 0.52,
                "B_GENERIC": 0.60,
                "CE_POST": 3.0,
                "CE_COMBINED": 3.01,
                "gap_PRE": 0.40,
                "gap_POST": 0.05,
                "gap_COMBINED": 0.35,
            }
        )
    out = assess_hypothesis(seeds)
    assert out["decision"] in {
        "SUPPORTED_FOR_FULL_ARCHITECTURE_FOLLOWUP",
        "REJECT_WRONG_FIREWALL_EXPLANATION_AS_PRIMARY",
        "INCONCLUSIVE",
    }
    assert "HARMFUL_RESPONSE_CRITERION" in out["criteria"]


def test_causal_shapes_forward():
    base = _micro_model()
    wrap = AdaptiveInterfaceWrapper(base, RepairVariant.COMBINED, observer_rank=4, actuator_rank=4)
    x = torch.randint(0, base.config.vocab_size, (3, 11))
    logits, risk = wrap(x, prompt_end_indices=torch.tensor([5, 5, 5]), mode="controlled")
    assert logits.shape == (3, 11, base.config.vocab_size)
    assert risk is not None and risk.shape == (3,)
