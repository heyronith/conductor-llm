"""Tests verifying autograd derivative accuracy against numerical finite differences and observation detach behavior."""

import torch

from ccpt.config import get_micro_dual_stream_config
from ccpt.modeling.dual_stream import CCPTDualStreamModel
from ccpt.training.gradients import set_requires_grad
from ccpt.training.losses import safe_generation_loss


def test_finite_difference_downstream_controller_gradient():
    """Verify analytical autograd gradient against central finite difference for parameters downstream of all observation detaches.

    For the final controller (steering_projections[-1]), there are no downstream stop_gradient operations.
    Therefore, the autograd gradient matches the full numerical derivative to near-machine precision.
    """
    torch.manual_seed(42)
    config = get_micro_dual_stream_config()

    # Instantiate model in float64 for high-precision numerical differentiation
    model = CCPTDualStreamModel(config).to(torch.float64)

    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)

    # Initialize controllers to a deterministic non-zero state
    with torch.no_grad():
        for steer in model.steering_projections:
            steer.weight.fill_(0.02)
        for gate in model.gate_projections:
            gate.weight.fill_(0.02)

    B, T = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    prompt_end_indices = torch.tensor([2, 4])

    # 1. Compute analytical gradient via autograd
    model.zero_grad(set_to_none=True)
    logits, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
    loss = safe_generation_loss(logits, input_ids, prompt_end_indices)
    loss.backward()

    # Final steering projection (layer 4 controller, downstream of all observation detaches)
    target_param = model.steering_projections[-1].weight
    row, col = 0, 0
    grad_analytical = target_param.grad[row, col].item()

    # 2. Compute numerical gradient via central finite difference
    eps = 1e-6
    orig_val = target_param.data[row, col].item()

    # Loss at w + eps
    target_param.data[row, col] = orig_val + eps
    with torch.no_grad():
        logits_pos, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
        loss_pos = safe_generation_loss(logits_pos, input_ids, prompt_end_indices).item()

    # Loss at w - eps
    target_param.data[row, col] = orig_val - eps
    with torch.no_grad():
        logits_neg, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
        loss_neg = safe_generation_loss(logits_neg, input_ids, prompt_end_indices).item()

    # Restore original parameter value
    target_param.data[row, col] = orig_val

    grad_numerical = (loss_pos - loss_neg) / (2.0 * eps)

    abs_err = abs(grad_analytical - grad_numerical)
    rel_err = abs_err / (abs(grad_analytical) + 1e-12)

    assert abs(grad_analytical) > 1e-7, "Analytical gradient is unexpectedly near zero"
    assert rel_err < 1e-5 or abs_err < 1e-7, (
        f"Finite difference mismatch on downstream controller: analytical={grad_analytical:.12e}, "
        f"numerical={grad_numerical:.12e}, rel_err={rel_err:.4e}, abs_err={abs_err:.4e}"
    )


def test_truncated_gradient_upstream_controller_observation_boundary():
    """Verify that parameters upstream of observation detaches receive surrogate/truncated gradients by design.

    For steering_projections[0] (controlling layer 2), parameter perturbations affect C_2 -> C_3 -> C_4 proposal -> N_2.
    However, autograd intentionally severs the C_4 proposal -> N_2 path via stop_gradient(C_tilde_4).
    Thus, autograd computes the intended truncated gradient rather than the full numerical perturbation derivative.
    """
    torch.manual_seed(42)
    config = get_micro_dual_stream_config()
    model = CCPTDualStreamModel(config).to(torch.float64)

    set_requires_grad(model.theta_C, False)
    set_requires_grad(model.theta_N, True)

    with torch.no_grad():
        for steer in model.steering_projections:
            steer.weight.fill_(0.02)
        for gate in model.gate_projections:
            gate.weight.fill_(0.02)

    B, T = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    prompt_end_indices = torch.tensor([2, 4])

    model.zero_grad(set_to_none=True)
    logits, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
    loss = safe_generation_loss(logits, input_ids, prompt_end_indices)
    loss.backward()

    # Upstream steering projection (layer 2 controller)
    target_param = model.steering_projections[0].weight
    row, col = 0, 0
    grad_analytical = target_param.grad[row, col].item()

    eps = 1e-5
    orig_val = target_param.data[row, col].item()

    target_param.data[row, col] = orig_val + eps
    with torch.no_grad():
        logits_pos, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
        loss_pos = safe_generation_loss(logits_pos, input_ids, prompt_end_indices).item()

    target_param.data[row, col] = orig_val - eps
    with torch.no_grad():
        logits_neg, _ = model(input_ids, prompt_end_indices=prompt_end_indices, mode="controlled")
        loss_neg = safe_generation_loss(logits_neg, input_ids, prompt_end_indices).item()

    target_param.data[row, col] = orig_val
    grad_numerical = (loss_pos - loss_neg) / (2.0 * eps)

    # Both must be finite and share sign/scale
    assert torch.isfinite(torch.tensor(grad_analytical))
    assert torch.isfinite(torch.tensor(grad_numerical))
    assert (grad_analytical * grad_numerical) > 0.0, "Analytical and numerical gradients have opposing signs"
