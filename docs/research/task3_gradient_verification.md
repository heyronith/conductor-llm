# Task 3: Gradient Firewall and Optimization Verification

## 1. Overview and Core Research Invariant
Task 3 formally verifies the mathematical gradient topology and optimization firewall of the Constitutional Control-Plane Transformer (CCPT) across all training modes.

The core autograd invariants proven in this task are:
1. **Ordinary LM Training (Mode A)**:
   $$\frac{\partial L_{\text{LM}}}{\partial \theta_N} = 0, \quad \frac{\partial L_{\text{LM}}}{\partial \theta_C} \neq 0$$
   Ordinary language-model cross-entropy updates capability parameters $\theta_C$ while normative/controller parameters $\theta_N$ receive zero gradients and remain bit-for-bit unchanged.
2. **Normative Safety Training (Mode B)**:
   $$\frac{\partial L_N}{\partial \theta_C} = 0, \quad \frac{\partial L_N}{\partial \theta_N} \neq 0$$
   With capability parameters frozen ($\theta_C.\text{requires\_grad} = \text{False}$), gradients from $L_N = L_{\text{risk}} + \lambda_{\text{gen}} L_{\text{safe}}$ flow backwards through the frozen capability operations to train $\theta_N$, while $\theta_C$ remains strictly bit-for-bit unchanged.
3. **Architectural Observation Firewall**:
   $$\frac{\partial L_{\text{risk}}}{\partial \theta_C} = 0$$
   Even when $\theta_C$ is marked as trainable (`requires_grad = True`), the explicit `stop_gradient` / `detach()` on the $C \to N$ observation edges prevents risk loss from leaking gradients into capability parameters.

---

## 2. Required Future Training Semantics

Future trainer implementations (e.g. Task 5/6) must adhere to the following verified execution policies:

### Policy 1: CCPT Ordinary LM Step
```python
# 1. Ensure capability parameters are trainable and normative parameters are bypassed
set_requires_grad(model.theta_C, True)
optimizer_C.zero_grad(set_to_none=True)

# 2. Execute purely in LM mode (skipping N and controller blocks entirely)
logits, _ = model(input_ids, mode="lm")
loss_lm = causal_lm_loss(logits, input_ids)

# 3. Backward and step
loss_lm.backward()
optimizer_C.step()
```

### Policy 2: CCPT Normative Training Step
```python
# 1. Freeze capability parameters; enable normative parameters
set_requires_grad(model.theta_C, False)
set_requires_grad(model.theta_N, True)
optimizer_N.zero_grad(set_to_none=True)

# 2. Execute in controlled mode WITHOUT torch.no_grad()
# (Crucial: capability forward operations must remain in the autograd graph so gradients
# flow from logits through frozen capability operations into controller outputs)
logits, risk_logits = model(
    input_ids,
    prompt_end_indices=prompt_end_indices,
    mode="controlled",
)

loss_normative = risk_classification_loss(risk_logits, risk_labels) + 1.0 * safe_generation_loss(
    logits, input_ids, prompt_end_indices
)

# 3. Backward and step
loss_normative.backward()
optimizer_N.step()

# 4. Re-enable capability trainability before the next LM batch
set_requires_grad(model.theta_C, True)
```

> [!CAUTION]
> **Autograd Invariant**: Never wrap the capability forward pass in `torch.no_grad()` during normative training. Parameter freezing (`requires_grad = False`) prevents parameter updates while preserving tensor differentiability with respect to controller activations. Wrapping in `torch.no_grad()` would sever the computation graph and prevent safe-generation loss from reaching the controller.

---

## 3. Discovered Zero-Controller Gradient Dynamics

Automated gradient probes confirmed the expected mathematical behavior at initialization:
1. **At exact zero initialization ($W_g = 0, W_s = 0$)**:
   - $\frac{\partial s}{\partial N} \propto W_s = 0$ and $\frac{\partial g}{\partial N} \propto W_g = 0$.
   - Consequently, safe-generation loss $L_{\text{safe}}$ backpropagates directly into $W_g$ and $W_s$ (gradient norm $> 0$), but does not yet propagate into deeper normative layers ($P_{\text{in}}$, $W_{\text{obs}}$, normative blocks).
   - Simultaneously, risk loss $L_{\text{risk}}$ connects directly from the prompt boundary to the risk head and deeper normative layers, immediately providing rich gradients to $P_{\text{in}}$, $W_{\text{obs}}$, and normative Transformer blocks.
2. **Transition After Controller Movement**:
   - As soon as controller weights $W_g, W_s$ move away from zero (via optimizer updates or small perturbation), $\frac{\partial s}{\partial N} \neq 0$, and safe-generation loss smoothly propagates through the controller into all deeper normative blocks and observation projections.

---

## 4. Surrogate / Truncated Gradients across Observation Boundaries

An important mathematical property of the CCPT architecture is that autograd gradients for parameters upstream of observation boundaries are intentionally **surrogate/truncated gradients**:
- For the final controller (e.g. `steering_projections[-1]`, controlling layer 4), there are no downstream observation detach operations. Autograd analytical derivatives match central finite differences to near-machine precision (relative error $\approx 1.38 \times 10^{-7}$).
- For upstream controllers (e.g. `steering_projections[0]`, controlling layer 2), parameter perturbations affect $C_2 \to C_3 \to \tilde C_4 \to N_2 \to \text{controller}_2 \to \text{loss}$. However, the architecture deliberately applies $\text{stop\_gradient}(\tilde C_4)$ at the observation edge. Consequently, autograd computes the intended surrogate gradient that excludes feedback through downstream normative observation.

---

## 5. Summary of Verification Results

| Verification Check | Target Invariant | Result | Evidence |
| :--- | :--- | :--- | :--- |
| **Test 1** | LM Mode Firewall | $\partial L_{\text{LM}} / \partial \theta_N = 0$ | PASSED (N grad norm = 0.0000) |
| **Test 2** | LM Optimizer Isolation | $\theta_N$ bit-for-bit identical | PASSED (`parameters_bit_identical` == True) |
| **Test 3** | Observation Edge Detach | $\partial L_{\text{risk}} / \partial \theta_C = 0$ with C trainable | PASSED (C grad norm = 0.0000) |
| **Test 4** | Normative Training Topology | $\partial L_N / \partial \theta_C = 0, \partial L_N / \partial \theta_N > 0$ | PASSED (C grad norm = 0.0000, N grad norm = 3.5161) |
| **Test 5** | Normative Optimizer Isolation | $\theta_C$ bit-for-bit identical | PASSED (`parameters_bit_identical` == True) |
| **Test 6** | Frozen Capability Autograd Path | $L_{\text{safe}} \to$ frozen C ops $\to$ controller $\to$ N | PASSED (N-block grad norm = 0.6027, C grad norm = 0.0) |
| **Test 7** | Zero-Init Controller Dynamics | $L_{\text{safe}}$ trains controllers only at init | PASSED (Controller grad $> 0$, deep N grad == 0) |
| **Test 8** | Risk Loss Immediate Training | $L_{\text{risk}}$ trains deep N at init | PASSED (Risk head, P_in, W_obs, N-blocks grad $> 0$) |
| **Test 9** | Perturbed Generation Dynamics | $L_{\text{safe}}$ reaches deep N after controller moves | PASSED (P_in, W_obs, N-blocks grad $> 0$) |
| **Test 10** | Combined Loss Coverage | All 11 logical modules receive signal | PASSED (All module grad norms $> 0$ and finite) |
| **Test 11 & 12** | Model B Joint Optimization | $L_{\text{LM}}$ trains both C and N streams | PASSED (Model B C norm = 4.7111, N norm = 0.1091) |
| **Test 13** | Model A Baseline Reference | Normal gradient propagation for LM and risk | PASSED (Core LM and risk head receive valid gradients) |
| **Test 14** | Tied Embedding Storage | Embedding weight shared by pointer in $\theta_C$ | PASSED (Zero duplicate parameters, no leak into $\theta_N$) |
| **Test 15** | Freeze/Unfreeze Reversibility | Parameter values preserved across freeze cycles | PASSED (Bit-identical values before and after) |
| **Test 16** | Stale Gradient Protection | `zero_grad(set_to_none=True)` clears gradients | PASSED (All grad None across mode switches) |
| **Test 17** | Safe-Generation Masking | Off-by-one boundary correctness | PASSED (Exact token masking verified) |
| **Test 18a** | Downstream Controller FD Check | Machine-precision autograd derivative match | PASSED (Relative error $\approx 1.38 \times 10^{-7}$ in float64) |
| **Test 18b** | Upstream Controller Truncated FD | Truncated gradient due to observation detach | PASSED (Consistent sign and order of magnitude) |
