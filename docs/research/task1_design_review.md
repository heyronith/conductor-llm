# CCPT Design Review
## Strongest reasons the architecture may work
- Strict gradient isolation prevents catastrophic forgetting of safety behaviors during normal LM capability updates.
- The controller can learn to dynamically suppress specific harmful activations rather than relying on global weight changes.

## Strongest reasons it may fail
- The controller is ignored: The capability stream might adapt its representations to bypass the controller's interventions if $s_l$ is too small, or if $g_l$ defaults to 1 strongly.
- Training instability: The alternating batch training might cause distribution drift where $N$ expects old $C$ activations.

## Mathematical concerns
- Optimization of multiplicative gating might be difficult due to variance in the capability residual update $\tilde C_l - C_{l-1}$. The bounding mechanisms ($\alpha=0.1, \beta=1.0$) should prevent explosive instability but might limit expressiveness.
- SwiGLU activations introduce more variance in the MLP output compared to GeLU, making the steering magnitude $s_l$ potentially less impactful unless scaled appropriately.

## Experimental confounds
- Model B (joint-training dual-stream) correctly isolates the structural components from the optimization components. However, it does not perfectly isolate each direction of the gradient firewall ($L_{LM} \rightarrow N$ vs $L_{norm} \rightarrow C$). If further granularity is required, separate ablation models will be needed.
- Model A has been meticulously parameter matched by expanding its MLP dimension ($d_{ff}=2496$) to ensure that performance gains are not simply due to the added parameters of the normative stream in CCPT.

## Implementation hazards
- Accidental parameter sharing in PyTorch optimizers.
- Using `torch.no_grad()` by mistake on the frozen $C$ pathway during Mode B, completely severing the gradient path to the controller.
- Getting the KV caching wrong during inference for the interleaved $C$ and $N$ blocks.
