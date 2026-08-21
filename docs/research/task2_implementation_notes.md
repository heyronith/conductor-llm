# Task 2 Implementation Notes

## 1. Repository Facts Discovered Before Editing
- The repository was initialized with standard Python 3.9+ runtime support (`python3 --version` yielded Python 3.9.6).
- PyTorch 2.8.0 and pytest 8.4.2 are installed in the Python environment.
- Prior to Task 2, no application source code, package configurations, or ML models existed in the repository.
- Git repository was initialized with no commits; Task 1 specifications (`task1_ccpt_architecture_spec.md`, `task1_experiment_contract.md`, `task1_design_review.md`, `task1_repo_inventory.md`) and `.cursor/rules/ccpt-research.mdc` were present.

## 2. Files and Components Implemented
- `pyproject.toml`: Minimal modern project configuration for the `ccpt` package.
- `README.md`: Setup and test instructions.
- `src/ccpt/config.py`:
  - `BaselineConfig`: Dataclass configuring Model A.
  - `DualStreamConfig`: Dataclass configuring Model B and Model C with strict validation.
  - Factory functions for smoke and micro configurations (`get_smoke_baseline_config`, `get_smoke_dual_stream_config`, `get_micro_baseline_config`, `get_micro_dual_stream_config`).
- `src/ccpt/modeling/layers.py`:
  - `RMSNorm`: Bias-free Root Mean Square Layer Normalization with learnable scaling.
  - `RotaryEmbedding`: Rotary Position Embedding (RoPE) with precomputed tables and device/dtype dynamic handling.
  - `CausalSelfAttention`: Multi-head self-attention with RoPE on query/key and native PyTorch causal SDP attention.
  - `SwiGLU`: Genuine 3-matrix bias-free MLP ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$).
  - `TransformerBlock`: Sequential pre-RMSNorm residual block ($U = x + \text{Attn}(\text{Norm}(x)), \text{out} = U + \text{SwiGLU}(\text{Norm}(U))$).
- `src/ccpt/modeling/baseline.py`:
  - `ParameterMatchedBaselineModel` (Model A): Standard causal decoder-only Transformer with tied embeddings ($d_{ff}=2496$) and auxiliary prompt-boundary risk classifier.
- `src/ccpt/modeling/dual_stream.py`:
  - `CCPTDualStreamModel` (Model C): Protected dual-stream architecture with frozen zero-initialized controllers, prompt-boundary risk classifier, and support for `mode="lm"` vs `mode="controlled"`.
  - `JointTrainingDualStreamModel` (Model B): Control model structurally identical to Model C with controlled forward pass by default.
- `tests/`:
  - `test_forward_shapes.py`: Verifies tensor shapes, batch sizes, sequence lengths, diagnostics, and config assertions.
  - `test_parameter_counts.py`: Asserts exact parameter counts for smoke and micro configurations.
  - `test_identity_initialization.py`: Proves exact numerical equivalence between LM and controlled modes at zero-initialization, and verifies controller output bounds under saturation.
  - `test_causality.py`: Asserts strict autoregressive causality and absence of future-token risk leakage.
  - `test_parameter_ownership.py`: Verifies disjoint and exhaustive parameter partitioning between $\theta_C$ and $\theta_N$.

## 3. Implementation Defaults Introduced in Task 2
The following engineering constants were frozen for implementation:
- `rms_norm_eps = 1e-6`
- `rope_theta = 10000.0`
- `init_std = 0.02` (standard normal initialization for linear layers and embeddings; RMSNorm weights initialized to 1.0; controllers initialized to 0.0)
- `dropout = 0.0` (bias-free architecture with deterministic evaluation)

## 4. Mapping from Task 1 Equations to Code
- Token embeddings: `self.embedding = nn.Embedding(vocab_size, d_C)`
- Initial normative state $N_0 = P_{\text{in}}(\text{stop\_gradient}(C_0))$: `self.p_in(c_0.detach())`
- Capability proposal $\tilde C_l$: `c_tilde = self.capability_layers[l_idx - 1](prev_c)`
- Normative observation $N_{l/2}^{\text{in}} = N_{l/2-1} + W_{\text{obs}, l/2}(\text{stop\_gradient}(\tilde C_l))$: `n_in = n + self.obs_projections[k](c_tilde.detach())`
- Normative block $N_{l/2}$: `n = self.normative_layers[k](n_in)`
- Gate projection $g_l = 1 + \alpha \tanh(W_{g, l/2} N_{l/2})$: `g_l = 1.0 + self.config.alpha * torch.tanh(self.gate_projections[k](n))`
- Steering projection $s_l = \beta \tanh(W_{s, l/2} N_{l/2})$: `s_l = self.config.beta * torch.tanh(self.steering_projections[k](n))`
- Residual capability update $C_l = C_{l-1} + g_l \odot (\tilde C_l - C_{l-1}) + s_l$: `c = prev_c + g_l * (c_tilde - prev_c) + s_l`
- LM Logits $\text{logits} = W_{\text{embed}}^T \text{RMSNorm}(C_{L_C})$: `F.linear(self.capability_final_norm(c), self.embedding.weight)`
- Normative Risk Head $\text{risk} = W_{\text{risk}} \text{RMSNorm}(N_{L_N}[p])$: `self.risk_head(self.normative_final_norm(n)[batch_idx, prompt_idx])`

## 5. Parameter Ownership: $\theta_C$ vs $\theta_N$
- **$\theta_C$ (Capability parameters)**:
  - `embedding.weight`: $32000 \times 512 = 16,384,000$
  - 4 capability blocks (each has 4 Attention projections of $512 \times 512$, 3 SwiGLU projections of $512 \times 2048$, 2 RMSNorms of $512$): $4 \times 4,195,328 = 16,781,312$
  - `capability_final_norm.weight`: $512$
  - Tied LM head: 0 (reuses `embedding.weight`)
  - **Total $\theta_C$**: **33,165,824** parameters.
- **$\theta_N$ (Normative / Controller parameters)**:
  - `p_in.weight`: $512 \times 256 = 131,072$
  - 2 observation projections $W_{\text{obs}, 1}, W_{\text{obs}, 2}$: $2 \times (512 \times 256) = 262,144$
  - 2 normative blocks (each has 4 Attention projections of $256 \times 256$, 3 SwiGLU projections of $256 \times 1024$, 2 RMSNorms of $256$): $2 \times 1,049,088 = 2,098,176$
  - 2 gate projections $W_{g, 1}, W_{g, 2}$: $2 \times (256 \times 1) = 512$
  - 2 steering projections $W_{s, 1}, W_{s, 2}$: $2 \times (256 \times 512) = 262,144$
  - `normative_final_norm.weight`: $256$
  - `risk_head.weight`: $256 \times 1 = 256$
  - **Total $\theta_N$**: **2,754,560** parameters.
- **Total Model C / Model B**: **35,920,384** parameters.
- Rigorous automated unit tests assert that $\theta_C \cap \theta_N = \emptyset$ and $\theta_C \cup \theta_N = \text{all parameters}$.

## 6. Model A Parameter-Count Clarification
- Model A core LM ($d=512, n_{\text{layers}}=4, d_{ff}=2496$ SwiGLU, tied embeddings): **35,918,336** parameters.
- Model A auxiliary prompt-boundary risk head (`baseline_risk_head` of shape $512 \times 1$ operating on final normalized state): **512** parameters.
- Model A total parameter count: **35,918,848** parameters.
- The remaining difference between Model A and CCPT (35,920,384) is only 1,536 parameters (~0.0043%), providing virtually perfect parameter matching.

## 7. Deferred Work
The following items are intentionally deferred to future research tasks as specified in the project roadmap:
- **Task 3**: Rigorous gradient isolation, autograd graph boundaries, and optimizer step invariance verification.
- **Task 4**: Data pipeline, tokenization, batch preparation, and prompt masking.
- **Task 5**: Micro-overfitting, failure mode detection, and convergence diagnostics.
- **Task 6**: Modal GPU training scripts and full smoke experiments.
- **Task 7**: Mechanistic activation steering and safety evaluation benchmarks.
