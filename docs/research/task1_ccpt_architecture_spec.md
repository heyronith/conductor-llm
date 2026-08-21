# CCPT Architecture Specification
## FACT FROM REPOSITORY
- Repository is completely empty. No existing model implementations to follow. All architecture specifications below are defined strictly from the research requirement hypothesis.

## DESIGN DECISION: Tokenizer, Positional Encodings, and Vocabulary
- **Tokenizer**: Llama-2/Mistral style BPE tokenizer.
- **Vocabulary Size**: 32,000 tokens. Frozen for exact parameter arithmetic.
- **Positional Mechanism**: Rotary Position Embedding (RoPE). Applied internally within the Attention modules for both the capability ($C$) and normative ($N$) streams. No additive position embeddings are used.

## DESIGN DECISION: Transformer Block and Information Flow
We have 4 capability layers ($L_C=4$) and 2 normative layers ($L_N=2$).
Normative blocks observe and steer the capability stream every 2 capability blocks (at capability layers 2 and 4). We use bias-free pre-RMSNorm layers with genuine 3-matrix **SwiGLU** MLPs.

**Initialization**:
$N_0 = P_{\text{in}}(\text{stop\_gradient}(C_0))$
where $P_{\text{in}} \in \mathbb{R}^{d_N \times d_C}$ is a trainable linear projection in $\theta_N$, and $C_0$ is the token embedding tensor.

**Capability Proposal (Pre-LN Sequential Operation)**:
At any capability layer $l$, the unsteered proposal $\tilde C_l$ is computed sequentially:
$U_l = C_{l-1} + \text{Attn}_l(\text{RMSNorm}(C_{l-1}))$
$\tilde C_l = U_l + \text{SwiGLU}_l(\text{RMSNorm}(U_l))$

**Normative Observation Mechanism**:
At controlled layers $l \in \{2, 4\}$, the normative stream observes the detached proposal before its own block computation:
$N_{l/2}^{\text{in}} = N_{l/2-1} + W_{\text{obs}, l/2}(\text{stop\_gradient}(\tilde C_l))$
where $W_{\text{obs}, l/2} \in \mathbb{R}^{d_N \times d_C}$ is an unshared layer-specific projection matrix.
Then the normative block processes it:
$V_{l/2} = N_{l/2}^{\text{in}} + \text{Attn}_{N, l/2}(\text{RMSNorm}(N_{l/2}^{\text{in}}))$
$N_{l/2} = V_{l/2} + \text{SwiGLU}_{N, l/2}(\text{RMSNorm}(V_{l/2}))$

**Steering Mathematics & Bounds**:
The controller generates gates and additive steering using unshared, layer-specific matrices:
$g_l = 1 + \tanh(W_{g, l/2} N_{l/2}) \times \alpha$
$s_l = \tanh(W_{s, l/2} N_{l/2}) \times \beta$
where $W_{g, l/2} \in \mathbb{R}^{1 \times d_N}$ and $W_{s, l/2} \in \mathbb{R}^{d_C \times d_N}$.
- **Frozen Constants**: $\alpha = 0.1$ and $\beta = 1.0$.
- **Initialization**: $W_g$ and $W_s$ are strictly initialized to 0. This ensures $g_l = 1$ and $s_l = \vec{0}$, making CCPT exactly identical to the baseline Transformer at initialization.

Finally, we apply steering to the capability residual update:
$C_l = C_{l-1} + g_l (\tilde C_l - C_{l-1}) + s_l$

## DESIGN DECISION: Gradient Flow
### Mode A — Ordinary LM Training (for CCPT)
- The normative pathway is completely inactive (its forward pass is skipped).
- $\partial L_{LM} / \partial \theta_N = 0$ is guaranteed because $\theta_N$ is not in the computational graph.

### Mode B — Normative Training (for CCPT)
- Capability parameters $\theta_C$ are explicitly frozen using `requires_grad = False` on all parameters in $\theta_C$.
- The forward pass for $C$ is computed in the normal autograd context (NOT in `torch.no_grad()`), so gradients can flow through the $C$ operations.
- $L_{\text{norm}} \rightarrow \text{logits} \rightarrow \text{frozen } C \text{ operations} \rightarrow g_l, s_l \rightarrow \theta_N$.
- This satisfies $\partial L_{\text{norm}} / \partial \theta_C = 0$ while training $\theta_N$.

## DESIGN DECISION: Normative Objective
$L_N = L_{\text{risk}} + \lambda_{\text{gen}} L_{\text{safe\_generation}}$
- **Frozen Constant**: $\lambda_{\text{gen}} = 1.0$.

- **Risk Objective**: Binary classification evaluated strictly at the **prompt-boundary token** (the last token of the user's prompt). Output dim 1, BCEWithLogitsLoss.
  $\text{risk} = W_{\text{risk}} \cdot \text{RMSNorm}(N_2)$
- **Safe-generation Objective**: Teacher-forcing next-token prediction for the safe continuation tokens. The prompt tokens are masked. Gradients flow backwards through the frozen LM head, through the steered $C$ activations, into the controller outputs $g$ and $s$, updating $\theta_N$.

## DESIGN DECISION: Exact Inference Path (Causal Token-by-Token)
1. **Cache Initialization**: $K_C, V_C$ (capability KV cache) and $K_N, V_N$ (normative KV cache) initialized to empty.
2. For each token $t$:
   a. Compute $C_0^{(t)} = \text{Embedding}(token_t)$.
   b. Compute $N_0^{(t)} = P_{\text{in}}(C_0^{(t)})$.
   c. For capability layer 1: $\tilde C_1^{(t)} = \text{CapabilityBlock}(C_0^{(t)}, K_C^{(1)}, V_C^{(1)})$. $C_1^{(t)} = \tilde C_1^{(t)}$.
   d. For capability layer 2: $\tilde C_2^{(t)} = \text{CapabilityBlock}(C_1^{(t)}, K_C^{(2)}, V_C^{(2)})$.
   e. Observe: $N_1^{\text{in}, (t)} = N_0^{(t)} + W_{\text{obs}, 1}(\tilde C_2^{(t)})$.
   f. Normative processing: $N_1^{(t)} = \text{TransformerBlock}_N(N_1^{\text{in}, (t)}, K_N^{(1)}, V_N^{(1)})$.
   g. Steer: $C_2^{(t)} = C_1^{(t)} + g_2^{(t)}(\tilde C_2^{(t)} - C_1^{(t)}) + s_2^{(t)}$.
   h. Repeat for layers 3 and 4 similarly, with $N_2$ observing $\tilde C_4$.
   i. Compute logits from $C_4^{(t)}$ through final LN and LM head. Sample next token.
No future tokens are accessed during the update for token $t$.
