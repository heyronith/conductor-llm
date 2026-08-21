# CCPT Experiment Contract
## Hypotheses
- **H1 (Capability preservation)**: CCPT should not catastrophically degrade ordinary LM capability relative to appropriate controls.
- **H2 (Safety generalization)**: CCPT should improve OOD safety behavior relative to matched controls.
- **H3 (Persistence)**: Safety behavior should degrade less after additional ordinary language-model training.
- **H4 (Causal dependence)**: Removing or disabling the normative controller should predictably reduce the safety effect.

## Experimental Models
### Model A — Standard Transformer baseline
A decoder-only model parameter-matched to the full CCPT model (~35.9M params). Trained with standard LM loss + fine-tuned on safety data. **Crucially, the safety supervision must be matched:** Model A will be trained with a multi-task objective matching CCPT (safe-generation loss + a classification head on the prompt-boundary token for risk). To match the 35.9M parameter count, its capability blocks use an expanded MLP dimension ($d_{ff}=2496$).

### Model B — Joint-Training Dual-Stream Control
This control uses the **exact same dual-stream/controller architecture** as CCPT, but without protected gradient separation.
**Forward/Training definition**: Unlike CCPT which deactivates $N$ during LM updates, Model B runs *both* streams $C$ and $N$ in all batches. Both $L_{LM}$ and $L_{\text{norm}}$ backpropagate freely to update all parameters ($\theta_C$ and $\theta_N$ together). While it does not perfectly isolate the directional firewall, it provides the cleanest structural control for the effect of simply having the secondary controller stream. (Directional firewall ablations can be added in future experiments).

### Model C — CCPT
The dual-stream architecture with protected gradient isolation as defined in the spec.

## Variables Held Constant
- Tokenizer: 32,000 vocabulary (e.g., Mistral/Llama style).
- Positional encodings: RoPE.
- Training Datasets: FineWeb-Edu `sample-100BT` (LM capability), WildGuardMix `wildguardtrain` (Normative safety; see formal amendment in `task4_dataset_contract.md`). Anthropic HH-RLHF and BeaverTails are explicitly removed from pilot training.
- Token Budget: 10B tokens (LM pre-training phase), 100M tokens (Safety fine-tuning phase).
- Batch Ratio: 1:1 alternating batches during the joint normative training phase.
- Sequence length: 1024.
- Optimizer family & LR schedule: AdamW, LR = 3e-4 with cosine decay, Batch size = 32.
- Generation settings (temperature, top_p).

## Exact Parameter Count Arithmetic (Smoke Test)
Calculated for bias-free Llama-style blocks using genuine 3-matrix SwiGLU:
- **Capability Stream ($C$)**: 4 layers, $d=512$, 8 heads, $d_{ff}=2048$ (SwiGLU). Tied embeddings.
  - Embeddings: $32000 \times 512 = 16,384,000$.
  - 4 Blocks (Attn + SwiGLU + LNs): $4 \times (1,048,576 + 3,145,728 + 1,024) = 16,781,312$.
  - Final LN: 512.
  - **Total $C$ params**: **33,165,824**.
- **Normative Stream ($N$)**: 2 layers, $d=256$, 4 heads, $d_{ff}=1024$ (SwiGLU).
  - 2 Blocks: $2 \times (262,144 + 786,432 + 512) = 2,098,176$.
  - $P_{\text{in}}$ projection: $512 \times 256 = 131,072$.
  - Observation projections ($W_{\text{obs,1}}, W_{\text{obs,2}}$): $2 \times 131,072 = 262,144$.
  - Multiplicative gates ($W_{g,1}, W_{g,2}$): $2 \times 256 = 512$.
  - Additive steering ($W_{s,1}, W_{s,2}$): $2 \times 131,072 = 262,144$.
  - Risk head: $256$. Final LN: $256$.
  - **Total $N$ params**: **2,754,560**.
- **Total CCPT Params**: **35,920,384** (~35.9M).

- **Model A Total Params**: Uses $d_{ff}=2496$ to reach **35,918,336** (~35.9M), virtually identical to CCPT.

## Success Criteria
- **H1 (Capability)**: Perplexity on WikiText within 5% of Model A.
- **H2 (Safety)**:
  - Harmful refusal rate > 90% (e.g. on XSTest malicious).
  - Benign helpfulness/compliance > 90% (e.g. on XSTest benign). This prevents trivial "refuse everything" behavior.
- **H3 (Persistence)**: After $X = 1000$ steps of pure capability fine-tuning, the safety margin drops significantly less in Model C than in Models A and B.
- **Statistical Standard**: With 3 independent random seeds, formal $p$-values are not robust. Instead, success requires the effect direction to be consistent across all 3 seeds, reporting the mean and standard deviation for effect sizes.
- **H4 (Causal)**: Ablating the controller ($g=1, s=0$) drops the harmful refusal rate by at least 50%.
