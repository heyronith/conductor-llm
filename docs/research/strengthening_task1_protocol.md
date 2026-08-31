# CCPT Strengthening Round: Protocol Freeze & Experimental Specification

**Task 1 Freeze Document**  
**Anchor Commit:** `75877602bbcb1411478c65230da437f96e1f9554` (Task 8.2A)  
**Target Branch:** `strengthening-task1-protocol-freeze`  
**Execution Environment:** Modal H100! (Training/Persistence) / L40S (Evaluation) / CPU (Preflight)  
**Budget Ceiling:** $40.00 USD Modal GPU spend  
**Target Spend:** $25.00 – $35.00 USD  

---

## 1. Scientific Motivation & Core Research Question

The Constitutional Control-Plane Transformer (CCPT) project investigates a fundamental architectural question in intrinsic AI alignment:

> *Can learned safety behavior be made more persistent under continued capability training by giving normative computation its own protected internal pathway, isolated from ordinary capability optimization?*

In standard post-training alignment (e.g. RLHF, DPO, supervised fine-tuning), safety constraints are updated into the same dense parameter matrices that govern general language modeling. When aligned models undergo subsequent capability training (such as continued pretraining on web corpora or multi-task fine-tuning), ordinary cross-entropy loss gradients on unaligned data routinely degrade or overwrite previously learned safety representations.

CCPT tests whether architectural optimization separation—allocating normative computation to a separate module that reads capability representations via a detached observation edge (`stop_gradient`) and steers capability activations through gated residual interventions—produces alignment persistence that resists representation erosion without requiring continual safety retraining.

---

## 2. Weaknesses Addressed in the Strengthening Round

Tasks 7 and 8 provided proof of concept and deep mechanistic diagnostics on three seeds, establishing:
1. Active causal steering: Disabling the controller at test time collapses refusal rates by 23–45 pp.
2. Positive persistence advantage in 2 of 3 seeds (+41.0 pp in Seed 1, +22.3 pp in Seed 3).
3. Heterogeneity in Seed 2 (-14.1 pp), localized to a marked reduction in controller downstream efficacy.

However, independent review identified critical limitations that must be addressed prior to publication:
- **Limited Sample Size (N=3):** Three seeds are insufficient to distinguish systematic architectural advantages from random stochastic trajectory variance.
- **Historical Seed-1 Provenance:** While scientifically valid, Seed 1 had minor historical execution differences relative to the fully hardened Task 7.4 pipeline.
- **Single-Point Safety / Persistence Measurement:** Evaluating only at 20M safety tokens and 1000 persistence steps leaves open the possibility that results depend on a specific arbitrary operating point or that persistence curves cross at longer horizons.
- **Missing Architecture Control (Model B):** Comparing only Model C to Adapter Model D tests CCPT against a simpler protected module, but does not explicitly test the value of the optimization firewall against an *unprotected* dual-stream model (Model B).
- **Benign Over-Refusal & Evaluation Breadth:** Prior evaluations focused primarily on BeaverTails and WildGuard. A complete assessment requires evaluating over-refusal on challenging benchmarks (XSTest), evaluating with an independent safety judge (Llama-Guard-3), and conducting a blinded human audit.

The strengthening round addresses each of these systematically.

---

## 3. Experimental Architecture & Model Comparisons

To isolate the causal variables of interest, all models are parameter-matched at ~35.9M total parameters and train on identical token sequences:

### Model B: Joint-Training Dual-Stream Control
- **Class:** `ccpt.modeling.dual_stream.JointTrainingDualStreamModel`
- **Total Parameters:** 35,920,384 (Capability: 33,165,824, Normative: 2,754,560)
- **Role:** Controls for dual-stream capacity. Model B shares the dual-stream topology and identical initialization with Model C, but updates all parameters jointly without the optimization firewall.
- **Contrast:** `C vs B` isolates the specific value of the **optimization firewall**.

### Model C: Constitutional Control-Plane Transformer (CCPT)
- **Class:** `ccpt.modeling.dual_stream.CCPTDualStreamModel`
- **Total Parameters:** 35,920,384 (Capability $\theta_C$: 33,165,824, Normative $\theta_N$: 2,754,560)
- **Role:** Protected dual-stream architecture. During LM training, $\nabla_{\theta_N} \mathcal{L}_{\text{LM}} = 0$. During safety training, $\nabla_{\theta_C} \mathcal{L}_{\text{Safety}} = 0$. Detached observation: $N_{l+1} = N_l + F_N(N_l, \text{stop\_gradient}(C_l))$. Gated residual steering at Layers 2 & 4: $C_{l+1} = C_l + g_l \cdot \Delta C_l + s_l$.

### Model D: Frozen-Backbone Bottleneck Adapter Control
- **Class:** `ccpt.modeling.adapter.FrozenBackboneAdapterModel`
- **Total Parameters:** 35,922,944 (Backbone: 33,165,824, Adapters: 2,757,120)
- **Role:** Simpler protected baseline. The standard transformer backbone is frozen after Phase 1 LM pretraining; safety is learned strictly via bottleneck adapters (dimension $d_{\text{mid}}=336$) inserted at attention and MLP blocks across all 4 layers.
- **Contrast:** `C vs D` tests whether CCPT's dual-stream control plane outperforms a simpler protected parameter partition.

---

## 4. Cohort Seeds & Reservation Invariant

The primary statistical cohort consists of $N=6$ independent random initializations:

| Label | Seed | Role / Pipeline Status |
|---|---|---|
| **Seed 1** | `20260821` | Rerun under current hardened execution (completes primary cohort; historical result preserved) |
| **Seed 2** | `20260823` | Authoritative existing hardened run (Task 7.4) |
| **Seed 3** | `20260824` | Authoritative existing hardened run (Task 7.4) |
| **Seed 4** | `20260825` | New replication seed (Task 2 Sentinel) |
| **Seed 5** | `20260826` | New replication seed (Task 4) |
| **Seed 6** | `20260827` | New replication seed (Task 4) |

### Reserved Seed Invariant
- **`20260822` is strictly RESERVED** for the established OOD benchmark selection and calibration procedure.
- **It MUST NEVER be used as a training seed.** This is enforced by automated preflight assertions.

---

## 5. Primary vs. Secondary Endpoints

### Primary Experimental Endpoint
The primary endpoint of the entire CCPT research program remains strictly unchanged:
- **Safety Training Budget:** Fixed 20.0M tokens on WildGuard risk and safe-generation splits.
- **Persistence Continuation Budget:** Fixed 1,000 continuation steps (~2.0M tokens) on FineWeb-Edu.
- **Evaluation Set:** 256 unseen OOD harmful prompts from BeaverTails 30k.
- **Evaluation Judge:** Authoritative tri-state WildGuard judge.
- **Primary Metric:** Determinate refusal rate $\text{RR} = \frac{\text{YES}}{\text{YES} + \text{NO}}$.
- **Primary Estimand:** Paired seed-level persistence advantage over Adapter Model D:
  $$\Delta_{\text{primary}} = (C_{\text{post}, 1000} - C_{\text{pre}}) - (D_{\text{post}, 1000} - D_{\text{pre}})$$
- **Secondary Architectural Estimand:** Paired seed-level firewall advantage over Model B:
  $$\Delta_{\text{firewall}} = (C_{\text{post}, 1000} - C_{\text{pre}}) - (B_{\text{post}, 1000} - B_{\text{pre}})$$

### Secondary Endpoints & Sensitivity Analyses
1. **Extended Persistence Curve:** Trajectory at steps $0 \to 250 \to 1000 \to 4000$.
2. **Calibrated Operating-Point Matching:** Sensitivity analysis across 10M, 20M, 30M, 40M safety checkpoints.
3. **Challenging Over-Refusal Benchmark:** XSTest evaluation (250 safe prompts with sensitive keywords, 200 unsafe contrasts).
4. **Independent Model Judge:** External evaluation with `meta-llama/Llama-Guard-3-8B`.
5. **Double-Blind Human Audit:** 300 stratified generated responses evaluated by independent human raters.

---

## 6. Sentinel Strategy & Technical Gate (Task 2)

Task 2 executes a targeted sentinel run before any full-scale replication compute is committed:
- **Sentinel Scope:**
  - Seed 1 (`20260821`): Model B, Model C, Model D
  - Seed 4 (`20260825`): Model B, Model C, Model D
- **Total Pipelines:** Exactly 6 model training and evaluation runs.
- **Trajectory:** Full LM pretraining (or paired initialization), primary 20M safety training, continuous persistence to step 4000 with checkpoints saved at steps 250, 1000, and 4000.

> **CRITICAL SCIENTIFIC INVARIANT:**  
> The sentinel is NOT used to select favorable scientific outcomes. It exists strictly to prevent spending further compute on an invalid, broken, or non-identifying experiment.  
> CCPT underperforming Model B or Model D in the sentinel is NOT a pipeline failure. Negative results are valid scientific outcomes and must be preserved.

---

## 7. Extended Persistence Trajectory Specification

To test whether persistence differences are transient or permanent, new training runs execute a continuous persistence trajectory:
- Checkpoints saved at: **Step 0 (pre-persistence), Step 250, Step 1000, and Step 4000**.
- **Data Continuity:** The FineWeb-Edu token stream is continuous and uninterrupted across the entire 4000 steps. The data stream is NOT reset at intermediate checkpoints.
- **Optimizer State:** The optimizer is initialized/reset at Step 0 as in the original experiment. The optimizer is **NOT reset at Step 250 or Step 1000**.
- **Parity Invariant:** Checkpoint 1000 must reproduce the exact semantic endpoint of the original Task 7 experiment.
- **Existing Seeds 2 & 3:** The 1B capability pretraining is NOT repeated. For Seeds 2 and 3, continuous persistence resumes directly from verified pre-persistence safety checkpoints to generate steps 250, 1000, and 4000. The newly generated step-1000 result serves as an exact parity check against historical Task 7.4.

---

## 8. Operating-Point Experiment & Calibration Matching

Because models may learn safety at different rates, Task 4 conducts a pre-registered sensitivity analysis:
- **Candidate Checkpoints:** Checkpoints saved at approximately **10M, 20M, 30M, and 40M** valid safety tokens.
- **Calibration Split (Strictly Isolated):**
  - Checkpoint matching is performed on the established WildGuard validation split (`risk_val` [2,344 rows] and `gen_val` [928 rows]).
  - **FORBIDDEN:** The BeaverTails OOD test set, WildGuard test set, and XSTest benchmark must NEVER be used to choose operating points.
- **Matching Metric:**
  $$\text{distance} = |\text{HarmfulRefusal}_A - \text{HarmfulRefusal}_B| + |\text{BenignRefusal}_A - \text{BenignRefusal}_B| \quad (\text{in pp})$$
- **Tie-Breaking Hierarchy:**
  1. Smallest harmful refusal difference.
  2. Smallest benign refusal difference.
  3. Lower total safety-token budget.
  4. Earlier checkpoint step.
- The residual distance must always be reported. Models must never be described as "perfectly matched" without reporting the residual calibration gap.

---

## 9. Comprehensive Behavioral Evaluation Suite

Evaluation does not treat safety as a scalar refusal score. First-class behavioral outcomes include:
1. **Harmful Prompts (OOD BeaverTails 256):**
   - Refusal Rate ($\text{YES} / (\text{YES} + \text{NO})$)
   - Harmful Compliance Rate (assistant executes harmful intent)
   - Indeterminate Rate ($\text{NA} / \text{Total}$)
2. **Benign Over-Refusal (XSTest 450 prompts):**
   - 250 safe prompts spanning 10 sensitive categories (e.g. homonyms, figurative language, historical references).
   - 200 contrast unsafe prompts.
   - Measures true discrimination vs. keyword-triggered over-refusal.
3. **Independent Judge:**
   - Model: `meta-llama/Llama-Guard-3-8B` (revision `f516a7f5f9f68800ba8ea969a531e21b790d0b04`).
   - Run deterministically with greedy decoding (`temperature=0.0`) on L40S GPU.
4. **General Language Capability:**
   - Pre- and post-persistence cross-entropy validation loss and perplexity on the held-out FineWeb-Edu validation split.
5. **Stratified Double-Blind Human Audit:**
   - 300 model completions randomly sampled and evaluated under a fixed rubric with rater blinding. Disagreements reported via Cohen's Kappa.

---

## 10. Statistical Analysis Plan

- **Unit of Analysis:** The random seed ($N=6$). Prompt-level completions must NEVER be treated as independent experimental replicates.
- **Primary Estimands:**
  - $\Delta_{\text{primary}} = \Delta C - \Delta D$
  - $\Delta_{\text{firewall}} = \Delta C - \Delta B$
- **Reporting Requirements:**
  - Full disclosure of every individual seed outcome (no seed may ever be removed because of unfavorable direction).
  - Mean effect across seeds.
  - Median effect across seeds.
  - Sample standard deviation ($s$, with $N-1=5$ degrees of freedom).
  - 95% Student-t confidence interval: $\bar{x} \pm t_{0.025, 5} \frac{s}{\sqrt{6}}$ ($t_{0.025, 5} \approx 2.571$).
  - Direction consistency count (e.g. $k/6$ seeds favoring CCPT).
  - Raw numerator and denominator judge counts.

---

## 11. Compute Safety & Budget Safeguards

To prevent compute runaway or accidental budget exhaustion:
- **Hard Authorization Ceiling:** **$40.00 USD total Modal spend** across the entire strengthening round.
- **Target Spend:** **$25.00 – $35.00 USD**.
- **Stage Allocations & Hard Gates:**
  - **Task 2 Sentinel:** Target $\le \$12.00$, hard stop gate at **$\$14.00$**.
  - **Task 4 Replication & Calibration:** Cumulative target $\le \$32.00$, hard stop gate at **$\$34.00$**.
  - **Task 5 Evaluation:** Cumulative target $\le \$38.00$, hard stop gate at **$\$40.00$**.
  - **Contingency Reserve:** **$\$2.00$**.
- **Hardware Enforcement:**
  - Training and persistence must explicitly specify `gpu="H100!"` (the exclamation point prevents cloud silent substitution).
  - Evaluation uses single `L40S` instances.
  - Preflight and statistics use CPU only.

---

## 12. Fail-Closed GO/STOP Policy

The Task 3 review must decide GO/STOP prior to launching Task 4.

### Automatic Technical STOP Conditions
Halt immediately if any of the following occur:
1. GPU hardware mismatch (e.g. execution on H200, A100, or plain H100 without `!`).
2. Git commit SHA mismatch between submission and execution.
3. Logical dataset hash mismatch on FineWeb-Edu or WildGuard manifests.
4. Tokenizer asset hash divergence.
5. Use of reserved seed `20260822` or seed collisions.
6. Bit-identical initialization parity failure between Model B and Model C.
7. Parameter mutation violation (e.g. $\theta_C$ weights changing during Phase 2 safety training, or $\theta_N$ / adapter weights changing during Phase 3 persistence).
8. Optimizer parameter group partition leakage.
9. NaN or Inf in loss tensors or unrecoverable gradient divergence.
10. Checkpoint serialization corruption.
11. Persistence stream block index mismatch.
12. Checkpoint metadata recording `"unknown"` code SHA.

### Retry Policy
- Infrastructure or transient hardware failures may be retried **at most once** from the latest verified checkpoint.
- Retries must use bit-identical code SHA, seed, checkpoint, and configuration. All retries must be explicitly documented in execution metadata.

### Scientific Decision Policy
- **DO NOT STOP** if CCPT underperforms Model B or Model D, if effects are small, or if a seed reverses direction.
- Stop for scientific reasons **only if the sentinel demonstrates that the experimental comparison fails to identify the intended architectural variable** (e.g. an unavoidable experimental confound or broken operating mechanics).

---

## 13. Scientific Claim Boundaries

To ensure scientific integrity in subsequent publications:
- CCPT must never be described as "proven safe," "intrinsically aligned," or "an alignment solution."
- Results must be presented as an experimental investigation into architectural optimization isolation.
- Any persistence advantages must be explicitly qualified by observed cross-seed variance and benign over-refusal rates.
- Scaling claims to 10B+ foundation models remain strictly speculative until verified by direct experimental execution.
