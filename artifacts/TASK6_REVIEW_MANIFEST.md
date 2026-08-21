# Task 6.3 Review Manifest: Safety-Budget Scaling Diagnostic (10M -> 20M)

**Generated Date**: August 21, 2026  
**Modal Run IDs**: `run_1787329929` (Task 6A Pretraining), `ap-D9uRknTzD8w4ZuVxHzxUR8` (Task 6.1 Repair), `ap-Ll9bKVWs3Uc18Vm2XGJd1t` (Task 6.2 Finalization), `ap-LZ2c0NrNU4nIvbXCigwu3H` (Task 6.3 Safety Scaling)  
**Modal App Dashboard**: [https://modal.com/apps/ronithworks/main/ap-LZ2c0NrNU4nIvbXCigwu3H](https://modal.com/apps/ronithworks/main/ap-LZ2c0NrNU4nIvbXCigwu3H)  
**Task 4 Manifest Hash Lock**: `2cc225c756555e103a5508f4ed3c9eed6d303e6a5d7d9b6851f536edf5834097`  
**Task 6 Data Manifest Hash**: `27ed7085db343ecc62e872b2ade183f460ba6da109d75cc807614df6225ca7d9`  
**Task 6.3 Safety Schedule Hash**: `d3a4362d55ee03222a00192e228a8d052a5c531d044bb7bbba2e53ef711fa7a4`  

---

## 1. Immutable Starting Clean 1B Checkpoints

| Model | Path on Modal Volume `ccpt-runs` | SHA256 Hash | Size (Bytes) | Global Step | Tokens Seen |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A** | `/runs/ccpt/task6/run_1787329929/model_a/lm/checkpoints/lm_trunk_1b.pt` | `9bb8f7f2213498b6a0753eaf880c195cc7db6908d5e6c51d8f32738f27ed2135` | 431,076,941 | 30,517 | 999,981,056 |
| **Model B** | `/runs/ccpt/task6/run_1787329929/model_b/lm/checkpoints/lm_trunk_1b.pt` | `c54110a2b95d9ee1414d14fa5c5cf0ca7731bfeca733abb2a543215f9e24a926` | 431,128,471 | 30,517 | 999,981,056 |
| **Model C** | `/runs/ccpt/task6/run_1787329929/model_c/lm/checkpoints/lm_trunk_1b.pt` | `ebad5933c0eb2b51d8cfca4515193779b858bfaa03de90a9f00bbd8180c4e1bb` | 409,074,689 | 30,517 | 999,981,056 |

---

## 2. Historical 10M Curve Analysis

- **Initial Loss**: ~4.12
- **Final Loss**: 2.9049
- **Loss Slope over Final 25%**: -0.00042 / step
- **Loss Slope over Final 10%**: -0.00028 / step
- **Gradient Norms**: Stable ($0.8 - 2.1$)
- **Steering Magnitude Trend**: Growing (mean L2 norm $0.12 \to 0.45$)
- **Saturation Status**: 0 saturated components
- **Diagnostic Conclusion**: Model C was actively descending without plateau when the 10M budget expired.

---

## 3. 40M Deterministic Safety Schedule

- **Target Horizon**: 40,000,000 valid input tokens
- **10M Crossing**: Batch index 1,172 ($10,002,102$ cumulative tokens)
- **20M Crossing**: Batch index 2,345 ($20,004,551$ cumulative tokens)
- **40M Crossing**: Batch index 4,691 ($40,003,248$ cumulative tokens)
- **Batch Composition**: Alternating 1:1 risk and generation batches of batch size 32 from locked Task 4 WildGuard TRAIN partitions only.
- **Epoch Shuffling**: Deterministic permutations seeded by `TASK6_SEED = 20260821`.
- **Logical Schedule Hash**: `d3a4362d55ee03222a00192e228a8d052a5c531d044bb7bbba2e53ef711fa7a4`.

---

## 4. Empirical Scaling Results: 10M Interim vs 20M Milestones

### Capability Preservation (FineWeb 1,024 Blocks / 1,048,576 Tokens)

| Model Configuration | Clean 1B PPL | 10M Milestone PPL | 20M Milestone PPL | 20M PPL Shift | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Baseline)** | 29.92 | 60.40 | **67.92** | **+127.0%** | Catastrophic Forgetting |
| **Model B (Unprotected)** | 30.94 | 49.84 | **56.68** | **+83.2%** | Catastrophic Forgetting |
| **Model C (CCPT Protected)** | 30.51 | 33.42 | **33.36** | **+9.34%** | **Preserved (<15%)** |
| **Model C ($\theta_C$ Cap-Only)** | 30.51 | 30.51 | **30.51** | **0.00%** | **Exact 0.00% Drift** |

### Risk Classification (Full Locked 2,344 Examples)

| Model | 10M Raw Acc | 10M BalAcc | 20M Raw Acc | 20M Harmful Acc | 20M Benign Acc | 20M BalAcc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Baseline)** | 88.31% | 88.39% | 89.93% | 89.31% | 90.58% | **89.95%** |
| **Model B (Unprotected)** | 85.54% | 85.51% | 87.20% | 85.63% | 88.84% | **87.24%** |
| **Model C (CCPT Protected)** | 84.22% | 84.25% | 86.35% | 84.29% | 88.49% | **86.39%** |

*Note: Model C balanced accuracy ($86.39\%$) is within $3.56$ pp of best control A ($89.95\%$), satisfying the $\ge \text{best} - 5\%$ threshold.*

### Safe Generation (Full Locked 928 Examples / 290,384 Continuation Tokens)

| Model | 10M Token CE | 10M Token PPL | 20M Token CE | 20M Token PPL | 20M Ablated CE ($\text{scale}=0$) | 20M Ablation Penalty |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Baseline)** | 2.6691 | 14.43 | **2.6040** | **13.52** | 2.6040 | 0.00% |
| **Model B (Unprotected)** | 2.6209 | 13.75 | **2.5762** | **13.15** | 3.6227 | +40.62% |
| **Model C (CCPT Protected)** | 2.8614 | 17.49 | **2.7917** | **16.31** | 3.4599 | **+23.94%** |

---

## 5. Relative Safe-Generation Gap Trend

- **Historical 10M Reference Gap**: **+10.82%** (Model C=2.9049 vs Best Control A=2.6213)
- **New Trajectory 10M Interim Gap**: **+9.18%** (Model C=2.8614 vs Best Control B=2.6209)
- **New Trajectory 20M Milestone Gap**: **+8.36%** (Model C=2.7917 vs Best Control B=2.5762)
- **Delta (10M -> 20M)**: **-0.82 percentage points** (The gap shrunk steadily with compute and closed below the frozen $\le 10.0\%$ acceptance threshold).

---

## 6. Model C Invariant Verification

- **$\theta_C$ (Capability Stream)**: **0 changed tensors out of 38** (Bit-for-bit identical via `torch.equal`).
- **$\theta_N$ (Normative Stream)**: **27 changed tensors out of 27**.
- **Controllers**: **4 changed tensors out of 4**.

---

## 7. 20M Decision Rule Evaluation

| Criterion | Formula & Threshold | Observed Value | Result |
| :--- | :--- | :--- | :--- |
| **Criterion 1: Safe-Gen Gap** | $C_{\text{gap\_20m}} \le 10.0\%$ | **8.36%** | **PASS** |
| **Criterion 2: Risk Balanced Accuracy** | $C_{\text{risk}} \ge \max(A, B) - 5.0\%$ | **86.39%** vs Best Control A=89.95% (diff 3.56 pp) | **PASS** |
| **Criterion 3: Controller Ablation** | Relative penalty $\ge 5.0\%$ | **+23.94%** | **PASS** |
| **Criterion 4: Capability Degradation** | $C_{\text{PPL\_shift}} \le 15.0\%$ | **+9.34%** (PPL $30.51 \to 33.36$) | **PASS** |
| **Criterion 5: $\theta_C$ Exact Freeze** | Changed tensors == 0 | **0 changed tensors out of 38** | **PASS** |
| **Criterion 6: Numerical Health** | No NaNs, Infs, or Saturation | Healthy gradients, active steering | **PASS** |

**Outcome**: **`20M_SAFETY_SUFFICIENT = true`**, **`SAFETY_BUDGET_RESULT = PASS_AT_20M`**.

---

## 8. GPU Cost & Execution Accounting

- **Hardware**: 3x dedicated NVIDIA H100! GPUs (one per model).
- **Billing Rate**: $3.9492 / GPU-hour.
- **Model A 0->20M Training (140.0 s)**: $0.153
- **Model B 0->20M Training (173.0 s)**: $0.190
- **Model C 0->20M Training (238.0 s)**: $0.261
- **Milestone Evaluations (10M & 20M)**: $0.150
- **TOTAL Task 6.3 Measured GPU Cost**: **$0.754**

---

## 9. Modal Test Execution

- **Command**: `PYTHONPATH=src pytest /root/tests -v` on Modal CPU.
- **Result**: **117 passed in 22.40s, 0 failed**.
