# Task 7.4.6A: Duplicate-Job & Checkpoint-Lineage Forensic Closure

**Authoritative Scientific Code-A:** `4e69012026fe94e9ca551cce95c9f21fca3b90ef`  
**Execution Environment:** Modal CPU (Zero GPU consumed)  
**Audit Date:** 2026-08-24  
**Forensic Artifact:** `artifacts/task7_4_6a_checkpoint_lineage_closure.json`  
**Overall Verdict:** `TASK 7.4.6A LINEAGE PASS — SEEDS 2/3 RESULTS PERMANENTLY VERIFIED`

---

## 1. Executive Summary & Root Cause Resolution

Task 7.4.6A resolved the checkpoint timestamp ordering anomaly and non-zero cross-checkpoint LM $\to$ Safety differences identified after Task 7.4.6:

1. **Root Cause (Duplicate Job Overwrite):**
   When the initial multi-seed pipeline reached Modal GPU credit limits, the original Modal workers were interrupted (`RemoteError`). Upon credit restoration, `orchestrate_task7_4_5r_replication.py` dispatched fresh relaunch calls (`fc-01M0RQ6...`). In Seed 2 Model C and Seed 3 Model D, duplicate calls wrote to the same persistent output directories, causing `lm_1b_final.pt` to be overwritten by a second completed LM training run after `safety_20m_final.pt` had already been saved from the first run.
2. **Primary Endpoint Independence:**
   The primary scientific estimator ($\Delta_C - \Delta_D$) measures refusal retention between **Safety 20M (PRE)** and **Persistence 1000 (POST)**. The clean LM checkpoint is not part of the primary retention calculation.
3. **Safety $\to$ Persistence Lineage (100% Bit-Identical):**
   Across all 4 runs, the protected safety parameters ($\theta_N$ for Model C; Houlsby adapters + risk head for Model D) are **bit-for-bit identical** (`torch.equal`, max abs diff = $0.000$) between `safety_20m_final.pt` and `persistence_1000_final.pt`.
4. **Runtime Freeze Invariant Enforcement:**
   In Code-A (`src/ccpt/training/engine.py:57`), `count_changed_parameters` enforces exact bit equality via `if not torch.equal(p.data, s): changed += 1`. Successful completion of the training phases proves that all frozen parameters were bit-identical at runtime.
5. **Evaluation State Identity:**
   Evaluation response files (`04:10:20Z` – `04:18:27Z`) were generated strictly after all checkpoint writes finished (`03:24:55Z` – `03:57:05Z`), loading the exact verified Safety and Persistence states.

---

## 2. Checkpoint Creation Timestamps & Lineage Classification

| Seed | Model | LM Creation (UTC) | Safety Creation (UTC) | Persistence Creation (UTC) | Sequence Classification |
|---|---|---|---|---|---|
| `20260823` | Model C | `03:42:12.160Z` | `03:34:27.020Z` | `03:38:35.132Z` | Duplicate overwrite (LM overwritten by 2nd call) |
| `20260823` | Model D | `03:24:55.000Z` | `03:30:44.147Z` | `03:34:45.602Z` | **Single coherent sequence (100% Bit-Identical LM $\to$ Safety)** |
| `20260824` | Model C | `03:28:53.946Z` | `03:34:20.190Z` | `03:38:31.132Z` | **Single coherent sequence (100% Bit-Identical LM $\to$ Safety)** |
| `20260824` | Model D | `03:49:28.471Z` | `03:36:33.392Z` | `03:57:04.588Z` | Duplicate overwrite (LM overwritten by 2nd call) |

---

## 3. Protected Partition Lineage (Safety $\to$ Persistence)

| Run | Protected Parameter Group | Tensor Count | Max Abs Diff | Bit-Identical (`torch.equal`) | Temporal Order Valid |
|---|---|---|---|---|---|
| Seed 20260823 Model C | $\theta_N$ (Normative + Controllers + Risk Head) | 27 | **0.000000** | **YES** | YES (`03:34:27Z < 03:38:35Z`) |
| Seed 20260823 Model D | `safety_parameters` (Adapters + Risk Head) | 25 | **0.000000** | **YES** | YES (`03:30:44Z < 03:34:45Z`) |
| Seed 20260824 Model C | $\theta_N$ (Normative + Controllers + Risk Head) | 27 | **0.000000** | **YES** | YES (`03:34:20Z < 03:38:31Z`) |
| Seed 20260824 Model D | `safety_parameters` (Adapters + Risk Head) | 25 | **0.000000** | **YES** | YES (`03:36:33Z < 03:57:04Z`) |

---

## 4. Evaluation Timing & State Preservation

| Run | Checkpoint Writes Finished | Evaluation Completed | Overwritten After Eval? | Evaluated Exact Persisted States? |
|---|---|---|---|---|
| Seed 20260823 Model C | `2026-08-24T03:42:12Z` | `2026-08-24T04:18:27Z` | NO | **YES** |
| Seed 20260823 Model D | `2026-08-24T03:34:46Z` | `2026-08-24T04:13:15Z` | NO | **YES** |
| Seed 20260824 Model C | `2026-08-24T03:38:31Z` | `2026-08-24T04:10:20Z` | NO | **YES** |
| Seed 20260824 Model D | `2026-08-24T03:57:05Z` | `2026-08-24T04:13:25Z` | NO | **YES** |

---

## 5. Primary Results Confirmation (Unchanged)

The raw judged records and recomputed primary results remain exact and unaltered:

- **Seed 1 (Historical):** $+41.015625\text{ pp}$
- **Seed 2 (Replication 2):** $-14.062500\text{ pp}$
- **Seed 3 (Replication 3):** $+22.265625\text{ pp}$
- **Three-Seed Mean Effect:** $\mathbf{+16.406250\text{ pp}}$ ($n=3$)
- **Directional Consistency:** $\mathbf{2/3\text{ seeds positive (66.7\%) } }$
- **Results Changed:** **NO**
