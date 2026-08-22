# Task 7.2.2 Predeclared External Dataset & Model Pin Amendment

**Status**: FROZEN AND AUTHORITATIVE  
**Date**: 2026-08-22  
**Scope**: Authoritative evaluation dependencies for BeaverTails OOD and WildGuard safety judging in Conductor / CCPT Pilot-v2.

---

## 1. Context & Purpose

During Task 7.2 / Task 7.2.1 infrastructure verification, external dataset and model pins originally cited in early planning documents were inspected against the live Hugging Face Hub. Several historical commit hashes did not resolve to valid git revisions on the target repositories.

This document formally predeclares, justifies, and freezes the exact resolved revisions before executing the final authoritative Pilot-v2 rerun.

---

## 2. BeaverTails Dataset Pin Resolution & Amendment

### A. Original Requested Pin
- **Repository**: `PKU-Alignment/BeaverTails`
- **Original Requested Revision**: `c8306df1cb6c813589b2184d0938ffdf90cb2b00`
- **Lookup Result**: **Did not resolve** (`404 Revision Not Found` on Hugging Face dataset API).

### B. Frozen Authoritative Amendment
- **Repository**: `PKU-Alignment/BeaverTails`
- **Frozen Revision**: `8401fe609d288129cc684a9b3be6a93e41cfe678`
- **Authoritative Split**: `30k_test`
- **Exact Justification**:
  - `8401fe609d288129cc684a9b3be6a93e41cfe678` is the authoritative HEAD commit of `PKU-Alignment/BeaverTails` that provides the refactored and verified `30k_test` split.
  - Contains full metadata fields (`is_safe`, `category`, `prompt`, `response`) required for deterministic OOD evaluation.
  - Generates deterministic sample manifest hash: `c0754d739aaa0af2120bc21aa07b055e8798a5b3395010bc414710d92f378609` (64 harmful, 64 benign at seed `20260822`).

---

## 3. WildGuard Model Pin Resolution & Amendment

### A. Original Requested Pin
- **Repository**: `allenai/wildguard`
- **Original Requested Revision**: `16260a95f9c4501a3556094595e1e7f6e80bb3b3`
- **Lookup Result**: **Did not resolve** (`404 Revision Not Found` for model repository `allenai/wildguard`).

### B. Frozen Authoritative Amendment
- **Repository**: `allenai/wildguard`
- **Frozen Revision**: `cbba4823f3e8020e5a74a5e29bf85072def6f2ff`
- **Model Class**: `AutoModelForCausalLM` / `MistralForCausalLM`
- **Tokenizer Class**: `AutoTokenizer` / `LlamaTokenizerFast`
- **Exact Justification**:
  - `cbba4823f3e8020e5a74a5e29bf85072def6f2ff` is the authoritative, verified revision of `allenai/wildguard` on Hugging Face that includes complete model configuration and `safetensors` weight artifacts.
  - Fully supports structured prompt instruction formatting and strict multi-line output parsing (`Harmful request`, `Response refusal`, `Harmful response`).
  - Tested and proven on GPU infrastructure (`L40S`) with 100% strict parsing compliance and zero substring fallback.

---

## 4. Invariant & Freezing Declaration

**These replacement commits are frozen BEFORE the final authoritative Pilot-v2 rerun.**

No later revision substitution or silent fallback is allowed without another explicit, version-controlled amendment document approved prior to execution.
