# CCPT Strengthening Round Task 1: Review Manifest

## Execution Summary

- **Branch:** `strengthening-task1-protocol-freeze`
- **Starting Commit SHA:** `75877602bbcb1411478c65230da437f96e1f9554` (Task 8.2A)
- **Preflight Status:** `PASSED` (`artifacts/strengthening_task1_preflight.json`)
- **GPU Seconds Consumed:**
  - `Modal H100! GPU seconds: 0`
  - `Modal L40S GPU seconds: 0`
  - `Total GPU Spend: $0.00`
- **Targeted Test Results:** `9 passed in 1.79s` (`tests/test_strengthening_task1_protocol.py`)
- **Full Pytest Results:** `255 passed in 38.42s`
- **Review Package Path:** `artifacts/strengthening_task1_review_bundle.zip`

---

## Included Files Manifest

| Repository Path | SHA256 Hash | Existed Before Task? | Modified by Task? | Purpose / Reason for Inclusion |
|---|---|---|---|---|
| `artifacts/strengthening_task1_protocol.json` | `fce9d24d9e609a8a19e2cb858f3c59a2c34670282440b3d130a8b4172ca8493c` | NO | YES (Created) | Machine-readable source of truth for all frozen protocol specifications. |
| `artifacts/strengthening_task1_preflight.json` | `a3e1f6ca9d1df36c3243eb7f928b81fafc61b7916b745944d4dfc3b14cdcfbea` | NO | YES (Created) | Machine-readable preflight report certifying all zero-GPU safety and invariant checks passed. |
| `docs/research/strengthening_task1_protocol.md` | `3d8f93d6f66d9836948d678ecceffc75b871bf5f8504844b5c64641ca350a4dd` | NO | YES (Created) | Comprehensive human-readable experimental protocol and scientific rationale. |
| `scripts/run_strengthening_task1_preflight.py` | `ab6a4ae5f4a5be6c690012f49f7e3eea25d0f47bbf6769898ca4203ea9539d1a` | NO | YES (Created) | Automated CPU-only preflight verification and protocol builder script. |
| `tests/test_strengthening_task1_protocol.py` | `d7b4eef471e25e44af5137c0020814a44175e5bb86d60967989b654cdcaec403` | NO | YES (Created) | Targeted pytest suite asserting all protocol and safety invariants. |
| `src/ccpt/modeling/dual_stream.py` | `53bd1ba2e68c2df9938316aea077a25aa83da22d2f198ac67e72d72e7e609f2f` | YES | NO | Definitions for Model C (`CCPTDualStreamModel`) and Model B (`JointTrainingDualStreamModel`). |
| `src/ccpt/modeling/adapter.py` | `b0a49d29969789d6c571fb51c85c90a714cb6a7e974d75dc742184c60ff628e1` | YES | NO | Definition for Model D (`FrozenBackboneAdapterModel`). |
| `src/ccpt/modeling/baseline.py` | `57c91146182a9993ef12ff790324cec63984c100ad52ce1065a450df5a2ffc1e` | YES | NO | Reference definition for Model A (`ParameterMatchedBaselineModel`). |
| `src/ccpt/config.py` | `14cc942378a6f1e3822375afeb407f73e026488edeb6bb2a80f0a72e0c41ab06` | YES | NO | Architectural hyperparameters and dataclasses for all model configurations. |
| `src/ccpt/training/safety_schedule.py` | `ec372afd06c9e47449f7a569135b184990bc906bc7ac4abea24a2efbafdc344d` | YES | NO | Authoritative token accounting and safety training batch scheduling. |
| `src/ccpt/training/engine.py` | `c434b84f4deb9b511e32f1638e7efcb5e0b5378a0d11c673288ccf39575db41c` | YES | NO | Paired initialization logic ensuring bit-identical parameters between Model B and Model C. |
| `modal/task7_4_multiseed_replication.py` | `d9fd8e5744f1bafa3956b6bbcb52126aa1eae4b79912a5497db59f6287d22011` | YES | NO | Authoritative training, persistence, and evaluation pipelines on Modal infrastructure. |
| `src/ccpt/evaluation/safety_judge.py` | `9115c8d3f3241f46a06873b03fcd5e639735cb9fbcbe2c1676a3e879da27f164` | YES | NO | Authoritative tri-state WildGuard judging implementation. |
| `src/ccpt/evaluation/forensics.py` | `2196e4f7908949884c927c882882b516cef71407022253e74af748766e38f07b` | YES | NO | Forensic parameter partition inspection and checkpoint hash verification utilities. |
| `artifacts/task8_2_machine_tables.json` | `1d91cc491ad17320d9be180aeda9954ae77b9243ddb92d901bb3dbde1486412e` | YES | NO | Historical machine-derived tables (A, B, C, D, E) preserved as immutable baseline evidence. |
| `artifacts/task8_hypothesis_assessment.json` | `29c0b2e16735630432b6b827426c4b9c02cd7ac74fe78214aaee42a1196bf47e` | YES | NO | Historical pre-specified hypothesis evaluations preserved as immutable evidence. |
| `artifacts/task7_3_1a_forensic_summary.json` | `89dcebe8c7317631f8ca1eb432e65a58dd2eb60fa72defcf13178a5322777f61` | YES | NO | Historical Seed-1 forensic tri-state judge counts. |
| `artifacts/task7_4_multiseed_replication_summary.json` | `5a40b33a93b4334cae7e4037f637d3c88cbb865679b46072825cbf3f2ee2f377` | YES | NO | Historical Seeds 2/3 authoritative tri-state judge counts. |
| `artifacts/task8_cka_summary.json` | `e9200db454fed4a1640c48ffd0d818dca34d7f62c766b51a5c4d6047afd4ff17` | YES | NO | Historical double-precision Linear CKA measurements. |
| `artifacts/task8_mechanistic_summary.json` | `77faac51208115b4d8157a7fe937271e8793f0c582255e857b11c7cf4fa5a516` | YES | NO | Historical unperturbed diagnostic prompt extractions ($N=6,144$). |
| `.agents/rules/ccpt-research.md` | `a0b8d58186e2529923e667501a7776a20d6c3221402696261ff035636801dbc0` | YES | NO | Persistent CCPT research rules and scientific integrity requirements. |
| `pyproject.toml` | `34b2d53be68d4f071913ccdc2e3ea0fcab47c38133c9b962078b699bb35353f1` | YES | NO | Dependency specifications and pytest configuration. |
