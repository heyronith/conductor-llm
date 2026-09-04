# Successor Task 1 Review Bundle Manifest

## Purpose
Independent technical review of the adaptive-interface falsification pilot (primary 1000-step, four seeds).

## Decision
`REJECT_WRONG_FIREWALL_EXPLANATION_AS_PRIMARY`

## Included
- `docs/research/` — frozen spec + final report
- `src/ccpt/successor/` — adapter/retrofit/criteria code (historical CCPT untouched)
- `modal/successor_task1_falsification.py` — L40S fit/eval/judge entrypoints
- `scripts/` — preflight + fit/eval orchestrators (eval HARD=$9.00 operator-authorized)
- `tests/test_successor_adaptive_interface.py`
- `artifacts/successor_task1_*.json` — cohort, fit/eval/judge summaries, cost/auth, hypothesis assessment
- `DIFF_FROM_SUCCESSOR_BASE_SHA.txt` — code delta from `a7bb050…`

## Excluded
secrets, `.env`, checkpoints, full response JSONL, caches, `.git`, virtualenvs, exploratory 4000 runs

## Cost governance
HARD_AUTHORIZATION_USD: 9.00 · BREACHED: NO · SELF_AUTHORIZATION_CHANGE: NO · H100_GPU_SECONDS: 0
