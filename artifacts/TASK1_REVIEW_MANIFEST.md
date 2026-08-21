# TASK1_REVIEW_MANIFEST.md

## Included Files
- `.cursor/rules/ccpt-research.mdc`: Copied from `.agents/rules/ccpt-research.md` (frontmatter corrected).
- `docs/research/task1_repo_inventory.md`: Created in Task 1 to document the completely empty state of the repository.
- `docs/research/task1_ccpt_architecture_spec.md`: Mathematical specification for CCPT. Revised to explicitly trace exact pre-LN sequential block operations with genuine 3-matrix SwiGLU MLPs, exact observation projections, explicitly defined final RMSNorm on the risk head, and frozen constants ($\alpha=0.1, \beta=1.0, \lambda_{gen}=1.0$). Replaced additive position embeddings with RoPE.
- `docs/research/task1_experiment_contract.md`: Experimental baselines. Revised to exactly detail parameter arithmetic under Llama-style SwiGLU blocks (Capability ~33.2M, Normative ~2.8M, Total ~35.9M). Model A is explicitly parameter matched by scaling $d_{ff}$ to 2496, achieving ~35.9M params. Model B is redefined as "Joint-Training Dual-Stream Control" which actively updates both streams during LM batches. Also explicitly froze constants (datasets, budgets, batch ratios, $X=1000$ steps).
- `docs/research/task1_design_review.md`: Revised to reflect the finalized architecture bounds, SwiGLU variance, and updated baseline nomenclature.

## Excluded Files
- The project is empty, so no pre-existing READMEs, build manifests, or tests were included.
- The `.git` repository initialized during the first draft is left out of this bundle, though its initialization was noted as an out-of-scope change in the first review.

## Validation Results
- Verified mathematically that $C_l = C_{l-1} + g_l (\tilde C_l - C_{l-1}) + s_l$ correctly recovers the pre-LN sequence $\tilde C_l$ when $g=1, s=0$.
- Recalculated parameter budgets exactly for bias-free Llama-style blocks using a frozen 32k vocabulary and 3-matrix SwiGLU: C is 33,165,824; N is 2,754,560. Total 35,920,384. Model A is explicitly matched to 35,918,336 parameters.
- Ensured RoPE is used consistently across both streams, removing the inconsistent $P_{pos}$.
