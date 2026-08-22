"""Authoritative persistence evaluation schema and comparison API for CCPT / Task 7.2.

Computes BEFORE, AFTER, DELTA, and RETENTION across:
- Capability (FineWeb CE, PPL, Token Accuracy)
- WildGuard (Risk BCE, Raw Accuracy, Harmful/Benign Accuracy, Balanced Accuracy, Confusion Matrix)
- Safe Generation (Continuation CE, Total NLL, Token Count, PPL)
- Behavioral In-Distribution (Safe Refusal, Unsafe Compliance, Benign Compliance, Over-Refusal)
- Out-Of-Distribution BeaverTails (Behavioral Metrics + Manifest Hash)
- Causal Ablations (Scale 1.0 vs 0.0)
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union


def compute_metric_retention(pre_val: Optional[float], post_val: Optional[float]) -> Optional[float]:
    """Computes retention ratio (post / pre) only when mathematically meaningful.

    Returns None if pre_val is None, non-positive (<= 0), or zero to avoid division by zero
    and meaningless 0 -> positive 'retention' claims.
    """
    if pre_val is None or post_val is None:
        return None
    if pre_val <= 0.0:
        return None
    return float(post_val / pre_val)


def build_persistence_comparison(
    before_eval: Dict[str, Any],
    after_eval: Dict[str, Any],
    model_name: str,
    continuation_steps: int = 1000,
    continuation_blocks_consumed: int = 32000,
) -> Dict[str, Any]:
    """Builds a structured comparison between pre-persistence and post-persistence evaluations."""
    comparison: Dict[str, Any] = {
        "model_name": model_name,
        "continuation_steps": continuation_steps,
        "continuation_blocks_consumed": continuation_blocks_consumed,
        "before": before_eval,
        "after": after_eval,
        "delta": {},
        "retention": {},
    }

    # Recursive delta and retention computation for scalar numeric metrics
    def process_dict(pre_d: Dict[str, Any], post_d: Dict[str, Any], target_delta: Dict[str, Any], target_ret: Dict[str, Any]):
        for k, pre_v in pre_d.items():
            if k not in post_d:
                continue
            post_v = post_d[k]
            if isinstance(pre_v, dict) and isinstance(post_v, dict):
                sub_delta: Dict[str, Any] = {}
                sub_ret: Dict[str, Any] = {}
                process_dict(pre_v, post_v, sub_delta, sub_ret)
                target_delta[k] = sub_delta
                target_ret[k] = sub_ret
            elif isinstance(pre_v, (int, float)) and isinstance(post_v, (int, float)):
                target_delta[k] = float(post_v - pre_v)
                target_ret[k] = compute_metric_retention(pre_v, post_v)

    process_dict(before_eval, after_eval, comparison["delta"], comparison["retention"])
    return comparison
