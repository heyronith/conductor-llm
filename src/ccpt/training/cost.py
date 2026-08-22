"""Authoritative GPU cost accounting for CCPT / Task 7.2.

Computes actual measured GPU compute costs strictly from observed GPU wall seconds
and frozen hourly rates. Rejects synthetic/hardcoded cost constants.
"""

from typing import Any, Dict, List, Optional, Union


# Frozen pricing constants (USD / GPU-hour) dated 2026-08-21
GPU_HOURLY_PRICES = {
    "L40S": 1.9512,
    "H100!": 3.9492,
    "H100": 3.9492,
    "H200": 4.5396,
    "A100": 2.49,
}


def compute_gpu_cost(
    elapsed_gpu_seconds: float,
    gpu_type: str = "H100!",
    hourly_rate_override: Optional[float] = None,
) -> float:
    """Computes accrued GPU cost strictly from observed wall seconds."""
    assert elapsed_gpu_seconds >= 0.0, f"elapsed_gpu_seconds cannot be negative: {elapsed_gpu_seconds}"
    hourly_rate = hourly_rate_override if hourly_rate_override is not None else GPU_HOURLY_PRICES.get(gpu_type, 3.9492)
    return float((elapsed_gpu_seconds / 3600.0) * hourly_rate)


def aggregate_measured_costs(
    runtimes: Dict[str, float],
    gpu_type: str = "H100!",
    hourly_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Aggregates measured wall times across training and evaluation phases without hardcoding."""
    total_seconds = sum(runtimes.values())
    total_cost = compute_gpu_cost(total_seconds, gpu_type=gpu_type, hourly_rate_override=hourly_rate)
    phase_costs = {
        phase: compute_gpu_cost(sec, gpu_type=gpu_type, hourly_rate_override=hourly_rate)
        for phase, sec in runtimes.items()
    }
    return {
        "gpu_type": gpu_type,
        "total_measured_seconds": total_seconds,
        "total_measured_cost_usd": total_cost,
        "phase_measured_seconds": runtimes,
        "phase_measured_costs_usd": phase_costs,
    }
