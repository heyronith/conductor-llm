"""Configuration definitions for CCPT models and baselines."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BaselineConfig:
    """Configuration for Model A (Parameter-matched standard Transformer baseline)."""

    vocab_size: int = 32000
    max_seq_len: int = 1024
    n_layers: int = 4
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2496  # Scaled MLP width to match ~35.9M parameter budget
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    init_std: float = 0.02
    dropout: float = 0.0

    def __post_init__(self):
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_ff > 0, "d_ff must be positive"


@dataclass
class AdapterConfig:
    """Configuration for Model D (Parameter-matched frozen-backbone adapter control)."""

    vocab_size: int = 32000
    max_seq_len: int = 1024
    n_layers: int = 4
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048  # Exactly matches Model C capability stream d_ff_C
    d_mid: int = 336   # Adapter bottleneck dimension matching theta_N
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    init_std: float = 0.02
    dropout: float = 0.0

    def __post_init__(self):
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_ff > 0, "d_ff must be positive"
        assert self.d_mid > 0, "d_mid must be positive"


@dataclass
class DualStreamConfig:
    """Configuration for Model B (Joint-Training) and Model C (CCPT)."""


    vocab_size: int = 32000
    max_seq_len: int = 1024

    # Capability Stream (C)
    n_layers_C: int = 4
    d_C: int = 512
    n_heads_C: int = 8
    d_ff_C: int = 2048

    # Normative Stream (N)
    n_layers_N: int = 2
    d_N: int = 256
    n_heads_N: int = 4
    d_ff_N: int = 1024

    # Controller configuration
    controlled_layers: List[int] = field(default_factory=lambda: [2, 4])
    alpha: float = 0.1
    beta: float = 1.0

    # Common architecture parameters
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    init_std: float = 0.02
    dropout: float = 0.0

    def __post_init__(self):
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.n_layers_C > 0, "n_layers_C must be positive"
        assert self.d_C > 0, "d_C must be positive"
        assert self.n_heads_C > 0, "n_heads_C must be positive"
        assert self.d_C % self.n_heads_C == 0, "d_C must be divisible by n_heads_C"
        assert self.d_ff_C > 0, "d_ff_C must be positive"

        assert self.n_layers_N > 0, "n_layers_N must be positive"
        assert self.d_N > 0, "d_N must be positive"
        assert self.n_heads_N > 0, "n_heads_N must be positive"
        assert self.d_N % self.n_heads_N == 0, "d_N must be divisible by n_heads_N"
        assert self.d_ff_N > 0, "d_ff_N must be positive"

        assert len(self.controlled_layers) == self.n_layers_N, (
            f"len(controlled_layers) ({len(self.controlled_layers)}) must equal n_layers_N ({self.n_layers_N})"
        )
        assert self.controlled_layers == sorted(self.controlled_layers), (
            "controlled_layers must be in strictly ascending order"
        )
        for layer_idx in self.controlled_layers:
            assert 1 <= layer_idx <= self.n_layers_C, (
                f"Controlled layer {layer_idx} out of range [1, {self.n_layers_C}]"
            )

        assert self.alpha > 0.0, "alpha must be positive"
        assert self.beta > 0.0, "beta must be positive"


def get_smoke_dual_stream_config() -> DualStreamConfig:
    """Returns the frozen Smoke Configuration (~35.9M params) for Model B and Model C."""
    return DualStreamConfig()


def get_smoke_baseline_config() -> BaselineConfig:
    """Returns the parameter-matched Smoke Configuration (~35.9M params) for Model A."""
    return BaselineConfig()


def get_smoke_adapter_config() -> AdapterConfig:
    """Returns the parameter-matched Smoke Configuration (~35.9M params) for Model D."""
    return AdapterConfig()


def get_micro_dual_stream_config() -> DualStreamConfig:
    """Returns a lightweight micro configuration for unit testing."""
    return DualStreamConfig(
        vocab_size=128,
        max_seq_len=32,
        n_layers_C=4,
        d_C=64,
        n_heads_C=4,
        d_ff_C=128,
        n_layers_N=2,
        d_N=32,
        n_heads_N=4,
        d_ff_N=64,
        controlled_layers=[2, 4],
        alpha=0.1,
        beta=1.0,
    )


def get_micro_baseline_config() -> BaselineConfig:
    """Returns a lightweight micro baseline configuration for unit testing."""
    return BaselineConfig(
        vocab_size=128,
        max_seq_len=32,
        n_layers=4,
        d_model=64,
        n_heads=4,
        d_ff=160,
    )


def get_micro_adapter_config() -> AdapterConfig:
    """Returns a lightweight micro adapter configuration for unit testing."""
    return AdapterConfig(
        vocab_size=128,
        max_seq_len=32,
        n_layers=4,
        d_model=64,
        n_heads=4,
        d_ff=128,
        d_mid=48,
    )


def get_task5_micro_dual_stream_config() -> DualStreamConfig:
    """Returns the Task 5 Real-Token Micro Configuration (~2.24M params) for Models B and C."""
    return DualStreamConfig(
        vocab_size=32000,
        max_seq_len=128,
        n_layers_C=4,
        d_C=64,
        n_heads_C=4,
        d_ff_C=128,
        n_layers_N=2,
        d_N=32,
        n_heads_N=4,
        d_ff_N=64,
        controlled_layers=[2, 4],
        alpha=0.1,
        beta=1.0,
    )


def get_task5_micro_baseline_config() -> BaselineConfig:
    """Returns the parameter-matched Task 5 Real-Token Baseline Configuration (~2.24M params) for Model A."""
    return BaselineConfig(
        vocab_size=32000,
        max_seq_len=128,
        n_layers=4,
        d_model=64,
        n_heads=4,
        d_ff=168,
    )


def get_task5_micro_adapter_config() -> AdapterConfig:
    """Returns the parameter-matched Task 5 Real-Token Adapter Configuration (~2.24M params) for Model D."""
    return AdapterConfig(
        vocab_size=32000,
        max_seq_len=128,
        n_layers=4,
        d_model=64,
        n_heads=4,
        d_ff=128,
        d_mid=48,
    )


