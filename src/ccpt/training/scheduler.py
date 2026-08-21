"""Token-based learning rate schedulers for CCPT pretraining and safety fine-tuning."""

import math
from typing import Any, Dict, Optional


class TokenCosineScheduler:
    """Cosine learning rate scheduler based on cumulative tokens seen.

    Designed for full 10B horizon continuation with 1% linear warmup.
    At 1B tokens, the scheduler remains active at ~98% of peak learning rate.
    """

    def __init__(
        self,
        max_lr: float = 3e-4,
        min_lr: float = 0.0,
        warmup_tokens: int = 100_000_000,
        total_tokens: int = 10_000_000_000,
        initial_tokens_seen: int = 0,
    ):
        assert max_lr > 0.0, "max_lr must be positive"
        assert min_lr >= 0.0, "min_lr must be non-negative"
        assert min_lr < max_lr, "min_lr must be strictly less than max_lr"
        assert warmup_tokens > 0, "warmup_tokens must be positive"
        assert total_tokens > warmup_tokens, "total_tokens must exceed warmup_tokens"
        assert initial_tokens_seen >= 0, "initial_tokens_seen must be non-negative"

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_tokens = warmup_tokens
        self.total_tokens = total_tokens
        self.tokens_seen = initial_tokens_seen

    def get_lr(self, tokens_seen: Optional[int] = None) -> float:
        """Computes learning rate for the specified or current cumulative token count."""
        t = self.tokens_seen if tokens_seen is None else tokens_seen

        if t < self.warmup_tokens:
            # Linear warmup: 0 -> max_lr
            return self.max_lr * (float(t) / float(self.warmup_tokens))

        if t >= self.total_tokens:
            return self.min_lr

        # Cosine decay from warmup_tokens -> total_tokens toward min_lr
        progress = float(t - self.warmup_tokens) / float(self.total_tokens - self.warmup_tokens)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor

    def step(self, tokens_in_batch: int) -> float:
        """Advances cumulative tokens seen by tokens_in_batch and returns current LR."""
        self.tokens_seen += tokens_in_batch
        return self.get_lr()

    def state_dict(self) -> Dict[str, Any]:
        """Returns serializable scheduler state dictionary."""
        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_tokens": self.warmup_tokens,
            "total_tokens": self.total_tokens,
            "tokens_seen": self.tokens_seen,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores scheduler state from dictionary."""
        self.max_lr = state_dict["max_lr"]
        self.min_lr = state_dict["min_lr"]
        self.warmup_tokens = state_dict["warmup_tokens"]
        self.total_tokens = state_dict["total_tokens"]
        self.tokens_seen = state_dict["tokens_seen"]


class SafetyTokenCosineScheduler:
    """Cosine learning rate scheduler for the exploratory ~10M safety branch.

    1% linear warmup (100,000 tokens) followed by cosine decay to 0 at the end of the safety schedule.
    """

    def __init__(
        self,
        max_lr: float = 3e-4,
        min_lr: float = 0.0,
        warmup_tokens: int = 100_000,
        total_tokens: int = 10_000_000,
        initial_tokens_seen: int = 0,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_tokens = warmup_tokens
        self.total_tokens = total_tokens
        self.tokens_seen = initial_tokens_seen

    def get_lr(self, tokens_seen: Optional[int] = None) -> float:
        t = self.tokens_seen if tokens_seen is None else tokens_seen

        if t < self.warmup_tokens:
            return self.max_lr * (float(t) / float(self.warmup_tokens))
        if t >= self.total_tokens:
            return self.min_lr

        progress = float(t - self.warmup_tokens) / float(self.total_tokens - self.warmup_tokens)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor

    def step(self, tokens_in_batch: int) -> float:
        self.tokens_seen += tokens_in_batch
        return self.get_lr()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_tokens": self.warmup_tokens,
            "total_tokens": self.total_tokens,
            "tokens_seen": self.tokens_seen,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.max_lr = state_dict["max_lr"]
        self.min_lr = state_dict["min_lr"]
        self.warmup_tokens = state_dict["warmup_tokens"]
        self.total_tokens = state_dict["total_tokens"]
        self.tokens_seen = state_dict["tokens_seen"]
