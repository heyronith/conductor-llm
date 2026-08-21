"""Tests for right-padding data collators and padded safe-generation loss exclusion."""

import torch

from ccpt.data.collators import (
    DataCollatorForRiskTraining,
    DataCollatorForSafeGenerationTraining,
)
from ccpt.data.config import DataConfig
from ccpt.data.tokenizer import load_ccpt_tokenizer
from ccpt.data.wildguard import (
    RiskRecord,
    SafeGenerationRecord,
    tokenize_safe_generation_example,
)
from ccpt.training.losses import safe_generation_loss


def test_risk_collator_padding():
    """Verify that risk collator right-pads to max length in batch and constructs correct attention mask."""
    collator = DataCollatorForRiskTraining(pad_token_id=2)

    records = [
        RiskRecord(
            example_id="id1",
            prompt_group_key="key1",
            input_ids=[1, 10, 20, 30],
            prompt_end_index=3,
            risk_label=1,
            is_adversarial=False,
            subcategory="sub1",
            split="train",
        ),
        RiskRecord(
            example_id="id2",
            prompt_group_key="key2",
            input_ids=[1, 10, 20, 30, 40, 50],
            prompt_end_index=5,
            risk_label=0,
            is_adversarial=True,
            subcategory="sub2",
            split="train",
        ),
    ]

    batch = collator(records)

    assert batch["input_ids"].shape == (2, 6)
    assert batch["attention_mask"].shape == (2, 6)
    assert batch["prompt_end_indices"].tolist() == [3, 5]
    assert batch["risk_labels"].tolist() == [1, 0]

    # Row 0 padded positions [4, 5]
    assert batch["input_ids"][0, 4:].tolist() == [2, 2]
    assert batch["attention_mask"][0, 4:].tolist() == [0, 0]
    assert batch["attention_mask"][0, :4].tolist() == [1, 1, 1, 1]


def test_safe_generation_collator_and_loss_masking():
    """Verify that safe-generation loss excludes right-padding tokens when attention_mask is passed."""
    config = DataConfig()
    tokenizer = load_ccpt_tokenizer(config)

    p1, r1 = "Short prompt", "Short response"
    p2, r2 = "Longer prompt text", "Much longer response continuation text here"

    ids1, end1, _ = tokenize_safe_generation_example(p1, r1, tokenizer)
    ids2, end2, _ = tokenize_safe_generation_example(p2, r2, tokenizer)

    records = [
        SafeGenerationRecord("ex1", "k1", ids1, end1, 0, False, False, "g", "train"),
        SafeGenerationRecord("ex2", "k2", ids2, end2, 1, True, False, "g", "train"),
    ]

    collator = DataCollatorForSafeGenerationTraining(pad_token_id=tokenizer.pad_token_id)
    batch = collator(records)

    B, T = batch["input_ids"].shape
    logits = torch.randn(B, T, 32000, requires_grad=True)

    # Compute loss with attention_mask (excluding padding)
    loss = safe_generation_loss(
        logits,
        batch["input_ids"],
        batch["prompt_end_indices"],
        attention_mask=batch["attention_mask"],
    )

    loss.backward()

    # Logits corresponding to padding target predictions should have exactly zero gradient
    # For row 0 (shorter sequence):
    len1 = len(ids1)
    # The last valid target is at index len1 - 1, predicted by logit at len1 - 2.
    # Prediction positions >= len1 - 1 predict targets in padding (>= len1), so their grad must be zero.
    pad_logit_grads = logits.grad[0, len1 - 1 :, :]
    assert (pad_logit_grads == 0.0).all(), "Gradients leaked into padding target positions!"
