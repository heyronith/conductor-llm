"""Right-padding PyTorch data collators for risk and safe-generation batches."""

from typing import Any, Dict, List, Sequence, Union

import torch

from ccpt.data.wildguard import RiskRecord, SafeGenerationRecord


class DataCollatorForRiskTraining:
    """Collator for batching variable-length risk classification examples with right-padding."""

    def __init__(self, pad_token_id: int = 2) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        records: Sequence[Union[RiskRecord, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        batch_size = len(records)
        if batch_size == 0:
            raise ValueError("Cannot collate empty batch")

        # Extract record attributes
        all_ids: List[List[int]] = []
        prompt_ends: List[int] = []
        labels: List[int] = []
        ex_ids: List[str] = []

        for rec in records:
            if isinstance(rec, RiskRecord):
                all_ids.append(rec.input_ids)
                prompt_ends.append(rec.prompt_end_index)
                labels.append(rec.risk_label)
                ex_ids.append(rec.example_id)
            else:
                all_ids.append(rec["input_ids"])
                prompt_ends.append(rec["prompt_end_index"])
                labels.append(rec["risk_label"])
                ex_ids.append(rec["example_id"])

        max_len = max(len(ids) for ids in all_ids)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, ids in enumerate(all_ids):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_end_indices": torch.tensor(prompt_ends, dtype=torch.long),
            "risk_labels": torch.tensor(labels, dtype=torch.long),
            "example_ids": ex_ids,
        }


class DataCollatorForSafeGenerationTraining:
    """Collator for batching variable-length safe-generation examples with right-padding."""

    def __init__(self, pad_token_id: int = 2) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        records: Sequence[Union[SafeGenerationRecord, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        batch_size = len(records)
        if batch_size == 0:
            raise ValueError("Cannot collate empty batch")

        all_ids: List[List[int]] = []
        prompt_ends: List[int] = []
        labels: List[int] = []
        is_refusals: List[bool] = []
        ex_ids: List[str] = []

        for rec in records:
            if isinstance(rec, SafeGenerationRecord):
                all_ids.append(rec.input_ids)
                prompt_ends.append(rec.prompt_end_index)
                labels.append(rec.risk_label)
                is_refusals.append(bool(rec.is_refusal))
                ex_ids.append(rec.example_id)
            elif isinstance(rec, dict):
                all_ids.append(rec["input_ids"])
                prompt_ends.append(rec["prompt_end_index"])
                labels.append(rec["risk_label"])
                is_refusals.append(bool(rec.get("is_refusal", False)))
                ex_ids.append(rec["example_id"])
            else:
                all_ids.append(getattr(rec, "input_ids"))
                prompt_ends.append(getattr(rec, "prompt_end_index"))
                labels.append(getattr(rec, "risk_label"))
                is_refusals.append(bool(getattr(rec, "is_refusal", False)))
                ex_ids.append(getattr(rec, "example_id"))

        max_len = max(len(ids) for ids in all_ids)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, ids in enumerate(all_ids):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_end_indices": torch.tensor(prompt_ends, dtype=torch.long),
            "risk_labels": torch.tensor(labels, dtype=torch.long),
            "is_refusals": torch.tensor(is_refusals, dtype=torch.bool),
            "example_ids": ex_ids,
        }


def pad_and_collate_risk_records(
    records: Sequence[Union[RiskRecord, Dict[str, Any]]],
    pad_token_id: int = 2,
):
    collator = DataCollatorForRiskTraining(pad_token_id=pad_token_id)
    res = collator(records)
    return res["input_ids"], res["prompt_end_indices"], res["risk_labels"].float(), res["attention_mask"]


def pad_and_collate_gen_records(
    records: Sequence[Union[SafeGenerationRecord, Dict[str, Any]]],
    pad_token_id: int = 2,
):
    """Pads and collates safe-generation records.

    Note: In historical Seed 1 code, the 4th return element erroneously returned
    `res["input_ids"]` instead of `res["is_refusals"]`. This did not affect Seed 1
    training gradients or losses because `compute_safe_generation_loss` consumes only
    `logits`, `input_ids`, `prompt_end_indices`, and `attention_mask`. In Task 7.4,
    this API is hardened to return the true `is_refusals` boolean tensor.
    """
    collator = DataCollatorForSafeGenerationTraining(pad_token_id=pad_token_id)
    res = collator(records)
    return res["input_ids"], res["prompt_end_indices"], res["risk_labels"].float(), res["is_refusals"], res["attention_mask"]

