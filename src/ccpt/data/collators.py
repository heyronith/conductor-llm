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
        ex_ids: List[str] = []

        for rec in records:
            if isinstance(rec, SafeGenerationRecord):
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
