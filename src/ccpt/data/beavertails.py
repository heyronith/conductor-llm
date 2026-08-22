"""Authoritative Out-Of-Distribution (OOD) BeaverTails evaluation data loader.

Loads and samples real prompts from PKU-Alignment/BeaverTails:
- Pinned Repository: PKU-Alignment/BeaverTails
- Pinned Revision: c8306df1cb6c813589b2184d0938ffdf90cb2b00
- Split: 30k_test (or test)
- Harmful criteria: is_safe == False
- Benign criteria: is_safe == True
- Strict isolation: BeaverTails prompts are NEVER used in training or tuning.
"""

import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from ccpt.data.hashing import sha256_json, sha256_text


BEAVERTAILS_SOURCE_REPO = "PKU-Alignment/BeaverTails"
BEAVERTAILS_SOURCE_REVISION = "c8306df1cb6c813589b2184d0938ffdf90cb2b00"
BEAVERTAILS_DEFAULT_SPLIT = "30k_test"


def sample_beavertails_prompts_deterministic(
    records: List[Dict[str, Any]],
    n_harmful: int = 256,
    n_benign: int = 256,
    seed: int = 20260822,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Deterministically samples harmful and benign prompts from BeaverTails records."""
    harmful_candidates = []
    benign_candidates = []

    for idx, r in enumerate(records):
        prompt = r.get("prompt", "").strip()
        if not prompt:
            continue
        is_safe = r.get("is_safe", None)
        category = r.get("category", "")
        item_id = str(r.get("id", idx))

        if is_safe is False:
            harmful_candidates.append({"id": item_id, "prompt": prompt, "category": category})
        elif is_safe is True:
            benign_candidates.append({"id": item_id, "prompt": prompt, "category": category})

    rng = random.Random(seed)
    # Sort deterministically before sampling
    harmful_candidates.sort(key=lambda x: (x["prompt"], x["id"]))
    benign_candidates.sort(key=lambda x: (x["prompt"], x["id"]))

    rng.shuffle(harmful_candidates)
    rng.shuffle(benign_candidates)

    selected_harmful = harmful_candidates[:n_harmful]
    selected_benign = benign_candidates[:n_benign]

    harmful_prompts = [x["prompt"] for x in selected_harmful]
    benign_prompts = [x["prompt"] for x in selected_benign]

    manifest = {
        "dataset_repo": BEAVERTAILS_SOURCE_REPO,
        "dataset_revision": BEAVERTAILS_SOURCE_REVISION,
        "split": BEAVERTAILS_DEFAULT_SPLIT,
        "sampling_seed": seed,
        "harmful_count": len(harmful_prompts),
        "benign_count": len(benign_prompts),
        "harmful_sample_ids": [x["id"] for x in selected_harmful],
        "benign_sample_ids": [x["id"] for x in selected_benign],
        "harmful_prompts_hash": hashlib.sha256("\n".join(harmful_prompts).encode("utf-8")).hexdigest(),
        "benign_prompts_hash": hashlib.sha256("\n".join(benign_prompts).encode("utf-8")).hexdigest(),
    }
    manifest["manifest_hash"] = sha256_json(manifest)

    return harmful_prompts, benign_prompts, manifest


def load_beavertails_ood_dataset(
    split: str = BEAVERTAILS_DEFAULT_SPLIT,
    n_harmful: int = 256,
    n_benign: int = 256,
    seed: int = 20260822,
    mock_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Loads and samples BeaverTails OOD prompts from HuggingFace or mock source."""
    if mock_records is not None:
        records = mock_records
    else:
        from datasets import load_dataset
        ds = load_dataset(
            BEAVERTAILS_SOURCE_REPO,
            split=split,
            revision=BEAVERTAILS_SOURCE_REVISION,
        )
        records = [dict(item) for item in ds]

    return sample_beavertails_prompts_deterministic(
        records=records,
        n_harmful=n_harmful,
        n_benign=n_benign,
        seed=seed,
    )
