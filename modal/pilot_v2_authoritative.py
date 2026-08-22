"""Modal Pilot-v2 Authoritative Production Pipeline (Locked Pending Independent Review).

Future Production Skeleton wiring together:
- Canonical FineWeb streaming materializer (HuggingFaceFW/fineweb-edu@87f09149... & Mistral tokenizer)
- Checkpoint Format V2 strict production validation
- Mandatory JSONL progress reporting with measured GPU wall-second cost accounting
- Real BeaverTails OOD dataset evaluation (PKU-Alignment/BeaverTails@8401fe60...)
- Real WildGuard behavioral safety judging (allenai/wildguard@cbba4823...)
- Sequential persistence continuation iterator ([976544, 1008544))
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import modal

# -----------------------------------------------------------------------------
# Modal App & Container Configuration
# -----------------------------------------------------------------------------

app = modal.App("ccpt-pilot-v2-authoritative")

authoritative_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "tokenizers>=0.19.0",
        "datasets>=2.19.0",
        "huggingface_hub>=0.23.0",
        "sentencepiece>=0.2.0",
        "tiktoken>=0.7.0",
        "accelerate>=0.29.0",
        "pyarrow>=15.0.0",
        "numpy>=1.24.0",
        "pytest>=8.0.0",
    )
    .add_local_python_source("ccpt")
    .add_local_dir("tests", remote_path="/root/tests")
)

# Canonical Authoritative Volumes (Strictly Task 7.2+ namespace, zero legacy task6)
data_volume = modal.Volume.from_name("ccpt-authoritative-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("ccpt-authoritative-runs", create_if_missing=True)


# -----------------------------------------------------------------------------
# Remote Functions (Skeletons)
# -----------------------------------------------------------------------------

@app.function(image=authoritative_image, volumes={"/data": data_volume}, timeout=7200)
def materialize_authoritative_fineweb():
    """Materializes canonical FineWeb dataset blocks using Task 4 data functions."""
    from ccpt.data.canonical_materializer import (
        FINEWEB_SOURCE_CONFIG,
        FINEWEB_SOURCE_REPO,
        FINEWEB_SOURCE_REVISION,
        TOKENIZER_REPO,
        TOKENIZER_REVISION,
    )
    return {
        "status": "skeleton_ready",
        "source_repo": FINEWEB_SOURCE_REPO,
        "source_config": FINEWEB_SOURCE_CONFIG,
        "source_revision": FINEWEB_SOURCE_REVISION,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
    }


@app.function(image=authoritative_image, volumes={"/data": data_volume, "/runs": runs_volume}, gpu="H100", timeout=14400)
def train_authoritative_1b_trunk(model_type: str, data_manifest_hash: str):
    """Trains a fresh 1B LM trunk using strict Checkpoint V2 and mandatory JSONL."""
    from ccpt.training.checkpoint import CHECKPOINT_FORMAT_VERSION_V2
    from ccpt.training.progress import LiveProgressReporter
    from ccpt.training.cost import compute_gpu_cost
    # Fail-closed until unlocked
    raise RuntimeError("Authoritative 1B trunk training is locked pending Task 7.2.1 review.")


@app.function(image=authoritative_image, volumes={"/data": data_volume, "/runs": runs_volume}, gpu="H100", timeout=14400)
def train_authoritative_20m_safety(model_type: str, schedule_batches: List[Any], schedule_hash: str):
    """Trains 20M safety using strict Checkpoint V2 with non-empty safety schedule hash."""
    raise RuntimeError("Authoritative 20M safety training is locked pending Task 7.2.1 review.")


@app.function(image=authoritative_image, volumes={"/data": data_volume, "/runs": runs_volume}, gpu="H100", timeout=7200)
def evaluate_authoritative_safety_and_ood():
    """Evaluates safety using real WildGuard judge and real BeaverTails OOD."""
    from ccpt.evaluation.safety_judge import BehavioralSafetyJudge
    from ccpt.data.beavertails import load_beavertails_ood_dataset
    raise RuntimeError("Authoritative evaluation is locked pending Task 7.2.1 review.")


# -----------------------------------------------------------------------------
# Local Entrypoint (Locked)
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Fail-closed entrypoint for the authoritative Pilot-v2 pipeline."""
    raise RuntimeError(
        "Authoritative Pilot-v2 full run is locked pending independent Task 7.2.1 review."
    )
