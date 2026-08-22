"""Canonical FineWeb Pilot-v2 materialization and manifest generation for Task 7.2.

Re-exports canonical Task 7.2 materializer functions.
"""

from ccpt.data.canonical_materializer import (
    FINEWEB_SOURCE_CONFIG,
    FINEWEB_SOURCE_REPO,
    FINEWEB_SOURCE_REVISION,
    TARGET_PERSISTENCE_BLOCKS,
    TARGET_TOTAL_TRAIN_BLOCKS,
    TARGET_TRAIN_PREFIX_BLOCKS,
    TARGET_VAL_BLOCKS,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    build_task7_2_data_manifest,
    compute_ordered_shards_hash,
    materialize_bounded_canonical_fineweb_proof,
)

# Backward-compatibility alias
build_task7_1_data_manifest = build_task7_2_data_manifest
