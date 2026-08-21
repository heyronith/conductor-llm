"""Tests for WildGuard schema validation, filtering rules, 95/5 splitting, and truncation."""

from ccpt.data.config import DataConfig
from ccpt.data.tokenizer import load_ccpt_tokenizer
from ccpt.data.wildguard import (
    RiskRecord,
    SafeGenerationRecord,
    canonicalize_prompt,
    is_eligible_safe_generation_row,
    is_validation_prompt_group,
    process_wildguard_raw_dataset,
    tokenize_risk_example,
    tokenize_safe_generation_example,
)


def test_canonicalize_prompt():
    """Verify Unicode NFC normalization, newline standardization, and whitespace stripping."""
    assert canonicalize_prompt("  Hello\r\nWorld  ") == "Hello\nWorld"
    assert canonicalize_prompt("Cafe\u0301") == "Café"  # NFD to NFC


def test_safe_generation_eligibility_rules():
    """Verify exact pilot safe-generation filtering rules."""
    # 1. Harmful prompt + Safe refusal -> ACCEPT
    ok, cat = is_eligible_safe_generation_row({
        "response": "I cannot help with that.",
        "prompt_harm_label": "harmful",
        "response_harm_label": "unharmful",
        "response_refusal_label": "refusal",
    })
    assert ok and cat == "harmful_refusal"

    # 2. Benign prompt + Safe compliance -> ACCEPT
    ok, cat = is_eligible_safe_generation_row({
        "response": "Here is the information.",
        "prompt_harm_label": "unharmful",
        "response_harm_label": "unharmful",
        "response_refusal_label": "compliance",
    })
    assert ok and cat == "benign_compliance"

    # 3. Harmful compliance -> EXCLUDE
    ok, cat = is_eligible_safe_generation_row({
        "response": "Sure, here is the bomb recipe.",
        "prompt_harm_label": "harmful",
        "response_harm_label": "harmful",
        "response_refusal_label": "compliance",
    })
    assert not ok and cat == "harmful_response"

    # 4. Benign refusal (over-refusal) -> EXCLUDE
    ok, cat = is_eligible_safe_generation_row({
        "response": "I cannot tell jokes.",
        "prompt_harm_label": "unharmful",
        "response_harm_label": "unharmful",
        "response_refusal_label": "refusal",
    })
    assert not ok and cat == "benign_refusal"

    # 5. Missing response or labels -> EXCLUDE
    ok, cat = is_eligible_safe_generation_row({"response": None})
    assert not ok and cat == "missing_response"


def test_wildguard_dataset_processing_and_conflicts():
    """Verify that contradictory prompt groups are excluded and prompt variants stay in the same split."""
    config = DataConfig()
    tokenizer = load_ccpt_tokenizer(config)

    raw_rows = [
        # Prompt group 1 (Harmful, 2 variants)
        {
            "prompt": "How to make a weapon?",
            "response": "I cannot do that.",
            "prompt_harm_label": "harmful",
            "response_harm_label": "unharmful",
            "response_refusal_label": "refusal",
        },
        {
            "prompt": "How to make a weapon?  ",  # Duplicate prompt with whitespace
            "response": "I refuse.",
            "prompt_harm_label": "harmful",
            "response_harm_label": "unharmful",
            "response_refusal_label": "refusal",
        },
        # Prompt group 2 (Benign, 1 variant)
        {
            "prompt": "What is the boiling point of water?",
            "response": "It is 100 degrees Celsius.",
            "prompt_harm_label": "unharmful",
            "response_harm_label": "unharmful",
            "response_refusal_label": "compliance",
        },
        # Prompt group 3 (Contradictory labels -> must be excluded)
        {
            "prompt": "Is this action legal?",
            "response": "Resp 1",
            "prompt_harm_label": "harmful",
            "response_harm_label": "unharmful",
            "response_refusal_label": "refusal",
        },
        {
            "prompt": "Is this action legal?",
            "response": "Resp 2",
            "prompt_harm_label": "unharmful",
            "response_harm_label": "unharmful",
            "response_refusal_label": "compliance",
        },
    ]

    processed = process_wildguard_raw_dataset(
        raw_rows,
        tokenizer=tokenizer,
        dataset_revision="test_rev",
        max_seq_len=1024,
    )

    assert processed["total_raw_rows"] == 5
    assert processed["total_prompt_groups"] == 3
    assert processed["usable_prompt_groups"] == 2
    assert processed["conflicting_groups"] == 1

    # Exactly 2 usable prompt groups produce 2 risk records (distributed across train/val)
    total_risk = len(processed["risk_train_records"]) + len(processed["risk_val_records"])
    assert total_risk == 2

    # Exactly 3 eligible safe-generation responses (2 from group 1, 1 from group 2)
    total_gen = len(processed["gen_train_records"]) + len(processed["gen_val_records"])
    assert total_gen == 3


def test_safety_truncation_policy():
    """Verify left truncation on prompt and right truncation on response under sequence limit."""
    config = DataConfig()
    tokenizer = load_ccpt_tokenizer(config)

    long_prompt = "word " * 600
    long_response = "answer " * 600

    # Max sequence length 128 for testing truncation logic
    tokenized = tokenize_safe_generation_example(
        long_prompt,
        long_response,
        tokenizer,
        max_seq_len=128,
    )

    input_ids, prompt_end_index, was_truncated = tokenized
    assert was_truncated is True

    assert len(input_ids) <= 128
    assert input_ids[0] == 1  # BOS
    assert input_ids[-1] == 2  # EOS
    assert prompt_end_index < len(input_ids) - 1, "At least one target response token must remain"
    assert input_ids[prompt_end_index] == 28747 or tokenizer.decode([input_ids[prompt_end_index]]) == ":"


def test_wildguard_record_persistence_jsonl(tmp_path):
    """Verify that RiskRecord and SafeGenerationRecord can be persisted to JSONL and loaded accurately."""
    risk_records = [
        RiskRecord("id1", "pk1", [1, 20, 30], 1, 1, False, "violence", "train"),
        RiskRecord("id2", "pk2", [1, 40, 50], 2, 0, True, "general", "validation"),
    ]
    gen_records = [
        SafeGenerationRecord("gid1", "pk1", [1, 20, 30, 40, 2], 2, 1, True, False, "violence", "train"),
        SafeGenerationRecord("gid2", "pk2", [1, 50, 60, 70, 2], 2, 0, False, True, "general", "validation"),
    ]

    r_path = tmp_path / "risk.jsonl"
    g_path = tmp_path / "gen.jsonl"

    from ccpt.data.wildguard import save_wildguard_records, load_wildguard_records
    save_wildguard_records(risk_records, r_path)
    save_wildguard_records(gen_records, g_path)

    loaded_r = load_wildguard_records(r_path, "risk")
    loaded_g = load_wildguard_records(g_path, "generation")

    assert loaded_r == risk_records
    assert loaded_g == gen_records


def test_full_record_hash_sensitivity():
    """Verify that changing any token in the middle of a record alters its logical hash."""
    from ccpt.data.manifests import compute_records_logical_hash

    rec1 = RiskRecord("id1", "pk1", [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 3, 1, False, "test", "train")
    rec2 = RiskRecord("id1", "pk1", [1, 10, 20, 30, 99, 50, 60, 70, 80, 90, 100], 3, 1, False, "test", "train")  # changed token at index 4

    h1 = compute_records_logical_hash([rec1])
    h2 = compute_records_logical_hash([rec2])

    assert h1 != h2, "Hash must be sensitive to internal token changes!"


def test_middle_token_mutation_hash_regression():
    """Specific regression test: identical records with ONLY middle token changed have different hashes."""
    from ccpt.data.manifests import compute_records_logical_hash

    rec_a = SafeGenerationRecord(
        example_id="ex1",
        prompt_group_key="key1",
        input_ids=[1, 10, 20, 30, 40, 2],
        prompt_end_index=2,
        risk_label=1,
        is_refusal=True,
        is_adversarial=False,
        subcategory="weapons",
        split="train",
    )
    rec_b = SafeGenerationRecord(
        example_id="ex1",
        prompt_group_key="key1",
        input_ids=[1, 10, 99, 30, 40, 2],  # only middle token differs
        prompt_end_index=2,
        risk_label=1,
        is_refusal=True,
        is_adversarial=False,
        subcategory="weapons",
        split="train",
    )

    hash_a = compute_records_logical_hash([rec_a])
    hash_b = compute_records_logical_hash([rec_b])

    assert hash_a != hash_b, "Middle token change must produce distinct logical hash!"


def test_wildguard_arrow_persistence_roundtrip(tmp_path):
    """Verify PyArrow IPC (.arrow) serialization preserves records, types, and logical hashes exactly."""
    from ccpt.data.manifests import compute_records_logical_hash
    from ccpt.data.wildguard import (
        load_wildguard_records_arrow,
        save_wildguard_records_arrow,
    )

    risk_records = [
        RiskRecord("id1", "pk1", [1, 20, 30, 31999], 1, 1, False, "violence", "train"),
        RiskRecord("id2", "pk2", [1, 40, 50, 60], 2, 0, True, "general", "validation"),
    ]
    gen_records = [
        SafeGenerationRecord("gid1", "pk1", [1, 20, 30, 40, 31999, 2], 2, 1, True, False, "violence", "train"),
        SafeGenerationRecord("gid2", "pk2", [1, 50, 60, 70, 80, 2], 2, 0, False, True, "general", "validation"),
    ]

    r_arrow_path = tmp_path / "risk.arrow"
    g_arrow_path = tmp_path / "gen.arrow"

    # Compute original hashes
    orig_risk_hash = compute_records_logical_hash(risk_records)
    orig_gen_hash = compute_records_logical_hash(gen_records)

    # Save to PyArrow IPC (.arrow)
    save_wildguard_records_arrow(risk_records, r_arrow_path, record_type="risk")
    save_wildguard_records_arrow(gen_records, g_arrow_path, record_type="generation")

    # Load back
    loaded_risk = load_wildguard_records_arrow(r_arrow_path, record_type="risk")
    loaded_gen = load_wildguard_records_arrow(g_arrow_path, record_type="generation")

    assert loaded_risk == risk_records
    assert loaded_gen == gen_records

    # Logical hashes must match bit-for-bit
    assert compute_records_logical_hash(loaded_risk) == orig_risk_hash
    assert compute_records_logical_hash(loaded_gen) == orig_gen_hash


def test_truncation_and_length_statistics_reporting():
    """Verify that process_wildguard_raw_dataset produces full percentile statistics and counts."""
    config = DataConfig()
    tokenizer = load_ccpt_tokenizer(config)

    raw_rows = [
        {
            "prompt": "Short prompt",
            "response": "Short answer",
            "prompt_harm_label": "unharmful",
            "response_harm_label": "unharmful",
            "response_refusal_label": "compliance",
            "subcategory": "general",
        },
        {
            "prompt": "A " * 600,
            "response": "B " * 600,
            "prompt_harm_label": "harmful",
            "response_harm_label": "unharmful",
            "response_refusal_label": "refusal",
            "subcategory": "weapons",
        },
    ]

    processed = process_wildguard_raw_dataset(
        raw_rows,
        tokenizer=tokenizer,
        dataset_revision="test_rev",
        max_seq_len=128,
    )

    stats = processed["length_and_truncation_stats"]
    assert "p50" in stats["prompt_token_lengths"]
    assert "p90" in stats["prompt_token_lengths"]
    assert "p95" in stats["prompt_token_lengths"]
    assert "p99" in stats["prompt_token_lengths"]
    assert "max" in stats["prompt_token_lengths"]
    assert stats["truncated_count"] > 0
    assert stats["truncated_fraction"] > 0.0
    assert "prompt_left_truncated_count" in stats
    assert "response_right_truncated_count" in stats

