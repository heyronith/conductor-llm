import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from transformers import PreTrainedTokenizerFast

from ccpt.data.config import DataConfig
from ccpt.data.hashing import sha256_file, sha256_json, sha256_text, stable_hash_int


@dataclass(frozen=True)
class RiskRecord:
    """Prepared tokenized record for risk classification training."""

    example_id: str
    prompt_group_key: str
    input_ids: List[int]
    prompt_end_index: int
    risk_label: int  # 0 = benign/unharmful, 1 = harmful
    is_adversarial: bool
    subcategory: str
    split: str  # 'train' or 'validation'


@dataclass(frozen=True)
class SafeGenerationRecord:
    """Prepared tokenized record for safe-generation training."""

    example_id: str
    prompt_group_key: str
    input_ids: List[int]
    prompt_end_index: int
    risk_label: int
    is_refusal: bool
    is_adversarial: bool
    subcategory: str
    split: str  # 'train' or 'validation'


RISK_ARROW_SCHEMA = pa.schema([
    ("example_id", pa.string()),
    ("prompt_group_key", pa.string()),
    ("input_ids", pa.list_(pa.uint16())),
    ("prompt_end_index", pa.int32()),
    ("risk_label", pa.int8()),
    ("is_adversarial", pa.bool_()),
    ("subcategory", pa.string()),
    ("split", pa.string()),
])

SAFE_GEN_ARROW_SCHEMA = pa.schema([
    ("example_id", pa.string()),
    ("prompt_group_key", pa.string()),
    ("input_ids", pa.list_(pa.uint16())),
    ("prompt_end_index", pa.int32()),
    ("risk_label", pa.int8()),
    ("is_refusal", pa.bool_()),
    ("is_adversarial", pa.bool_()),
    ("subcategory", pa.string()),
    ("split", pa.string()),
])


def canonicalize_prompt(prompt: str) -> str:
    """Produce canonical prompt key for grouping and splitting.

    Applies Unicode NFC normalization, converts all CRLF/CR to LF, and strips outer whitespace.
    """
    if not prompt:
        return ""
    nfc_text = unicodedata.normalize("NFC", prompt)
    lf_text = nfc_text.replace("\r\n", "\n").replace("\r", "\n")
    return lf_text.strip()


def is_validation_prompt_group(canonical_key: str, val_percentage: int = 5) -> bool:
    """Deterministic 95/5 prompt-group split rule using SHA256."""
    val = stable_hash_int(f"wildguard_group_split_v1:{canonical_key}", modulo=100)
    return val < val_percentage


def format_safety_prefix(prompt: str) -> str:
    """Format the canonical prompt framing string."""
    clean_prompt = prompt.replace("\r\n", "\n").replace("\r", "\n")
    return f"User: {clean_prompt}\nAssistant:"


def format_safety_response(response: str) -> str:
    """Format the safe continuation response string."""
    clean_resp = response.replace("\r\n", "\n").replace("\r", "\n")
    return f" {clean_resp}"


def tokenize_risk_example(
    prompt: str,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 1024,
) -> Tuple[List[int], int]:
    """Tokenize prompt framing for risk classification with left-truncation if needed."""
    prefix_text = format_safety_prefix(prompt)
    encoded = tokenizer.encode(prefix_text, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + encoded

    if len(input_ids) > max_seq_len:
        suffix_ids = tokenizer.encode("\nAssistant:", add_special_tokens=False)
        suffix_len = len(suffix_ids)
        content_budget = max_seq_len - 1 - suffix_len
        content_ids = input_ids[1:-suffix_len] if suffix_len > 0 else input_ids[1:]
        truncated_content = content_ids[-content_budget:]
        input_ids = [tokenizer.bos_token_id] + truncated_content + suffix_ids

    prompt_end_index = len(input_ids) - 1
    return input_ids, prompt_end_index


def tokenize_safe_generation_example(
    prompt: str,
    response: str,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 1024,
) -> Optional[Tuple[List[int], int, bool]]:
    """Tokenize prompt framing and response continuation with left/right truncation.

    Returns:
        (input_ids, prompt_end_index, was_truncated) or None if response exhausted.
    """
    prefix_text = format_safety_prefix(prompt)
    resp_text = format_safety_response(response)

    prefix_ids = [tokenizer.bos_token_id] + tokenizer.encode(prefix_text, add_special_tokens=False)
    resp_ids = tokenizer.encode(resp_text, add_special_tokens=False) + [tokenizer.eos_token_id]

    total_len = len(prefix_ids) + len(resp_ids)
    was_truncated = False

    if total_len > max_seq_len:
        was_truncated = True
        excess = total_len - max_seq_len
        suffix_ids = tokenizer.encode("\nAssistant:", add_special_tokens=False)
        suffix_len = len(suffix_ids)

        # Step 1: Truncate prompt from the LEFT first
        prompt_content_ids = prefix_ids[1:-suffix_len] if suffix_len > 0 else prefix_ids[1:]
        prompt_truncatable = max(0, len(prompt_content_ids) - 8)
        prompt_cut = min(excess, prompt_truncatable)

        if prompt_cut > 0:
            prompt_content_ids = prompt_content_ids[prompt_cut:]
            prefix_ids = [tokenizer.bos_token_id] + prompt_content_ids + suffix_ids
            excess -= prompt_cut

        # Step 2: If still too long, truncate response from the RIGHT (preserving final EOS)
        if excess > 0:
            resp_content_ids = resp_ids[:-1]  # exclude EOS
            resp_cut = excess
            if resp_cut >= len(resp_content_ids):
                # No response tokens remain
                return None
            resp_content_ids = resp_content_ids[:-resp_cut]
            resp_ids = resp_content_ids + [tokenizer.eos_token_id]

    input_ids = prefix_ids + resp_ids
    prompt_end_index = len(prefix_ids) - 1

    if prompt_end_index >= len(input_ids) - 1:
        return None

    return input_ids, prompt_end_index, was_truncated


def is_eligible_safe_generation_row(row: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Determine whether a WildGuard raw row satisfies the pilot safe-generation filtering rules."""
    resp = row.get("response")
    if not resp or not str(resp).strip():
        return False, "missing_response"

    p_harm = row.get("prompt_harm_label")
    r_harm = row.get("response_harm_label")
    r_refusal = row.get("response_refusal_label")

    if not p_harm or not r_harm or not r_refusal:
        return False, "missing_labels"

    p_harm = str(p_harm).strip().lower()
    r_harm = str(r_harm).strip().lower()
    r_refusal = str(r_refusal).strip().lower()

    if r_harm == "harmful":
        return False, "harmful_response"

    if p_harm == "harmful" and r_harm == "unharmful" and r_refusal == "refusal":
        return True, "harmful_refusal"

    if p_harm == "unharmful" and r_harm == "unharmful" and r_refusal == "compliance":
        return True, "benign_compliance"

    if p_harm == "harmful" and r_refusal == "compliance":
        return False, "harmful_compliance"

    if p_harm == "unharmful" and r_refusal == "refusal":
        return False, "benign_refusal"

    return False, "unrecognized_combination"


def compute_length_percentiles(lengths: List[int]) -> Dict[str, float]:
    """Compute standard summary percentiles for a list of integer lengths."""
    if not lengths:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.array(lengths)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def process_wildguard_raw_dataset(
    raw_rows: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    dataset_revision: str,
    max_seq_len: int = 1024,
) -> Dict[str, Any]:
    """Process a raw list of WildGuard dictionaries into verified risk and generation datasets with full statistics."""
    prompt_groups: Dict[str, List[Dict[str, Any]]] = {}
    prompt_only_rows_count = 0
    response_containing_rows_count = 0

    for row in raw_rows:
        prompt = row.get("prompt", "")
        key = canonicalize_prompt(prompt)
        if not key:
            continue
        prompt_groups.setdefault(key, []).append(row)
        resp = row.get("response")
        if resp and str(resp).strip():
            response_containing_rows_count += 1
        else:
            prompt_only_rows_count += 1

    usable_groups: Dict[str, Dict[str, Any]] = {}
    conflicting_groups = 0
    total_prompt_groups = len(prompt_groups)

    for key, rows in prompt_groups.items():
        harm_labels = {
            str(r.get("prompt_harm_label", "")).strip().lower()
            for r in rows
            if r.get("prompt_harm_label") is not None
        }
        harm_labels.discard("")
        if len(harm_labels) != 1:
            conflicting_groups += 1
            continue

        label_str = next(iter(harm_labels))
        if label_str not in ("harmful", "unharmful"):
            conflicting_groups += 1
            continue

        risk_val = 1 if label_str == "harmful" else 0
        first_row = rows[0]
        usable_groups[key] = {
            "representative_prompt": first_row.get("prompt", ""),
            "risk_label": risk_val,
            "is_adversarial": bool(first_row.get("adversarial", False)),
            "subcategory": str(first_row.get("subcategory", "")),
            "rows": rows,
        }

    risk_train_records: List[RiskRecord] = []
    risk_val_records: List[RiskRecord] = []
    harmful_risk_count = 0
    benign_risk_count = 0
    risk_prompt_lengths: List[int] = []
    risk_truncated_count = 0

    for key, grp in usable_groups.items():
        is_val = is_validation_prompt_group(key)
        split_name = "validation" if is_val else "train"
        raw_prefix_tokens = [tokenizer.bos_token_id] + tokenizer.encode(
            format_safety_prefix(grp["representative_prompt"]), add_special_tokens=False
        )
        risk_prompt_lengths.append(len(raw_prefix_tokens))
        if len(raw_prefix_tokens) > max_seq_len:
            risk_truncated_count += 1

        input_ids, prompt_end_index = tokenize_risk_example(
            grp["representative_prompt"],
            tokenizer,
            max_seq_len=max_seq_len,
        )
        example_id = sha256_text(f"{dataset_revision}:risk:{key}")
        rec = RiskRecord(
            example_id=example_id,
            prompt_group_key=key,
            input_ids=input_ids,
            prompt_end_index=prompt_end_index,
            risk_label=grp["risk_label"],
            is_adversarial=grp["is_adversarial"],
            subcategory=grp["subcategory"],
            split=split_name,
        )
        if grp["risk_label"] == 1:
            harmful_risk_count += 1
        else:
            benign_risk_count += 1

        if is_val:
            risk_val_records.append(rec)
        else:
            risk_train_records.append(rec)

    gen_train_records: List[SafeGenerationRecord] = []
    gen_val_records: List[SafeGenerationRecord] = []
    eligible_harmful_refusal = 0
    eligible_benign_compliance = 0
    excluded_stats: Dict[str, int] = {}

    gen_prompt_token_lengths: List[int] = []
    gen_resp_token_lengths: List[int] = []
    gen_combined_token_lengths: List[int] = []
    gen_truncated_examples_count = 0
    prompt_left_truncated_count = 0
    response_right_truncated_count = 0
    total_eligible_candidates = 0

    for key, grp in usable_groups.items():
        is_val = is_validation_prompt_group(key)
        split_name = "validation" if is_val else "train"
        prompt_text = grp["representative_prompt"]

        for row in grp["rows"]:
            is_eligible, reason = is_eligible_safe_generation_row(row)
            if not is_eligible:
                excluded_stats[reason] = excluded_stats.get(reason, 0) + 1
                continue

            total_eligible_candidates += 1
            if reason == "harmful_refusal":
                eligible_harmful_refusal += 1
                is_refusal = True
            else:
                eligible_benign_compliance += 1
                is_refusal = False

            resp_text = str(row["response"])

            p_len = len(tokenizer.encode(format_safety_prefix(prompt_text), add_special_tokens=False)) + 1
            r_len = len(tokenizer.encode(format_safety_response(resp_text), add_special_tokens=False)) + 1
            gen_prompt_token_lengths.append(p_len)
            gen_resp_token_lengths.append(r_len)
            comb_len = p_len + r_len
            gen_combined_token_lengths.append(comb_len)

            if comb_len > max_seq_len:
                gen_truncated_examples_count += 1
                excess = comb_len - max_seq_len
                suffix_len = len(tokenizer.encode("\nAssistant:", add_special_tokens=False))
                prompt_content_len = max(0, p_len - 1 - suffix_len)
                prompt_truncatable = max(0, prompt_content_len - 8)
                prompt_cut = min(excess, prompt_truncatable)
                if prompt_cut > 0:
                    prompt_left_truncated_count += 1
                if excess - prompt_cut > 0:
                    response_right_truncated_count += 1

            tokenized_res = tokenize_safe_generation_example(
                prompt_text,
                resp_text,
                tokenizer,
                max_seq_len=max_seq_len,
            )
            if tokenized_res is None:
                excluded_stats["truncation_exhausted_response"] = (
                    excluded_stats.get("truncation_exhausted_response", 0) + 1
                )
                continue

            input_ids, prompt_end_index, was_truncated = tokenized_res

            example_id = sha256_text(f"{dataset_revision}:gen:{key}:{resp_text}")
            rec = SafeGenerationRecord(
                example_id=example_id,
                prompt_group_key=key,
                input_ids=input_ids,
                prompt_end_index=prompt_end_index,
                risk_label=grp["risk_label"],
                is_refusal=is_refusal,
                is_adversarial=bool(row.get("adversarial", False)),
                subcategory=str(row.get("subcategory", "")),
                split=split_name,
            )
            if is_val:
                gen_val_records.append(rec)
            else:
                gen_train_records.append(rec)

    length_stats = {
        "risk_prompt_token_lengths": compute_length_percentiles(risk_prompt_lengths),
        "risk_total_records": len(usable_groups),
        "risk_truncated_count": risk_truncated_count,
        "risk_truncated_fraction": (risk_truncated_count / max(1, len(usable_groups))),
        "prompt_token_lengths": compute_length_percentiles(gen_prompt_token_lengths),
        "response_token_lengths": compute_length_percentiles(gen_resp_token_lengths),
        "combined_token_lengths": compute_length_percentiles(gen_combined_token_lengths),
        "total_eligible_candidates": total_eligible_candidates,
        "truncated_count": gen_truncated_examples_count,
        "truncated_fraction": (gen_truncated_examples_count / max(1, total_eligible_candidates)),
        "prompt_left_truncated_count": prompt_left_truncated_count,
        "response_right_truncated_count": response_right_truncated_count,
        "exhausted_response_count": excluded_stats.get("truncation_exhausted_response", 0),
    }

    return {
        "total_raw_rows": len(raw_rows),
        "prompt_only_rows_count": prompt_only_rows_count,
        "response_containing_rows_count": response_containing_rows_count,
        "total_prompt_groups": total_prompt_groups,
        "usable_prompt_groups": len(usable_groups),
        "conflicting_groups": conflicting_groups,
        "harmful_risk_count": harmful_risk_count,
        "benign_risk_count": benign_risk_count,
        "risk_train_records": risk_train_records,
        "risk_val_records": risk_val_records,
        "gen_train_records": gen_train_records,
        "gen_val_records": gen_val_records,
        "eligible_harmful_refusal": eligible_harmful_refusal,
        "eligible_benign_compliance": eligible_benign_compliance,
        "length_and_truncation_stats": length_stats,
        "excluded_stats": excluded_stats,
    }


def save_wildguard_records_arrow(
    records: Sequence[Union[RiskRecord, SafeGenerationRecord]],
    output_path: Path,
    record_type: str = "risk",
) -> str:
    """Save prepared records to a PyArrow IPC (.arrow) file and return file SHA256."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = RISK_ARROW_SCHEMA if record_type == "risk" else SAFE_GEN_ARROW_SCHEMA

    example_ids = [r.example_id for r in records]
    prompt_group_keys = [r.prompt_group_key for r in records]
    input_ids = [r.input_ids for r in records]
    prompt_end_indices = [r.prompt_end_index for r in records]
    risk_labels = [r.risk_label for r in records]
    is_adversarials = [r.is_adversarial for r in records]
    subcategories = [r.subcategory for r in records]
    splits = [r.split for r in records]

    if record_type == "risk":
        arrays = [
            pa.array(example_ids, type=pa.string()),
            pa.array(prompt_group_keys, type=pa.string()),
            pa.array(input_ids, type=pa.list_(pa.uint16())),
            pa.array(prompt_end_indices, type=pa.int32()),
            pa.array(risk_labels, type=pa.int8()),
            pa.array(is_adversarials, type=pa.bool_()),
            pa.array(subcategories, type=pa.string()),
            pa.array(splits, type=pa.string()),
        ]
    else:
        is_refusals = [r.is_refusal for r in records]
        arrays = [
            pa.array(example_ids, type=pa.string()),
            pa.array(prompt_group_keys, type=pa.string()),
            pa.array(input_ids, type=pa.list_(pa.uint16())),
            pa.array(prompt_end_indices, type=pa.int32()),
            pa.array(risk_labels, type=pa.int8()),
            pa.array(is_refusals, type=pa.bool_()),
            pa.array(is_adversarials, type=pa.bool_()),
            pa.array(subcategories, type=pa.string()),
            pa.array(splits, type=pa.string()),
        ]

    table = pa.Table.from_arrays(arrays, schema=schema)
    with pa.OSFile(str(output_path), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_table(table)

    return sha256_file(output_path)


def load_wildguard_records_arrow(
    input_path: Path,
    record_type: str = "risk",
) -> List[Union[RiskRecord, SafeGenerationRecord]]:
    """Load prepared records from a PyArrow IPC (.arrow) file."""
    with pa.OSFile(str(input_path), "rb") as source:
        with ipc.open_file(source) as reader:
            table = reader.read_all()

    pydict = table.to_pydict()
    num_rows = len(table)
    records: List[Union[RiskRecord, SafeGenerationRecord]] = []
    for i in range(num_rows):
        raw_ids = pydict["input_ids"][i]
        input_ids = [int(tok) for tok in raw_ids]
        if record_type == "risk":
            rec = RiskRecord(
                example_id=str(pydict["example_id"][i]),
                prompt_group_key=str(pydict["prompt_group_key"][i]),
                input_ids=input_ids,
                prompt_end_index=int(pydict["prompt_end_index"][i]),
                risk_label=int(pydict["risk_label"][i]),
                is_adversarial=bool(pydict["is_adversarial"][i]),
                subcategory=str(pydict["subcategory"][i]),
                split=str(pydict["split"][i]),
            )
        elif record_type == "generation":
            rec = SafeGenerationRecord(
                example_id=str(pydict["example_id"][i]),
                prompt_group_key=str(pydict["prompt_group_key"][i]),
                input_ids=input_ids,
                prompt_end_index=int(pydict["prompt_end_index"][i]),
                risk_label=int(pydict["risk_label"][i]),
                is_refusal=bool(pydict["is_refusal"][i]),
                is_adversarial=bool(pydict["is_adversarial"][i]),
                subcategory=str(pydict["subcategory"][i]),
                split=str(pydict["split"][i]),
            )
        else:
            raise ValueError(f"Unknown record_type: {record_type}")
        records.append(rec)
    return records


def save_wildguard_records(
    records: Sequence[Union[RiskRecord, SafeGenerationRecord]],
    output_path: Path,
    record_type: str = "risk",
) -> str:
    """Save prepared records to disk (Arrow IPC if .arrow, JSONL if .jsonl)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".arrow":
        return save_wildguard_records_arrow(records, output_path, record_type=record_type)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    return sha256_file(output_path)


def load_wildguard_records(
    input_path: Union[Path, str],
    record_type: str = "risk",
) -> List[Union[RiskRecord, SafeGenerationRecord]]:
    """Load prepared records from disk (Arrow IPC if .arrow, JSONL if .jsonl)."""
    input_path = Path(input_path)
    if input_path.suffix == ".arrow":
        return load_wildguard_records_arrow(input_path, record_type=record_type)

    records: List[Union[RiskRecord, SafeGenerationRecord]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if record_type == "risk":
                records.append(RiskRecord(**data))
            elif record_type == "generation":
                records.append(SafeGenerationRecord(**data))
            else:
                raise ValueError(f"Unknown record_type: {record_type}")
    return records


def sample_wildguard_id_behavior_prompts(
    records: Sequence[RiskRecord],
    tokenizer: PreTrainedTokenizerFast,
    n_harmful: int = 256,
    n_benign: int = 256,
    salt: str = "task7_3_id_behavior_v1:",
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Deterministically selects harmful and benign prompt texts from WildGuard risk validation records using salted hash ranking."""
    import hashlib
    harmful_candidates = []
    benign_candidates = []

    for r in records:
        if r.risk_label == 1:
            key = stable_hash_int(salt + r.example_id, modulo=2**63)
            harmful_candidates.append((key, r.example_id, r))
        elif r.risk_label == 0:
            key = stable_hash_int(salt + r.example_id, modulo=2**63)
            benign_candidates.append((key, r.example_id, r))

    harmful_candidates.sort(key=lambda x: (x[0], x[1]))
    benign_candidates.sort(key=lambda x: (x[0], x[1]))

    if len(harmful_candidates) < n_harmful:
        raise ValueError(f"Insufficient harmful candidates: {len(harmful_candidates)} < {n_harmful}")
    if len(benign_candidates) < n_benign:
        raise ValueError(f"Insufficient benign candidates: {len(benign_candidates)} < {n_benign}")

    selected_harmful = harmful_candidates[:n_harmful]
    selected_benign = benign_candidates[:n_benign]

    harmful_prompts = []
    harmful_ids = []
    for _, ex_id, rec in selected_harmful:
        prompt_tokens = rec.input_ids[: rec.prompt_end_index + 1]
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        harmful_prompts.append(prompt_text)
        harmful_ids.append(ex_id)

    benign_prompts = []
    benign_ids = []
    for _, ex_id, rec in selected_benign:
        prompt_tokens = rec.input_ids[: rec.prompt_end_index + 1]
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        benign_prompts.append(prompt_text)
        benign_ids.append(ex_id)

    manifest = {
        "selection_algorithm": "salted_hash_ranking_v1",
        "salt": salt,
        "n_harmful": n_harmful,
        "n_benign": n_benign,
        "harmful_example_ids": harmful_ids,
        "benign_example_ids": benign_ids,
        "harmful_ids_hash": hashlib.sha256("\n".join(harmful_ids).encode("utf-8")).hexdigest(),
        "benign_ids_hash": hashlib.sha256("\n".join(benign_ids).encode("utf-8")).hexdigest(),
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return harmful_prompts, benign_prompts, manifest

