"""Chat-template-friendly formatters for mmlu_pro_chat variant. No "Answer:" in user or assistant messages."""

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def doc_to_text(example):
    """Return question + options only (no Answer, no CoT)."""
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(example["options"]):
        if i >= len(choices):
            break
        prompt += f"{choices[i]}. {opt.strip()}\n"
    return (
        prompt
        + '\n\nThink step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.'
    )


def fewshot_doc_to_target(example):
    """Assistant message for fewshot: CoT + answer letter, without 'Answer:' prefix."""
    cot_content = example["cot_content"].replace(
        "A: Let's think step by step.", "Let's think step by step."
    )
    return cot_content


# --- Frozen 1,000-question subset (standardized fast protocol) -------------
#
# The manifest (mmlu-pro-chat-subsample-1000-seed-1234.json, kept next to this
# file) freezes a seed-1234 uniform sample of 1,000 rows from the pinned
# TIGER-Lab/MMLU-Pro test split (12,032 rows). Selection uses each item's
# `category_index` (zero-based position AFTER filtering the full test split by
# category), so it composes with the per-category task structure.
#
# Default behavior of the mmlu_pro_chat tasks is the 1,000-question subset.
# Set MMLU_PRO_CHAT_FULL=1 to run the full 12,032-question split instead.
# `--limit 1000` is NOT an acceptable substitute (it does not reproduce the
# frozen sample).

import functools
import hashlib
import json
import os

_MANIFEST_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mmlu-pro-chat-subsample-1000-seed-1234.json",
)


@functools.lru_cache(maxsize=1)
def _load_manifest():
    with open(_MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    items = manifest["items"]
    if len(items) != manifest["sampling"]["sample_size"]:
        raise ValueError("manifest item count != declared sample_size")
    row_indices = sorted(item["row_index"] for item in items)
    if len(set(row_indices)) != len(row_indices):
        raise ValueError("manifest row_index values are not unique")
    # The manifest fingerprint is sha256 over the comma-joined sorted indices.
    digest = hashlib.sha256(
        ",".join(str(i) for i in row_indices).encode("utf-8")
    ).hexdigest()
    expected = manifest["sampling"]["row_indices_sha256"]
    if digest != expected:
        raise ValueError(
            f"manifest row_indices sha256 mismatch: {digest} != {expected}"
        )
    per_category = {}
    for item in items:
        per_category.setdefault(item["category"], []).append(item)
    total = sum(len(v) for v in per_category.values())
    if total != manifest["sampling"]["sample_size"]:
        raise ValueError("per-category split does not sum to sample_size")
    return manifest, per_category


def _subset_enabled():
    return os.environ.get("MMLU_PRO_CHAT_FULL", "").strip() not in ("1", "true")


def process_docs_subset(dataset, subject):
    """Filter the full test split to `subject`, then (by default) select the
    frozen manifest rows for that category via category_index."""
    filtered = dataset.filter(lambda x: x["category"] == subject)
    if not _subset_enabled():
        return filtered
    _, per_category = _load_manifest()
    items = sorted(
        per_category.get(subject, []), key=lambda it: it["category_index"]
    )
    indices = [it["category_index"] for it in items]
    if indices and indices[-1] >= len(filtered):
        raise ValueError(
            f"manifest category_index out of range for '{subject}': "
            f"{indices[-1]} >= {len(filtered)} — dataset revision drift?"
        )
    selected = filtered.select(indices)
    for it, row in zip(items, selected):
        if row["question_id"] != it["question_id"]:
            raise ValueError(
                f"question_id mismatch in '{subject}' at category_index "
                f"{it['category_index']}: dataset {row['question_id']} != "
                f"manifest {it['question_id']} — dataset revision drift?"
            )
    return selected


def _make_subset_fn(subject):
    return functools.partial(process_docs_subset, subject=subject)


process_biology = _make_subset_fn("biology")
process_business = _make_subset_fn("business")
process_chemistry = _make_subset_fn("chemistry")
process_computer_science = _make_subset_fn("computer science")
process_economics = _make_subset_fn("economics")
process_engineering = _make_subset_fn("engineering")
process_health = _make_subset_fn("health")
process_history = _make_subset_fn("history")
process_law = _make_subset_fn("law")
process_math = _make_subset_fn("math")
process_other = _make_subset_fn("other")
process_philosophy = _make_subset_fn("philosophy")
process_physics = _make_subset_fn("physics")
process_psychology = _make_subset_fn("psychology")
