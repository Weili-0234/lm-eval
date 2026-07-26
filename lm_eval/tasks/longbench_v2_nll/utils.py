"""LongBench v2 scored by choice-NLL (no generation).

Official zero-shot prompt (THUDM/LongBench-v2 prompts/0shot.txt) with the
official middle-truncation rule (keep first + last half of the tokenized
prompt). Instead of sampling a CoT and extracting a letter, we compare the
loglikelihood of the four choice letters after the official answer stem —
deterministic, extraction-free, and robust to thinking-mode response drift.

Dataset is revision-pinned; truncation is deterministic given the tokenizer
(identity asserted upstream by the runner) and LB2_PROMPT_TOKENS, so all
served rows score the exact same rendered documents.
"""

import os

import datasets

from lm_eval.tasks.ruler.common_utils import get_tokenizer


LB2_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"

# verbatim from THUDM/LongBench-v2 prompts/0shot.txt, minus the trailing
# 'The correct answer is (' stem which the task supplies as gen_prefix
TEMPLATE = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Format your response as follows: "The correct answer is (insert answer here)"."""

LENGTH_BUCKETS = ("short", "medium", "long")


def get_longbench_v2(**kwargs):
    pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", ""))
    tok = get_tokenizer(pretrained)
    budget = int(os.environ.get("LB2_PROMPT_TOKENS", "130560"))

    ds = datasets.load_dataset(
        "THUDM/LongBench-v2", split="train", revision=LB2_REVISION
    )

    def render(doc):
        prompt = TEMPLATE.format(
            context=doc["context"],
            question=doc["question"],
            choice_A=doc["choice_A"],
            choice_B=doc["choice_B"],
            choice_C=doc["choice_C"],
            choice_D=doc["choice_D"],
        )
        ids = tok(prompt, add_special_tokens=False).input_ids
        if len(ids) > budget:
            half = budget // 2
            prompt = tok.decode(ids[:half], skip_special_tokens=False) + tok.decode(
                ids[-half:], skip_special_tokens=False
            )
        return {"input": prompt, "prompt_tokens": min(len(ids), budget)}

    ds = ds.map(render, remove_columns=["context"])
    return {"test": ds}


def process_results(doc: dict, results: list) -> dict[str, float]:
    lls = [float(r[0]) for r in results]
    pred = max(range(len(lls)), key=lambda i: lls[i])
    gold = "ABCD".index(doc["answer"])
    acc = float(pred == gold)
    out = {"acc": acc}
    for bucket in LENGTH_BUCKETS:
        out[f"acc_{bucket}"] = acc if doc["length"] == bucket else -1.0
    return out


def aggregate_bucket(values: list[float]) -> float:
    kept = [v for v in values if v != -1]
    if not kept:
        return -1
    return sum(kept) / len(kept)
