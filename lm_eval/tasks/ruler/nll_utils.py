"""Gold-answer NLL variants of RULER tasks (Panel A).

The generate-mode RULER protocol is invalid for thinking-native chat models
(greedy raw completion + tiny gen budget elicits `<think>` openings instead of
answers). These variants instead measure the loglikelihood the served model
assigns to the gold answer appended after the task's own gen_prefix — no
generation, no answer extraction, robust to response-style drift.

Doc identity across rows is load-bearing (we report per-row NLL deltas on the
SAME documents): niah_multikey_3 draws uuid4 needles and vt draws unseeded
random chains, so regenerating per invocation yields different docs. The tasks
therefore ONLY load a frozen dataset produced once by
scripts/freeze_ruler_nll.py, addressed via RULER_NLL_DATA and integrity-checked
against the freeze manifest.
"""

import hashlib
import json
import os

import datasets


# override only for smoke tests on tiny frozen datasets; the standard panel
# must match the metric_list keys in the *_nll yamls
PANEL_LENGTHS = [
    int(x)
    for x in os.environ.get("RULER_NLL_PANEL", "32768,65536,131072").split(",")
]

_WS = (" ", "\n", "\t")


def _load_frozen(name: str) -> dict[str, datasets.Dataset]:
    root = os.environ.get("RULER_NLL_DATA")
    assert root, (
        f"{name}: RULER_NLL_DATA must point to the frozen dataset dir "
        "produced by scripts/freeze_ruler_nll.py (docs are non-deterministic "
        "to regenerate; refusing to build them ad hoc)."
    )
    manifest = json.load(open(os.path.join(root, "MANIFEST.json")))
    fname = f"{name}.jsonl"
    blob = open(os.path.join(root, fname), "rb").read()
    digest = hashlib.md5(blob).hexdigest()
    want = manifest["files"][fname]["md5"]
    assert digest == want, f"{fname}: md5 {digest} != manifest {want}; abort"
    rows = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line]
    return {
        "test": datasets.Dataset.from_list(rows, split=datasets.Split.TEST)
    }


def niah_multikey_3_nll(**kwargs):
    return _load_frozen("niah_multikey_3_nll")


def vt_nll(**kwargs):
    return _load_frozen("ruler_vt_nll")


def qa_hotpot_nll(**kwargs):
    return _load_frozen("ruler_qa_hotpot_nll")


def _lead(doc: dict) -> str:
    # ll continuations are concatenated raw after the gen_prefix-terminated
    # context, so the target supplies its own boundary space
    return "" if doc["gen_prefix"].endswith(_WS) else " "


def target_join(doc: dict) -> str:
    """All outputs form the answer (niah needles, vt variable set)."""
    return _lead(doc) + " ".join(doc["outputs"])


def target_first(doc: dict) -> str:
    """Outputs are alternative aliases (QA); score the canonical first one."""
    return _lead(doc) + doc["outputs"][0]


def process_results_nll(doc: dict, results: list) -> dict[str, float]:
    ll, is_greedy = results[0]
    length = doc["max_length"]
    assert length in PANEL_LENGTHS, (
        f"doc max_length {length} not in panel {PANEL_LENGTHS}; frozen dataset "
        "and task metric_list disagree"
    )
    metrics: dict[str, float] = {}
    for panel_len in PANEL_LENGTHS:
        metrics[f"nll_{panel_len}"] = -1.0
        metrics[f"greedy_{panel_len}"] = -1.0
    metrics[f"nll_{length}"] = -float(ll)
    metrics[f"greedy_{length}"] = float(bool(is_greedy))
    return metrics
