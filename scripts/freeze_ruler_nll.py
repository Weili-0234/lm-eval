#!/usr/bin/env python3
"""Freeze the Panel A (RULER gold-answer NLL) documents ONCE.

niah_multikey_3 draws uuid4 needles and vt draws unseeded random chains, so
regenerating per lm-eval invocation yields different documents per row — fatal
for paired per-row NLL deltas. This script runs the stock RULER generators one
time, subsets to the first N docs per length, and writes jsonl + MANIFEST.json
(md5 per file). The *_nll tasks refuse to run without this artifact
(RULER_NLL_DATA env var) and hard-verify the md5s.

Usage:
  python scripts/freeze_ruler_nll.py --tokenizer /path/to/Qwen3.5-27B \
      --out /path/to/ruler-nll-data [--lengths 32768 65536 131072] [--samples 100]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[32768, 65536, 131072])
    ap.add_argument("--samples", type=int, default=100,
                    help="docs kept per task per length (first N of the "
                         "generator's 500)")
    args = ap.parse_args()

    from lm_eval.tasks.ruler import niah_utils, qa_utils, vt_utils

    if os.path.isdir(args.out) and os.listdir(args.out):
        sys.exit(f"refusing to overwrite nonempty {args.out}")
    os.makedirs(args.out, exist_ok=True)

    builder_kwargs = {"tokenizer": args.tokenizer,
                      "max_seq_lengths": args.lengths}
    builders = {
        "niah_multikey_3_nll": niah_utils.niah_multikey_3,
        "ruler_vt_nll": vt_utils.get_vt_dataset,
        "ruler_qa_hotpot_nll": qa_utils.get_hotpotqa,
    }

    harness_sha = subprocess.run(
        ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
         "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    manifest = {
        "tokenizer": args.tokenizer,
        "lengths": args.lengths,
        "samples_per_length": args.samples,
        "harness_sha": harness_sha,
        "files": {},
    }

    for name, builder in builders.items():
        ds = builder(**dict(builder_kwargs))["test"]
        keep, seen = [], {}
        for i, length in enumerate(ds["max_length"]):
            if seen.get(length, 0) < args.samples:
                keep.append(i)
                seen[length] = seen.get(length, 0) + 1
        subset = ds.select(keep)
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        blob = open(path, "rb").read()
        manifest["files"][f"{name}.jsonl"] = {
            "rows": len(subset),
            "per_length": seen,
            "bytes": len(blob),
            "md5": hashlib.md5(blob).hexdigest(),
        }
        print(f"FROZEN {name}: {len(subset)} rows {seen}", flush=True)

    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("FREEZE_RULER_NLL_DONE", flush=True)


if __name__ == "__main__":
    main()
