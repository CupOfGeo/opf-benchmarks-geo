"""Adapter for nvidia/Nemotron-PII (test split).

Note: NVIDIA's GLiNER-PII model was trained on this dataset's train split, so
its 0.87 strict F1 is partially in-distribution. OPF's score on this benchmark
is fully out-of-distribution and the comparison should be framed accordingly.
"""

from __future__ import annotations

import argparse
import json
from ast import literal_eval as parse_python_literal
from pathlib import Path

from datasets import load_dataset

from . import write_opf_jsonl


HF_PATH = "nvidia/Nemotron-PII"
SPLIT = "test"
# Pin the HF dataset commit so results are reproducible. Set to None to use
# whatever HEAD is at run time (NOT recommended for the published numbers).
HF_REVISION: str | None = None


def iter_examples(*, max_examples: int | None = None):
    ds = load_dataset(HF_PATH, split=SPLIT, streaming=True, revision=HF_REVISION)
    yielded = 0
    for i, row in enumerate(ds):
        if max_examples is not None and yielded >= max_examples:
            return
        text = row.get("text") or ""
        raw_spans = row.get("spans") or []
        if isinstance(raw_spans, str):
            # Nemotron stores spans as a Python-literal string, not JSON.
            try:
                raw_spans = parse_python_literal(raw_spans)
            except (ValueError, SyntaxError):
                try:
                    raw_spans = json.loads(raw_spans)
                except json.JSONDecodeError:
                    raw_spans = []
        spans = [
            {"label": s["label"], "start": int(s["start"]), "end": int(s["end"])}
            for s in raw_spans
            if isinstance(s, dict)
            and s.get("label")
            and s.get("start") is not None
            and s.get("end") is not None
        ]
        yield {
            "id": row.get("uid") or f"nemotron-{i}",
            "text": text,
            "spans": spans,
        }
        yielded += 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data", type=Path)
    p.add_argument("--max-examples", type=int, default=None)
    args = p.parse_args()

    stats = write_opf_jsonl(
        benchmark="nemotron",
        out_dir=args.out_dir,
        examples=iter_examples(max_examples=args.max_examples),
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
