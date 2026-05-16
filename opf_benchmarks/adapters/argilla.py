"""Adapter for argilla/textcat-tokencat-pii-per-domain.

NOTE: this dataset's annotations are mDeBERTa model suggestions, not
human-validated gold. We use it because it's the dataset NVIDIA's GLiNER-PII
model card points at for the "Argilla PII" benchmark (Strict F1 = 0.70).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from . import write_opf_jsonl


HF_PATH = "argilla/textcat-tokencat-pii-per-domain"
SPLIT = "train"
# Pin the HF dataset commit so results are reproducible. Set to None to use
# whatever HEAD is at run time (NOT recommended for the published numbers).
HF_REVISION: str | None = None


def iter_examples(*, max_examples: int | None = None):
    ds = load_dataset(HF_PATH, split=SPLIT, revision=HF_REVISION)
    for i, row in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            return
        text = row.get("source-text") or ""
        gold = row.get("pii.suggestion") or []
        spans = [
            {"label": s["label"], "start": int(s["start"]), "end": int(s["end"])}
            for s in gold
            if s and s.get("label") and s.get("start") is not None
        ]
        yield {
            "id": row.get("id") or f"argilla-{i}",
            "text": text,
            "spans": spans,
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data", type=Path)
    p.add_argument("--max-examples", type=int, default=None)
    args = p.parse_args()

    stats = write_opf_jsonl(
        benchmark="argilla",
        out_dir=args.out_dir,
        examples=iter_examples(max_examples=args.max_examples),
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
