"""Adapter for ai4privacy/pii-masking-300k (validation split, English only).

Filters to English to keep parity with OPF's primary training language. The
dataset's ``privacy_mask`` field is the gold span list.
"""

from __future__ import annotations

import json

from datasets import load_dataset


HF_PATH = "ai4privacy/pii-masking-300k"
SPLIT = "validation"
# Pin the HF dataset commit so results are reproducible. Set to None to use
# whatever HEAD is at run time (NOT recommended for the published numbers).
HF_REVISION: str | None = None


def iter_examples(*, max_examples: int | None = None, language: str = "English"):
    ds = load_dataset(HF_PATH, split=SPLIT, streaming=True, revision=HF_REVISION)
    yielded = 0
    for i, row in enumerate(ds):
        if max_examples is not None and yielded >= max_examples:
            return
        if language and row.get("language") != language:
            continue
        text = row.get("source_text") or ""
        gold = row.get("privacy_mask") or []
        if isinstance(gold, str):
            try:
                gold = json.loads(gold)
            except json.JSONDecodeError:
                gold = []
        spans = [
            {"label": s["label"], "start": int(s["start"]), "end": int(s["end"])}
            for s in gold
            if isinstance(s, dict)
            and s.get("label")
            and s.get("start") is not None
            and s.get("end") is not None
        ]
        yield {
            "id": row.get("id") or f"ai4privacy-{i}",
            "text": text,
            "spans": spans,
        }
        yielded += 1
