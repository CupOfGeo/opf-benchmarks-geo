"""Adapter for gretelai/gretel-pii-masking-en-v1 (test split).

NVIDIA's GLiNER-PII model card lists ``Gretel PII Dataset V1/V2`` as one of
their three evaluation benchmarks. V2 does not exist on the gretelai org as of
2026-05; only V1 is publicly available. We use V1's test split (~5k examples).

V1's ``entities`` field is a Python-literal string of ``{entity, types}`` pairs
with no character offsets, so we reconstruct offsets by locating each entity
value in the text. Entities that appear multiple times become multiple gold
spans. Entities that cannot be located are skipped (rare; usually whitespace
or punctuation differences).
"""

from __future__ import annotations

import json
from ast import literal_eval as parse_python_literal

from datasets import load_dataset


HF_PATH = "gretelai/gretel-pii-masking-en-v1"
SPLIT = "test"
# Pin the HF dataset commit so results are reproducible. Set to None to use
# whatever HEAD is at run time (NOT recommended for the published numbers).
HF_REVISION: str | None = None


def _find_all_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    """Return [(start, end), ...] for every occurrence of value in text."""
    if not value:
        return []
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(value, start)
        if idx < 0:
            return out
        out.append((idx, idx + len(value)))
        start = idx + 1  # allow overlapping matches; harmless for PII


def iter_examples(*, max_examples: int | None = None):
    ds = load_dataset(HF_PATH, split=SPLIT, streaming=True, revision=HF_REVISION)
    yielded = 0
    for i, row in enumerate(ds):
        if max_examples is not None and yielded >= max_examples:
            return
        text = row.get("text") or ""
        raw_entities = row.get("entities") or []
        if isinstance(raw_entities, str):
            try:
                raw_entities = parse_python_literal(raw_entities)
            except (ValueError, SyntaxError):
                try:
                    raw_entities = json.loads(raw_entities)
                except json.JSONDecodeError:
                    raw_entities = []
        spans = []
        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue
            value = ent.get("entity")
            types = ent.get("types") or []
            if not value or not types:
                continue
            label = types[0]  # first type is the primary category
            for start, end in _find_all_occurrences(text, value):
                spans.append({"label": label, "start": start, "end": end})
        yield {
            "id": row.get("uid") or f"gretel-{i}",
            "text": text,
            "spans": spans,
        }
        yielded += 1
