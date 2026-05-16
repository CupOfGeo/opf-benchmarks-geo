"""Adapter for argilla/textcat-tokencat-pii-per-domain.

NOTE: this dataset's annotations are mDeBERTa model suggestions, not
human-validated gold. We use it because it's the dataset NVIDIA's GLiNER-PII
model card points at for the "Argilla PII" benchmark (Strict F1 = 0.70).
"""

from __future__ import annotations

from datasets import load_dataset


HF_PATH = "argilla/textcat-tokencat-pii-per-domain"
SPLIT = "train"
# Pin the HF dataset commit so results are reproducible. Set to None to use
# whatever HEAD is at run time (NOT recommended for the published numbers).
HF_REVISION: str | None = None


def iter_examples(*, max_examples: int | None = None):
    ds = load_dataset(HF_PATH, split=SPLIT, streaming=True, revision=HF_REVISION)
    yielded = 0
    for i, row in enumerate(ds):
        if max_examples is not None and yielded >= max_examples:
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
        yielded += 1
