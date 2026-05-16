"""Adapters that convert benchmark datasets into OPF eval JSONL format.

Each adapter writes two JSONL files per benchmark:

* ``<name>_full.jsonl`` — every gold span, original benchmark labels preserved.
  Use with ``opf eval --eval-mode untyped`` for the penalty view +
  per-source-label recall breakdown.

* ``<name>_opfscope.jsonl`` — only gold spans whose label maps to an OPF
  category; labels rewritten to OPF taxonomy. Use with both ``--eval-mode
  untyped`` (fair span-detection F1) and ``--eval-mode typed`` (fair
  categorical F1).

OPF JSONL schema (one record per line):

    {
      "text": "...",
      "spans": {"<label>: <span_text>": [[start, end]], ...},
      "info": {"id": "...", "source": "..."}
    }
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from ..label_map import is_known_label, map_label


def _coalesce_spans(
    spans: Iterable[tuple[str, int, int, str]],
) -> dict[str, list[list[int]]]:
    """Group (label, start, end, span_text) into OPF's 'label: text' -> [[s,e]] form.

    Duplicate (label, start, end) entries are deduplicated.
    """
    seen: set[tuple[str, int, int]] = set()
    grouped: dict[str, list[list[int]]] = {}
    for label, start, end, span_text in spans:
        if (label, start, end) in seen:
            continue
        seen.add((label, start, end))
        key = f"{label}: {span_text}"
        grouped.setdefault(key, []).append([start, end])
    return grouped


def write_opf_jsonl(
    *,
    benchmark: str,
    out_dir: Path,
    examples: Iterable[Mapping[str, object]],
) -> dict[str, int]:
    """Write ``<benchmark>_full.jsonl`` and ``<benchmark>_opfscope.jsonl``.

    ``examples`` yields dicts with keys: ``id`` (str), ``text`` (str),
    ``spans`` (list of ``{label, start, end}``).

    Returns counts dict: examples_full, examples_opfscope, spans_in_scope,
    spans_out_of_scope, plus per-label counts under ``label_counts``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"{benchmark}_full.jsonl"
    opfscope_path = out_dir / f"{benchmark}_opfscope.jsonl"

    n_full = 0
    n_opfscope = 0
    n_spans_in = 0
    n_spans_out = 0
    label_counts: Counter[str] = Counter()
    unknown_labels: Counter[str] = Counter()

    with full_path.open("w", encoding="utf-8") as f_full, opfscope_path.open(
        "w", encoding="utf-8"
    ) as f_scope:
        for ex in examples:
            text = ex["text"]
            assert isinstance(text, str)
            ex_id = str(ex["id"])
            gold_spans = ex["spans"]
            assert isinstance(gold_spans, list)

            full_entries: list[tuple[str, int, int, str]] = []
            scope_entries: list[tuple[str, int, int, str]] = []
            for s in gold_spans:
                label = str(s["label"])
                start = int(s["start"])
                end = int(s["end"])
                if start < 0 or end > len(text) or start >= end:
                    continue
                span_text = text[start:end]
                label_counts[label] += 1

                opf_cat = map_label(benchmark, label)
                if opf_cat is None:
                    n_spans_out += 1
                    if not is_known_label(benchmark, label):
                        unknown_labels[label] += 1
                    full_entries.append((label, start, end, span_text))
                else:
                    n_spans_in += 1
                    full_entries.append((opf_cat, start, end, span_text))
                    scope_entries.append((opf_cat, start, end, span_text))

            full_record = {
                "text": text,
                "spans": _coalesce_spans(full_entries),
                "info": {"id": ex_id, "source": benchmark},
            }
            f_full.write(json.dumps(full_record, ensure_ascii=False) + "\n")
            n_full += 1

            if scope_entries:
                scope_record = {
                    "text": text,
                    "spans": _coalesce_spans(scope_entries),
                    "info": {"id": ex_id, "source": benchmark},
                }
                f_scope.write(json.dumps(scope_record, ensure_ascii=False) + "\n")
                n_opfscope += 1

    if unknown_labels:
        print(
            f"[{benchmark}] WARNING: {len(unknown_labels)} label(s) not in label_map; "
            f"treated as out-of-scope: {dict(unknown_labels.most_common(20))}",
            file=sys.stderr,
        )

    return {
        "examples_full": n_full,
        "examples_opfscope": n_opfscope,
        "spans_in_scope": n_spans_in,
        "spans_out_of_scope": n_spans_out,
        "label_counts": dict(label_counts),
        "unknown_labels": dict(unknown_labels),
        "full_path": str(full_path),
        "opfscope_path": str(opfscope_path),
    }
