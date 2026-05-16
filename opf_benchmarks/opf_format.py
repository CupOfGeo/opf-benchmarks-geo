"""Pure transform: source-benchmark example -> OPF eval JSONL records.

OPF JSONL schema (one record per line):

    {
      "text": "...",
      "spans": {"<label>: <span_text>": [[start, end]], ...},
      "info": {"id": "...", "source": "..."}
    }

Each source example produces a "full" record (every valid gold span, with
unmapped labels kept as-is) and an "opfscope" record (only spans whose label
maps to an OPF category, relabeled to the OPF taxonomy). The opfscope record
is None when no spans survive the label-map filter.
"""

from __future__ import annotations

from typing import Iterable, Literal

from .label_map import is_known_label, map_label


SpanCategory = Literal["in", "out_known", "out_unknown"]


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


def example_to_opf_records(
    *,
    benchmark: str,
    text: str,
    ex_id: str,
    gold_spans: list[dict],
) -> tuple[dict, dict | None, list[tuple[str, SpanCategory]]]:
    """Convert one source example to OPF full + opfscope records.

    Returns:
        full_record: OPF JSONL record covering every valid gold span.
        scope_record: same restricted to label-mapped spans, or None if empty.
        categories: per-valid-span (source_label, "in" | "out_known" | "out_unknown")
            for the caller to accumulate stats from.
    """
    full_entries: list[tuple[str, int, int, str]] = []
    scope_entries: list[tuple[str, int, int, str]] = []
    categories: list[tuple[str, SpanCategory]] = []

    for s in gold_spans:
        label = str(s["label"])
        start = int(s["start"])
        end = int(s["end"])
        if start < 0 or end > len(text) or start >= end:
            continue
        span_text = text[start:end]

        opf_cat = map_label(benchmark, label)
        if opf_cat is None:
            cat: SpanCategory = (
                "out_known" if is_known_label(benchmark, label) else "out_unknown"
            )
            categories.append((label, cat))
            full_entries.append((label, start, end, span_text))
        else:
            categories.append((label, "in"))
            full_entries.append((opf_cat, start, end, span_text))
            scope_entries.append((opf_cat, start, end, span_text))

    full_record = {
        "text": text,
        "spans": _coalesce_spans(full_entries),
        "info": {"id": ex_id, "source": benchmark},
    }
    scope_record: dict | None = None
    if scope_entries:
        scope_record = {
            "text": text,
            "spans": _coalesce_spans(scope_entries),
            "info": {"id": ex_id, "source": benchmark},
        }
    return full_record, scope_record, categories
