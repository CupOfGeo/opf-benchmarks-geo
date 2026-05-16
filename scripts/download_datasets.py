"""Pre-fetch the three HF datasets and convert them to OPF eval JSONL.

Each benchmark produces two files under ``--out-dir`` (default ``data/``):
  <name>_full.jsonl       — every gold span (unmapped labels kept as-is)
  <name>_opfscope.jsonl   — only spans whose label maps to an OPF category

The HF datasets library caches the raw data under ~/.cache/huggingface/.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping

from opf_benchmarks.adapters import ai4privacy, argilla, nemotron
from opf_benchmarks.opf_format import example_to_opf_records


ADAPTERS: dict[str, Callable[..., Iterable[Mapping[str, object]]]] = {
    "argilla": argilla.iter_examples,
    "ai4privacy": ai4privacy.iter_examples,
    "nemotron": nemotron.iter_examples,
}


def write_benchmark(
    *,
    benchmark: str,
    iter_fn: Callable[..., Iterable[Mapping[str, object]]],
    out_dir: Path,
    max_examples: int | None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"{benchmark}_full.jsonl"
    scope_path = out_dir / f"{benchmark}_opfscope.jsonl"

    n_full = n_scope = n_in = n_out = 0
    unknown_labels: Counter[str] = Counter()

    with full_path.open("w", encoding="utf-8") as f_full, scope_path.open(
        "w", encoding="utf-8"
    ) as f_scope:
        for ex in iter_fn(max_examples=max_examples):
            full_rec, scope_rec, categories = example_to_opf_records(
                benchmark=benchmark,
                text=str(ex["text"]),
                ex_id=str(ex["id"]),
                gold_spans=list(ex["spans"]),  # type: ignore[arg-type]
            )
            f_full.write(json.dumps(full_rec, ensure_ascii=False) + "\n")
            n_full += 1
            if scope_rec is not None:
                f_scope.write(json.dumps(scope_rec, ensure_ascii=False) + "\n")
                n_scope += 1
            for label, cat in categories:
                if cat == "in":
                    n_in += 1
                else:
                    n_out += 1
                    if cat == "out_unknown":
                        unknown_labels[label] += 1

    if unknown_labels:
        print(
            f"[{benchmark}] WARNING: {len(unknown_labels)} label(s) not in label_map; "
            f"treated as out-of-scope: {dict(unknown_labels.most_common(20))}",
            file=sys.stderr,
        )

    return {
        "examples_full": n_full,
        "examples_opfscope": n_scope,
        "spans_in_scope": n_in,
        "spans_out_of_scope": n_out,
        "unknown_labels": dict(unknown_labels),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(ADAPTERS),
        choices=list(ADAPTERS),
    )
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap examples per benchmark. None = full split. "
             "For Argilla's mDeBERTa-suggested train split (~50k rows of "
             "shaky-quality labels) you probably want 5000-10000.",
    )
    args = p.parse_args()

    for bench in args.benchmarks:
        print(f"\n=== {bench} ===")
        stats = write_benchmark(
            benchmark=bench,
            iter_fn=ADAPTERS[bench],
            out_dir=args.out_dir,
            max_examples=args.max_examples,
        )
        print(
            f"  examples_full={stats['examples_full']} "
            f"examples_opfscope={stats['examples_opfscope']} "
            f"spans_in={stats['spans_in_scope']} spans_out={stats['spans_out_of_scope']}"
        )


if __name__ == "__main__":
    main()
