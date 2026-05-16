"""Pre-fetch the three HF datasets and convert them to OPF eval JSONL.

The actual dataset files are cached by the `datasets` library under
~/.cache/huggingface/. The JSONL artifacts this script produces under
``data/`` are what `opf_benchmarks.run_eval` actually consumes.

After running, you'll have:
  data/argilla_full.jsonl       data/argilla_opfscope.jsonl
  data/ai4privacy_full.jsonl    data/ai4privacy_opfscope.jsonl
  data/nemotron_full.jsonl      data/nemotron_opfscope.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opf_benchmarks.adapters import argilla, ai4privacy, nemotron
from opf_benchmarks.adapters import write_opf_jsonl


ADAPTERS = {
    "argilla": argilla.iter_examples,
    "ai4privacy": ai4privacy.iter_examples,
    "nemotron": nemotron.iter_examples,
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
        iter_fn = ADAPTERS[bench]
        stats = write_opf_jsonl(
            benchmark=bench,
            out_dir=args.out_dir,
            examples=iter_fn(max_examples=args.max_examples),
        )
        print(
            f"  examples_full={stats['examples_full']} "
            f"examples_opfscope={stats['examples_opfscope']} "
            f"spans_in={stats['spans_in_scope']} spans_out={stats['spans_out_of_scope']}"
        )
        if stats["unknown_labels"]:
            print(f"  unknown labels (added to OOS): {stats['unknown_labels']}")


if __name__ == "__main__":
    main()
