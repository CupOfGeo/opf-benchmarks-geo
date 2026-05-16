"""Investigate what OPF actually predicts for out-of-scope gold spans.

For every gold span in ``data/<benchmark>_full.jsonl`` whose source label is
*not* one of OPF's 8 categories, find any overlapping OPF prediction and tally
which OPF category caught it. Outputs a per-source-label confusion table.

This is what explains "OPF covers more than its taxonomy admits": e.g. Argilla
USERNAME gets 91% character recall in the untyped eval, but the label_map
treats USERNAME as out-of-scope. This script answers "what is OPF actually
tagging USERNAME spans as?" (likely private_person or account_number).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from opf import OPF

from opf_benchmarks.label_map import OPF_CATEGORIES


def iter_examples(path: Path) -> Iterator[tuple[str, list[tuple[str, int, int]]]]:
    """Yield (text, [(label, start, end), ...]) for each example in a _full JSONL."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"]
            gold: list[tuple[str, int, int]] = []
            for key, ranges in rec["spans"].items():
                # Key format: "{label}: {span_text}" — split on the first ": ".
                label = key.split(": ", 1)[0]
                for start, end in ranges:
                    gold.append((label, int(start), int(end)))
            yield text, gold


def overlap_chars(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="argilla", choices=["argilla", "ai4privacy", "nemotron"])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--max-examples", type=int, default=200,
                   help="Cap examples to analyze. Inference cost is the dominant factor; "
                        "200 examples is usually enough to see the pattern.")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to write the Markdown report. Always printed to stdout.")
    args = p.parse_args()

    full_path = args.data_dir / f"{args.benchmark}_full.jsonl"
    if not full_path.exists():
        raise SystemExit(
            f"missing {full_path} — run `python -m scripts.download_datasets` first"
        )

    print(f"Loading OPF on {args.device}...", flush=True)
    opf = OPF(device=args.device, output_mode="typed")

    # confusion[source_label][opf_predicted_label or "<none>"] = count
    confusion: dict[str, Counter[str]] = {}

    for i, (text, gold) in enumerate(iter_examples(full_path)):
        if i >= args.max_examples:
            break
        if i % 25 == 0:
            print(f"  example {i}/{args.max_examples}", flush=True)
        preds = list(opf.redact(text).detected_spans)

        for gold_label, gs, ge in gold:
            if gold_label in OPF_CATEGORIES:
                continue  # in-scope — not what we're investigating
            best: tuple[int, str] | None = None  # (overlap_chars, opf_label)
            for pred in preds:
                ov = overlap_chars(gs, ge, pred.start, pred.end)
                if ov > 0 and (best is None or ov > best[0]):
                    best = (ov, pred.label)
            outcome = best[1] if best else "<none>"
            confusion.setdefault(gold_label, Counter())[outcome] += 1

    labels_sorted = sorted(confusion.items(), key=lambda kv: -sum(kv[1].values()))

    lines = [
        f"# Where OPF tags out-of-scope `{args.benchmark}` labels",
        "",
        f"On the first {args.max_examples} examples from `data/{args.benchmark}_full.jsonl`. "
        f"For each gold span whose source label is *not* in OPF's 8 categories, "
        f"this shows which OPF category OPF actually predicted for that span "
        f"(top 3, by share of spans). `<none>` = OPF predicted nothing overlapping that span.",
        "",
        "| Source label | Gold spans | OPF predicted (top 3 by share) |",
        "|---|---|---|",
    ]
    for label, ctr in labels_sorted:
        total = sum(ctr.values())
        top = ctr.most_common(3)
        top_str = ", ".join(f"`{l}` {n / total:.0%}" for l, n in top)
        lines.append(f"| `{label}` | {total} | {top_str} |")

    report = "\n".join(lines)
    print()
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
