"""Aggregate the 9 metrics JSON files into a blog-ready Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BENCHMARKS = ("argilla", "ai4privacy", "nemotron")
BENCHMARK_DISPLAY = {
    "argilla": "Argilla PII",
    "ai4privacy": "AI4Privacy",
    "nemotron": "Nemotron-PII",
}
NVIDIA_REPORTED = {
    "argilla": 0.70,
    "ai4privacy": 0.64,
    "nemotron": 0.87,
}
MODES = ("untyped_full", "untyped_opfscope", "typed_opfscope")


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _metric(payload: dict, key: str):
    return payload.get("metrics", {}).get(key)


def per_source_label_recall(payload: dict) -> list[tuple[str, float, int, int]]:
    """Extract per-source-label recall rows from an untyped_full payload."""
    m = payload.get("metrics", {})
    recalled_prefix = "ground_truth_label_recall.recalled_chars."
    total_prefix = "ground_truth_label_recall.ground_truth_chars."
    recall_prefix = "ground_truth_label_recall.recall."
    rows: dict[str, dict[str, float]] = {}
    for key, val in m.items():
        if key.startswith(recalled_prefix):
            rows.setdefault(key[len(recalled_prefix) :], {})["recalled"] = float(val)
        elif key.startswith(total_prefix):
            rows.setdefault(key[len(total_prefix) :], {})["total"] = float(val)
        elif key.startswith(recall_prefix):
            rows.setdefault(key[len(recall_prefix) :], {})["recall"] = float(val)
    out = []
    for label, d in rows.items():
        out.append(
            (
                label,
                d.get("recall", 0.0),
                int(d.get("recalled", 0)),
                int(d.get("total", 0)),
            )
        )
    out.sort(key=lambda row: (-row[3], row[0]))
    return out


def per_class_typed(payload: dict) -> list[tuple[str, float, float, float]]:
    """Extract per-OPF-category P/R/F1 from a typed_opfscope payload."""
    m = payload.get("metrics", {})
    classes: dict[str, dict[str, float]] = {}
    prefix = "by_class."
    for key, val in m.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        if not rest.startswith(("private_", "account_", "secret")):
            continue
        parts = rest.split(".")
        if len(parts) == 3 and parts[1] == "span" and parts[2] in ("precision", "recall", "f1"):
            classes.setdefault(parts[0], {})[parts[2]] = float(val)
    out = []
    for label, d in sorted(classes.items()):
        out.append(
            (
                label,
                d.get("precision", 0.0),
                d.get("recall", 0.0),
                d.get("f1", 0.0),
            )
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("results/REPORT.md"))
    args = p.parse_args()

    payloads: dict[str, dict[str, dict]] = {}
    for bench in BENCHMARKS:
        payloads[bench] = {}
        for mode in MODES:
            f = args.results_dir / bench / f"{mode}_metrics.json"
            payloads[bench][mode] = _load(f) if f.exists() else {}

    sample_sizes = []
    for bench in BENCHMARKS:
        n = payloads[bench]["untyped_full"].get("summary", {}).get("examples")
        if n is not None:
            sample_sizes.append(int(n))
    if sample_sizes and len(set(sample_sizes)) == 1:
        size_desc = f"a {sample_sizes[0]}-example sample of each benchmark"
    elif sample_sizes:
        size_desc = (
            "samples of size "
            + ", ".join(
                f"{BENCHMARK_DISPLAY[b]}={payloads[b]['untyped_full']['summary']['examples']}"
                for b in BENCHMARKS
                if payloads[b]["untyped_full"].get("summary", {}).get("examples") is not None
            )
        )
    else:
        size_desc = "unknown samples"

    lines: list[str] = []
    lines.append("# OpenAI Privacy Filter vs NVIDIA GLiNER-PII — Benchmark Results\n")
    lines.append(
        f"All evals run on {size_desc} using OPF's built-in `opf eval` (Apache 2.0, "
        f"default checkpoint). Untyped mode ignores category identity "
        f"(label-agnostic span detection). Typed mode scores the OPF-mapped "
        f"category as well.\n"
    )

    # ----- Headline table -----
    lines.append("## Headline F1 (strict span match)\n")
    lines.append(
        "| Benchmark | NVIDIA GLiNER-PII (reported) | OPF — untyped × full | OPF — untyped × OPF-scope | OPF — typed × OPF-scope |"
    )
    lines.append("|---|---|---|---|---|")
    for bench in BENCHMARKS:
        nv = NVIDIA_REPORTED[bench]
        uf = _metric(payloads[bench]["untyped_full"], "detection.span.f1")
        uo = _metric(payloads[bench]["untyped_opfscope"], "detection.span.f1")
        to = _metric(payloads[bench]["typed_opfscope"], "detection.span.f1")
        lines.append(
            f"| {BENCHMARK_DISPLAY[bench]} | {nv:.2f} | {_fmt(uf)} | {_fmt(uo)} | {_fmt(to)} |"
        )
    lines.append("")
    lines.append(
        "*untyped × full* = penalty view (every gold span counts, including categories "
        "OPF wasn't trained on). *untyped × OPF-scope* = label-agnostic F1 restricted to "
        "OPF-supported categories. *typed × OPF-scope* = strictest fair view, requires "
        "OPF predicts the right category too. The typed × OPF-scope number is the "
        "closest analogue to NVIDIA's published strict F1.\n"
    )

    # ----- Token-level F1 as supplementary view -----
    lines.append("## Token-level F1 (lenient)\n")
    lines.append(
        "Token-level F1 gives partial credit for any token overlap, even if span "
        "boundaries don't match exactly. Useful for sanity-checking the detector "
        "even when the boundary policy disagrees with the benchmark.\n"
    )
    lines.append(
        "| Benchmark | untyped × full | untyped × OPF-scope | typed × OPF-scope |"
    )
    lines.append("|---|---|---|---|")
    for bench in BENCHMARKS:
        uf = _metric(payloads[bench]["untyped_full"], "detection.f1")
        uo = _metric(payloads[bench]["untyped_opfscope"], "detection.f1")
        to = _metric(payloads[bench]["typed_opfscope"], "detection.f1")
        lines.append(
            f"| {BENCHMARK_DISPLAY[bench]} | {_fmt(uf)} | {_fmt(uo)} | {_fmt(to)} |"
        )
    lines.append("")

    # ----- Per-source-label recall (the gem) -----
    lines.append("## Per-source-label recall (untyped × full)\n")
    lines.append(
        "Shows what fraction of each gold label's character span is covered by *any* "
        "OPF prediction. Labels where OPF wasn't trained will naturally show low "
        "recall — this is the breakdown that explains the penalty-view F1.\n"
    )
    for bench in BENCHMARKS:
        rows = per_source_label_recall(payloads[bench]["untyped_full"])
        if not rows:
            continue
        lines.append(f"### {BENCHMARK_DISPLAY[bench]}\n")
        lines.append("| Source label | Recall | Recalled chars | Gold chars |")
        lines.append("|---|---|---|---|")
        for label, recall, recalled, total in rows:
            lines.append(f"| `{label}` | {recall:.3f} | {recalled} | {total} |")
        lines.append("")

    # ----- Per OPF category breakdown -----
    lines.append("## Per-OPF-category P/R/F1 (typed × OPF-scope)\n")
    for bench in BENCHMARKS:
        rows = per_class_typed(payloads[bench]["typed_opfscope"])
        if not rows:
            continue
        lines.append(f"### {BENCHMARK_DISPLAY[bench]}\n")
        lines.append("| OPF category | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|")
        for label, prec, rec, f1 in rows:
            lines.append(f"| `{label}` | {prec:.3f} | {rec:.3f} | {f1:.3f} |")
        lines.append("")

    # ----- Sample sizes -----
    lines.append("## Sample sizes\n")
    lines.append("| Benchmark | Examples (full) | Examples (OPF-scope) | Tokens (full) |")
    lines.append("|---|---|---|---|")
    for bench in BENCHMARKS:
        s_full = payloads[bench]["untyped_full"].get("summary", {})
        s_scope = payloads[bench]["untyped_opfscope"].get("summary", {})
        ef = s_full.get("examples", "—")
        eo = s_scope.get("examples", "—")
        tk = s_full.get("tokens", "—")
        lines.append(
            f"| {BENCHMARK_DISPLAY[bench]} | {ef} | {eo} | {tk} |"
        )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print("\n--- Preview ---\n")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
