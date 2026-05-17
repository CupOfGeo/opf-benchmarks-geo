"""Chart generation for opf-benchmarks results.

Reads metrics JSONs (default: `vast_results/`, the committed full-run numbers)
and produces seaborn/matplotlib charts. Renders inline in the notebook and saves
PNGs to `results/figs/` for blog/LinkedIn use.

CLI:
    python -m opf_benchmarks.charts --save          # writes all 5 PNGs to vast_results/figs/
    python -m opf_benchmarks.charts --results-dir vast_results --save
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from opf_benchmarks.aggregate import (
    BENCHMARK_DISPLAY,
    BENCHMARKS,
    MODES,
    NVIDIA_REPORTED,
    _metric,
    per_class_typed,
    per_source_label_recall,
)
from opf_benchmarks.label_map import MAPS


MODE_DISPLAY = {
    "untyped_full": "OPF untyped × full",
    "untyped_opfscope": "OPF untyped × scope",
    "typed_opfscope": "OPF typed × scope",
}

SERIES_ORDER = [
    "NVIDIA reported",
    "OPF untyped × full",
    "OPF untyped × scope",
    "OPF typed × scope",
]

OPF_CATEGORIES_ORDER = [
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
]

NVIDIA_GREEN = "#76B900"
OPF_PALETTE = ["#9bb7ff", "#5980ff", "#1d4cff"]  # light → dark blue for the 3 OPF modes
IN_SCOPE_COLOR = "#1d4cff"
OUT_OF_SCOPE_COLOR = "#bbbbbb"


def load_metrics(results_dir: str | Path = "vast_results") -> dict[str, dict[str, dict]]:
    """Load all 9 metrics JSONs into the {benchmark: {mode: payload}} shape."""
    results_dir = Path(results_dir)
    out: dict[str, dict[str, dict]] = {}
    for bench in BENCHMARKS:
        out[bench] = {}
        for mode in MODES:
            f = results_dir / bench / f"{mode}_metrics.json"
            out[bench][mode] = json.loads(f.read_text()) if f.exists() else {}
    return out


# ---------------------------------------------------------------------------
# Chart 1 — Headline F1 (grouped bar)
# ---------------------------------------------------------------------------

def _headline_frame(payloads: dict[str, dict[str, dict]]) -> pd.DataFrame:
    """Long-format DataFrame of strict-span F1, one row per (benchmark, series)."""
    rows = []
    for bench in BENCHMARKS:
        nv = NVIDIA_REPORTED.get(bench)
        if nv is not None:
            rows.append({
                "benchmark": BENCHMARK_DISPLAY[bench],
                "series": "NVIDIA reported",
                "f1": nv,
            })
        for mode in MODES:
            f1 = _metric(payloads[bench][mode], "detection.span.f1")
            rows.append({
                "benchmark": BENCHMARK_DISPLAY[bench],
                "series": MODE_DISPLAY[mode],
                "f1": float(f1) if f1 is not None else float("nan"),
            })
    return pd.DataFrame(rows)


def plot_headline_f1(
    payloads: dict[str, dict[str, dict]],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Grouped bar of strict-span F1 across benchmarks × {NVIDIA + 3 OPF modes}."""
    df = _headline_frame(payloads)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    palette = {
        "NVIDIA reported": NVIDIA_GREEN,
        "OPF untyped × full": OPF_PALETTE[0],
        "OPF untyped × scope": OPF_PALETTE[1],
        "OPF typed × scope": OPF_PALETTE[2],
    }

    sns.barplot(
        data=df,
        x="benchmark",
        y="f1",
        hue="series",
        hue_order=SERIES_ORDER,
        palette=palette,
        ax=ax,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Strict-span F1")
    ax.set_xlabel("")
    ax.set_title("OpenAI Privacy Filter vs NVIDIA GLiNER-PII — strict-span F1")
    ax.legend(title="", loc="lower center", bbox_to_anchor=(0.5, -0.22), ncols=4, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 2 — Per-OPF-category F1 heatmap (typed × OPF-scope)
# ---------------------------------------------------------------------------

def plot_per_category_heatmap(
    payloads: dict[str, dict[str, dict]],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Heatmap of typed_opfscope F1 by (OPF category × benchmark)."""
    rows = []
    for bench in BENCHMARKS:
        for label, _p, _r, f1 in per_class_typed(payloads[bench]["typed_opfscope"]):
            rows.append({
                "benchmark": BENCHMARK_DISPLAY[bench],
                "category": label,
                "f1": f1,
            })
    df = pd.DataFrame(rows)
    pivot = (
        df.pivot(index="category", columns="benchmark", values="f1")
          .reindex(OPF_CATEGORIES_ORDER)
          .reindex(columns=[BENCHMARK_DISPLAY[b] for b in BENCHMARKS])
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "F1"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Per-OPF-category F1 (typed × OPF-scope)")
    ax.set_xlabel("")
    ax.set_ylabel("OPF category")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 3 — Per-source-label recall (untyped × full), in-scope vs out-of-scope
# ---------------------------------------------------------------------------

def _is_in_scope(label: str, benchmark: str) -> bool:
    """A label is OPF in-scope if it IS an OPF category, or maps to one.

    `opf_format.example_to_opf_records` renames mapped source labels to their
    OPF target before the data hits the eval, so the per-source-label-recall
    keys can be either (a) original out-of-scope source labels (e.g. `IPV4`,
    `GENDER`) or (b) OPF category names (e.g. `private_email`, `account_number`)
    for spans that were mapped. The MAPS lookup covers (a)-style labels that
    somehow leaked through unmapped; the OPF_CATEGORIES_ORDER check covers (b).
    """
    if label in OPF_CATEGORIES_ORDER:
        return True
    return MAPS.get(benchmark, {}).get(label) is not None


def plot_per_source_label_recall(
    payloads: dict[str, dict[str, dict]],
) -> plt.Figure:
    """One horizontal-bar panel per benchmark, stacked vertically.

    Bars sorted by recall, colored by in-scope vs out-of-scope. The "what OPF
    catches and what it misses" chart.
    """
    # Per-benchmark height proportional to label count, capped so the figure
    # stays sane on small screens.
    rows_per_bench = []
    for bench in BENCHMARKS:
        rows = per_source_label_recall(payloads[bench]["untyped_full"])
        rows_per_bench.append((bench, rows))
    heights = [max(2.0, len(rows) * 0.22) for _, rows in rows_per_bench]
    fig, axes = plt.subplots(
        nrows=len(BENCHMARKS),
        ncols=1,
        figsize=(11, sum(heights) + 1.5),
        gridspec_kw={"height_ratios": heights},
    )

    for ax, (bench, rows) in zip(axes, rows_per_bench):
        if not rows:
            ax.set_title(f"{BENCHMARK_DISPLAY[bench]} (no data)")
            ax.axis("off")
            continue

        df = pd.DataFrame(rows, columns=["label", "recall", "recalled", "total"])
        df = df.sort_values("recall", ascending=True)  # asc → highest at top in barh
        df["color"] = df["label"].map(
            lambda lbl: IN_SCOPE_COLOR if _is_in_scope(lbl, bench) else OUT_OF_SCOPE_COLOR
        )

        ax.barh(df["label"], df["recall"], color=df["color"])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Recall" if ax is axes[-1] else "")
        ax.set_title(BENCHMARK_DISPLAY[bench], loc="left", fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        sns.despine(ax=ax)

    legend_handles = [
        mpatches.Patch(color=IN_SCOPE_COLOR, label="OPF in-scope"),
        mpatches.Patch(color=OUT_OF_SCOPE_COLOR, label="Out of OPF scope"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncols=2,
        bbox_to_anchor=(0.5, 1.00),
        frameon=False,
    )
    fig.suptitle(
        "Per-source-label recall (untyped × full) — what OPF catches and what it misses",
        y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


# ---------------------------------------------------------------------------
# Chart 4 — Token-level vs span-level F1 (slope chart)
# ---------------------------------------------------------------------------

def plot_token_vs_span(
    payloads: dict[str, dict[str, dict]],
) -> plt.Figure:
    """Slope chart per benchmark showing token-level → span-level F1 drop per mode."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    for ax, bench in zip(axes, BENCHMARKS):
        for i, mode in enumerate(MODES):
            payload = payloads[bench][mode]
            token_f1 = _metric(payload, "detection.f1")
            span_f1 = _metric(payload, "detection.span.f1")
            if token_f1 is None or span_f1 is None:
                continue
            color = OPF_PALETTE[i]
            ax.plot([0, 1], [token_f1, span_f1], marker="o", color=color, label=MODE_DISPLAY[mode])
            ax.annotate(
                f"{token_f1:.2f}", (0, token_f1),
                textcoords="offset points", xytext=(-6, 0),
                ha="right", va="center", fontsize=8, color=color,
            )
            ax.annotate(
                f"{span_f1:.2f}", (1, span_f1),
                textcoords="offset points", xytext=(6, 0),
                ha="left", va="center", fontsize=8, color=color,
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Token\n(lenient)", "Span\n(strict)"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(0, 1)
        ax.set_title(BENCHMARK_DISPLAY[bench])
        if ax is axes[0]:
            ax.set_ylabel("F1")
        sns.despine(ax=ax)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncols=3,
        bbox_to_anchor=(0.5, 1.02), frameon=False,
    )
    fig.suptitle("Token-level vs span-level F1 — boundary strictness penalty", y=1.08)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 5 — Precision vs Recall scatter with F1 isolines
# ---------------------------------------------------------------------------

def plot_pr_scatter(
    payloads: dict[str, dict[str, dict]],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """P vs R scatter for all (benchmark, mode); F1 contours overlaid."""
    rows = []
    for bench in BENCHMARKS:
        for mode in MODES:
            payload = payloads[bench][mode]
            p = _metric(payload, "detection.span.precision")
            r = _metric(payload, "detection.span.recall")
            if p is None or r is None:
                continue
            rows.append({
                "benchmark": BENCHMARK_DISPLAY[bench],
                "mode": MODE_DISPLAY[mode],
                "precision": float(p),
                "recall": float(r),
            })
    df = pd.DataFrame(rows)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # F1 isolines: F1 = 2pr/(p+r)  =>  p = F1*r/(2r - F1)
    r = np.linspace(0.001, 1, 200)
    for f1 in [0.5, 0.6, 0.7, 0.8, 0.9]:
        p_curve = (f1 * r) / (2 * r - f1)
        valid = (p_curve >= 0) & (p_curve <= 1.05)
        ax.plot(r[valid], p_curve[valid], "--", color="gray", linewidth=0.5, alpha=0.6)
        # Place label where the curve crosses x=0.97 (right side)
        idx_end = np.argmin(np.abs(r - 0.97))
        if valid[idx_end]:
            ax.text(
                0.99, p_curve[idx_end], f"F1={f1}",
                fontsize=7, color="gray", alpha=0.85, va="center",
            )

    sns.scatterplot(
        data=df,
        x="recall", y="precision",
        hue="benchmark", style="mode",
        s=180, ax=ax,
        hue_order=[BENCHMARK_DISPLAY[b] for b in BENCHMARKS],
        style_order=list(MODE_DISPLAY.values()),
    )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall (strict span)")
    ax.set_ylabel("Precision (strict span)")
    ax.set_title("OPF Precision vs Recall across benchmarks and modes")
    ax.legend(title="", loc="lower left", frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Registry + CLI
# ---------------------------------------------------------------------------

CHARTS = {
    "headline_f1": plot_headline_f1,
    "per_category_heatmap": plot_per_category_heatmap,
    "per_source_label_recall": plot_per_source_label_recall,
    "token_vs_span": plot_token_vs_span,
    "pr_scatter": plot_pr_scatter,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="vast_results",
                   help="Directory containing per-benchmark metrics JSONs (default: vast_results)")
    p.add_argument("--figs-dir", default="vast_results/figs",
                   help="Output directory for saved PNGs (default: vast_results/figs — co-located with the full-run metrics they're derived from)")
    p.add_argument("--save", action="store_true",
                   help="Save all charts as PNGs.")
    p.add_argument("--only", default=None, choices=list(CHARTS),
                   help="Render/save only this chart by name.")
    args = p.parse_args()

    payloads = load_metrics(args.results_dir)
    out_dir = Path(args.figs_dir)
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    charts = {args.only: CHARTS[args.only]} if args.only else CHARTS
    for name, fn in charts.items():
        fig = fn(payloads)
        if args.save:
            out_path = out_dir / f"{name}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Wrote {out_path}")
        plt.close(fig)

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
