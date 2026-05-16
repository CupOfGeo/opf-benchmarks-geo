"""Run ``opf eval`` 3x per benchmark and collect metrics.

For each benchmark we run:
  1. untyped on the full file        -> penalty F1 + per-source-label recall
  2. untyped on the OPF-scope file   -> fair span-detection F1
  3. typed   on the OPF-scope file   -> fair categorical F1 + per-category

All metrics are written under results/<benchmark>/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BENCHMARKS = ("argilla", "ai4privacy", "nemotron")


def run_one(
    *,
    benchmark: str,
    data_dir: Path,
    results_dir: Path,
    device: str,
    extra: list[str],
) -> dict[str, dict]:
    full = data_dir / f"{benchmark}_full.jsonl"
    scope = data_dir / f"{benchmark}_opfscope.jsonl"

    if not full.exists():
        raise FileNotFoundError(f"missing {full}; run the adapter first")
    if not scope.exists():
        raise FileNotFoundError(f"missing {scope}; run the adapter first")

    out_dir = results_dir / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "untyped_full": dict(
            input=full,
            args=["--eval-mode", "untyped"],
            metrics_out=out_dir / "untyped_full_metrics.json",
        ),
        "untyped_opfscope": dict(
            input=scope,
            args=["--eval-mode", "untyped"],
            metrics_out=out_dir / "untyped_opfscope_metrics.json",
        ),
        "typed_opfscope": dict(
            input=scope,
            args=["--eval-mode", "typed", "--per-class"],
            metrics_out=out_dir / "typed_opfscope_metrics.json",
        ),
    }

    results: dict[str, dict] = {}
    for name, cfg in runs.items():
        cmd = [
            "opf",
            "eval",
            str(cfg["input"]),
            "--device",
            device,
            "--metrics-out",
            str(cfg["metrics_out"]),
            *cfg["args"],
            *extra,
        ]
        print(f"\n=== {benchmark}/{name} ===", flush=True)
        print(" ".join(cmd), flush=True)
        # Inherit terminal for both streams so opf eval's tqdm bar is visible live.
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode})", file=sys.stderr)
            results[name] = {"error": f"exit {proc.returncode}"}
            continue

        with cfg["metrics_out"].open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        results[name] = metrics
        m = metrics.get("metrics", {})
        summary = metrics.get("summary", {})
        f1 = m.get("detection.span.f1")
        print(
            f"  examples={summary.get('examples')} F1(strict span)={f1}",
            flush=True,
        )

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARKS),
        choices=BENCHMARKS,
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--device",
        default="cpu",
        help="Pass through to `opf eval --device`. Use cpu, cuda, or mps.",
    )
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args appended to every `opf eval` call (e.g. --window-batch-size 8)",
    )
    args = p.parse_args()

    summary: dict[str, dict] = {}
    for bench in args.benchmarks:
        summary[bench] = run_one(
            benchmark=bench,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            device=args.device,
            extra=args.extra,
        )

    summary_path = args.results_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
