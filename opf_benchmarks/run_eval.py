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
import re
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


BENCHMARKS = ("argilla", "ai4privacy", "nemotron", "gretel")

# (mode_name, dataset_suffix, mode_args)
MODES = (
    ("untyped_full",     "full",     ["--eval-mode", "untyped"]),
    ("untyped_opfscope", "opfscope", ["--eval-mode", "untyped"]),
    ("typed_opfscope",   "opfscope", ["--eval-mode", "typed", "--per-class"]),
)

PROGRESS_RE = re.compile(r"progress: examples=(\d+)(?:/(\d+))?")


def run_opf_eval(cmd: list[str], total: int, desc: str) -> int:
    """Run `opf eval`, converting its `progress: examples=N/M` stderr into a tqdm bar."""
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, bufsize=1)
    bar = tqdm(total=total, desc=desc, unit="ex")
    last = 0
    assert proc.stderr is not None
    for line in proc.stderr:
        m = PROGRESS_RE.search(line)
        if not m:
            tqdm.write(line.rstrip(), file=sys.stderr)
            continue
        n = int(m.group(1))
        if m.group(2):  # OPF tells us the real total when --max-examples is set
            bar.total = int(m.group(2))
            bar.refresh()
        bar.update(n - last)
        last = n
    bar.close()
    return proc.wait()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    p.add_argument("--force", action="store_true",
                   help="Re-run modes whose metrics JSON already exists.")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra args appended to every `opf eval` call (e.g. --window-batch-size 8)")
    args = p.parse_args()

    extra = list(args.extra)
    if not any(t.startswith("--progress-every") for t in extra):
        extra += ["--progress-every", "10"]

    plans = [
        (
            bench, mode,
            args.data_dir / f"{bench}_{suffix}.jsonl",
            mode_args,
            args.results_dir / bench / f"{mode}_metrics.json",
        )
        for bench in args.benchmarks
        for mode, suffix, mode_args in MODES
    ]

    for i, (bench, mode, input_path, mode_args, metrics_out) in enumerate(plans, 1):
        tag = f"[{i}/{len(plans)}] {bench}/{mode}"

        if metrics_out.exists() and not args.force:
            print(f"\n=== {tag} === SKIP ({metrics_out})", flush=True)
            continue
        if not input_path.exists():
            sys.exit(f"missing {input_path}; run the dataset adapter first")
        metrics_out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "opf", "eval", str(input_path),
            "--device", args.device,
            "--metrics-out", str(metrics_out),
            *mode_args, *extra,
        ]
        print(f"\n=== {tag} ===\n{' '.join(cmd)}", flush=True)

        total = sum(1 for _ in input_path.open("rb"))
        rc = run_opf_eval(cmd, total=total, desc=tag)
        if rc != 0:
            sys.exit(f"{tag} FAILED (exit {rc})")

        m = json.loads(metrics_out.read_text())
        f1 = m.get("metrics", {}).get("detection.span.f1")
        examples = m.get("summary", {}).get("examples")
        print(f"  examples={examples} F1(strict span)={f1}", flush=True)


if __name__ == "__main__":
    main()
