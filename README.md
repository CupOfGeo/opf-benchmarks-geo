# opf-benchmarks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CupOfGeo/opf-benchmarks-geo/blob/main/run_benchmarks.ipynb)

Independent PII benchmark evaluations of [OpenAI Privacy Filter (OPF)](https://github.com/openai/privacy-filter) against the three benchmarks NVIDIA published numbers on in the [GLiNER-PII](https://huggingface.co/nvidia/gliner-PII) model card:

- **Argilla PII** (`argilla/textcat-tokencat-pii-per-domain`) — NVIDIA reports strict F1 = 0.70
- **AI4Privacy** (`ai4privacy/pii-masking-300k`) — NVIDIA reports strict F1 = 0.64
- **Nemotron-PII** (`nvidia/Nemotron-PII` test split) — NVIDIA reports strict F1 = 0.87

OPF and NVIDIA's GLiNER-PII use different label taxonomies, so we evaluate three ways per benchmark:

1. **`untyped × full`** — every gold span counts (incl. categories OPF wasn't trained on). Penalty view.
2. **`untyped × OPF-scope`** — label-agnostic F1, restricted to OPF-supported categories. Fair span detection.
3. **`typed × OPF-scope`** — strictest fair view. Requires OPF to also predict the right category (after a hand-written label map). This is the closest analogue to NVIDIA's published strict F1.

See `opf_benchmarks/label_map.py` for the full mapping from each benchmark's labels to OPF's 8 categories.

## Repro

```bash
git clone https://github.com/<you>/opf-benchmarks
cd opf-benchmarks
uv sync                                                          # installs OPF (pinned), datasets, etc.

uv run python -m scripts.download_datasets --max-examples 5000   # downloads & converts to JSONL
uv run python -m opf_benchmarks.run_eval --device cpu            # 9 evals: 3 benchmarks × 3 modes
uv run python -m opf_benchmarks.aggregate                        # writes results/REPORT.md
```

- OPF is pinned to a specific commit of `openai/privacy-filter` in `pyproject.toml`. Bump deliberately when you want to re-run against a newer release.
- HuggingFace dataset revisions are not yet pinned (`HF_REVISION = None` in each adapter). Set those before publishing if exact reproducibility matters to you.
- `data/` and `results/` are gitignored — regenerate them locally with the commands above.

## Caveats worth noting before publishing numbers

- **Argilla labels are not gold.** The `pii.suggestion` field is mDeBERTa output, not human-validated. The 0.70 NVIDIA reports is against the same noisy labels.
- **Nemotron is NVIDIA's home turf.** GLiNER-PII was trained on this dataset's train split. OPF's score here is fully out-of-distribution.
- **AI4Privacy overlap.** Some AI4Privacy-derived data appears in GLiNER-PII's training mix (per NVIDIA's card). OPF here is out-of-distribution.


vast.ai 
https://cloud.vast.ai/cli/
