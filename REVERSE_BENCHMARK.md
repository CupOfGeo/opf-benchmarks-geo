# Running NVIDIA GLiNER-PII on OpenAI's benchmark suite

The mirror of this repo's existing experiment. We've evaluated **OPF** on NVIDIA's three benchmarks; this doc sketches what it would take to evaluate **GLiNER-PII** on OpenAI's three benchmarks (the ones reported in the OPF Privacy Filter Model Card, April 2026, §7.2).

## OpenAI's evaluation suite

Per OPF model card §7.2:

| Dataset | What it is | Where | Already in this repo? |
|---|---|---|---|
| **PII-Masking-300k** | AI4Privacy's `pii-masking-300k` multilingual PII | `ai4privacy/pii-masking-300k` (HF, gated) | ✅ yes |
| **CredData** | Credential detection in source code (Samsung) | `Samsung/CredData` (GitHub) | ❌ no |
| **SPY** | Synthetic medical-consultation + legal-question PII | Not directly cited in PDF; likely [`spy-dataset` on HF](https://huggingface.co/datasets) — needs lookup | ❌ no |

The headline GLiNER-PII numbers we'd want to report match OPF's "All labels" rows (token + span F1, baseline + corrected) from PDF Table 1.

## What needs building

### 1. Loader for CredData

The Samsung repo ships labeled CSV/JSONL with three classes per credential candidate: `T` (true positive), `F` (false positive), `X` (unknown). OPF maps `T → secret` and drops F/X (§7.2.2). We'd replicate that. Adapter goes alongside `opf_benchmarks/adapters/{argilla,ai4privacy,nemotron}.py`. Per-file token/char offsets in CredData are line-oriented; need to verify our scoring tooling handles that or convert to character offsets.

### 2. Loader for SPY

Source isn't linked in the PDF page I have. First step is to confirm what dataset OPF actually used (search HF and the OPF code repo for "spy" or check `opf/_eval/` for any builtin reference). If SPY isn't publicly hosted, this benchmark may be reproducible only by OpenAI; drop it and report on the other two.

### 3. GLiNER-PII inference wrapper

GLiNER is open-vocabulary: at inference you pass the list of labels you want detected, and it returns spans with confidence scores. Per the build.nvidia.com card, GLiNER-PII was evaluated at `threshold=0.3`. Stub:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("nvidia/gliner-PII")

def predict(text: str, labels: list[str], threshold: float = 0.3):
    return model.predict_entities(text, labels, threshold=threshold)
```

We'd construct the `labels` list from the union of GLiNER-PII's 55+ official categories (listed on the model card) so we're using the full advertised capability, then map predicted labels to each benchmark's gold categories (mirroring what `opf_benchmarks/label_map.py` does in reverse — gold labels stay as-is, GLiNER outputs get collapsed to gold categories).

### 4. Scoring

We can reuse the existing scoring logic in `opf_benchmarks/` if we coerce GLiNER's output into the same `{text, start, end, label}` span shape. Three modes (untyped × full, untyped × scope, typed × scope) make sense for GLiNER too — GLiNER has its own 55-category scope; some OPF benchmark labels (e.g. CredData's `secret`) collapse onto GLiNER's `password`/`api_key` cleanly, others (`secret` for arbitrary high-entropy tokens) may not have a direct GLiNER equivalent.

### 5. Hardware

GLiNER is small (570M params, transformer encoder). Should run on a single 3090 or even CPU with patience — easier than OPF.

## Things to flag in the writeup

- **GLiNER on PII-Masking-300k is the head-to-head** that closes the loop on the existing experiment. OpenAI reports OPF span F1 ≈ 0.93 on it; NVIDIA's card doesn't break out per-benchmark numbers in the same units, so this would be the first apples-to-apples comparison in either direction.
- **CredData is OPF's home turf** in the same way Nemotron-PII is NVIDIA's — OPF's training data was tuned for software credentials. Expect GLiNER to underperform OPF here. Symmetry with Nemotron makes the framing easy: "each model has one benchmark on its home turf."
- **SPY may be out of reach** if it's not publicly hosted. Acknowledge and move on rather than chasing.

## Order of operations (when we get to this)

1. Confirm SPY source — if unhosted, drop it.
2. Build CredData loader → smoke test on 100 examples.
3. Build GLiNER wrapper using `predict_entities` + threshold=0.3 → smoke test on existing AI4Privacy sample.
4. Run all three (or two) benchmarks × three scoring modes.
5. Write a sibling notebook (`run_gliner_benchmarks.ipynb`) and a sibling REPORT.md so the two experiments sit side-by-side.

## Decision needed

Whether to commit to this before or after the OPF re-run lands. Two options:

- **Now**, in parallel: scope is small (one new dataset adapter + one inference wrapper); could ship together as a single "both directions" content piece.
- **Later**, as a follow-up post: cleaner narrative, lets the OPF result breathe first.
