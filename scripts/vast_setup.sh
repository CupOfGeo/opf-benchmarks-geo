#!/usr/bin/env bash
# Bootstrap a vast.ai box (NGC PyTorch image) for opf-benchmarks runs.
#
# Run this AFTER ssh-ing into the box. HF_TOKEN is expected to be present
# via vast.ai's --env injection (set by vast_provision.sh). If you provisioned
# manually, export it yourself before running.
#
# SECURITY: only use READ-ONLY HF tokens here — vast hosts have root on the
# physical box and can read container env vars. Revoke the token after the run.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/CupOfGeo/opf-benchmarks-geo/main/scripts/vast_setup.sh | bash

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/CupOfGeo/opf-benchmarks-geo.git}"
REPO_DIR="${REPO_DIR:-opf-benchmarks-geo}"

# Pin to /workspace — that's where vast's NGC images start onstart_cmd. Keeps the
# repo path predictable so the laptop-side poller knows exactly where to look,
# regardless of where setup.sh was invoked from.
mkdir -p /workspace && cd /workspace

# vast.ai writes --env values into /etc/environment but non-login shells don't
# always source it. Pull HF_TOKEN from there if it's not already in the env.
if [[ -z "${HF_TOKEN:-}" && -r /etc/environment ]]; then
    HF_TOKEN="$(grep -E '^HF_TOKEN=' /etc/environment | tail -n1 | cut -d= -f2- | tr -d '"')"
    export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set. Provision via vast_provision.sh (which passes --env)," >&2
    echo "or export it manually: export HF_TOKEN=hf_..." >&2
    exit 1
fi

# 1. OS packages
apt-get update -qq
apt-get install -y -qq git tmux build-essential

# 2. Clone repo if not present, then cd in
if [[ ! -d "$REPO_DIR" ]]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# 3. Python deps — pip install -e . so we keep NGC's pre-tested torch+triton.
# Do NOT `uv sync` here — that would reinstall torch from PyPI and clobber
# the NGC-tested combo, re-introducing the dtype-assertion bug.
python -m pip install --quiet -e .

# 4. OPF MoE workaround — their grouped_swiglu_w2 triton kernel mixes fp32/bf16
# in tl.dot, which every modern triton refuses. See OPF issue #20.
# Setting this falls back to a pure-PyTorch SwiGLU+w2 path on the failing
# kernel only; other triton kernels still run.
if ! grep -q OPF_MOE_FUSED_SWIGLU_W2 ~/.bashrc 2>/dev/null; then
    echo 'export OPF_MOE_FUSED_SWIGLU_W2=0' >> ~/.bashrc
fi
export OPF_MOE_FUSED_SWIGLU_W2=0

# Persist HF_TOKEN too so a reattached tmux still has it.
if ! grep -q "^export HF_TOKEN=" ~/.bashrc 2>/dev/null; then
    echo "export HF_TOKEN=${HF_TOKEN}" >> ~/.bashrc
fi

# 5. Verify
echo
echo "--- Versions ---"
python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| cuda available:', torch.cuda.is_available())"
echo

# 6. Launch the full benchmark in a detached tmux session.
# Takes ~14h on a 4090. tmux means an ssh drop won't kill it.
# Set SKIP_RUN=1 to stop after bootstrap (for smoke-test work).
if [[ "${SKIP_RUN:-}" == "1" ]]; then
    cat <<'EOF'
--- Bootstrap complete (SKIP_RUN=1, eval not launched) ---

Smoke (one benchmark, 50 examples, ~20 sec):
  python -m scripts.download_datasets --benchmarks argilla --max-examples 50
  python -m opf_benchmarks.run_eval --device cuda --benchmarks argilla --extra --window-batch-size 32
EOF
    exit 0
fi

if tmux has-session -t eval 2>/dev/null; then
    echo "tmux session 'eval' already exists — not relaunching."
    echo "Attach with: tmux attach -t eval"
    exit 0
fi

REPO_PATH="$(pwd)"
# Sentinel file lets the laptop-side orchestrator poll for completion.
# Removed up-front so a re-run doesn't see a stale flag.
rm -f "$REPO_PATH/results/.eval_done" "$REPO_PATH/results/.eval_failed"

# OPF_SMOKE=1 (passed from vast_run.py --smoke): one benchmark, 50 examples, ~30s.
# Used to validate the provision → run → sentinel → rsync → destroy lifecycle without
# burning 14h of GPU time.
if [[ "${OPF_SMOKE:-}" == "1" ]]; then
    DOWNLOAD_ARGS="--benchmarks argilla --max-examples 50"
    EVAL_ARGS="--device cuda --benchmarks argilla --extra --window-batch-size 32"
    RUN_LABEL="smoke (argilla, 50 ex)"
else
    DOWNLOAD_ARGS=""
    EVAL_ARGS="--device cuda --extra --window-batch-size 32"
    RUN_LABEL="full eval (~14h on a 4090)"
fi

tmux new-session -d -s eval "bash -c 'cd $REPO_PATH && \
    python -m scripts.download_datasets $DOWNLOAD_ARGS 2>&1 | tee download.log && \
    python -m opf_benchmarks.run_eval $EVAL_ARGS 2>&1 | tee run.log && \
    mkdir -p results && touch results/.eval_done || (mkdir -p results && touch results/.eval_failed); \
    echo \"--- eval exited with code \$? ---\"; exec bash'"

cat <<EOF

--- Eval launched in tmux session 'eval': $RUN_LABEL ---

Watch:    tmux attach -t eval        (detach: Ctrl-b then d)
Logs:     tail -f $REPO_PATH/run.log

When it finishes, from your laptop in the repo root:
  ./scripts/vast_pull_results.sh <instance_id>      # results/ -> ./vast_results/

Then revoke the HF token at https://huggingface.co/settings/tokens
and destroy the instance:  vastai destroy instance <instance_id>
EOF
