#!/usr/bin/env bash
# Pull benchmark results from a vast.ai instance back to the local repo.
#
# Run from your laptop in the repo root. Copies the remote box's
# opf-benchmarks-geo/results/ into ./vast_results/ via rsync.
#
# Usage:
#   ./scripts/vast_pull_results.sh <instance_id>

set -euo pipefail

INSTANCE_ID="${1:-}"
if [[ -z "$INSTANCE_ID" ]]; then
    echo "Usage: $0 <instance_id>" >&2
    echo "Find it with: vastai show instances" >&2
    exit 1
fi

command -v vastai >/dev/null || { echo "vastai CLI not found. pip install vastai" >&2; exit 1; }
command -v rsync  >/dev/null || { echo "rsync not found." >&2; exit 1; }

REMOTE_DIR="${REMOTE_DIR:-opf-benchmarks-geo/results/}"
LOCAL_DIR="${LOCAL_DIR:-./vast_results/}"

# vastai ssh-url emits 'ssh://root@host:port'. rsync needs host + port split.
SSH_URL="$(vastai ssh-url "$INSTANCE_ID")"
SSH_URL="${SSH_URL#ssh://}"                 # strip scheme
SSH_HOST="${SSH_URL%:*}"                    # root@host
SSH_PORT="${SSH_URL##*:}"                   # port

mkdir -p "$LOCAL_DIR"

echo "Pulling ${SSH_HOST}:${REMOTE_DIR} -> ${LOCAL_DIR} (port ${SSH_PORT})"
rsync -av --partial --progress \
    -e "ssh -p ${SSH_PORT} -o ForwardAgent=no" \
    "${SSH_HOST}:${REMOTE_DIR}" \
    "$LOCAL_DIR"

echo
echo "Done. Results in $LOCAL_DIR"
echo "Reminder: revoke the HF token used for this run at"
echo "  https://huggingface.co/settings/tokens"
