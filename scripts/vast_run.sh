#!/usr/bin/env bash
# End-to-end vast.ai benchmark run from your laptop.
#
#   1. Pick a GPU offer (interactive) or use the one you pass in.
#   2. Create the instance with HF_TOKEN injected via --env.
#   3. Wait for SSH to come up.
#   4. Push & run vast_setup.sh on the box — kicks off the full eval in tmux.
#   5. Poll for the sentinel file the eval writes when done.
#   6. rsync results -> ./vast_results/
#   7. (Optional) destroy the instance.
#
# Usage:
#   ./scripts/vast_run.sh                       # interactive offer pick
#   ./scripts/vast_run.sh <offer_id>            # provision a specific offer
#   INSTANCE_ID=12345 ./scripts/vast_run.sh     # resume polling on existing instance
#
# Flags (env vars):
#   DESTROY_ON_SUCCESS=1   destroy the box after a clean pull (default: no)
#   POLL_SECS=300          how often to poll for eval completion (default: 5 min)
#
# Laptop sleep / network drop: the eval runs in tmux on the box and survives.
# Just rerun with INSTANCE_ID=<id> to resume polling + pull.
#
# SECURITY: only use a READ-ONLY HF token (revoke it after — vast hosts have
# root on the physical box).

set -euo pipefail

# ----- config -----
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.05-py3}"
DISK_GB="${DISK_GB:-60}"
QUERY="${QUERY:-gpu_name in [RTX_4090,RTX_3090,RTX_3080_Ti] num_gpus=1 cuda_vers>=12.4 disk_space>=${DISK_GB} inet_down>=100 rentable=true verified=true}"
REPO_URL="${REPO_URL:-https://github.com/CupOfGeo/opf-benchmarks-geo.git}"
REPO_DIR="${REPO_DIR:-opf-benchmarks-geo}"
REMOTE_RESULTS="${REMOTE_RESULTS:-$REPO_DIR/results/}"
LOCAL_RESULTS="${LOCAL_RESULTS:-./vast_results/}"
SETUP_URL="${SETUP_URL:-https://raw.githubusercontent.com/CupOfGeo/opf-benchmarks-geo/main/scripts/vast_setup.sh}"
POLL_SECS="${POLL_SECS:-300}"
SSH_WAIT_SECS="${SSH_WAIT_SECS:-600}"

# ----- preflight -----
command -v vastai >/dev/null || { echo "vastai CLI not found. pip install vastai" >&2; exit 1; }
command -v rsync  >/dev/null || { echo "rsync not found." >&2; exit 1; }
command -v jq     >/dev/null || { echo "jq not found. brew install jq" >&2; exit 1; }

HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "No HF_TOKEN found in env or ~/.cache/huggingface/token." >&2
    echo "Run 'huggingface-cli login' with a READ-ONLY token, or 'export HF_TOKEN=hf_...'." >&2
    exit 1
fi

# ----- provision (or reuse) -----
if [[ -n "${INSTANCE_ID:-}" ]]; then
    echo "Reusing existing instance $INSTANCE_ID (skipping provision)."
else
    OFFER_ID="${1:-}"
    if [[ -z "$OFFER_ID" ]]; then
        echo "Searching offers matching:"
        echo "  $QUERY"
        echo
        vastai search offers "$QUERY" --order dph --limit 10
        echo
        read -rp "Offer ID to rent: " OFFER_ID
        [[ -z "$OFFER_ID" ]] && { echo "No offer given. Aborting." >&2; exit 1; }
    fi

    echo "Creating instance from offer $OFFER_ID..."
    CREATE_OUT="$(vastai create instance "$OFFER_ID" \
        --image "$IMAGE" --disk "$DISK_GB" --ssh --direct \
        --env "-e HF_TOKEN=$HF_TOKEN -e OPF_MOE_FUSED_SWIGLU_W2=0" \
        --raw)"
    INSTANCE_ID="$(echo "$CREATE_OUT" | jq -r '.new_contract // empty')"
    if [[ -z "$INSTANCE_ID" ]]; then
        echo "Failed to parse instance ID from create response:" >&2
        echo "$CREATE_OUT" >&2
        exit 1
    fi
    echo "Created instance $INSTANCE_ID."
fi

# ----- wait for SSH -----
echo "Waiting for SSH on instance $INSTANCE_ID (up to ${SSH_WAIT_SECS}s)..."
deadline=$(( $(date +%s) + SSH_WAIT_SECS ))
SSH_URL=""
while (( $(date +%s) < deadline )); do
    SSH_URL="$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)"
    if [[ -n "$SSH_URL" && "$SSH_URL" == ssh://* ]]; then
        SSH_HOST_PORT="${SSH_URL#ssh://}"
        SSH_HOST="${SSH_HOST_PORT%:*}"
        SSH_PORT="${SSH_HOST_PORT##*:}"
        if ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
               -o ConnectTimeout=8 -o BatchMode=yes -o ForwardAgent=no \
               "$SSH_HOST" true 2>/dev/null; then
            break
        fi
    fi
    sleep 15
done

if [[ -z "${SSH_HOST:-}" ]] || ! ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -o BatchMode=yes \
        -o ForwardAgent=no "$SSH_HOST" true 2>/dev/null; then
    echo "SSH never came up. Check 'vastai show instance $INSTANCE_ID'." >&2
    exit 1
fi
echo "SSH ready: $SSH_HOST (port $SSH_PORT)"

SSH=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o ForwardAgent=no "$SSH_HOST")

# ----- bootstrap + launch (idempotent: skips if tmux session already running) -----
if "${SSH[@]}" "tmux has-session -t eval 2>/dev/null"; then
    echo "tmux session 'eval' already exists on the box — not relaunching."
else
    echo "Running vast_setup.sh on the box..."
    "${SSH[@]}" "curl -sL '$SETUP_URL' | bash"
fi

# ----- poll for completion -----
echo
echo "Polling for completion every ${POLL_SECS}s..."
echo "(eval keeps running in tmux even if this script dies; re-run with INSTANCE_ID=$INSTANCE_ID to resume)"
while true; do
    STATUS="$("${SSH[@]}" "
        if [ -f $REPO_DIR/results/.eval_done ]; then echo done;
        elif [ -f $REPO_DIR/results/.eval_failed ]; then echo failed;
        elif tmux has-session -t eval 2>/dev/null; then echo running;
        else echo gone; fi
    " 2>/dev/null || echo unreachable)"

    case "$STATUS" in
        done)        echo "$(date '+%H:%M:%S')  eval finished — pulling results"; break ;;
        failed)      echo "$(date '+%H:%M:%S')  eval reported failure. Pulling partial results anyway."; break ;;
        gone)        echo "$(date '+%H:%M:%S')  tmux session gone and no sentinel — eval likely died. Pulling what's there."; break ;;
        unreachable) echo "$(date '+%H:%M:%S')  ssh unreachable, retrying" ;;
        *)           echo "$(date '+%H:%M:%S')  $STATUS" ;;
    esac
    sleep "$POLL_SECS"
done

# ----- pull results -----
mkdir -p "$LOCAL_RESULTS"
echo "rsync ${SSH_HOST}:${REMOTE_RESULTS} -> ${LOCAL_RESULTS}"
rsync -av --partial --progress \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ForwardAgent=no" \
    "${SSH_HOST}:${REMOTE_RESULTS}" "$LOCAL_RESULTS"

# ----- teardown -----
echo
if [[ "${DESTROY_ON_SUCCESS:-}" == "1" ]]; then
    echo "Destroying instance $INSTANCE_ID (DESTROY_ON_SUCCESS=1)..."
    vastai destroy instance "$INSTANCE_ID"
else
    echo "Instance $INSTANCE_ID left running. Destroy with:"
    echo "  vastai destroy instance $INSTANCE_ID"
fi

cat <<EOF

--- Done ---
Results: $LOCAL_RESULTS
Revoke the HF token used for this run: https://huggingface.co/settings/tokens
EOF
