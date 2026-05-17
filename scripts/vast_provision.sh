#!/usr/bin/env bash
# Provision a vast.ai instance for opf-benchmarks.
#
# Run from your laptop. Searches for matching GPUs, shows the top offers,
# prompts for which one to rent, and creates the instance with a tested
# NVIDIA NGC PyTorch image (avoids the torch/triton/CUDA version dance
# that the plain `pytorch/pytorch` image causes).
#
# Usage:
#   ./scripts/vast_provision.sh            # interactive: shows offers, asks
#   ./scripts/vast_provision.sh <offer_id> # skip search, rent this offer
#
# Requires: vastai CLI installed and authenticated.

set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.05-py3}"
DISK_GB="${DISK_GB:-60}"
QUERY="${QUERY:-gpu_name in [RTX_4090,RTX_3090,RTX_3080_Ti] num_gpus=1 cuda_vers>=12.4 disk_space>=${DISK_GB} inet_down>=100 rentable=true verified=true}"

command -v vastai >/dev/null || { echo "vastai CLI not found. pip install vastai" >&2; exit 1; }

# HF_TOKEN: prefer env, fall back to huggingface-cli's cached token.
# SECURITY: vast.ai hosts have root on the physical box and can read container
# env vars. Use a READ-ONLY fine-grained HF token here, never a write token,
# and revoke it after the run.
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "No HF_TOKEN found in env or ~/.cache/huggingface/token." >&2
    echo "Run 'huggingface-cli login' with a READ-ONLY token, or 'export HF_TOKEN=hf_...'." >&2
    exit 1
fi

OFFER_ID="${1:-}"

if [[ -z "$OFFER_ID" ]]; then
    echo "Searching offers matching:"
    echo "  $QUERY"
    echo
    vastai search offers "$QUERY" --order dph --limit 10
    echo
    read -rp "Offer ID to rent (paste from column 'ID' above): " OFFER_ID
fi

if [[ -z "$OFFER_ID" ]]; then
    echo "No offer ID provided. Aborting." >&2
    exit 1
fi

echo
echo "Creating instance from offer $OFFER_ID:"
echo "  image: $IMAGE"
echo "  disk:  ${DISK_GB} GB"
echo

vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh \
    --direct \
    --env "-e HF_TOKEN=$HF_TOKEN -e OPF_MOE_FUSED_SWIGLU_W2=0"

echo
echo "--- Next steps ---"
echo "1. Wait ~30-60s for the instance to start, then:"
echo "     vastai show instances"
echo "2. Note the new instance/contract ID from the output and SSH in:"
echo "     ssh \$(vastai ssh-url <instance_id>)"
echo "3. On the box, HF_TOKEN is already exported via --env. Run:"
echo "     curl -sL https://raw.githubusercontent.com/CupOfGeo/opf-benchmarks-geo/main/scripts/vast_setup.sh | bash"
echo
echo "--- After the run ---"
echo "Revoke the HF token you used at https://huggingface.co/settings/tokens"
echo "(vast hosts have root on the box; treat any token sent there as compromised)."
