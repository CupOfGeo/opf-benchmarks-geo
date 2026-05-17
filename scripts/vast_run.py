"""End-to-end vast.ai benchmark run, driven from your laptop.

Picks a GPU offer, creates the instance with HF_TOKEN injected via --env, fires
vast_setup.sh as the container onstart (which downloads datasets and launches
the full eval in tmux), polls for the sentinel file the eval writes when done,
rsyncs results into ./vast_results/, and optionally destroys the instance.

The eval runs in tmux on the box, so a laptop sleep / network drop is fine —
just rerun with `--instance-id <id>` to resume polling + pull.

Usage:
    python scripts/vast_run.py                          # interactive offer pick
    python scripts/vast_run.py --offer-id 12345
    python scripts/vast_run.py --instance-id 67890      # resume on existing instance
    python scripts/vast_run.py --destroy-on-success     # auto-destroy after pull

SECURITY: only use a READ-ONLY HF token. Vast hosts have root on the physical
box and can read container env vars; revoke the token at
https://huggingface.co/settings/tokens after the run.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from vastai_sdk import VastAI

IMAGE = "nvcr.io/nvidia/pytorch:24.05-py3"
DISK_GB = 60
QUERY = (
    # 24GB+ only: OPF + window-batch-size 32 OOMs on the 12GB 3080 Ti at nemotron's
    # longer sequences. RTX 3090 is the sweet spot (24GB, cheap).
    "gpu_name in [RTX_4090,RTX_3090] num_gpus=1 gpu_ram>=20000 cuda_vers>=12.4 "
    f"disk_space>={DISK_GB} inet_down>=100 rentable=true verified=true"
)
SETUP_URL = "https://raw.githubusercontent.com/CupOfGeo/opf-benchmarks-geo/main/scripts/vast_setup.sh"
REPO_DIR = "opf-benchmarks-geo"
REMOTE_RESULTS = f"{REPO_DIR}/results/"
LOCAL_RESULTS = Path("./vast_results/")
SENTINEL_DONE = f"{REPO_DIR}/results/.eval_done"
SENTINEL_FAILED = f"{REPO_DIR}/results/.eval_failed"

POLL_RUNNING_SECS = 15      # while waiting for actual_status=running
POLL_EVAL_SECS = 300        # while waiting for the eval to finish
RUNNING_TIMEOUT_SECS = 900  # ~15 min for container to come up
SSH_TIMEOUT_SECS = 600


def resolve_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    cached = Path("~/.cache/huggingface/token").expanduser()
    if cached.exists():
        return cached.read_text().strip()
    sys.exit(
        "No HF_TOKEN found in env or ~/.cache/huggingface/token.\n"
        "Run `huggingface-cli login` with a READ-ONLY token, or `export HF_TOKEN=hf_...`."
    )


def pick_offer(sdk: VastAI) -> int:
    print(f"Searching offers:\n  {QUERY}\n")
    offers = sdk.search_offers(query=QUERY, order="dph_total", limit=10)
    if not offers:
        sys.exit("No offers matched. Try loosening QUERY.")
    print(f"{'#':>2}  {'ID':>10}  {'GPU':<14}  {'$/hr':>6}  {'down':>5}  reliability")
    for i, o in enumerate(offers):
        print(
            f"{i:>2}  {o.get('id'):>10}  {o.get('gpu_name', '?'):<14}  "
            f"{o.get('dph_total', 0):>6.3f}  {int(o.get('inet_down') or 0):>4}M  "
            f"{o.get('reliability2', 0):.3f}"
        )
    while True:
        choice = input("\nPick by index or paste offer ID (q to quit): ").strip()
        if choice.lower() in ("q", "quit", "exit", ""):
            sys.exit("Aborted.")
        if choice.isdigit():
            n = int(choice)
            if n < len(offers):
                return int(offers[n]["id"])
            return n  # treat as offer ID
        print(f"  '{choice}' isn't a number — try again.")


def create_instance(sdk: VastAI, offer_id: int, hf_token: str, smoke: bool) -> int:
    # The API wants env as a dict {"KEY": "VAL"}; the docker-flag string "-e KEY=VAL"
    # is a CLI convenience that gets parsed (vastai.cli.util.parse_env) before send.
    env = {"HF_TOKEN": hf_token, "OPF_MOE_FUSED_SWIGLU_W2": "0"}
    if smoke:
        env["OPF_SMOKE"] = "1"
    # onstart runs once during container init. tmux daemonizes, so the eval persists
    # after onstart returns.
    onstart = f"curl -sL {SETUP_URL} | bash >> /var/log/opf_bootstrap.log 2>&1"
    print(f"Creating instance from offer {offer_id} (image {IMAGE}, disk {DISK_GB}GB)...")
    resp = sdk.create_instance(
        id=offer_id,
        image=IMAGE,
        disk=DISK_GB,
        env=env,
        onstart_cmd=onstart,
        runtype="ssh_direc ssh_proxy",  # literal — CLI spelling for --ssh --direct
    )
    instance_id = resp.get("new_contract") or resp.get("id")
    if not instance_id:
        sys.exit(f"Failed to parse instance ID from create response: {resp}")
    print(f"Created instance {instance_id}.")
    return int(instance_id)


def wait_for_running(sdk: VastAI, instance_id: int) -> tuple[str, int]:
    """Poll show_instance until actual_status=running and SSH host/port are populated."""
    print(f"Waiting for instance {instance_id} to enter 'running' state...")
    deadline = time.time() + RUNNING_TIMEOUT_SECS
    last_status = None
    while time.time() < deadline:
        info = sdk.show_instance(id=instance_id)
        status = info.get("actual_status")
        if status != last_status:
            print(f"  status={status}")
            last_status = status
        if status == "running":
            try:
                url = sdk.ssh_url(id=instance_id)
            except Exception as e:
                print(f"  ssh_url not ready yet ({e}); retrying")
                url = None
            if url:
                parsed = urlparse(url)
                host = f"{parsed.username or 'root'}@{parsed.hostname}"
                port = parsed.port or 22
                if probe_ssh(host, port):
                    return host, port
                print("  ssh probe failed, retrying")
        time.sleep(POLL_RUNNING_SECS)
    sys.exit(f"Instance {instance_id} never became reachable. Check `vastai show instance {instance_id}`.")


def probe_ssh(host: str, port: int) -> bool:
    return subprocess.run(
        [
            "ssh", "-p", str(port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=8",
            "-o", "BatchMode=yes",
            "-o", "ForwardAgent=no",
            host, "true",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def ssh_run(host: str, port: int, cmd: str, *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh", "-p", str(port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ForwardAgent=no",
            "-o", "ConnectTimeout=15",
            host, cmd,
        ],
        capture_output=capture, text=True,
    )


def poll_for_done(host: str, port: int) -> str:
    """Returns 'done' or 'failed'.

    The onstart bootstrap (apt + git clone + pip install OPF from source) can run
    for 5-10 min before tmux is even launched. So "no tmux session" is NOT a
    signal that the eval died — it could just mean we're still bootstrapping.
    We only terminate on the explicit sentinel files. Ctrl-C if something's truly
    stuck; instance stays alive so you can ssh in and inspect.
    """
    print(
        f"\nPolling for eval completion every {POLL_EVAL_SECS}s.\n"
        "(eval runs in tmux on the box; safe to Ctrl-C and resume later "
        "with --instance-id <id>)"
    )
    # Tail whichever log is most informative for the current phase:
    # bootstrap log during pip install, download.log during dataset fetch, run.log during eval.
    check = (
        f"if [ -f {SENTINEL_DONE} ]; then echo done; "
        f"elif [ -f {SENTINEL_FAILED} ]; then echo failed; "
        f"elif tmux has-session -t eval 2>/dev/null; then echo running; "
        f"elif [ -d {REPO_DIR} ]; then echo bootstrapping; "
        f"else echo cloning; fi; "
        f"for f in {REPO_DIR}/run.log {REPO_DIR}/download.log /var/log/opf_bootstrap.log; do "
        f"  [ -s \"$f\" ] && echo \"--- $f ---\" && tail -n 3 \"$f\" && break; "
        f"done | sed 's/^/  /'"
    )
    while True:
        r = ssh_run(host, port, check)
        lines = (r.stdout or "").strip().splitlines() or ["unreachable"]
        state = lines[0].strip()
        ts = time.strftime("%H:%M:%S")
        print(f"{ts}  {state}")
        for line in lines[1:]:
            print(f"          {line}")
        if state in ("done", "failed"):
            return state
        time.sleep(POLL_EVAL_SECS)


def pull_results(host: str, port: int) -> None:
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"\nrsync {host}:{REMOTE_RESULTS} -> {LOCAL_RESULTS}/")
    ssh_opts = (
        f"ssh -p {port} -o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null -o ForwardAgent=no"
    )
    subprocess.run(
        [
            "rsync", "-av", "--partial", "--progress",
            "-e", ssh_opts,
            f"{host}:{REMOTE_RESULTS}",
            f"{LOCAL_RESULTS}/",
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offer-id", type=int, help="Skip interactive search, provision this offer.")
    p.add_argument("--instance-id", type=int, help="Resume polling/pulling on an existing instance.")
    p.add_argument("--destroy-on-success", action="store_true", help="Destroy the instance after a clean pull.")
    p.add_argument("--smoke", action="store_true",
                   help="Run a tiny smoke eval (argilla, 50 examples, ~30s) instead of the full 14h benchmark. "
                        "Use this to validate the full lifecycle end-to-end before committing to an overnight run.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    hf_token = resolve_hf_token()
    sdk = VastAI()  # auto-reads ~/.config/vastai/vast_api_key

    if args.instance_id:
        instance_id = args.instance_id
        print(f"Reusing instance {instance_id} (skipping provision).")
    else:
        offer_id = args.offer_id or pick_offer(sdk)
        instance_id = create_instance(sdk, offer_id, hf_token, args.smoke)

    # Smoke eval finishes in ~30s once tmux starts; tight poll catches it quickly.
    # Bootstrap is still the slow part (~5 min) so first poll likely sees "bootstrapping".
    global POLL_EVAL_SECS
    if args.smoke:
        POLL_EVAL_SECS = 30

    host, port = wait_for_running(sdk, instance_id)
    print(f"SSH ready: {host} (port {port})")

    state = poll_for_done(host, port)
    if state == "failed":
        print("Eval reported failure; pulling partial results.")

    pull_results(host, port)

    print()
    if args.destroy_on_success and state == "done":
        print(f"Destroying instance {instance_id} (--destroy-on-success)...")
        sdk.destroy_instance(id=instance_id)
    else:
        print(f"Instance {instance_id} left running. Destroy with:")
        print(f"  vastai destroy instance {instance_id}")

    print(
        f"\n--- Done ---\nResults: {LOCAL_RESULTS}/\n"
        "Revoke the HF token used for this run: https://huggingface.co/settings/tokens"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
