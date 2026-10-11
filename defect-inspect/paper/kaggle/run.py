#!/usr/bin/env python3
"""Push the resolution experiment to Kaggle, wait for it, fetch the results.

    python paper/kaggle/run.py              # push and poll until done
    python paper/kaggle/run.py --status     # just report state
    python paper/kaggle/run.py --fetch      # just download output

Needs ~/.kaggle/kaggle.json (kaggle.com > Settings > API > Create New Token)
and a phone-verified account -- Kaggle withholds both GPU and internet access
from unverified accounts, and this notebook needs internet to clone NEU-DET.

The kernel slug is derived from the username in kaggle.json, so nothing here
carries a hardcoded account.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "defect-inspect-resolution-retrain"  # must match what the title slugifies to
CREDS = Path.home() / ".kaggle" / "kaggle.json"
# The CLI lives in the evaluation venv, not on PATH.
KAGGLE = Path.home() / "dv-eval" / "bin" / "kaggle"


def die(msg: str) -> "None":
    sys.exit(f"error: {msg}")


def username() -> str:
    if not CREDS.exists():
        die(f"{CREDS} not found.\n"
            "  1. kaggle.com > Settings > API > Create New Token\n"
            f"  2. mkdir -p {CREDS.parent} && mv ~/Downloads/kaggle.json {CREDS.parent}/\n"
            f"  3. chmod 600 {CREDS}")
    mode = CREDS.stat().st_mode & 0o777
    if mode & 0o077:
        print(f"warning: {CREDS} is mode {mode:o}; tightening to 600")
        CREDS.chmod(0o600)
    return json.loads(CREDS.read_text())["username"]


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    if not KAGGLE.exists():
        die(f"{KAGGLE} not found. Install with: {Path.home()}/dv-eval/bin/pip install kaggle")
    p = subprocess.run([str(KAGGLE), *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        die(f"kaggle {' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")
    return p


def write_metadata(user: str) -> None:
    """kernel-metadata.json, regenerated so the account is never committed."""
    meta = {
        "id": f"{user}/{SLUG}",
        "title": "defect-inspect resolution retrain",
        "code_file": "resolution_retrain.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        # Required: the notebook clones the NEU-DET mirror from GitHub.
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"metadata -> {user}/{SLUG}  (private, GPU on, internet on)")


def status(ref: str) -> str:
    out = run("kernels", "status", ref, check=False).stdout.strip()
    print(out or "(no status yet)")
    for state in ("complete", "error", "cancelAcknowledged", "running", "queued"):
        if state in out.lower():
            return state
    return "unknown"


def fetch(ref: str) -> None:
    dest = HERE / "output"
    dest.mkdir(exist_ok=True)
    run("kernels", "output", ref, "-p", str(dest))
    files = sorted(p.name for p in dest.rglob("*") if p.is_file())
    print(f"\n{len(files)} files in {dest}:")
    for f in files:
        print("  ", f)
    csv = dest / "resolution_matrix.csv"
    if csv.exists():
        print("\n--- resolution_matrix.csv ---")
        print(csv.read_text().strip())
    else:
        print("\nno resolution_matrix.csv -- the run did not reach the evaluation cell.")
        log = dest / f"{SLUG}.log"
        if log.exists():
            print("last 25 log lines:")
            print("\n".join(log.read_text().splitlines()[-25:]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--timeout-min", type=int, default=120)
    args = ap.parse_args()

    ref = f"{username()}/{SLUG}"

    if args.status:
        status(ref)
        return
    if args.fetch:
        fetch(ref)
        return

    write_metadata(username())
    print("pushing...")
    print(run("kernels", "push", "-p", str(HERE)).stdout.strip())

    # Two training runs on a T4: roughly 15 min at 320 and 40 at 640, plus
    # queueing. Poll on a long interval -- a tight loop achieves nothing here
    # and Kaggle rate-limits the status endpoint.
    deadline = time.time() + args.timeout_min * 60
    while time.time() < deadline:
        time.sleep(120)
        print(f"\n[{time.strftime('%H:%M:%S')}] ", end="")
        st = status(ref)
        if st in ("complete", "error", "cancelAcknowledged"):
            print(f"\nfinished: {st}")
            fetch(ref)
            return
    print(f"\nstill running after {args.timeout_min} min. Check later:")
    print(f"  python {Path(__file__).name} --status")
    print(f"  python {Path(__file__).name} --fetch")


if __name__ == "__main__":
    main()
