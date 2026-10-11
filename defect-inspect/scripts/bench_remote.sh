#!/usr/bin/env bash
# Run the CPU benchmark on a remote ARM machine and bring the numbers back.
#
# Why: the development laptop is x86-64 with AVX-512 and VNNI. The Pi 5 is four
# Cortex-A76 cores with 128-bit NEON. Laptop FPS says nothing about the Pi, so
# the benchmark has to run on an ARM CPU to mean anything.
#
# Oracle Cloud's Always Free tier gives 4 Ampere A1 cores (Neoverse-N1): the
# same ARMv8.2 generation as the A76, same two 128-bit NEON FMA pipes, same
# SDOT instruction for INT8. Not a Pi, but a real ARM measurement whose ratios
# and rough magnitude transfer.
#
# Ships only what the benchmark needs -- about 18 MB, no dataset, no weights
# beyond the two ONNX files, and nothing that requires torch on the far end.
#
# Usage:
#   bash scripts/bench_remote.sh ubuntu@<public-ip>
#   bash scripts/bench_remote.sh ubuntu@<public-ip> ~/.ssh/oracle_key

set -euo pipefail

HOST="${1:?usage: bench_remote.sh user@host [ssh-key]}"
KEY="${2:-}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[ -n "$KEY" ] && SSH_OPTS+=(-i "$KEY")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="defect-inspect-bench"
BUNDLE="$(mktemp -t defect-bench-XXXXXX).tar.gz"
trap 'rm -f "$BUNDLE"' EXIT

echo "==> packing bundle"
tar -czf "$BUNDLE" -C "$REPO_ROOT" \
    pyproject.toml \
    uv.lock \
    src/inspector \
    configs \
    bench_fixtures \
    models/model.onnx \
    models/model_int8.onnx
echo "    $(du -h "$BUNDLE" | cut -f1)"

echo "==> shipping to $HOST"
scp "${SSH_OPTS[@]}" "$BUNDLE" "$HOST:/tmp/bench.tar.gz"

echo "==> running on the remote ARM host"
ssh "${SSH_OPTS[@]}" "$HOST" REMOTE_DIR="$REMOTE_DIR" 'bash -s' <<'REMOTE'
set -euo pipefail

echo "--- what CPU is this actually ---"
uname -m
nproc
lscpu | grep -Ei 'model name|architecture|^cpu\(s\)|mhz|bogomips' || true
grep -m1 Features /proc/cpuinfo || true
echo

if [ "$(uname -m)" != "aarch64" ]; then
    echo "REFUSING: remote host is $(uname -m), not aarch64."
    echo "The entire point of this script is to measure ARM. Aborting."
    exit 1
fi

rm -rf "$REMOTE_DIR"
mkdir -p "$REMOTE_DIR"
tar -xzf /tmp/bench.tar.gz -C "$REMOTE_DIR"
cd "$REMOTE_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "--- installing uv ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

echo "--- installing runtime dependencies (no torch, no ultralytics) ---"
uv sync --no-dev

echo "--- confirming the runtime is deployment-clean ---"
uv run python - <<'PY'
import importlib.util as u
leaked = [m for m in ("torch", "ultralytics", "torchvision") if u.find_spec(m)]
assert not leaked, f"runtime pulled in {leaked}; the Pi must not need these"
print("clean: no torch, no ultralytics, no torchvision")
PY

echo
echo "--- self-checks ---"
uv run python -m inspector.config
uv run python -m inspector.preprocess

echo
echo "======================= 4 threads, matching Pi 5 core count ================"
uv run python -m inspector.bench --threads 4 --frames 60 --repeats 3

echo
echo "======================= FP32 vs INT8 across thread counts =================="
uv run python -m inspector.bench --compare --frames 60 --repeats 3
REMOTE

echo
echo "==> done. Paste the two tables into reports/benchmark.md, replacing the"
echo "    projection with the measured ARM numbers, and label the CPU model."
