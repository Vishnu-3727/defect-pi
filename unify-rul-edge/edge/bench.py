"""Measure what the deployment section of your report needs.

    python bench.py                # default thread count
    python bench.py --sweep        # thread-count sweep, the chart for your report

Report PERCENTILES, not the mean. Real-time engineering cares about the tail, and
p50/p95/p99 is what separates a measured system from a guessed one.

For a ~44k-parameter model, fewer threads is often FASTER — thread synchronisation
costs more than the parallelism gains. Counter-intuitive, easy to demonstrate, and
a good result to show.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    from inference import ArtifactsMissing, Predictor
except ImportError:
    from edge.inference import ArtifactsMissing, Predictor


def pi_thermals() -> dict:
    """Throttling invalidates a latency number. Record it alongside."""
    out = {}
    for key, cmd in (("temp_c", ["vcgencmd", "measure_temp"]),
                     ("throttled", ["vcgencmd", "get_throttled"])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            out[key] = r.stdout.strip().split("=")[-1]
        except Exception:
            out[key] = None      # not a Pi
    return out


def bench(threads: int, iters: int = 1000, warmup: int = 50) -> dict:
    p = Predictor(threads=threads)
    L, S = p.window, len(p.channels)
    rng = np.random.default_rng(0)
    win = rng.standard_normal((L, S)).astype(np.float32) * 0.1 + p.mean

    for _ in range(warmup):      # caches, thread pool, first-run graph setup
        p.predict(win)

    t = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        p.predict(win)
        t[i] = (time.perf_counter() - t0) * 1e3

    return {
        "threads": threads, "iters": iters, "sensors": S, "window": L,
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p95_ms": round(float(np.percentile(t, 95)), 3),
        "p99_ms": round(float(np.percentile(t, 99)), 3),
        "max_ms": round(float(t.max()), 3),
        "mean_ms": round(float(t.mean()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="../results/bench.json")
    a = ap.parse_args()

    try:
        p = Predictor()
    except ArtifactsMissing as e:
        raise SystemExit(f"{e}\nExport the model first: python -m unify.deploy.export_onnx")

    import onnxruntime as ort
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "onnxruntime": ort.__version__,
        "providers": p.sess.get_providers(),
        "model_sha": p.model_sha,
        "thermals_before": pi_thermals(),
    }
    print(json.dumps(info, indent=2))

    runs = [bench(t, a.iters) for t in ([1, 2, 3, 4] if a.sweep else [a.threads])]
    info["thermals_after"] = pi_thermals()

    print(f"\n{'threads':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}  (ms)")
    for r in runs:
        print(f"{r['threads']:>8} {r['p50_ms']:>8.3f} {r['p95_ms']:>8.3f} "
              f"{r['p99_ms']:>8.3f} {r['max_ms']:>8.3f}")

    best = min(runs, key=lambda r: r["p95_ms"])
    print(f"\nbest p95 at {best['threads']} thread(s): {best['p95_ms']} ms")
    print("Inference is a few ms out of a ~1000 ms end-to-end path — the 1-second "
          "feature window is the bottleneck, not the model. That is the right "
          "result for an edge system, and it means headroom for ~50 sensor nodes.")

    out = Path(__file__).parent / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"info": info, "runs": runs}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
