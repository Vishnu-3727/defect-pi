"""Measure per-stage latency, frame rate and peak memory on the CPU.

Every speed number in the report comes from this file. None of them come from a
blog post about a different Raspberry Pi running a different model.

The laptop is not a Pi 5, so a laptop run is a *proxy*, not the answer. Pinning
to four threads and disabling the GPU makes it a defensible one -- both are
four-core ARM-class CPU inference of the same graph -- but the Pi number must
be re-measured on a Pi. The output labels itself accordingly.

Run:  uv run python -m inspector.bench --model models/model_int8.onnx --threads 4
      uv run python -m inspector.bench --compare
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import REPO_ROOT, load_config
from .detector import Detector


def peak_rss_mb() -> float:
    """Peak resident set size in MB, or 0.0 if the platform will not say.

    Two implementations because the project runs on both Windows (development)
    and Linux (the Pi), and neither exposes the other's API.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = COUNTERS()
        counters.cb = ctypes.sizeof(COUNTERS)
        # Since Windows 7 the function lives in kernel32 as K32GetProcessMemoryInfo;
        # the psapi.dll export is a compatibility shim that is not always present.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_info = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if get_info is None:
            get_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        get_info.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        handle = ctypes.c_void_p(handle)
        ok = get_info(handle, ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 1e6 if ok else 0.0

    try:
        import resource
        # Linux reports kilobytes, macOS reports bytes.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1e3 if sys.platform.startswith("linux") else peak / 1e6
    except ImportError:
        return 0.0


def load_frames(folder: Path, count: int) -> list[np.ndarray]:
    """Real images, class-balanced, decoded once and held in RAM.

    Benchmarking on a blank array would be dishonest in a specific way: NMS cost
    scales with how many candidate boxes survive the confidence threshold, and a
    blank frame produces none. The measured post-processing time would be zero.
    """
    paths = sorted(folder.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no images in {folder}; run scripts/prepare_dataset.py")

    by_class: dict[str, list[Path]] = {}
    for p in paths:
        by_class.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)

    per_class = max(1, count // len(by_class))
    chosen = [p for cls in sorted(by_class) for p in by_class[cls][:per_class]]
    return [cv2.imread(str(p)) for p in chosen]


def benchmark(model: Path, cfg, threads: int, frames: list[np.ndarray],
              warmup: int, repeats: int) -> dict:
    det = Detector(model, cfg.classes, cfg.detector.imgsz,
                   cfg.detector.conf_threshold, cfg.detector.iou_threshold, threads)

    # The first few inferences allocate arenas and populate caches, and are
    # several times slower than steady state. Including them would understate
    # the frame rate the line actually sustains.
    for frame in frames[:warmup]:
        det.detect(frame)

    pre, inf, post, total, boxes = [], [], [], [], 0
    for _ in range(repeats):
        for frame in frames:
            t0 = time.perf_counter()
            boxes += len(det.detect(frame))
            total.append((time.perf_counter() - t0) * 1e3)
            pre.append(det.timing.preprocess_ms)
            inf.append(det.timing.inference_ms)
            post.append(det.timing.postprocess_ms)

    return {
        "model": model.name,
        "size_mb": model.stat().st_size / 1e6,
        "threads": threads,
        "n": len(total),
        "preprocess_ms": statistics.median(pre),
        "inference_ms": statistics.median(inf),
        "postprocess_ms": statistics.median(post),
        "total_ms": statistics.median(total),
        "p95_ms": sorted(total)[int(len(total) * 0.95)],
        "fps": 1000.0 / statistics.median(total),
        "boxes_per_frame": boxes / len(total),
        "peak_rss_mb": peak_rss_mb(),
    }


# Cortex-A76 roofline, from the published core specification rather than from a
# benchmark blog. Each core has two 128-bit NEON pipes; each pipe issues one
# fused multiply-add over 4 float32 lanes per cycle, and an FMA counts as two
# floating-point operations.
#
#   2 pipes x 4 lanes x 2 FLOP = 16 FLOP per cycle per core
#   16 x 2.4 GHz x 4 cores     = 153.6 GFLOP/s peak for a Pi 5
#
# Peak is not achievable. Real convolution efficiency on a CPU is limited by
# memory traffic and by layers too small to fill the vector units, and lands
# between roughly 25% and 40% of peak. That spread is why the projection below
# is a band and not a number.
PI5_PEAK_GFLOPS = 153.6
PI5_EFFICIENCY = (0.25, 0.40)
# Preprocessing is memory-bound and does not use the vector units well, so it
# scales worse than inference. Measured 2.1-2.6 ms on a Zen 5 core; on an A76
# expect roughly 3x that, and more if the camera delivers larger frames.
PI5_PREPROCESS_MS = (6.0, 9.0)
PI5_POSTPROCESS_MS = (1.5, 3.0)


def project_pi5(gflops: float) -> tuple[float, float]:
    """Projected total ms per frame on a Pi 5, as a (best, worst) band.

    This is arithmetic from a published core specification, not a measurement.
    It exists so the report contains a defensible number instead of a laptop
    figure wearing a Pi label. Replace it with a real measurement the day the
    hardware arrives.
    """
    fast = (gflops / (PI5_PEAK_GFLOPS * PI5_EFFICIENCY[1])) * 1e3 \
        + PI5_PREPROCESS_MS[0] + PI5_POSTPROCESS_MS[0]
    slow = (gflops / (PI5_PEAK_GFLOPS * PI5_EFFICIENCY[0])) * 1e3 \
        + PI5_PREPROCESS_MS[1] + PI5_POSTPROCESS_MS[1]
    return fast, slow


def print_table(rows: list[dict]) -> None:
    print()
    print(f"{'model':<20}{'thr':>4}{'pre':>7}{'infer':>8}{'post':>7}"
          f"{'total':>8}{'p95':>8}{'FPS':>8}{'MB':>7}")
    print("-" * 77)
    for r in rows:
        print(f"{r['model']:<20}{r['threads']:>4}{r['preprocess_ms']:>7.2f}"
              f"{r['inference_ms']:>8.2f}{r['postprocess_ms']:>7.2f}"
              f"{r['total_ms']:>8.2f}{r['p95_ms']:>8.2f}{r['fps']:>8.1f}"
              f"{r['size_mb']:>7.1f}")
    print("-" * 77)
    print("times are median milliseconds per frame; p95 is the tail latency")
    print(f"peak RSS: {rows[-1]['peak_rss_mb']:.0f} MB   "
          f"boxes/frame: {rows[-1]['boxes_per_frame']:.2f}")
    print(f"host: {platform.processor() or platform.machine()} ({platform.machine()})")


def print_projection(gflops: float) -> None:
    """Print the Pi 5 projection, but only when the host is not ARM."""
    if platform.machine() in {"aarch64", "arm64", "armv7l"}:
        print("\nHost is ARM. The table above is a real measurement for this device.")
        return

    fast, slow = project_pi5(gflops)
    print()
    print("=" * 77)
    print("THE TABLE ABOVE IS NOT A RASPBERRY PI RESULT AND DOES NOT PREDICT ONE.")
    print("=" * 77)
    print("This host is x86-64. Matching the thread count does not make it")
    print("comparable to a Cortex-A76, because the difference is per-core, not")
    print("per-core-count:")
    print()
    print("    clock          ~5.0 GHz boost      vs   2.4 GHz")
    print("    SIMD width     AVX-512 (512-bit)   vs   NEON (128-bit)")
    print("    INT8 per instr 64 lanes (VNNI)     vs   16 lanes (SDOT)")
    print()
    print("An x86 core here does roughly 4x the INT8 lanes per instruction at")
    print("about twice the clock. Quoting this FPS for the Pi would overstate it")
    print("by a large and unknown factor.")
    print()
    print(f"Projection for a Pi 5 at {gflops:.1f} GFLOPs per frame, computed from the")
    print("A76 roofline (153.6 GFLOP/s peak, 25-40% achieved efficiency):")
    print()
    print(f"    {slow:.0f} ms/frame ({1000 / slow:.0f} FPS)  to  "
          f"{fast:.0f} ms/frame ({1000 / fast:.0f} FPS)")
    print()
    print("That is an ESTIMATE derived from a published core specification.")
    print("It is not a measurement and must be labelled as an estimate wherever")
    print("it appears. Re-run this command on the Pi to replace it.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="defaults to the configured model")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--compare", action="store_true",
                    help="benchmark FP32 and INT8 across thread counts")
    ap.add_argument("--gflops", type=float, default=2.0,
                    help="model GFLOPs per frame; ultralytics prints this on export")
    ap.add_argument("--images-dir", default=None,
                    help="frame source; defaults to the test split, falls back to "
                         "bench_fixtures/images so CI can run without the dataset")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.images_dir:
        source = Path(args.images_dir)
    else:
        source = cfg.resolve("data/yolo/test/images")
        # CI checks out the repo without the dataset, which is git-ignored.
        # bench_fixtures holds a committed, class-balanced sample so the ARM
        # runner measures the same frames the laptop did.
        if not source.is_dir():
            source = cfg.resolve("bench_fixtures/images")
    frames = load_frames(source, args.frames)
    print(f"frames from {source}")
    print(f"benchmarking on {len(frames)} real test images, "
          f"{args.repeats} repeats, {args.warmup} warmup")

    rows = []
    if args.compare:
        for name in ("model.onnx", "model_int8.onnx"):
            path = REPO_ROOT / "models" / name
            if not path.exists():
                print(f"skipping {name}: not built")
                continue
            for threads in (1, 2, 4):
                rows.append(benchmark(path, cfg, threads, frames,
                                      args.warmup, args.repeats))
    else:
        path = Path(args.model) if args.model else cfg.resolve(cfg.detector.model)
        rows.append(benchmark(path, cfg, args.threads, frames,
                              args.warmup, args.repeats))

    print_table(rows)
    print_projection(args.gflops)


if __name__ == "__main__":
    main()
