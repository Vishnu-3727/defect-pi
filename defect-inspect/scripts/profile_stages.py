"""Time each stage of the frame loop separately, on the machine that runs it.

    python scripts/profile_stages.py [--repeats 3] [--frames 60]

WHY THIS EXISTS
`inspector.bench` measures inference alone, and tests/test_pipeline.py measures
the system end to end. Neither says where the difference between them goes, and
that difference was attributed for a while to SQLite commits and failure-image
encoding. Measured, those two cost under 2% of the frame budget; the gap is
inference plus the gated anomaly backstop. Attribution by intuition was wrong by
more than an order of magnitude, which is the reason to keep a profiler in the
repository rather than reason about it.

READING THE OUTPUT
Stages that do not run every frame -- imwrite only on a FAIL, the anomaly model
only when gated in -- are reported twice: per call, and amortised per frame. The
per-frame column is the one that adds up to the budget, and the one to optimise
against. A stage with a large mean and a near-zero median runs rarely and costs
a lot when it does; that is the anomaly backstop by design.

VALIDITY
Run this only on a board that is not throttling. `vcgencmd get_throttled` must
read 0x0, or have only the sticky bits 16+ set, before and after. Under
under-voltage the Pi 5 drops from 2400 to 1000 MHz and the inference mean
inflates while the median does not -- which shows up here as a wide gap between
the two, and makes the numbers worthless for comparison.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2                                          # noqa: E402
from inspector.config import load_config            # noqa: E402
from inspector.decision import decide               # noqa: E402
from inspector.detector import draw                 # noqa: E402
from inspector.pipeline import Pipeline             # noqa: E402
from inspector.storage import InspectionRecord      # noqa: E402

STAGES = ("detect", "anomaly", "draw", "imwrite", "imencode", "db")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    fixtures = REPO_ROOT / "bench_fixtures" / "images"
    p = Pipeline(cfg, source_override=str(fixtures), loop=True)

    it = iter(p.source)
    frames = [next(it) for _ in range(args.frames)]

    T: dict[str, list[float]] = {k: [] for k in STAGES}
    scratch = []

    for _ in range(args.repeats):
        for i, frame in enumerate(frames):
            t = time.perf_counter()
            dets = p.detector.detect(frame)
            T["detect"].append((time.perf_counter() - t) * 1e3)

            best = max((d.confidence for d in dets), default=None)

            t = time.perf_counter()
            score = (p.anomaly.maybe_score(frame, best, cfg.anomaly.gate_conf_below)
                     if p.anomaly else None)
            T["anomaly"].append((time.perf_counter() - t) * 1e3)

            verdict = decide(dets, score, cfg.decision.min_defect_area_px,
                             cfg.anomaly.score_threshold)

            t = time.perf_counter()
            ann = draw(frame, dets)
            T["draw"].append((time.perf_counter() - t) * 1e3)

            if verdict.is_fail:
                out = p.failure_dir / f"_profile_{i}.jpg"
                scratch.append(out)
                t = time.perf_counter()
                cv2.imwrite(str(out), ann)
                T["imwrite"].append((time.perf_counter() - t) * 1e3)

            t = time.perf_counter()
            cv2.imencode(".jpg", ann,
                         [cv2.IMWRITE_JPEG_QUALITY, cfg.dashboard.jpeg_quality])
            T["imencode"].append((time.perf_counter() - t) * 1e3)

            t = time.perf_counter()
            p.storage.insert(InspectionRecord(
                time.time(), verdict.verdict, verdict.reason, verdict.defect_class,
                best, len(dets), score, 1.0, verdict.is_fail, None))
            T["db"].append((time.perf_counter() - t) * 1e3)

    n_frames = len(frames) * args.repeats
    print(f"\n{args.frames} frames x {args.repeats} repeats = {n_frames} frames\n")
    print(f"{'stage':<10} {'calls':>6} {'median':>8} {'mean':>8} {'p95':>8} {'/frame':>8}")
    print("-" * 56)

    budget = 0.0
    for k in STAGES:
        v = sorted(T[k])
        if not v:
            print(f"{k:<10} {0:>6}   never ran")
            continue
        per_frame = st.mean(v) * len(v) / n_frames
        budget += per_frame
        print(f"{k:<10} {len(v):>6} {st.median(v):>8.2f} {st.mean(v):>8.2f} "
              f"{v[int(len(v) * .95)]:>8.2f} {per_frame:>8.2f}")

    print("-" * 56)
    print(f"{'TOTAL':<10} {'':>6} {'':>8} {'':>8} {'':>8} {budget:>8.2f} ms")
    print(f"\nceiling {1000 / budget:.1f} FPS  (all stages serial, single thread)")

    det = sorted(T["detect"])
    skew = st.mean(det) / st.median(det) if st.median(det) else 1.0
    if skew > 1.25:
        print(f"\nWARNING: detect mean/median = {skew:.2f}. That skew is what "
              f"throttling looks like.\n         Check `vcgencmd get_throttled` "
              f"-- these numbers are not comparable if it is non-zero\n"
              f"         in bits 0-3.")

    for f in scratch:
        f.unlink(missing_ok=True)
    p.storage.conn.execute("DELETE FROM inspections")
    p.storage.conn.commit()
    p.close()


if __name__ == "__main__":
    main()
