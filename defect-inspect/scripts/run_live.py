"""Run the inspection system.

    uv run python scripts/run_live.py --source data/yolo/test/images --loop
    uv run python scripts/run_live.py --source 0            # USB webcam
    uv run python scripts/run_live.py                        # whatever config says
    uv run python scripts/run_live.py --show                 # with a preview window

On the Pi with a CSI camera, `capture.source: auto` picks picamera2 by itself.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from inspector.config import load_config      # noqa: E402
from inspector.pipeline import Pipeline       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None,
                    help="device index, video file, or image folder")
    ap.add_argument("--loop", action="store_true", help="replay the source forever")
    ap.add_argument("--max-frames", type=int, default=0, help="0 means run until stopped")
    ap.add_argument("--actuator", default=None, choices=["mock", "serial", "gpio"],
                    help="override the configured backend")
    ap.add_argument("--show", action="store_true", help="open a preview window")
    ap.add_argument("--quiet", action="store_true", help="only log failures")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)

    if args.actuator:
        import dataclasses
        cfg = dataclasses.replace(
            cfg, actuator=dataclasses.replace(cfg.actuator, backend=args.actuator)
        )

    if args.show:
        import cv2

    last_report = time.perf_counter()

    def on_result(verdict, detections, annotated):
        nonlocal last_report
        if verdict.is_fail or not args.quiet:
            level = logging.WARNING if verdict.is_fail else logging.INFO
            logging.log(level, "%-4s  %s", verdict.verdict, verdict.reason)

        if args.show:
            import cv2
            cv2.imshow("inspection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        now = time.perf_counter()
        if now - last_report >= 5.0:
            s = pipe.stats
            logging.info("--- %d frames, %.1f FPS, %d pass / %d fail, %d ejected",
                         s.frames, s.fps, s.passes, s.fails, s.ejected)
            last_report = now

    with Pipeline(cfg, source_override=args.source, loop=args.loop) as pipe:
        stats = pipe.run(max_frames=args.max_frames, on_result=on_result)

        if args.show:
            import cv2
            cv2.destroyAllWindows()

        print()
        print(f"{'frames':<16}{stats.frames}")
        print(f"{'pass':<16}{stats.passes}")
        print(f"{'fail':<16}{stats.fails}")
        print(f"{'ejected':<16}{stats.ejected}")
        print(f"{'anomaly calls':<16}{stats.anomaly_calls}")
        print(f"{'throughput':<16}{stats.fps:.1f} FPS")
        print(f"\nresults in {cfg.resolve(cfg.storage.db)}")
        print("dashboard: uv run python dashboard/app.py")


if __name__ == "__main__":
    main()
