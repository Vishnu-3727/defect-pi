"""Flask dashboard: live view, recent results, and running statistics.

Architecture, and the one thing that matters about it:

    [ pipeline thread ]  ---- writes latest JPEG ---->  [ Flask thread ]
            |                                                  |
            +------------ writes -----> [ SQLite ] <--- reads --+

The inspection loop runs in its own thread and never waits for a browser. The
MJPEG stream reads whatever the newest frame happens to be. If the browser is
slow, or twelve people open the page, the stream drops frames -- the inspection
does not slow down.

Getting this wrong is the classic mistake: generating frames inside the Flask
request handler couples inspection throughput to network throughput, and the
conveyor then runs at the speed of someone's WiFi.

Why every rate is windowed
--------------------------
Rates on this page are reported over a rolling window, not since the beginning
of time. A cumulative reject rate is the wrong statistic for a line: it is
dominated by whatever happened first and it can never recover. After replay runs
this database read 91.7% rejected, which described a fixture folder full of
defect samples rather than anything a machine was doing. The all-time totals are
still shown, clearly labelled, because "how many parts has this station handled"
is a fair question with a cumulative answer.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from flask import (Flask, Response, abort, jsonify, render_template,
                   request, send_file)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from inspector.config import load_config          # noqa: E402
from inspector.pipeline import Pipeline           # noqa: E402

log = logging.getLogger(__name__)

# Selectable windows, in seconds. 0 means all time.
WINDOWS = (60, 300, 900, 3600, 0)
DEFAULT_WINDOW = 300


def _window_arg() -> int:
    """The requested window in seconds, clamped to the offered set.

    Validated rather than trusted: this value becomes a SQL timestamp bound,
    and an arbitrary string from a query parameter has no business going there.
    """
    try:
        w = int(request.args.get("window", DEFAULT_WINDOW))
    except (TypeError, ValueError):
        return DEFAULT_WINDOW
    return w if w in WINDOWS else DEFAULT_WINDOW


def read_soc_temp_c() -> float | None:
    """SoC temperature in Celsius, or None where that file does not exist.

    sysfs rather than `vcgencmd measure_temp` because it needs no subprocess
    and no video group membership. The kernel reports millidegrees.
    """
    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        return round(int(raw) / 1000.0, 1)
    except (OSError, ValueError):
        return None


def read_throttled() -> dict | None:
    """Decoded Pi throttling flags, or None when vcgencmd is unavailable.

    reports/benchmark.md argues that an FPS figure from a throttling SoC is not
    reproducible, so the same state is worth showing live. The low bits are
    happening right now; bits 16 and up are sticky "has occurred since boot"
    flags, which is why a perfectly healthy Pi commonly reads 0x50000.
    """
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=2)
        if out.returncode != 0:
            return None
        value = int(out.stdout.strip().split("=")[1], 16)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    return {
        "raw": f"0x{value:x}",
        "now": {
            "under_voltage": bool(value & 0x1),
            "arm_capped": bool(value & 0x2),
            "throttled": bool(value & 0x4),
            "soft_temp_limit": bool(value & 0x8),
        },
        "since_boot": {
            "under_voltage": bool(value & 0x10000),
            "arm_capped": bool(value & 0x20000),
            "throttled": bool(value & 0x40000),
            "soft_temp_limit": bool(value & 0x80000),
        },
    }


def create_app(pipeline: Pipeline) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html", classes=pipeline.cfg.classes,
                               windows=WINDOWS, default_window=DEFAULT_WINDOW)

    @app.route("/stream")
    def stream():
        def generate():
            # Cap the stream rate independently of the inspection rate. The eye
            # cannot use 50 FPS and the network should not carry it.
            interval = 1 / 15
            while True:
                frame = pipeline.latest_jpeg()
                if frame:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(frame)).encode() +
                           b"\r\n\r\n" + frame + b"\r\n")
                time.sleep(interval)

        return Response(generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/stats")
    def stats():
        s = pipeline.stats
        window = _window_arg()
        since = (time.time() - window) if window else None

        win = pipeline.storage.summary(since=since)
        alltime = pipeline.storage.summary()
        return jsonify({
            "live": {
                "frames": s.frames,
                "passes": s.passes,
                "fails": s.fails,
                "ejected": s.ejected,
                "anomaly_calls": s.anomaly_calls,
                "fps": round(s.fps, 1),
                "latency_ms": round(s.latency_ms, 1),
            },
            "window": {
                "seconds": window,
                "total": win["total"],
                "passes": win["passes"],
                "fails": win["fails"],
                "ejected": win["ejected"],
                "reject_rate": round(win["reject_rate"] * 100, 1),
                "avg_latency_ms": round(win["avg_latency_ms"], 1),
                # p95 over the window only: sorting the whole table by a column
                # with no index costs more than the number is worth.
                "p95_latency_ms": round(
                    pipeline.storage.latency_percentile(0.95, since=since), 1),
                "by_class": win["by_class"],
            },
            "database": {
                "total": alltime["total"],
                "passes": alltime["passes"],
                "fails": alltime["fails"],
                "ejected": alltime["ejected"],
                "reject_rate": round(alltime["reject_rate"] * 100, 1),
                "avg_latency_ms": round(alltime["avg_latency_ms"], 1),
                "by_class": alltime["by_class"],
            },
            "config": {
                "source": pipeline.source.name,
                "actuator": pipeline.actuator.name,
                "anomaly": pipeline.anomaly is not None,
                "conf_threshold": pipeline.cfg.detector.conf_threshold,
            },
        })

    @app.route("/api/series")
    def series():
        """Throughput and rejects per bucket over the window, for the chart."""
        window = _window_arg() or 3600
        # Aim for ~60 points whatever the window, so the chart stays readable
        # and the payload stays small.
        bucket = max(1.0, window / 60)
        return jsonify({
            "window": window,
            "bucket": bucket,
            "points": pipeline.storage.timeseries(time.time() - window, bucket),
        })

    @app.route("/api/health")
    def health():
        """Is the pipeline alive, and is the SoC in a state we can trust?"""
        s = pipeline.stats
        age = (time.time() - pipeline.last_frame_at
               if pipeline.last_frame_at else None)
        return jsonify({
            "running": pipeline.is_running(),
            # A thread that is alive but has produced no frame for seconds is a
            # wedged pipeline. The page should say so rather than show a frozen
            # image beside confidently stale counters.
            "last_frame_age_s": round(age, 1) if age is not None else None,
            "frames": s.frames,
            "soc_temp_c": read_soc_temp_c(),
            "throttled": read_throttled(),
            "db_bytes": (pipeline.storage.path.stat().st_size
                         if pipeline.storage.path.exists() else 0),
        })

    @app.route("/api/recent")
    def recent():
        return jsonify(pipeline.storage.recent(50))

    @app.route("/failure/<int:inspection_id>")
    def failure_image(inspection_id: int):
        """The saved frame for one inspection.

        image_path comes from our own database, but it still reaches the
        filesystem, so it is resolved and confined to the repository before
        anything is opened. A row naming /etc/shadow gets a 404, not a file.
        """
        rows = [r for r in pipeline.storage.recent(200)
                if r["id"] == inspection_id]
        if not rows or not rows[0].get("image_path"):
            abort(404)

        path = Path(rows[0]["image_path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        try:
            path = path.resolve(strict=True)
            path.relative_to(REPO_ROOT.resolve())
        except (OSError, ValueError):
            abort(404)
        return send_file(path, mimetype="image/jpeg")

    @app.route("/example/<name>")
    def example_image(name: str):
        """One example thumbnail per defect class, from the fixture folder."""
        allowed = {
            "crazing_101.jpg",
            "inclusion_121.jpg",
            "patches_135.jpg",
            "pitted_surface_123.jpg",
            "rolled-in_scale_102.jpg",
            "scratches_122.jpg",
        }
        if name not in allowed:
            abort(404)
        return send_file(REPO_ROOT / "bench_fixtures" / "images" / name,
                         mimetype="image/jpeg")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None,
                    help="device index, video file, or folder; overrides the config")
    ap.add_argument("--loop", action="store_true", help="replay the source forever")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)

    pipeline = Pipeline(cfg, source_override=args.source, loop=args.loop)
    pipeline.run_in_background()

    app = create_app(pipeline)
    log.info("dashboard on http://%s:%d", cfg.dashboard.host, cfg.dashboard.port)
    try:
        # threaded=True so the MJPEG stream, which never returns, does not block
        # the stats endpoints. use_reloader=False because the reloader would
        # start a second process and a second inspection pipeline.
        app.run(host=cfg.dashboard.host, port=cfg.dashboard.port,
                threaded=True, use_reloader=False)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
