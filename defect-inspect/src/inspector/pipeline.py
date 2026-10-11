"""The inspection loop: everything above, wired together.

    frame -> detector -> gate -> anomaly -> decision -> storage -> actuator
                                                            |
                                                            +-> latest frame
                                                                for the dashboard

Design rules this file obeys, and why:

  * It owns no algorithm. Every decision lives in the module responsible for it,
    so this file stays readable and the parts stay testable alone.
  * Nothing here knows which actuator backend is active, or where frames come
    from. Both are chosen in config and resolved before the loop starts.
  * A failure in storage or the ejector logs and continues. The inspection line
    stopping is worse than a lost record.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .actuator import build_actuator
from .anomaly import build_anomaly_detector
from .capture import open_source
from .decision import decide
from .detector import Detector, draw
from .storage import InspectionRecord, Storage

log = logging.getLogger(__name__)


@dataclass
class Stats:
    frames: int = 0
    passes: int = 0
    fails: int = 0
    ejected: int = 0
    anomaly_calls: int = 0
    started: float = field(default_factory=time.perf_counter)
    latency_ms: float = 0.0

    @property
    def fps(self) -> float:
        elapsed = time.perf_counter() - self.started
        return self.frames / elapsed if elapsed > 0 else 0.0


class Pipeline:
    def __init__(self, cfg, source_override: str | None = None, loop: bool = False,
                 annotate: bool = True):
        self.cfg = cfg
        self.annotate = annotate

        self.detector = Detector(
            cfg.resolve(cfg.detector.model), cfg.classes, cfg.detector.imgsz,
            cfg.detector.conf_threshold, cfg.detector.iou_threshold,
            cfg.detector.num_threads,
        )
        self.anomaly = build_anomaly_detector(cfg)
        self.actuator = build_actuator(cfg)
        self.storage = Storage(cfg.resolve(cfg.storage.db))
        self.source = open_source(cfg, source_override, loop)

        self.failure_dir = cfg.resolve(cfg.storage.image_dir)
        if cfg.storage.save_fail_images:
            self.failure_dir.mkdir(parents=True, exist_ok=True)

        self.stats = Stats()
        # Wall-clock time of the last processed frame, for the dashboard's
        # liveness check. Wall clock, not perf_counter, because it is compared
        # against time.time() in another thread and shown to a human.
        self.last_frame_at: float | None = None
        self._thread: threading.Thread | None = None
        # The dashboard reads these from another thread. Guarded by the lock
        # because a half-written JPEG buffer renders as a broken image.
        self._latest_jpeg: bytes | None = None
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()

        log.info("source   : %s", self.source.name)
        log.info("detector : %s", Path(cfg.detector.model).name)
        log.info("anomaly  : %s", "enabled" if self.anomaly else "disabled")
        log.info("actuator : %s", self.actuator.name)

    def process(self, frame):
        """One frame, end to end. Returns (verdict, detections, annotated)."""
        start = time.perf_counter()

        detections = self.detector.detect(frame)
        best_conf = max((d.confidence for d in detections), default=None)

        score = None
        if self.anomaly is not None:
            score = self.anomaly.maybe_score(frame, best_conf,
                                             self.cfg.anomaly.gate_conf_below)
            if score is not None:
                self.stats.anomaly_calls += 1

        verdict = decide(detections, score, self.cfg.decision.min_defect_area_px,
                         self.cfg.anomaly.score_threshold)

        latency_ms = (time.perf_counter() - start) * 1e3
        annotated = draw(frame, detections) if self.annotate else frame

        image_path = None
        if verdict.is_fail and self.cfg.storage.save_fail_images:
            image_path = self._save_failure(annotated)

        ejected = False
        if verdict.is_fail:
            # The delay models the physical gap between the camera and the
            # ejector: the part photographed now reaches the solenoid later.
            # Zero on a bench, non-zero on a real belt, and calibrated by
            # measuring distance divided by belt speed.
            if self.cfg.actuator.eject_delay_s > 0:
                time.sleep(self.cfg.actuator.eject_delay_s)
            ejected = self.actuator.eject()

        self._record(verdict, latency_ms, ejected, image_path)

        self.stats.frames += 1
        self.stats.latency_ms = latency_ms
        self.last_frame_at = time.time()
        if verdict.is_fail:
            self.stats.fails += 1
            self.stats.ejected += int(ejected)
        else:
            self.stats.passes += 1

        with self._frame_lock:
            ok, buf = cv2.imencode(".jpg", annotated,
                                   [cv2.IMWRITE_JPEG_QUALITY,
                                    self.cfg.dashboard.jpeg_quality])
            if ok:
                self._latest_jpeg = buf.tobytes()

        return verdict, detections, annotated

    # ponytail: fixed cap, not a config knob. An unbounded writer filled this
    # directory with 472,973 JPEGs (9.3 GB) during one unattended systemd run.
    # A ring buffer of recent evidence is what a demo actually needs; a real
    # line archives off-box instead. Raise MAX_FAILURE_IMAGES if a longer
    # forensic window is wanted.
    MAX_FAILURE_IMAGES = 2000
    _PRUNE_EVERY = 200

    def _prune_failures(self) -> None:
        """Keep only the newest MAX_FAILURE_IMAGES JPEGs in failure_dir."""
        try:
            files = sorted(self.failure_dir.glob("*.jpg"),
                           key=lambda f: f.stat().st_mtime, reverse=True)
            for stale in files[self.MAX_FAILURE_IMAGES:]:
                stale.unlink(missing_ok=True)
        except Exception as exc:
            log.warning("failure-image prune failed: %s", exc)

    def _save_failure(self, annotated) -> str | None:
        try:
            name = f"fail_{time.strftime('%Y%m%d_%H%M%S')}_{self.stats.frames:06d}.jpg"
            path = self.failure_dir / name
            cv2.imwrite(str(path), annotated)
            self._saved_images = getattr(self, "_saved_images", 0) + 1
            if self._saved_images % self._PRUNE_EVERY == 0:
                self._prune_failures()
            # Store a repo-relative path when the directory is inside the repo,
            # so the database stays portable between the laptop and the Pi.
            # Path.relative_to raises for anything outside, and an absolute path
            # is a perfectly good fallback -- losing the record over a cosmetic
            # preference would be absurd.
            try:
                return str(path.relative_to(self.cfg.resolve(".")))
            except ValueError:
                return str(path)
        except Exception as exc:
            log.error("could not save failure image: %s", exc)
            return None

    def _record(self, verdict, latency_ms, ejected, image_path) -> None:
        try:
            self.storage.insert(InspectionRecord(
                ts=time.time(), verdict=verdict.verdict, reason=verdict.reason,
                defect_class=verdict.defect_class, confidence=verdict.confidence,
                n_defects=verdict.n_defects, anomaly_score=verdict.anomaly_score,
                latency_ms=latency_ms, ejected=ejected, image_path=image_path,
            ))
        except Exception as exc:
            # Losing a row must not stop the line. It is logged loudly instead.
            log.error("could not record inspection: %s", exc)

    def latest_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def run(self, max_frames: int = 0, on_result=None) -> Stats:
        self.stats = Stats()
        try:
            for frame in self.source:
                if self._stop.is_set():
                    break
                verdict, detections, annotated = self.process(frame)
                if on_result is not None:
                    on_result(verdict, detections, annotated)
                if max_frames and self.stats.frames >= max_frames:
                    break
        except KeyboardInterrupt:
            log.info("interrupted")
        return self.stats

    def run_in_background(self, max_frames: int = 0) -> threading.Thread:
        """Run the loop in a daemon thread, for the dashboard to serve alongside."""
        thread = threading.Thread(target=self.run, kwargs={"max_frames": max_frames},
                                  daemon=True, name="inspection")
        thread.start()
        self._thread = thread
        return thread

    def is_running(self) -> bool:
        """Is the background inspection thread alive?

        False also when nothing was ever started, which is the honest answer
        for a dashboard asking whether inspection is happening.
        """
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()
        self.source.close()
        self.actuator.close()
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def demo() -> None:
    """Self-check: a full run over real images with a mock ejector.

    This is the end-to-end test. It exercises capture, detection, gating,
    decision, database write and ejection in one pass, and asserts the numbers
    are consistent with each other rather than merely non-zero.
    """
    import dataclasses
    import tempfile

    from .config import load_config

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    cfg = load_config()

    images = cfg.resolve("bench_fixtures/images")
    if not images.is_dir():
        images = cfg.resolve("data/yolo/test/images")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(
            cfg,
            storage=dataclasses.replace(cfg.storage, db=str(Path(tmp) / "t.db"),
                                        image_dir=str(Path(tmp) / "fail")),
            actuator=dataclasses.replace(cfg.actuator, backend="mock"),
        )
        with Pipeline(cfg, source_override=str(images)) as pipe:
            stats = pipe.run(max_frames=40)

            assert stats.frames == 40, stats.frames
            assert stats.passes + stats.fails == stats.frames
            assert stats.fails > 0, "40 defect images produced no failures"
            assert stats.ejected == stats.fails, \
                f"mock ejector should fire on every fail: {stats.ejected} vs {stats.fails}"
            assert pipe.latest_jpeg(), "no frame was published for the dashboard"

            summary = pipe.storage.summary()
            assert summary["total"] == 40, summary
            assert summary["fails"] == stats.fails, summary
            assert summary["avg_latency_ms"] > 0

            saved = list(Path(tmp).joinpath("fail").glob("*.jpg"))
            assert len(saved) == stats.fails, \
                f"{stats.fails} failures but {len(saved)} images saved"

            # Every failure row must carry a usable path, not None. This caught
            # a real bug: Path.relative_to raises when the failure directory
            # sits outside the repo, and the image was saved but unreferenced.
            failures = [r for r in pipe.storage.recent(100) if r["verdict"] == "FAIL"]
            assert failures, "no FAIL rows recorded"
            missing = [r["id"] for r in failures if not r["image_path"]]
            assert not missing, f"FAIL rows with no image_path: {missing}"

            print(f"pipeline self-check passed: {stats.frames} frames, "
                  f"{stats.passes} pass / {stats.fails} fail, "
                  f"{stats.ejected} ejected, {stats.anomaly_calls} anomaly calls, "
                  f"{stats.fps:.1f} FPS end to end")


if __name__ == "__main__":
    demo()
