"""Frame sources, behind one interface.

The pipeline must not know or care whether frames come from a CSI camera on a
Pi, a USB webcam on a laptop, a recorded video, or a folder of stills. Every
`if source == ...` lives here and nowhere else, which is what lets the same
pipeline run on a demo video during development and on real hardware later with
a one-line config change.

On Raspberry Pi OS Bookworm the legacy camera stack was removed, so
`cv2.VideoCapture` cannot open a Camera Module 3. That needs picamera2. Code
written and tested against a laptop webcam therefore fails on the Pi with
"camera not found", which is a deployment surprise this module exists to avoid.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class FrameSource:
    """Base interface. Iterating yields BGR frames until the source is done."""

    name = "base"

    def __iter__(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoCaptureSource(FrameSource):
    """USB webcam, video file, or any URL OpenCV understands."""

    def __init__(self, target, width: int = 0, height: int = 0, fps: int = 0,
                 loop: bool = False):
        self.cap = cv2.VideoCapture(target)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open capture source {target!r}. For a device index, "
                f"check the camera is connected and not held by another process. "
                f"On a Pi 5 with a CSI camera module, use source: picamera instead."
            )
        # These are requests, not commands. The driver may ignore any of them,
        # so the accepted values are read back below rather than assumed.
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.actual = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.loop = loop
        self.name = f"capture({target}) {self.actual[0]}x{self.actual[1]}"

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                # End of file, or the camera went away. Looping is for demos:
                # a short clip can drive the pipeline indefinitely.
                if self.loop and self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    continue
                return
            yield frame

    def close(self) -> None:
        self.cap.release()


class FolderSource(FrameSource):
    """A directory of stills, replayed at a fixed rate.

    Useful without hardware: it turns the test split into a synthetic conveyor.
    """

    def __init__(self, folder: Path, fps: float = 2.0, loop: bool = False):
        self.paths = sorted(p for p in Path(folder).iterdir()
                            if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.paths:
            raise RuntimeError(f"no images in {folder}")
        self.interval = 1.0 / fps if fps > 0 else 0.0
        self.loop = loop
        self.name = f"folder({folder}) {len(self.paths)} images @ {fps:g} fps"

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            for path in self.paths:
                frame = cv2.imread(str(path))
                if frame is None:
                    continue
                start = time.perf_counter()
                yield frame
                # Sleep for whatever is left of the frame interval after the
                # consumer finished, so a slow pipeline does not get slower.
                remaining = self.interval - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
            if not self.loop:
                return


class PiCameraSource(FrameSource):
    """Raspberry Pi CSI camera through picamera2.

    picamera2 delivers RGB, OpenCV expects BGR, so the conversion happens here
    and the rest of the pipeline stays in one colour convention.
    """

    def __init__(self, width: int = 640, height: int = 480):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is not installed. It ships with Raspberry Pi OS; on "
                "another platform use capture.source: device instead."
            ) from exc

        self.cam = Picamera2()
        config = self.cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.cam.configure(config)
        self.cam.start()
        time.sleep(0.5)          # sensor auto-exposure needs a moment to settle
        self.name = f"picamera2 {width}x{height}"

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            yield cv2.cvtColor(self.cam.capture_array(), cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self.cam.stop()


def _picamera_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("picamera2") is not None


def open_source(cfg, override: str | None = None, loop: bool = False) -> FrameSource:
    """Build the configured frame source.

    `override` accepts what a user would type on the command line: a device
    index, a path to a video or a folder. It wins over the config file so the
    same config works for both a demo and live capture.
    """
    capture = cfg.capture

    if override is not None:
        if override.isdigit():
            return VideoCaptureSource(int(override), capture.width, capture.height,
                                      capture.fps, loop)
        path = Path(override)
        if path.is_dir():
            # Replay, not capture: paced for a person, not a sensor.
            return FolderSource(path, capture.replay_fps, loop)
        if not path.exists():
            raise RuntimeError(f"source {override!r} is neither a device index, "
                               f"a folder, nor an existing file")
        return VideoCaptureSource(str(path), capture.width, capture.height,
                                  capture.replay_fps, loop)

    source = capture.source
    if source == "auto":
        # Prefer the CSI camera when the library is present, which is the case
        # on a Pi and almost nowhere else.
        source = "picamera" if _picamera_available() else "device"

    if source == "picamera":
        return PiCameraSource(capture.width, capture.height)
    if source == "device":
        return VideoCaptureSource(capture.device_index, capture.width,
                                  capture.height, capture.fps, loop)
    if source == "folder":
        return FolderSource(cfg.resolve(capture.path), capture.replay_fps, loop)
    if source == "video":
        return VideoCaptureSource(str(cfg.resolve(capture.path)), capture.width,
                                  capture.height, capture.replay_fps, loop)
    raise ValueError(f"unhandled capture source {source!r}")


def demo() -> None:
    """Self-check: the folder source replays real images at the requested rate."""
    from .config import load_config

    cfg = load_config()
    folder = cfg.resolve("bench_fixtures/images")
    if not folder.is_dir():
        folder = cfg.resolve("data/yolo/test/images")
    assert folder.is_dir(), "no images available; run scripts/prepare_dataset.py"

    src = FolderSource(folder, fps=50)
    frames = []
    start = time.perf_counter()
    for frame in src:
        frames.append(frame)
        if len(frames) == 10:
            break
    elapsed = time.perf_counter() - start

    assert len(frames) == 10
    assert all(f.ndim == 3 and f.shape[2] == 3 for f in frames), "expected BGR frames"
    # 10 frames at 50 fps is 0.2 s of pacing; allow generous slack for slow IO.
    assert elapsed >= 0.15, f"pacing ignored: 10 frames in {elapsed:.3f}s"

    # A folder with no images must fail loudly, not yield nothing forever.
    try:
        FolderSource(cfg.resolve("configs"))
        raise AssertionError("a folder with no images should raise")
    except RuntimeError:
        pass

    src.close()
    print(f"capture self-check passed: 10 frames in {elapsed:.2f}s, "
          f"shape {frames[0].shape}")


if __name__ == "__main__":
    demo()
