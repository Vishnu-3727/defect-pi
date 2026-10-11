"""PatchCore at runtime: how unlike normal steel is this frame?

Runs on ONNX Runtime and NumPy only, like everything else that has to survive
on the Pi. The bank and the backbone were built once on the laptop by
scripts/build_patchcore.py.

What this adds over the detector: YOLOv8 can only report the six defect classes
it was trained on. A seventh kind of damage -- a new alloy flaw, a handling
dent, an oil smear -- produces no detection at all, and the part passes.
PatchCore never learned defect classes, so it has no such blind spot. What it
cannot do is say which defect it found. The two are complementary, which is why
decision.py fuses them with OR.

The cost is real: a backbone forward pass plus a nearest-neighbour search
against 2000 vectors, per frame. So it does not run per frame. It runs only
when the detector is unsure, and never more than a few times a second.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AnomalyDetector:
    def __init__(self, backbone_path: str | Path, bank_path: str | Path,
                 imgsz: int = 256, num_threads: int = 4,
                 max_calls_per_second: float = 3.0):
        import onnxruntime as ort

        backbone_path, bank_path = Path(backbone_path), Path(bank_path)
        for p in (backbone_path, bank_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Run scripts/build_patchcore.py first."
                )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(backbone_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        data = np.load(bank_path)
        self.bank = data["bank"].astype(np.float32)
        projection = data["projection"]
        self.projection = projection.astype(np.float32) if projection.size else None
        self.imgsz = int(data["imgsz"]) if "imgsz" in data else imgsz

        # ||b||^2 precomputed once. The squared-distance expansion below needs
        # it every call, and it never changes.
        self.bank_sq = np.einsum("ij,ij->i", self.bank, self.bank)

        self.min_interval = 1.0 / max_calls_per_second if max_calls_per_second > 0 else 0.0
        self._last_call = 0.0
        self.calls = 0
        self.skipped = 0
        self.last_latency_ms = 0.0

    def _embed(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        x = resized[:, :, ::-1].astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1))[np.newaxis, ...]

        feats = self.session.run(None, {self.input_name: x})[0]
        patches = feats[0].reshape(feats.shape[1], -1).T     # (n_patches, channels)
        if self.projection is not None:
            patches = patches @ self.projection
        return np.ascontiguousarray(patches, dtype=np.float32)

    def score(self, frame: np.ndarray) -> float:
        """Anomaly score: the distance from the most unusual patch to the bank.

        Taking the maximum over patches, not the mean, is the point. A defect
        occupying 5% of the frame moves the mean by almost nothing but moves the
        maximum by a lot. Averaging would hide exactly what we are looking for.
        """
        start = time.perf_counter()
        patches = self._embed(frame)

        # ||a-b||^2 = ||a||^2 - 2ab + ||b||^2. Expanding it turns the whole
        # nearest-neighbour search into one matrix multiply, which BLAS does with
        # NEON on the Pi. Building an explicit (n_patches, bank) difference
        # tensor instead would allocate ~100 MB per frame and be far slower.
        patch_sq = np.einsum("ij,ij->i", patches, patches)
        d2 = patch_sq[:, None] - 2.0 * (patches @ self.bank.T) + self.bank_sq[None, :]
        nearest = np.sqrt(np.maximum(d2.min(axis=1), 0.0))

        self.calls += 1
        self.last_latency_ms = (time.perf_counter() - start) * 1e3
        return float(nearest.max())

    def maybe_score(self, frame: np.ndarray, detector_conf: float | None,
                    gate_below: float) -> float | None:
        """Run only when the detector is unsure, and not too often.

        Two gates, for two different reasons.

        The confidence gate is about value: when YOLOv8 has already reported a
        defect at 0.9, a second opinion changes no decision, so the compute is
        wasted. The frames worth a second opinion are the ones where it saw
        nothing, or saw something faintly.

        The rate gate is about the frame budget. Without it, a run of uncertain
        frames would call PatchCore on every one and the frame rate would
        collapse exactly when the line is producing marginal parts -- the worst
        possible time. Capping the rate means the slow path can never dominate.
        """
        if detector_conf is not None and detector_conf >= gate_below:
            return None

        now = time.perf_counter()
        if self.min_interval and (now - self._last_call) < self.min_interval:
            self.skipped += 1
            return None

        self._last_call = now
        return self.score(frame)


def build_anomaly_detector(cfg) -> AnomalyDetector | None:
    """Construct it if enabled and built; otherwise return None and say why.

    Returning None rather than raising keeps the system running with the
    detector alone, which is a degraded but useful state. A missing memory bank
    should not stop an inspection line.
    """
    if not cfg.anomaly.enabled:
        return None
    try:
        return AnomalyDetector(
            cfg.resolve("models/patchcore_backbone.onnx"),
            cfg.resolve(cfg.anomaly.bank),
            cfg.anomaly.imgsz,
            cfg.detector.num_threads,
            cfg.anomaly.max_calls_per_second,
        )
    except FileNotFoundError as exc:
        log.warning("anomaly module disabled: %s", exc)
        return None


def demo() -> None:
    """Self-check, and a calibration report for the score threshold.

    Scores defect-free crops against defect crops from the test split. If the
    bank is any good the two distributions separate, and the printed suggestion
    is a defensible starting value for anomaly.score_threshold.
    """
    from .config import load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()

    det = build_anomaly_detector(cfg)
    if det is None:
        print("skipped: run scripts/build_patchcore.py first")
        return

    images_dir = cfg.resolve("data/yolo/test/images")
    labels_dir = cfg.resolve("data/yolo/test/labels")
    paths = sorted(images_dir.glob("*.jpg"))
    assert paths, "no test images"

    defect_scores, clean_scores = [], []
    for path in paths[::4][:60]:
        frame = cv2.imread(str(path))
        h, w = frame.shape[:2]
        label = labels_dir / f"{path.stem}.txt"
        boxes = []
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5:
                    boxes.append([float(v) for v in p[1:]])
        if not boxes:
            continue

        # A crop centred on the largest annotated defect.
        cx, cy, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
        x1, x2 = int(max(0, (cx - bw / 2) * w)), int(min(w, (cx + bw / 2) * w))
        y1, y2 = int(max(0, (cy - bh / 2) * h)), int(min(h, (cy + bh / 2) * h))
        if x2 - x1 > 16 and y2 - y1 > 16:
            defect_scores.append(det.score(frame[y1:y2, x1:x2]))

        # A crop from the corner least covered by any box, as a normal sample.
        corner = min(
            [(0, 0), (0, w // 2), (h // 2, 0), (h // 2, w // 2)],
            key=lambda yx: sum(
                1 for b in boxes
                if abs(b[0] * w - (yx[1] + w // 4)) < b[2] * w / 2 + w // 8
                and abs(b[1] * h - (yx[0] + h // 4)) < b[3] * h / 2 + h // 8
            ),
        )
        crop = frame[corner[0]:corner[0] + h // 2, corner[1]:corner[1] + w // 2]
        if crop.size:
            clean_scores.append(det.score(crop))

    assert defect_scores and clean_scores, "calibration produced no samples"
    d, c = np.array(defect_scores), np.array(clean_scores)

    print(f"\n{'region':<12}{'n':>5}{'median':>10}{'p90':>10}{'max':>10}")
    print("-" * 47)
    print(f"{'normal':<12}{len(c):>5}{np.median(c):>10.3f}"
          f"{np.percentile(c, 90):>10.3f}{c.max():>10.3f}")
    print(f"{'defect':<12}{len(d):>5}{np.median(d):>10.3f}"
          f"{np.percentile(d, 90):>10.3f}{d.max():>10.3f}")
    print("-" * 47)

    suggested = float(np.percentile(c, 95))
    separation = float(np.median(d) - np.median(c))
    print(f"suggested anomaly.score_threshold: {suggested:.2f} "
          f"(95th percentile of normal regions)")
    print(f"median separation defect - normal: {separation:+.3f}")
    if separation <= 0:
        print("\nWARNING: defect regions do not score higher than normal ones.")
        print("The bank is contaminated or the backbone features are unsuitable.")
        print("Do not trust this module until that is resolved -- see docs/14.")

    assert det.last_latency_ms > 0
    print(f"\nanomaly self-check passed: {det.calls} scorings, "
          f"{det.last_latency_ms:.1f} ms for the last one")


if __name__ == "__main__":
    demo()
