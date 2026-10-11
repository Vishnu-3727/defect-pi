"""YOLOv8 inference through ONNX Runtime. No PyTorch, no ultralytics.

This is the file that has to run on the Pi, so it depends on onnxruntime,
numpy and opencv only. Everything ultralytics did for us at training time --
decoding the raw output tensor, filtering by confidence, suppressing duplicate
boxes, mapping coordinates back to the original frame -- has to happen here
explicitly. See docs/08_yolo_v1_to_v8.md.

Output tensor layout for YOLOv8 at 320x320 with 6 classes:

    (1, 10, 2100)
     |   |    +--- 2100 candidate positions (anchor points across 3 scales:
     |   |         40x40 + 20x20 + 10x10 = 1600 + 400 + 100)
     |   +-------- 4 box values (cx, cy, w, h in letterboxed pixels)
     |             followed by 6 class scores, already sigmoid-activated
     +------------ batch of 1

Note there is no objectness score. YOLOv8 is anchor-free and dropped it; the
class score alone is the confidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .preprocess import preprocess, unletterbox_boxes


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in original frame pixels

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Timing:
    """Per-stage milliseconds for the last frame. Feeds bench.py and the dashboard."""
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


class Detector:
    def __init__(self, model_path: str | Path, class_names: list[str], imgsz: int = 320,
                 conf_threshold: float = 0.35, iou_threshold: float = 0.45,
                 num_threads: int = 4):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run scripts/export_onnx.py --int8 first."
            )

        opts = ort.SessionOptions()
        # One thread per physical core. The Pi 5 has four Cortex-A76 cores, so
        # more threads than that buys context switching, not throughput.
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = list(class_names)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.timing = Timing()

        expected = self.session.get_inputs()[0].shape
        # A static export pins the spatial dims; catch a mismatch here rather
        # than as a shape error 2000 frames into a demo.
        if isinstance(expected[2], int) and expected[2] != imgsz:
            raise ValueError(
                f"model expects input {expected[2]}x{expected[3]} but imgsz={imgsz}. "
                f"Re-export with --imgsz {expected[2]} or fix configs/default.yaml."
            )

        n_out = self.session.get_outputs()[0].shape[1]
        if isinstance(n_out, int) and n_out != 4 + len(self.class_names):
            raise ValueError(
                f"model outputs {n_out} rows, which implies {n_out - 4} classes, "
                f"but config lists {len(self.class_names)}: {self.class_names}"
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        tensor, scale, pad = preprocess(frame, self.imgsz)

        t1 = time.perf_counter()
        raw = self.session.run(None, {self.input_name: tensor})[0]

        t2 = time.perf_counter()
        detections = self._decode(raw, scale, pad, frame.shape)

        t3 = time.perf_counter()
        self.timing = Timing((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
        return detections

    def _decode(self, raw: np.ndarray, scale: float, pad: tuple[int, int],
                orig_shape: tuple[int, ...]) -> list[Detection]:
        # (1, 4+nc, N) -> (N, 4+nc). Transposing first means every filter below
        # is a row operation on a contiguous-ish array instead of a strided one.
        preds = raw[0].T

        boxes_cxcywh = preds[:, :4]
        scores_all = preds[:, 4:]

        # One class per box: the highest-scoring one. Doing the argmax before
        # thresholding costs one pass over 2100 rows and lets everything after
        # this work on a handful of survivors.
        class_ids = scores_all.argmax(axis=1)
        confidences = scores_all[np.arange(scores_all.shape[0]), class_ids]

        # The INT8 model can emit values slightly above 1.0 -- measured up to
        # 1.28. The score head ends in a sigmoid, whose output is bounded in
        # [0, 1], but quantising that tensor rounds it to a grid whose top step
        # lands above 1. Clipping restores the meaning of the number without
        # changing which boxes survive: the ordering is untouched, and the
        # threshold sits far below 1. The FP32 model never trips this.
        np.clip(confidences, 0.0, 1.0, out=confidences)

        keep = confidences >= self.conf_threshold
        if not keep.any():
            return []

        boxes_cxcywh = boxes_cxcywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # cxcywh -> xyxy, still in letterboxed coordinates.
        xy = boxes_cxcywh[:, :2]
        wh = boxes_cxcywh[:, 2:4]
        boxes_xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)

        # NMS in letterboxed space, before the coordinate mapping: IoU is
        # scale-invariant, so the result is identical and we map fewer boxes.
        # cv2's implementation is compiled C++; a NumPy one would be slower and
        # would be one more thing to get wrong.
        xywh = np.concatenate([boxes_xyxy[:, :2],
                               boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]], axis=1)
        indices = cv2.dnn.NMSBoxesBatched(
            xywh.tolist(), confidences.tolist(), class_ids.tolist(),
            self.conf_threshold, self.iou_threshold,
        )
        if len(indices) == 0:
            return []
        indices = np.array(indices).flatten()

        mapped = unletterbox_boxes(boxes_xyxy[indices], scale, pad, orig_shape)

        return [
            Detection(
                class_id=int(class_ids[i]),
                class_name=self.class_names[int(class_ids[i])],
                confidence=float(confidences[i]),
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            )
            for i, b in zip(indices, mapped)
        ]


def draw(frame: np.ndarray, detections: list[Detection],
         color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """Annotate a copy of the frame. Never mutates the original."""
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def demo() -> None:
    """Self-check: the ONNX model loads and produces sane detections on real data."""
    from .config import load_config

    cfg = load_config()
    model = cfg.resolve(cfg.detector.model)
    if not model.exists():
        print(f"skipped: {model} not built yet")
        return

    det = Detector(model, cfg.classes, cfg.detector.imgsz,
                   cfg.detector.conf_threshold, cfg.detector.iou_threshold,
                   cfg.detector.num_threads)

    all_images = sorted((cfg.resolve("data/yolo/test/images")).glob("*.jpg"))
    assert all_images, "no test images; run scripts/prepare_dataset.py"

    # Sample per class, not the first 20 alphabetically. Sorting puts every
    # crazing image first, and crazing has the lowest recall of the six classes,
    # so an alphabetical sample can legitimately return zero detections and look
    # exactly like a broken decoder. It did, once.
    by_class: dict[str, list] = {}
    for p in all_images:
        by_class.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)
    images = [p for cls in sorted(by_class) for p in by_class[cls][:4]]

    total = 0
    for path in images:
        frame = cv2.imread(str(path))
        dets = det.detect(frame)
        total += len(dets)
        h, w = frame.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = d.box
            assert 0 <= x1 <= x2 <= w + 1, (path.name, d.box)
            assert 0 <= y1 <= y2 <= h + 1, (path.name, d.box)
            assert 0.0 <= d.confidence <= 1.0, d.confidence
            assert d.class_name in cfg.classes

    assert total > 0, f"{len(images)} test images produced zero detections; decode is wrong"
    print(f"detector self-check passed: {total} detections over {len(images)} images "
          f"({len(by_class)} classes), {det.timing.total_ms:.1f} ms on the last frame")


if __name__ == "__main__":
    demo()
