"""Prove the deployed pipeline agrees with the trained model.

Between `best.pt` and what runs on the Pi there are three chances to silently
break something: the ONNX export, the INT8 quantisation, and our own
hand-written decode in `detector.py`. None of them raise an error when wrong.
The model still runs, still emits boxes, and the boxes are subtly misplaced or
missing.

This script closes that gap two ways:

  1. Box-level parity -- run ultralytics on best.pt and our Detector on each
     ONNX file over the same images, then match the boxes. Catches decode bugs,
     letterbox mismatches and coordinate-mapping errors.
  2. mAP delta -- run the standard evaluation on both ONNX files. Quantifies
     what quantisation actually cost, instead of assuming it was free.

Run:  uv run scripts/parity.py
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def sample_images(folder: Path, count: int) -> list[Path]:
    paths = sorted(folder.glob("*.jpg"))
    by_class: dict[str, list[Path]] = {}
    for p in paths:
        by_class.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)
    per_class = max(1, count // max(1, len(by_class)))
    return [p for cls in sorted(by_class) for p in by_class[cls][:per_class]]


def reference_predictions(weights: Path, images: list[Path], imgsz: int, conf: float,
                          iou: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Ground truth for this comparison: what ultralytics itself predicts."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in images:
        r = model.predict(str(path), imgsz=imgsz, conf=conf, iou=iou,
                          device="cpu", verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        classes = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros(0, int)
        out[path.name] = (boxes, classes)
    return out


def compare(model_path: Path, cfg, images: list[Path], reference: dict,
            iou_match: float) -> dict:
    from inspector.detector import Detector

    det = Detector(model_path, cfg.classes, cfg.detector.imgsz,
                   cfg.detector.conf_threshold, cfg.detector.iou_threshold,
                   cfg.detector.num_threads)

    matched = ref_total = ours_total = wrong_class = 0
    ious: list[float] = []

    for path in images:
        frame = cv2.imread(str(path))
        dets = det.detect(frame)
        ours = np.array([d.box for d in dets]) if dets else np.zeros((0, 4))
        ours_cls = np.array([d.class_id for d in dets], int) if dets else np.zeros(0, int)

        ref_boxes, ref_cls = reference[path.name]
        ref_total += len(ref_boxes)
        ours_total += len(ours)

        if len(ref_boxes) == 0 or len(ours) == 0:
            continue

        m = iou_matrix(ref_boxes, ours)
        # Greedy one-to-one matching, highest IoU first.
        used: set[int] = set()
        for ref_i in np.argsort(-m.max(axis=1)):
            order = np.argsort(-m[ref_i])
            for ours_j in order:
                if ours_j in used or m[ref_i, ours_j] < iou_match:
                    continue
                used.add(int(ours_j))
                matched += 1
                ious.append(float(m[ref_i, ours_j]))
                if ref_cls[ref_i] != ours_cls[ours_j]:
                    wrong_class += 1
                break

    return {
        "model": model_path.name,
        "ref_boxes": ref_total,
        "our_boxes": ours_total,
        "matched": matched,
        "wrong_class": wrong_class,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "recall_vs_ref": matched / ref_total if ref_total else 0.0,
    }


def map_scores(model_path: Path, data: str, imgsz: int) -> tuple[float, float]:
    from ultralytics import YOLO

    m = YOLO(str(model_path), task="detect")
    r = m.val(data=data, split="test", imgsz=imgsz, device="cpu", plots=False,
              verbose=False, project=str(REPO_ROOT / "runs"),
              name=f"parity_{model_path.stem}", exist_ok=True)
    return float(r.box.map50), float(r.box.map)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(REPO_ROOT / "models" / "best.pt"))
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "yolo" / "data.yaml"))
    ap.add_argument("--images", type=int, default=60)
    ap.add_argument("--tail-conf", type=float, default=0.001,
                    help="second parity pass at the confidence floor mAP sweeps "
                         "from; the operating-threshold pass agrees closely even "
                         "when the tail does not -- see reports/imgsz-rect-trap.md")
    ap.add_argument("--iou-match", type=float, default=0.70,
                    help="IoU above which two boxes are considered the same detection")
    ap.add_argument("--skip-map", action="store_true", help="box parity only, no mAP run")
    args = ap.parse_args()

    from inspector.config import load_config

    cfg = load_config()
    images = sample_images(cfg.resolve("data/yolo/test/images"), args.images)
    print(f"parity over {len(images)} class-balanced test images\n")

    ref = reference_predictions(Path(args.weights), images, cfg.detector.imgsz,
                                cfg.detector.conf_threshold, cfg.detector.iou_threshold)

    rows = []
    for name in ("model.onnx", "model_int8.onnx"):
        path = REPO_ROOT / "models" / name
        if path.exists():
            rows.append(compare(path, cfg, images, ref, args.iou_match))

    # Second pass at the metric's confidence floor. A parity check at the
    # operating threshold licenses claims about the operating point and nothing
    # else: mAP integrates from 0.001, and an export can agree perfectly at 0.35
    # while diverging in the tail that the sweep counts.
    tail_ref = reference_predictions(Path(args.weights), images, cfg.detector.imgsz,
                                     args.tail_conf, cfg.detector.iou_threshold)
    # The config dataclasses are frozen, which is the right call for a config
    # object; build a modified copy rather than mutating one.
    tail_cfg = dataclasses.replace(
        cfg, detector=dataclasses.replace(cfg.detector,
                                          conf_threshold=args.tail_conf))
    tail_rows = []
    for name in ("model.onnx", "model_int8.onnx"):
        path = REPO_ROOT / "models" / name
        if path.exists():
            r = compare(path, tail_cfg, images, tail_ref, args.iou_match)
            r["model"] = r["model"] + f" @{args.tail_conf}"
            tail_rows.append(r)
    rows.extend(tail_rows)

    print(f"{'model':<20}{'ref':>6}{'ours':>6}{'matched':>9}{'recall':>9}"
          f"{'mean IoU':>10}{'cls err':>9}")
    print("-" * 69)
    for r in rows:
        print(f"{r['model']:<20}{r['ref_boxes']:>6}{r['our_boxes']:>6}{r['matched']:>9}"
              f"{r['recall_vs_ref']:>9.3f}{r['mean_iou']:>10.4f}{r['wrong_class']:>9}")
    print("-" * 69)
    print("'ref' is what ultralytics predicts from best.pt on the same images.")
    print(f"A box counts as matched at IoU >= {args.iou_match}.")

    if not args.skip_map:
        print("\nmAP on the test split (this takes a minute per model):\n")
        print(f"{'model':<20}{'mAP@50':>10}{'mAP@50-95':>12}")
        print("-" * 42)
        base = None
        for name in ("best.pt", "model.onnx", "model_int8.onnx"):
            path = REPO_ROOT / "models" / name
            if not path.exists():
                continue
            m50, m = map_scores(path, args.data, cfg.detector.imgsz)
            base = base or m50
            print(f"{name:<20}{m50:>10.4f}{m:>12.4f}")
        print("-" * 42)
        print("A drop of more than about 3 points of mAP@50 from quantisation is")
        print("worth investigating before shipping. See docs/13_cpu_optimization_pi5.md.")


if __name__ == "__main__":
    main()
