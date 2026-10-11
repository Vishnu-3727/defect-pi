"""Escapes and false-positive proxy as a function of the confidence threshold.

    python scripts/threshold_study.py [--split test] [--imgsz 320]

WHY THIS EXISTS
`detector.conf_threshold` is applied in detector.py as a hard cut on raw model
output, before decide() sees anything. It therefore sets two unrelated policies
at once: which boxes are shown (precision) and whether the part is ejected
(recall). Those want opposite values. This script measures the cost of tying
them together, and it is the evidence behind reports/threshold-decoupling.md.

WHAT AN "ESCAPE" IS
Every NEU-DET image contains a defect, so a frame with no surviving detection is
a defective part the system passes. That is the number an inspection line cares
about and it does not appear in mAP, which is swept at conf 0.001 and never sees
the operating threshold.

WHAT THIS CANNOT MEASURE
The rate at which a good part is wrongly ejected. NEU-DET has no defect-free
images, so every figure here is conditioned on the part being defective. The
wrong-class column is a proxy for false-positive propensity and nothing more.
Do not lower the deployed threshold on the strength of this script alone.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

THRESHOLDS = (0.35, 0.25, 0.15, 0.10, 0.05, 0.01)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--weights", default=str(REPO_ROOT / "models" / "best.pt"))
    ap.add_argument("--gate-conf-below", type=float, default=0.55,
                    help="the PatchCore gate, for the redundancy table")
    args = ap.parse_args()

    from inspector.config import load_config
    from ultralytics import YOLO

    cfg = load_config()
    classes = list(cfg.classes)
    model = YOLO(args.weights)

    labels = sorted(glob.glob(str(REPO_ROOT / "data" / "yolo" / args.split / "labels" / "*.txt")))
    if not labels:
        sys.exit(f"no labels under data/yolo/{args.split}/labels")

    rows = []
    for f in labels:
        recs = [l.split() for l in Path(f).read_text().split("\n") if l.strip()]
        if not recs:
            continue
        truth = int(recs[0][0])
        img = f.replace("/labels/", "/images/").replace(".txt", ".jpg")
        # conf=0.01 once, then re-threshold in numpy: running the model per
        # threshold would be six times the work for identical output.
        r = model.predict(img, imgsz=args.imgsz, conf=0.01, verbose=False)[0]
        conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.empty(0)
        cls = (r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes)
               else np.empty(0, int))
        rows.append((truth, len(recs), conf, cls))

    print(f"\n{len(rows)} images, split={args.split}, imgsz={args.imgsz}")
    print(f"weights: {Path(args.weights).name}")
    print(f"ground truth: {np.mean([g for _, g, _, _ in rows]):.2f} boxes/image\n")

    print(f"{'conf':<7}{'correct label':>15}{'escapes':>10}{'boxes/img':>11}"
          f"{'wrong-cls/img':>15}")
    print("-" * 58)
    for thr in THRESHOLDS:
        ok = esc = 0
        boxes, wrong = [], []
        for truth, _, conf, cls in rows:
            keep = conf >= thr
            boxes.append(int(keep.sum()))
            wrong.append(int((cls[keep] != truth).sum()))
            if keep.any():
                if cls[keep][conf[keep].argmax()] == truth:
                    ok += 1
            else:
                esc += 1
        n = len(rows)
        print(f"{thr:<7.2f}{ok / n:>15.2f}{esc / n:>10.3f}"
              f"{np.mean(boxes):>11.2f}{np.mean(wrong):>15.2f}")

    # Per class, at the deployed threshold and at a candidate decision threshold.
    print(f"\nper-class correct label")
    print(f"{'class':<18}" + "".join(f"{t:>8.2f}" for t in (0.35, 0.10)))
    print("-" * 34)
    for c, name in enumerate(classes):
        sub = [r for r in rows if r[0] == c]
        cells = []
        for thr in (0.35, 0.10):
            hit = sum(1 for truth, _, conf, cls in sub
                      if (conf >= thr).any()
                      and cls[conf >= thr][conf[conf >= thr].argmax()] == truth)
            cells.append(hit / max(1, len(sub)))
        print(f"{name:<18}" + "".join(f"{v:>8.2f}" for v in cells))

    # Is the anomaly backstop reachable by the detector alone?
    gate = [r for r in rows if (r[2].max() if len(r[2]) else 0.0) < args.gate_conf_below]
    print(f"\nframes entering the anomaly gate (max conf < {args.gate_conf_below}): "
          f"{len(gate)}/{len(rows)} = {len(gate) / len(rows):.1%}")
    print(f"{'conf':<7}{'correct label on those frames':>32}")
    print("-" * 39)
    for thr in (0.35, 0.25, 0.10, 0.05):
        hit = sum(1 for truth, _, conf, cls in gate
                  if (conf >= thr).any()
                  and cls[conf >= thr][conf[conf >= thr].argmax()] == truth)
        print(f"{thr:<7.2f}{hit / max(1, len(gate)):>32.2f}")
    print("\nPatchCore costs ~110 ms per call (scripts/profile_stages.py) and "
          "separates\nweakly on this dataset (2.40 vs 2.26, reports/anomaly.md). "
          "Re-thresholding costs 0 ms.")


if __name__ == "__main__":
    main()
