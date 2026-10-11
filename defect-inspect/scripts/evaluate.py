"""Evaluate the trained detector and write a report a human can read.

Ultralytics already computes the metrics and draws the curves. What it does not
do is put the numbers in one file with the context needed to interpret them, so
this script does that and writes reports/metrics.md.

Evaluated on the TEST split by default, not val. Val was used to pick the best
epoch during training, so val mAP is mildly optimistic — the model was selected
on it. Test has been untouched until now. See docs/11_evaluation_metrics.md.

Run:  uv run scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def write_report(metrics, names: list[str], split: str, weights: Path,
                 imgsz: int, save_dir: Path, out: Path) -> None:
    box = metrics.box
    lines: list[str] = []
    add = lines.append

    add("# Detection metrics")
    add("")
    add(f"- Generated: {date.today().isoformat()}")
    add(f"- Weights: `{weights.name}`")
    add(f"- Split: **{split}** (held out; never used for model selection)")
    add(f"- Image size: {imgsz}")
    add("")
    add("## Overall")
    add("")
    add("| Metric | Value |")
    add("|--------|-------|")
    add(f"| mAP@50 | **{box.map50:.4f}** |")
    add(f"| mAP@50-95 | **{box.map:.4f}** |")
    add(f"| Precision (mean) | {box.mp:.4f} |")
    add(f"| Recall (mean) | {box.mr:.4f} |")
    add(f"| F1 (from mean P and R) | {f1(box.mp, box.mr):.4f} |")
    add("")
    add("## Per class")
    add("")
    add("| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 |")
    add("|-------|-----------|--------|----|--------|-----------|")

    # box.ap_class_index maps each row of the per-class arrays back to a class id.
    for row, cls_id in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.p[row], box.r[row], box.ap50[row], box.ap[row]
        add(f"| {names[int(cls_id)]} | {p:.4f} | {r:.4f} | {f1(p, r):.4f} "
            f"| {ap50:.4f} | {ap:.4f} |")

    add("")
    add("## How to read this")
    add("")
    add("- **Precision** — of the boxes the model reported, what fraction were real.")
    add("  Low precision means false alarms: good parts get ejected.")
    add("- **Recall** — of the real defects, what fraction the model found.")
    add("  Low recall means escapes: defective parts reach the customer.")
    add("- **mAP@50** — average precision across recall levels, counting a box")
    add("  correct when it overlaps the truth by at least 50%. The standard")
    add("  headline number for detection.")
    add("- **mAP@50-95** — the same, averaged over IoU thresholds 0.50 to 0.95.")
    add("  Much stricter: it grades how tightly the box fits, not just whether")
    add("  the defect was found. Always lower, and it is meant to be.")
    add("")
    add("For an inspection system these two errors are not equally costly.")
    add("Shipping a defective part is normally worse than re-checking a good one,")
    add("so the operating point should favour recall. That is a business decision")
    add("expressed as one number: `detector.conf_threshold` in `configs/default.yaml`.")
    add("")
    add("## Plots")
    add("")
    for plot in ("confusion_matrix_normalized.png", "PR_curve.png",
                 "F1_curve.png", "R_curve.png", "P_curve.png"):
        if (out.parent / plot).exists():
            add(f"### {plot.replace('_', ' ').replace('.png', '')}")
            add("")
            add(f"![{plot}]({plot})")
            add("")

    add(f"Raw ultralytics output: `{save_dir}`")
    add("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(REPO_ROOT / "models" / "best.pt"))
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "yolo" / "data.yaml"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"{weights} not found. Run scripts/train.py first.")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    # rect=False is not a tuning choice, it is a correctness one. The validator
    # defaults to rectangular batching and pads to a stride multiple using imgsz
    # as a LOWER BOUND, so a 200x200 NEU-DET image at imgsz=320 is fed to the
    # network at 352. The resulting mAP is not the mAP at the size in the title,
    # the bias is model-dependent (+0.021, -0.025, +0.016 on three checkpoints
    # here), and it is large enough to reverse a ranking. The deployed ONNX graph
    # has a fixed input and genuinely runs at imgsz, so this is also the only
    # setting under which the report describes the deployed system.
    # See reports/imgsz-rect-trap.md.
    metrics = model.val(
        rect=False,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        project=str(REPO_ROOT / "runs"),
        name=f"eval_{args.split}",
        exist_ok=True,
    )

    names = [model.names[i] for i in sorted(model.names)]
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)

    save_dir = Path(metrics.save_dir)
    for png in save_dir.glob("*.png"):
        shutil.copy2(png, reports / png.name)

    out = reports / "metrics.md"
    write_report(metrics, names, args.split, weights, args.imgsz, save_dir, out)

    print(f"\nmAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP@50-95 : {metrics.box.map:.4f}")
    print(f"report    : {out}")


if __name__ == "__main__":
    main()
