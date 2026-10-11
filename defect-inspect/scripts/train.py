"""Train YOLOv8n on NEU-DET.

Why YOLOv8n and not a bigger variant: the model has to hold a real-time frame
rate on four Cortex-A76 cores with no GPU. Parameter count is the budget.
See docs/08_yolo_v1_to_v8.md for the size comparison and docs/13 for the
measured latency that justifies the choice.

Why imgsz=320: NEU-DET images are natively 200x200. Training at 640 would
upsample every image, doubling compute for invented detail. 320 is the nearest
stride-32 multiple above native resolution.

Run:  uv run scripts/train.py --epochs 100
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_args(a: argparse.Namespace) -> dict:
    """Training hyperparameters, each one chosen for this dataset.

    The augmentation block matters more than the learning rate here. NEU-DET is
    1236 training images, which is small; augmentation is what stops the model
    memorising them.
    """
    return dict(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        seed=a.seed,
        deterministic=True,
        project=str(REPO_ROOT / "runs"),
        name=a.name,
        exist_ok=True,
        patience=a.patience,          # stop if val mAP has not improved in N epochs
        optimizer="auto",             # ultralytics picks AdamW and an LR for small sets
        cos_lr=True,                  # smooth LR decay, no step cliffs
        val=True,
        plots=True,

        # --- augmentation ---
        # Steel defects have no canonical orientation: a scratch is a scratch
        # upside down. So both flips are free training data.
        fliplr=0.5,
        flipud=0.5,
        degrees=10.0,                 # small rotations; the strip itself is axis-aligned
        translate=0.1,
        scale=0.3,                    # simulates camera height variation on the rig
        shear=0.0,
        perspective=0.0,              # the camera looks straight down; no perspective
        # The images are grayscale, so hue and saturation jitter would be a no-op
        # on the data and a lie in the report. Only brightness is meaningful,
        # and it stands in for the LED ring light ageing or dimming.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        mosaic=1.0,                   # 4-image tiling, the main regulariser for small sets
        close_mosaic=10,              # off for the last 10 epochs so the model
                                      # finishes on undistorted, realistic frames
        mixup=0.0,                    # blends two images; meaningless for surface texture
        erasing=0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "yolo" / "data.yaml"))
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained weights to fine-tune")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0", help="'0' for the first GPU, 'cpu' otherwise")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="yolov8n_neu_320")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} not found. Run scripts/prepare_dataset.py first.")

    # Imported here, not at module scope: importing ultralytics costs several
    # seconds and pulls in torch, so --help should not pay for it.
    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(**build_args(args))

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training finished but {best} is missing")

    dest = REPO_ROOT / "models" / "best.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(best, dest)

    print(f"\nbest weights: {best}")
    print(f"copied to   : {dest}")
    print(f"run artifacts (curves, confusion matrix): {save_dir}")
    print("\nnext: uv run scripts/evaluate.py")


if __name__ == "__main__":
    main()
