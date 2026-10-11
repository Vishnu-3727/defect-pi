"""Export best.pt to ONNX, then quantise to INT8 for the Pi.

Three artefacts come out of this script:

  models/model.onnx        FP32, portable, no PyTorch needed to run
  models/model_int8.onnx   weights and activations as 8-bit integers
  models/calib/            the images used to calibrate the quantiser

Why quantise at all: a Cortex-A76 executes integer multiply-accumulate faster
than float, and INT8 also quarters the bytes moved between RAM and cache.
Convolution is often memory-bound, so the speedup usually exceeds the arithmetic
ratio. Cost is a small accuracy drop, which we measure rather than assume.

Static quantisation needs *calibration*: real images are pushed through the
FP32 graph while the tool records the actual range of every activation tensor,
so it can choose a scale factor per tensor. Calibrating on random noise would
produce ranges no real image ever hits, and the INT8 model would be useless.

Run:  uv run scripts/export_onnx.py --int8
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def export_fp32(weights: Path, imgsz: int, opset: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # simplify=True folds constants and removes training-only nodes, which both
    # shrinks the graph and avoids operators the quantiser cannot handle.
    exported = model.export(format="onnx", imgsz=imgsz, opset=opset,
                            simplify=True, dynamic=False)

    dest = REPO_ROOT / "models" / "model.onnx"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(exported, dest)
    return dest


def collect_calibration(src: Path, out: Path, count: int, imgsz: int, seed: int) -> Path:
    """Copy a class-balanced sample of real training images for calibration."""
    import cv2

    from inspector.preprocess import letterbox

    images = sorted(src.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images found in {src}")

    # Balanced across classes: filenames are <class>_<n>.jpg, and calibrating on
    # 200 crazing images would give ranges skewed to one texture.
    by_class: dict[str, list[Path]] = {}
    for p in images:
        by_class.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)

    rng = random.Random(seed)
    per_class = max(1, count // len(by_class))
    chosen: list[Path] = []
    for cls in sorted(by_class):
        pool = sorted(by_class[cls])
        rng.shuffle(pool)
        chosen += pool[:per_class]

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for p in chosen:
        img = cv2.imread(str(p))
        padded, _, _ = letterbox(img, imgsz)
        cv2.imwrite(str(out / p.name), padded)

    print(f"calibration set: {len(chosen)} images across {len(by_class)} classes -> {out}")
    return out


class CalibrationReader:
    """Feeds preprocessed images to the quantiser, one at a time.

    onnxruntime.quantization calls get_next() until it returns None. The tensors
    must go through exactly the same preprocessing the runtime uses, or the
    recorded activation ranges describe a different distribution than the one
    the deployed model will see.
    """

    def __init__(self, folder: Path, input_name: str):
        import cv2

        from inspector.preprocess import to_tensor

        self.input_name = input_name
        self._batches = iter([
            {input_name: to_tensor(cv2.imread(str(p)))}
            for p in sorted(folder.glob("*.jpg"))
        ])

    def get_next(self):
        return next(self._batches, None)


def head_nodes(model_path: Path, prefix: str) -> list[str]:
    """Names of the detection-head nodes, so quantisation can skip them.

    Measured, not assumed: quantising the whole graph dropped mAP@50 from 0.735
    to 0.408 on the test split, with recall collapsing from 0.68 to 0.42 while
    precision rose. That pattern means low-confidence detections were being lost,
    and the head is where confidence is computed.

    The head does three things that quantise badly. The Softmax and DFL
    convolution turn a distribution over 16 bins into a sub-pixel coordinate,
    so 8-bit bins visibly coarsen box positions. The Sigmoid compresses a wide
    logit range into 0..1, where a whole INT8 step can span the gap between a
    detection and a miss. And the Sub/Add/Div box arithmetic operates on
    absolute pixel coordinates whose range is far wider than the activations
    the calibrator saw elsewhere.

    The head is also cheap: it runs on small feature maps, so leaving it in FP32
    costs little speed. The backbone and neck, which hold nearly all the
    multiply-accumulates, stay INT8.
    """
    import onnx

    graph = onnx.load(str(model_path)).graph
    return [n.name for n in graph.node if n.name.startswith(prefix)]


def quantise(fp32: Path, calib_dir: Path, exclude: list[str],
             calib_method: str) -> Path:
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.preprocess import quant_pre_process
    import onnxruntime as ort

    prepared = fp32.with_name("model_prepared.onnx")
    # Shape inference and graph cleanup. Skipping this is the usual cause of
    # "node not quantised" warnings and a model that is barely faster.
    quant_pre_process(str(fp32), str(prepared), skip_symbolic_shape=False)

    input_name = ort.InferenceSession(
        str(prepared), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name

    dest = fp32.with_name("model_int8.onnx")
    quantize_static(
        model_input=str(prepared),
        model_output=str(dest),
        calibration_data_reader=CalibrationReader(calib_dir, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,          # a scale per output channel; noticeably better mAP
        calibrate_method=getattr(CalibrationMethod, calib_method),
        nodes_to_exclude=exclude,
    )
    prepared.unlink(missing_ok=True)
    return dest


def report_sizes(paths: list[Path]) -> None:
    print()
    print(f"{'artefact':<28}{'size':>12}")
    print("-" * 40)
    for p in paths:
        if p.exists():
            print(f"{p.name:<28}{p.stat().st_size / 1e6:>9.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(REPO_ROOT / "models" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=320)
    # opset 13 is a floor, not a preference: per-channel quantisation emits
    # DequantizeLinear with an `axis` attribute, which does not exist before 13.
    # Exporting at 12 produces an INT8 file that loads with
    # "Unrecognized attribute: axis for operator DequantizeLinear".
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--int8", action="store_true", help="also produce the INT8 model")
    ap.add_argument("--calib-count", type=int, default=200)
    ap.add_argument("--calib-method", default="MinMax",
                    choices=["MinMax", "Entropy", "Percentile"])
    ap.add_argument("--head-prefix", default="/model.22/",
                    help="node-name prefix of the detection head")
    ap.add_argument("--quantise-head", action="store_true",
                    help="quantise the head too; measured to cost ~33 points of mAP@50")
    ap.add_argument("--calib-src",
                    default=str(REPO_ROOT / "data" / "yolo" / "train" / "images"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"{weights} not found. Run scripts/train.py first.")

    fp32 = export_fp32(weights, args.imgsz, args.opset)
    print(f"FP32 ONNX: {fp32}")

    artefacts = [weights, fp32]
    if args.int8:
        calib = collect_calibration(Path(args.calib_src),
                                    REPO_ROOT / "models" / "calib",
                                    args.calib_count, args.imgsz, args.seed)
        exclude = [] if args.quantise_head else head_nodes(fp32, args.head_prefix)
        if exclude:
            print(f"keeping {len(exclude)} detection-head nodes in FP32 "
                  f"(prefix {args.head_prefix})")
        int8 = quantise(fp32, calib, exclude, args.calib_method)
        print(f"INT8 ONNX: {int8}")
        artefacts.append(int8)

    report_sizes(artefacts)
    print("\nnext: uv run python -m inspector.bench --model models/model_int8.onnx")


if __name__ == "__main__":
    main()
