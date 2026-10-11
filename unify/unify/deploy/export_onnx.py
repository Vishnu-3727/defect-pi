"""Export the trained model to ONNX, verify it, and stage the three Pi artifacts.

    python -m unify.deploy.export_onnx

Everything happens in ONE script so the model and its normalisation statistics can
never drift apart. Exactly three files go to the Pi:

    unify_rul.onnx      the frozen graph
    norm_stats.json     the training statistics + per-dataset life scales
    sensor_vocab.json   the sensor-ID map

If you find yourself copying a .py model definition to the Pi, the export is
incomplete.

The critical line is dynamic_axes marking axis 1 of `x` as "sensors". Without it
the exported graph is welded to whatever S the dummy input had, and every claim in
the project title dies at export time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from ..models.unify_rul import build as build_model

ROOT = Path(__file__).resolve().parents[2]
OPSET = 17


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(ckpt_path: Path, out_dir: Path, opset: int = OPSET,
           dummy_sensors: int = 14, window: int | None = None) -> Path:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["cfg"]
    L = window or cfg["data"]["window"]

    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    # BatchNorm uses batch statistics in train mode and running statistics in eval
    # mode. Exporting in train mode bakes in the wrong behaviour permanently, and
    # with batch size 1 on the Pi it divides by zero variance. This one call is the
    # difference between a working deployment and a silently broken one.
    model.eval()

    S = dummy_sensors
    args = (torch.randn(1, S, L),
            torch.ones(1, S, dtype=torch.bool),
            torch.arange(1, S + 1, dtype=torch.long).unsqueeze(0),
            torch.zeros(1, dtype=torch.long))

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "unify_rul.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model, args, str(onnx_path),
            input_names=["x", "mask", "sensor_ids", "domain_id"],
            output_names=["rul", "attn"],
            dynamic_axes={
                "x": {0: "batch", 1: "sensors"},           # <- the line that matters
                "mask": {0: "batch", 1: "sensors"},
                "sensor_ids": {0: "batch", 1: "sensors"},
                "domain_id": {0: "batch"},
                "rul": {0: "batch"},
                "attn": {0: "batch", 3: "sensors"},
            },
            opset_version=opset,
            do_constant_folding=True,
            # Force the TorchScript TRACING exporter. torch>=2.6 defaults to the
            # dynamo exporter, which on this graph (a) ignores opset_version and
            # silently emits opset 18, (b) does not propagate dynamic_axes to the
            # OUTPUT shapes, so ORT logs a shape-mismatch warning on every batched
            # call, and (c) produces a graph that quantize_dynamic rejects with
            # "Inferred shape and existing shape differ".
            # Tracing gets all three right. The model is written to be
            # trace-friendly on purpose — no .item(), no data-dependent branching.
            dynamo=False,
        )
    return onnx_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/best.pt")
    ap.add_argument("--out", default="edge/artifacts")
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--quantize", action="store_true",
                    help="also emit an int8 model and report the size delta")
    a = ap.parse_args()

    ckpt_path = ROOT / a.ckpt
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}. Train first.")

    out_dir = ROOT / a.out
    onnx_path = export(ckpt_path, out_dir, a.opset)
    print(f"exported {onnx_path}  ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # ── structural check ──────────────────────────────────────────────────────
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))
    print("onnx.checker: ok")

    # ── numerical + variable-S check ──────────────────────────────────────────
    from .verify_onnx import verify
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ok = verify(onnx_path, ckpt)
    if not ok:
        raise SystemExit("verification FAILED — do not deploy this file")

    # ── stage the other two artifacts ─────────────────────────────────────────
    shutil.copy(ROOT / "config" / "sensor_vocab.json", out_dir / "sensor_vocab.json")

    # ALWAYS regenerate from the checkpoint, never keep whatever happens to be
    # sitting in out_dir. This used to be guarded by `if not stats_path.exists()`,
    # which did the exact opposite of the "the two files can never disagree"
    # promise in its own comment: exporting a --ckpt from one training run into a
    # directory left over from another silently paired new weights with old
    # normalisation statistics. Nothing downstream catches that — verify_onnx
    # compares ONNX against PyTorch, both of which see already-normalised input,
    # so the export passes and the Pi quietly z-scores with the wrong mean and
    # std. The checkpoint's own norm/scales match its weights by construction.
    stats_path = out_dir / "norm_stats.json"
    stats_path.write_text(json.dumps(
        {"datasets": ckpt["norm"], "scales": ckpt["scales"],
         "window": ckpt["cfg"]["data"]["window"],
         "dataset_config": {k: v for k, v in ckpt["cfg"].get("dataset_config", {}).items()}},
        indent=2))
    print(f"staged {stats_path.name} (datasets: {sorted(ckpt['norm'])}), "
          f"sensor_vocab.json")

    digest = sha256(onnx_path)
    (out_dir / "model.sha256").write_text(digest)
    manifest = {
        "model": onnx_path.name,
        "sha256": digest,
        "opset": a.opset,
        "window": ckpt["cfg"]["data"]["window"],
        "scales": ckpt["scales"],
        "epoch": ckpt["epoch"],
        "val": ckpt["val"],
        "size_bytes": onnx_path.stat().st_size,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"sha256 {digest[:16]}...")

    if a.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = out_dir / "unify_rul_int8.onnx"
        quantize_dynamic(str(onnx_path), str(q), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm"])
        f32, i8 = onnx_path.stat().st_size, q.stat().st_size
        print(f"int8: {i8/1024:.1f} KB vs fp32 {f32/1024:.1f} KB ({f32/i8:.1f}x smaller)")
        print("NOW RE-MEASURE ACCURACY. A quantisation claim without an RMSE "
              "number beside it is not a result.")

    print("\nDeploy:\n"
          f"  scp {out_dir}/unify_rul.onnx {out_dir}/norm_stats.json "
          f"{out_dir}/sensor_vocab.json pi@raspberrypi.local:~/unify/edge/artifacts/")


if __name__ == "__main__":
    main()
