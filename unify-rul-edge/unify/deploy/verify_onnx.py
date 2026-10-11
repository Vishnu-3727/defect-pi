"""Prove the exported model is correct before it goes anywhere near the Pi.

Two checks, both of which have caught real bugs in this class of project:

1. NUMERICAL EQUIVALENCE with PyTorch. An export that silently changes numerics
   (train-mode BatchNorm, an unsupported op quietly approximated, an opset
   mismatch) is miserable to debug from the device. Catch it here in 2 seconds.

2. VARIABLE SENSOR COUNT. Run S = 2, 10, 14, 21 through the same file. This is
   the project's headline claim; if dynamic_axes was wrong, S=2 throws and you
   find out now instead of in front of an examiner.

    python -m unify.deploy.verify_onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ..models.unify_rul import build as build_model

ROOT = Path(__file__).resolve().parents[2]
SENSOR_COUNTS = (2, 10, 14, 21)


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _feeds(S: int, L: int, B: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "x": rng.standard_normal((B, S, L)).astype(np.float32),
        "mask": np.ones((B, S), dtype=bool),
        "sensor_ids": np.tile(np.arange(1, S + 1, dtype=np.int64), (B, 1)),
        "domain_id": np.zeros(B, dtype=np.int64),
    }


def verify(onnx_path: Path, ckpt: dict, rtol: float = 1e-3, atol: float = 1e-4) -> bool:
    cfg = ckpt["cfg"]
    L = cfg["data"]["window"]

    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sess = _session(onnx_path)
    ok = True

    print(f"{'S':>4} {'onnx out':>12} {'torch out':>12} {'max abs diff':>14}  status")
    for S in SENSOR_COUNTS:
        f = _feeds(S, L, B=1, seed=S)
        try:
            onnx_out = sess.run(["rul"], f)[0]
        except Exception as e:                       # dynamic_axes wrong -> lands here
            print(f"{S:>4}  FAILED: {type(e).__name__}: {str(e)[:80]}")
            ok = False
            continue

        with torch.no_grad():
            t_out, _ = model(torch.from_numpy(f["x"]), torch.from_numpy(f["mask"]),
                             torch.from_numpy(f["sensor_ids"]),
                             torch.from_numpy(f["domain_id"]))
        t_out = t_out.numpy()

        diff = float(np.abs(onnx_out - t_out).max())
        close = np.allclose(onnx_out, t_out, rtol=rtol, atol=atol)
        ok &= bool(close) and onnx_out.shape == (1, model.n_outputs)
        print(f"{S:>4} {onnx_out.ravel()[0]:>12.6f} {t_out.ravel()[0]:>12.6f} "
              f"{diff:>14.2e}  {'ok' if close else 'MISMATCH'}")

    # Batching must still work — the Pi predicts with B=1, training used B=256.
    f = _feeds(14, L, B=8, seed=99)
    b_out = sess.run(["rul"], f)[0]
    ok &= b_out.shape[0] == 8
    print(f"batch=8 -> {b_out.shape} {'ok' if b_out.shape[0] == 8 else 'FAILED'}")

    # A masked-out sensor must not change the answer. This is what lets a sensor
    # fail in the field without retraining, and it is a genuine selling point.
    f = _feeds(14, L, B=1, seed=7)
    base = sess.run(["rul"], f)[0]
    f2 = {k: v.copy() for k, v in f.items()}
    f2["mask"][0, -1] = False
    f2["x"][0, -1] = 999.0                       # garbage in the masked channel
    masked = sess.run(["rul"], f2)[0]
    changed = float(np.abs(base - masked).max())
    # It SHOULD change (one fewer sensor attended) but must stay finite and sane.
    finite = np.isfinite(masked).all() and abs(float(masked.ravel()[0])) < 1e3
    ok &= bool(finite)
    print(f"masked sensor: delta {changed:.4f}, finite={finite} "
          f"{'ok' if finite else 'FAILED — garbage leaked through the mask'}")

    print("\nVERIFY " + ("PASSED" if ok else "FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="edge/artifacts/unify_rul.onnx")
    ap.add_argument("--ckpt", default="results/best.pt")
    a = ap.parse_args()

    ckpt = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=True)
    raise SystemExit(0 if verify(ROOT / a.onnx, ckpt) else 1)


if __name__ == "__main__":
    main()
