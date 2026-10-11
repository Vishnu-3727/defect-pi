"""Build the self-contained demo site under site/.

The site runs the REAL exported model in the visitor's browser via
onnxruntime-web. Nothing here is a recording: this script only exports the
inputs (real C-MAPSS test engines) and the artifacts the model needs, then
verifies that the Python predictor and the browser will be fed identical
tensors. The predictions themselves are computed client-side at view time.

Run:  python scripts/build_demo_site.py
      python scripts/build_demo_site.py --demo     # self-check only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unify.common.normalize import Normalizer  # noqa: E402

SITE = ROOT / "site"
ARTIFACTS = ROOT / "edge" / "artifacts"
INTERIM = ROOT / "data" / "interim" / "cmapss"

# Chosen deliberately, not sampled: three different sensor counts through one
# weight set is the entire claim in the project title, so the picker must be able
# to demonstrate it. Two of them also run under six operating conditions, which
# is where per-condition normalisation earns its keep.
ENGINES = [
    ("FD001", "u34", "15 sensors, 1 operating condition - runs to near failure"),
    ("FD001", "u93", "15 sensors - still healthy at the end of its record"),
    ("FD003", "u24", "16 sensors, 475 cycles - the longest record shipped"),
    ("FD002", "u11", "21 sensors, 6 operating conditions"),
    ("FD004", "u158", "21 sensors, 6 operating conditions"),
]

# ponytail: 4 decimals. The sensors are z-scored against statistics whose std is
# order 1, so the 5th decimal is far below the noise the model was trained on,
# and it halves the JSON. verify() proves the rounding does not move the RUL.
ROUND = 4


def load_unit(subset, unit):
    path = INTERIM / f"cmapss_{subset}_test_{unit}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path} - run: make data")
    # No allow_pickle: the four arrays read here are plain numeric arrays, so the
    # default (pickle disabled) loads them fine and the exporter never executes
    # anything from a .npz. Only the unread "meta" field would need it.
    d = np.load(path)
    return d["signals"], d["sensor_ids"], d["rul"], d["condition"]


def normalise_per_condition(norm, signals, sensor_ids, condition):
    """Exactly what unify.common.normalize.Normalizer.apply does for C-MAPSS.

    Kept in one function so the exporter and the verifier cannot drift apart.
    """
    out = np.empty_like(signals, dtype=np.float32)
    for c in np.unique(condition):
        m = condition == c
        mean, std = norm.vector_for("cmapss", sensor_ids, str(int(c)))
        out[m] = (signals[m] - mean) / std
    return out


def build_payload():
    stats = json.loads((ARTIFACTS / "norm_stats.json").read_text())
    vocab = {k: v for k, v in json.loads((ARTIFACTS / "sensor_vocab.json").read_text()).items()
             if not k.startswith("_")}
    inv_vocab = {v: k for k, v in vocab.items()}

    window = int(stats.get("window", 30))
    life_scale = float(stats["scales"]["cmapss"])

    # C-MAPSS labels are capped during training (config/datasets.yaml, cap_mode
    # absolute - the Heimes 2008 field standard). The model therefore CANNOT
    # output more than the cap, so the demo must score it against the capped
    # label. Comparing a capped prediction against an uncapped label manufactures
    # a 50-cycle "error" on every healthy engine that is purely an artefact.
    # Read from config rather than hardcoded: if someone retrains at a different
    # cap, the page must not keep scoring against the old one.
    ds_cfg = yaml.safe_load((ROOT / "config" / "datasets.yaml").read_text())
    cm = ds_cfg["cmapss"] if "cmapss" in ds_cfg else ds_cfg["datasets"]["cmapss"]
    assert cm["cap_mode"] == "absolute", f"unexpected cap_mode {cm['cap_mode']!r} for cmapss"
    cap = float(cm["cap"])

    norm = Normalizer()
    norm.stats = stats["datasets"]

    engines = []
    for subset, unit, note in ENGINES:
        signals, sensor_ids, rul, condition = load_unit(subset, unit)

        # Normalise HERE, per condition, using the shipped training statistics -
        # then the browser never reimplements the lookup and cannot drift from the
        # trainer. The rule this project will not bend is that the statistics
        # travel with the model and are reused verbatim.
        normed = normalise_per_condition(norm, signals, sensor_ids, condition)

        engines.append({
            "id": f"{subset}_{unit}",
            "subset": subset,
            "unit": unit,
            "note": note,
            "sensors": int(len(sensor_ids)),
            "cycles": int(len(signals)),
            "conditions": int(len(np.unique(condition))),
            "sensor_ids": [int(s) for s in sensor_ids],
            "sensor_names": [inv_vocab.get(int(s), f"id{int(s)}") for s in sensor_ids],
            "condition": [int(c) for c in condition],
            "true_rul": [float(r) for r in rul],
            # What the model was actually trained to predict. The page scores
            # against this and shows the uncapped value beside it.
            "true_rul_capped": [min(float(r), cap) for r in rul],
            # (T, S), already z-scored. The browser slices a window and transposes
            # it to (1, S, L) - that is all the preprocessing left on the client.
            "x": [[round(float(v), ROUND) for v in row] for row in normed],
            # Raw values for the sensor readout. Rounded harder: displayed, never
            # fed to the model.
            "raw": [[round(float(v), 2) for v in row] for row in signals],
        })

    return {
        "window": window,
        "life_scale": life_scale,
        "domain_id": 0,
        "cap": cap,
        "model": "unify_rul.onnx",
        "engines": engines,
    }


def verify(payload, tol_rul_cycles=0.05):
    """Prove the browser gets tensors that reproduce the Python predictor.

    Feeds the exported (rounded) window through onnxruntime in Python and
    compares against the same window normalised at full precision. If the
    rounding or the per-condition lookup were wrong, the RUL moves and this
    fails - which is the point of shipping a check instead of a promise.
    """
    import onnxruntime as ort

    stats = json.loads((ARTIFACTS / "norm_stats.json").read_text())
    norm = Normalizer()
    norm.stats = stats["datasets"]
    sess = ort.InferenceSession(str(ARTIFACTS / "unify_rul.onnx"),
                                providers=["CPUExecutionProvider"])
    L = payload["window"]
    scale = payload["life_scale"]

    worst = 0.0
    for eng in payload["engines"]:
        signals, sensor_ids, _, condition = load_unit(eng["subset"], eng["unit"])
        exported = np.asarray(eng["x"], dtype=np.float32)
        full = normalise_per_condition(norm, signals, sensor_ids, condition)

        sids = np.asarray(eng["sensor_ids"], dtype=np.int64)[None]
        common = {
            "mask": np.ones((1, sids.shape[1]), dtype=bool),
            "sensor_ids": sids,
            "domain_id": np.array([payload["domain_id"]], dtype=np.int64),
        }
        # Check several windows, not just the last: a per-condition bug on
        # FD002/FD004 shows up wherever the condition changes, not only at the end.
        for end in sorted({L, len(signals) // 2, len(signals)}):
            if end < L:
                continue
            a = sess.run(["rul"], dict(x=exported[end - L:end].T[None], **common))[0]
            b = sess.run(["rul"], dict(x=full[end - L:end].T[None], **common))[0]
            av = float(np.clip(a, 0, 1).ravel()[0])
            bv = float(np.clip(b, 0, 1).ravel()[0])
            worst = max(worst, abs(av - bv) * scale)

    print(f"  export parity: worst |dRUL| from {ROUND}-decimal rounding = {worst:.4f} cycles")
    assert worst < tol_rul_cycles, (
        f"rounding to {ROUND} decimals moved RUL by {worst:.3f} cycles "
        f"(tolerance {tol_rul_cycles}) - raise ROUND")


def demo():
    """Self-check: shapes are consistent and the export reproduces the predictor."""
    payload = build_payload()
    assert len(payload["engines"]) == len(ENGINES)
    counts = {e["sensors"] for e in payload["engines"]}
    assert counts == {15, 16, 21}, f"expected 3 distinct sensor counts, got {counts}"
    for e in payload["engines"]:
        assert len(e["x"]) == e["cycles"] == len(e["true_rul"]) == len(e["raw"])
        assert len(e["x"][0]) == e["sensors"] == len(e["sensor_ids"]) == len(e["sensor_names"])
        assert e["cycles"] >= payload["window"], f"{e['id']} is shorter than the window"
    verify(payload)
    print("  demo OK: 3 sensor counts, shapes consistent, parity within tolerance")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="run the self-check and exit")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    payload = build_payload()
    verify(payload)

    assets = SITE / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ARTIFACTS / "unify_rul.onnx", assets / "unify_rul.onnx")
    (assets / "engines.json").write_text(json.dumps(payload, separators=(",", ":")))

    mb = (assets / "engines.json").stat().st_size / 1e6
    onnx_kb = (assets / "unify_rul.onnx").stat().st_size / 1e3
    print(f"  wrote {assets / 'engines.json'}  {mb:.2f} MB")
    print(f"  wrote {assets / 'unify_rul.onnx'}  {onnx_kb:.0f} KB")
    for e in payload["engines"]:
        print(f"    {e['id']:12s} {e['sensors']:2d} sensors  {e['cycles']:3d} cycles  "
              f"{e['conditions']} condition(s)")


if __name__ == "__main__":
    main()
