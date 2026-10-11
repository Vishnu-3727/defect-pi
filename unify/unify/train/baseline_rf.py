"""RandomForest baseline on hand-crafted window summaries.

Every serious ML report has a non-deep baseline; almost no student project does.
If UNIFY-RUL cannot beat a random forest on summary statistics, the deep model has
not justified itself and you have learned something worth reporting either way.

    python -m unify.train.baseline_rf --datasets synthetic synthetic_small
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

from ..common.normalize import Normalizer
from ..common.windowing import window_unit
from ..datasets.build import dataset_config, load_cached
from ..datasets.joint import hold_out_official_test, split_units
from ..models.losses import mae, nasa_score, rmse

ROOT = Path(__file__).resolve().parents[2]


def summarize(X: np.ndarray) -> np.ndarray:
    """(N, S, L) -> (N, S*5): per-channel mean, std, slope, first, last.

    The slope is the important one — it is the crude version of what the CNN
    learns, so beating this baseline means the CNN found something a linear trend
    does not capture.
    """
    N, S, L = X.shape
    t = np.arange(L, dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-8)
    slope = (X * t).mean(-1)
    return np.concatenate([X.mean(-1), X.std(-1), slope, X[..., 0], X[..., -1]], axis=1)


def pad_to(x: np.ndarray, width: int) -> np.ndarray:
    """RandomForest needs a fixed column count — pad the shorter feature vectors.

    This is exactly the limitation UNIFY-RUL removes: a classical model must be
    told the sensor count up front, whereas the mask handles it in the deep model.
    Worth saying out loud in the report.
    """
    if x.shape[1] >= width:
        return x[:, :width]
    return np.pad(x, ((0, 0), (0, width - x.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--out", default="results/baseline_rf.json")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / a.config).read_text())
    names = a.datasets or cfg["data"]["datasets"]
    ds_cfg = dataset_config()

    units = load_cached(names, ROOT / cfg["data"]["interim_dir"])
    if not units:
        raise SystemExit(f"no cached units for {names}; run unify.datasets.build first")

    # Same three-way split and the same test protocol as unify.train.train. A
    # baseline scored on a different split is not a baseline, it is a second
    # unrelated number that happens to sit next to the first one in the table.
    pool_units, test_units = hold_out_official_test(units)
    train_units, val_units = split_units(pool_units, cfg["data"]["val_frac"], cfg["seed"])
    norm = Normalizer().fit(train_units, per_condition={"cmapss"} & set(names))
    L = cfg["data"]["window"]
    width = max(u.S for u in units) * 5

    def make(us, last_window_only: bool = False):
        Xs, ys, dss, scales = [], [], [], {}
        for u in us:
            c = ds_cfg[u.dataset]
            X, y = window_unit(norm.apply(u), u.capped_rul(c["cap_mode"], c["cap"]), L, 1)
            if len(X) == 0:
                continue
            if last_window_only:      # published C-MAPSS protocol: one per engine
                X, y = X[-1:], y[-1:]
            Xs.append(pad_to(summarize(X), width))
            ys.append(y)
            dss += [u.dataset] * len(y)
            scales.setdefault(u.dataset, []).append(u.life_scale(c["cap_mode"], c["cap"]))
        return (np.concatenate(Xs), np.concatenate(ys), np.array(dss),
                {k: float(np.mean(v)) for k, v in scales.items()})

    Xtr, ytr, _, _ = make(train_units)
    Xva, yva, dsva, scales = make(val_units)
    print(f"baseline: {Xtr.shape[0]} train windows, {Xtr.shape[1]} features")

    rf = RandomForestRegressor(n_estimators=a.n_estimators, n_jobs=-1,
                               random_state=cfg["seed"], min_samples_leaf=5)
    rf.fit(Xtr, ytr)
    pred = np.clip(rf.predict(Xva), 0.0, 1.0)

    def report(pred, y_true, ds_of_row, label):
        res = {}
        for ds in np.unique(ds_of_row):
            m = ds_of_row == ds
            s = scales.get(ds, 1.0)
            p, t = pred[m] * s, y_true[m] * s
            res[str(ds)] = {"rmse": rmse(p, t), "mae": mae(p, t), "n": int(m.sum()),
                            "nasa": nasa_score(p, t) if ds == "cmapss" else None}
            print(f"  {label} {ds}: rmse={res[str(ds)]['rmse']:.2f} "
                  f"mae={res[str(ds)]['mae']:.2f} n={res[str(ds)]['n']}")
        res["mean"] = {"rmse": float(np.mean([v["rmse"] for v in res.values()]))}
        print(f"  {label} MEAN rmse={res['mean']['rmse']:.3f}")
        return res

    out = report(pred, yva, dsva, "val")

    if test_units:
        Xte, yte, dste, _ = make(test_units, last_window_only=True)
        pred_te = np.clip(rf.predict(Xte), 0.0, 1.0)
        out = {"val": out, "official_test": report(pred_te, yte, dste, "test")}

    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
