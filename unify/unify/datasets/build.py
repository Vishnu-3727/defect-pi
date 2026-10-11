"""Parse every configured dataset once, cache to data/interim/*.npz.

FEMTO is ~5 GB of tiny CSVs and XJTU ~10 GB. Re-parsing them on every training
run wastes hours. Parse once, then training loads only the cache.

    python -m unify.datasets.build                     # everything configured
    python -m unify.datasets.build --datasets synthetic
    python -m unify.datasets.build --force             # ignore the cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..common.schema import Unit

ROOT = Path(__file__).resolve().parents[2]
LOADERS = {}


def _loaders():
    if LOADERS:
        return LOADERS
    from . import cmapss, femto, synthetic, xjtu
    LOADERS.update({
        "cmapss": lambda c, s: cmapss.load(c, s),
        "femto": lambda c, s: femto.load(c, s),
        "xjtu": lambda c, s: xjtu.load(c, s),
        "synthetic": lambda c, s: synthetic.load(c, s),
        "synthetic_small": lambda c, s: synthetic.load_small(c, s),
    })
    return LOADERS


def dataset_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "datasets.yaml").read_text())


def build(names: list[str], interim: Path, force: bool = False) -> dict[str, int]:
    ds_cfg = dataset_config()
    counts: dict[str, int] = {}

    for name in names:
        out = interim / name
        if out.exists() and any(out.glob("*.npz")) and not force:
            counts[name] = len(list(out.glob("*.npz")))
            print(f"{name}: cached, {counts[name]} units (--force to rebuild)")
            continue

        cfg = ds_cfg[name]
        units: list[Unit] = []
        for split in ("train", "test"):
            try:
                units += _loaders()[name](cfg, split)
            except FileNotFoundError:
                pass

        if not units:
            print(f"{name}: NO DATA FOUND under {cfg.get('raw', '(generated)')} — skipping")
            counts[name] = 0
            continue

        out.mkdir(parents=True, exist_ok=True)
        for u in units:
            u.save(out / f"{u.unit_id}.npz")
        counts[name] = len(units)
        T = sum(u.T for u in units)
        print(f"{name}: {len(units)} units, {T} timesteps, S={units[0].S} -> {out}")

    return counts


def load_cached(names: list[str], interim: Path) -> list[Unit]:
    units: list[Unit] = []
    for name in names:
        d = interim / name
        if not d.exists():
            continue
        units += [Unit.load(p) for p in sorted(d.glob("*.npz"))]
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["cmapss", "femto", "xjtu", "synthetic", "synthetic_small"])
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(a.datasets, ROOT / a.interim, a.force)


if __name__ == "__main__":
    main()
