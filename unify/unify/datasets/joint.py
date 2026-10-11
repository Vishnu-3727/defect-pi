"""Joint multi-dataset Dataset, unit-level splitting, and the padding collate.

Two things here carry the project's core claims:

1. `split_units` splits by UNIT, never by window. Adjacent windows overlap by
   ~97%, so random window splitting puts near-identical samples in train and test
   and your RMSE becomes fiction. If your numbers beat published state of the
   art, look here first.

2. `collate` pads a batch to its largest sensor count and returns a mask. That is
   what lets one batch contain a 10-channel FEMTO window and a 14-channel C-MAPSS
   window at the same time. Without the mask the model attends to zero-padding
   and validation loss starts depending on batch composition.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..common.windowing import window_unit


def split_units(units: list, val_frac: float, seed: int) -> tuple[list, list]:
    """Unit-level split, stratified per dataset so every dataset appears in both."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    by_ds: dict[str, list] = {}
    for u in units:
        by_ds.setdefault(u.dataset, []).append(u)

    for ds, us in sorted(by_ds.items()):
        idx = rng.permutation(len(us))
        n_val = max(1, int(round(val_frac * len(us)))) if len(us) > 1 else 0
        val += [us[i] for i in idx[:n_val]]
        train += [us[i] for i in idx[n_val:]]
    return train, val


def hold_out_official_test(units: list) -> tuple[list, list]:
    """Separate units that belong to a dataset's own published test split.

    C-MAPSS ships train_FDxxx.txt and test_FDxxx.txt, and every published RMSE is
    measured on the latter. The parser caches both, so without this the random
    80/20 in split_units dissolves the official test units into training — the
    model would be fitted on the exact engines the reported number is supposed to
    hold out, and the result could not be compared to any paper.

    Keyed on the unit_id the parsers already emit ("cmapss_FD001_test_u1").

    Which datasets this actually catches, because the name oversells it:

    * cmapss  — yes, the published test files.
    * synthetic / synthetic_small — yes, and deliberately. The generators draw
      their test partition from an independent seed (`seed + 1`), so those units
      are genuinely unseen, not a relabelling of the training ones. Holding them
      out costs half the synthetic training pool, which is why synthetic val RMSE
      is worse than it was before this function existed. Those numbers never
      appear in results (see config/datasets.yaml) so the uniform protocol is
      worth more than the accuracy.
    * femto — only when data/raw/femto/test_rul.csv is supplied; without it the
      parser skips test bearings entirely and there is nothing to hold out.
    * xjtu — no. It has no published split, so everything falls through to the
      ordinary random split.
    """
    held = [u for u in units if "_test_" in u.unit_id]
    pool = [u for u in units if "_test_" not in u.unit_id]
    return pool, held


def leave_one_out(units: list, dataset: str):
    """Leave-one-bearing-out folds. With 15 XJTU bearings a single random split
    is statistically meaningless; LOBO gives you an error bar."""
    targets = [u for u in units if u.dataset == dataset]
    rest = [u for u in units if u.dataset != dataset]
    for held in targets:
        yield [u for u in targets if u is not held] + rest, [held]


class WindowDataset(Dataset):
    """Windows from a set of units, already normalised, with normalised targets."""

    def __init__(self, units: list, normalizer, ds_cfg: dict, L: int, stride: int,
                 last_window_only: bool = False):
        """last_window_only implements the published C-MAPSS test protocol.

        A test engine's series is truncated at a random point before failure and
        RUL_FDxxx.txt gives its remaining life at that final cycle — so the task
        is ONE prediction per engine, at the end of its record. Scoring every
        window instead averages in ~120 easy early-life predictions per engine,
        which flatters RMSE and inflates the NASA score by the window count, and
        produces a number no published result can be compared against.
        """
        self.items: list[dict] = []
        self.per_dataset_counts: dict[str, int] = {}
        self.scales: dict[str, float] = {}
        contributing: dict[str, int] = {}

        for u in units:
            cfg = ds_cfg[u.dataset]
            sig = normalizer.apply(u)                             # (T, S) z-scored
            y = u.capped_rul(cfg["cap_mode"], cfg["cap"])         # (T,) in [0,1]
            X, yw = window_unit(sig, y, L, stride)                # (N, S, L), (N,)
            if len(X) == 0:
                continue

            sids = torch.from_numpy(u.sensor_ids)
            keep = [len(X) - 1] if last_window_only else range(len(X))
            for i in keep:
                self.items.append({
                    "x": torch.from_numpy(X[i]),
                    "y": float(yw[i]),
                    "ids": sids,
                    "dom": u.domain_id,
                    "ds": u.dataset,
                    "unit": u.unit_id,
                })
            n_kept = 1 if last_window_only else len(X)
            self.per_dataset_counts[u.dataset] = \
                self.per_dataset_counts.get(u.dataset, 0) + n_kept
            # life_scale turns normalised RUL back into native units. Averaged over
            # units because at inference we do not know which specific unit we face.
            s = u.life_scale(cfg["cap_mode"], cfg["cap"])
            self.scales[u.dataset] = self.scales.get(u.dataset, 0.0) + s
            contributing[u.dataset] = contributing.get(u.dataset, 0) + 1

        # Average over the units that actually produced windows. Dividing by ALL
        # units of the dataset — including ones shorter than L, which contributed
        # nothing to the sum — silently deflates the scale and therefore every
        # physical-unit metric reported downstream.
        for ds in self.scales:
            self.scales[ds] /= max(contributing.get(ds, 1), 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def balanced_sampler(self) -> WeightedRandomSampler:
        """Inverse-frequency weights per dataset.

        C-MAPSS yields ~18k windows from 100 engines; XJTU yields far fewer. Without
        this the "universal" model is really a C-MAPSS model that occasionally
        guesses about bearings.
        """
        w = [1.0 / self.per_dataset_counts[it["ds"]] for it in self.items]
        return WeightedRandomSampler(w, num_samples=len(self.items), replacement=True)


def collate(batch: list[dict]):
    """Pad to the batch's largest S and build the attention mask."""
    B = len(batch)
    S_max = max(b["x"].shape[0] for b in batch)
    L = batch[0]["x"].shape[1]

    x = torch.zeros(B, S_max, L, dtype=torch.float32)
    mask = torch.zeros(B, S_max, dtype=torch.bool)
    ids = torch.zeros(B, S_max, dtype=torch.long)       # 0 == PAD in sensor_vocab.json

    for i, b in enumerate(batch):
        S = b["x"].shape[0]
        x[i, :S] = b["x"]
        mask[i, :S] = True
        ids[i, :S] = b["ids"]

    assert mask.any(dim=1).all(), "a sample has zero valid sensors — softmax would NaN"

    return {
        "x": x, "mask": mask, "sensor_ids": ids,
        "domain_id": torch.tensor([b["dom"] for b in batch], dtype=torch.long),
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.float32),
        "ds": [b["ds"] for b in batch],
    }
