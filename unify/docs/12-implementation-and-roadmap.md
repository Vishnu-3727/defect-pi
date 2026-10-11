# Sections 14 & 15 — Implementation and Roadmap

## 14.1 Folder structure

```
unify-rul/
├── README.md                     project overview, how to run, results table
├── pyproject.toml                uv/pip metadata, pinned deps
├── uv.lock                       exact resolved versions (committed)
├── requirements-pi.txt           LIGHT deps for the Pi: numpy, onnxruntime, fastapi, uvicorn
├── Makefile                      make data / train / export / deploy / bench
├── config/
│   ├── default.yaml              L, stride, cap, d_model, lr, epochs, seed …
│   ├── sensor_vocab.json         global sensor-ID map (shared by trainer AND Pi)
│   └── datasets.yaml             paths, per-dataset caps, life_scale, unit
│
├── data/
│   ├── raw/{cmapss,femto,xjtu}/  downloaded archives, .gitignored
│   ├── interim/                  per-unit .npz after parsing (cached)
│   └── processed/                windowed .npy tensors + manifest.json
│
├── unify/                        THE PACKAGE — importable by trainer and Pi alike
│   ├── __init__.py
│   ├── common/
│   │   ├── features.py           RMS, kurtosis, crest… (mirrors the ESP32 C++)
│   │   ├── normalize.py          fit/apply z-score; save/load norm_stats.json
│   │   ├── windowing.py          sliding_window_view, unit-boundary safe
│   │   └── schema.py             the Unit dataclass — one schema to rule them all
│   ├── datasets/
│   │   ├── base.py               abstract parser → yields Unit objects
│   │   ├── cmapss.py             parse train/test/RUL txt, per-condition norm
│   │   ├── femto.py              walk acc_*.csv, extract features per snapshot
│   │   ├── xjtu.py               walk N.csv, extract features per snapshot
│   │   └── joint.py              JointDataset + collate_fn (pad + mask)
│   ├── models/
│   │   ├── encoder.py            SensorEncoder (shared 1-D CNN)
│   │   ├── attention.py          CrossAttentionPool (masked, multi-head)
│   │   ├── unify_rul.py          the full UnifyRUL module
│   │   └── losses.py             SmoothL1, weighted-asymmetric, NASA score, pinball
│   ├── train/
│   │   ├── train.py              main loop, per-dataset val logging, checkpointing
│   │   ├── evaluate.py           RMSE/MAE/NASA in PHYSICAL units, per dataset
│   │   └── baseline_rf.py        RandomForest baseline for the results table
│   └── deploy/
│       ├── export_onnx.py        eval() → export with dynamic_axes → verify → hash
│       ├── verify_onnx.py        PyTorch↔ORT equivalence + S∈{2,14,16,21} test
│       └── quantize.py           dynamic/static int8 + accuracy re-measure
│
├── edge/                         everything that runs ON THE PI
│   ├── app.py                    FastAPI: /ingest, /health, /metrics, /ws, /
│   ├── inference.py              ORT session, window deque, normalise, predict
│   ├── postprocess.py            EWMA smoothing, hysteresis alarm
│   ├── db.py                     SQLite schema, batched writes, retention job
│   ├── bench.py                  latency p50/p95/p99, RSS, temp, throttle
│   ├── static/                   index.html, chart.min.js (LOCAL, not CDN), app.js
│   ├── artifacts/                unify_rul.onnx, norm_stats.json, sensor_vocab.json
│   └── unify.service             systemd unit
│
├── firmware/esp32/               PlatformIO project
│   ├── platformio.ini            board, framework, pinned lib versions
│   ├── src/main.cpp              setup/loop, task creation
│   ├── src/mpu6050.cpp/.h        I2C init, burst read, DC removal, unit conversion
│   ├── src/features.cpp/.h       RMS/std/kurtosis/skew/crest — MIRRORS unify/common/features.py
│   ├── src/net.cpp/.h            Wi-Fi connect, HTTP POST, store-and-forward buffer
│   ├── src/config.h              SSID, PI_URL, FS_HZ, WINDOW_N, ACCEL_RANGE
│   └── test/test_features.cpp    asserts against golden values from Python
│
├── notebooks/
│   ├── 01_explore_cmapss.ipynb   sensor traces, constant-sensor detection
│   ├── 02_explore_bearings.ipynb RMS/kurtosis vs life for all 32 bearings
│   └── 03_results.ipynb          all figures for the report
│
├── tests/
│   ├── test_shapes.py            S∈{2,14,16,21} → same output shape
│   ├── test_windowing.py         no window spans a unit boundary
│   ├── test_normalize.py         save/load round-trip; stats fit on train only
│   ├── test_features_parity.py   Python features == ESP32 golden values
│   └── test_onnx_parity.py       PyTorch == ORT within 1e-4
│
├── results/
│   ├── metrics.json              per-dataset RMSE/MAE/NASA, all runs
│   ├── figures/                  loss curves, predicted-vs-true, attention maps
│   └── bench_pi.json             latency/memory/thermal on the Pi
│
└── docs/                         ← these mentor notes, plus the report
```

### Why this structure

- **`unify/` is a package, not loose scripts.** `from unify.common.normalize import apply` works identically on the laptop and the Pi. That single fact eliminates train/serve skew.
- **`edge/` contains only what ships.** If a file in `edge/` imports torch, you have a bug.
- **`data/raw/` is `.gitignore`d.** Never commit 15 GB of bearing CSVs.
- **`tests/` exists.** Four small test files prove your three headline claims (variable sensor count, no leakage, export fidelity). That is the cheapest credibility you will ever buy.

---

## 14.2 Every important file, explained

### `unify/common/schema.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Unit:
    unit_id: str          # "cmapss_FD001_u17"
    dataset: str          # "cmapss" | "femto" | "xjtu"
    signals: np.ndarray   # (T, S) float32
    sensor_ids: list[int] # length S, indices into sensor_vocab
    rul: np.ndarray       # (T,) float32, native units, uncapped
    life_total: float
    unit_of_time: str     # "cycle" | "second" | "minute"
```

Every parser returns `list[Unit]`. Everything downstream consumes only this. Adding a fourth dataset = writing one parser.

### `unify/datasets/cmapss.py`

Parses the space-separated text, drops zero-variance sensors, clusters the 3 operational settings with KMeans(6) for FD002/FD004, computes `RUL = max_cycle − cycle` per unit. Returns `Unit` objects.

### `unify/datasets/femto.py` / `xjtu.py`

Walks the snapshot CSVs in numeric order (**sort numerically, not lexically** — `acc_10.csv` sorts before `acc_2.csv` as a string, and this bug silently scrambles your time series), computes the feature vector per snapshot via `common/features.py`, stacks into `(T, S)`. Caches to `data/interim/<unit_id>.npz`.

### `unify/datasets/joint.py`

```python
def collate_fn(batch):
    S_max = max(b["x"].shape[0] for b in batch)
    B, L = len(batch), batch[0]["x"].shape[1]
    x    = torch.zeros(B, S_max, L)
    mask = torch.zeros(B, S_max, dtype=torch.bool)
    ids  = torch.zeros(B, S_max, dtype=torch.long)
    for i, b in enumerate(batch):
        S = b["x"].shape[0]
        x[i, :S], mask[i, :S], ids[i, :S] = b["x"], True, b["ids"]
    return x, mask, ids, torch.tensor([b["dom"] for b in batch]), torch.stack([b["y"] for b in batch])
```

The `JointDataset` concatenates windows from all three datasets and shuffles. Consider a `WeightedRandomSampler` so C-MAPSS (which has far more windows) does not dominate — inverse-frequency weights per dataset.

### `unify/train/train.py`

Argparse or Hydra over `config/default.yaml`. Logs per-dataset validation RMSE every epoch. Saves `best.pt` on the mean of per-dataset RMSE. Fixed seed, logged.

### `unify/deploy/export_onnx.py`

Loads `best.pt` → `model.eval()` → export with `dynamic_axes` → `onnx.checker` → parity test → `sha256sum` → writes `norm_stats.json` and copies `sensor_vocab.json` into `edge/artifacts/`. **All in one script**, so the three artefacts can never drift apart.

### `edge/app.py`

FastAPI with `/ingest` (POST, Pydantic-validated), `/health`, `/metrics`, `/ws` (WebSocket), `/` (dashboard). Holds the deque, ORT session and DB handle in module state.

### `firmware/esp32/src/features.cpp`

Must produce bit-comparable results to `unify/common/features.py`. Golden-value test:

```cpp
// test/test_features.cpp — 1000 known samples, expected values from Python
TEST_CASE("rms matches python") { assert(fabs(rms(golden, 1000) - 0.041237f) < 1e-4f); }
```

---

## 14.3 How the modules talk

```
unify/datasets/*  ──Unit──▶  unify/datasets/joint.py  ──tensors──▶  unify/train/train.py
                                                                          │ best.pt
                                                                          ▼
                                                             unify/deploy/export_onnx.py
                                                                          │
                                    ┌─────────────────────────────────────┘
                                    │ .onnx + norm_stats.json + sensor_vocab.json
                                    ▼
firmware/esp32 ──HTTP JSON──▶  edge/app.py ──▶ edge/inference.py ──▶ edge/postprocess.py
                                    │                                        │
                                    └──▶ edge/db.py (SQLite)                 └──▶ /ws ──▶ browser

shared code:  unify/common/*  imported by BOTH unify/train and edge/inference
```

---

## 15.1 Twelve-week roadmap (from 7 Aug 2026)

Assumes ~10–12 hours/week between two people. Adjust dates to your review calendar; the ordering is what matters.

### Week 1 — 7–13 Aug: End-to-end skeleton with a stub model

**Goal: a number moves on a screen, driven by a real sensor, by Sunday.**

- ESP32: I2C read at 1 kHz, DC removal, compute RMS + kurtosis, POST JSON at 1 Hz.
- Pi: FastAPI `/ingest`, deque of 30, **stub `predict()` returning 42.0**, SQLite insert, minimal HTML page.
- Verify with `curl` before writing firmware.

**Deliverable:** live dashboard showing a constant 42 and real feature traces.
**Why first:** integration is where projects die. Kill that risk in week 1 while you have time.

### Week 2 — 14–20 Aug: Data parsers and the common schema

- `schema.py`, `cmapss.py`, `femto.py`, `xjtu.py` → `Unit` objects, cached to `interim/`.
- Assert per unit: RUL monotonically decreasing, ends at 0, no NaN.
- Notebook `02`: plot RMS/kurtosis vs life for all 32 bearings and 5 engines.

**Deliverable:** `python -m unify.datasets.build_all` produces `interim/` for all three datasets; plots in `notebooks/`.

### Week 3 — 21–27 Aug: Preprocessing and the DataLoader

- z-score fit-on-train, save/load `norm_stats.json`; per-condition normalisation for FD002/FD004.
- Sliding window (unit-boundary safe), RUL cap, RUL normalisation to [0,1].
- `JointDataset` + `collate_fn` with padding and mask.
- Tests: `test_windowing.py`, `test_normalize.py`.

**Deliverable:** one batch printed with shapes `(256, S_max, 30)`, mask correct, mixed datasets in one batch.

### Week 4 — 28 Aug–3 Sep: Model, and overfit one batch

- `encoder.py`, `attention.py`, `unify_rul.py`, `losses.py`.
- `test_shapes.py`: S ∈ {2,14,16,21} → identical output shape.
- **Overfit 8 samples to loss ≈ 0.** Do not proceed until this passes.

**Deliverable:** model trains on a tiny subset; shape test green.

### Week 5 — 4–10 Sep: First real training run + baseline

- Full training: AdamW, 1e-3, cosine, 100 epochs, early stopping.
- Per-dataset validation RMSE logged and plotted.
- `baseline_rf.py`: RandomForest on hand-crafted features.

**Deliverable:** first `results/metrics.json`. Target: C-MAPSS FD001 RMSE < 20. Do not panic if it is 25 — tune next week.

### Week 6 — 11–17 Sep: Tune, ablate, and lock the model

- Sweep: L ∈ {20,30,50}, d_model ∈ {32,64}, cap ∈ {110,125,140}, with/without per-condition norm, with/without sensor embedding.
- Ablation table: each preprocessing step's contribution.
- 3 seeds, report mean ± std.

**Deliverable:** the ablation table (a whole slide), and a locked `best.pt`. Target FD001 RMSE 13–17.

### Week 7 — 18–24 Sep: ONNX export and verification

- `export_onnx.py` with `dynamic_axes`, checker, parity test, hashing.
- `verify_onnx.py`: S ∈ {2,14,16,21} + numerical equivalence.
- Netron screenshot for the report.

**Deliverable:** `edge/artifacts/` populated; `pytest tests/test_onnx_parity.py` green.

### Week 8 — 25 Sep–1 Oct: Real inference on the Pi

- Install `requirements-pi.txt` in a fresh venv on the Pi; swap the stub for the real ORT session.
- `bench.py`: p50/p95/p99, RSS, `vcgencmd measure_temp`, `get_throttled`.
- Thread-count sweep {1,2,3,4}.
- systemd service enabled.

**Deliverable:** `results/bench_pi.json` + the latency chart. This closes the "Any Hardware" claim.

### Week 9 — 2–8 Oct: Live rig experiments

- Record baseline (motor off), healthy (10 min), imbalance (10 min), heavy imbalance (10 min) — **three trials each**.
- Calibrate the health index against the healthy baseline; set the alarm threshold at mean − 3σ.
- EWMA smoothing + N-consecutive hysteresis.

**Deliverable:** a plot showing the health index dropping repeatably when the fault is induced. This is your live-demo evidence.

### Week 10 — 9–15 Oct: Dashboard, robustness, alerting

- WebSocket live charts, attention bar panel, status strip, alert banner.
- Staleness detection, warm-up state, ESP32 store-and-forward, LED status.
- `/health` and `/metrics` endpoints; SQLite retention job.
- `--source replay` fallback mode.

**Deliverable:** the full dashboard; robustness table demonstrated.

### Week 11 — 16–22 Oct: Improvements and the results package

Pick two or three, in this priority order:

1. Quantile regression head → uncertainty intervals → alarm on the 10th percentile (**solves P4**).
2. int8 quantisation study (size/latency/RMSE table).
3. Leave-one-bearing-out cross-validation for FEMTO/XJTU.
4. Self-attention among sensors before pooling.
5. Benchmark on a second device.

Then: all figures for the report, the specialist-vs-universal comparison table, the ablation table.

**Deliverable:** `notebooks/03_results.ipynb` producing every figure reproducibly.

### Week 12 — 23–29 Oct: Report, slides, rehearsal

- Report: abstract, literature (your three base papers + their gaps), methodology, results, deployment, limitations, future scope.
- Slides following the section structure of these notes.
- **Rehearse the 11-step demo three times.** Record a video fallback.
- Run through the viva bank in `13-defense-and-industry.md` out loud, both of you.

**Deliverable:** submission-ready package.

### Buffer — 30 Oct onward

Something will slip. Weeks 11–12 have the most compressible content. Protect weeks 1, 4, 7 and 8 — they are the structural ones.

---

## 15.2 Division of labour (two people)

| | Vishnu | Kamlesh |
|---|---|---|
| Weeks 1–3 | Pi service, FastAPI, SQLite, dashboard skeleton | ESP32 firmware, I2C, features, Wi-Fi |
| Weeks 2–4 | — | Dataset parsers, preprocessing, DataLoader |
| Weeks 4–6 | Model, training, tuning | Data exploration notebooks, RF baseline, ablations |
| Weeks 7–8 | ONNX export, verification, Pi benchmarking | Rig assembly, mounting, motor experiments |
| Weeks 9–10 | Dashboard, robustness, alerting | Live experiments, calibration, data collection |
| Weeks 11–12 | Improvements, figures | Report writing, slides |

**Both must understand everything.** In viva, examiners deliberately ask each member about the other's part. Swap explanations weekly: each of you explains the other's module out loud for 10 minutes.

---

## 15.3 What to have working before each review

| Review | Minimum bar | Ideal |
|---|---|---|
| Review 1 (done, 11 Jul) | Datasets downloaded, preprocessing, model designed | ✔ as per your slides |
| Review 2 (~mid Sep) | Model trained, per-dataset metrics, hardware acquiring live | Ablation table, RF baseline, live dashboard with a stub or real model |
| Review 3 (~mid Oct) | ONNX on the Pi, live end-to-end, latency measured | Fault-injection demo, quantisation study, robustness features |
| Final | Everything + report | Second-device benchmark, uncertainty intervals, LOBO cross-validation |

---

## Summary

- One importable package (`unify/`) shared by trainer and Pi; `edge/` contains only what ships; `firmware/` mirrors the feature code in C++.
- Four small test files prove the three headline claims.
- Twelve weeks: skeleton → data → preprocessing → model → training → tuning → export → Pi → rig experiments → dashboard → improvements → report.
- Build the end-to-end skeleton with a stub model in week 1. Integration risk first, always.

## Key takeaways

1. Week 1 delivers a moving number on a screen. Everything else is easier after that.
2. `unify/common/` imported by both sides is the structural defence against train/serve skew.
3. Overfit 8 samples before any real training run.
4. Protect weeks 1, 4, 7, 8. Compress 11 and 12 if you must.

## Common interview questions

1. How do you structure an ML project so training and serving cannot diverge?
2. Why cache intermediate parsed data?
3. What do you test in an ML codebase, given you cannot unit-test "accuracy"? (Shapes, invariants, leakage, export parity, feature parity.)
4. How do you version models and tie predictions to the model that made them?

## Common viva questions

1. "Show me your folder structure and explain what each part does." → Have it open.
2. "Which files run on the Pi?" → Only `edge/` plus `unify/common/`. Nothing importing torch.
3. "How do you ensure the ESP32's kurtosis matches Python's?" → The golden-value test in `firmware/test/`.
4. "What was your biggest technical challenge?" → Have a real, specific answer with a number in it. (Good candidates: the unit-normalisation problem across three time bases; masking in attention for variable sensor counts; motor EMI corrupting I2C.)

## Practical implementation tips

- `make` targets for everything. `make train`, `make export`, `make deploy`, `make bench`. Six months later, and in front of an examiner, you will be grateful.
- Commit `results/metrics.json` after every real run so you have a history of what worked.
- Write the report's methodology section in week 6, while the decisions are fresh. Writing it in week 12 from memory always loses detail.
- Keep a `LOGBOOK.md` with one dated line per session. It becomes your report's timeline and your viva's memory.

## Suggested improvements for this project

1. **Add CI** (GitHub Actions) running `pytest` on push. Two files, and it means your export-parity and shape tests actually run.
2. **Add `make demo`** that starts the replay source, the service and opens the browser — one command for a reliable demo.
3. **Publish the repo** (public, with a good README and the results table). A student project with tests, a Makefile and reproducible results is a genuinely strong portfolio artefact.
