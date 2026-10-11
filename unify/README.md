# UNIFY-RUL

Universal Remaining Useful Life prediction — any dataset, any hardware.
ECE PBL, Chennai Institute of Technology, AY 2026-27.

One 44k-parameter model trained jointly on C-MAPSS, FEMTO and XJTU-SY, accepting
any sensor count without architectural change, exported to ONNX with dynamic
sensor axes, running on a CPU-only ARM edge device.

Design rationale and the full mentor curriculum: [`docs/00-index.md`](docs/00-index.md).

## Prove it works in five minutes, with no downloads

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/Pi: .venv/bin/activate
pip install -r requirements-train.txt

make smoke        # or run the four commands below by hand
```

`make smoke` runs the entire stack on generated run-to-failure data:

```
python -m unify.datasets.build --datasets synthetic synthetic_small
python -m unify.train.train    --datasets synthetic synthetic_small --overfit
python -m unify.train.train    --datasets synthetic synthetic_small --epochs 40
python -m unify.deploy.export_onnx --quantize
python -m pytest tests/ -q
```

Then watch it run end to end:

```bash
# terminal 1
cd edge && UNIFY_LIVE_MAP=../config/live_map.synthetic.json \
  python -m uvicorn app:app --host 0.0.0.0 --port 8000

# terminal 2 — stream a real cached unit from healthy to failure
python edge/fake_node.py --replay data/interim/synthetic/synthetic_test_u0.npz --period 0.05
```

Open <http://localhost:8000>.

## The real datasets

Download into `data/raw/` (see `config/datasets.yaml` for the expected layout):

| Dataset | Where | Size | Status |
|---|---|---|---|
| NASA C-MAPSS | `phm-datasets.s3.amazonaws.com`, "6. Turbofan Engine Degradation Simulation Data Set" | ~12 MB | **downloaded** |
| FEMTO / PRONOSTIA | IEEE PHM 2012 Data Challenge | ~2–5 GB | manual |
| XJTU-SY | XJTU / Changxing Sumyoung bearing dataset | ~10 GB | manual |

C-MAPSS is scriptable — the S3 mirror is a plain HTTPS GET, no login:

```bash
curl -L -o cmapss.zip \
  "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
# it is a zip containing CMAPSSData.zip; extract the inner one into data/raw/cmapss/
```

The bearing sets are not. Both sit behind Google Drive / Mendeley landing pages
and need a browser. Until they are present, `make data` prints
`NO DATA FOUND … skipping` for them and training runs on what exists.

### How the splits work

Three splits, not two:

| Split | Built from | Used for |
|---|---|---|
| train | 80 % of `train_FDxxx.txt` units | gradient steps |
| val | 20 % of `train_FDxxx.txt` units | early stopping, checkpoint selection |
| official test | all of `test_FDxxx.txt` | the reported number, looked at once |

Splitting is by **unit**, never by window — two windows from the same engine
overlap by 29 of 30 timesteps, so a window-level split leaks almost perfectly.
The official test units are held out by `hold_out_official_test`, which keys on
the `unit_id` the parser emits, and scored with **one prediction per engine at
its final cycle** — the published C-MAPSS protocol. Scoring every window instead
averages in ~120 easy early-life predictions per engine and is not comparable to
any paper.

```bash
make data       # parse once, cache to data/interim (FEMTO/XJTU take a while)
make train
make baseline   # RandomForest, for the results table
make export
make test
```

## Layout

| Path | Runs where | What |
|---|---|---|
| `unify/common/` | **both** | schema, features, z-score, windowing — imported by the trainer *and* the Pi, so the two can never diverge |
| `unify/datasets/` | laptop | one parser per dataset, all emitting the same `Unit` |
| `unify/models/` | laptop | shared CNN encoder, masked cross-attention, head, losses |
| `unify/train/` | laptop | joint training, per-dataset evaluation, RF baseline |
| `unify/deploy/` | laptop | ONNX export + verification |
| `edge/` | **Pi** | FastAPI ingest, ONNX Runtime, SQLite, dashboard, benchmark |
| `firmware/esp32/` | ESP32 | 1 kHz sampling, features, Wi-Fi POST |
| `config/` | both | model/training config, sensor vocabulary, live channel map |
| `tests/` | CI | 39 tests |

Only three files ever go to the Pi: `unify_rul.onnx`, `norm_stats.json`,
`sensor_vocab.json`. Nothing in `edge/` may import torch.

## The claims, and the tests that hold them up

| Claim | Test |
|---|---|
| Any sensor count, one weight set | `tests/test_shapes.py::test_any_sensor_count_same_output_shape` (S = 2…21) |
| …survives ONNX export | `tests/test_onnx_parity.py` (S = 2, 10, 14, 21, matched to PyTorch within 3e-8) |
| A failed sensor doesn't break it | `test_mask_blocks_padded_sensors`, `test_masked_channel_cannot_inject_garbage` |
| No train/test leakage | `test_split_is_by_unit_not_by_window` |
| int8 quantisation is safe | `test_int8_quantisation_preserves_predictions` (2.6× smaller) |
| ESP32 and Python compute the same features | `tests/test_preprocessing.py` feature block |

## Measured

Latency is measured **on the target**, a Raspberry Pi 5 running 64-bit Raspberry
Pi OS (2026-08-09), with the synthetic build described above — an x86 laptop is
not an ARM proxy and its numbers do not belong in the report. Accuracy is the
C-MAPSS build, on the laptop.

| | |
|---|---|
| Parameters | 44,417 |
| Model, fp32 | 190 KB |
| Model, int8 | 79 KB (2.4× smaller) |
| Inference p50 / p95 / p99 (Pi 5, 4 threads) | **0.423 / 0.441 / 0.446 ms** |
| Inference p50 (Pi 5, 1 thread) | 0.539 ms |
| ONNX vs PyTorch max abs diff | 1.8e-7 |

### Accuracy, real data

C-MAPSS, all four subsets trained jointly as one model with one weight set.
Scored on the published test files — 690 engines, one prediction each at the
final cycle. 17 of the 707 test engines are shorter than the 30-cycle window and
cannot be scored at all.

| | RMSE (cycles) | MAE (cycles) | NASA score |
|---|---|---|---|
| UNIFY-RUL | 14.51 | **9.99** | 3,417 |
| RandomForest baseline | **14.41** | 10.50 | — |

**Read this honestly: the baseline wins on RMSE.** By 0.1 cycles, which is
noise, so the fair statement is that they tie — while the deep model is clearly
better on MAE, 10.0 against 10.5. That combination has a specific meaning: the
deep model is better on the typical engine and worse on a few outliers, since
RMSE squares large errors and MAE does not. Find those engines before claiming
anything else.

That first run looked undertrained — it stopped at epoch 23 of a budgeted 100
with training loss still falling — so the obvious suspicion was that early
stopping had fired on noise. A sweep says otherwise:

| lr | patience | seed | best epoch | val RMSE | test RMSE | NASA |
|---|---|---|---|---|---|---|
| 1e-3 | 15 | 42 | 8 | 14.87 | 14.51 | 3,417 |
| 3e-4 | 30 | 42 | 13 | 14.81 | 14.81 | 3,826 |
| 1e-3 | 30 | 42 | 13 | 15.07 | 14.79 | 3,356 |
| 3e-4 | 30 | 1 | 18 | 16.80 | **14.45** | 2,941 |

Three things fall out of that table, and all three are worth defending in a viva:

1. **Test RMSE spans 14.45–14.81 across a 3× learning-rate range, double the
   patience, and two seeds.** A 0.36-cycle band. The hyperparameters are not the
   lever, and neither was the early-stopping rule — with patience 30 the 1e-3 run
   still peaked at epoch 13 and spent 30 more epochs finding nothing better.
2. **Every run peaks between epoch 13 and 18, then training loss keeps falling
   while validation degrades.** That is overfitting, not undertraining. 44k
   parameters exhaust 567 engines in about 15 epochs.
3. **Validation does not predict test here.** The run with the *worst* validation
   RMSE (16.80, seed 1) produced the *best* test RMSE (14.45). With only 142
   validation units, the selection signal is weaker than the noise. The shipping
   checkpoint is still chosen on validation, because choosing it on test is how
   the leakage this repo just removed gets reintroduced — but the honest reading
   is that the choice is close to arbitrary. Fixing it properly means k-fold or
   leave-one-out over the training units; `leave_one_out` in
   `unify/datasets/joint.py` is written and tested but not yet wired to a flag.

So: not a tuning problem. The remaining levers are regularisation, augmentation,
and the two datasets still sitting behind a browser download.

What the model does buy, and the baseline cannot: the RandomForest needs a fixed
column count, so `baseline_rf.py` pads every dataset to the widest one and
retrains per feature layout. UNIFY-RUL takes 15, 16 or 21 sensors through one
weight set. That is the claim in the title, and it holds regardless of which
number in the table is lower.

Two things worth saying out loud in the viva:

- **One core is enough.** Going from 1 thread to 4 buys 0.12 ms on a path where
  the 1-second feature window costs 1000 ms. The model is not the bottleneck and
  never will be; parallelism here is free headroom, not a requirement.
- **The tail is flat.** p99 is 5 % above p50. Compare a vision pipeline on the
  same board, where p95 runs ~2× the median at 4 threads. A 44k-parameter model
  that fits in cache does not thrash, so the latency distribution stays tight —
  which is what a control loop actually cares about.

Thermals during the sweep: 46.1 °C, `throttled 0x0`.

```bash
cd edge && python bench.py --sweep     # thread-count sweep on the Pi
```

## Flash the sensor node

1. Edit `firmware/esp32/src/config.h` — SSID, password, Pi IP. The ESP32 is
   **2.4 GHz only**; a 5 GHz hotspot will never connect.
2. `cd firmware/esp32 && pio run -t upload -t monitor`
3. MPU-6050 SDA→GPIO21, SCL→GPIO22, VCC→3V3. L298N ENA→25, IN1→26, IN2→27.
   Motor on its own 12 V supply, **all grounds common**.

## What this is not

The motor rig has **no ground-truth RUL** — you cannot run it to destruction in a
semester, and one bearing's death is one sample. The MPU-6050 also tops out at
1 kHz against the bearing datasets' 25.6 kHz, so it cannot see the kHz-region
resonances that bearing spalls excite.

So: **validated accuracy comes from the datasets; the rig demonstrates the live
pipeline and detects the low-frequency faults you induce.** Two claims, kept
separate. [`docs/11-realtime-pipeline.md`](docs/11-realtime-pipeline.md) §13.1
explains how to present this — leading with the limitation earns more credit than
being caught by it.

### Why the Pi runs a different model from the one in the results table

Two builds, on purpose:

| | Trained on | Runs where | What it is |
|---|---|---|---|
| Accuracy result | C-MAPSS | laptop | the number reported below |
| Live demo | synthetic generators | Raspberry Pi 5 | the pipeline, end to end |

The C-MAPSS model knows twenty-one *turbofan* sensor channels. The rig is a
*bearing* rig, and `config/live_map.json` maps its accelerometer features onto
bearing sensor IDs. Feeding a turbofan-trained model a vibration feature vector
would produce a number with no meaning — every sensor embedding it looks up
would be one it was never trained on. The synthetic build is the one whose
channel map matches, so it is the one on the Pi, and its role is exactly the
"demonstrates the live pipeline" claim above and nothing more.

This collapses to one model the moment FEMTO or XJTU is downloaded: a
bearing-trained build serves the rig, `UNIFY_LIVE_MAP` comes out of
`deploy/unify-rul.service`, and `live_map.json` goes back to its real
`ref_dataset: xjtu`.

## Next

- Train on the real datasets and fill in the results table
- `loss.kind: pinball` in `config/default.yaml` → prediction intervals, and alarm
  on the 10th percentile. That is the fix for problem P4 from Review 1.
- Benchmark on a second device so "Any Hardware" is demonstrated, not asserted
- Leave-one-bearing-out CV for FEMTO/XJTU — with 15 bearings a single split is
  statistically meaningless

Roadmap: [`docs/12-implementation-and-roadmap.md`](docs/12-implementation-and-roadmap.md).

## Run the dashboard (Pi or laptop)

```powershell
.\run-dashboard.ps1            # Windows, or double-click run-dashboard.cmd
```

```bash
./run-dashboard.sh             # macOS / Linux
./run-dashboard.sh local       # force laptop mode; $PI_HOST also forces one host
```

If the Pi (`defect-pi.local`, else the IP in `.pi_last_ip`) answers on port
22, the launcher uses it: it opens `http://<pi>:8000/` if `/health` answers
with a `features` key (UNIFY's marker), otherwise it runs `ssh vishnu@<pi>
demo start unify-rul` and waits up to 30 s (stop with `ssh vishnu@<pi> demo
stop`). Otherwise it starts the server plus the simulator locally with `uv`
on `$UNIFY_PORT` if set, else the first free port among 8000, 8010, 8020,
8030, and opens `http://127.0.0.1:<port>/`.

No ESP32 exists yet, so every reading is simulated: `edge/fake_node.py
--degrade --loop` posts a degradation ramp that restarts instead of holding at
failure. The dashboard says `LIVE · SIMULATED SENSOR` when the feed's node is
`fake`, so a demo is never mistaken for a real machine.
