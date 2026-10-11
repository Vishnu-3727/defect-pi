# Real-Time Industrial Defect Detection System

Surface defect inspection for steel components, running on a Raspberry Pi 5 with
no GPU. A camera watches parts on a conveyor, YOLOv8n localises defects, an
unsupervised anomaly model gives a second opinion on uncertain frames, and a
solenoid ejects the rejects. Every decision is logged to SQLite and shown on a
web dashboard.

ECE Project Based Learning, Chennai Institute of Technology.

---

## Status

| Stage | State |
|-------|-------|
| Dataset (NEU-DET, 1770 images, 6 classes) | done |
| YOLOv8n training at 320x320 | done — **v2**, test mAP@50 **0.7624** (PyTorch); deployed INT8 artefact **0.7353**, `reports/model-v2.md` |
| Evaluation on held-out test split | done — `reports/metrics.md` |
| Resolution study | done — 640 is **not** better; train-high/deploy-low refuted, `reports/resolution-retrain.md` |
| ONNX export and INT8 quantisation | done — 0.3 mAP cost, `reports/benchmark.md` |
| ONNX pipeline parity vs ultralytics | done — 132/132 boxes, IoU 1.0000 |
| CPU benchmark, x86 | done — 50.6 FPS at 4 threads. **x86-only; does not predict the Pi** |
| CPU benchmark, ARM | done on the Pi 5 — **36.7 FPS** INT8 at 4 threads, `reports/benchmark.md` |
| Per-stage profile | done — `scripts/profile_stages.py`; storage and failure JPEGs are 2% of the budget, not the bottleneck |
| PatchCore anomaly module | done — weak separation, `reports/anomaly.md` |
| SQLite, dashboard, actuator, pipeline | done — self-checks pass on the Pi |
| Dashboard | done — windowed rates, p95, SoC health, failure-frame viewer |
| Arduino ejector firmware | written, **not yet tested on hardware** |
| Camera | **not connected** — no USB or CSI device present on the Pi |
| Hardware integration | Pi 5 deployed and running; camera, Arduino, relay and solenoid outstanding |
| Power supply | **faulty — board is under-volting, see below** |

The Pi 5 is deployed: the repository runs on it, the dashboard serves from it,
and the ARM benchmark above was measured on it. What is still simulated is
everything downstream of the decision — no camera is attached, so frames come
from `bench_fixtures/images` by folder replay, and the actuator runs the `mock`
backend that logs instead of firing. Switching either to real hardware is a
one-line config change.

> **Power warning.** As of 2026-09-18 this board is under-volting: `dmesg`
> reports `Undervoltage detected!`, `vcgencmd get_throttled` reaches `0x50005`
> on the live bits, and the ARM clock drops from 2400 MHz to 1000 MHz at 62 C
> with the fan running. The FPS figures above predate this and are not currently
> reproducible. Fix the supply — official 27 W USB-C PD — before taking any
> further measurement. `scripts/profile_stages.py` warns while the fault
> persists.

---

## Quick start

```bash
uv sync --extra train                      # laptop, with GPU training deps
uv run python scripts/fetch_dataset.py     # sparse-clone NEU-DET, ~45 MB
uv run python scripts/prepare_dataset.py   # VOC XML -> YOLO, stratified 70/20/10
uv run python scripts/train.py --epochs 100
uv run python scripts/evaluate.py          # writes reports/metrics.md
uv run python scripts/export_onnx.py --int8
```

On the Raspberry Pi, install the runtime only — no PyTorch, no CUDA:

```bash
uv sync --extra pi
uv run python scripts/run_live.py --source 0
```

## Measuring on ARM

The laptop is x86-64 with AVX-512 and VNNI; the Pi 5 is four Cortex-A76 cores
with 128-bit NEON. Laptop FPS carries no information about the Pi, so the
benchmark must run on ARM to mean anything.

```bash
bash scripts/bench_remote.sh ubuntu@<arm-host-ip> ~/.ssh/key
```

Ships an 18 MB bundle, installs runtime dependencies only, refuses to run if
the remote host is not `aarch64`, and prints the same tables the laptop does.
`.github/workflows/bench-arm.yml` does the same thing on GitHub's free
`ubuntu-24.04-arm` runners, but those require a public repository.

## Self-checks

Every non-trivial module carries a runnable check:

```bash
uv run python -m inspector.config          # config validation
uv run python -m inspector.preprocess      # letterbox round trip, channel order
uv run python scripts/prepare_dataset.py --self-check
```

---

## Architecture

```
 camera ─► preprocess ─► YOLOv8n (ONNX INT8) ─► confident? ─yes─► decision
                                                    │                │
                                                   no                │
                                                    ▼                │
                                             PatchCore ──────────────┤
                                                                     ▼
                                          SQLite ◄── PASS / FAIL ──► actuator
                                             │                          │
                                             ▼                     Arduino ─► relay ─► solenoid
                                         dashboard
```

Every arrow is explained in [docs/18_system_architecture.md](docs/18_system_architecture.md).

---

## Repository layout

| Path | Contents |
|------|----------|
| `src/inspector/` | The deployable runtime. Imports only what runs on a Pi. |
| `scripts/` | One-shot tools: fetch, prepare, train, evaluate, export. May assume a GPU. |
| `configs/default.yaml` | Every tunable value, validated on load. |
| `dashboard/` | Flask web interface. |
| `firmware/ejector/` | Arduino Nano sketch for the ejector. |
| `docs/` | A 22-module course covering the whole project from first principles. |
| `reports/` | Generated metrics and plots. |

The package is named `inspector`, not `inspect`, because `inspect` is a Python
standard-library module and would be shadowed.

---

## Documentation

Start at [docs/00_start_here.md](docs/00_start_here.md), which contains the
reading order and a dependency graph. The modules assume no prior knowledge and
teach prerequisites before dependants.

---

## Dataset

NEU-DET: 1770 grayscale images, 200x200 px, six defect classes — crazing,
inclusion, patches, pitted surface, rolled-in scale, scratches. Sourced via
sparse clone from a public mirror; `scripts/fetch_dataset.py` handles it.

Median defect box covers 11.8% of image area, and only 1.2% of boxes fall below
1%. These are large, diffuse regions, not micro-scale features — which is why
the model uses the standard detection heads rather than an added
high-resolution small-object head.

## Run the dashboard (Pi or laptop)

One launcher per OS opens the dashboard, using the Raspberry Pi if it is
reachable, otherwise running it on the laptop:

```powershell
.\run-dashboard.ps1            # Windows (or double-click run-dashboard.cmd)
```

```bash
bash run-dashboard.sh          # macOS / Linux
```

The Pi is `vishnu@defect-pi.local` by default; override with `PI_HOST` or a
first argument (e.g. `bash run-dashboard.sh 192.168.1.8`), or pass `local`
to force laptop mode. With no explicit host the launchers try
`defect-pi.local` then the last good IP saved in `.pi_last_ip`, using the
first host that answers on TCP 22; the answering host's IPv4 is saved back
to `.pi_last_ip` for next time. Pi mode starts the demo with
`ssh vishnu@<host> demo start defect-inspect` and stops it with
`ssh vishnu@<host> demo stop`. Laptop mode runs
`uv run python dashboard/app.py --source bench_fixtures/images --loop`.
