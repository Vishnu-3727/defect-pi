# Section 8 — Software Stack

For each tool: what it is, where it sits in your project, and what you could have used instead.

## 8.1 The stack at a glance

```
LAPTOP (training)                          RASPBERRY PI (inference)          ESP32
─────────────────                          ────────────────────────          ─────
Python 3.11                                Python 3.11                       C++ (Arduino)
  ├─ NumPy        arrays                     ├─ NumPy      arrays              ├─ Wire.h    I2C
  ├─ Pandas       CSV parsing                ├─ onnxruntime  inference         ├─ WiFi.h    802.11
  ├─ SciPy        signal features            ├─ Flask/FastAPI  API+dashboard   ├─ HTTPClient
  ├─ scikit-learn splits, KMeans, metrics    ├─ sqlite3    historian           └─ ArduinoJson
  ├─ PyTorch      model + training           └─ Chart.js (browser)
  ├─ Matplotlib   plots
  └─ onnx         export
```

---

## 8.2 Python

**What.** Interpreted, dynamically-typed, garbage-collected language. Slow at raw loops, fast at *calling* fast things.

**Why here.** The entire ML ecosystem is Python-first. More importantly, the *same* language runs on your laptop and on the Pi, so preprocessing code (the z-score, the windowing) is literally the same file in both places — which eliminates the classic train/serve skew bug where the Python trainer and the C++ server disagree by a rounding rule.

**Version.** 3.11 or 3.12. Avoid 3.13 until every wheel you need has an aarch64 build.

**Environment.** Your slide says `uv + pyenv` with a lockfile — that is a genuinely good choice, better than most student projects. `uv` resolves and installs in seconds and produces a `uv.lock` that pins transitive dependencies. Keep two lockfiles: `requirements-train.txt` (heavy: torch, pandas, sklearn) and `requirements-pi.txt` (light: numpy, onnxruntime, flask). Installing PyTorch on the Pi is both unnecessary and painful.

**Alternatives.** Julia (fast, tiny ecosystem), C++ with LibTorch (fast, miserable to iterate in), MATLAB (used in industry for signal processing; licensing kills it).

---

## 8.3 NumPy

**What.** N-dimensional arrays with C-speed vectorised operations. The lingua franca — every other library speaks NumPy.

**Where in your project.** Loaded sensor arrays, windowing (see the stride trick below), z-scoring, the input tensor handed to ONNX Runtime, feature computation.

**Why it is fast.** A Python `for` loop over a million floats takes seconds; `arr.mean()` takes milliseconds because it is one call into compiled, SIMD-vectorised C. Rule: if you write a Python loop over samples, you are doing it wrong.

**The one trick worth knowing** — zero-copy sliding windows:

```python
from numpy.lib.stride_tricks import sliding_window_view
w = sliding_window_view(signals, window_shape=L, axis=0)   # (T-L+1, S, L), NO data copied
```

This turns windowing from a slow, memory-doubling loop into a view. On XJTU it is the difference between 40 seconds and 40 milliseconds.

**Alternatives.** None realistically. (CuPy for GPU arrays; irrelevant here.)

---

## 8.4 Pandas

**What.** Labelled 2-D tables (DataFrames) with SQL-like grouping and joining.

**Where.** Parsing C-MAPSS's space-separated text; the RUL label computation is a two-line `groupby`:

```python
df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
```

**Where NOT.** Do not use Pandas for the bearing datasets' 25.6 kHz waveforms or in the training loop — DataFrame overhead per row is enormous. Parse with Pandas, convert to NumPy immediately, never look back.

**Alternatives.** Polars (much faster, Rust-based, worth knowing about), plain `np.loadtxt` (fine for C-MAPSS honestly), DuckDB.

---

## 8.5 SciPy

**What.** Scientific computing on top of NumPy — signal processing, statistics, optimisation, FFT.

**Where.** Feature extraction for bearings:

```python
from scipy.stats import kurtosis, skew
from scipy.signal import welch, hilbert, butter, filtfilt
```

`kurtosis` and `skew` for time-domain features, `welch` for power spectral density, `hilbert` for envelope analysis, `butter`+`filtfilt` for band-pass filtering. If you add the envelope-spectrum features from `05-preprocessing.md`, this is where they come from.

**Note:** you did not list SciPy in your original stack. Add it — you need it for kurtosis alone.

**Alternatives.** Implement kurtosis by hand (5 lines, and you must anyway for the ESP32's C++ side — keep the two implementations consistent and unit-test them against each other).

---

## 8.6 scikit-learn

**What.** Classical ML: preprocessing, model selection, metrics.

**Where.**

- `train_test_split(groups=...)` / `GroupKFold` — unit-level splitting, so windows from one engine never straddle the split. This is the leakage guard from `05-preprocessing.md` §5.8.
- `KMeans(n_clusters=6)` — clustering C-MAPSS operating conditions for per-condition normalisation.
- `StandardScaler` — z-scoring, with `fit` on train and `transform` elsewhere. Its `.mean_` / `.scale_` are exactly what you serialise for the Pi.
- `mean_squared_error`, `mean_absolute_error`, `r2_score` — metrics.
- **A baseline model.** Fit a `RandomForestRegressor` on hand-crafted features and report its RMSE. If your deep model cannot beat it, you have learned something important. Every serious ML report includes a non-deep baseline; almost no student project does. Free credibility.

**Alternatives.** Do the z-score by hand with NumPy (arguably cleaner, since you control the serialisation format).

---

## 8.7 PyTorch

**What.** Deep-learning framework: tensors with GPU support, automatic differentiation (`autograd`), neural-network layers, optimisers.

**Why PyTorch over TensorFlow here.**

| | PyTorch | TensorFlow/Keras |
|---|---|---|
| Graph | Define-by-run (dynamic) | Static graph by default (TF2 has eager) |
| Debugging | Standard Python debugger works | Harder inside graph mode |
| Variable shapes | Natural — your S varies per batch | Requires more care |
| Research adoption | Dominant | Declining |
| ONNX export | `torch.onnx.export`, first-class | Works, extra conversion steps |
| Edge story | Via ONNX / ExecuTorch | TFLite is excellent (a genuine TF advantage) |

Dynamic shapes matter specifically for you: your batches contain different S values, and PyTorch simply doesn't care. Version 2.3 as pinned on your slide is fine.

**Where.** `models/`, `training/`. Laptop only — never installed on the Pi.

**Alternatives.** TensorFlow/Keras (TFLite would be a legitimate alternative deployment path — mention in viva that you evaluated ONNX vs TFLite and chose ONNX for framework independence), JAX (great, small ecosystem for this), fastai (too high-level).

---

## 8.8 ONNX and ONNX Runtime

Two different things; students conflate them and lose marks.

- **ONNX** = the *file format* and operator specification. A `.onnx` file is a protobuf containing a computation graph (nodes, weights, shapes). It is a **noun**, a document. Nothing executes it.
- **ONNX Runtime (ORT)** = the *engine* by Microsoft that loads a `.onnx` and executes it, with graph optimisations and hardware-specific backends.

Analogy: ONNX is a PDF; ONNX Runtime is Adobe Reader.

**Where.** Export at the end of training (`deployment/export_onnx.py`); ORT is the inference engine on the Pi.

Full treatment in `10-onnx-and-deployment.md`.

**Alternatives.** TensorFlow Lite (excellent on ARM, but you'd need to convert PyTorch → ONNX → TF → TFLite, which is fragile); Apache TVM (compiles to optimised native code, best performance, steep learning curve); TorchScript (keeps you inside PyTorch — 800 MB on the Pi, defeats the purpose); ExecuTorch (PyTorch's newer edge runtime, promising, less mature); OpenVINO (Intel-focused, not for ARM).

---

## 8.9 Flask vs FastAPI

Both are Python web frameworks. You need one HTTP server on the Pi that: accepts POSTs from the ESP32, runs inference, serves the dashboard.

| | Flask | FastAPI |
|---|---|---|
| Style | Synchronous WSGI | Async ASGI |
| Request validation | Manual | Automatic via Pydantic type hints |
| Auto API docs | No | Yes — `/docs` Swagger UI, free |
| WebSockets | Needs Flask-SocketIO | Built in |
| Templating | Jinja2 built in | Jinja2 available |
| Learning curve | Lower | Slightly higher |
| Throughput | Fine at your scale | Higher |

**Recommendation: FastAPI**, for three project-specific reasons:

1. The auto-generated `/docs` page lets you test the ESP32 endpoint from a browser before the ESP32 firmware exists — a real development accelerator.
2. Built-in WebSockets give you a live-updating dashboard without polling.
3. Pydantic validation rejects malformed ESP32 packets automatically, which is a trust-boundary concern (never trust bytes off the network).

If you already have Flask working, do not rewrite it. At 1 request/second, both are 1000× more than adequate. Ponytail rule: the one that already works wins.

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

class Packet(BaseModel):
    seq: int
    ts_ms: int
    features: list[float] = Field(min_length=1, max_length=64)   # validation is free

app = FastAPI()

@app.post("/ingest")
def ingest(p: Packet):
    buf.append(p.features)
    if len(buf) < L:
        return {"status": "warming_up", "have": len(buf), "need": L}
    rul = predict(buf)
    db.log(p.seq, p.ts_ms, rul)
    return {"rul": rul}
```

**Production note:** `flask run` / `uvicorn --reload` are development servers. On the Pi, run `uvicorn app:app --host 0.0.0.0 --port 8000` under a **systemd** unit so it restarts on crash and starts at boot. That's the difference between a demo and a deployment, and it is worth mentioning.

---

## 8.10 SQLite

**What.** A complete SQL database in a single file, with no server process. Embedded in the Python standard library (`import sqlite3` — no installation).

**Where.** The historian: every prediction, timestamped, with the model hash.

```sql
CREATE TABLE IF NOT EXISTS predictions (
  id        INTEGER PRIMARY KEY,
  ts        REAL NOT NULL,        -- unix seconds
  seq       INTEGER,              -- ESP32 sequence number
  rul       REAL NOT NULL,
  rul_lo    REAL,                 -- 10th percentile, if you add quantiles
  rul_hi    REAL,                 -- 90th percentile
  health    REAL,
  latency_ms REAL,
  model_sha TEXT
);
CREATE INDEX IF NOT EXISTS idx_ts ON predictions(ts);
```

**Why SQLite and not PostgreSQL/InfluxDB.** No server, no config, no network port, one file you can `scp` off the Pi for analysis. At 1 row/second that is 86,400 rows/day — SQLite handles millions without effort.

**Two Pi-specific cautions:**

1. **SD-card wear.** Every commit is a write. At 1 Hz with default settings you will do ~86k transactions/day. Use WAL mode and batch commits:
   ```python
   con.execute("PRAGMA journal_mode=WAL")
   con.execute("PRAGMA synchronous=NORMAL")
   # commit every 10 rows, not every row
   ```
2. **Retention.** Add a nightly job that downsamples anything older than 24 hours to 1-minute averages, or the file grows forever and the card fills.

**Alternatives.** InfluxDB/TimescaleDB (proper time-series DBs — right answer at plant scale, overkill here), plain CSV/JSONL append (simplest of all, and honestly fine — but you lose querying), Redis (in-memory, loses data on power cut).

---

## 8.11 The dashboard

Not in your original list explicitly, but it is where marks are won.

**Simplest thing that looks good:** one HTML page served by FastAPI, using **Chart.js** (a single ~200 KB JS file, served locally from the Pi — do *not* rely on a CDN, since your offline demo would break). WebSocket pushes new predictions; the chart appends.

What to display:

1. Big number: current RUL with units, and (if you add quantiles) the interval.
2. Health-index bar, 0–100 %.
3. Time-series chart of RUL over the last 10 minutes.
4. Sensor traces (raw features).
5. **Attention weight bar chart** — which sensor the model is currently using. This is your differentiator.
6. Status strip: last packet age, inference p95 latency, model hash, uptime.
7. Alert banner, red, when the alarm condition holds.

**Alternatives.** Streamlit (fastest to write, heavier at runtime, awkward for real-time push), Grafana + InfluxDB (industry-standard look, more setup), Node-RED (popular in IIoT, drag-and-drop, genuinely worth mentioning in the industry section).

---

## 8.12 OpenCV

Listed in your original section with "(if used)". **It is not used.** You have no camera and no images. Do not include a dependency you do not need — an examiner asking "where do you use OpenCV?" when the answer is "nowhere" is an avoidable own goal.

*If* you wanted a vision element (an entirely optional extension): a USB camera watching for oil leaks or belt slippage as a second modality, fused with vibration. That is a different project. Mention it only as future scope, if at all.

---

## 8.13 Arduino IDE / ESP-IDF

**What.** Arduino IDE is a C++ toolchain plus board-support packages and a simple editor. ESP-IDF is Espressif's native FreeRTOS-based SDK.

**Where.** The ESP32 firmware: I2C reading, feature computation, Wi-Fi, HTTP POST.

**Libraries you will use:**

| Library | Purpose |
|---|---|
| `Wire.h` | I2C master |
| `WiFi.h` | Station-mode connection |
| `HTTPClient.h` | POST to the Pi |
| `ArduinoJson` | Serialise the feature packet |
| `MPU6050` (Electronic Cats / jrowberg) | Register-level sensor driver |

**Arduino IDE vs PlatformIO vs ESP-IDF.**

- Arduino IDE: easiest, fine for this project.
- **PlatformIO** (VS Code extension): proper dependency management, `platformio.ini` under version control, much better editor. Recommended if you are comfortable — it makes your firmware reproducible, which matters for a report.
- ESP-IDF: full FreeRTOS control, needed if you want precise sampling timing via hardware timers and DMA. Consider it if your MPU-6050 sampling jitter becomes a problem (it might — see `08-hardware.md`).

**Alternatives.** MicroPython on ESP32 (slower, easier), Zephyr RTOS (heavyweight).

---

## 8.14 What each thing is *for*, one line each

| Software | One-line justification for the viva |
|---|---|
| Python | Same language on laptop and Pi → no train/serve skew |
| NumPy | Vectorised array maths; the format every other library speaks |
| Pandas | Parsing C-MAPSS text files and computing RUL with `groupby` |
| SciPy | Kurtosis, skewness, PSD, envelope analysis for bearing features |
| scikit-learn | Group-wise splitting, KMeans for operating conditions, metrics, RF baseline |
| PyTorch | Dynamic graphs handle variable sensor counts naturally; autograd |
| ONNX | Framework-independent model format — the "any hardware" contract |
| ONNX Runtime | Executes it on ARM with NEON, without PyTorch's 800 MB |
| FastAPI | Typed HTTP ingest + auto docs + WebSocket dashboard |
| SQLite | Zero-config embedded historian, one file |
| Chart.js | Local, dependency-free live charts |
| Arduino/PlatformIO | ESP32 firmware toolchain |

---

## Summary

- Two environments with two dependency sets: heavy on the laptop, light on the Pi.
- PyTorch trains, ONNX carries, ONNX Runtime executes.
- FastAPI + SQLite + Chart.js is a complete, offline-capable edge application stack.
- SciPy belongs in your stack (kurtosis); OpenCV does not.

## Key takeaways

1. ONNX ≠ ONNX Runtime. Format vs engine.
2. Never install PyTorch on the Pi. If you feel the need to, your export step is broken.
3. Serve JS/CSS locally so the offline demo actually works offline.
4. Run the server under systemd, not `--reload`.

## Common interview questions

1. Difference between ONNX and ONNX Runtime.
2. Why is NumPy fast if Python is slow?
3. WSGI vs ASGI; when do you need async?
4. When would you choose SQLite over a client-server database?
5. Static vs dynamic computation graphs.
6. What is a lockfile and why does it matter for reproducibility?

## Common viva questions

1. "Why PyTorch and not TensorFlow?" → Dynamic shapes for variable sensor counts, easier debugging, first-class ONNX export. Acknowledge TFLite is a strong alternative for edge.
2. "Do you need PyTorch on the Pi?" → No. That is the entire reason for ONNX. ~50 MB vs ~800 MB.
3. "Where do you use OpenCV?" → Nowhere. Remove it from your slide before someone asks.
4. "How do you guarantee the laptop and Pi preprocess identically?" → Same Python module imported by both, plus the serialised normalisation statistics.
5. "What happens if the Pi reboots?" → systemd restarts the service; SQLite retains history; the window buffer refills in 30 s and the dashboard shows "warming up".

## Practical implementation tips

- Keep preprocessing in `common/preprocess.py`, imported by both the trainer and the Pi service. One code path, one behaviour.
- Pin versions. `onnxruntime` and `onnx` opset compatibility is the most common "works on my laptop" failure.
- On the Pi: `pip install onnxruntime` (the aarch64 wheel exists for 64-bit Raspberry Pi OS). If pip tries to build from source, you are on a 32-bit OS or an unsupported Python — fix that rather than fighting the build.
- Write `requirements-pi.txt` and test it in a fresh venv on the Pi early, not the night before the demo.

## Suggested improvements for this project

1. **Add a `RandomForestRegressor` baseline** on hand-crafted features. One afternoon; enormously improves your results section's credibility.
2. **Serve the dashboard's JS locally** and verify by disabling the Pi's internet.
3. **Add a `/health` endpoint** and a `/metrics` endpoint (last-packet age, p50/p95 latency, model SHA, uptime).
4. **Switch to PlatformIO** for the firmware so `platformio.ini` pins library versions in git. Reproducible firmware is rare in student projects and easy to point at.
