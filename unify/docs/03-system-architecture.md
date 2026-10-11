# Section 3 — Complete System Architecture

## 3.1 The two loops

The single most common mistake in presenting this project is drawing one long arrow from "Datasets" to "Dashboard". That is wrong, because your system has **two separate loops that run at different times, on different machines, at different rates.**

```
╔══════════════════════ LOOP A — OFFLINE / TRAINING (laptop, once) ═════════════════════╗
║                                                                                        ║
║  C-MAPSS ─┐                                                                            ║
║  FEMTO   ─┼─▶ Parsers ─▶ Preprocess ─▶ Sliding ─▶ Joint    ─▶ MODEL ─▶ Loss ─▶ Adam   ║
║  XJTU-SY ─┘   (to a      (z-score,     window     DataLoader   (CNN +   (MSE +  (back- ║
║               common      RUL cap)     (L, stride)             attn)    score)  prop)  ║
║               schema)                                             │                    ║
║                                                                   ▼                    ║
║                                              best.pt ──▶ torch.onnx.export             ║
║                                                              (dynamic_axes)            ║
║                                                                   │                    ║
╚═══════════════════════════════════════════════════════════════════│════════════════════╝
                                                                    │  scp / USB
                                                                    ▼
╔══════════════════════ LOOP B — ONLINE / INFERENCE (Pi, forever @ 1 Hz) ═══════════════╗
║                                                                                        ║
║  DC motor ─▶ MPU-6050 ─I2C─▶ ESP32 ─▶ feature ─Wi-Fi/HTTP─▶ Pi ─▶ same z-score ─▶ ORT ║
║  (12 V,       (accel+         (240      extract    (JSON)     (Flask)   (saved stats)  ║
║   L298N)       gyro)          MHz)     (RMS,                                     │     ║
║                                        kurtosis…)                                ▼     ║
║                                                                              RUL scalar║
║                                                                                  │     ║
║                        SQLite ◀── log ──┬──────────────────────────────────────┘      ║
║                                          │                                             ║
║                        Dashboard ◀───────┤                                             ║
║                        Alert     ◀───────┘  (if RUL < threshold for N consecutive)     ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
```

The only thing crossing from Loop A to Loop B is **two files**: `unify_rul.onnx` and `norm_stats.json`. If you forget the second one, your predictions will be silently, catastrophically wrong (see §3.4 — this is the #1 deployment bug in the entire field).

---

## 3.2 Every block, explained

### Block 1 — Datasets

**What.** Three public run-to-failure archives (details in `04-datasets.md`).

**Why here.** They are the only source of ground-truth RUL labels you have. Your motor rig cannot provide them.

**Output artefact.** A common in-memory schema. Insist on one from day one:

```python
# Every dataset parser must return this, no exceptions
{
  "unit_id":   str,           # "cmapss_FD001_u17", "xjtu_c1_b3"
  "dataset":   str,           # "cmapss" | "femto" | "xjtu"
  "signals":   np.ndarray,    # (T, S) float32 — T timesteps, S sensors
  "sensor_ids": list[int],    # length S — global sensor vocabulary indices
  "rul":       np.ndarray,    # (T,) float32 — remaining life at each timestep
  "life_total": float,        # total life of this unit, in native units
  "unit_of_time": str,        # "cycle" | "second" | "minute"
}
```

Everything downstream consumes only this. Adding a fourth dataset later becomes a 60-line parser, not a refactor. This is the entire "Any Dataset" claim in your title, made concrete.

### Block 2 — Preprocessing

**What.** Convert raw, differently-scaled, differently-sampled records into tensors a network can digest.

**Steps.** Sensor selection → resampling/feature extraction (bearings only) → per-dataset z-score → RUL capping → RUL normalisation.

**Why here.** A network cannot learn when one input is 1400 (turbine temperature in °R) and another is 0.03 (a bleed ratio). Gradients from the large-scale feature swamp everything else. Covered fully in `05-preprocessing.md`.

**Critical:** the normalisation statistics computed here must be **saved to disk** and reused verbatim on the Pi.

### Block 3 — Sliding Window

**What.** Cut each long time series into overlapping fixed-length chunks.

```
Engine #17, 192 cycles long, window L=30, stride=1:

cycles   1..30  ──▶ window 0,  label = RUL at cycle 30  = 162 (capped 125)
cycles   2..31  ──▶ window 1,  label = RUL at cycle 31  = 161 (capped 125)
...
cycles 163..192 ──▶ window 162, label = RUL at cycle 192 = 0
                                          ↑
                          label is ALWAYS the RUL at the LAST timestep
```

**Why here.** Three reasons: (a) neural networks want fixed-size inputs; (b) it turns 100 engines into ~18,000 training samples — essential given how tiny prognostics datasets are; (c) it matches inference reality, where the Pi will only ever have the last L seconds in its buffer.

**Shape produced.** `(N, S, L)` — N windows, S sensors, L timesteps. Labels `(N,)`.

### Block 4 — Shared per-sensor CNN encoder

**What.** A small 1-D convolutional network that reads **one sensor's** window and emits a fixed-length descriptor vector.

**The key word is "shared".** The *same weights* process sensor 1, sensor 2, … sensor S. Loop over sensors, or better, fold sensors into the batch dimension:

```
in:   (B, S, L)  ──reshape──▶  (B·S, 1, L)  ──CNN──▶  (B·S, D)  ──reshape──▶  (B, S, D)
```

**Why here.** This one trick is what makes the model sensor-count-agnostic. A conventional model does `nn.Linear(21*30, …)` and is permanently welded to 21 sensors. A shared encoder doesn't care whether S is 2 (XJTU) or 21 (C-MAPSS) — S just changes the batch size of an internal reshape. This is your answer to problem **P1**.

### Block 5 — Sensor embeddings

**What.** A learned vector `e_s ∈ R^D` per sensor identity, added to that sensor's CNN descriptor.

**Why here.** The shared encoder deliberately erased sensor identity (that's what "shared" means). But identity matters: a temperature trend and a vibration RMS trend with identical shapes mean different things. The embedding gives the model back a "name tag" per channel. Also, attention over a set is permutation-invariant — without embeddings the model literally cannot tell which sensor is which.

### Block 6 — Cross-attention aggregator

**What.** A mechanism that combines a variable-length set of sensor tokens into one fixed-length context vector, weighting each sensor by how informative it is.

**Why here.** Consider the alternatives:

| Aggregation | Handles variable S? | Learns which sensor matters? | Verdict |
|---|---|---|---|
| Concatenate `(B, S·D)` | ❌ shape is welded to S | yes | Defeats the whole project |
| Mean pool over S | ✅ | ❌ every sensor equally important | Weak baseline; noise dilutes signal |
| Max pool over S | ✅ | partially | Loses magnitude info |
| **Cross-attention** | ✅ (with a mask) | ✅ learned, input-dependent | ✔ what you use |

Cross-attention with a small number of *learned query vectors* attending over the sensor tokens gives a fixed-size output regardless of S, while learning a data-dependent weighting. Full mechanics and shapes in `06-model.md` §6.6.

### Block 7 — Regression head

**What.** A small MLP: context vector → single scalar.

**Why here.** Attention produced a rich representation; something must turn it into a number. Kept small (D → 64 → 1) deliberately — this is where overfitting on tiny prognostics datasets creeps in.

### Block 8 — Loss function

**What.** The number that says how wrong the prediction is, and whose gradient teaches the network.

**Why here.** MSE alone is symmetric and does not know that late predictions kill people. You combine MSE with the NASA asymmetric score (or use it as a reported metric). Details in `06-model.md` §6.8.

### Block 9 — Training

**What.** Repeatedly: sample a batch, forward, compute loss, backpropagate, update weights via Adam/AdamW, evaluate, keep the best checkpoint.

**Why here.** Obvious, but note one project-specific subtlety: your batches must be **mixed across datasets** (your slide already says "Batches from all 3 per step"). If you train dataset-by-dataset, the network catastrophically forgets. Mixed batches are what force one shared representation — the heart of the "Universal" claim.

### Block 10 — ONNX export

**What.** Trace the PyTorch model and serialise it as a framework-independent computation graph.

**Why here.** The Pi should not need PyTorch (~800 MB, slow to import, painful on ARM). ONNX + ONNX Runtime is ~50 MB and faster. `dynamic_axes` on the batch **and sensor** dimensions is what preserves the variable-S property through export. Details in `10-onnx-and-deployment.md`.

### Block 11 — Raspberry Pi

**What.** The edge node: runs the ONNX Runtime session, the HTTP endpoint the ESP32 posts to, the SQLite log, and the Flask dashboard.

**Why here.** It is the "Any Hardware" half of your title, and the demonstration that a model trained on an x86 laptop with an NVIDIA GPU runs unchanged on ARM.

### Block 12 — ESP32

**What.** The sensing endpoint. Reads the MPU-6050 over I2C, buffers a window, computes time-domain features, sends them over Wi-Fi.

**Why here.** The Pi *could* read I2C directly, but then the sensor must be physically at the Pi. Splitting acquisition from inference mirrors real plants (sensor nodes near machines, one gateway per line) and lets you show a wireless architecture. It also demonstrates edge preprocessing: sending 8 features instead of 1000 raw samples is a 100× bandwidth reduction *before* the data even reaches the edge node.

### Block 13 — Sensor + motor

**What.** MPU-6050 IMU on a 12 V DC motor driven through an L298N.

**Why here.** Provides live, real, physically-caused data so the demo is not a CSV replay. Its limitations (bandwidth, no failure label) are discussed honestly in `08-hardware.md` and `11-realtime-pipeline.md`.

### Block 14 — Dashboard + alerting

**What.** Flask/FastAPI serving a page with live RUL, health index, sensor traces and an alert banner; SQLite as the historian.

**Why here.** Layer 6–7 of OSA-CBM. A prediction nobody sees is not maintenance. Also, it is what the examiner will actually look at.

---

## 3.3 End-to-end data flow with concrete shapes

### Training path (one batch)

```
raw CSV / .mat files
        │  parsers
        ▼
per-unit arrays              signals (T, S) float32,  rul (T,)
        │  z-score with train-split stats
        ▼
normalised arrays            same shapes, mean≈0 std≈1 per sensor
        │  sliding window L=30, stride=1
        ▼
X (N, S, L)   y (N,)         e.g. C-MAPSS FD001: (17731, 14, 30), y in [0,125]
        │  DataLoader, batch=256, shuffled ACROSS datasets
        ▼
x (256, S_max, 30)  mask (256, S_max) bool   y (256,)
        │  reshape for shared encoder
        ▼
(256·S_max, 1, 30)
        │  1-D CNN: Conv(1→32,k=5) → Conv(32→64,k=3) → Conv(64→64,k=3) → GAP
        ▼
(256·S_max, 64)
        │  reshape back, add sensor embedding
        ▼
tokens (256, S_max, 64)
        │  cross-attention: 1 learned query, mask padded sensors
        ▼
context (256, 64)
        │  MLP 64 → 64 → 1
        ▼
rul_norm_hat (256, 1)  in [0, 1]
        │  loss vs y_norm
        ▼
scalar loss → backward() → AdamW.step()
```

### Inference path (one prediction on the Pi)

```
MPU-6050 registers                6 × int16, at 500–1000 Hz
        │ I2C burst read on ESP32
        ▼
ring buffer                       e.g. 1000 samples × 6 axes
        │ feature extraction on ESP32 (RMS, std, kurtosis, crest, p2p …)
        ▼
feature vector                    (S_live,) float32, S_live = 2 (or 8 with more features)
        │ JSON over HTTP POST, Wi-Fi
        ▼
Pi: rolling feature history        (L, S_live) — Pi holds the last L=30 seconds
        │ z-score with SAVED stats
        ▼
np.ndarray (1, S_live, 30) float32
        │ ort_session.run(None, {"x": arr, "mask": mask})
        ▼
rul_norm (1, 1)
        │ × life_scale for the reference dataset
        ▼
RUL in minutes → SQLite → dashboard → alert if below threshold N times in a row
```

Note the asymmetry: training sees `(256, S, 30)`, inference sees `(1, S, 30)`. Both must be valid, which is precisely why `dynamic_axes` marks axis 0 as dynamic. And S differs between them, which is why axis 1 must be dynamic too.

---

## 3.4 The three interfaces that break projects

Write these on a sticky note.

**Interface 1 — normalisation statistics.**
The Pi must apply *exactly* the mean and std computed on the training set. Not recomputed live, not approximated. Serialise them next to the model:

```json
{"dataset": "xjtu", "sensor_ids": [40, 41],
 "mean": [0.0031, 0.0027], "std": [0.412, 0.388],
 "rul_cap": 125, "life_scale": 2100.0, "unit": "minute"}
```

If you recompute statistics from live data, you normalise away the very degradation trend you are trying to detect. The prediction will look stable and be meaningless. This bug is silent — no exception, no NaN, just wrong answers.

**Interface 2 — sensor ID vocabulary.**
The sensor embedding table is indexed by a global integer ID. `C-MAPSS T24 = 3`, `XJTU horizontal accel = 40`, and so on. That mapping must be identical at train and inference time. Store it in a single `sensor_vocab.json` that both the trainer and the Pi load. Never hard-code it in two places.

**Interface 3 — window length and stride.**
The Pi must accumulate exactly `L` timesteps at the same effective sampling interval before it can predict. If training used 1 sample/second and the ESP32 posts at 4 Hz, you must decimate or re-window on the Pi. Mismatch here produces confidently wrong output.

---

## 3.5 Failure modes to design against

| Failure | Symptom | Mitigation |
|---|---|---|
| ESP32 drops off Wi-Fi | Predictions freeze on stale data | Timestamp every packet; Pi shows "STALE" if last packet > 3 s old |
| Packet loss / out-of-order | Ragged window | Sequence-number every packet; drop or interpolate gaps |
| Window not yet full at startup | Garbage first prediction | Suppress output until `len(buffer) == L`; show "warming up" |
| Model outputs negative RUL | Nonsense on dashboard | Clamp to `[0, cap]` — or better, use a softplus/sigmoid output activation |
| Alert flapping | Alarm on/off every second | Require N consecutive breaches (hysteresis) — this is your P4 fix, cheaply |
| SD card fills with SQLite rows | Pi dies at 3 a.m. | Retention policy: keep 1 Hz for 24 h, then downsample to 1/min |

---

## Summary

- Two loops: offline training on the laptop, online inference on the Pi. They meet at exactly two files — `.onnx` and `norm_stats.json`.
- A common per-unit schema (`signals`, `rul`, `sensor_ids`, `unit_of_time`) is what makes "Any Dataset" real.
- The shared per-sensor encoder makes the model sensor-count-agnostic; cross-attention aggregates a variable-size set into a fixed vector; the head produces the scalar.
- Three interfaces silently break deployments: normalisation stats, sensor-ID vocabulary, window length/rate.

## Key takeaways

1. Draw two loops, never one arrow.
2. `(B, S, L) → (B·S, 1, L) → (B·S, D) → (B, S, D)` is the reshape that carries the entire novelty claim. Be able to write it from memory.
3. Saving and shipping the normalisation statistics is as important as saving the weights.
4. Robustness features (staleness, hysteresis, warm-up) cost ~20 lines and are what separates a demo from a system.

## Common interview questions

1. Walk me through the data flow from CSV to prediction, with shapes at every step.
2. Why is the encoder shared across sensors instead of one encoder per sensor?
3. What exactly does the Pi need besides the model file?
4. Why must batches mix datasets rather than train sequentially?
5. Where in this pipeline would you add uncertainty estimation? (At the head — output multiple quantiles instead of one mean.)

## Common viva questions

1. "Show me where in your diagram the variable sensor count is handled." → The reshape into the batch dimension, plus the attention mask.
2. "What happens if I unplug the ESP32?" → Staleness detection; predictions stop updating and the dashboard says so.
3. "Is your dashboard reading live data or a recording?" → Be honest about which parts are live; have a switchable `--replay` flag so you can demo both deliberately.
4. "Why do you preprocess on the ESP32 rather than sending raw data?" → 100× bandwidth reduction, and it keeps the Wi-Fi packet inside one MTU.
5. "What is the slowest part of your pipeline?" → Measure it. Usually it is the 1-second feature-window accumulation on the ESP32, not the inference.

## Practical implementation tips

- Build the pipeline **end-to-end with a dummy model first** (a model that returns `42.0`). Get ESP32 → Wi-Fi → Pi → dashboard working in week 1 with a stub. Then swap in the real model. Debugging one unknown at a time is the only sane way.
- Log every packet the Pi receives to a JSONL file from day one. When something goes wrong at demo time, you will have the trace.
- Put a `--source {live,replay}` flag on the Pi service. Replay mode reads a recorded JSONL at the same rate. This lets you rehearse the demo without the motor running.

## Suggested improvements for this project

1. Add a **sequence number and timestamp** to every ESP32 packet. Costs 8 bytes, saves hours of debugging.
2. Add a **`/health` endpoint** on the Pi returning uptime, last-packet age, inference p95, model hash. Trivial to write, looks extremely professional in a demo.
3. Store the **model file's SHA-256** in the SQLite log alongside predictions, so every logged prediction is traceable to the exact model that produced it. This is real MLOps practice (model lineage) and is a great talking point.
