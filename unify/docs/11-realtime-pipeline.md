# Section 13 — The Real-Time Pipeline

## 13.1 The honesty layer — read this first

Before the walkthrough, settle the question your examiner will ask: **what does the RUL number on your dashboard actually mean when it is driven by your motor rig?**

Your model learned degradation-to-failure from turbofans and bearings. Your motor has never failed and has no label. So a raw physical RUL in minutes, displayed for the motor, is **not** a validated prediction. Presenting it as one is the mistake that turns a good project into a criticised one.

### Three defensible framings — pick one and be consistent

**Framing A — "Pipeline demonstrator" (safest, recommended for Review 2).**
The rig demonstrates the live path: acquisition → features → wireless → edge inference → dashboard → alert, with measured latency. The dashboard shows the model's **health index** (`1 − RUL_norm`, a 0–1 bar) and the raw features, and labels the RUL panel clearly as *"model output on live features — unvalidated domain"*. Accuracy claims come only from the dataset results table.

**Framing B — "Relative health index" (better demo, still honest).**
Same as A, but you calibrate: record 10 minutes of healthy running, take the mean health index as the 100 % reference, and display *change* from that baseline. Then induce a fault (imbalance mass on the shaft, loosened mount, increased load) and show the index drop in real time. You are now demonstrating **detection**, which is a real, defensible claim, and it looks dramatic on stage.

**Framing C — "Transfer with fine-tuning" (strongest, most work).**
Run 3–5 short accelerated-degradation experiments on cheap bearings (progressively loosen, add contaminant, or overload until measurable change), record them to failure-ish, and fine-tune only the head and the sensor embeddings on that small set while freezing the encoder. Now you have a genuinely fine-tuned model for your own domain, with real labels you created. This is a proper transfer-learning result and it is the difference between a semester project and something you could write up.

**Recommendation:** implement A now, add B for the demo (it is a couple of hours of work: one baseline recording plus a mass taped to the shaft), and state C as future scope. Say all of this out loud in your presentation — leading with the limitation earns far more credit than being caught by it.

---

## 13.2 Step-by-step walkthrough

### Step 1 — The motor rotates

12 V DC motor, driven by the L298N with a PWM duty from the ESP32. The mechanical system produces vibration from: shaft rotation (1× frequency), residual imbalance, bearing element passage, commutator switching, and mounting resonances.

```
motor 3000 rpm → f_r = 50 Hz  → the 1× peak
imbalance      → energy at 1×, grows with the eccentric mass
misalignment   → energy at 2×
looseness      → many harmonics, raised noise floor
```

Vary the PWM duty to change speed; this is your controllable operating condition and a good analogue to the datasets' multiple operating conditions.

### Step 2 — The sensor measures

MPU-6050 mounted radially on the bearing housing (epoxy or bracket — not foam tape), configured for `DLPF_CFG=0` (260 Hz bandwidth), `SMPLRT_DIV=0` (1 kHz), `±4g` or `±8g`.

Each sample: 6 int16 registers. Convert to physical units:

```cpp
float ax_g = ax_raw / 4096.0f;    // for ±8 g range (4096 LSB/g)
```

**Remove the DC component.** Gravity puts a constant ~1 g on whichever axis is vertical, which dominates RMS and swamps the vibration signal. Subtract a running mean (or high-pass at ~2 Hz) before computing features. Forgetting this is an extremely common bug and makes all your RMS values look constant.

### Step 3 — The ESP32 acquires and extracts features

```cpp
// Core 1: hardware-timer ISR at 1 kHz
void IRAM_ATTR onTimer() {
  BaseType_t hp = pdFALSE;
  vTaskNotifyGiveFromISR(samplerTaskHandle, &hp);   // wake the sampler, don't do I2C in the ISR
  portYIELD_FROM_ISR(hp);
}

// Sampler task, core 1
for (;;) {
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
  readMPU(&ax, &ay, &az);               // I2C burst, ~350 µs @ 400 kHz
  buf[idx++] = {ax, ay, az};
  if (idx == N) {                       // N = 1000 → one window per second
    Features f = computeFeatures(buf, N);
    xQueueSend(txQueue, &f, 0);         // hand to the network task, never block here
    idx = 0;
  }
}
```

Features per axis, computed in one pass where possible:

```cpp
struct Features { float rms, std, kurt, skew, crest, p2p, mean_abs; };
```

Kurtosis in a numerically-stable single pass (Welford-style moments) rather than the naive `E[x⁴]/σ⁴` formula, which loses precision in float32. Keep the C++ implementation and the Python implementation in agreement — write a test that feeds the same 1000 samples to both and asserts they match to 1e-4. Train/serve skew in the *feature* layer is just as fatal as in the normalisation layer, and much harder to spot.

**Diagnostics to emit alongside the features:** clipping count, actual sample interval jitter, Wi-Fi RSSI, free heap. These cost nothing and are the first things you will want when something misbehaves.

### Step 4 — Preprocessing split across the boundary

| Done on the ESP32 | Done on the Pi |
|---|---|
| DC removal | z-score with the saved training statistics |
| Feature computation (RMS, kurtosis, …) | Window assembly (deque of the last L feature vectors) |
| Unit conversion to g | Mapping features → sensor IDs |
| — | Optional EWMA smoothing |

**Why this split.** Feature extraction must happen where the raw high-rate data is (bandwidth). Normalisation must happen where the training statistics are (correctness). Do not move either.

### Step 5 — Wi-Fi transport

```cpp
// Core 0: network task
Features f;
xQueueReceive(txQueue, &f, portMAX_DELAY);
StaticJsonDocument<512> doc;
doc["node"] = "motor1"; doc["seq"] = seq++; doc["ts_ms"] = millis();
JsonArray a = doc.createNestedArray("feat");
a.add(f.rms); a.add(f.kurt); /* … */
String body; serializeJson(doc, body);

http.begin(client, PI_URL);
http.addHeader("Content-Type", "application/json");
http.setReuse(true);
int code = http.POST(body);
if (code != 200) buffer_for_retry(f);      // store-and-forward
```

~300 bytes, one packet, ~2–10 ms over Wi-Fi.

### Step 6 — The Pi receives, validates, windows

```python
@app.post("/ingest")
def ingest(p: Packet):
    if p.seq - state.last_seq > 1:
        state.gaps += p.seq - state.last_seq - 1
        log.warning("gap: lost %d packets", p.seq - state.last_seq - 1)
    state.last_seq  = p.seq
    state.last_time = time.time()

    buf.append(p.feat)                       # deque(maxlen=L)
    if len(buf) < L:
        return {"status": "warming_up", "have": len(buf), "need": L}
    ...
```

### Step 7 — ONNX inference

```python
    window = np.array(buf, dtype=np.float32)              # (L, S)
    x = ((window - mu) / sd).T[None]                      # (1, S, L)
    t0 = time.perf_counter()
    rul_norm, attn = sess.run(["rul", "attn"], {
        "x": x, "mask": MASK, "sensor_ids": IDS, "domain_id": DOM})
    lat_ms = (time.perf_counter() - t0) * 1e3
```

~1–4 ms.

### Step 8 — Post-processing: smoothing, denormalisation, alarm

```python
    state.ewma = 0.3 * float(rul_norm[0, 0]) + 0.7 * state.ewma   # smooth
    health = state.ewma                                            # 0..1
    rul_physical = state.ewma * LIFE_SCALE                         # native units

    if health < ALARM_THRESHOLD:
        state.breaches += 1
    else:
        state.breaches = 0
    alarm = state.breaches >= N_CONSECUTIVE                        # hysteresis
```

**Why smoothing and hysteresis matter.** A raw per-second prediction jitters by a few percent. Without smoothing, an alarm at a fixed threshold flips on and off many times per minute — the false-alarm problem from your **P4**. EWMA plus "N consecutive breaches" removes essentially all of it for six lines of code. If you also implemented quantile regression (`06-model.md` §6.12), alarm on the **10th-percentile** curve instead, which is the principled version.

### Step 9 — Persist

```python
    db.execute("INSERT INTO predictions(ts,seq,rul,health,latency_ms,model_sha) "
               "VALUES (?,?,?,?,?,?)",
               (time.time(), p.seq, rul_physical, health, lat_ms, MODEL_SHA))
```

Batch commits every 10 rows; WAL mode. See `07-software-stack.md` §8.10.

### Step 10 — Dashboard

WebSocket push to the browser; Chart.js appends. Panels:

```
┌─────────────────────────────────────────────────────────────┐
│  UNIFY-RUL  ·  motor1                        ● LIVE  38 ms   │
├──────────────────────┬──────────────────────────────────────┤
│   HEALTH INDEX       │   RUL (model output, live domain)     │
│   ████████████░░ 78% │   47 min   [10–90%: 31 – 68]          │
├──────────────────────┴──────────────────────────────────────┤
│   RUL over time    ╱╲___                                     │
│                          ╲___                                │
├──────────────────────────────────────────────────────────────┤
│   Features:  ax_rms ▁▂▂▃▃▄▅▆   ax_kurt ▂▂▂▃▇▅▃              │
├──────────────────────────────────────────────────────────────┤
│   Attention:  ax_rms ███████ 61%   ax_kurt ███ 24%  ...      │
├──────────────────────────────────────────────────────────────┤
│   ⚠ MAINTENANCE ADVISED — health below 30% for 12 s          │
└──────────────────────────────────────────────────────────────┘
Status: last packet 0.4 s ago · p95 latency 3.1 ms · model a3f9c2… · uptime 4 h
```

The attention panel is your unique selling point — no other student project will show which sensor the model is currently using.

### Step 11 — The alert

Minimum: a red banner. Better, and cheap:

- A red LED on the ESP32, driven by the POST response (the machine itself signals, independent of the dashboard).
- A row in an `alerts` table with a timestamp and the triggering value.
- Optional: a webhook, an email, or a Telegram message. One `requests.post` call. Impressive in a demo, ~10 lines.

---

## 13.3 End-to-end timing budget

```
sensor sampling window (the dominant term)      1000 ms   ← by design: 1 s of data
feature computation on ESP32                       2 ms
JSON serialisation                                 1 ms
Wi-Fi transmission + TCP                        2–15 ms
FastAPI parse + validate                         1–3 ms
normalisation + tensor build                       1 ms
ONNX Runtime inference                           1–4 ms
smoothing + alarm logic                          <1 ms
SQLite insert (batched)                          <1 ms
WebSocket push + browser render                 5–20 ms
────────────────────────────────────────────────────────
end-to-end AFTER the window closes              13–47 ms
total sensor-to-screen                        ~1013–1047 ms
```

**The insight to state:** inference is 1–4 ms out of ~1000 ms. **The model is not the bottleneck — the physics of collecting a 1-second window is.** That is exactly the right result for an edge system and it means you have headroom for 100× more model, or for 50 sensor nodes on one Pi. Say this; it reframes "our model is small" from an apology into a design win.

If you need faster response, shorten the feature window (0.25 s) or overlap windows (compute features every 0.25 s over the last 1 s). Trade-off: shorter windows give noisier features.

---

## 13.4 Demo script — rehearse this

A demo that fails is worth less than no demo. Rehearse the exact sequence:

1. **Boot** the Pi, show `systemctl status unify` → active. (Proves it is a service, not a script.)
2. **Show the dashboard** in "warming up 8/30" state — proves the window logic is real.
3. **Window fills**, predictions start. Point at the p95 latency figure.
4. **Show the attention bars** — "the model is currently weighting horizontal RMS most heavily."
5. **Induce a fault**: attach the imbalance mass while running (safely — guard the shaft) or step the PWM load up. Watch RMS rise and the health index fall on screen.
6. **Alarm fires** after N consecutive breaches; red banner, red LED on the ESP32.
7. **Pull the network cable/uplink** from the Pi → dashboard still live; edge inference is genuinely local.
8. **Unplug the ESP32** → "SENSOR OFFLINE" within 3 s. Plug back in → buffered packets flush and it recovers.
9. **Run `bench.py`** live → the latency/memory/temperature table on screen.
10. **Run the S ∈ {2,14,16,21} script** live → proves the sensor-count claim.
11. **Show the dataset results table** — the validated accuracy numbers, clearly separated from the live rig.

Steps 7, 8, 10 and 11 are the ones examiners remember.

### Have a fallback

Record a JSONL of a successful run and add `--source replay`. If the motor, the Wi-Fi or the ESP32 fails at the worst moment, you switch to replay in five seconds and continue. Say openly that you are in replay mode if you use it — a prepared fallback is professional; a pretended live demo is not.

---

## 13.5 Things that will go wrong (and their fixes)

| Symptom | Likely cause | Fix |
|---|---|---|
| Health index dead flat | DC gravity not removed, or features saturated | High-pass / subtract running mean |
| RUL pinned at 100 % | Sigmoid saturation from labels capped at exactly 1.0 | Cap at 0.98, or remove the sigmoid |
| Predictions wildly noisy | No smoothing; window too short | EWMA α=0.3 |
| Everything freezes when the motor starts | I2C corrupted by motor EMI | Capacitors, shorter I2C leads, Wire timeout + bus recovery |
| ESP32 reboots randomly | Brownout from Wi-Fi TX current spikes | Better 5 V supply, bulk capacitor on the board's 5 V |
| Dashboard lags then catches up | Blocking work in the request handler | Move SQLite writes and WebSocket pushes off the hot path |
| ESP32 never connects | 5 GHz-only hotspot, or captive portal | 2.4 GHz; ideally run the Pi as an access point |
| Prediction differs from laptop | Wrong `norm_stats.json`, or train-mode export | Hash-check the artefacts; run the equivalence test |

---

## Summary

- The live loop: motor → MPU-6050 → ESP32 (1 kHz sampling, 1 Hz features) → Wi-Fi JSON → Pi (validate, window, normalise, ONNX) → smooth → alarm → SQLite → dashboard.
- Feature extraction lives with the raw data; normalisation lives with the training statistics.
- Total sensor-to-screen ≈ 1 s, of which inference is 1–4 ms. The window, not the model, is the bottleneck.
- Smoothing + hysteresis (or quantile-based alarming) is what makes the alert usable.
- Be explicit that the rig demonstrates the pipeline and fault *detection*; validated RUL accuracy comes from the datasets.

## Key takeaways

1. Choose a framing (A/B/C) for what the live RUL means, and state it before anyone asks.
2. Remove the DC gravity component or all your vibration features are constants.
3. Never block the sampling task on network I/O — queue between two FreeRTOS tasks.
4. The inference-is-2ms-of-1000ms observation is a headline, not a footnote.
5. Rehearse the demo, and build a replay fallback.

## Common interview questions

1. Where in a distributed pipeline should preprocessing run, and why?
2. What is an EWMA and why use it on predictions?
3. What is hysteresis in alarming and what problem does it solve?
4. How do you detect that a sensor node has died?
5. Why compute features on the endpoint rather than sending raw data?
6. What is the end-to-end latency of your system, and what dominates it?

## Common viva questions

1. "Is that RUL number real?" → The honesty-layer answer, delivered confidently and first.
2. "How do you know the model is responding to the fault and not to noise?" → The baseline recording, the controlled imbalance, and the repeatability of the drop across trials. Do at least three trials so you can say "repeatable".
3. "What is your alarm threshold and how did you choose it?" → Have a real answer: chosen from the healthy-baseline distribution (e.g. mean − 3σ), or from the quantile model. "We picked 30 %" is a weak answer; "3σ below the healthy baseline over 10 minutes of data" is a strong one.
4. "How many sensor nodes could one Pi handle?" → Compute it: at 4 ms per inference and 1 Hz, the CPU could serve ~250 nodes; the practical limit is network and application overhead. Nice, quantitative answer.
5. "What happens if the model is wrong and the machine fails anyway?" → Discuss the asymmetric loss, the conservative alarm on the lower quantile, and the fact that this is an advisory system with a human in the loop — not an autonomous shutdown. Correct, mature answer.

## Practical implementation tips

- Build the whole loop with a **stub model returning 42.0** in week 1. Swap in the real model later. One unknown at a time.
- Log every ingest packet to JSONL from day one. It is your replay source and your post-mortem evidence.
- Put the model SHA and the config in the dashboard footer. When you have three model versions the week before the demo, you will need it.
- Test the demo on the actual demo network, in the actual room, before demo day.

## Suggested improvements for this project

1. **Do the controlled-fault experiment** (baseline / imbalance / heavy imbalance, three trials each). This converts your live demo from "numbers moving" to "a measured, repeatable detection result" — the highest-value few hours in the whole project.
2. **Alarm on the 10th percentile** from quantile regression rather than on the point estimate. Directly and principledly solves P4.
3. **Add a second sensor node** (a second ESP32 + MPU-6050 on a second motor, or just a second sensor position on the same motor). Proves multi-node scaling and makes MQTT's advantage concrete.
4. **Add motor current sensing** (INA219/ACS712). A second physical modality, and it genuinely improves detectability of load-related faults.
5. **Record and publish a 1-minute video** of the demo. Costs nothing, and it survives hardware failure on review day.
