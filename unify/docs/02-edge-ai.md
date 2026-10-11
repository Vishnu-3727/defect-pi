# Section 2 — Edge AI

## 2.1 What is Edge AI?

**Plain language.** Normally, a smart device sends your data to a big computer somewhere on the internet, which thinks about it and sends the answer back. Edge AI means the thinking happens *on the device itself* — no internet round trip.

**Technical.** Edge AI = executing the inference phase of a trained machine-learning model on compute resources physically located at or near the data source, rather than in a centralised datacentre. The model weights are shipped to the device; the data never leaves.

The word "edge" comes from network topology. Picture the network as a tree:

```
        ┌──────────────────────────┐
        │   CLOUD  (datacentre)    │   GPUs, unlimited power, 50–300 ms away
        └────────────┬─────────────┘
                     │  internet / WAN
        ┌────────────┴─────────────┐
        │   FOG / GATEWAY          │   industrial PC, plant server, 5–20 ms away
        └────────────┬─────────────┘
                     │  LAN / Wi-Fi
        ┌────────────┴─────────────┐
        │   EDGE  ← Raspberry Pi   │   your device, 0 ms — it IS the data source's neighbour
        └────────────┬─────────────┘
                     │  I2C / SPI / UART
        ┌────────────┴─────────────┐
        │   ENDPOINT / TinyML      │   ESP32 + MPU-6050 — sensing, sometimes tiny inference
        └──────────────────────────┘
```

Your project spans the bottom three tiers. The Pi is the edge node; the ESP32 is the endpoint. Being able to draw and name this diagram is worth easy marks.

### Related terms you should not confuse

- **TinyML** — ML on microcontrollers (KB of RAM, no OS). Your ESP32 tier. Frameworks: TFLite Micro, CMSIS-NN, Edge Impulse.
- **Fog computing** — Cisco's term for the gateway tier between edge and cloud.
- **On-device AI** — same idea as edge AI, usually said about phones.
- **Federated learning** — training (not just inference) distributed across edge devices, with only gradients sent to the server. Excellent "future scope" answer for your viva.

---

## 2.2 Cloud AI vs Edge AI

| Dimension | Cloud AI | Edge AI (your Pi) |
|---|---|---|
| Where compute lives | Remote datacentre | On/near the machine |
| Latency | 50–300 ms typical, unbounded tail | 1–20 ms, bounded |
| Determinism | No — network jitter, queuing | Yes — same code, same silicon, every time |
| Bandwidth cost | Every sample uploaded | Only alerts/summaries uploaded |
| Works offline | No | Yes |
| Data privacy | Data leaves the premises | Data stays on the premises |
| Compute available | Effectively unlimited (A100s) | 4 cores, 1.5 GHz, no GPU |
| Memory | 100s of GB | 8 GB total, shared with OS |
| Power | Irrelevant (grid + cooling) | 3–7 W, sometimes battery |
| Model size practical limit | 100s of GB | ~10–500 MB |
| Cost model | Per-inference, forever (OPEX) | One-time hardware (CAPEX) |
| Update/retrain | Instant, centralised | Requires pushing files to fleet |
| Best for | Training, large models, fleet analytics | Real-time control, always-on monitoring |

### The economics, made concrete for your rig

Your ESP32 samples the MPU-6050 at 1 kHz on 6 axes. Raw upload:

```
1000 samples/s × 6 axes × 2 bytes = 12 kB/s
                                  = 43 MB/hour
                                  = 1.04 GB/day  per sensor node
                                  ≈ 380 GB/year  per node
```

A plant with 200 monitored motors → **76 TB/year** of raw vibration, essentially all of it uninteresting. Cloud AI means paying to transport, store and process all of it. Edge AI means the Pi consumes it locally and uploads only "Motor 47: RUL 41 min, alert" — a few hundred bytes per event. That is a ~10⁶× bandwidth reduction. Put this calculation on a slide; it is the most persuasive single argument for the whole architecture.

---

## 2.3 Why the Raspberry Pi is an Edge AI device

The Pi 4 (8 GB) qualifies because it hits every requirement:

| Requirement | Pi 4 provides |
|---|---|
| Runs a full OS | Raspberry Pi OS 64-bit (Debian) — so Python, pip, systemd, networking all work |
| Enough RAM for a model + runtime | 8 GB; your model needs < 50 MB |
| SIMD-capable CPU | Cortex-A72, ARMv8-A with **NEON** 128-bit SIMD — this is what makes ONNX Runtime fast |
| Networking | Gigabit Ethernet + 802.11ac Wi-Fi (talks to ESP32) |
| Physical I/O | 40-pin GPIO with I2C, SPI, UART (talks to sensors directly if needed) |
| Low power | ~3 W idle, ~6–7 W under full 4-core load |
| Deployable | Fanless, DIN-rail cases exist, boots from SD/USB, no moving parts |

**Cortex-A72 details worth knowing for viva:**

- ARMv8-A, 64-bit, **out-of-order superscalar**, 3-wide decode — a real application-class core, not a microcontroller.
- 4 cores @ 1.5 GHz on the Pi 4 (BCM2711).
- Caches: 32 KB L1-I + 32 KB L1-D per core, **1 MB shared L2**. LPDDR4-3200 main memory.
- **NEON** (Advanced SIMD): 128-bit vectors → 4× fp32 lanes or 16× int8 lanes per instruction. This is why int8 quantisation gives real speedups here.
- **No NPU, no usable GPU compute** for ONNX (the VideoCore VI is not an ML accelerator you can practically target). All your inference is CPU.
- Thermal: throttles around 80–85 °C. Under sustained 4-thread inference in a closed case, expect throttling. Use a heatsink; measure with `vcgencmd measure_temp`.

**Cache is the number that should shape your model design.** 1 MB of L2 shared between four cores. A model whose weights are under ~500 KB stays hot in cache and runs dramatically faster than one that streams weights from LPDDR4 every inference. Your UNIFY-RUL model, at roughly 100–300 KB of fp32 weights, fits. This is not an accident — design it to stay that way.

---

## 2.4 Why inference on the Pi and training on the laptop

### Why inference belongs on the Pi

1. **Latency** — the decision loop (sensor → prediction → alert) must close in tens of milliseconds, deterministically. A Wi-Fi hop to a cloud server adds unbounded jitter.
2. **Availability** — a factory floor with flaky Wi-Fi still needs the machine monitored. Edge inference degrades gracefully; cloud inference stops dead.
3. **Bandwidth** — see the 380 GB/year/node calculation above.
4. **Privacy / IP** — vibration signatures reveal production rates, process recipes and machine health. Many plants contractually forbid raw telemetry leaving the site.
5. **Cost** — one ₹6,000 Pi versus an indefinite per-inference cloud bill.
6. **It is the point of the project** — "Any Hardware" in your title is a claim about deployment. You must actually deploy to prove it.

### Why training must NOT happen on the Pi

| Aspect | Training | Inference |
|---|---|---|
| Passes over the data | Hundreds of epochs | Once |
| Direction | Forward **and** backward | Forward only |
| Memory | Weights + gradients + optimiser state + all activations for backprop | Weights + one activation buffer |
| Arithmetic per sample | ~3× forward cost | 1× forward cost |
| Typical hardware | RTX GPU (thousands of cores, high memory bandwidth) | Any CPU |

Concretely: your training set is C-MAPSS + FEMTO + XJTU-SY, on the order of 10⁵–10⁶ windows. One epoch on an RTX laptop GPU is seconds to a couple of minutes; 100 epochs is comfortably under an hour. The same job on a Pi 4 CPU is **50–200× slower** — days, with thermal throttling and SD-card wear, and 8 GB of RAM shared with the OS.

**The rule to state in viva:** *train once, centrally, with the fleet's data; deploy the frozen artefact everywhere.* This is exactly what industry does — it is called the train/serve split, and it is the reason ONNX exists.

Note also: Adam's optimiser state is 2 extra float32 tensors per parameter, so training memory ≈ 4× the model, plus activations for every layer in the batch. Inference memory ≈ model + one window. That single fact explains the whole architecture.

---

## 2.5 The four classic advantages, with numbers for your report

### Latency

```
Cloud path:  ESP32 →Wi-Fi→ Pi →internet→ cloud →infer→ back
             2 ms      +   40–200 ms RTT   +  5 ms  +  40–200 ms
             ≈ 90–400 ms, high variance, fails entirely if the link drops

Edge path:   ESP32 →Wi-Fi→ Pi →ONNX Runtime→ result
             2 ms      +   1–5 ms
             ≈ 3–10 ms, low variance
```

Measure and report your actual number. `time.perf_counter()` around `session.run()`, 1000 iterations, report p50 / p95 / p99 — not just the mean. Percentiles are what real-time engineers care about, and reporting them signals maturity.

### Privacy

Raw vibration is process-revealing. Under edge inference the raw stream never crosses the plant boundary; only the derived RUL scalar does. Under Indian DPDP-style and EU GDPR-style regimes, minimising transmitted data is the cheapest compliance strategy there is.

### Bandwidth

12 kB/s raw vs ~50 bytes per prediction at 1 Hz. Ratio ≈ 240:1 even before you account for TCP/TLS overhead and the fact that cloud ingest is billed per GB.

### Offline operation

Your demo should include this: **pull the Ethernet/Wi-Fi uplink out during the presentation and show the Pi still predicting.** It takes five seconds and proves the point better than any slide.

Two more advantages usually omitted, worth adding:

- **Energy per inference.** Pi 4 at ~6 W producing ~200 inferences/s → ~30 mJ per inference, versus the joules spent by radios transmitting to a datacentre plus the datacentre's own draw. Edge AI is usually the greener option once transmission is counted.
- **Regulatory/functional safety.** A safety interlock cannot depend on an internet link. Anything that must act within a bounded time must be local.

---

## 2.6 Honest disadvantages of Edge AI — say these before the examiner does

1. **Model updates are a logistics problem.** Retraining is easy; getting the new `.onnx` onto 500 devices in a plant is an MLOps problem (see industry section: NVIDIA Fleet Command, Azure IoT Edge, balena).
2. **No fleet-wide learning at inference time.** The device sees only its own machine.
3. **Compute ceiling.** You cannot run a 7B-parameter model on a Pi. Model design is constrained by the target — this is a feature of your project, not a bug: it forces the small architecture you chose.
4. **Physical security.** Someone can steal the SD card and extract your model. Cloud weights cannot be pocketed.
5. **Debugging is harder.** No easy `print` into a dashboard; you need remote logging (your SQLite + Flask setup partially solves this).
6. **Drift goes unnoticed.** With no central monitoring, a device whose sensor has drifted keeps confidently producing garbage. Mitigation: periodically upload prediction summaries for central drift monitoring — a hybrid edge/cloud design, which is what industry actually deploys.

---

## Summary

- Edge AI = inference at the data source; the topology is cloud → fog → edge → endpoint, and your project occupies the bottom three tiers.
- The Pi 4 qualifies as an edge device: full OS, 8 GB RAM, Cortex-A72 with NEON SIMD, Wi-Fi, GPIO, ~6 W.
- Inference is one forward pass; training is hundreds of epochs of forward + backward + optimiser state. That asymmetry is why you train on the laptop and infer on the Pi.
- Advantages: latency, privacy, bandwidth, offline operation, energy, safety. Disadvantages: update logistics, compute ceiling, physical security, silent drift.

## Key takeaways

1. "Edge" is a network-topology word, not a marketing word.
2. NEON SIMD, not clock speed, is what makes ARM inference viable — and it is why int8 quantisation pays off (4 fp32 lanes → 16 int8 lanes).
3. 1 MB of shared L2 cache is the design constraint that should keep your model small.
4. The bandwidth calculation (380 GB/year/node) is your strongest single justification slide.

## Common interview questions

1. Define edge computing and give a topology diagram.
2. Why is inference cheaper than training in both memory and FLOPs? Quantify.
3. What hardware features make a CPU good at inference? (SIMD width, cache size, memory bandwidth, int8 support.)
4. When would you *not* use edge AI? (Model too large; task needs global context across many machines; retraining loop must be tight; device fleet too small to justify engineering.)
5. What is TinyML and how is it different from edge AI?
6. What is federated learning and how does it relate to edge deployment?

## Common viva questions

1. "Why not just run the model on the ESP32 itself?" → 520 KB SRAM, no FPU-friendly runtime for attention, and the model plus window buffer would not fit. The ESP32's job is acquisition and feature extraction. (Add: a distilled fixed-sensor variant *could* run there via TFLite Micro — good future scope.)
2. "Why not use a Jetson Nano/Orin?" → Overkill and over-budget for a model this small; the Pi proves the "any hardware" claim on a CPU-only ARM target, which is the harder case.
3. "What is the actual inference latency you measured?" → Have p50/p95/p99 numbers ready. Do not guess.
4. "Does the Pi use its GPU?" → No. The VideoCore VI has no practical ONNX Runtime execution provider; all inference is CPU with NEON.
5. "What if the Wi-Fi fails?" → The Pi keeps inferring on the last data; the ESP32 buffers. Demonstrate it live.

## Practical implementation tips

- Use **64-bit** Raspberry Pi OS. The 32-bit build costs you ~30 % on ONNX Runtime and there is no aarch64 wheel story on it.
- Pin `intra_op_num_threads` explicitly (start with 2 or 3, not 4 — leave a core for the Flask app and the OS). Benchmark; more threads is often slower for small models due to synchronisation overhead.
- Add a heatsink. Log `vcgencmd measure_temp` alongside latency so you can prove throttling did or did not occur.
- Boot from a USB SSD if you have one — SD cards are the number-one cause of mysterious Pi failures during long logging runs.

## Suggested improvements for this project

1. **Measure and plot a latency histogram** on the Pi, not just a mean. It is a five-line matplotlib addition and it makes the deployment section look professional.
2. **Add a power measurement** (USB inline power meter, ~₹500) and report joules per inference. Almost no student project does this; it is a differentiator.
3. **Implement the offline-resilience demo** as an explicit test case: kill the network, show buffered operation, restore, show catch-up.
4. **Add a hybrid uplink**: the Pi keeps full local operation but posts a daily summary to a cloud endpoint. That mirrors real industrial architecture and gives you a slide comparing pure-edge vs hybrid.
