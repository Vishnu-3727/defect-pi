# 13 — CPU Optimisation for the Raspberry Pi 5

**Prerequisites:** [12 — ONNX and the runtime](12_onnx_and_runtime.md).

---

## 1. Concept

The Pi 5 has four Cortex-A76 cores at 2.4 GHz, 128-bit NEON vector units, and no
usable GPU for neural networks. Everything the inspection system does happens on
those four cores, shared between the detector, the anomaly module, the camera,
the database and a web server.

CPU optimisation here means three things: making each operation cheaper
(quantisation), using all four cores (threading), and not doing work that buys
nothing (input size, gating).

---

## 2. Why it is needed

A conveyor sets a deadline. If a part passes the camera every 100 ms and
inspection takes 150 ms, parts go by uninspected — not slowly, but *not at all*.

Latency is therefore a correctness property, not a comfort. And unlike accuracy,
it cannot be improved after deployment by collecting more data.

---

## 3. Real-world example

The classic edge-AI failure: a model that reaches 0.95 mAP on a workstation is
deployed to a device where it runs at 2 FPS, and the line runs at 20 parts per
second. The project is technically successful and operationally useless.

The discipline that avoids it is to **measure on the target early and let the
budget constrain the model choice**, rather than picking the most accurate model
and hoping. That is why YOLOv8n was chosen in Module 08 before training began,
not after benchmarking a larger model.

---

## 4. How it works

### First: know what you are optimising

Measured with `python -m inspector.bench`, INT8, four threads:

| Stage | ms | Share | Scales with threads? |
|-------|---:|------:|----------------------|
| Preprocess | 2.58 | 13% | **no** |
| Inference | 16.49 | 84% | yes, sub-linearly |
| Decode + NMS | 0.55 | 3% | no |
| **Total** | **19.74** | | |

Inference dominates, so that is where effort goes. But notice preprocessing does
not thread at all — as inference gets faster, its 13% share grows. On a Pi, with
a larger camera frame to letterbox, it will matter more.

**Optimising the 3% would be a waste of a week.** Measure first.

### Quantisation

Weights and activations are stored as 8-bit integers instead of 32-bit floats.

$$ x_{int8} = \text{round}\left(\frac{x_{float}}{s}\right) + z $$

where $s$ is a scale and $z$ a zero point, chosen per tensor — or, with
per-channel quantisation, per output channel.

Three separate benefits, and the third is usually the largest:

1. **Arithmetic.** Integer multiply-accumulate is cheaper than float.
2. **Memory.** A quarter of the bytes. The model went 12.1 MB → 5.6 MB.
3. **Bandwidth.** A quarter of the traffic between RAM and cache. Convolution on
   a CPU is frequently memory-bound, so this often dominates the other two.

**Static** quantisation needs calibration: real images are pushed through the
FP32 graph while the tool records each activation's actual range. Calibrating on
noise would produce ranges no real image reaches, and the model would be useless.
Our calibration set is 198 images, class-balanced:

```python
# Balanced across classes: calibrating on 200 crazing images would give
# ranges skewed to one texture.
```

### The quantisation study — what went wrong and how it was found

This is the most instructive result in the project.

The first INT8 export quantised the entire graph. It was fast and broken:

| model | mAP@50 | precision | recall | FPS |
|-------|-------:|----------:|-------:|----:|
| FP32 | 0.7354 | 0.643 | 0.681 | 41.8 |
| INT8, whole graph | **0.4077** | 0.763 | **0.417** | 67.5 |

A 33-point collapse. **The diagnostic signal is the direction the two error
rates moved: precision went up while recall went down.** The model was not
becoming confused — it was going silent. Marginal detections were falling below
threshold and vanishing. That points at the part of the network that computes
confidence.

A second symptom confirmed it: the quantised model emitted **confidence values
above 1.0, measured up to 1.28**, from a tensor that is the output of a sigmoid
and is therefore mathematically bounded by 1. Quantisation had rounded it onto a
grid whose top step lands above 1.

**Why the head (`model.22`, 66 nodes) quantises badly:**

1. **Softmax and the DFL convolution.** YOLOv8 predicts each box edge as a
   distribution over 16 bins and takes its expectation for sub-pixel precision.
   Rounding those probabilities to 8-bit steps coarsens every box edge.
2. **Sigmoid.** It compresses a wide logit range into 0 to 1. Near the decision
   threshold, one INT8 step spans the difference between a detection and a miss.
3. **Box arithmetic** (`Sub`, `Add`, `Div`, `Mul`) on absolute pixel coordinates
   spanning 0 to 320 — a far wider range than the backbone activations the
   calibrator saw, so a shared scale fits them badly.

**The fix:** exclude those 66 nodes. The backbone and neck, which hold nearly
all the multiply-accumulates, stay INT8. The head is cheap because it runs on
small feature maps.

| | whole-graph INT8 | INT8, FP32 head | change |
|---|---:|---:|---:|
| mAP@50 | 0.4077 | 0.7324 | **+0.325** |
| Recall vs FP32 boxes | 0.508 | 0.924 | +0.416 |
| FPS at 4 threads | 67.5 | 50.6 | −16.9 |

**A 25% speed loss bought back 32 points of mAP.** Final quantisation cost:
0.3 points of mAP@50, against a 3-point budget.

An ARM caveat: the head now runs FP32, and FP32 NEON has no VNNI-style advantage
to lose, so the 25% cost measured on x86 may be larger on the Pi. If it is, the
next step is selective — the DFL softmax and the sigmoid caused the damage, not
all 66 nodes.

### Threading

```python
opts.intra_op_num_threads = 4    # one per Cortex-A76 core
opts.inter_op_num_threads = 1    # our graph is a chain
```

Measured scaling: 1 → 2 → 4 threads gives 32.4 → 40.3 → **50.6** FPS.
**1.57x from four cores, not 4x.**

The shortfall is not a bug. Convolutions on small feature maps are partly
memory-bound, and the head operates on tensors too small to split usefully.
Practical consequences:

- **More threads than cores is worse, not better.** It adds context switching.
- **Do not budget on linear scaling.** If a model needs 4x, four cores will not
  provide it.
- **The system has other work.** Flask, SQLite and the camera also need CPU.
  Reserving all four cores for inference can starve them.

### Input size

The strongest single lever, and free.

Compute scales roughly with the square of the input dimension:

| imgsz | relative compute | justification |
|-------|-----------------:|---------------|
| 640 | 4.0x | tutorial default |
| **320** | **1.0x** | **ours — nearest multiple of 32 above native 200x200** |
| 256 | 0.64x | below native; would discard real detail |

NEU-DET is natively 200x200. Training at 640 would upsample every image and
quadruple the compute to process interpolated pixels carrying no information.
Choosing 320 was worth roughly 4x before a single optimisation was applied.

The constraint is that the backbone downsamples five times, so the input must be
a multiple of 32. `config.py` enforces it:

```python
if cfg.detector.imgsz % 32 != 0:
    raise ValueError(f"detector.imgsz must be a multiple of 32 (the network "
                     f"stride), got {cfg.detector.imgsz}")
```

### Not doing work: the two gates

The cheapest operation is one you skip. PatchCore costs **36.7 ms** — more than
the detector — so it does not run per frame:

```yaml
gate_conf_below: 0.55        # only when the detector is unsure
max_calls_per_second: 3      # and never more than 3 times a second
```

Two gates for two reasons. The **confidence gate** is about value: when YOLOv8
already reports a defect at 0.9, a second opinion changes no decision. The
**rate gate** is about the budget: without it, a run of uncertain frames would
call PatchCore on every one and the frame rate would collapse exactly when the
line is producing marginal parts — the worst possible moment.

`conf_threshold` is a speed knob too. It is applied to 2100 candidates *before*
NMS, which is why post-processing costs 0.55 ms rather than tens of milliseconds.

---

## 5. Mathematical intuition

**Quantisation error.** Mapping a range $[a, b]$ onto 256 levels gives a step
size $s = (b-a)/255$ and a maximum rounding error of $s/2$. Accuracy therefore
depends entirely on how tight $[a,b]$ is:

- A tensor with range $[0, 1]$ → step 0.004. Fine.
- A tensor with range $[0, 320]$ → step 1.25. That is over a pixel of box error
  before anything else goes wrong — which is exactly the box-arithmetic problem
  above.

**Per-channel** quantisation gives each output channel its own scale. Channels
in a convolution routinely differ by an order of magnitude in range, so one
shared scale wastes most of the 256 levels on the quiet channels. This is
measurably better, and it is what forced opset 13.

**The Cortex-A76 roofline.** Each core has two 128-bit NEON pipes; each issues
one fused multiply-add over 4 float32 lanes per cycle, and an FMA is 2 FLOPs:

$$ 2 \times 4 \times 2 = 16 \text{ FLOP/cycle/core} $$
$$ 16 \times 2.4\,\text{GHz} \times 4 \text{ cores} = 153.6 \text{ GFLOP/s peak} $$

Real convolution achieves 25 to 40% of peak. At 2.0 GFLOPs per frame:

$$ \frac{2.0}{153.6 \times 0.40} = 32.5\,\text{ms} \qquad \frac{2.0}{153.6 \times 0.25} = 52\,\text{ms} $$

Adding preprocessing and NMS gives **40 to 64 ms per frame, 16 to 25 FPS**.

**That is a projection from a published specification, not a measurement**, and
it is labelled as one everywhere it appears — including in the benchmark tool's
own output. The Pi has been ordered; these numbers get replaced by measurements.

---

## 6. Software implementation

`src/inspector/bench.py` refuses to let an x86 measurement masquerade as a Pi
result. When the host is not ARM it prints:

```
=============================================================================
THE TABLE ABOVE IS NOT A RASPBERRY PI RESULT AND DOES NOT PREDICT ONE.
=============================================================================
    clock          ~5.0 GHz boost      vs   2.4 GHz
    SIMD width     AVX-512 (512-bit)   vs   NEON (128-bit)
    INT8 per instr 64 lanes (VNNI)     vs   16 lanes (SDOT)
```

This matters because an x86 core with AVX-512 VNNI processes four times the INT8
lanes per instruction at twice the clock. **Matching the thread count controls
how many cores are used; it does nothing about what each core can do.** Quoting
laptop FPS for a Pi overstates it by a large and unknown factor.

The benchmark also refuses to measure on synthetic input:

```python
# Benchmarking on a blank array would be dishonest in a specific way: NMS cost
# scales with how many candidate boxes survive the confidence threshold, and a
# blank frame produces none. The measured post-processing time would be zero.
```

and it discards warm-up frames, which are several times slower than steady state
while ONNX Runtime allocates arenas and populates caches.

---

## 7. Hardware implementation

### The upgrade ladder, if the Pi proves too slow

In order of cost:

| Step | Cost | Expected gain | Trade |
|------|-----:|--------------|-------|
| Reuse preprocessing buffers | free | ~1 ms/frame | none |
| Drop the stride-8 detection head | free | large | loses small-defect recall — possibly affordable at 11.8% median box area |
| Selective head quantisation | free | recovers part of the 25% | needs care to avoid the mAP collapse |
| Active cooling | ~₹500 | prevents thermal throttling | none |
| Hailo-8L on the Pi 5 AI kit | ~₹8000 | 10x+ | needs model recompilation |
| Jetson Orin Nano | ~₹25000 | large | different platform entirely |

### Thermal throttling — the one that surprises people

The Pi 5 throttles at 80 °C. A benchmark run for thirty seconds gives a number a
production line will never sustain, because the line runs for eight hours.

**Always measure sustained throughput, not peak**, and fit a heatsink or the
active cooler before drawing conclusions. This is the most common reason a
demo-day number cannot be reproduced.

### Storage

SQLite on an SD card, at 25 inserts per second, is a real wear concern. Our
`storage.py` addresses it:

```python
self.conn.execute("PRAGMA journal_mode=WAL")
self.conn.execute("PRAGMA synchronous=NORMAL")
```

plus batched commits — except for failures, which commit immediately:

```python
# A FAIL commits immediately. Losing a pass record on power loss is a
# statistic; losing the record of a part that was physically ejected is a
# traceability hole.
```

---

## 8. Common mistakes

- **Optimising before measuring.** Our decode is 3% of the frame. A week spent
  there would gain nothing.
- **Assuming quantisation is free.** Ours cost 33 points of mAP before the head
  was excluded.
- **Assuming linear thread scaling.** We measured 1.57x from four cores.
- **Quoting x86 numbers for ARM.** Different SIMD width, different clock,
  different instruction set.
- **Benchmarking on blank frames.** Post-processing cost disappears.
- **Ignoring warm-up.** The first frames are several times slower.
- **Measuring for ten seconds.** Thermal throttling appears later.
- **Forgetting the rest of the system.** Flask, SQLite and the camera also want
  CPU time.

---

## 9. Viva questions

**Q: How did you make the model fast enough for a Raspberry Pi?**
Four things, in order of value. Training at 320 rather than 640, since the images
are natively 200x200 — worth 4x before anything else. INT8 quantisation of the
backbone and neck — 1.21x measured, and a quarter of the memory traffic. Four
threads, one per core — 1.57x. And gating the expensive anomaly module so it
runs at most three times a second instead of per frame.

**Q: What is quantisation and what did it cost you?**
Storing weights and activations as 8-bit integers instead of 32-bit floats,
which reduces arithmetic cost, memory and — usually most importantly — bandwidth
between RAM and cache. Quantising the whole graph cost 33 points of mAP@50. We
diagnosed it from the error direction: precision rose while recall collapsed,
meaning marginal detections were vanishing, which pointed at the confidence head.
Excluding the head's 66 nodes recovered 32 of those points for a 25% speed loss.
Final cost, 0.3 points.

**Q: Why does the head quantise so badly?**
Three reasons. The DFL softmax turns a 16-bin distribution into a sub-pixel
coordinate, and 8-bit bins coarsen it. The sigmoid compresses a wide logit range
into 0 to 1, where one INT8 step spans detect-versus-miss. And the box arithmetic
works on 0-to-320 pixel coordinates, a much wider range than the activations the
calibrator saw elsewhere, so a shared scale fits it poorly. A confirming symptom:
the fully quantised model emitted confidences of 1.28 from a sigmoid output that
cannot exceed 1.

**Q: What frame rate do you expect on the Pi 5?**
16 to 25 FPS, and that is a projection rather than a measurement. It comes from
the A76 roofline — two 128-bit NEON FMA pipes, 16 FLOP per cycle per core,
2.4 GHz, four cores, 153.6 GFLOP/s peak, at 25 to 40% achieved efficiency for
real convolution, on a 2.0 GFLOP model. Our laptop numbers are x86 with AVX-512
and cannot be quoted for ARM. The Pi is on order and the tool will print the real
figure.

**Q: Why did you not use a GPU or an accelerator?**
Cost and scope. The Pi 5 has no usable GPU for neural networks, and the projected
CPU frame rate already exceeds what the conveyor needs. A Hailo-8L would give
more than 10x for about ₹8000 and model recompilation — documented as the
upgrade path if the measurement disappoints.

**Q: What is thermal throttling and how does it affect your numbers?**
The Pi 5 reduces clock speed above 80 °C. A thirty-second benchmark can look
fine while an eight-hour shift runs slower. Sustained throughput must be measured
with adequate cooling fitted, or the demo-day number will not survive contact
with production.

---

## 10. Summary

1. Measure before optimising. Inference is 84% of our frame; decode is 3%.
2. Input size was the biggest lever and it was free — 320 instead of 640 is 4x.
3. INT8 gives 1.21x and quarters memory traffic, but quantising the detection
   head destroyed 33 points of mAP. Exclude it.
4. Four cores give 1.57x, not 4x. Never budget on linear scaling.
5. The cheapest work is skipped work — two gates keep a 36.7 ms module
   affordable.
6. Projected 16 to 25 FPS on the Pi from the A76 roofline, labelled an estimate
   until the hardware arrives.

**Next:** [14 — PatchCore and anomaly detection](14_patchcore_anomaly.md)
