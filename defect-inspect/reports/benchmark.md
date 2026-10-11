# CPU benchmark and quantisation study

Regenerate with:

```bash
uv run python -m inspector.bench --compare
uv run python scripts/parity.py
```

**All numbers below were measured on this project's own code and data.** Two
machines appear and they are never mixed: the **Raspberry Pi 5**, which is the
deployment target and the only platform whose FPS figures may be quoted as
results, and an **AMD Ryzen AI 9 HX 370** laptop with the GPU unused, which is
the development host and appears only for comparison.

---

## Measured on the Raspberry Pi 5

Raspberry Pi 5 Model B Rev 1.1, four Cortex-A76 at 2.4 GHz, Raspberry Pi OS
Bookworm 64-bit, Python 3.11.2, onnxruntime 1.28.0, Active Cooler running.
60 class-balanced real test images, 3 repeats, 10 warm-up frames.

| model | threads | pre | infer | post | total | p95 | FPS | size MB |
|-------|--------:|----:|------:|-----:|------:|----:|----:|--------:|
| model.onnx (FP32) | 1 | 1.75 | 88.65 | 0.36 | 90.77 | 90.90 | 11.0 | 12.1 |
| model.onnx (FP32) | 2 | 1.75 | 49.24 | 0.36 | 51.36 | 51.47 | 19.5 | 12.1 |
| model.onnx (FP32) | 4 | 2.02 | 41.11 | 0.40 | 44.70 | 78.58 | 22.4 | 12.1 |
| model_int8.onnx | 1 | 1.75 | 52.16 | 0.37 | 54.29 | 54.36 | 18.4 | 5.6 |
| model_int8.onnx | 2 | 1.76 | 31.08 | 0.37 | 33.23 | 33.31 | 30.1 | 5.6 |
| model_int8.onnx | 4 | 2.01 | 23.23 | 0.38 | 27.27 | 51.50 | **36.7** | 5.6 |

Peak RSS 122 MB. Average 2.22 boxes per frame after NMS.

**36.7 FPS against a design target of 20.** The full pipeline — capture,
detect, gate, decide, store, eject — measured 17.1 FPS end to end in
`tests/test_pipeline.py`, which is the honest system-level figure.

### Where the difference goes

This was previously attributed to SQLite writes and JPEG encoding of failure
images. That attribution was wrong, and wrong by more than an order of
magnitude. Measured per stage with `scripts/profile_stages.py` on the Pi 5,
180 frames:

| stage | calls | median ms | mean ms | p95 ms | per frame ms |
|---|--:|--:|--:|--:|--:|
| detect | 180 | 29.30 | 38.62 | 86.59 | 38.62 |
| anomaly (gated) | 180 | 0.00 | 9.30 | 107.77 | 9.30 |
| draw | 180 | 0.16 | 0.23 | 0.43 | 0.23 |
| imwrite (failures) | 165 | 0.57 | 0.80 | 1.20 | 0.73 |
| imencode (preview) | 180 | 0.30 | 0.53 | 0.80 | 0.53 |
| db insert | 180 | 0.14 | 0.28 | 0.24 | 0.28 |

Storage and failure-image encoding together cost **1.01 ms of a 49.7 ms
budget — 2%**. The gap is inference plus the anomaly backstop.

The anomaly row is the one to read carefully: a median of 0.00 ms with a mean
of 9.30 ms means it rarely runs and costs about 110 ms when it does, which is
roughly four inferences. The rate cap of 3 calls/second is therefore not a
minor safeguard but the single most important throughput parameter in the
configuration — at the cap it can claim a third of the frame budget on its own.
Raising `anomaly.max_calls_per_second` trades throughput away faster than
anything else exposed in `configs/default.yaml`.

**These particular numbers were taken on a throttling board** — detect
mean/median is 1.32, and the profiler warns when that ratio exceeds 1.25. The
2% conclusion about storage holds regardless, being a ratio between stages
measured in the same run, but the absolute figures must be retaken on an
adequate supply. See the under-voltage note below.

### Thermal validity of the measurement

An FPS number taken from a throttling SoC is not reproducible, so the run was
bracketed: **46.6 C before, 57.6 C after**, fan at 2402 rpm, and
`vcgencmd get_throttled` reading `0x50000` both before and after — unchanged.
Bits 16 and 18 are sticky "has occurred" flags left by a brief under-voltage
event two seconds into boot, visible in `dmesg` as `Undervoltage detected!`
followed by `Voltage normalised`. No new bit was raised and no current-throttle
bit (0 to 3) was ever set, so nothing throttled while the benchmark ran.

`/boot/firmware/config.txt` carries a deliberately aggressive fan curve
(`fan_temp0=30000`, stepping to full speed by 60 C) so that a long run cannot
drift into a thermal cap. Without it the stock curve idles the fan until 50 C
and the SoC coasts into the 60s during sustained inference.

### Under-voltage, 2026-09-18: the board is no longer in that state

The bracketing above described a healthy board. It is not healthy now, and
every FPS figure in this file predates the change.

Measured 2026-09-18 during a live dashboard run:

- `dmesg` — `hwmon3: Undervoltage detected!` at 5679 s uptime, with no
  matching `Voltage normalised`.
- `vcgencmd get_throttled` — alternating between `0x50000` and **`0x50005`**.
  Bits 0 and 2 are the *live* under-voltage and throttled flags, not the
  sticky bits 16+, so this is happening rather than having happened.
- `vcgencmd measure_clock arm` — collapsing from 2400 MHz to **1000 MHz**,
  sampled at 2-second intervals, at 62 C with the fan running. Temperature is
  therefore not the cause.
- Live p95 latency through the dashboard read 147-157 ms against the 51 ms in
  the table above.

The cause is supply, not cooling: a 2.4× clock drop at 62 C with an under-
voltage flag set is a power-delivery fault. The remedy is the official 27 W
USB-C PD supply and a known-good cable, after which `get_throttled` should read
`0x0` at boot and stay there.

**Consequence for this file and for any paper drawn from it:** the numbers
above are not currently reproducible on this hardware. They were taken under
verified-good conditions and are not withdrawn, but they must be retaken and
re-bracketed before publication, and the profiler in
`scripts/profile_stages.py` will warn while the fault persists.

### The roofline projection was wrong, and wrong in the safe direction

An earlier version of this file carried a projection derived from the A76
specification — two 128-bit NEON FMA pipes, 16 FLOP per cycle per core, 2.4 GHz
across four cores, 153.6 GFLOP/s peak, assumed 25 to 40 percent achieved:

> Projected Pi 5 performance: 40 to 64 ms per frame, i.e. 16 to 25 FPS.

The device delivered **36.7 FPS**, some 47 percent above the optimistic end of
that band. The efficiency assumption, not the arithmetic, was at fault:
onnxruntime's ARM64 kernels achieve considerably better than 40 percent of peak
on this workload. The lesson is that a roofline band errs in both directions,
and "estimate" has to mean estimate — not "conservative floor".

### Threads: the fastest setting is not the steadiest

Read the `p95` column against `total`, not just the FPS column:

| threads | median ms | p95 ms | spread | FPS |
|--------:|----------:|-------:|-------:|----:|
| 2 | 33.23 | 33.31 | **+0.2%** | 30.1 |
| 4 | 27.27 | 51.50 | **+89%** | 36.7 |

Four threads win on throughput and lose badly on determinism: the worst frame
takes nearly twice the median. Two threads are effectively jitter-free. The
cause is that with all four cores committed to inference there is nothing left
for the OS, the capture thread, the SQLite writer and the Flask streamer, so
those steal time from the inference pool at unpredictable moments.

For a conveyor this matters more than the headline. A line running at 20 parts
per second has a 50 ms budget per part; the 4-thread configuration exceeds it
on the tail while the 2-thread configuration never comes close. **The default
stays at 4 threads because the current target is 20 FPS with margin, but if the
mechanical design ever needs a hard latency guarantee, drop to 2** — it costs
18 percent of throughput and buys a bound.

---

## Development-host reference (x86, not a result)

The rest of this section is the same benchmark on the laptop. It is kept for
one reason: the quantisation study below was carried out there, and its
accuracy numbers are hardware-independent. **The FPS column in the next table
is a fact about a Zen 5 laptop and must never be quoted as a Pi figure.**

| | Ryzen AI 9 HX 370 (Zen 5) | Pi 5 (Cortex-A76) |
|---|---|---|
| Clock | ~5.0 GHz boost | 2.4 GHz |
| SIMD | AVX-512, 512-bit | NEON, 128-bit |
| INT8 lanes per instruction | 64 (VNNI) | 16 (SDOT) |
| Cores available | 12 | 4 |

## Latency and frame rate

60 class-balanced real test images, 3 repeats, 10 warm-up frames.
Times are median milliseconds per frame; `p95` is the tail.

| model | threads | pre | infer | post | total | p95 | FPS | size MB |
|-------|--------:|----:|------:|-----:|------:|----:|----:|--------:|
| model.onnx (FP32) | 1 | 2.11 | 38.22 | 0.49 | 41.13 | 47.61 | 24.3 | 12.1 |
| model.onnx (FP32) | 2 | 2.51 | 29.55 | 0.56 | 32.63 | 42.12 | 30.6 | 12.1 |
| model.onnx (FP32) | 4 | 2.53 | 20.81 | 0.55 | 23.94 | 25.89 | **41.8** | 12.1 |
| model_int8.onnx | 1 | 2.17 | 27.86 | 0.49 | 30.89 | 33.98 | 32.4 | 5.6 |
| model_int8.onnx | 2 | 2.48 | 21.69 | 0.53 | 24.80 | 29.75 | 40.3 | 5.6 |
| model_int8.onnx | 4 | 2.58 | 16.49 | 0.55 | 19.74 | 21.82 | **50.6** | 5.6 |

Peak RSS 108 MB. Average 2.22 boxes per frame after NMS.

### How well did the ratios actually transfer?

This file previously claimed that although absolute FPS cannot cross an
architecture boundary, ratios can. Now that both platforms have been measured,
that claim can be checked instead of asserted — and it only half survives:

| ratio (INT8, unless stated) | x86 | Pi 5 | error |
|---|---:|---:|---:|
| INT8 over FP32, 4 threads | 1.21x | **1.64x** | 36% |
| 4 threads over 1 thread | 1.56x | **1.99x** | 28% |
| 4 threads over 1 thread (FP32) | 1.72x | **2.04x** | 19% |

Both ratios moved by roughly a third, and both moved the same way: **ARM gains
more from quantisation and more from threading than x86 does.**

The reasons are structural, not noise. AVX-512 makes x86 FP32 unusually strong,
so INT8 has proportionally less to win there, whereas NEON's FP32 throughput is
modest and `SDOT` gives INT8 a bigger relative jump. And the laptop's poor
thread scaling is partly a boost-clock artefact — one active core runs near
5.0 GHz while four active cores clock down, so the 1-thread baseline is
inflated. The Pi's four A76 cores all run at 2.4 GHz regardless, which is why
it scales closer to linear.

**Conclusion: treat a host ratio as a direction, not a coefficient.** It will
tell you that INT8 is worth doing. It will not tell you by how much.

Three things worth reading off this table:

- **Threading scales sub-linearly.** Four threads give 1.75x over one, not 4x.
  Convolution on small feature maps is partly memory-bound, and the head runs
  on tensors too small to split usefully.
- **Post-processing is nearly free here** — 0.55 ms — because the confidence
  threshold discards all but a couple of dozen candidates before NMS sees them.
  This is a direct consequence of `conf_threshold: 0.35`. Lowering it to catch
  more crazing would raise this number.
- **Preprocessing does not scale with threads at all** (2.1 to 2.6 ms flat) and
  becomes 13% of the frame budget in the fastest configuration. This file used
  to predict that it would "matter more on the Pi". **It does not.** The Pi
  measured 1.75 to 2.02 ms — slightly *faster* than the laptop, because the
  letterbox is a single-threaded `cv2.resize` plus a copy, which is bound by
  memory bandwidth on a 200x200 image rather than by core width, and the
  laptop's figure includes scheduler overhead from a busier machine. The
  prediction was wrong; the preallocated-buffer optimisation is not needed.
  This changes once a real camera delivers 1920x1080 frames instead of 200x200
  files, which is the point at which to re-measure.

---

## The quantisation study

### What happened

The first INT8 export quantised the whole graph. It was fast and badly broken:

| model | mAP@50 | mAP@50-95 | precision | recall | FPS (4 thr) |
|-------|-------:|----------:|----------:|-------:|------------:|
| FP32 ONNX | 0.7354 | 0.3746 | 0.643 | 0.681 | 41.8 |
| INT8, whole graph | **0.4077** | 0.2462 | 0.763 | **0.417** | 67.5 |
| INT8, FP32 head | **0.7324** | 0.3898 | — | — | 50.6 |

A 33-point collapse in mAP@50. The tell is the direction the two error rates
moved: **precision went up while recall went down.** The model was not becoming
confused, it was becoming silent — every marginal detection fell below
threshold and disappeared. That points at the part of the network that computes
confidence, not at the features.

### Why the head quantises badly

The detection head, `model.22` in the exported graph, does three things that
8-bit arithmetic handles poorly, all of them in 66 nodes that carry very little
of the total compute:

1. **Softmax + DFL convolution.** YOLOv8 predicts each box edge as a
   distribution over 16 bins and takes its expectation to get a sub-pixel
   coordinate. Rounding that distribution to 8-bit steps coarsens every box.
2. **Sigmoid.** It compresses a wide logit range into 0 to 1. Near the decision
   threshold, one INT8 step can be the whole difference between a detection and
   a miss.
3. **Box arithmetic (Sub, Add, Div, Mul).** These operate on absolute pixel
   coordinates spanning 0 to 320, a far wider range than the activations the
   calibrator saw in the backbone, so a shared scale fits them badly.

A separate symptom confirmed the diagnosis: the fully quantised model emitted
confidence values **above 1.0**, measured up to 1.28, from a tensor that is the
output of a sigmoid and therefore mathematically bounded by 1.
`detector.py` clips this, and the clip is a safety net, not the fix.

### The fix and what it cost

`scripts/export_onnx.py` now excludes the 66 head nodes from quantisation by
default. Backbone and neck, which hold nearly all the multiply-accumulates,
stay INT8; the head runs FP32.

| | whole-graph INT8 | INT8 with FP32 head | change |
|---|---:|---:|---:|
| mAP@50 | 0.4077 | 0.7324 | **+0.325** |
| Recall vs FP32 boxes | 0.508 | 0.924 | +0.416 |
| Mean IoU vs FP32 boxes | 0.9035 | 0.9442 | +0.041 |
| FPS at 4 threads | 67.5 | 50.6 | -16.9 |
| Model size | 3.4 MB | 5.6 MB | +2.2 MB |

**A 25% speed loss bought back 32 points of mAP.** The trade is not close.
`--quantise-head` re-enables the fast, broken version for anyone who wants to
reproduce the failure.

One caveat specific to ARM, and it is still **open**. This fix leaves the head
in FP32, and FP32 NEON has no VNNI-style advantage to lose, so the 25 percent
cost measured on x86 could be larger on the Pi. That has not been measured:
only the head-excluded model was deployed, so there is no whole-graph INT8
figure from ARM to compare against.

It is a one-command experiment — `export_onnx.py --quantise-head`, copy, run
`bench.py` — and worth doing, because the indirect evidence points both ways.
INT8 buys more on ARM than on x86 (1.64x against 1.21x), which argues the FP32
head is proportionally more expensive there; but 36.7 FPS already clears the
target by 84 percent, so there is no pressure to spend accuracy on speed.

If it ever does prove expensive, the next step is not to quantise all 66 nodes
but to quantise selectively: the DFL softmax and the sigmoid are the two that
caused the damage.

Final accuracy cost of quantisation: **0.3 points of mAP@50** (0.7354 to
0.7324), against a 3-point budget.

---

## Pipeline correctness

`scripts/parity.py` compares boxes from our hand-written ONNX decode against
what ultralytics itself predicts from `best.pt`, over 60 class-balanced images.

| model | ref boxes | our boxes | matched | recall | mean IoU | class errors |
|-------|----------:|----------:|--------:|-------:|---------:|-------------:|
| model.onnx (FP32) | 132 | 132 | 132 | **1.000** | **1.0000** | 0 |
| model_int8.onnx | 132 | 133 | 122 | 0.924 | 0.9442 | 0 |

The FP32 row is the important one. A perfect match on all 132 boxes at IoU
1.0000 proves that our letterboxing, tensor layout, output decode, NMS and
inverse coordinate mapping are all correct — none of which raises an error when
wrong. Because that row is exact, any difference in the INT8 row is
attributable to quantisation alone.

---

## Accuracy on the held-out test split

`best.pt`, evaluated by `scripts/evaluate.py` on the test split (181 images,
438 boxes) which was never used for model selection.

| | mAP@50 | mAP@50-95 |
|---|---:|---:|
| Overall | 0.7107 | 0.4063 |

| class | precision | recall | mAP@50 |
|-------|----------:|-------:|-------:|
| crazing | 0.549 | **0.222** | **0.385** |
| inclusion | 0.710 | 0.762 | 0.871 |
| patches | 0.750 | 0.895 | 0.911 |
| pitted_surface | 0.560 | 0.728 | 0.745 |
| rolled-in_scale | 0.546 | 0.507 | 0.514 |
| scratches | 0.649 | 0.860 | 0.837 |

**Crazing is the open problem.** Recall of 0.222 means the model misses roughly
four out of five crazing defects. Crazing is a fine network of hairline cracks
with very low contrast against the mill finish, and its box boundaries are
genuinely ambiguous — annotators disagree about where one region ends. It is
the lowest-AP class in most published NEU-DET results too, so this is a
property of the problem, not only of our training run.

This is a real limitation and is stated as one. Options, in order of expected
value: a per-class confidence threshold; contrast-limited adaptive histogram
equalisation in preprocessing; oversampling crazing during training; and
PatchCore, which does not need to recognise crazing at all in order to notice
that the surface is not normal.
