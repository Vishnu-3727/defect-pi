# Sections 11 & 12 — ONNX and Edge Deployment

## 11.1 What ONNX is

**Plain language.** A model is two things: a recipe (what operations, in what order) and ingredients (the learned numbers). PyTorch stores them in a PyTorch-specific way that only PyTorch can read. ONNX is a neutral way of writing both down, so any tool can read it.

**Technical.** ONNX (Open Neural Network Exchange) is:

1. A **file format** — Protocol Buffers (`onnx.proto`) serialising a directed acyclic graph.
2. An **operator set (opset)** — a versioned specification of ~190 standardised operators (`Conv`, `MatMul`, `Softmax`, `Gather`, `Relu`, …) with precisely defined semantics.
3. A **type system** — tensor element types and shapes, with support for symbolic (dynamic) dimensions.

A `.onnx` file contains:

```
ModelProto
├── ir_version, producer_name ("pytorch"), opset_import (e.g. version 17)
└── GraphProto
    ├── input:      [ {name:"x", type: float32[batch, sensors, 30]},
    │                 {name:"mask", type: bool[batch, sensors]}, ... ]
    ├── output:     [ {name:"rul", type: float32[batch, 1]} ]
    ├── initializer: the learned weights, as raw tensors
    └── node[]:     Conv → BatchNormalization → Relu → … → Sigmoid
                    (each node: op_type, inputs, outputs, attributes)
```

Inspect it:

```python
import onnx
m = onnx.load("unify_rul.onnx")
print(onnx.helper.printable_graph(m.graph))   # human-readable dump
onnx.checker.check_model(m)                   # validates structure
```

Or visually with **Netron** (netron.app — drag and drop the file). Put a Netron screenshot of your graph in the report; it is a free, professional-looking figure and it proves the export worked.

## 11.2 Why ONNX exists

Before ONNX, deploying model M from framework F to hardware H required an `F × H` matrix of converters. ONNX turns that into `F + H`:

```
Without ONNX (N×M problem)          With ONNX (N+M)
PyTorch ─┬─▶ ARM CPU                 PyTorch ─┐
         ├─▶ NVIDIA GPU              TF      ─┼─▶ ONNX ─┬─▶ ORT (CPU/ARM)
         └─▶ Qualcomm DSP            JAX     ─┘         ├─▶ TensorRT (NVIDIA)
TF      ─┬─▶ ARM CPU                                    ├─▶ SNPE/QNN (Qualcomm)
         ├─▶ …                                          ├─▶ OpenVINO (Intel)
                                                        └─▶ TVM (anything)
```

Second reason: **dependency weight**. PyTorch on ARM is ~800 MB installed, slow to import, and awkward to build. `onnxruntime` is ~50 MB, imports in ~0.2 s, and has prebuilt aarch64 wheels. For an edge device, that is the whole argument.

Third: **stability**. An ONNX graph is frozen. It has no Python, no custom classes, no pickle. It will load in five years. A `.pt` file depends on your exact Python class definitions being importable.

## 11.3 How conversion works — tracing

`torch.onnx.export` **traces**: it runs your model once with a dummy input, records every operation the tensors actually pass through, and writes those operations into the graph.

**What tracing means in practice — the rules:**

| Do | Don't |
|---|---|
| Use tensor ops throughout | Use `.item()`, `int(tensor)`, `float(tensor)` — they bake the value in as a constant |
| Use `torch.where`, `masked_fill` | Use Python `if tensor > 0:` — only the traced branch survives |
| Derive shapes with `x.shape[0]` (traced symbolically) | Hard-code `x.reshape(256, …)` — the batch size gets frozen |
| Call `model.eval()` first | Export in train mode — BatchNorm/Dropout behave wrongly forever |
| Use `torch.no_grad()` | – |
| Loop a fixed number of times | Loop `for i in range(x.shape[1])` — unrolls to that exact count |

Your model is trace-friendly by construction: pure tensor ops, no control flow, no data-dependent shapes. That is not luck — it is why the architecture is written the way it is. Say that in viva; "designed for exportability" is a real engineering consideration.

**Newer alternative:** `torch.onnx.export(..., dynamo=True)` uses TorchDynamo instead of tracing and handles control flow better. For your model, plain tracing is fine and better documented.

## 11.4 Dynamic axes — the critical part for this project

By default, tracing freezes **every** dimension at the dummy input's size. Export with `(1, 21, 30)` and the model refuses any other shape. That would destroy both your batching and — far worse — your entire "any sensor count" claim.

`dynamic_axes` marks dimensions as symbolic:

```python
import torch

model.eval()

dummy_x    = torch.randn(1, 21, 30)
dummy_mask = torch.ones(1, 21, dtype=torch.bool)
dummy_ids  = torch.arange(21).unsqueeze(0)
dummy_dom  = torch.zeros(1, dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_x, dummy_mask, dummy_ids, dummy_dom),
    "unify_rul.onnx",
    input_names  = ["x", "mask", "sensor_ids", "domain_id"],
    output_names = ["rul", "attn"],
    dynamic_axes = {
        "x":          {0: "batch", 1: "sensors"},   # ← BOTH axes dynamic
        "mask":       {0: "batch", 1: "sensors"},
        "sensor_ids": {0: "batch", 1: "sensors"},
        "domain_id":  {0: "batch"},
        "rul":        {0: "batch"},
        "attn":       {0: "batch", 3: "sensors"},
    },
    opset_version = 17,
    do_constant_folding = True,
)
```

Marking axis 1 of `x` as `"sensors"` is **the single most important line in your deployment code.** Without it, the exported model is welded to 21 sensors and every claim in your title collapses.

### Verify it — do not assume

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("unify_rul.onnx", providers=["CPUExecutionProvider"])
for S in (2, 14, 16, 21):                      # every sensor count you care about
    out = sess.run(None, {
        "x":          np.random.randn(1, S, 30).astype(np.float32),
        "mask":       np.ones((1, S), dtype=bool),
        "sensor_ids": np.arange(S, dtype=np.int64)[None],
        "domain_id":  np.zeros(1, dtype=np.int64),
    })
    print(S, out[0].shape, out[0].ravel()[0])
# must print a valid scalar for every S. If S=2 throws, dynamic_axes is wrong.
```

Also verify **numerical equivalence** with PyTorch:

```python
torch_out = model(x, mask, ids, dom)[0].detach().numpy()
onnx_out  = sess.run(None, {...})[0]
np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-5)
```

Make both of these a `pytest` test that runs in CI (or at least before every demo). An export that silently changes numerics is a nightmare to debug later, and this test catches it in 2 seconds.

### Common export errors and their causes

| Error / symptom | Cause | Fix |
|---|---|---|
| `Unsupported: ATen operator ...` | An op with no ONNX equivalent at your opset | Raise `opset_version`, or rewrite with basic ops |
| Output shape frozen at the dummy size | `dynamic_axes` missing or misspelled key | Names must match `input_names` exactly |
| Predictions differ from PyTorch | Exported in train mode | `model.eval()` before export |
| `NaN` in output | `-inf` masking with a fully-masked row | Use `-1e4` and assert ≥1 valid sensor |
| Model runs but ignores the mask | Mask compared with `!=0` inconsistently | Keep the mask `bool` end-to-end |
| Huge file | Weights not constant-folded | `do_constant_folding=True` |

## 11.5 ONNX Runtime — the engine

ORT loads a `.onnx`, applies graph optimisations, and executes via an **Execution Provider** (EP) chosen per hardware.

**Graph optimisations ORT performs automatically:**

- **Constant folding** — evaluates subgraphs with constant inputs at load time.
- **Operator fusion** — `Conv + BatchNorm + Relu` becomes one fused kernel. This is a big win: BatchNorm at inference is an affine transform that folds directly into the convolution's weights and bias, so it costs literally zero at runtime.
- **Layout optimisation**, redundant-node elimination, memory-reuse planning.

**Execution providers:**

| EP | Hardware | Relevant to you |
|---|---|---|
| CPUExecutionProvider | Any CPU (uses MLAS, with NEON on ARM) | ✔ your Pi |
| XNNPACK | ARM CPUs, quantised models | ✔ worth benchmarking |
| CUDA / TensorRT | NVIDIA | Laptop only |
| ACL / Arm NN | Arm Compute Library | Optional on Pi, sometimes faster |
| QNN / SNPE | Qualcomm DSP/NPU | Industry context |
| CoreML / NNAPI | Apple / Android | – |

On the Pi you use `CPUExecutionProvider`, which internally dispatches to NEON-vectorised MLAS kernels. That is where your speed comes from.

## 11.6 Deploying to the Pi

### Copy the artefacts

```bash
# from the laptop — exactly three files
scp deployment/unify_rul.onnx   pi@raspberrypi.local:~/unify/
scp deployment/norm_stats.json  pi@raspberrypi.local:~/unify/
scp deployment/sensor_vocab.json pi@raspberrypi.local:~/unify/
```

Nothing else crosses. If you find yourself copying `.py` model definitions, your export is incomplete.

**Record the hash** so you always know which model produced which prediction:

```bash
sha256sum unify_rul.onnx | tee model.sha256
```

### Install and run

```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install numpy onnxruntime fastapi uvicorn
```

```python
import onnxruntime as ort, numpy as np, json

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2          # BENCHMARK this: 1, 2, 3, 4
opts.inter_op_num_threads = 1
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.enable_mem_pattern = True

sess  = ort.InferenceSession("unify_rul.onnx", opts, providers=["CPUExecutionProvider"])
stats = json.load(open("norm_stats.json"))
mu, sd = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)

def predict(window):                   # window: (L, S) most recent first-in-last-out
    x = ((window - mu) / sd).T[None].astype(np.float32)      # (1, S, L)
    mask = np.ones((1, x.shape[1]), dtype=bool)
    rul_norm = sess.run(["rul"], {"x": x, "mask": mask,
                                  "sensor_ids": SENSOR_IDS[None],
                                  "domain_id":  np.zeros(1, np.int64)})[0]
    return float(rul_norm[0, 0]) * stats["life_scale"]
```

### Run it as a service

```ini
# /etc/systemd/system/unify.service
[Unit]
Description=UNIFY-RUL edge inference
After=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/unify
ExecStart=/home/pi/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now unify
journalctl -u unify -f          # live logs
```

`Restart=always` means a crash at 2 a.m. self-heals. This one file is the difference between "a script I run" and "a deployed system", and it is worth saying so in your report.

## 12.1 Measuring memory, CPU and latency

**Do not guess these numbers. Measure them and put a table in your report.**

### Latency benchmark

```python
import time, numpy as np
x = np.random.randn(1, 16, 30).astype(np.float32)
feeds = {"x": x, "mask": np.ones((1,16), bool),
         "sensor_ids": np.arange(16, dtype=np.int64)[None],
         "domain_id":  np.zeros(1, np.int64)}

for _ in range(50): sess.run(None, feeds)          # warm-up: caches, thread pool

t = []
for _ in range(1000):
    s = time.perf_counter(); sess.run(None, feeds); t.append((time.perf_counter()-s)*1e3)
t = np.array(t)
print(f"p50 {np.percentile(t,50):.2f} ms  p95 {np.percentile(t,95):.2f}  "
      f"p99 {np.percentile(t,99):.2f}  max {t.max():.2f}")
```

**Report percentiles, not the mean.** Real-time engineering cares about the tail. Also log `vcgencmd measure_temp` before and after so you can state whether thermal throttling occurred.

### Expected numbers (verify against your own)

| Metric | Expected on Pi 4 |
|---|---|
| Model file (fp32) | 150–350 KB |
| Model file (int8) | 50–100 KB |
| RSS of the Python process | 80–150 MB (mostly numpy + ORT + FastAPI) |
| Peak inference working set | < 5 MB |
| Inference p50, 1 thread | 1–4 ms |
| Inference p50, 2 threads | 1–3 ms |
| CPU during steady state at 1 Hz | < 5 % of one core |
| End-to-end (sensor → dashboard) | 15–60 ms |

If your inference is above ~20 ms, something is wrong: 32-bit OS, a debug build, thread thrashing, or graph optimisations disabled.

### Memory measurement

```bash
ps -o rss= -p $(pgrep -f uvicorn)     # KB
/usr/bin/time -v python bench.py      # peak RSS
free -h                               # system-wide
```

### CPU measurement

```bash
top -p $(pgrep -f uvicorn)
pidstat -p $(pgrep -f uvicorn) 1
vcgencmd measure_temp && vcgencmd get_throttled   # 0x0 means never throttled
```

## 12.2 Optimisation, in the order you should try it

1. **Thread count.** Benchmark `intra_op_num_threads` ∈ {1,2,3,4}. For a model this small, **1 or 2 usually wins** — thread synchronisation costs more than the parallelism gains. Counter-intuitive, easy to demonstrate, and a great result to show in a chart.
2. **Graph optimisation level.** `ORT_ENABLE_ALL` (the default) fuses Conv+BN+ReLU. Verify it is on.
3. **Dynamic int8 quantisation.** Weights → int8, activations quantised at runtime. Two lines:

   ```python
   from onnxruntime.quantization import quantize_dynamic, QuantType
   quantize_dynamic("unify_rul.onnx", "unify_rul_int8.onnx", weight_type=QuantType.QInt8)
   ```

   Expect ~4× smaller file and 1.5–3× faster on MatMul-heavy graphs. **Always re-measure accuracy after quantising** — report RMSE before and after. A "quantisation impact" table (size, latency, RMSE) is exactly the kind of edge-AI result an examiner wants to see and almost no student produces.

   Note: dynamic quantisation mainly accelerates `MatMul`/`Gemm`. Your graph is Conv-dominated, so gains may be modest. **Static quantisation** (with a calibration dataset) quantises convolutions too and gives more — try it if you have time. Being able to explain *why* dynamic quantisation helped less than expected is a better answer than a big number you can't explain.

4. **Reduce the window or model size** if needed. You almost certainly will not need to.
5. **Batch multiple sensor nodes.** If you add more ESP32s, batch their windows into one `sess.run` — near-free throughput.
6. **Pin the process** (`taskset -c 2,3`) and consider `chrt` for scheduling priority if you need tighter tail latency. Advanced; mention it, only do it if the p99 is bad.

### What NOT to do

- Do not try to use the Pi's GPU. There is no practical ONNX EP for VideoCore VI.
- Do not install PyTorch "just to compare". You can compare on the laptop.
- Do not optimise before measuring. Your bottleneck is almost certainly the 1-second feature window on the ESP32, not the 2 ms inference.

## 12.3 The deployment checklist

```
[ ] model.eval() called before export
[ ] dynamic_axes marks BOTH batch and sensors
[ ] onnx.checker.check_model passes
[ ] ORT output matches PyTorch within 1e-4
[ ] S ∈ {2,14,16,21} all run
[ ] norm_stats.json exported and its values match the training scaler
[ ] sensor_vocab.json identical on both machines
[ ] SHA-256 of the model recorded and logged with predictions
[ ] onnxruntime installs from a wheel on the Pi (no source build)
[ ] latency p50/p95/p99 measured and recorded
[ ] memory RSS recorded
[ ] temperature and throttle state recorded
[ ] systemd service enabled, survives reboot
[ ] tested with the network cable pulled
```

---

## Summary

- ONNX is a format + operator spec + type system; ONNX Runtime is the engine that executes it. Two different things.
- Export by tracing: `model.eval()`, no `.item()`, no data-dependent branches, no hard-coded shapes.
- `dynamic_axes` on batch **and** sensors is what preserves the universality claim through export.
- ORT on the Pi uses NEON-vectorised CPU kernels and fuses Conv+BN+ReLU automatically.
- Measure p50/p95/p99, RSS, CPU % and temperature; optimise thread count first, then quantisation.

## Key takeaways

1. Three files go to the Pi: `.onnx`, `norm_stats.json`, `sensor_vocab.json`. Nothing else.
2. `dynamic_axes = {"x": {0:"batch", 1:"sensors"}}` is the most important line in your deployment code.
3. Always verify ONNX numerics against PyTorch — make it a test.
4. Fewer threads is often faster for tiny models. Benchmark; don't assume.
5. Quantisation must be reported with an accuracy number beside it.

## Common interview questions

1. What is ONNX and what problem does it solve? Draw the N×M → N+M diagram.
2. How does tracing-based export differ from scripting, and what breaks under each?
3. What are dynamic axes and when do you need them?
4. What is operator fusion? Give an example. (Conv+BN+ReLU; BN at inference is affine and folds into the conv weights.)
5. Difference between dynamic and static quantisation. Which needs calibration data?
6. Why can int8 be ~4× faster on NEON? (128-bit registers: 4 fp32 lanes vs 16 int8 lanes.)
7. What is an Execution Provider?

## Common viva questions

1. "Show me that your model really accepts different sensor counts." → Run the S ∈ {2,14,16,21} loop live. Have it in a script ready to run; do not fumble.
2. "Why not just run PyTorch on the Pi?" → 800 MB vs 50 MB, slower, ARM builds are painful, and it pins you to one framework. ONNX is the portability contract in your title.
3. "What is your inference latency?" → p50/p95/p99, measured, with the thread count and temperature stated.
4. "Did quantisation hurt accuracy?" → Have the before/after table. If you did not quantise, say so and explain that at 2 ms you had no need — that is a legitimate engineering answer.
5. "How would you deploy to a different device — say a Jetson or a phone?" → Same `.onnx`; swap the Execution Provider (TensorRT, NNAPI, CoreML). That is precisely the point of ONNX and precisely the "Any Hardware" claim.
6. "What if the ONNX output differs from PyTorch?" → It shouldn't; I have an automated equivalence test. Usual causes: train-mode export, opset mismatch, unsupported op silently approximated.

## Practical implementation tips

- Export **inside** the training script, right after loading the best checkpoint, so the model and the stats can never drift apart.
- Version the artefacts: `unify_rul_v3_20260815.onnx`, and log the version with every prediction.
- Keep a `bench.py` on the Pi that prints the whole latency/memory/temperature table in one command. Run it during the demo, live.
- Screenshot the Netron graph for your report.

## Suggested improvements for this project

1. **Publish a quantisation study**: fp32 vs int8 (dynamic) vs int8 (static) — file size, p50/p95 latency, RMSE per dataset. One table, and it is genuine edge-AI research content.
2. **Benchmark on a second device** — an old laptop CPU, a friend's Pi 3, an Android phone via `onnxruntime-mobile`. Two devices from one file makes "Any Hardware" a demonstrated result rather than a slogan. Even one extra device transforms that claim.
3. **Add ORT profiling** (`opts.enable_profiling = True`) and show which operator dominates. A per-operator time breakdown is a strong figure.
4. **Automate the deployment** with a single `make deploy` that exports, checks, hashes, copies and restarts the service. Small, and it shows engineering discipline.
