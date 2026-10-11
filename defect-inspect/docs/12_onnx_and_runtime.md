# 12 — ONNX and the Runtime

**Prerequisites:** [07 — PyTorch](07_pytorch.md), [08 — YOLO](08_yolo_v1_to_v8.md).

---

## 1. Concept

ONNX — Open Neural Network Exchange — is a file format that stores a trained
network as a **graph of operations plus a table of weights**, in a way that does
not depend on the framework that produced it.

ONNX Runtime is the engine that executes such a graph. Together they let a model
trained in PyTorch run somewhere PyTorch is not installed.

---

## 2. Why it is needed

A `.pt` file is a PyTorch object. Loading it requires PyTorch, which on ARM is a
2.6 GB install carrying an entire automatic differentiation engine, an optimiser
library and CUDA bindings — for a device that will never compute a gradient.

| | PyTorch on the Pi | ONNX Runtime on the Pi |
|---|---|---|
| Install size | ~2.6 GB | ~50 MB |
| Contains | training + inference | inference only |
| Cold start | seconds | fast |
| Graph optimisation | limited | constant folding, operator fusion |
| Our INT8 support | awkward | first class |

The size difference is the visible argument. The real one is that **inference and
training are different problems**, and a runtime built only for inference can
optimise in ways a training framework cannot — it knows the weights will never
change, so it can fold constants, fuse operators, and pick specialised kernels
once at load time.

---

## 3. Real-world example

The same separation exists throughout engineering. You compile C on a
workstation and flash the binary to a microcontroller; the microcontroller does
not carry GCC. Export is the compile step, ONNX is the binary, and ONNX Runtime
is the loader.

The interoperability matters commercially too. A model exported to ONNX can run
on ONNX Runtime, TensorRT, OpenVINO, CoreML, or a vendor NPU SDK — without
retraining. If this project later moves to a Hailo accelerator or a Jetson, the
ONNX file is the portable asset. The `.pt` is not.

---

## 4. How it works

### What is inside the file

```
model_int8.onnx
├── graph
│   ├── inputs      images  float32 [1, 3, 320, 320]
│   ├── nodes       Conv -> SiLU -> Conv -> Concat -> ... (231 of them)
│   └── outputs     output0 float32 [1, 10, 2100]
├── initializers    the weight tensors
└── opset           13
```

The graph is a directed acyclic graph. Each node names an operator (`Conv`,
`Sigmoid`, `Concat`), its inputs, and its outputs. Nothing is Python; nothing is
executable code. That is the point — no arbitrary code is deserialised, unlike a
pickle-based checkpoint.

### Opset versions, and the one that bit us

The **opset** is the version of the operator specification. Operators gain
attributes over time, and an older opset simply does not have them.

We exported at opset 12 and the INT8 model refused to load:

```
Error Unrecognized attribute: axis for operator DequantizeLinear
```

**Cause:** per-channel quantisation stores one scale factor per output channel,
which requires `DequantizeLinear` to carry an `axis` attribute saying which
dimension the scales index. That attribute was added in **opset 13**. At opset
12 the exporter emitted it anyway and the runtime rejected it.

The fix is one line, and it is now a documented floor rather than a preference:

```python
# opset 13 is a floor, not a preference: per-channel quantisation emits
# DequantizeLinear with an `axis` attribute, which does not exist before 13.
ap.add_argument("--opset", type=int, default=13)
```

This is the general shape of opset problems: the export succeeds, the file
exists, and it fails at load. Export and load are separate steps and must both
be tested.

### The export path

```
best.pt  ──torch.onnx.export──►  best.onnx  ──onnxslim──►  model.onnx
                                                               │
                                                    quant_pre_process
                                                               │
                                                        quantize_static
                                                               ▼
                                                       model_int8.onnx
```

Two intermediate steps that are easy to skip and expensive to skip:

**`simplify=True` / onnxslim** folds constants and removes training-only nodes.
Beyond shrinking the graph, it removes operators the quantiser cannot handle,
so skipping it produces a model that quantises partially and is barely faster.

**`quant_pre_process`** runs shape inference and graph cleanup before
quantisation. Skipping it is the usual cause of "node not quantised" warnings.

### Static shapes

We export with `dynamic=False`, pinning the input to `[1, 3, 320, 320]`.

A dynamic model accepts any resolution and any batch size. That flexibility
costs speed: the runtime cannot pre-plan memory allocation or select
shape-specialised kernels at load time, so it re-plans per call.

Our camera resolution never changes and our batch is always one. Fixing the
shape lets ONNX Runtime do all of its planning once. `detector.py` checks the
match at startup rather than discovering it mid-demo:

```python
if isinstance(expected[2], int) and expected[2] != imgsz:
    raise ValueError(
        f"model expects input {expected[2]}x{expected[3]} but imgsz={imgsz}. "
        f"Re-export with --imgsz {expected[2]} or fix configs/default.yaml."
    )
```

There is a second check on the output shape, because a model trained for a
different number of classes loads perfectly and produces silently wrong results:

```python
n_out = self.session.get_outputs()[0].shape[1]
if isinstance(n_out, int) and n_out != 4 + len(self.class_names):
    raise ValueError(f"model outputs {n_out} rows, which implies {n_out - 4} "
                     f"classes, but config lists {len(self.class_names)}")
```

### Execution providers

ONNX Runtime dispatches operators to an **execution provider** — a backend for
particular hardware.

| Provider | Hardware | Relevance here |
|----------|----------|----------------|
| `CPUExecutionProvider` | any CPU | **what we use**, on x86 and on the Pi |
| `CUDAExecutionProvider` | NVIDIA GPU | not on a Pi |
| `OpenVINOExecutionProvider` | Intel | not applicable |
| `NNAPIExecutionProvider` | Android | not applicable |
| Vendor NPU providers | Hailo, Coral, RK3588 | the upgrade path if the Pi is too slow |

Our session configuration, and why each line:

```python
opts.intra_op_num_threads = num_threads   # 4, one per Cortex-A76 core
opts.inter_op_num_threads = 1             # our graph is a chain; no parallel branches
opts.graph_optimization_level = ORT_ENABLE_ALL
opts.execution_mode = ORT_SEQUENTIAL
```

`intra_op` splits a single operator across threads — that is where the
parallelism in a convolution lives. `inter_op` runs independent operators
concurrently, which a mostly-linear graph like ours cannot exploit, so setting
it above 1 adds scheduling overhead for nothing.

### Graph optimisation, concretely

`ORT_ENABLE_ALL` applies transformations at load time. The most valuable is
**operator fusion**:

```
 before:  Conv ──► BatchNorm ──► SiLU        three passes over the data
 after:   FusedConv                          one pass, one kernel
```

Batch normalisation at inference is a fixed affine transform, so it can be
folded into the preceding convolution's weights and bias. The result is
arithmetically identical and strictly faster — fewer kernel launches and, more
importantly on a memory-bound CPU, fewer round trips through cache.

This is also why ultralytics reports "Model summary (fused): 73 layers" — some
of the fusion happens before export as well.

---

## 5. Mathematical intuition

**Fusing batch norm into convolution.** Batch norm at inference computes:

$$ y = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta $$

with $\mu, \sigma, \gamma, \beta$ all constant. Substituting the convolution
$x = W * i + b$ gives:

$$ y = \underbrace{\frac{\gamma W}{\sqrt{\sigma^2+\epsilon}}}_{W'} * i \;+\; \underbrace{\frac{\gamma (b - \mu)}{\sqrt{\sigma^2+\epsilon}} + \beta}_{b'} $$

which is a convolution with modified weights $W'$ and bias $b'$. Two operations
become one, exactly, with no approximation.

**Constant folding**: any subgraph whose inputs are all constants is evaluated
once at load time and replaced by its result. Training graphs are full of these;
inference never recomputes them.

---

## 6. Software implementation

Export, measured:

```
$ uv run python scripts/export_onnx.py --int8

FP32 ONNX: models/model.onnx
keeping 66 detection-head nodes in FP32 (prefix /model.22/)
calibration set: 198 images across 6 classes
INT8 ONNX: models/model_int8.onnx

artefact                            size
----------------------------------------
best.pt                           6.2 MB
model.onnx                       12.1 MB
model_int8.onnx                   5.6 MB
```

The runtime side is `src/inspector/detector.py`, and the whole file imports
`onnxruntime`, `numpy` and `cv2` — nothing else. That constraint is enforced by
a test rather than by discipline:

```python
def test_runtime_has_no_training_dependency():
    banned = {"torch", "ultralytics", "torchvision"}
    ...
    assert f"import {word}" not in source, \
        f"{name} imports {word}; the Pi runtime must not need it"
```

**Verification that the export is faithful** is `scripts/parity.py`, and the
result is the strongest evidence in this project:

| model | ref boxes | our boxes | matched | mean IoU | class errors |
|-------|----------:|----------:|--------:|---------:|-------------:|
| model.onnx (FP32) | 132 | 132 | 132 | **1.0000** | 0 |

An exact match against ultralytics' own predictions on all 132 boxes. Export,
letterboxing, tensor layout, decode, NMS and inverse coordinate mapping are
jointly correct — and none of those would have raised an error if they were not.

---

## 7. Hardware implementation

Measured with `python -m inspector.bench --compare`. **The absolute figures are
from an x86 host with AVX-512 and do not predict the Pi** — see
`reports/benchmark.md`. The ratios do transfer:

| model | threads | inference ms | FPS | size |
|-------|--------:|-------------:|----:|-----:|
| FP32 | 1 | 38.22 | 24.3 | 12.1 MB |
| FP32 | 4 | 20.81 | 41.8 | 12.1 MB |
| INT8 | 1 | 27.86 | 32.4 | 5.6 MB |
| INT8 | 4 | 16.49 | 50.6 | 5.6 MB |

**INT8 is 1.21x FP32. Four threads are 1.57x one thread.**

Threading scales sub-linearly — 1.57x from 4 cores, not 4x — because
convolutions on small feature maps are partly memory-bound, and the head runs on
tensors too small to split usefully. This is the single most useful number to
have before tuning the Pi: adding cores will not rescue a model that is too big.

The deployment install on the Pi is:

```bash
uv sync --extra pi     # runtime + lgpio. No torch, no ultralytics.
```

Every runtime dependency — `onnxruntime`, `opencv-python`, `numpy`, `flask`,
`pyserial`, `lgpio` — publishes an `aarch64` wheel, so installation is a
download rather than an hour-long compile. Checking wheel availability before
choosing a dependency is a deployment skill, and one missing wheel would change
the plan.

---

## 8. Common mistakes

- **Exporting at too low an opset.** Ours failed at 12 and needed 13. The export
  succeeds and the load fails.
- **Skipping `simplify` or `quant_pre_process`.** Produces a graph that
  quantises partially and is barely faster.
- **Assuming the ONNX model is identical to the `.pt`.** Verify with a parity
  check. Ours matched exactly — but only because we looked.
- **Using dynamic shapes without needing them.** Costs load-time planning for
  flexibility you do not use.
- **Setting `inter_op_num_threads` above 1 on a linear graph.** Scheduling
  overhead, no gain.
- **Installing `onnxruntime-gpu` on a Pi.** There is no GPU provider for it.
- **Forgetting that preprocessing must match training exactly.** ONNX stores the
  graph, not the letterbox or the normalisation. Those are your code's
  responsibility, and getting them wrong fails silently.

---

## 9. Viva questions

**Q: What is ONNX and why did you export to it?**
An open format storing a trained model as a graph of operators plus weights,
independent of the training framework. We export so the Raspberry Pi can run
inference through ONNX Runtime — about 50 MB — instead of PyTorch's 2.6 GB. The
Pi never computes a gradient, so it never needs a training framework.

**Q: What is an opset and did it cause you any trouble?**
The version of the ONNX operator specification. Yes: we exported at opset 12 and
the INT8 model failed to load with "Unrecognized attribute: axis for operator
DequantizeLinear". Per-channel quantisation needs that attribute, which was
introduced in opset 13. The export had succeeded, so the failure appeared only
at load — export and load are separate steps and both need testing.

**Q: What does ONNX Runtime do that PyTorch does not?**
It knows the weights are fixed, so at load time it folds constants and fuses
operators — for instance merging batch normalisation into the preceding
convolution, which is exact and eliminates a whole pass over the data. A
training framework must keep those operations separate because they participate
in gradients.

**Q: What are execution providers?**
Hardware-specific backends that ONNX Runtime dispatches operators to. We use
`CPUExecutionProvider` on both the laptop and the Pi. The same ONNX file would
run on CUDA, OpenVINO, or a vendor NPU without retraining, which is the upgrade
path if the Pi proves too slow.

**Q: How do you know the exported model behaves like the original?**
`scripts/parity.py` runs ultralytics on `best.pt` and our ONNX pipeline on the
same 60 images and matches the boxes. The FP32 export matched all 132 reference
boxes at mean IoU 1.0000 with zero class errors, which verifies export and our
hand-written decode together.

**Q: Why fix the input shape instead of allowing dynamic shapes?**
Our camera resolution and batch size never change. A static shape lets ONNX
Runtime plan memory and select kernels once at load time instead of per call.
`detector.py` validates the configured `imgsz` against the model's declared
input at startup so a mismatch is an immediate clear error.

---

## 10. Summary

1. ONNX stores a model as a graph plus weights; ONNX Runtime executes it without
   the training framework.
2. The Pi runs 50 MB of runtime instead of 2.6 GB of PyTorch, because inference
   never needs gradients.
3. Opset 13 is a hard floor for per-channel quantisation — ours failed at 12,
   at load time rather than export time.
4. Load-time graph optimisation fuses batch norm into convolution exactly, and
   fixed input shapes let the runtime plan once.
5. Measured: INT8 is 1.21x FP32, four threads 1.57x one thread, and the FP32
   export matches ultralytics box for box.

**Next:** [13 — CPU optimisation for the Pi 5](13_cpu_optimization_pi5.md)
