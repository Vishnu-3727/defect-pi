# `val(imgsz=320)` does not evaluate at 320

Root cause of the PyTorch↔ONNX gap flagged as "indicated, not demonstrated" in
`quantisation-corrected.md`. It is now demonstrated, and it is not a runtime
difference.

---

## The measurement

All three calls pass `imgsz=320`, same weights, same split, batch=1. The
preprocessing hook records what the network is actually fed:

| config | input actually fed | mAP@50 | mAP@50-95 |
|---|---|--:|--:|
| `.pt`, `rect=True` (**the default**) | **352×352** | **0.7624** | 0.4373 |
| `.pt`, `rect=False` | 320×320 | 0.7411 | 0.4063 |
| `.onnx` | 320×320 | **0.7411** | **0.4063** |

**PyTorch at 320 and ONNX at 320 agree exactly.** The runtimes are identical to
four decimals. The entire discrepancy was resolution.

## Why it happens

Ultralytics' validator defaults to `rect=True`, rectangular batching: it derives
a batch shape from the data and pads to a multiple of the model stride (32),
using `imgsz` as a lower bound rather than the target. NEU-DET's 200×200 images
land on **352**, one stride step above 320.

An exported ONNX graph has a **fixed input shape**, so it cannot be rect-batched
and genuinely runs at the requested 320. The comparison "PyTorch vs ONNX" is
therefore "352 vs 320" wearing a disguise, and it is the *only* thing that ever
distinguished them.

The preprocessed tensors are byte-identical in the `predict` path — checked
first, which is why the search moved to `val`.

## The bias is model-dependent, and it reverses rankings

Measured per checkpoint, the gap between the default and the true 320 has no
consistent sign:

| model | fed 352 (default) | fed 320 (deployed) | delta |
|---|--:|--:|--:|
| v2 (deployed) | 0.7624 | 0.7411 | **+0.0213** |
| v1 (archived) | 0.7107 | 0.7354 | **−0.0248** |
| kaggle t640 | 0.7277 | 0.7121 | **+0.0155** |

So this is not a constant offset that cancels in a comparison. It is a
per-model perturbation of up to ±0.025, and it is large enough to reorder:

| | ranking |
|---|---|
| default evaluation (352) | v2 > **t640** > **v1** |
| true deployed resolution (320) | v2 > **v1** > **t640** |

**v1 and t640 swap places.** At 352, `t640` appears better by 0.0170; at 320,
`v1` is better by 0.0233. A default setting nobody typed decides which of two
models you would ship.

That is the strongest form of the finding: not a mis-stated number, but model
selection pointed at the wrong checkpoint.

## What this invalidates

**Every PyTorch mAP figure in this project was measured at 352, not 320.**

- `reports/metrics.md` reports **0.7624** as the accuracy of the 320-px model.
  The deployed ONNX model at 320 scores **0.7411**. The headline number is from
  a resolution the system never runs.
- The same applies to the v1 figure of 0.7106/0.7107 and to `defectvision`'s
  0.744 and 0.726.
- It fully explains the "INT8 sign flip": v1 PyTorch 0.7107 (at 352) against v1
  INT8 ONNX 0.7324 (at 320) was a resolution comparison. At matched resolution
  v1 is 0.7354 → 0.7341, a −0.0013 quantisation cost like every other model.

**What it does not invalidate:** any comparison where both sides had the same
treatment. The ten-model quantisation curve was ONNX→ONNX throughout, so
`quantisation-corrected.md` stands. The resolution retrain compared PyTorch
against PyTorch, so its *ranking* holds — but its axis labels are wrong, since
the "320" and "640" runs were both rect-padded.

## Why this is worth publishing

It is a silent, default-on trap with a plausible-looking output. Nothing errors.
`imgsz=320` appears in the call, in the log line and in the generated report.
The number is simply not the number for the resolution named, and the error is
in the **optimistic** direction here (+0.021 mAP@50), so nothing looks wrong.

It bites precisely when comparing a deployed export against its source
checkpoint — the single most common reason to run both. A practitioner
concluding "our ONNX export costs 2 points of mAP" would be wrong, and would
likely go looking for a bug in the export.

The project's own `scripts/parity.py` could not catch it: it compares boxes at
conf 0.35 and reports 121/121 at IoU 1.0000, because at a fixed operating
threshold the two agree closely. The divergence lives in the low-confidence tail
that a swept metric integrates over. That is the same gap recorded in
`quantisation-corrected.md`, now with a demonstrated mechanism behind it.

## The parity gap, now quantified

`parity.py` gained a second pass at conf 0.001, the floor mAP sweeps from:

| model | conf | ref | ours | matched | recall | mean IoU | cls err |
|---|--:|--:|--:|--:|--:|--:|--:|
| `model.onnx` | 0.35 | 121 | 121 | 121 | 1.000 | 1.0000 | 0 |
| `model_int8.onnx` | 0.35 | 121 | 117 | 107 | 0.884 | 0.9518 | 0 |
| `model.onnx` | 0.001 | 2153 | 2153 | 2153 | **1.000** | 1.0000 | 0 |
| `model_int8.onnx` | 0.001 | 2153 | 2051 | 1472 | **0.684** | 0.8904 | **7** |

The FP32 export is exact at both thresholds — 2153 of 2153 boxes at IoU 1.0000 —
which is a stronger result than the original check established, and it is why
the PyTorch/ONNX gap had to be resolution rather than export.

INT8 is the opposite: agreement looks good at the operating point (0.884) and
**falls to 0.684 in the tail**, where seven class errors also appear that the
0.35 pass reports as zero.

So the original 121/121 at IoU 1.0000 was true and did not mean what it was used
to mean. It licensed carrying an accuracy number across the export boundary; the
number is swept from 0.001, and at 0.001 barely two thirds of the INT8 boxes
agree.

## What to do

1. Re-measure and re-report every PyTorch figure with `rect=False`, or state the
   evaluated resolution beside each number.
2. Quote **0.7411** as the deployed accuracy of this system, not 0.7624.
3. In `scripts/evaluate.py`, pass `rect=False` and print the input shape the
   validator actually used, so the report cannot disagree with its own title.
4. Extend `parity.py` to compare at the metric's confidence floor (0.001) as
   well as at the operating threshold.

## Prior art, checked

**The mechanism is in the library source, and I verified it directly.**
`ultralytics/engine/validator.py` (8.4.155, line 223):

```python
if not (pt or (getattr(model, "dynamic", False) and fmt != "imx")):
    self.args.rect = False
```

Ultralytics *knows* a static export cannot be rect-batched, so it silently
disables the setting for non-PyTorch models. The asymmetry is deliberate. What
is undocumented is its consequence: the same `val(imgsz=N)` call measures a
`.pt` and its own export at **different input sizes**, with no warning.

A literature and issue sweep (raw output in `../paper/lit/rect-trap-sweep.md`)
found:

- **The mechanism is discussed in scattered GitHub issues**, including one
  reporting a PyTorch-vs-ONNX mAP gap restored by `rect=False`. So the
  behaviour is known to some users. It is **not** in any paper, and nowhere is
  it quantified.
- **Not found:** any paper attributing a PyTorch-vs-export accuracy gap to this
  artefact. Several quantisation papers and benchmark posts compare INT8/ONNX/
  TensorRT against a PyTorch baseline **without stating letterbox or rect
  handling**, so their reported "export cost" may be partly this ~0.02 bias.
  The sweep names specific candidates; each would need checking before the claim
  is made about any individual paper.
- **Not found:** any paper showing `rect`/`imgsz` reversing a model ranking.

### Where this sits in the evaluation-protocol literature

There is an established line showing that protocol details change reported
conclusions — this result belongs in it rather than standing alone:

- **Dave, Dollár, Ramanan, Kirillov, Girshick, *Evaluating Large-Vocabulary
  Object Detectors: The Devil is in the Details*, arXiv:2102.01066, 2021.**
  *(Verified directly.)* The default AP implementation "produces a gameable
  metric, where a simple, un-intuitive re-ranking policy can improve AP by a
  large margin", and under a corrected implementation "many reported gains do
  not translate to improvements". The closest precedent in spirit: a default
  nobody chose, silently deciding what a benchmark reports.
- COCO-ReM (ECCV 2024) and Llerena & Jung (WACV 2025) are reported by the sweep
  as showing annotation-quality and noise-aware corrections that shift rankings.
  *Not verified here — check before citing.*

**Positioning:** that line concerns the *metric*. This concerns the *input*:
the model under test never saw the resolution the report names, and the library
chose that for you based on the file format. Same failure class, different
stage of the pipeline, and it discriminates specifically against exported
models — the ones actually deployed.

## Limits

- Ultralytics 8.4.155. The default and the stride arithmetic may differ by
  version; the behaviour should be re-checked rather than assumed.
- The 352 figure is specific to 200×200 inputs at `imgsz=320`. Other input
  aspect ratios and sizes land elsewhere.
