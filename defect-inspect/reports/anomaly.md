# PatchCore: what it does here, and how well

> **SEPARATION RESTATED 2026-09-19.** The figures below ("median defect 2.40 vs
> median normal 2.26, separation +0.141") come from `inspector.anomaly.demo()`,
> which compares medians over **46 regions per group**. Measured over **534 defect
> and 135 clean regions**, the same statistic is **+0.273** and the threshold-free
> **AUROC is 0.819**.
>
> PatchCore separates considerably better than this project has been assuming.
> "Weak separation" was an artefact of sample size and should not be carried into
> the paper. The calibration in `demo()` should be replaced by an AUROC over the
> largest available sample; see `patchcore-denoise.md`.


Regenerate with:

```bash
uv run python scripts/build_patchcore.py
uv run python -m inspector.anomaly
```

---

## Why the module exists

YOLOv8 can only report the six classes it was trained on. A seventh kind of
damage — a handling dent, an oil smear, a flaw in a new alloy — produces no
detection at all, and the part passes inspection. That is a blind spot no
amount of extra training data closes, because the failure is *definitional*.

PatchCore has no defect classes. It stores what normal steel looks like and
measures distance from it. It therefore cannot have that blind spot — and
cannot name what it finds. The two models are complementary, which is why
`decision.py` fuses them with OR rather than AND.

---

## What was built

| Component | Choice | Why |
|-----------|--------|-----|
| Backbone | ResNet-18, layers 2+3 | ~10x cheaper than the paper's WideResNet-50; mid-level features describe texture, which is all this needs |
| Feature grid | 16x16, 384 channels | stride-8 layer 2 with layer 3 upsampled and concatenated, then 3x3 average pooled |
| Dimensionality | 384 → 128, random projection | Johnson-Lindenstrauss preserves distances; 3x cheaper search |
| Memory bank | 2000 vectors, greedy k-centre coreset | from 63,476 candidate patches |
| Runtime | ONNX Runtime + NumPy | keeps torch off the Pi |
| Bank file | 1.13 MB | |
| Latency | 36.7 ms per call | which is why it is gated |

---

## The dataset problem, and the workaround

**PatchCore is defined to train on defect-free images. NEU-DET has none.**
All 1770 images carry at least one labelled defect. There is no good-parts
folder to point at.

The workaround: build the bank from image regions lying *outside* every
annotated bounding box, with boxes dilated 1.3x first so defect bleed is
excluded. That yielded **63,476 patches from 153,600 — 41.3% of the image
area was usable as "normal"**.

Its limitations, which belong in the viva answer rather than a footnote:

1. **The "normal" patches come from defective images.** Where a defect extends
   past its annotation, defect texture enters the bank and the model learns to
   tolerate it.
2. **Crazing covers most of its image**, so few of its patches survive the mask.
   The bank under-represents that background texture, and crazing is already the
   detector's worst class.
3. **Annotation rectangles bound irregular defects.** The corners of a box are
   normal surface being discarded — harmless, but it means the mask is
   conservative in one direction and leaky in the other.

---

## Measured separation, and the honest verdict

46 defect crops against 46 defect-free crops from the test split:

| region | n | median | p90 | max |
|--------|--:|-------:|----:|----:|
| normal | 46 | 2.261 | 2.726 | 3.482 |
| defect | 46 | 2.402 | 3.240 | 3.949 |

**Median separation: +0.141.** The distributions overlap substantially — the
90th percentile of normal (2.726) sits above the median of defect (2.402).

At the calibrated threshold of 2.77 (95th percentile of normal), this module
catches a minority of defects at a ~5% false-alarm rate on normal surface. **It
is not a second accurate classifier and must not be described as one.**

### Why the separation is weak

Two causes, in order of importance:

1. **A contaminated bank.** Building "normal" from defective images is a
   compromise, and it shows. Every leaked defect patch teaches the model that
   the defect is normal.
2. **Steel mill finish is genuinely high-variance.** Rolling produces
   directional texture that varies between coils. A feature distance that is
   large for a rolled-in scale defect is also large for an ordinary change in
   surface finish, so the signal and the nuisance live in the same range.

### What would fix it

**Photograph 200 known-good parts under the production lighting rig and rebuild
the bank from those.** This is not a modelling change — it is the correct input
that the dataset cannot supply. It needs the conveyor and the LED ring light,
so it is the first thing to redo once hardware is assembled. Expect the
separation to improve substantially, because the two causes above are both
artefacts of the workaround rather than properties of the method.

Secondary options, if a clean bank still separates poorly: a larger bank
(2000 → 8000 vectors), keeping layer 1 features for finer texture detail, or
per-patch score maps to localise the anomaly instead of one frame-level maximum.

---

## Why it runs on ~3 frames per second, not all of them

36.7 ms per call against a ~20 ms detector frame is not affordable per frame.
Two independent gates in `anomaly.maybe_score`:

- **Confidence gate** (`gate_conf_below: 0.55`) — value. When YOLOv8 already
  reports a defect at 0.9, a second opinion changes no decision.
- **Rate gate** (`max_calls_per_second: 3`) — budget. Without it, a run of
  uncertain frames calls PatchCore on every one and the frame rate collapses
  precisely when the line is producing marginal parts, which is the worst
  possible moment to slow down.

The gating is what makes the module affordable. It also means PatchCore samples
the line rather than inspecting every part, and on a fast conveyor an
unclassified anomaly may be seen a few parts late. For a backstop against
unknown defect types that is an acceptable trade; for a primary detector it
would not be.
