# One threshold is doing two incompatible jobs

> **SEPARATION FIGURE SUPERSEDED 2026-09-19.** The "median defect 2.40 vs median
> normal 2.26, separation +0.141" quoted from this project's calibration was
> measured on 46 regions per group. On 669 regions the same statistic reads
> **+0.273 with AUROC 0.819**. PatchCore separates better than this project has
> been assuming, and "weak separation" should not be used as a premise until
> re-measured. See `patchcore-denoise.md`.


Measured 2026-09-18 on the held-out test split (181 images, 438 boxes) with the
v2 320 weights. Reproduce with `scripts/threshold_study.py`.

---

## The observation

`configs/default.yaml` has a single `detector.conf_threshold: 0.35`. It is
applied in `detector.py` as a hard cut on raw model output, before `decide()`
ever sees a box. That one number therefore controls two things at once:

1. **Which boxes are drawn and logged** — a *precision* concern. Too low and
   the dashboard fills with speculative rectangles and the operator stops
   trusting it.
2. **Whether the part is ejected** — a *recall* concern. Too high and defective
   parts ship.

These want opposite settings, and the second is the one that costs money.

## What it costs

Every NEU-DET image contains a defect, so a frame with no detection above the
threshold is an **escape** — a defective part the system passes.

| conf | image-level correct label | escapes | boxes/img | wrong-class/img |
|---|--:|--:|--:|--:|
| **0.35** (deployed) | 0.90 | **8.8%** | 2.19 | 0.13 |
| 0.25 | 0.93 | 4.4% | 2.78 | 0.20 |
| 0.15 | — | 1.7% | 3.93 | 0.33 |
| **0.10** | **0.97** | **0.6%** | 5.03 | 0.42 |
| 0.05 | 0.98 | 0.0% | 7.97 | 0.61 |

Ground truth averages 2.42 boxes per image.

**Lowering the decision threshold from 0.35 to 0.10 cuts escapes from 8.8% to
0.6% — a factor of fifteen — at zero computational cost and with no change to
the model.** The detector already produces these boxes; they are being discarded
before the decision is made.

### The effect is concentrated in the hardest class

| class | correct label @0.35 | @0.10 |
|---|--:|--:|
| **crazing** | **0.57** | **0.93** |
| rolled-in_scale | 0.90 | 1.00 |
| scratches | 0.90 | 0.90 |
| inclusion / patches / pitted_surface | 1.00 | 1.00 |

Crazing has been the weakest class in every measurement in this project and its
sibling, at every resolution and under two training stacks. Box-level recall at
IoU ≥ 0.5 is 0.33. But when the model does place a crazing box it labels it
correctly **100%** of the time, 98% of its predictions on crazing images say
"crazing", and the median best-IoU against ground truth is 0.243.

So the model recognises crazing and fails to *localise* it.

> **The mechanism first proposed here — that the box *partition* is arbitrary —
> was tested and falsified. See `crazing-hypothesis-failed.md`.** Region-level
> scoring does not rescue crazing specifically, and crazing remains the
> worst-localised class even when boxes are unioned. The surviving and
> better-supported framing is the published *things vs stuff* distinction
> (COCO-Stuff, CVPR 2018; Panoptic Segmentation, CVPR 2019): crazing is a
> "stuff" class annotated with a "thing" formulation. Crazing being the worst
> NEU-DET class is also the published norm across 16/17, 8/8 and 5/5 models in
> three separate comparison tables, so it is not a fault of this model.

**For a PASS/FAIL ejector, that grading is irrelevant.** The system needs to
know the part is defective, not where the defect's edges were drawn.

## It replicates on an independent system

The sibling project `defectvision` is a separate codebase with its own model
(YOLOv8n trained at **640**, not 320), its own stratified split (182 test images
from a different seed), and its own training run. Nothing is shared but the
dataset pool. Running the same measurement there:

| project | escapes @0.35 | escapes @0.10 | reduction | crazing @0.35 | crazing @0.10 |
|---|--:|--:|--:|--:|--:|
| defect-inspect | 8.8% | 0.6% | **15×** | 0.57 | 0.93 |
| defectvision | 15.4% | 0.5% | **31×** | 0.27 | 0.97 |

The effect is not a quirk of one training run. It is larger on the second
system, and the crazing collapse is more extreme there — 0.27 at the deployed
threshold against 0.97 at 0.10.

One detail sharpens the point. `defectvision`'s model was trained at 640 and is
evaluated here at 320, the mismatched-but-graceful direction (`resolution-retrain.md`).
Its confidences are therefore systematically lower, which is why the *same*
0.35 threshold leaks 15.4% of parts instead of 8.8%.

**A confidence threshold is not a transferable constant.** 0.35 encodes a
property of one model's score distribution, yet it is written in the config as
if it were a property of the inspection task. Two projects built to the same
brief inherited the same number and it means something different in each. That
is the deeper version of the finding: the constant is not merely doing two jobs,
it is not well-defined across models at all.

## Consequence for the anomaly backstop

PatchCore runs when detector confidence falls below 0.55 and costs ~110 ms per
call — by `scripts/profile_stages.py`, the most throughput-sensitive parameter
in the configuration. 27.1% of test frames (49/181) enter that gate.

On exactly those frames:

| | correct label |
|---|--:|
| detector @ 0.35 (deployed) | 0.67 |
| detector @ 0.25 | 0.82 |
| **detector @ 0.10** | **0.96** |
| detector @ 0.05 | 0.98 |

PatchCore's own measured separation on this dataset is weak — median defect
score 2.40 against median normal 2.26 (`anomaly.md`).

**The detector, at a lower threshold, recovers 96% of the frames PatchCore was
built to rescue, for 0 ms instead of 110 ms, and with better separation.** For
in-distribution defects the backstop is redundant.

This does **not** refute PatchCore's stated purpose. It was built for the
*seventh* defect type — damage no class covers — and every image here belongs to
one of the six trained classes. That argument stands and is untested by this
experiment. What changes is the justification: PatchCore should be reserved for
out-of-distribution damage, not used as a general safety net for frames the
detector is merely quiet about.

## The proposed change

Split the one threshold into two, by role:

```yaml
detector:
  conf_threshold: 0.35      # what gets drawn and logged  (precision)
decision:
  eject_threshold: 0.10     # what triggers the ejector   (recall)
```

`detector.py` filters at the lower of the two, so the decision sees the evidence;
`decide()` ejects on `eject_threshold`; the dashboard and the stored boxes render
at `conf_threshold`. Nothing else changes: no retraining, no extra inference, no
new model.

This is a small change and the reason it is worth writing up is that the failure
is invisible. Nothing errors. The dashboard looks clean and plausible precisely
*because* the low-confidence evidence was thrown away, and the reject rate it
reports is the rate after discarding it.

## Limits of this result — read before acting

1. **The false-alarm cost is not measured, and cannot be measured on this
   dataset.** NEU-DET contains no defect-free images. Every number above is
   conditioned on the part being defective, so "escapes" is a true recall figure
   while the rate at which a *good* part would be wrongly ejected at 0.10 is
   unknown. The wrong-class column (0.13 → 0.42 boxes/image) is a proxy for
   false-positive propensity, not a substitute.

   This is the single thing to resolve before deploying 0.10. It needs perhaps
   50 images of defect-free steel under the production lighting rig — an
   afternoon's work once a camera exists, and the same images would fix
   PatchCore's weak separation, which stems from building its "normal" bank out
   of defect-free *regions* of defective images.

2. One model, one test split, 181 images. The escape figures at 0.05 and 0.01
   (0.0%) sit on a handful of frames and should not be read as exact.

3. `min_defect_area_px` already exists to catch small high-confidence false
   positives, and it is unchanged here. Its interaction with a lower threshold
   is untested: more low-confidence boxes may be small ones, in which case the
   area filter absorbs part of the effect.

## Why this is the interesting result

Most of the measurement in this project improved a number by a few points —
quantisation, resolution, retraining. All of those were bounded by the model.

This one is not about the model at all. The information needed to catch 15×
more escapes was already being computed on every frame and discarded by a
configuration default, and no metric in the system reported its loss: mAP is
computed at a 0.001 sweep and never sees the operating threshold, while the
dashboard reports rates *after* the cut. A system can be instrumented,
benchmarked, parity-checked and still lose most of its recall to a constant in
a YAML file.
