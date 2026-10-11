# Prevalence, measured on a dataset that has real defect-free images

`overkill-and-prevalence.md` argued that a benchmark where **every image contains
a defect** biases operating-point selection toward recall, because the overkill
term is multiplied by (1 − p) = 0. Its overkill rates were estimated from
defect-free *regions*, an upper bound.

**KolektorSDD2** removes the proxy: 894 genuinely normal against 110 defective
test images, a natural prevalence of **11.0%**. Both error rates are measured
directly. Two arms differ only in training composition, and both are judged on
the same test set including the normals.

Run on Kaggle, `paper/kaggle/ksdd2/ksdd2_prevalence.ipynb`. YOLOv8n, 640 px,
60 epochs, seed 42, single class, `rect=False` throughout.

---

## Both error rates, directly measured

| thr | escape (defect-only) | escape (with-normals) | overkill (defect-only) | overkill (with-normals) |
|---|--:|--:|--:|--:|
| 0.50 | 0.282 | 0.382 | 0.006 | 0.001 |
| 0.35 | 0.164 | 0.236 | 0.017 | 0.004 |
| 0.25 | 0.118 | 0.173 | 0.043 | 0.007 |
| 0.15 | 0.082 | 0.145 | 0.111 | 0.020 |
| 0.10 | 0.064 | 0.118 | 0.195 | 0.035 |
| 0.05 | 0.036 | 0.100 | 0.322 | 0.067 |
| 0.01 | 0.009 | 0.055 | 0.706 | 0.230 |

**The overkill cost of a low threshold is real and severe.** At 0.01 the
defect-only model rejects **70.6% of good parts** to reach a 0.9% escape rate.
This is the number NEU-DET structurally cannot show, and it is measured here,
not estimated.

## Result 1: prevalence alone moves the recommendation by 25–35×

Cost-optimal threshold, by prevalence and cost ratio R = C_escape / C_overkill:

| prevalence | R=1 | R=10 | R=100 | R=1000 |
|---|--:|--:|--:|--:|
| 0.010 | 0.50 | 0.35 | 0.25 | 0.05 |
| 0.050 | 0.50 | 0.35 | 0.05 | 0.01 |
| **0.110 (this dataset)** | **0.35** | 0.25 | 0.05 | 0.01 |
| 0.500 | 0.25 | 0.05 | 0.01 | 0.01 |
| **1.000 (the NEU-DET regime)** | **0.01** | **0.01** | **0.01** | **0.01** |

*(defect-only arm; the with-normals arm shows the same collapse.)*

**At prevalence 1.00 the optimal threshold is 0.01 for every cost ratio,
including R = 1 where escapes and false rejects cost the same.** The overkill
term vanishes, so nothing opposes lowering the threshold and the recommendation
bottoms out regardless of economics.

At this dataset's real 11%, the same model wants **0.35** at R = 1 — a factor
of 35 higher.

That is the artefact, reproduced on a dataset that does not suffer from it, by
varying only the prevalence parameter. It is no longer an argument about
NEU-DET; it is a measured property of the cost function.

## Result 2: the bias is partly baked in at training time

Overkill at **matched escape rate**, interpolated on each arm's curve:

| escape | defect-only | with-normals | ratio |
|---|--:|--:|--:|
| 0.05 | 0.258 | 0.230 | 1.1× |
| 0.08 | 0.119 | 0.139 | **0.9×** |
| 0.10 | 0.077 | 0.067 | 1.1× |
| 0.15 | 0.024 | 0.018 | 1.4× |
| 0.20 | 0.013 | 0.006 | **2.3×** |
| 0.25 | 0.009 | 0.004 | 2.1× |

Area under the escape-overkill curve: **defect-only 0.0293, with-normals
0.0096** — a **3× better operating characteristic** from adding background
images alone, with no change to architecture, schedule or seed.

**But state the nuance rather than the headline.** The curves **cross** near
escape 0.08, where the defect-only arm is marginally better (0.9×). The
advantage is concentrated in the low-overkill region — escape 0.15–0.25, where
with-normals rejects half to a third as many good parts — which is where a real
line with 11% prevalence would operate, but it is not a uniform improvement and
should not be reported as one.

So training on defect-only data does not simply produce a worse model. It
produces one whose operating characteristic is worse **in the region that
matters for a low-prevalence line**, and indistinguishable elsewhere.

## What this closes

- The upper-bound caveat on `overkill-and-prevalence.md`: overkill is now
  measured on real negatives, and the direction of that report's conclusion is
  confirmed on a second dataset and a different defect morphology (median defect
  covers 1.65% of the image here against 39% on NEU-DET).
- The claim that the prevalence effect is a property of the cost function rather
  than of NEU-DET: demonstrated by sweeping p on data that has real negatives.

## Limits

- 110 defective test images. Escape rates move ~0.9 points per image, so the
  fine structure of the escape column is noisy; the overkill column rests on
  894 images and is firmer.
- One architecture, one dataset, 60 epochs, one seed per arm. The 3× AUC gap is
  a single comparison, not a distribution — a second seed per arm would say
  whether it survives training variance, and has not been run.
- Both arms trained at 640 on 631×228 images. KSDD2's aspect ratio is far from
  square, so the rect trap would bite differently here; `rect=False` was forced
  throughout, but the training-time letterboxing was not audited.
