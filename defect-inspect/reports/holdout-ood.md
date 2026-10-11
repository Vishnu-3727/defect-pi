# The blind spot is real, and lowering the threshold does not close it

> **SEPARATION FIGURE SUPERSEDED 2026-09-19.** The "median defect 2.40 vs median
> normal 2.26, separation +0.141" quoted from this project's calibration was
> measured on 46 regions per group. On 669 regions the same statistic reads
> **+0.273 with AUROC 0.819**. PatchCore separates better than this project has
> been assuming, and "weak separation" should not be used as a premise until
> re-measured. See `patchcore-denoise.md`.


Run on Kaggle 2026-09-18, `paper/kaggle/holdout_ood.ipynb`, kernel
`vishnu3727/defect-inspect-holdout-ood`. Six folds after DRA's hard setting
(Ding, Pang, Shen, CVPR 2022): hold one class out of training entirely, test on
images carrying that class and nothing else.

Hold-out is by **label content**, not filename — 121 of 1770 NEU-DET images
carry boxes of more than one class, so a filename hold-out for `scratches` would
have left 49 scratches boxes in training via images named `inclusion_*`. OOD test
sets are images whose label set is exactly the held-out class: **24–30 images per
fold**, which is small and bounds everything below.

---

## Result 1: the detector goes quiet on defect types it has not seen

Fraction of pure-OOD images producing **any** detection. For a PASS/FAIL ejector
a wrong label is still a correct ejection, so this is the operational number —
only silence is a failure.

| held out | @0.35 (deployed) | @0.25 | @0.10 | @0.05 |
|---|--:|--:|--:|--:|
| crazing | **0.069** | 0.069 | 0.207 | 0.345 |
| pitted_surface | 0.148 | 0.185 | 0.333 | 0.444 |
| rolled-in_scale | 0.167 | 0.300 | 0.600 | 0.867 |
| inclusion | 0.233 | 0.300 | 0.667 | 0.800 |
| patches | 0.296 | 0.481 | 0.926 | 1.000 |
| scratches | 0.625 | 0.750 | 0.917 | 0.958 |
| **mean** | **0.256** | 0.347 | 0.608 | 0.736 |
| *(for contrast: seen classes)* | *0.910* | *0.961* | *0.997* | *0.998* |

**At the deployed threshold, 74% of parts carrying an unseen defect type pass
silently.** On crazing, 93% do.

## Result 2: this refutes our own extrapolation

`threshold-decoupling.md` showed that lowering the threshold to 0.10 recovers
96% of the frames PatchCore was gated to rescue, and concluded the backstop was
redundant. It then noted — correctly — that every test defect was in-distribution
and that the claim could not extend to unseen types without this experiment.

It does not extend. At 0.10 the unseen-defect escape rate is still **39%**, and
for crazing and pitted_surface it is 79% and 67%. The threshold is a lever on
*familiar* defects; it does almost nothing for unfamiliar ones, which is what
the two curves above show when read side by side — seen-class firing saturates
at 0.997 while unseen-class firing is still at 0.608.

**PatchCore's rationale stands. Do not remove it.** Our earlier claim narrows to
exactly what was measured: the backstop is redundant *for defects the detector
was trained on*.

This also reproduces DRA's finding on our own data and hardware: naive
supervised training has a real blind spot for unseen anomaly classes.

## Result 3: the gate is well-placed; the backstop behind it is the weak part

AUROC separating seen from unseen at image level (higher = better, 0.5 = none):

| held out | MSP | max-logit | energy | n OOD |
|---|--:|--:|--:|--:|
| crazing | 0.993 | 0.993 | 0.993 | 29 |
| pitted_surface | 0.970 | 0.970 | 0.962 | 27 |
| rolled-in_scale | 0.963 | 0.963 | 0.955 | 30 |
| inclusion | 0.930 | 0.930 | 0.931 | 30 |
| patches | 0.900 | 0.900 | 0.874 | 27 |
| scratches | 0.763 | 0.763 | 0.743 | 24 |

Confidence separates seen from unseen **well** — 0.76 to 0.99. So the existing
design is right: `anomaly.gate_conf_below: 0.55` routes unfamiliar frames to the
backstop, and it routes them accurately. Crazing, the class the detector is most
blind to, is also the one it is most reliably *uncertain* about (0.993).

The architecture is not the problem. What sits behind the gate is: PatchCore's
measured separation on this dataset is median defect 2.40 against median normal
2.26 (`anomaly.md`). **The routing works and the destination is weak** — which
relocates the open problem from "should the backstop exist" to "make the
backstop actually separate", and the known cause of the latter is that its
memory bank is built from defect-free regions of defective images rather than
genuinely clean ones.

## Result 4: the proposed "free improvement" was mathematically empty

`paper/lit/synthesis.md` recommended switching the gate from max softmax to
max-logit or energy, on IJCV 2024's finding that MSP is the weakest head-derived
OOD score.

**max-logit and MSP are identical to three decimals in all six folds, and this
is necessary rather than coincidental.** max-logit is `log(p/(1−p))`, a strictly
monotonic function of `p`, so it induces the same ranking, and AUROC is
rank-based.

The recommendation does not transfer because it assumes a **softmax** head,
where softmax normalises across classes and the logit therefore retains
magnitude information the normalised probability discards. **YOLOv8 uses
independent per-class sigmoids with no cross-class normalisation**, so the logit
is a monotone re-mapping of the probability and carries exactly the same
information.

Energy does aggregate across classes and is genuinely different — and it is
slightly *worse* in five of six folds.

There is no free improvement here. The recommendation is withdrawn.

## Limits

- 24–30 OOD images per fold. A fold's rate moves by 3–4 points per image, so
  treat fold-level differences under ~0.1 as noise and rely on the mean and the
  monotone trend across thresholds.
- Each fold trains on ~970–1030 images instead of 1236, so these models are
  weaker than the deployed one in absolute terms. Compare folds to each other,
  not to `metrics.md`.
- "Unseen" here means unseen *within NEU-DET*. A genuinely novel defect on a
  real line may be further from the training distribution than one NEU-DET class
  is from the other five, in which case these figures are optimistic.
