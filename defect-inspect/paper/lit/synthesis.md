# Research synthesis, 2026-09-18

> **SEPARATION RESTATED 2026-09-19.** The "median defect 2.40 vs median
> normal 2.26, separation +0.141" quoted from this project's calibration was
> measured on 46 regions per group. On 669 regions the same statistic reads
> **+0.273 with AUROC 0.819**. PatchCore separates better than this project has
> been assuming, and "weak separation" should not be used as a premise until
> re-measured. See `../../reports/patchcore-denoise.md`.


Five parallel literature sweeps plus four local experiments. What survives, what
died, and what to do next. Citations marked *verified* were fetched and checked
here; the rest come from the sweeps and must be re-checked before submission.

---

## Scoreboard

| claim | status |
|---|---|
| Threshold decoupling cuts escapes 15× | **Holds. Replicated 31× on the sibling system.** Gap is real but thin. |
| Quantisation delta variance is unreported | **Holds.** Effect is established; its variance is not. |
| Crazing fails because its box partition is arbitrary | **FALSIFIED by our own test.** |
| Crazing is ill-posed for bounding boxes | **Not novel** — the things/stuff distinction. |
| PatchCore is redundant for in-distribution defects | **Holds, and is novel as a measured claim** — but does not generalise. |
| PatchCore is redundant generally | **Not supported. The literature says the opposite.** |

---

## 1. Threshold decoupling — the strongest result

Zero hits for the specific claim: two simultaneous thresholds split by consumer,
the actuator versus the display. Every ingredient is published separately;
nobody assembles them or names the conflation as a defect.

The idea is nonetheless one sentence from folklore — a metal-detector product
page draws the identical distinction between what a machine senses and when it
alerts. **Lead with the measurement, not the idea.** Use the canonical
vocabulary: escape rate / overkill rate, formalised in IJAMT 2011
(<https://doi.org/10.1007/s00170-011-3342-2>).

Position against UAI 2024 (<https://arxiv.org/abs/2404.17427>) and Algorithms
19(5):409 2026 (<https://doi.org/10.3390/a19050409>), both of which select **one**
operating point well. *Verified: the latter's own abstract says
"operating-threshold recommendation", singular.* Its full text is still unread —
MDPI was in maintenance all session. **Read it before submitting.**

## 2. Quantisation variance — a clean small contribution

The effect is established and named: QReg (<https://arxiv.org/abs/2206.12372>,
*verified*) models quantisation as additive noise and reports 8-bit as a reliable
regulariser; AQD (CVPR 2021, *verified*) shows 4-bit RetinaNet beating full
precision by 1.8% AP.

**Nobody reports the variance.** Every source quotes a single "costs X mAP".
Our two runs of one recipe gave +0.0217 and −0.0271 — a 0.049 spread, wider than
most improvements this project has measured. The experiment to make this
publishable is cheap and unrun: quantise one model N times with different
calibration draws, and K independently trained models, then report a
distribution.

## 3. Crazing — our mechanism was wrong, and the real framing was already published

We claimed the ~2.3-box partition is arbitrary. That predicts region-level
scoring should rescue crazing specifically. It does not: crazing gains +0.100
while `pitted_surface` gains +0.149, and crazing stays worst. **Falsified**
(`../../reports/crazing-hypothesis-failed.md`).

The correct framing was published years ago: crazing is a **"stuff"** class
annotated with a **"thing"** formulation — COCO-Stuff (CVPR 2018,
<https://arxiv.org/abs/1612.03716>) and Panoptic Segmentation (CVPR 2019,
<https://arxiv.org/abs/1801.00868>).

And crazing being NEU-DET's worst class is the **published norm**: worst in
16/17 models in one table, 8/8 in another, 5/5 in a third. Our crazing mAP@50
0.4666 sits within noise of the two published YOLOv8n values (0.473, 0.476) and
our recall 0.2467 beside SH-DETR's 0.239. **Nothing in our pipeline needs
fixing, and no novelty claim survives here.**

What is unclaimed: no per-class **TIDE** decomposition (ECCV 2020,
<https://arxiv.org/abs/2008.08115>, `pip install tidecv`) on NEU-DET isolating
Loc versus Cls error, and no inter-annotator study for defect boxes. Either
would be a small real contribution. Neither is run.

*Caveat on one citation:* arXiv:2606.07659 asserts the formulation-level claim
for NEU-DET crazing directly, but its abstract claims 98.5% mAP while its own
tables report 73–74% — *verified by fetching the abstract*. Treat as unreliable.

## 4. PatchCore — the finding is real, and narrower than we stated

**Novel as a measured claim.** No paper ablates an unsupervised branch out of a
supervised+unsupervised pipeline and reports its marginal contribution. Nearest
prior art, and the right thing to cite: the fabric cascade paper
(<https://arxiv.org/abs/2608.14727>), which decomposed its own 1.36× speedup and
found **91% came from overlapping JPEG decode with inference — the cascade
itself contributed 5.1%**. That is our finding in mirror image.

Supporting: ADBench (NeurIPS 2022, <https://arxiv.org/abs/2206.09426>) — "with
merely 1% labeled anomalies, most semi-supervised methods can outperform the
best unsupervised method."

**But the generalisation does not hold, and this corrects our report.** DRA
(CVPR 2022, <https://arxiv.org/abs/2203.14506>) runs the hard setting — train on
one anomaly class, test only on unseen classes. At 10 training anomalies,
unsupervised KDAD scores **0.847** while every naive supervised baseline scores
**0.751–0.762**. A stock YOLOv8n is a naive supervised baseline. **The blind
spot PatchCore was built for is real**, and our measurement — all six classes
in-distribution — cannot see it.

So: keep PatchCore. Our claim is only that it is redundant *for defects the
detector was trained on*, which is 100% of what NEU-DET contains.

### The free improvement we should make

Our gate uses **softmax confidence**, which the OOD literature identifies as the
**weakest** head-derived score. IJCV 2024
(<https://arxiv.org/html/2408.16757v1>) finds max-logit and energy best across
OOD and OSR benchmarks. On a detection benchmark: MSP FPR95 ≈ 71% versus energy
≈ 57%.

Switching the gate from softmax confidence to an energy or max-logit score costs
nothing — no second model, no memory bank, no clean images — and is better
supported than either of our current options. YOLOv8 applies a sigmoid in the
head, so the logit is recoverable as `log(p/(1−p))`.

### Context for our 110 ms PatchCore measurement

PatchCore's own paper reports 0.17–0.6 s/image **on a V4 GPU**. On a Raspberry
Pi 4, an independent study measured **~3.5 s/image** and concluded the system
"would not work on a high-frequency system"
(<https://arxiv.org/abs/2409.15980>). Our 110 ms on a Pi 5 is far faster than
both, which is explained by our very small coreset (`patchcore_bank.npz` is
1.1 MB) — worth stating explicitly in the paper, because a reviewer will
otherwise disbelieve it.

Also relevant to our weak 2.40-vs-2.26 separation: FR-PatchCore (Sensors 2024)
reports PatchCore's similarity principle breaking under rotation and pixel
misalignment, since MVTec categories are near-aligned and real samples are not.

---

## The two experiments that would settle everything

1. **DRA hard-setting protocol, for the PatchCore question.** Retrain on five of
   the six classes, remove the sixth from training entirely, test only on the
   held-out class. That directly measures whether the detector's blind spot is
   real for *our* model and whether PatchCore covers it. The Kaggle
   infrastructure for this already exists (`paper/kaggle/`); it is six training
   runs, one per held-out class, and nothing else needs building.

2. **TIDE per-class decomposition, for the crazing question.** One run. Predicts
   Loc+Miss ≫ Cls for crazing alone. Unpublished for NEU-DET.

Both are cheap. Neither is done. Everything above is bounded by them.

## Still blocking any deployment change

NEU-DET has no defect-free images, so the overkill rate at a lower threshold
remains unmeasurable. ~50 clean images from the production rig would unblock the
threshold change *and* fix PatchCore's memory bank, which is currently built
from defect-free regions of defective images. No paper isolates that
construction as an ablation — SoftPatch (NeurIPS 2022) treats normal patches of
defective images as usable — so it is untested rather than known-bad.
