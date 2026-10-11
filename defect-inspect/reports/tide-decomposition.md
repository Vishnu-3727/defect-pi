# TIDE: on NEU-DET, classification is never the problem

Run 2026-09-19 with `tidecv` (Bolya, Foley, Hays, Hoffman, *TIDE: A General
Toolbox for Identifying Object Detection Errors*, ECCV 2020,
<https://arxiv.org/abs/2008.08115>), v2 weights, test split, predictions at
conf 0.05.

TIDE isolates each error type by measuring the ΔAP recovered if that error
alone were fixed: **Cls** (right box, wrong class), **Loc** (right class, box
off), **Both**, **Dupe**, **Bkg** (fired where there is nothing), **Miss**
(ground truth with no detection).

No per-class TIDE decomposition on NEU-DET was found in the literature.

---

## Whole dataset

| Cls | Loc | Both | Dupe | Bkg | Miss |
|--:|--:|--:|--:|--:|--:|
| **0.14** | **12.68** | 0.23 | 1.81 | 2.03 | 2.12 |

AP@50 = 72.52. **Localisation error is 90× classification error.**

## Per class

Each row restricts ground truth to one class; predictions are all classes on
those images, which inflates `Bkg` relative to the whole-dataset run. Compare
rows to each other, not to the table above.

| class | imgs | AP50 | Cls | Loc | Dupe | Bkg | Miss |
|---|--:|--:|--:|--:|--:|--:|--:|
| crazing | 30 | 18.8 | **0.00** | 15.50 | 1.24 | **20.65** | 0.25 |
| inclusion | 40 | 20.5 | 0.11 | 1.93 | 0.30 | 7.49 | 0.67 |
| patches | 35 | 21.9 | 0.11 | 1.43 | 0.15 | 0.47 | 0.23 |
| pitted_surface | 31 | 29.4 | **0.00** | 1.97 | 0.22 | 15.02 | 0.84 |
| rolled-in_scale | 30 | 61.2 | **0.00** | **20.00** | 3.08 | 1.03 | 3.02 |
| scratches | 30 | 41.7 | **0.00** | 3.22 | 1.55 | 0.83 | 1.60 |

## What this confirms, and what it refutes

**Confirmed: the model recognises, it cannot localise.** `Cls` is 0.00 for
crazing and never exceeds 0.11 for any class. The earlier observation — that
crazing boxes are labelled correctly 100% of the time when they land — is not a
quirk of the confidence threshold; classification error is essentially absent
from this detector on this dataset.

**Refuted: that this is special to crazing.** The prediction recorded in
`crazing-hypothesis-failed.md` was that crazing's budget would be Loc+Miss
dominated with near-zero Cls *while other classes were not*. Cls is ~0
everywhere, and `rolled-in_scale` has the **highest** Loc error of any class
(20.00 against crazing's 15.50). Crazing is not qualitatively different in the
Loc dimension.

**What is distinctive about crazing is `Bkg`, at 20.65** — by far its largest
component, and the largest Bkg of any class. Bkg means the detector produced
confident boxes where the ground truth has nothing at all. For a defect
distributed across a repeating texture that the annotator marked in only 2.3
places, that is the expected signature: the model fires across the texture, the
labels cover part of it, and the surplus scores as background error.

So the surviving statement is narrower and better evidenced than the one it
replaces: **crazing's error is dominated by firing on unlabelled texture, not by
misrecognition and not by missing the defect** (`Miss` is 0.25, the lowest of
any class — the model almost never fails to respond to crazing entirely).

## The generalisable observation

Across every class, this detector's error is localisation and background, and
essentially never classification. The six-way taxonomy is easy; the boxes are
hard. That is worth one line in the paper, because the standard mAP headline
conflates the two and invites reviewers to read a low mAP as "the model cannot
tell the defect types apart", which on this evidence is the opposite of true.

It also supports the design argument elsewhere in this project: for a PASS/FAIL
ejector, the quantity the system needs (which defect, if any, is present) is the
one the detector gets right, and the quantity it is graded on (where exactly)
is the one it gets wrong.

## Limits

- Per-class subsets include out-of-class predictions, inflating `Bkg`. A stricter
  variant would filter predictions to the class under test, but that would hide
  exactly the cross-class firing this table is measuring.
- 30–40 images per class. Differences under ~2 dAP are not meaningful.
- conf 0.05 was used to admit the low-confidence boxes the threshold study
  showed matter. At 0.35 the Bkg column would shrink and Miss would grow.
