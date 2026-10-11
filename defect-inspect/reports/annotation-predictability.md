# Is crazing's background error unannotated defect? Mostly no — but crazing is still the outlier

`tide-decomposition.md` found crazing's error dominated by **Bkg = 20.65**, the
largest of any class: confident boxes where the ground truth has nothing. One
explanation is that those boxes are real defects the annotator missed, in which
case they are not model error at all but a ceiling imposed by the labels.

That is testable without re-annotating anything. A box that hits a missed
annotation should score like a true positive; a hallucination should not.

---

## Measurement

Every predicted box on the test split at conf 0.05, split by whether it matches
any ground-truth box at IoU ≥ 0.5.

| class | matched n | median conf | unmatched n | median conf | ratio |
|---|--:|--:|--:|--:|--:|
| **crazing** | 113 | **0.118** | 307 | **0.079** | **0.67** |
| rolled-in_scale | 92 | 0.253 | 124 | 0.085 | 0.34 |
| inclusion | 145 | 0.505 | 185 | 0.149 | 0.30 |
| scratches | 75 | 0.555 | 77 | 0.141 | 0.25 |
| patches | 92 | 0.750 | 120 | 0.129 | 0.17 |
| pitted_surface | 57 | 0.706 | 55 | 0.100 | 0.14 |
| **all** | 574 | 0.414 | 868 | 0.102 | 0.25 |

## The hypothesis mostly fails

Across the dataset, unmatched boxes score a median 0.102 against 0.414 for
matched ones, and **only 3% of the 868 unmatched boxes exceed confidence 0.5**.
They look like ordinary low-confidence false positives, not like confident
detections of something the annotator missed.

So "the Bkg error is really missed annotation" is **not supported** as a general
claim, and the label-ceiling argument cannot be made this way.

## What survives, and it is specific

**Crazing's ratio is 0.67 against 0.14–0.34 for every other class.** Its matched
boxes score a median 0.118 — barely above its unmatched 0.079, and lower than
*every other class's unmatched boxes*.

Read plainly: for crazing, the detector's confidence carries almost no
information about whether a box will agree with the annotation. For patches, a
confident box is 4–6× more likely to match; for crazing, confidence and
agreement are nearly independent.

That is a quantity worth reporting, because it says something the mAP number
does not: crazing's annotation is not predictable from the image content the
model has learned. It is consistent with the things/stuff framing
(`crazing-hypothesis-failed.md`) without depending on the partition mechanism
that was falsified there — a texture has no boundary the model could agree with,
so agreement becomes close to chance regardless of how confident the model is.

## What this does not establish

- **No label-convergence ceiling is computed.** Tschirschwitz & Rodehorst's
  method (WACV 2025) needs repeated independent annotations; this has one. The
  confidence ratio is a cheap proxy for "is agreement predictable", not a bound
  on achievable mAP.
- The 3% figure depends on the 0.05 confidence floor and the IoU 0.5 match
  criterion. Both are conventional, neither is neutral.
- A genuine audit — two or three annotators re-labelling 50 crazing images, and
  reporting pairwise IoU — remains the experiment that would settle it, and is
  the one piece of this that needs a person rather than a GPU.
