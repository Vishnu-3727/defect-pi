# TIDE on the independent system: classification error is absent there too

`tide-decomposition.md` found `Cls` error near zero across defect-inspect and
concluded the detector recognises but cannot localise. This checks it on
`defectvision` — separate codebase, separate model (trained at 640), separate
split, separate training run.

Predictions at conf 0.05, `rect=False`.

| class | imgs | AP50 | Cls | Loc | Dupe | Bkg | Miss |
|---|--:|--:|--:|--:|--:|--:|--:|
| crazing | 30 | 19.4 | **0.00** | 20.67 | 0.61 | 19.95 | 0.38 |
| inclusion | 39 | 24.8 | **0.00** | 4.04 | 0.67 | 0.62 | 1.72 |
| patches | 34 | 21.9 | **0.00** | 1.91 | 0.08 | 7.64 | 0.49 |
| pitted_surface | 30 | 28.3 | **0.00** | 3.04 | 0.30 | 14.47 | 0.61 |
| rolled-in_scale | 30 | 27.6 | **0.00** | 15.06 | 1.04 | **28.34** | 0.43 |
| scratches | 29 | 21.1 | **0.00** | 1.27 | 0.77 | 22.28 | 0.45 |

## Replicates: classification error is not the problem

**`Cls` is 0.00 for every class**, against a maximum of 0.11 on defect-inspect.
Two independently trained models, different training resolutions, different
splits, same conclusion: this detector family does not confuse NEU-DET's six
classes. The six-way taxonomy is easy; the boxes are hard.

That is now a two-system result and can be stated in the paper without
qualification. It also sharpens the reading of a low mAP on this dataset: it
does **not** mean the model cannot tell the defect types apart.

## Does not replicate: that crazing is distinctively background-dominated

`tide-decomposition.md` noted crazing's `Bkg = 20.65` as "by far its largest
component, and the largest Bkg of any class". On defectvision crazing's Bkg is
19.95 — but **rolled-in_scale reaches 28.34 and scratches 22.28**, both higher.

So crazing is not uniquely characterised by firing on unlabelled texture. It is
high on that axis in both systems, and on the independent system two other
classes are higher.

This is the third crazing-specific claim to fail (after the partition mechanism
and the "Loc-dominated only for crazing" prediction). The pattern is consistent:
**crazing is hard, but nothing measured so far makes it qualitatively different
from the other difficult classes.** The things/stuff framing remains the
defensible statement, and it was already published.

## Limits

- 29–39 images per class. Differences under ~2 dAP are not meaningful, which
  covers the crazing-vs-scratches Bkg comparison but not crazing-vs-rolled-in.
- Per-class subsets include out-of-class predictions, inflating `Bkg` in both
  systems equally; the comparison between them is fair, the absolute values are
  not comparable to a whole-dataset run.
- defectvision's model was trained at 640 and evaluated here at 320, the
  mismatched-but-graceful direction. Its confidences sit lower, which shifts
  what survives the 0.05 floor.
