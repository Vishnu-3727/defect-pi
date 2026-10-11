# The "arbitrary partition" explanation for crazing does not survive testing

## What was claimed

`threshold-decoupling.md` proposed a mechanism for crazing's poor box-level
performance: crazing is a diffuse texture covering ~50% of the image, cut into
~2.3 boxes by annotator judgement, so the *partition* is arbitrary and the
detector cannot reproduce it even when it sees the defect correctly.

That predicted something testable. If the partition is the problem but the
region is not, then scoring at region level — union of ground-truth boxes
against union of predicted boxes — should rescue crazing **specifically**.

## The test

Predictions at conf 0.10, test split, images with at least one detection.
"Partition" = median best IoU per ground-truth box. "Region" = IoU of the
unioned ground truth against the unioned predictions.

| class | imgs | median partition IoU | median region IoU | gain |
|---|--:|--:|--:|--:|
| crazing | 29 | 0.557 | 0.657 | **+0.100** |
| inclusion | 30 | 0.753 | 0.848 | +0.095 |
| patches | 30 | 0.820 | 0.909 | +0.089 |
| pitted_surface | 31 | 0.782 | 0.931 | **+0.149** |
| rolled-in_scale | 30 | 0.711 | 0.789 | +0.078 |
| scratches | 30 | 0.783 | 0.588 | **−0.194** |

## The hypothesis fails

Crazing's region-level gain (+0.100) is unremarkable — `pitted_surface` gains
half again as much, and three other classes gain about the same. Unioning boxes
helps almost everything a little, which is what you would expect from a looser
scoring rule, and tells us nothing specific about crazing.

More decisively: crazing remains the **worst-localised class at region level
too** (0.657, above only scratches). If the partition were the obstacle, region
scoring should have moved crazing into the pack. It did not.

So the failure is not that the boxes are carved up arbitrarily. Crazing is
simply harder to localise at every granularity tested.

Scratches moving the other way (−0.194) is a sanity check that the metric
behaves sensibly: scratches are thin diagonal lines, and unioning several of
them produces a large rectangle that is mostly background, so region scoring
should and does punish that class.

## What the literature says instead

The concept was never novel, and the honest framing is stronger than the
discarded mechanism. Crazing is a **"stuff"** class annotated with a **"thing"**
formulation:

- Caesar, Uijlings, Ferrari, *COCO-Stuff: Thing and Stuff Classes in Context*,
  CVPR 2018 — <https://arxiv.org/abs/1612.03716>. "Things" have well-defined
  shape and are countable; "stuff" is amorphous, has no distinct parts and no
  fixed shape.
- Kirillov, He, Girshick, Rother, Dollár, *Panoptic Segmentation*, CVPR 2019 —
  <https://arxiv.org/abs/1801.00868>. Stuff has no instances, which is why
  instance/box formulations do not apply to it.

That published distinction covers the observation, and it does not depend on the
partition mechanism this test just falsified.

Independently, crazing being the worst NEU-DET class is the **published norm**,
not a defect of this model: worst in 16/17 models in one comparison table, 8/8
in another, 5/5 in a third. Our crazing mAP@50 of 0.4666 is within noise of the
two published YOLOv8n figures (0.473, 0.476), and our crazing recall of 0.2467
sits beside SH-DETR's 0.239. Nothing here needs fixing in our pipeline.

## The experiment that would actually settle it

**TIDE** (Bolya, Foley, Hays, Hoffman, *A General Toolbox for Identifying Object
Detection Errors*, ECCV 2020 — <https://arxiv.org/abs/2008.08115>, `pip install
tidecv`) decomposes detection error into Cls, Loc, Both, Dupe, Bkgd and Miss,
each scored by its isolated ΔAP.

The surviving claim — the model recognises crazing but cannot localise it —
predicts crazing's error budget is dominated by **Loc** and **Miss** with near
zero **Cls**, while other classes are not. That is a single run and it either
holds or it does not.

No TIDE decomposition per class on NEU-DET was found in the literature, so
running it is both the rigorous test and a small contribution. It is the next
thing to do and it has not been done here.

## Why this is recorded rather than deleted

The mechanism was asserted in a report and committed before it was tested. It
took one experiment to falsify. Leaving the claim in place and quietly not
mentioning it would have put a wrong explanation into a paper draft that already
cites the correct one.
