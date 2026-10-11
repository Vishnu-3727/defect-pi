# Input resolution, and what the train/inference mismatch costs

> **Superseded in part, 2026-09-18.** The "train at 640, deploy at 320" hypothesis
> below was tested by controlled retrain and **refuted** — it costs 0.035 mAP@50
> rather than gaining 0.033. See `resolution-retrain.md`. The cross-project
> comparison that motivated it was confounded by the training environment, which
> accounts for +0.052 on its own. The measurements in this file are still correct;
> the inference drawn from them was not.


Regenerate with:

```bash
python scripts/evaluate.py --split test --imgsz 320 --device cpu
python scripts/evaluate.py --split test --imgsz 640 --device cpu
```

---

## Setup

Same weights (`models/best.pt`, YOLOv8n, **trained at imgsz 320**, 100 epochs),
same held-out test split (181 images, 438 instances), only `imgsz` differs.
Measured 2026-09-18 on a 12th-gen Core i5-1235U, CPU only. Accuracy is
device-independent; for deployment timings see `benchmark.md`.

The 320 row reproduces the published figure in `metrics.md` — 0.7107 measured
against 0.7106 recorded — which is the check that makes the rest of this table
trustworthy.

## Result

| imgsz | P | R | mAP@50 | mAP@50-95 | inference ms/img |
|---|---|---|---|---|---|
| **320** (native) | 0.627 | 0.662 | **0.7107** | **0.4063** | 22.1 |
| 640 | 0.302 | 0.338 | 0.3091 | 0.1436 | 90.2 |

**Inferring at twice the training resolution destroys the model**: mAP@50 falls
0.711 → 0.309, a 57% relative loss, for 4.1× the compute. Precision halves and
recall halves.

This is worth stating plainly because it is a deployment hazard, not a curiosity.
`imgsz` appears in `configs/default.yaml` as an innocuous tunable next to thread
count and confidence threshold. Thread count trades speed for nothing; `imgsz`
silently invalidates the model. Anyone who raises it expecting "more detail, more
accuracy" gets a detector that misses two defects in three, with no error and no
warning — every stage still runs, boxes still appear, and the dashboard still
reports a plausible reject rate.

## The asymmetry, read against the sibling project

`defectvision` ran the same experiment on a model **trained at 640**
(`../defectvision/reports/resolution-study.md`). Putting both together:

| model | trained at | eval 320 | eval 640 |
|---|---|---|---|
| defect-inspect | 320 | **0.7107** | 0.3091 |
| defectvision | 640 | **0.7440** | 0.7260 |

The mismatch penalty is strongly **asymmetric**:

- Evaluating **above** the training resolution is catastrophic: −0.402 mAP@50.
- Evaluating **below** it is mildly beneficial: +0.018 mAP@50.

A model trained at 640 therefore degrades gracefully when run at 320, while one
trained at 320 does not survive being run at 640 at all. If a single set of
weights has to serve more than one input size, train at the larger one.

## The actionable consequence

The pairing suggests a free accuracy gain for this project: **train at 640,
deploy at 320.** `defectvision` does exactly that and scores 0.744 where this
project scores 0.711 — at identical inference cost, since both run 320 at
deployment. The training run costs GPU time once; inference cost is unchanged.

That is the most promising accuracy lever identified so far, and it is cheaper
than anything in the inference path, which is already at its resolution optimum.

### Why this is suggestive and not yet proven

**The two models were evaluated on different test splits.** This project's split
holds 181 images and 438 instances; `defectvision`'s holds 182 and 427. Both are
stratified 70/20/10 draws from the same 1770-image NEU-DET pool, but with
different seeds, so the 0.744-vs-0.711 comparison crosses two different test
sets and is not a controlled result.

Other differences between the two training runs (augmentation settings,
learning-rate schedule, early-stop checkpoint) have not been diffed either. Both
are YOLOv8n at 100 epochs from `yolov8n.pt`, which is where the similarity is
established and where it ends.

To make the claim properly: retrain this project's weights at 640 and evaluate on
**this** project's 320 split. One training run, one evaluation, and the
comparison becomes controlled. Until then the cross-project row above is a
hypothesis with supporting evidence, and any paper should present it as such.

## Portability fix needed to run this

`data/yolo/data.yaml` hardcoded `path: C:/Users/vishn/Projects/defect-inspect/data/yolo`,
so evaluation worked only on the original Windows machine. The `path:` key is now
removed entirely rather than made relative — ultralytics resolves a relative
`path` against the wrong base, but with no `path:` key it uses the yaml's own
directory, which is correct on all three machines.
