# Two evaluation settings span 0.025 mAP — more than most published improvements

Extends `imgsz-rect-trap.md`. That report showed `rect=True` feeds 352 instead of
320. A second setting does comparable damage, and the two compose.

---

## The grid

`defectvision`, one checkpoint, one test split. Only two evaluation settings
vary — neither is part of the model:

| NMS IoU | fed | mAP@50 | mAP@50-95 |
|--:|---|--:|--:|
| 0.45 | 352 | **0.7443** | 0.4218 |
| 0.45 | 320 | 0.7360 | 0.3994 |
| 0.70 | 352 | 0.7350 | **0.4343** |
| 0.70 | 320 | **0.7189** | 0.4054 |

- **Range on mAP@50: 0.0254.** From settings alone.
- The published figure for this project, 0.744, is the top-left cell: NMS IoU
  0.45 from its own config, and the rect default. Reproduced to four decimals,
  so the number was never wrong — it was under-specified.
- **The deployed configuration is the third row down**: 320 input, NMS 0.45.
  **0.7360**, not 0.744.

## The two metrics rank the settings oppositely

mAP@50 prefers NMS IoU 0.45 (0.7443 > 0.7350 at 352). mAP@50-95 prefers 0.70
(0.4343 > 0.4218 at the same input size). A practitioner tuning on one headline
metric is detuning the other, and neither choice is visible in a results table
that reports only "mAP@50 = 0.744, imgsz 320".

## Why this is the important version of the finding

0.0254 is larger than:

- the gap between our v1 and v2 models at matched resolution (**0.0057**);
- the margin `defectvision`'s resolution study rests on (**0.0095** as published);
- the quantisation cost measured across ten checkpoints (**0.0067**);
- and most increments separating adjacent methods in the published NEU-DET
  baseline table, where papers are commonly 1–3 points apart.

So on this dataset, **the evaluation protocol has more leverage on the headline
number than the contributions being compared.** A NEU-DET leaderboard entry that
does not state its input size, rect setting and NMS IoU is not reproducible, and
none of the papers surveyed state all three.

## What survives, checked rather than assumed

`defectvision`'s conclusion that 320 beats 640 **holds** at true resolutions, and
the margin grows: +0.0095 as published against **+0.0135** measured at 320 vs 640
rather than 352 vs 672. Its ranking was never at risk; only its axis labels and
absolute values were.

One hypothesis was tested and **failed**: batch size does not affect the result
here. rect computes a padded shape per batch from the batch's aspect ratios, so
mixed-aspect datasets would vary — but NEU-DET is uniformly 200×200, so every
batch lands on 352 regardless. Recorded because it is the natural next worry and
it is not a problem for this dataset.

## What to do

1. Every reported number states: input size actually fed, `rect`, NMS IoU,
   confidence floor, and runtime. Five fields. Four of the five silently altered
   a number in this project today.
2. `scripts/evaluate.py` already forces `rect=False`; it should also print all
   five fields into `metrics.md` so the report carries its own provenance.
3. For the paper, quote the deployed configuration and say so explicitly:
   defect-inspect **0.7411**, defectvision **0.7360**.

## Limits

- Ultralytics 8.4.155; defaults and stride arithmetic may differ by version.
- Two settings explored. Confidence floor, letterbox padding colour, and
  `single_cls` were not swept and may add further spread.
- The 0.0254 range is specific to this dataset and model. The argument that it
  exceeds the contributions being compared is specific to NEU-DET's literature.
