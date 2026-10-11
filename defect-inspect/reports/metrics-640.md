# Detection metrics

- Generated: 2026-09-18
- Weights: `best.pt`
- Split: **test** (held out; never used for model selection)
- Image size: 640

## Overall

| Metric | Value |
|--------|-------|
| mAP@50 | **0.3091** |
| mAP@50-95 | **0.1436** |
| Precision (mean) | 0.3019 |
| Recall (mean) | 0.3380 |
| F1 (from mean P and R) | 0.3190 |

## Per class

| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|-------|-----------|--------|----|--------|-----------|
| crazing | 0.0000 | 0.0000 | 0.0000 | 0.0858 | 0.0253 |
| inclusion | 0.5503 | 0.6825 | 0.6093 | 0.5659 | 0.2665 |
| patches | 0.5645 | 0.7209 | 0.6332 | 0.6736 | 0.3717 |
| pitted_surface | 0.0290 | 0.0238 | 0.0262 | 0.0145 | 0.0015 |
| rolled-in_scale | 0.4803 | 0.2500 | 0.3288 | 0.3300 | 0.1390 |
| scratches | 0.1873 | 0.3509 | 0.2443 | 0.1848 | 0.0578 |

## How to read this

- **Precision** — of the boxes the model reported, what fraction were real.
  Low precision means false alarms: good parts get ejected.
- **Recall** — of the real defects, what fraction the model found.
  Low recall means escapes: defective parts reach the customer.
- **mAP@50** — average precision across recall levels, counting a box
  correct when it overlaps the truth by at least 50%. The standard
  headline number for detection.
- **mAP@50-95** — the same, averaged over IoU thresholds 0.50 to 0.95.
  Much stricter: it grades how tightly the box fits, not just whether
  the defect was found. Always lower, and it is meant to be.

For an inspection system these two errors are not equally costly.
Shipping a defective part is normally worse than re-checking a good one,
so the operating point should favour recall. That is a business decision
expressed as one number: `detector.conf_threshold` in `configs/default.yaml`.

## Plots

### confusion matrix normalized

![confusion_matrix_normalized.png](confusion_matrix_normalized.png)

Raw ultralytics output: `/media/vishnu/2E44590D4458D8E3/Vishnu/defect-inspect/runs/eval_test`
