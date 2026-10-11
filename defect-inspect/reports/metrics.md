# Detection metrics

- Generated: 2026-09-19
- Weights: `best.pt`
- Split: **test** (held out; never used for model selection)
- Image size: 320

## Overall

| Metric | Value |
|--------|-------|
| mAP@50 | **0.7411** |
| mAP@50-95 | **0.4063** |
| Precision (mean) | 0.6883 |
| Recall (mean) | 0.6804 |
| F1 (from mean P and R) | 0.6843 |

## Per class

| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|-------|-----------|--------|----|--------|-----------|
| crazing | 0.6119 | 0.2698 | 0.3745 | 0.3835 | 0.1317 |
| inclusion | 0.7194 | 0.7460 | 0.7325 | 0.7940 | 0.4080 |
| patches | 0.7269 | 0.8488 | 0.7831 | 0.8876 | 0.5797 |
| pitted_surface | 0.8056 | 0.8095 | 0.8076 | 0.8950 | 0.5335 |
| rolled-in_scale | 0.6364 | 0.5312 | 0.5791 | 0.6402 | 0.3013 |
| scratches | 0.6295 | 0.8772 | 0.7330 | 0.8465 | 0.4834 |

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
