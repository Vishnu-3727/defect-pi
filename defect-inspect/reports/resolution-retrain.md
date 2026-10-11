# Controlled retrain: the train-high/deploy-low hypothesis is wrong

Run on Kaggle, 2026-09-18. Notebook `paper/kaggle/resolution_retrain.ipynb`,
driver `paper/kaggle/run.py`, kernel `vishnu3727/defect-inspect-resolution-retrain`.

Tesla T4, ultralytics 8.4.155, torch 2.10.0+cu128, Python 3.12.13.
YOLOv8n from `yolov8n.pt`, 100 epochs each (neither early-stopped), seed 42,
`deterministic=True`. Split pinned by `paper/split-manifest.json` — the same
1236/353/181 images and 2887/801/438 boxes as the deployed model, asserted before
training. 11.4 min at 320, 18.7 min at 640.

---

## The hypothesis under test

`reports/resolution-study.md` observed that `defectvision` (trained at 640) scored
0.744 at 320 while this project (trained at 320) scored 0.711, and proposed that
**training at 640 and deploying at 320** was a free accuracy gain. That comparison
crossed two different test splits and two different training environments, so it
was recorded as a hypothesis requiring one controlled retrain.

This is that retrain. **The hypothesis is refuted.**

## Result

| model | eval 320 | eval 640 |
|---|---|---|
| trained at 320 | **0.7624** | 0.3438 |
| trained at 640 | 0.7277 | **0.7629** |

mAP@50 on the held-out test split. Full matrix with P/R/mAP50-95 in
`../paper/kaggle/output/resolution_matrix.csv`.

**Training at 640 and deploying at 320 costs 0.0347 mAP@50** against training at
320 directly (0.7277 vs 0.7624). It is not a free gain; it is a loss. The
deployed configuration — train at 320, infer at 320 — was already the right one.

## What the matrix does show

**Each resolution is equally accurate at its own native size.** 0.7624 at 320
against 0.7629 at 640, a difference of 0.0005 which is noise on 427 instances. The
extra pixels buy nothing on a dataset whose source images are 200×200. Since 640
costs 4.2× the inference time (measured on both a T4 and the Pi 5), 320 is
strictly better for deployment — the same conclusion `defectvision` reached, now
by a controlled route.

**The mismatch asymmetry is real, and larger than previously estimated.**

- Evaluating **above** the training resolution: 0.7624 → 0.3438, **−0.419**.
- Evaluating **below** it: 0.7629 → 0.7277, **−0.035**.

A model trained at 640 therefore degrades gracefully when run at 320, while one
trained at 320 collapses when run at 640 — twelve times the penalty. The earlier
report had the direction right and the sign of the down-scaling case wrong: it is
a small loss, not a small gain.

## Why the cross-project comparison misled

The control cell in the notebook failed, deliberately and informatively:

```
control trained320@320 = 0.7624  (original: 0.7107)
CONTROL FAILED -- environment differs from the original; do not compare.
```

Retraining at 320 in this environment scores **0.7624** where the deployed
weights score **0.7107** — a gap of **+0.0517** attributable to the training
stack, not to resolution. ultralytics 8.4.155 on a T4 produces a materially
better model from identical data, seed and epoch count than whatever version
produced `models/best.pt`.

That gap is larger than the 0.033 difference the original cross-project
observation was built on. The `defectvision`-beats-`defect-inspect` result is
therefore explained by training-environment differences and two different test
splits — not by input resolution. Had the notebook compared the new 640 model
against the old 320 weights, it would have shown +0.017 and "confirmed" a
hypothesis that the internal comparison refutes.

**This is the reason the control was included.** A two-variable comparison
dressed as one would have sent a wrong finding into a paper.

## The actionable result

The experiment refuted its hypothesis and produced a better model anyway:
`best_trained320.pt` scores **0.7624** against the deployed **0.7107**, at
identical inference cost, on the same split.

Before adopting it:

1. Run `scripts/parity.py` after ONNX export. The accuracy above is PyTorch on a
   GPU; the Pi runs ONNX INT8 on CPU, and the existing parity check (132/132
   boxes, IoU 1.0000) is what licenses carrying an accuracy number across that
   boundary. It has not been re-run for these weights.
2. Re-derive the INT8 calibration. Quantisation cost 0.3 mAP on the old weights;
   it cannot be assumed identical here.
3. Re-check `anomaly.score_threshold`. It is calibrated at the 95th percentile of
   defect-free scores under the old detector's confidence distribution, and the
   PatchCore gate fires on `detector_conf < 0.55`. A detector whose confidences
   have shifted changes how often the slow path runs, which is the most sensitive
   throughput parameter in the config.
4. Do not re-benchmark latency until the Pi's power supply is fixed — it is
   under-volting and dropping 2400 → 1000 MHz.

`best_trained640_320input.onnx` was exported before the result was known. It is
the *worse* model and should not be deployed; it is kept only so the matrix is
fully reproducible.
