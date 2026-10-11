# Model v2: what changed, and what the Pi will actually see

> **EXPLANATION CORRECTED 2026-09-19.** The cause given below -- quantisation
> moving the two models in opposite directions -- is wrong. v2's advantage is
> +0.0517 in PyTorch, +0.0057 in FP32 ONNX and +0.0029 in INT8: it is lost at
> the PyTorch->ONNX boundary, not at quantisation. The conclusion, that the
> deployed gain is marginal, still holds. See `quantisation-corrected.md`.


Adopted 2026-09-18. Weights from the controlled Kaggle retrain
(`resolution-retrain.md`): YOLOv8n, imgsz 320, 100 epochs, seed 42,
ultralytics 8.4.155 on a Tesla T4. Same split as v1.

v1 is archived under `models/v1-2026-08-08/` and can be restored by copying it
back over `models/`.

---

## The headline number is not the deployed number

| artefact | mAP@50 | mAP@50-95 |
|---|---|---|
| v1 `best.pt` (PyTorch) | 0.7107 | 0.4063 |
| **v1 `model_int8.onnx`** — what the Pi ran | **0.7324** | **0.3898** |
| v2 `best.pt` (PyTorch) | 0.7624 | 0.4373 |
| v2 `model.onnx` (FP32) | 0.7411 | 0.4063 |
| **v2 `model_int8.onnx`** — what the Pi runs | **0.7353** | **0.4109** |

At PyTorch level v2 gains **+0.0517 mAP@50**. On the artefact that actually
runs on the Pi it gains **+0.0029 mAP@50 and +0.0211 mAP@50-95**.

The mAP@50 gain is noise. The mAP@50-95 gain is real but modest, and means the
boxes are placed more tightly rather than more defects being found.

**Report the INT8 row, not the PyTorch row.** Quoting +0.052 for this upgrade
would be quoting a number no deployed component produces.

### Why the gain collapses

Quantisation moves the two models in opposite directions:

- v1: 0.7107 → 0.7324 on INT8, **+0.0217**
- v2: 0.7624 → 0.7353 on INT8, **−0.0271**

INT8 *helped* v1 and *hurt* v2. At a 0.001 confidence floor, quantisation
perturbs the low-confidence tail, which is where mAP@50 is won and lost; that
perturbation is not required to have a consistent sign. It does mean the
existing documentation line — "INT8 costs 0.3 mAP" — describes v1 only and does
not generalise.

The direction reverses on mAP@50-95 (v1 −0.0165, v2 −0.0264), so both models
lose localisation precision under quantisation, as expected. Only the loose-IoU
metric is unstable.

## Parity

FP32 ONNX reproduces ultralytics exactly at the operating threshold:

| model | ref boxes | ours | matched | recall | mean IoU | class errors |
|---|--:|--:|--:|--:|--:|--:|
| `model.onnx` | 121 | 121 | 121 | 1.000 | 1.0000 | 0 |
| `model_int8.onnx` | 121 | 117 | 107 | 0.884 | 0.9518 | 0 |

The FP32 row is the check that licenses carrying accuracy across the export
boundary, and it is perfect — matching v1's documented 132/132 at IoU 1.0000.

The INT8 row is worth watching: it finds 117 boxes where the reference finds
121, and only 107 agree at IoU ≥ 0.70. No box is given the wrong class. This is
consistent with the mAP@50-95 drop and is the honest cost of quantisation at the
operating point.

## Per-class, v1 → v2 (PyTorch, test split)

| Class | mAP@50 v1 | mAP@50 v2 | Recall v1 | Recall v2 |
|---|--:|--:|--:|--:|
| crazing | 0.3855 | **0.4666** | 0.2222 | 0.2467 |
| inclusion | 0.8711 | 0.8446 | 0.7619 | 0.7540 |
| patches | 0.9111 | 0.9171 | 0.8953 | 0.8837 |
| pitted_surface | 0.7451 | 0.8511 | 0.7281 | 0.7662 |
| rolled-in_scale | 0.5138 | 0.6288 | 0.5068 | 0.5625 |
| scratches | 0.8370 | 0.8663 | 0.8596 | 0.8772 |

The two weakest classes improve most — crazing +0.081 and rolled-in_scale
+0.115 mAP@50. **Crazing recall is still 0.2467**, so roughly three in four
crazing defects still escape. That limitation is unchanged and must stay in the
limitations section of any paper.

## Effect on the anomaly gate

PatchCore runs when detector confidence is below 0.55, and each call costs about
110 ms — the most throughput-sensitive parameter in the configuration. Measured
over 120 test images, per-image maximum confidence:

| | median | mean | fraction < 0.55 |
|---|--:|--:|--:|
| v1 INT8 | 0.800 | 0.684 | 0.292 |
| v2 INT8 | 0.795 | 0.696 | **0.233** |

The distribution barely moves, so `anomaly.score_threshold` does not need
recalibrating for this swap. The gate fires on 23.3% of frames instead of 29.2%
— about a fifth fewer PatchCore calls, which is a small throughput gain rather
than a risk.

## Still outstanding

1. **Not deployed to the Pi.** The board is powered off. `models/` on the drive
   holds v2; the Pi still has v1 and must be synced before it runs anything.
2. **No latency measurement.** The Pi is under-volting (2400 → 1000 MHz), so no
   FPS or p95 figure taken now is usable. Re-benchmark only after the supply is
   replaced.
3. `reports/benchmark.md` still describes v1 throughout. Its FPS numbers are
   architecture-bound rather than weights-bound and should carry over, but that
   is an assumption until measured.
