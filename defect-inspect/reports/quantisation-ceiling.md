# INT8 imposes a ceiling: 91% of the FP32 advantage does not survive it

> **WITHDRAWN 2026-09-19.** There is no ceiling. Ten checkpoints spanning
> FP32 0.550-0.741 give INT8 = -0.023 + 1.025*FP32, R^2 0.997, with INT8 spread
> equal to 100% of FP32 spread. This report's central claim is false and its
> cause was a cross-runtime comparison. See `quantisation-corrected.md`.


`quantisation-variance.md` recorded that the same quantisation procedure gave
+0.0217 mAP@50 on the v1 model and −0.0271 on v2, and proposed measuring the
variance. The variance turns out to be small. The explanation is something else,
and more useful.

Method: the same FP32 model quantised repeatedly through the project's own
`scripts/export_onnx.py --int8`, varying only `--seed`, which changes which 200
images are drawn for calibration. Evaluated on the held-out test split each time.

---

## Calibration draw is not the source

| model | n draws | FP32 mAP@50 | INT8 mean | INT8 sd | delta | min | max |
|---|--:|--:|--:|--:|--:|--:|--:|
| v1 (Aug 2026) | 6 | 0.7107 | 0.7324 | 0.0036 | **+0.0217** | 0.7255 | 0.7361 |
| v2 (Sep 2026) | 8 | 0.7411 | 0.7350 | 0.0027 | **−0.0061** | 0.7310 | 0.7378 |

Within a model the delta is **stable**: sd 0.0027–0.0036, and every one of v2's
eight draws is negative while every one of v1's is positive. The sign does not
flip with the calibration set. It is a property of the trained model.

## The models converge

| | v1 | v2 | gap |
|---|--:|--:|--:|
| FP32 mAP@50 | 0.7107 | 0.7411 | **+0.0304** |
| INT8 mAP@50 | 0.7324 | 0.7350 | **+0.0026** |

**Two models separated by 0.0304 mAP@50 in FP32 are separated by 0.0026 after
INT8 quantisation — 91% of the advantage is gone, and what remains is smaller
than the calibration standard deviation, so it is not distinguishable from
zero.**

Both land at ~0.734. The weaker model is lifted, the stronger is pulled down,
and the sign of the "quantisation cost" is simply which side of ~0.734 the FP32
model started on. There is no paradox in the earlier sign flip; there is a
ceiling.

## Why this matters more than the variance question

It explains the disappointing v2 upgrade directly. `model-v2.md` recorded a
+0.0517 FP32 gain collapsing to +0.0029 on the artefact the Pi actually runs,
and called the cause "quantisation moves the two models in opposite
directions". The real cause is that **INT8 caps this architecture on this
dataset at about 0.734, and v2's FP32 gain was mostly above the cap.**

The practical consequence is sharp and applies beyond this project:

> **For a model deployed in INT8, FP32 model selection is close to
> uninformative once the FP32 score is near the ceiling.** Choosing between
> checkpoints, resolutions or training recipes on FP32 mAP can rank models that
> are indistinguishable after quantisation — which is what happened here across
> a full retrain, a resolution study and a Kaggle GPU run.

Anything that would improve the deployed system has to be shown to improve the
**quantised** model. Every accuracy comparison in this project that used FP32
numbers should be re-read with that in mind.

## Relation to the literature

The regularisation account (QReg, <https://arxiv.org/abs/2206.12372>; AQD,
CVPR 2021) explains why quantisation can *help* a model — noise injection acting
as a regulariser. It does not by itself predict **convergence** of two models to
a common level, which is what the table above shows and which is the more
actionable half.

The sweep found no paper characterising the run-to-run distribution of the PTQ
delta; it also found none reporting that the delta's sign is determined by the
FP32 model's position relative to a quantised ceiling. Both remain unclaimed as
far as those searches reached.

## Limits

- One architecture (YOLOv8n), one dataset, one quantisation scheme (ONNX Runtime
  static, MinMax, detection head kept in FP32). The ceiling value 0.734 is
  specific to all of those.
- Two models is two points. The claim "there is a ceiling" would be much stronger
  with a third and fourth FP32 model spanning a wider range — for example the
  640-trained checkpoint from the resolution study, which is already on disk and
  has not been quantised.
- v1's sweep is 6 draws rather than 8; the run was still completing. Its sd is
  computed over 6.
