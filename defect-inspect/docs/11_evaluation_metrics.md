# 11 — Evaluation Metrics

**Prerequisites:** [08 — YOLO](08_yolo_v1_to_v8.md), [10 — Training](10_training_hyperparameters.md).

---

## 1. Concept

A detector's output is a list of boxes with classes and confidences. The truth
is a different list. Evaluation is the machinery for turning "how close are
these two lists" into a number.

Accuracy is not that number, and cannot be. "95% accurate" is meaningless for
detection: accuracy needs a denominator of decisions, and a detector makes a
variable number of them per image. What we use instead is precision, recall, and
mean Average Precision.

---

## 2. Why it is needed

Two errors matter in inspection, and they have opposite costs:

| Error | Name | Consequence on the line |
|-------|------|------------------------|
| Reported a defect that is not there | **False positive** | A good part is ejected. Cost: scrap, or a re-inspection |
| Missed a defect that is there | **False negative** | A defective part ships. Cost: a customer complaint |

One number cannot express both. Worse, **you can trade one for the other with a
single config value** — lowering `conf_threshold` catches more real defects and
also more phantom ones. So evaluation must report the trade-off, not a point on
it.

That is precisely what mAP does: it summarises performance across all possible
thresholds.

---

## 3. Real-world example

A rolling mill produces 1000 parts, of which 50 are genuinely defective.

**System A** flags 60 parts, and 45 of them are truly defective.
- Precision = 45/60 = **0.75** — three quarters of the ejections were justified.
- Recall = 45/50 = **0.90** — nine of ten defects were caught.
- 5 defective parts shipped. 15 good parts were scrapped.

**System B** flags 200 parts, and all 50 defective ones are among them.
- Precision = 50/200 = **0.25**
- Recall = 50/50 = **1.00**
- Nothing shipped defective. 150 good parts scrapped.

Neither is "better". B is right for aerospace fasteners, where an escape is
catastrophic. A is right for construction rebar, where scrapping 150 good parts
per thousand destroys the margin. **The metric does not choose; the business
does, and it expresses that choice as `conf_threshold`.**

---

## 4. How it works

### The four outcomes

For detection, a prediction counts as correct only if the **class matches** and
the **IoU with a ground-truth box exceeds a threshold** — usually 0.5.

| | Defect present | No defect |
|---|---|---|
| **Predicted defect** | True Positive (TP) | False Positive (FP) |
| **Predicted nothing** | False Negative (FN) | True Negative (TN) |

Detection has no meaningful TN — there is no count of "correctly reported
nothing" at every position. This is exactly why accuracy, which needs TN, does
not apply.

### Precision and recall

$$ \text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN} $$

Read them as questions:

- **Precision**: of everything I flagged, how much was real? *(Do I cry wolf?)*
- **Recall**: of everything real, how much did I flag? *(Do I miss things?)*

They move in opposite directions as the confidence threshold changes.
Lower the threshold and you report more boxes: recall rises, precision falls.

### F1

$$ F_1 = 2 \cdot \frac{P \cdot R}{P + R} $$

The harmonic mean, which punishes imbalance — a model with precision 1.0 and
recall 0.1 scores 0.18, not 0.55. Useful as one summary number when both errors
cost roughly the same. **For our system they do not**, which is why F1 is
reported but not optimised.

### Average Precision, and mAP

Sweep the confidence threshold from 1.0 down to 0. At each value, compute
precision and recall. Plot precision against recall:

```
 precision
   1.0 |‾‾‾‾‾‾‾‾‾‾\
       |            \___
       |                \____          Average Precision =
       |                     \___      the area under this curve
       |                         \__
   0.0 +--------------------------\--
       0                          1.0  recall
```

**AP** is the area under that curve for one class. **mAP** is the mean of AP
across all classes.

- **mAP@50** — a prediction counts as correct at IoU ≥ 0.5. The headline number.
- **mAP@50-95** — the same, averaged over IoU thresholds 0.50, 0.55, … 0.95.
  Far stricter: it grades how *tightly* the box fits, not merely whether the
  defect was found. It is always lower, and it is meant to be.

A model can have excellent mAP@50 and poor mAP@50-95 — it finds every defect but
draws sloppy boxes. Ours: 0.711 against 0.407. That ratio is normal for a
dataset where the defect boundary is genuinely ambiguous, as crazing's and
rolled-in scale's are.

### Our results, and how to read them

From `reports/metrics.md`, on the **test** split — 181 images, 438 boxes, never
used for model selection:

| | mAP@50 | mAP@50-95 |
|---|---:|---:|
| **Overall** | **0.7107** | **0.4066** |

| class | precision | recall | F1 | mAP@50 |
|-------|----------:|-------:|---:|-------:|
| crazing | 0.549 | **0.222** | 0.316 | **0.385** |
| inclusion | 0.710 | 0.762 | 0.735 | 0.871 |
| patches | 0.750 | 0.895 | 0.816 | 0.911 |
| pitted_surface | 0.560 | 0.728 | 0.633 | 0.745 |
| rolled-in_scale | 0.546 | 0.507 | 0.526 | 0.514 |
| scratches | 0.649 | 0.860 | 0.740 | 0.837 |

**The per-class table is the useful one, and the average hides the problem.**
0.711 sounds respectable. It is the mean of a 0.911 and a 0.385. Crazing recall
of 0.222 means **roughly four in five crazing defects are missed** — and an
average that conceals that is worse than no average.

Reporting the per-class breakdown, and naming the weak class before an examiner
finds it, is the difference between a defensible result and a discovered one.

### Confusion matrix

The confusion matrix (`reports/confusion_matrix_normalized.png`) shows *which*
classes get mistaken for which. Rows are truth, columns are prediction.

What to look for:

- **Off-diagonal clusters** between two classes → they look alike; consider
  merging them, or adding examples that distinguish them.
- **A heavy "background" column** → defects missed entirely, which is a recall
  problem, not a confusion problem.
- **A heavy "background" row** → detections on empty surface, a precision problem.

For us, crazing's failure is mostly into background — the model does not
misclassify crazing as something else, it does not see it at all. That points at
contrast and annotation ambiguity, not class confusion, and it rules out
"add more classes" as a fix.

### ROC, and why it is not used here

An ROC curve plots true-positive rate against false-positive rate. It needs true
negatives, which detection does not have. **ROC is for classification;
precision-recall is for detection.** Knowing why is a good viva answer — ROC is
also misleading under heavy class imbalance, since a large TN count flatters the
false-positive rate.

Where ROC *would* apply in this project is PatchCore, which produces one score
per frame and is effectively a binary classifier. See `reports/anomaly.md`.

---

## 5. Mathematical intuition

**AP as an integral** over the precision-recall curve:

$$ AP = \int_0^1 p(r)\, dr \;\approx\; \sum_n (r_n - r_{n-1})\, p_{interp}(r_n) $$

where $p_{interp}(r) = \max_{\tilde{r} \ge r} p(\tilde{r})$ — interpolated
precision, taking the maximum precision at any recall at least $r$. The
interpolation removes the small zigzags caused by individual predictions and
makes AP stable.

**mAP** across $C$ classes and, for mAP@50-95, ten IoU thresholds:

$$ mAP = \frac{1}{C}\sum_{c=1}^{C} AP_c \qquad
   mAP_{50\text{-}95} = \frac{1}{10}\sum_{t \in \{.50,\dots,.95\}} mAP_t $$

Note that mAP averages classes **equally**, regardless of how many boxes each
has. Crazing's 678 boxes and inclusion's 1001 count the same. This is usually
what you want — it stops a common class from masking failure on a rare one —
but it means our overall 0.711 is dragged down by crazing exactly as much as it
is lifted by patches.

---

## 6. Software implementation

`scripts/evaluate.py` runs on the **test** split by default, and the docstring
says why:

```python
# Evaluated on the TEST split by default, not val. Val was used to pick the best
# epoch during training, so val mAP is mildly optimistic -- the model was
# selected on it. Test has been untouched until now.
```

The report it writes includes a plain-language section, because a table of
numbers is not a result until someone can act on it:

```
- Precision -- of the boxes the model reported, what fraction were real.
  Low precision means false alarms: good parts get ejected.
- Recall -- of the real defects, what fraction the model found.
  Low recall means escapes: defective parts reach the customer.
```

There is a second kind of verification in this project that is not an accuracy
metric at all. `scripts/parity.py` checks that the *deployed pipeline* agrees
with the *trained model*:

| model | ref boxes | our boxes | matched | mean IoU | class errors |
|-------|----------:|----------:|--------:|---------:|-------------:|
| model.onnx (FP32) | 132 | 132 | 132 | 1.0000 | 0 |
| model_int8.onnx | 132 | 133 | 122 | 0.9442 | 0 |

mAP tells you whether the model is good. Parity tells you whether what you
*deployed* is the model you *evaluated*. Both can fail independently, and only
the second catches a broken export or a wrong letterbox.

---

## 7. Hardware implementation

Metrics interact with the frame budget in two ways students usually miss.

**`conf_threshold` is a speed knob as well as an accuracy knob.** It is applied
to 2100 candidates before NMS. Raising it from 0.35 to 0.5 discards more
candidates, so NMS handles fewer boxes. We measured post-processing at 0.55 ms
with the current threshold; lowering it to chase crazing recall would raise both
the false-positive rate and the frame time.

**Quantisation changes the metrics, and must be re-measured, not assumed.** Our
INT8 experiment showed precision *rising* (0.643 → 0.763) while recall
*collapsed* (0.681 → 0.417). Anyone tracking only mAP would see a 33-point drop
and know something broke; anyone tracking only precision would think it improved.
The direction the two moved is what identified the cause as the confidence head.

---

## 8. Common mistakes

- **Reporting accuracy for a detector.** It has no true negatives.
- **Reporting only the mean.** Our 0.711 hides a 0.385.
- **Reporting validation instead of test.** Validation chose the best epoch.
- **Comparing mAP across datasets.** COCO mAP and NEU-DET mAP are different
  scales entirely.
- **Comparing mAP@50 with someone else's mAP@50-95.** Always state which.
- **Using ROC for detection.** No TN, and it flatters imbalanced problems.
- **Optimising F1 when the two errors have different costs.** For inspection
  they nearly always do.
- **Assuming quantisation preserves accuracy.** Ours cost 33 points before it
  was fixed, and 0.3 after.

---

## 9. Viva questions

**Q: What is mAP?**
Mean Average Precision. For each class, sweep the confidence threshold, plot
precision against recall, and take the area under that curve — that is AP. The
mean across classes is mAP. mAP@50 counts a prediction correct at IoU ≥ 0.5;
mAP@50-95 averages over ten IoU thresholds and so also grades box tightness.

**Q: Why not report accuracy?**
Accuracy requires true negatives, and detection has no meaningful count of
correctly-reported-nothing. It is also uninformative under class imbalance.
Precision, recall and mAP are the standard because they describe both error
types separately.

**Q: Your mAP@50 is 0.711 but mAP@50-95 is 0.407. Why the gap?**
The stricter metric grades how tightly boxes fit, up to IoU 0.95. Our defects —
crazing especially, and rolled-in scale — have genuinely ambiguous boundaries,
so the box is approximately right but rarely within 5% of the annotation. The
ratio is normal for surface-defect datasets.

**Q: Which class performs worst, and what would you do about it?**
Crazing: recall 0.222, mAP@50 0.385. It is a network of fine, low-contrast
thermal cracks with ambiguous extent, and it is the lowest-AP class in published
NEU-DET results too. Options in order of expected value: a per-class confidence
threshold, CLAHE contrast enhancement in preprocessing, oversampling crazing
during training, and PatchCore — which does not need to recognise crazing at all
to notice the surface is abnormal.

**Q: How would you tune the system for a customer who cannot tolerate escapes?**
Lower `detector.conf_threshold` in `configs/default.yaml`. That raises recall and
lowers precision — more good parts ejected, fewer defects shipped. The whole
policy is one number, and the precision-recall curve in `reports/` shows exactly
what each value costs.

**Q: How do you know the model you deployed is the model you evaluated?**
`scripts/parity.py`. It runs ultralytics on `best.pt` and our ONNX pipeline on
the same images and matches the boxes. FP32 matched all 132 reference boxes at
mean IoU 1.0000 with zero class errors, which verifies export, letterboxing,
decode, NMS and coordinate mapping together. mAP alone would not catch a broken
deployment.

---

## 10. Summary

1. Precision is "did I cry wolf", recall is "did I miss it". They trade against
   each other through one config value.
2. mAP is the area under the precision-recall curve, averaged over classes;
   mAP@50-95 additionally grades box tightness.
3. Report the per-class table. Our 0.711 average conceals crazing at 0.385 and
   recall 0.222.
4. Report test, not validation. Ours agree (0.711 vs 0.732), which is evidence
   against selection overfitting.
5. Accuracy and ROC do not apply to detection — no true negatives.
6. mAP measures the model; parity measures the deployment. You need both.

**Next:** [12 — ONNX and the runtime](12_onnx_and_runtime.md)
