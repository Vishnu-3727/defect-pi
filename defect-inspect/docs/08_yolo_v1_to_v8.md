# 08 — YOLO, v1 to v8

**Prerequisites:** [05 — CNN fundamentals](05_cnn_fundamentals.md),
[06 — CV task taxonomy](06_cv_task_taxonomy.md).

---

## 1. Concept

YOLO — *You Only Look Once* — is a family of object detectors whose defining
idea is that detection should be **one forward pass**, not a search.

Before YOLO, detectors worked by proposing thousands of candidate regions and
classifying each one. YOLO instead divides the image into a grid and has every
grid cell predict boxes directly. One pass, one output tensor, done.

That single design choice is why a 3-million-parameter model can inspect a
conveyor on a Raspberry Pi.

---

## 2. Why it is needed

The alternative, R-CNN and its descendants, runs in two stages: propose regions,
then classify each. It is accurate and it is slow — Faster R-CNN reaches roughly
5 to 10 FPS on a desktop GPU. Your project's own Review 1 base-paper table lists
this as the reason it was rejected.

Our budget is a conveyor moving parts past a camera, and a CPU with no GPU. Two
stages is not affordable. One stage is.

The trade is real: single-stage detectors have historically been slightly less
accurate, especially for small or overlapping objects. YOLOv8 has closed most of
that gap, and for our defect sizes (median 11.8% of image area) the gap does not
apply at all.

---

## 3. Real-world example

The name is the idea. A human inspector does not scan a strip of steel by
examining ten thousand rectangles. They look once, and the defect is where it
is. YOLO formalises that: every position in the image is examined exactly once,
in parallel, by the same network.

---

## 4. How it works

### The history, and what each version actually changed

| Version | Year | The change that mattered | Consequence |
|---------|------|-------------------------|-------------|
| **v1** | 2016 | Grid-based single-pass detection. 7x7 grid, 2 boxes per cell | 45 FPS, but poor at small and clustered objects |
| **v2** | 2017 | **Anchor boxes** — predict offsets from preset shapes, not raw sizes. Batch norm. Higher resolution | Much better recall; anchors became standard for six years |
| **v3** | 2018 | **Three detection scales** with a feature pyramid. Independent per-class sigmoid instead of softmax | Small objects finally worked; multi-label became possible |
| **v4/v5** | 2020 | Engineering, not new theory: mosaic augmentation, CSP blocks, better training recipes. v5 brought a usable PyTorch codebase | Accessible to non-researchers |
| **v6/v7** | 2022 | Reparameterised blocks, richer training-time supervision | Incremental accuracy at equal speed |
| **v8** | 2023 | **Anchor-free**. Decoupled head. Distribution Focal Loss for box regression | Simpler, no anchor tuning, better small-object localisation |

**Why v8 for this project:** anchor-free removes a dataset-specific
hyperparameter we would otherwise have to tune for NEU-DET's box shapes; it
ships as a maintained library with ONNX export built in; and the `n` variant is
the smallest credible detector available. Version 9 through 12 exist and are
marginally better on COCO, but v8 has the deepest deployment tooling and the
most published NEU-DET baselines to compare against — which matters more for a
project that has to be defended than half a point of mAP.

### Anchors, and why anchor-free is simpler

**With anchors (v2 to v7):** you define, say, nine box shapes up front. Each
grid cell predicts adjustments relative to each shape. Good shapes make training
easy; bad shapes make it hard. The shapes are usually found by running k-means
on your dataset's box dimensions — a step you must remember to redo for every
new dataset.

**Anchor-free (v8):** each grid cell predicts the four distances from itself to
the box edges. No presets, no k-means, no dataset-specific tuning. One fewer
thing to get wrong.

```
          anchor-free prediction from one grid cell

                    ^ top
                    |
       left  <---- (.) ----> right          four distances,
                    |                        directly regressed
                    v bottom
```

### The three parts of the network

```
 image 320x320x3
        |
   [ BACKBONE ]   CSPDarknet. Extracts features, downsampling five times.
        |         Outputs three feature maps at strides 8, 16, 32.
        |
   [   NECK   ]   PAN-FPN. Mixes those three maps so each carries both
        |         fine detail and broad context.
        |
   [   HEAD   ]   Three parallel detection heads, one per scale.
        |         Decoupled: separate branches for class and for box.
        v
   (1, 10, 2100)  4 box values + 6 class scores, over 2100 positions
```

**Backbone** — the feature extractor from Module 05. Spatial size shrinks,
channel count grows.

**Neck** — the part that solves the problem raised in Module 05. Deep feature
maps know *what* but have lost *where*; shallow maps know *where* but not
*what*. The FPN passes semantic information downward from deep to shallow, and
the PAN passes spatial information back upward. Every scale ends up with both.

**Head** — "decoupled" means classification and box regression have separate
convolutional branches. They are different tasks — one wants
translation-invariant features, the other translation-sensitive ones — and
sharing a branch forces a compromise.

### The output tensor, for our model exactly

```
(1, 10, 2100)
 |   |    |
 |   |    +--- 2100 positions = 40x40 (stride 8) + 20x20 (stride 16) + 10x10 (stride 32)
 |   |                        = 1600 + 400 + 100
 |   +-------- 10 rows = 4 box values + 6 class scores
 +------------ batch of 1
```

Three scales exist so that a large defect is caught by the coarse 10x10 map
where each cell sees a lot of context, and a small one by the fine 40x40 map.

**There is no objectness score.** v3 through v7 predicted "is there an object
here at all" separately from "which class". v8 dropped it; the class score is
the confidence. This is why `detector.py` reads rows 4 onward as class scores
directly with no extra multiplication — a detail that silently halves your
confidences if you carry over v5 decoding code.

### IoU and NMS

**Intersection over Union** measures overlap between two boxes:

$$ \text{IoU} = \frac{\text{area of intersection}}{\text{area of union}} $$

0 means disjoint, 1 means identical. It appears in three distinct places, and
conflating them causes confusion:

1. In training, to decide which predictions are responsible for which truth.
2. In **NMS**, to decide whether two predictions describe the same defect.
3. In **evaluation**, to decide whether a prediction counts as correct
   (mAP@50 uses a threshold of 0.5).

**Non-Maximum Suppression** removes duplicates. The network happily predicts the
same scratch from four neighbouring cells:

```
before NMS                         after NMS
 ┌───────────┐                      ┌───────────┐
 │ ┌─────────┼─┐   0.91             │           │  0.91
 │ │┌────────┼─┼─┐ 0.88     ---->   │           │
 └─┼┼────────┘ │ │ 0.85             └───────────┘
   └┼──────────┘ │ 0.79
    └────────────┘
```

The algorithm: sort by confidence, take the highest, delete everything
overlapping it by more than the IoU threshold, repeat.

Our `detector.py` calls `cv2.dnn.NMSBoxesBatched` — the *batched* variant does
NMS per class, so a scratch box and an inclusion box that happen to overlap do
not suppress each other. Using the non-batched version would silently drop real
detections when two defect types coincide.

Two implementation details from our code worth knowing:

- **NMS runs in letterboxed coordinates, before mapping back.** IoU is a ratio,
  so it is scale-invariant, and doing it in this order means fewer boxes go
  through the coordinate transform.
- **NMS is CPU work outside the neural network.** It does not appear in FLOP
  counts. We measured 0.55 ms per frame, small only because
  `conf_threshold: 0.35` discards all but a couple of dozen candidates from 2100
  first. Lowering that threshold raises NMS cost as well as false positives.

### Model sizes

| Model | Parameters | GFLOPs @640 | COCO mAP@50-95 | Suitable for a Pi? |
|-------|-----------:|------------:|---------------:|--------------------|
| **YOLOv8n** | **3.2 M** | **8.7** | 37.3 | **yes** |
| YOLOv8s | 11.2 M | 28.6 | 44.9 | marginal |
| YOLOv8m | 25.9 M | 78.9 | 50.2 | no |
| YOLOv8l | 43.7 M | 165.2 | 52.9 | no |
| YOLOv8x | 68.2 M | 257.8 | 53.9 | no |

Our fused model reports **3,006,818 parameters and 2.0 GFLOPs at 320x320**.

Why `n`: FLOPs scale roughly linearly with inference time on a CPU. YOLOv8s
would cost about 3.3x the compute for a few points of mAP — and on NEU-DET,
where our limiting factor is crazing's inherent low contrast rather than model
capacity, those points are not there to be had. Bigger would be slower for
nothing.

---

## 5. Mathematical intuition

**Box decoding, anchor-free.** For a grid cell at $(c_x, c_y)$ with predicted
distances $(l, t, r, b)$ to the four edges:

$$ x_1 = c_x - l, \quad y_1 = c_y - t, \quad x_2 = c_x + r, \quad y_2 = c_y + b $$

**Distribution Focal Loss.** v8 does not regress each distance as a single
number. It predicts a probability distribution over 16 discrete bins and takes
the expectation:

$$ \hat{d} = \sum_{i=0}^{15} i \cdot p_i $$

Two benefits: the network can express uncertainty about an edge, which matters
when a defect's boundary is genuinely ambiguous — and NEU-DET's are; and the
expectation of a distribution gives sub-pixel precision from discrete bins.

This is also the mechanism that quantised so badly in our INT8 experiment.
Rounding those 16 probabilities to 8-bit steps coarsens every box edge, which is
one of the three reasons the head is now excluded from quantisation. See
`reports/benchmark.md`.

**The total loss** combines three terms:

$$ L = \lambda_{box} L_{CIoU} + \lambda_{cls} L_{BCE} + \lambda_{dfl} L_{DFL} $$

These are the `box_loss`, `cls_loss` and `dfl_loss` columns printed during our
training run.

---

## 6. Software implementation

The decode in `src/inspector/detector.py`, which is the part we wrote by hand
and therefore the part most likely to be wrong:

```python
preds = raw[0].T                        # (1, 10, 2100) -> (2100, 10)
boxes_cxcywh = preds[:, :4]
scores_all   = preds[:, 4:]             # no objectness in v8

class_ids   = scores_all.argmax(axis=1)
confidences = scores_all[np.arange(len(class_ids)), class_ids]
np.clip(confidences, 0.0, 1.0, out=confidences)

keep = confidences >= self.conf_threshold   # 2100 -> a few dozen
```

Thresholding before anything else is what keeps post-processing at 0.55 ms:
every later operation runs on tens of rows, not 2100.

**We proved this decode correct rather than assuming it.**
`scripts/parity.py` compares our output against ultralytics' own predictions:

| model | ref boxes | our boxes | matched | mean IoU | class errors |
|-------|----------:|----------:|--------:|---------:|-------------:|
| model.onnx (FP32) | 132 | 132 | 132 | **1.0000** | 0 |

A perfect match on all 132 boxes. Letterboxing, transpose, argmax, NMS and the
inverse coordinate mapping are all correct — none of which raises an error when
wrong.

---

## 7. Hardware implementation

Measured on this project (x86 host, so treat absolutes as architecture-specific
— see `reports/benchmark.md`):

| Stage | ms | Where it runs |
|-------|---:|---------------|
| Preprocess | 2.6 | OpenCV, single-threaded |
| Inference | 16.5 | ONNX Runtime, 4 threads |
| NMS + decode | 0.55 | NumPy and OpenCV |

Two hardware-relevant observations:

- **Inference is the only stage that threads.** Preprocessing stays flat at
  ~2.6 ms regardless of thread count, so it becomes a larger fraction of the
  budget as the model gets faster. On a Pi it will matter more.
- **The three detection scales have different costs.** The stride-8 map has
  1600 of the 2100 positions and dominates. Dropping it would be a large speedup
  and would destroy small-defect recall — which for our 11.8%-median boxes might
  actually be affordable, and is worth measuring if the Pi disappoints.

---

## 8. Common mistakes

- **Applying an objectness multiplication in v8.** There is no objectness. Code
  ported from v5 halves every confidence.
- **Using non-batched NMS.** Overlapping boxes of different classes suppress
  each other, silently deleting real detections.
- **Forgetting to undo the letterbox.** Boxes land in the wrong place. Nothing
  errors. Our `preprocess.demo()` asserts the round trip for this reason.
- **Assuming `(1, 10, 2100)` is `(1, 2100, 10)`.** Transposing wrongly produces
  boxes from class scores and confidences from coordinates. Output is garbage,
  no exception.
- **Choosing a bigger model to fix a data problem.** Our crazing recall is 0.222
  because crazing is faint and ambiguously bounded, not because the model is
  small.
- **Quoting COCO mAP as if it applied to your dataset.** COCO mAP@50-95 of 37.3
  says nothing about steel defects.

---

## 9. Viva questions

**Q: Why is it called You Only Look Once?**
Because detection is a single forward pass over the whole image, rather than
proposing regions and classifying each one separately. Every position is
examined exactly once, in parallel.

**Q: What is the difference between YOLO and Faster R-CNN?**
Faster R-CNN is two-stage: a region proposal network suggests candidates, then a
second network classifies them. Accurate but roughly 5 to 10 FPS on a desktop
GPU. YOLO is single-stage and predicts everything in one pass. We need real-time
inference on a CPU, which rules out two stages.

**Q: What does anchor-free mean and why does it help?**
Anchor-based versions predict offsets from preset box shapes that must be tuned
per dataset, usually by k-means over the training boxes. YOLOv8 predicts the
four distances from each grid cell to the box edges directly. It removes a
dataset-specific hyperparameter and improves localisation of unusually shaped
objects.

**Q: Explain NMS and IoU.**
IoU is intersection area over union area, measuring how much two boxes overlap.
NMS uses it to remove duplicates: sort predictions by confidence, keep the
highest, delete everything overlapping it beyond a threshold, repeat. We use the
per-class variant so different defect types cannot suppress each other.

**Q: Why YOLOv8n rather than YOLOv8s?**
YOLOv8s costs about 3.3x the compute for a few mAP points, and our frame budget
is four ARM cores shared with a second model. Our accuracy ceiling is set by
crazing's low contrast rather than by model capacity, so the extra parameters
would buy latency, not accuracy.

**Q: How do you know your ONNX decoding is correct?**
`scripts/parity.py` runs ultralytics on `best.pt` and our ONNX pipeline on the
same 60 images and matches the boxes. The FP32 model matched all 132 reference
boxes at mean IoU 1.0000 with zero class errors, which verifies letterboxing,
tensor layout, decode, NMS and the inverse coordinate mapping together.

---

## 10. Summary

1. One forward pass over a grid, not a region search — that is the whole idea,
   and it is what makes CPU deployment possible.
2. v8's anchor-free head removes per-dataset anchor tuning; DFL gives sub-pixel
   box edges and is the part that quantises worst.
3. Backbone extracts, neck fuses fine detail with broad context, decoupled head
   predicts class and box separately.
4. Our output is `(1, 10, 2100)`: 4 box values plus 6 class scores over 2100
   positions, no objectness.
5. YOLOv8n at 3.0 M parameters and 2.0 GFLOPs is chosen because FLOPs are the
   frame budget, and our accuracy ceiling is set by the data, not the model.

**Next:** [09 — The NEU-DET dataset](09_dataset_neu_det.md)
