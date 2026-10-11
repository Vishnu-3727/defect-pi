# 06 — Classification, Detection, Segmentation

**Prerequisites:** [04 — OpenCV](04_opencv.md), [05 — CNN fundamentals](05_cnn_fundamentals.md).

---

## 1. Concept

"Computer vision" is not one task. Four distinct tasks share the same
convolutional backbone and differ only in what comes out the other end.

```
 INPUT: one image of a steel strip with a scratch on the left

 ┌─────────────────────┬──────────────────────────────────────────────┐
 │ Image classification│ "scratches"                                  │
 │                     │ one label for the whole image                │
 ├─────────────────────┼──────────────────────────────────────────────┤
 │ Object detection    │ "scratches", 0.87, box (30, 44, 96, 120)     │
 │  <-- THIS PROJECT   │ label + rectangle, one per defect            │
 ├─────────────────────┼──────────────────────────────────────────────┤
 │ Semantic            │ a per-pixel mask: which pixels are "scratch" │
 │ segmentation        │ two scratches merge into one region          │
 ├─────────────────────┼──────────────────────────────────────────────┤
 │ Instance            │ per-pixel mask, per object                   │
 │ segmentation        │ scratch #1 and scratch #2 stay separate      │
 └─────────────────────┴──────────────────────────────────────────────┘
```

Everything above the line is cheap. Everything below is expensive, both in
compute and in labelling effort.

---

## 2. Why it is needed

You must choose one, and the choice determines the dataset you need, the model
you train, the annotation hours you spend, and the frame rate you achieve. It
is the highest-leverage decision in the project, and it is made before any code
is written.

Choosing wrongly is expensive in a specific way: **the annotation format is not
convertible upward.** Boxes can be derived from masks, but masks cannot be
derived from boxes. If you label 1770 images with boxes and later decide you
need segmentation, you label them all again.

---

## 3. Real-world example

The task follows from what the downstream *action* requires:

| Factory question | Task needed | Why |
|------------------|-------------|-----|
| "Reject this part, yes or no?" | Classification | The ejector needs one bit |
| "Where is the defect, and what kind?" | Detection | Operator sees it; defect type drives process correction |
| "What fraction of the surface is corroded?" | Semantic segmentation | Area is a per-pixel quantity |
| "How many separate cracks, and how long is each?" | Instance segmentation | Counting and measuring need object identity |
| "Is this part unlike any good part?" | Anomaly detection | No labelled defects exist yet |

**Our system ejects defective parts and shows the operator what was wrong.**
Detection is the smallest task that supports both. Classification would eject
without explaining. Segmentation would explain in more detail than an ejector
can use, at several times the cost.

A rolling mill genuinely does want segmentation eventually — defect *area* per
square metre is a process quality metric — but that is a later product, not a
semester project.

---

## 4. How it works

### The shared structure

All four tasks use the same backbone from Module 05. Only the head differs.

```
                     ┌──────────────────────────► [ dense layer ] ──► one label
                     │                                              CLASSIFICATION
                     │
 image ──► [ CNN backbone ] ──► feature maps
                     │
                     ├──────────────────────────► [ conv head ] ──► boxes + classes
                     │                                              DETECTION
                     │
                     └──────────────────────────► [ upsampling decoder ] ──► per-pixel map
                                                                            SEGMENTATION
```

This is why transfer learning works across tasks (Module 07): a backbone
trained for classification on millions of images is a good starting point for
detection on 1236.

### Image classification

Output: a vector of class probabilities, one per class, summing to 1.

```
[0.02, 0.01, 0.05, 0.03, 0.02, 0.87]   ->  "scratches"
```

Cheapest to run and cheapest to label — one word per image. NEU-DET has a
classification variant used exactly this way. Its weakness for us: it assumes
one label per image, and a real strip can carry two defect types at once.

### Object detection

Output: a variable-length list. Each entry is
`(class, confidence, x1, y1, x2, y2)`.

The variable length is the hard part. A network produces a fixed-size output
tensor, but an image may contain zero defects or nine. Detectors resolve this by
predicting a fixed grid of candidate boxes — thousands of them — each with a
confidence, then discarding the low-confidence ones and merging duplicates with
Non-Maximum Suppression. Module 08 covers this in full.

Labelling cost: draw a rectangle per defect. Roughly 10 to 20 seconds per image.
For 1770 images, about 6 hours of human work. This dataset came pre-labelled;
that labour was done for us and is worth acknowledging honestly.

### Semantic segmentation

Output: one array the size of the image, each pixel holding a class index.

Labelling cost: paint every defect pixel. Roughly 2 to 10 minutes per image.
For 1770 images that is **60 to 300 hours**. This alone rules it out for a
semester project, before any discussion of model speed.

Semantic segmentation cannot separate two touching objects of the same class —
both are simply "scratch pixels".

### Instance segmentation

Output: for each detected object, a class, a box, and a pixel mask.

The most informative and most expensive. YOLOv8 has a segmentation variant
(`yolov8n-seg`) that adds a mask head to the same backbone; it costs roughly 10
to 20 percent more inference time. The blocker is not the model, it is the
labelling.

### Where anomaly detection fits

None of the four. Anomaly detection answers "is this unlike normal?" and needs
**no defect labels at all** — only images of good parts. It is a fifth category,
orthogonal to the others, and this project uses it as a second opinion because
it can flag a defect type nobody has ever labelled. Module 14.

---

## 5. Mathematical intuition

The tasks differ in the shape of the output space and therefore in the loss.

**Classification** — output is a probability vector over $C$ classes;
loss is cross-entropy:

$$ L = -\sum_{c=1}^{C} y_c \log \hat{y}_c $$

**Detection** — output is a set, so the loss must combine three things: is
there an object, which class, and where. YOLOv8's total loss is a weighted sum:

$$ L = \lambda_{box} L_{box} + \lambda_{cls} L_{cls} + \lambda_{dfl} L_{dfl} $$

You see exactly these three numbers scrolling past during training as
`box_loss`, `cls_loss`, `dfl_loss`. Module 10 explains each.

**Segmentation** — cross-entropy applied per pixel, averaged over all
$H \times W$ positions, often combined with a Dice term to handle the fact that
defect pixels are vastly outnumbered by background pixels:

$$ L = \frac{1}{HW}\sum_{y,x} \text{CE}\big(y_{y,x}, \hat{y}_{y,x}\big) $$

The output space grows from $C$ numbers, to roughly $5N$ numbers for $N$
objects, to $H \times W \times C$ numbers. Cost tracks that growth.

---

## 6. Software implementation

The choice is visible throughout the repo:

| File | What it assumes |
|------|----------------|
| `scripts/prepare_dataset.py` | Converts VOC **boxes** to YOLO **boxes**; a mask format would need different code entirely |
| `data/yolo/*/labels/*.txt` | Five numbers per line: `class cx cy w h`. This is the detection format |
| `src/inspector/detector.py` | Decodes box coordinates and runs NMS |
| `src/inspector/decision.py` | Consumes a list of boxes, applies a minimum-area rule |
| `src/inspector/anomaly.py` | Consumes a whole frame and returns a single score — the orthogonal task |

A label file for a detection dataset looks like this:

```
5 0.487500 0.512500 0.235000 0.910000
```

Class 5 (scratches), centre at 48.75% across and 51.25% down, 23.5% of the
width and 91% of the height. Five numbers. A segmentation label for the same
image would be a polygon with dozens of vertices, or a PNG mask the size of the
image.

---

## 7. Hardware implementation

Approximate cost on a Pi 5 CPU at 320x320, from the architecture rather than
measurement — Module 13 gives the measured numbers for the one we chose:

| Task | Model | Relative inference cost | Output size |
|------|-------|------------------------|-------------|
| Classification | ResNet-18 | 1.0x | 6 floats |
| **Detection** | **YOLOv8n** | **~1.3x** | **~2100 candidate boxes, then NMS** |
| Instance segmentation | YOLOv8n-seg | ~1.5x | boxes + 160x160 mask prototypes |
| Semantic segmentation | U-Net | ~3x | 320x320x6 |

Note that detection's post-processing is not free. NMS runs on the CPU in
Python/NumPy, over thousands of candidates, and on an underpowered device it
can rival the network itself. Raising `conf_threshold` cuts candidates before
NMS sees them, which is a real speed knob, not just an accuracy one.

The segmentation column also hides a memory problem: a 320x320x6 float32 output
is 2.4 MB per frame versus about 50 KB for detection. On a shared-memory device
that bandwidth is not free.

---

## 8. Common mistakes

- **Saying "image classification" in the report when the model outputs boxes.**
  Examiners notice, and it is the easiest correction to make.
- **Labelling with boxes and later needing masks.** Non-recoverable; you relabel.
- **Assuming segmentation is "more accurate".** It is more *detailed*. A
  segmentation model can have worse defect-level recall than a detector.
- **Ignoring NMS cost when budgeting frame time.** It is CPU work outside the
  neural network and does not appear in FLOP counts.
- **Treating anomaly detection as a kind of classification.** It is trained
  without defect labels at all, which is precisely its value.
- **Choosing the task from what sounds impressive rather than from what the
  actuator needs.** The solenoid consumes one bit. Everything beyond that bit
  exists for the human operator, and should be justified on those grounds.

---

## 9. Viva questions

**Q: Why did you choose object detection rather than classification?**
The system must both eject the part and tell the operator what was wrong and
where. Classification gives a single label with no location, so the operator
cannot verify the decision and the process engineer cannot tell whether the
defect is edge-biased or centre-biased. Detection is the smallest task that
supports both requirements.

**Q: Why not segmentation, which is more precise?**
Annotation cost, primarily. Boxes take 10 to 20 seconds per image; masks take
2 to 10 minutes. For 1770 images that is roughly 6 hours versus 60 to 300
hours. Inference is also 1.5 to 3 times slower, against a fixed frame budget.
Since the ejector consumes a binary decision, the extra pixel precision would
buy nothing operationally. It is documented as future work for defect-area
metrics.

**Q: What is the difference between semantic and instance segmentation?**
Semantic labels each pixel with a class, so two touching scratches become one
region. Instance segmentation keeps them as separate objects with individual
masks, which you need if you want to count defects or measure each one.

**Q: Your system uses both YOLOv8 and PatchCore. Are they doing the same job?**
No. YOLOv8 is supervised detection: it recognises the six defect types it was
trained on and localises them. PatchCore is unsupervised anomaly detection: it
learns only what a good part looks like and flags anything unlike it. YOLOv8
cannot detect a seventh defect type; PatchCore can, but cannot name it.

**Q: If a part has two different defects, what happens?**
Detection handles it naturally — it returns a list, so both appear with their
own class and box. A classification model would be forced to output one label,
which is a concrete reason detection suits real inspection better.

---

## 10. Summary

1. Four tasks share one backbone and differ only in the head and the output shape.
2. Detection was chosen because the system needs a location and a class, and
   nothing more.
3. Annotation cost, not model capability, is what rules out segmentation here:
   about 6 hours versus 60 to 300.
4. Label formats do not convert upward. Boxes come from masks; masks never come
   from boxes.
5. Anomaly detection is a fifth, orthogonal category needing no defect labels —
   which is exactly why this project runs it alongside the detector.

**Next:** [07 — PyTorch](07_pytorch.md)
