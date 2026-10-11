# 14 — PatchCore and Unsupervised Anomaly Detection

**Prerequisites:** [05 — CNN fundamentals](05_cnn_fundamentals.md),
[12 — ONNX and the runtime](12_onnx_and_runtime.md).

---

## 1. Concept

Every model so far has been **supervised**: you showed it 1,800 labelled images
of six named defects, and it learned to draw boxes around those six things.

PatchCore asks a different question. Instead of *"which of my six defects is
this?"* it asks *"does this look like anything I have ever seen on a good
part?"* It is trained on **defect-free images only**, memorises what normal
looks like, and at inference reports a distance — how far this image is from
anything normal.

It cannot name a defect. It can only say *unlike normal, by this much*.

---

## 2. Why it is needed

YOLOv8 has a hard boundary: it detects the six classes in NEU-DET and nothing
else. Present it with a burn mark, a paint smear, a drill hole, or a defect type
that appeared last week when a new supplier's steel arrived, and it will
confidently output nothing. **An empty prediction from a detector is
indistinguishable from a good part.**

That is the failure mode that costs money. A defect the model was never trained
on passes inspection with full confidence.

PatchCore closes that hole from the other direction. It never needed to see the
defect. It only needs the surface to differ from normal.

Two systems with complementary blind spots:

| | YOLOv8n | PatchCore |
|---|---|---|
| Needs labelled defects | yes | **no** |
| Names the defect | yes | no |
| Detects unseen defect types | **no** | yes |
| Localises | box | coarse heat map |
| Cost per frame (Pi 5) | 27 ms | ~90 ms |

---

## 3. Real-world example

Semiconductor wafer inspection is the canonical case. A fab produces millions of
essentially identical dies, and the overwhelming majority are good. Collecting
thousands of labelled examples of every possible defect is impossible — some
defect modes appear once a year when a specific tool drifts.

So the industry inverted the problem: model the *good* die, which you have in
effectively unlimited supply, and flag deviation. The same logic drives the
MVTec AD benchmark, which is built entirely from defect-free training sets, and
on which PatchCore was state of the art when published (Roth et al., 2022).

The economics are the point. **Good parts are free and abundant. Labelled
defects are expensive and rare.**

---

## 4. How it works

Four stages. Two happen once, offline; two happen per frame.

### Stage 1 — Embed (offline and online)

Push the image through a pretrained CNN and stop partway. We use **ResNet-18**
and take the outputs of `layer2` and `layer3`.

Why the middle and not the end? The last layer of a classifier has learned
*"this is a cat"* — too abstract, and specific to ImageNet's classes. The first
layer has learned edges — too generic to distinguish anything. The middle layers
carry **texture**: the scale at which a steel surface's mill finish lives.

`layer2` sees fine texture; `layer3` sees coarser structure. `layer3` is
upsampled to `layer2`'s grid and the two are concatenated, so each grid cell
carries both scales:

```
input 256x256x3
  -> layer2: 32x32x128
  -> layer3: 16x16x256  --upsample--> 32x32x256
  -> concat: 32x32x384
  -> 3x3 average pool, stride 2 -> 16x16x384
```

The 3×3 average pool is not compression for its own sake. It makes each patch
descriptor depend on its neighbours, so a single noisy activation cannot create
an anomaly on its own.

Result: **256 patches of 384 numbers each** per image.

### Stage 2 — Build the memory bank (offline, once)

Run every defect-free training image through Stage 1 and keep all the patch
vectors. For 1,000 images that is 256,000 vectors × 384 dimensions × 4 bytes =
**393 MB**. Far too much to hold, and far too slow to search.

### Stage 3 — Coreset subsampling (offline, once)

Most of those 256,000 vectors are near-duplicates — flat mill finish looks like
flat mill finish. A **greedy k-centre coreset** keeps a small subset that still
covers the space:

```
pick one vector at random
repeat until you have k:
    for every remaining vector, find its distance to the nearest chosen vector
    choose the vector whose nearest-chosen-distance is largest
```

Each step picks the point currently worst represented. The result is a spread
sample, not a random one — random sampling would over-represent the flat regions
and drop the rare-but-normal features like grain boundaries.

Typically 1% is kept. Our bank ended at **1.1 MB**.

### Stage 4 — Score (online, per frame)

For each of the 256 patches in the new image, find the distance to its nearest
neighbour in the bank. The image score is the **maximum** over patches.

Maximum, not mean — this matters enormously and is covered in section 8.

---

## 5. Mathematical intuition

### The distance trick

The naive nearest-neighbour computation builds a difference tensor: for 256
query patches against a 20,000-vector bank in 384 dimensions, that is
256 × 20,000 × 384 floats = **7.9 GB**. On a Pi with 8 GB total, that is fatal.

Expand the square instead:

$$\|a - b\|^2 = \|a\|^2 - 2\,a{\cdot}b + \|b\|^2$$

- $\|b\|^2$ — one number per bank vector, computed once when the bank loads.
- $\|a\|^2$ — one number per query patch, 256 numbers.
- $a{\cdot}b$ — a single matrix multiply, `(256, 384) @ (384, 20000)`.

Peak memory becomes the 256 × 20,000 result matrix: **20 MB**, and BLAS runs the
matmul at full NEON throughput. This is the single optimisation that makes
PatchCore possible on a Pi, and it is implemented in `anomaly.py`.

### Why random projection is safe

PatchCore optionally projects 384 dimensions down before the coreset search. The
**Johnson–Lindenstrauss lemma** says that projecting $n$ points into
$O(\log n / \epsilon^2)$ dimensions with a random matrix preserves all pairwise
distances within a factor of $1 \pm \epsilon$.

The consequence is counter-intuitive but load-bearing: the target dimension
depends on the *number of points*, not the original dimension. You can go from
384 to 128 dimensions and keep distances almost exactly, which speeds the
coreset construction by 3× at no measurable cost in accuracy.

### Why maximum and not mean

A defect occupies a small fraction of the image. Suppose 4 of 256 patches are
anomalous with distance 8.0 and the other 252 are normal at 2.0:

$$\text{mean} = \frac{4(8.0) + 252(2.0)}{256} = 2.09 \qquad \text{max} = 8.0$$

The mean moved 0.09 from the normal value of 2.0. The signal is **diluted by the
252 patches that are working correctly**. The larger the image relative to the
defect, the worse this gets. Maximum is the only aggregation that survives.

---

## 6. Software implementation

`src/inspector/anomaly.py`. The backbone is exported to ONNX by
`scripts/build_patchcore.py` so the runtime never imports torch.

```python
def score(self, frame):
    patches = self._embed(frame)              # (256, 384)
    q = (patches ** 2).sum(axis=1)            # ||a||^2
    dots = patches @ self.bank.T              # (256, N)
    d2 = q[:, None] - 2.0 * dots + self.bank_sq[None, :]
    np.maximum(d2, 0.0, out=d2)               # kill FP negatives
    return float(np.sqrt(d2.min(axis=1)).max())
```

`np.maximum(d2, 0.0)` is not defensive padding. The expansion is algebraically
non-negative but numerically is a difference of similar magnitudes, so it
produces values like `-1e-7` for identical vectors. `sqrt` of that is `nan`, and
a single `nan` propagates through `min` and silently disables the module.

### The gate

PatchCore costs ~90 ms on the Pi against the detector's 27 ms. Running it every
frame would take the system from 36.6 FPS to about 8.5. So it does not run every
frame:

```python
def maybe_score(self, frame, detections):
    top = max((d.confidence for d in detections), default=0.0)
    if top >= self.gate_conf_below:      # detector is sure -> trust it
        return None
    if self._calls_this_second >= self.max_calls_per_second:
        return None                      # hard budget, never negotiable
    return self.score(frame)
```

Two independent guards. The first is semantic: a confident detection needs no
second opinion. The second is a hard cap that holds even if every frame is
ambiguous — without it, a badly lit batch would collapse the frame rate at
exactly the moment inspection matters.

Measured: 14 anomaly calls across 139 frames, about 10%.

---

## 7. Hardware implementation

Three things the rig determines, all of which dominate the algorithm:

**Lighting must be constant.** PatchCore measures distance from normal. If
normal was photographed at 09:00 with the window open and inspection runs at
17:00 under fluorescents, every part is an anomaly. This is why the design
specifies a **ring light and an enclosure**, not ambient light.

**Working distance and focus must be fixed.** A patch descriptor is
scale-sensitive. Move the camera 5 cm and the mill finish changes spatial
frequency, and the bank no longer describes it.

**The bank is specific to the part.** A bank built on cold-rolled sheet does not
transfer to galvanised. Every new product line needs its own bank — which is
cheap, because it only needs good parts.

Rebuild whenever the lighting, the camera position, or the material changes.

---

## 8. Common mistakes

**Building the bank from data that contains defects.** The bank defines normal.
Contaminate it and the defect becomes normal.

*This project is guilty of a compromise here, and it is stated openly.* NEU-DET
contains **no defect-free images at all** — every one of the 1,800 has at least
one annotated defect. The bank was therefore built from image regions *outside*
bounding boxes dilated by 1.3×, giving 63,476 usable patches from 153,600
(41.3%). Three consequences:

1. Unannotated defects in "normal" regions are now in the bank.
2. NEU-DET boxes are loose, so the dilation may exclude genuinely normal area.
3. The bank never sees defect-adjacent context, which is where subtle
   transitions live.

Measured result: **normal median 2.261, defect median 2.402, separation only
+0.141**, with heavily overlapping distributions. This is weak. It is reported
in `reports/anomaly.md` rather than buried, and the threshold of **2.77** is the
95th percentile of the normal distribution, calibrated from measurement rather
than guessed.

The fix is not algorithmic. It is **photographing 200 known-good parts on the
real rig**, and it is the first task once the hardware exists.

**Using the mean instead of the maximum.** See section 5. This one silently
halves your detection rate and produces no error.

**Guessing the threshold.** The score is an unbounded distance in an arbitrary
feature space. `2.77` means nothing on its own — it means the 95th percentile of
*this* bank under *this* lighting. Recalibrate whenever the bank is rebuilt:
`python -m inspector.anomaly` prints the distribution.

**Using the final CNN layer.** Too abstract and ImageNet-specific. Texture lives
in the middle.

**Forgetting the module can be absent.** `build_anomaly_detector()` returns
`None` when the bank file is missing, and the pipeline runs on the detector
alone. That is deliberate — a missing bank should degrade the system, not stop
the line — but it is a *silent* downgrade, which is why the artefacts are now
committed to git.

---

## 9. Interview questions

**Q: What is the difference between anomaly detection and object detection?**
Object detection is supervised multi-class localisation: it needs labelled
examples of each class and outputs class plus box. Anomaly detection is
unsupervised one-class: it models only the normal distribution and outputs a
scalar deviation. The practical difference is what happens with an unseen defect
— the detector outputs nothing, the anomaly model still flags it.

**Q: Why does PatchCore need only good images?**
Because it does not learn a decision boundary between two classes. It builds a
description of one class and measures distance from it. That is also its
limitation: it can say *different*, never *why*.

**Q: What is a coreset and why not sample randomly?**
A coreset is a subset chosen so that every original point is close to some
selected point — greedy k-centre minimises the maximum such distance. Random
sampling reproduces the *density* of the data, which over-represents common
patterns and can drop rare-but-normal features entirely, causing false alarms on
legitimate parts.

**Q: Your memory bank is 1.1 MB from 393 MB of features. What was lost?**
Redundancy, mostly. The greedy k-centre keeps coverage rather than density, so
the discarded vectors are the ones already within a small radius of a kept one.
What is genuinely lost is fine-grained density information — the bank knows what
normal *can* look like but not how *often*.

**Q: Why is your separation only 0.141?**
Because the bank was built from a dataset with no defect-free images, using
regions outside dilated boxes as a proxy. It is a documented compromise. The
module is a backstop for unknown defect types, not a second accurate classifier,
and the fix is data collection, not a different algorithm.

**Q: Why not run it on every frame?**
90 ms against the detector's 27 ms. Every frame would give roughly 8.5 FPS
against a 20 FPS requirement. It runs only when the detector's top confidence is
below 0.55, capped at 3 calls per second.

---

## 10. Summary

- PatchCore models **normal** and reports distance from it, so it needs no
  labelled defects and catches defect types the detector never saw.
- Four stages: embed with mid-level ResNet-18 features, collect a bank, shrink
  it with a greedy k-centre coreset, score by nearest-neighbour distance.
- Score is the **maximum** over patches. The mean dilutes a small defect into
  invisibility.
- The distance expansion $\|a-b\|^2 = \|a\|^2 - 2ab + \|b\|^2$ turns 7.9 GB into
  20 MB and one matmul. Without it this does not run on a Pi.
- It is **gated**, not continuous: only when the detector is unsure, capped at 3
  calls per second.
- **Our separation is weak (+0.141)** because NEU-DET has no defect-free images.
  This is a data problem with a data fix: photograph good parts.
- Lighting and camera geometry must be fixed. Rebuild the bank when either
  changes, and recalibrate the threshold when the bank changes.

Next: [15 — SQLite and result storage](15_sqlite.md).
