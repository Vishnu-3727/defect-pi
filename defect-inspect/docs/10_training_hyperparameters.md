# 10 — Training and Hyperparameters

**Prerequisites:** [07 — PyTorch](07_pytorch.md), [09 — NEU-DET](09_dataset_neu_det.md).

---

## 1. Concept

Training is the loop from Module 07 run until the loss stops falling.
Hyperparameters are the settings you choose *before* training that control how
that loop behaves — how many passes, how many images at a time, how big a step,
which optimiser.

Weights are learned. Hyperparameters are chosen. The distinction is exactly what
a viva question about "parameters" is testing.

---

## 2. Why it is needed

Two runs on identical data with different hyperparameters can differ by twenty
points of mAP. Learning rate too high and the loss diverges; too low and it
never arrives. Batch size too large on 8 GB of VRAM and training crashes.
Epochs too many and the model memorises 1236 images instead of learning from
them.

None of these produce an error message saying what went wrong. They produce a
worse number, and the reason has to be reasoned about.

---

## 3. Real-world example

Tuning a PID controller: gain too high oscillates, too low responds sluggishly,
and the right value depends on the plant. Hyperparameters are the same kind of
problem, with two differences — there are more of them, and each experiment
costs ten minutes rather than ten seconds.

That cost is why sensible defaults matter more than exhaustive search for a
semester project. Our run used ultralytics' automatic optimiser selection and
spent the effort on augmentation choices instead, which is where domain
knowledge actually pays.

---

## 4. How it works

### Our actual configuration

From `scripts/train.py`, with the reasoning:

| Hyperparameter | Value | Why this value |
|----------------|-------|----------------|
| `epochs` | 100 | Enough to converge on 1236 images; `patience` stops earlier if not |
| `imgsz` | **320** | NEU-DET is natively 200x200. 640 would upsample every image, doubling compute for invented detail. 320 is the nearest multiple of 32 above native |
| `batch` | 64 | Fits in 8 GB VRAM at this resolution; larger batches give smoother gradients |
| `optimizer` | auto → **AdamW, lr 0.001** | Ultralytics inspects the dataset size and picks; it chose AdamW with momentum 0.9 |
| `cos_lr` | True | Cosine decay — a smooth ramp down rather than sudden step drops |
| `patience` | 30 | Stop if validation mAP has not improved for 30 epochs |
| `seed` + `deterministic` | 42, True | The run is reproducible; two runs give the same number |
| `close_mosaic` | 10 | Mosaic off for the final 10 epochs so training ends on realistic images |

**What actually happened:** 100 epochs in 10.4 minutes on an RTX 4060.
Best validation mAP@50 0.732, test 0.711.

### Epochs, and what "enough" means

One epoch is one pass of all 1236 training images. Our batch size of 64 makes
that 20 optimiser steps per epoch, so 100 epochs is 2000 weight updates.

Too few epochs and the model is **underfit** — it has not extracted what is
there. Too many and it is **overfit** — it has begun memorising individual
images, and validation performance decays while training loss keeps improving.

The way to tell them apart is the gap between the two curves:

```
 loss
  |  \
  |   \___  training loss
  |    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ keeps falling
  |     \
  |      \___                validation loss
  |          ‾‾‾‾‾‾‾\____/‾‾ flattens, then rises
  |                  ^
  |                  best epoch: stop here
  +--------------------------------- epoch
```

`patience: 30` automates this. Ultralytics also saves `best.pt` separately from
`last.pt`, so the deployed weights are from the best validation epoch, not the
final one — even if the final epochs were spent overfitting.

### Batch size

The number of images processed before the weights update once.

| | Small batch (8) | Large batch (64) |
|---|---|---|
| Gradient quality | noisy | smooth, averaged |
| VRAM | low | high |
| Steps per epoch | many | few |
| Regularisation | noise helps generalisation | less inherent noise |

There is a coupling worth knowing: **larger batches usually want larger learning
rates**, because averaging more samples gives a more reliable gradient that can
be trusted with a bigger step. Ultralytics' `auto` optimiser accounts for this,
which is one reason we let it choose.

Our 64 at 320x320 used about 2.6 GB of the 4060's 8 GB — comfortable.

### Learning rate

The single most important hyperparameter.

```
  too high              about right            too low
    |  /\  /\             |\                     |‾‾‾--___
    | /  \/  \            | \___                 |        ‾‾‾---___
    |/        \           |     ‾‾‾---___        |
    +-----------          +--------------        +----------------
    diverges              converges              still going at epoch 100
```

Our run used AdamW at 0.001, chosen automatically. **Cosine decay** then anneals
it smoothly toward zero:

$$ \eta_t = \eta_{min} + \tfrac{1}{2}(\eta_0 - \eta_{min})\left(1 + \cos\frac{t\pi}{T}\right) $$

The intuition: large steps early to find the right region of the loss surface,
tiny steps late to settle precisely into a minimum. Smooth decay avoids the
transient instability that a sudden step drop causes.

### The three losses

YOLOv8 prints three numbers per epoch, and they answer three different
questions:

| Loss | What it penalises | Question |
|------|-------------------|----------|
| `box_loss` | CIoU between predicted and true boxes | Is the box in the right place, the right size, the right shape? |
| `cls_loss` | Binary cross-entropy per class | Is the defect type right? |
| `dfl_loss` | Distribution Focal Loss on box edges | Are the edge distributions sharp and correctly placed? |

Reading them diagnostically:

- `cls_loss` falling while `box_loss` stalls → the model recognises defect types
  but localises poorly. Usually a resolution or augmentation issue.
- `box_loss` falling while `cls_loss` stalls → it finds defects but confuses
  classes. Usually visually similar classes, or too few examples of one.

Our run started at `box 2.10, cls 3.91, dfl 1.98` and converged smoothly, with
mAP@50 climbing 0.126 → 0.631 → 0.712 → 0.732 across the run.

### Overfitting: the six defences we used

1. **Transfer learning** — pretrained COCO weights, so the model does not have
   to learn edges from 1236 images (Module 07).
2. **Augmentation** — flips, rotation, scale, brightness, mosaic (Module 09).
3. **A small model** — 3.0 M parameters cannot memorise as readily as 68 M.
4. **Weight decay** — AdamW's decoupled decay pulls weights toward zero.
5. **Early stopping** — `patience: 30` on validation mAP.
6. **Best-epoch checkpointing** — `best.pt` is the best validation epoch.

**Evidence it worked**: validation mAP@50 0.732 against test 0.711. A model that
had overfit its validation set would show a much larger gap between a split it
was selected on and one it had never seen.

---

## 5. Mathematical intuition

**CIoU loss**, the box term, extends IoU with two corrections:

$$ L_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v $$

- $1 - \text{IoU}$ — the overlap term. On its own it has a fatal flaw: two
  non-overlapping boxes have IoU 0 and therefore **zero gradient**, so the model
  gets no signal about which direction to move.
- $\rho^2/c^2$ — normalised centre distance. This is what fixes that: even with
  no overlap, the model learns to move the box closer.
- $\alpha v$ — an aspect-ratio consistency term, so a tall thin box is not
  matched to a short wide one of equal area.

**Binary cross-entropy** for classification, applied per class independently:

$$ L_{BCE} = -\sum_c \big[ y_c \log \hat{y}_c + (1-y_c)\log(1-\hat{y}_c) \big] $$

Independent sigmoids rather than a softmax means the classes are not forced to
compete, so one region can carry two labels. That is correct behaviour for
surface defects, where a patch can genuinely overlap rolled-in scale.

---

## 6. Software implementation

The augmentation block in `scripts/train.py` is where domain knowledge is
encoded, and each line has a comment explaining the physics rather than the API:

```python
fliplr=0.5,
flipud=0.5,        # a scratch is a scratch upside down
degrees=10.0,      # the strip is axis-aligned; large rotations are unrealistic
scale=0.3,         # camera height variation on the rig
hsv_h=0.0,
hsv_s=0.0,         # the images are grayscale; jitter here is a no-op and a lie
hsv_v=0.3,         # stands in for the LED ring light dimming as it ages
mosaic=1.0,
close_mosaic=10,   # finish on undistorted, realistic frames
mixup=0.0,         # blending two images is meaningless for surface texture
```

Reproducibility is enforced rather than hoped for:

```python
seed=42,
deterministic=True,
```

Without both, two runs of identical code give different numbers, and you cannot
tell whether a change helped or the random seed did.

---

## 7. Hardware implementation

| | Our training run |
|---|---|
| GPU | RTX 4060 Laptop, 8 GB |
| VRAM used | ~2.6 GB at batch 64, 320x320 |
| Wall clock | **10.4 minutes for 100 epochs** |
| Per epoch | ~6 seconds |

Three practical consequences:

- **Training is cheap here.** Six seconds per epoch means experiments are
  affordable — a different augmentation setting costs ten minutes to evaluate,
  not a day. Use that.
- **VRAM sets the batch ceiling.** A CUDA out-of-memory error is a signal to
  halve the batch, not to buy a GPU. Gradient accumulation simulates a larger
  batch on smaller hardware.
- **Training does not touch the Pi.** The Pi has no role until `export_onnx.py`
  has run. This is worth restating whenever someone asks whether the Pi is fast
  enough to train.

---

## 8. Common mistakes

- **Training at 640 because tutorials do.** Our images are 200x200. Upsampling
  triples the compute to invent detail that was never captured.
- **Reporting the last epoch instead of the best.** Use `best.pt`.
- **Changing several hyperparameters at once.** You learn nothing about which
  one mattered.
- **Not fixing the seed.** Then run-to-run noise is indistinguishable from an
  improvement.
- **Treating a falling training loss as success.** Training loss falls while
  overfitting. Watch validation.
- **Copying augmentation settings across domains.** Vertical flip is right for
  steel texture and wrong for photographs of streets.
- **Increasing epochs to fix underperformance.** If validation mAP has plateaued
  for 30 epochs, more epochs give more overfitting, not more accuracy.

---

## 9. Viva questions

**Q: What is the difference between a parameter and a hyperparameter?**
Parameters are the 3 million weights learned by gradient descent.
Hyperparameters are settings chosen before training that control the learning —
epochs, batch size, learning rate, optimiser, augmentation strengths.

**Q: What learning rate and optimiser did you use?**
AdamW at 0.001 with momentum 0.9, selected automatically by ultralytics from the
dataset size, with cosine decay across the run. AdamW gives each weight an
adaptive step from running averages of the gradient and its square, which suits
a small imbalanced dataset, and its decoupled weight decay adds regularisation.

**Q: How do you know the model is not overfitting?**
Six defences: transfer learning, augmentation, a 3 M parameter model,
weight decay, early stopping with patience 30, and best-epoch checkpointing. The
evidence is that validation mAP@50 (0.732) and test mAP@50 (0.711) agree
closely — a model that had overfit the split it was selected on would show a
much larger gap.

**Q: What do box_loss, cls_loss and dfl_loss mean?**
Box loss is CIoU, penalising wrong position, size and aspect ratio. Class loss is
per-class binary cross-entropy. DFL loss trains the distribution over 16 bins
that YOLOv8 uses per box edge to get sub-pixel precision. Reading them separately
is diagnostic: class falling while box stalls means the model knows what it sees
but not exactly where.

**Q: Why train at 320 rather than 640?**
NEU-DET images are natively 200x200. Training at 640 upsamples every image,
roughly quadrupling compute to process interpolated pixels that carry no
information. 320 is the smallest multiple of 32 above native resolution, and 32
is required because the backbone downsamples five times.

**Q: What is early stopping?**
Training halts when validation performance stops improving — 30 epochs without
gain, in our case. It prevents the wasted compute and the overfitting that
continued training would produce, and pairs with saving the best-epoch weights
rather than the last.

---

## 10. Summary

1. Weights are learned; hyperparameters are chosen. Know which is which.
2. Our run: YOLOv8n, 320px, batch 64, AdamW at 0.001 with cosine decay,
   100 epochs in 10.4 minutes, test mAP@50 0.711.
3. The three losses are diagnostic, not decorative — read them separately.
4. Six defences against overfitting, and val-versus-test agreement is the
   evidence they worked.
5. Augmentation is where domain knowledge enters training; every setting in
   `train.py` is justified by the physics of steel inspection.

**Next:** [11 — Evaluation metrics](11_evaluation_metrics.md)
