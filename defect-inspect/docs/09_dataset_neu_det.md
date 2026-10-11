# 09 — The NEU-DET Dataset

**Prerequisites:** [06 — CV task taxonomy](06_cv_task_taxonomy.md).

---

## 1. Concept

NEU-DET is a public dataset of hot-rolled steel strip surface defects, published
by Northeastern University, China. It is the standard benchmark for this
problem, which is why the Review 1 base papers use it and why our results can be
compared against published ones.

**What we actually have**, measured by `scripts/prepare_dataset.py` rather than
quoted from a paper:

| | |
|---|---|
| Images | **1770** (not the 1800 usually cited) |
| Resolution | 200 x 200 px |
| Colour | grayscale, stored as 3-channel JPEG |
| Classes | 6 |
| Annotations | Pascal VOC XML, one file per image |
| Total boxes | 4126 |
| Median box area | **11.8% of the image** |

---

## 2. Why it is needed

A model is its data. Every property of the dataset becomes a property of the
system, whether or not anyone looked. The class imbalance below is not trivia —
it *is* the reason crazing recall is 0.222.

The second reason to study it: viva questions about a dataset are the easiest to
ask and the most revealing to answer badly.

---

## 3. Real-world example

Hot rolling passes a steel slab through rollers at high temperature. Each defect
class has a physical cause on that line:

| Class | Physical origin | Visual character |
|-------|----------------|------------------|
| **crazing** | Thermal stress cracking of the surface layer | A fine network of hairline cracks, very low contrast |
| **inclusion** | Non-metallic material rolled into the surface | Dark elongated streaks, high contrast |
| **patches** | Uneven oxidation or scale adhesion | Large irregular bright or dark areas |
| **pitted_surface** | Corrosion or descaling faults | Clusters of small dark pits |
| **rolled-in_scale** | Oxide scale pressed into the surface by rollers | Diffuse dark smudges, soft edges |
| **scratches** | Mechanical contact during handling | Long thin lines, often diagonal |

This matters diagnostically. Our two worst classes are **crazing** (mAP 0.385)
and **rolled-in_scale** (0.514) — precisely the two described above as "very low
contrast" and "diffuse, soft edges". The model's weaknesses line up exactly with
physical visibility, which is a much better viva answer than "the model needs
more epochs".

---

## 4. How it works

### Pascal VOC XML format

```xml
<annotation>
    <filename>crazing_1.jpg</filename>
    <size><width>200</width><height>200</height><depth>1</depth></size>
    <object>
        <name>crazing</name>
        <bndbox>
            <xmin>2</xmin><ymin>2</ymin><xmax>193</xmax><ymax>194</ymax>
        </bndbox>
    </object>
</annotation>
```

VOC stores **absolute pixel corners**. YOLO needs **normalised centre and size**:

```
class  cx/W  cy/H  w/W  h/H
0      0.487500  0.490000  0.955000  0.960000
```

Normalising by image size is what lets one model train on mixed resolutions, and
it is why a YOLO label file contains no image dimensions at all.

### What our converter found

`scripts/prepare_dataset.py` reports rather than assumes:

```
classes discovered (6): crazing, inclusion, patches, pitted_surface,
                        rolled-in_scale, scratches

class                train     val    test   total
--------------------------------------------------
crazing                484     131      63     678
inclusion              696     179     126    1001
patches                600     180      86     866
pitted_surface         298      83      42     423
rolled-in_scale        429     127      64     620
scratches              380     101      57     538
--------------------------------------------------
boxes                 2887     801     438    4126
images                1236     353     181    1770

median box area: 11.8% of the image
boxes under 1% of image area: 1.2%
```

Four findings worth carrying into the report:

**1. 1770 images, not 1800.** The commonly cited figure is 300 per class. The
public mirror we used ships roughly 295 per class. Stating the number you
actually have, and where it came from, is more defensible than repeating a
figure from an abstract.

**2. Images are balanced; boxes are not.** Each class has ~295 images, but
inclusion contributes 1001 boxes while pitted_surface contributes 423 — a 2.4x
imbalance, because an inclusion image typically contains several separate
streaks and a pitted_surface image one large region. **The model trains on
boxes, not images**, so this is the imbalance that matters.

**3. The defects are large, not small.** Median box area is 11.8% of the image
and only 1.2% of boxes fall under 1%. This directly contradicts the Review 1
slide claiming a *"small-object detection head for micro-scale surface
defects"*. Adding a high-resolution P2 head would cost frame rate to detect
objects that are not there. **The standard three heads are the correct choice,
and the data says so.**

**4. Class names are discovered, not assumed.** The converter reads every XML
and builds the class list from what is present. Hardcoding six names means a
silently renamed class becomes class 0 — a corruption with no error message.

### Splitting: why stratified

We split 70/20/10 into train, val and test, **per class**, with a fixed seed.

A plain random split of 1770 images would leave class proportions lumpy in the
smaller splits by chance. Validation mAP would then partly measure the luck of
the draw, and comparing two training runs would compare two different problems.
Stratifying means every split holds the same class mix.

The three splits have three different jobs, and mixing them up invalidates the
result:

| Split | Used for | Seen by |
|-------|----------|---------|
| **train** | gradient updates | the optimiser |
| **val** | choosing the best epoch, early stopping | model selection |
| **test** | the number you report | nothing until the end |

**Val mAP is mildly optimistic** because the best epoch was chosen using it. Our
`scripts/evaluate.py` defaults to the test split for that reason. The numbers
agree closely — val 0.732, test 0.711 — which is itself evidence that no
selection overfitting occurred.

### Data augmentation

1236 training images is small. Augmentation manufactures variety, and each
choice in `train.py` is justified by the physics of the problem rather than
copied from a tutorial:

| Augmentation | Setting | Why |
|--------------|---------|-----|
| Horizontal flip | 0.5 | A scratch is a scratch mirrored |
| **Vertical flip** | **0.5** | Also true here. Not true for photographs of the world, where the sky is up — a good example of tuning augmentation to the domain |
| Rotation | ±10° | The strip is roughly axis-aligned; large rotations would be unrealistic |
| Scale | ±30% | Simulates camera height variation on the rig |
| Translate | ±10% | Part position on the belt varies |
| **Brightness (HSV-V)** | **0.3** | Stands in for the LED ring light dimming as it ages |
| **Hue, Saturation** | **0.0** | The images are grayscale. Jittering hue would be a no-op on the data and a lie in the report |
| Mosaic | 1.0, off for last 10 epochs | Tiles four images into one; the main regulariser for small datasets |
| Perspective | 0.0 | The camera looks straight down |
| MixUp | 0.0 | Blending two images is meaningless for surface texture |

**`close_mosaic: 10`** deserves a note. Mosaic creates artificial images with
four scenes and unnatural boundaries. Excellent for regularisation, but the
model should finish training on realistic frames — so mosaic is switched off for
the final 10 epochs. This is standard practice and it is a good viva detail.

---

## 5. Mathematical intuition

**VOC to YOLO**, for image width $W$ and height $H$:

$$ c_x = \frac{x_{min}+x_{max}}{2W}, \quad c_y = \frac{y_{min}+y_{max}}{2H}, \quad w = \frac{x_{max}-x_{min}}{W}, \quad h = \frac{y_{max}-y_{min}}{H} $$

All four outputs lie in $[0, 1]$ — provided the input is valid. NEU-DET
annotations occasionally run a pixel or two past the image edge, so
`prepare_dataset.py` clamps before converting. Without the clamp a normalised
value exceeds 1.0 and ultralytics rejects the whole label file at load time.

**Effective dataset size under augmentation.** With $k$ independent binary
augmentations the model can see up to $2^k$ variants of each image. Both flips
alone give 4; adding continuous rotation, scale and brightness makes the space
effectively continuous. This is why 1236 images can train 3 million parameters
without immediate memorisation.

---

## 6. Software implementation

Acquisition is scripted, not manual:

```bash
uv run python scripts/fetch_dataset.py     # sparse git clone, ~45 MB
uv run python scripts/prepare_dataset.py   # VOC -> YOLO, stratified split
```

`fetch_dataset.py` uses a **partial, sparse clone** — the upstream repository is
423 MB of notebooks and checkpoints, and we want two folders totalling 45 MB:

```python
run(["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1", UPSTREAM, dest])
run(["git", "sparse-checkout", "set", "IMAGES", "ANNOTATIONS"], cwd=dest)
```

`--filter=blob:none` downloads no file contents up front; `sparse-checkout` then
fetches blobs only for the paths named.

Two safety measures in `prepare_dataset.py` worth pointing at:

**Orphan detection.** Images and annotations are matched by stem, and anything
unpaired is reported rather than silently skipped. The upstream repository's
file listing suggested 1800 XML against 1770 JPEG; the check exists so a
mismatch is a printed number, not 30 quietly missing samples.

**XML hardening.** The annotations come from a third-party GitHub repository, so
they are untrusted input. Python's expat parser expands internal entities, which
is the billion-laughs denial of service. VOC files never need a DTD, so any file
declaring one is rejected:

```python
if "<!doctype" in lowered or "<!entity" in lowered:
    raise ValueError(f"{xml_path.name}: DTD or entity declaration is not allowed")
```

One guard, no extra dependency.

The coordinate conversion carries its own assertions:

```python
assert to_yolo((0, 0, 100, 200), 200, 200) == (0.25, 0.5, 0.5, 1.0)
assert to_yolo((-5, -5, 205, 205), 200, 200) == (0.5, 0.5, 1.0, 1.0)  # clamped
assert to_yolo((100, 200, 0, 0), 200, 200) == (0.25, 0.5, 0.5, 1.0)   # reversed
assert to_yolo((50, 50, 50, 60), 200, 200) is None                     # degenerate
```

---

## 7. Hardware implementation

NEU-DET is 200x200 grayscale under controlled laboratory lighting. Your conveyor
will not be.

**The domain gap, and what to do about it:**

| Property | NEU-DET | Your rig |
|----------|---------|----------|
| Resolution | 200x200 | 640x480 or higher |
| Lighting | uniform, laboratory | LED ring, ageing, ambient leakage |
| Colour | grayscale | colour sensor |
| Framing | defect fills much of the frame | part is one object among background |
| Focus | fixed, correct | depends on mounting |

A model trained purely on NEU-DET will underperform on your rig. Three mitigations,
in increasing order of effort:

1. **Match the imaging conditions.** Mount the camera so a part fills the frame
   the way NEU-DET's defects do. Free, and the largest single win.
2. **Fine-tune on rig images.** Photograph 100 to 200 parts, label them, and
   continue training from `best.pt`. A few hundred domain images are worth
   thousands of out-of-domain ones.
3. **Rebuild the PatchCore bank from your own good parts** — see
   `reports/anomaly.md`. This one needs no labelling at all.

This gap is the honest answer to "will it work in a factory", and stating it
plainly is stronger than claiming it will.

---

## 8. Common mistakes

- **Quoting 1800 images when you have 1770.** Report what you measured.
- **Balancing images and calling the dataset balanced.** Boxes are what train
  the model, and boxes range from 423 to 1001 per class.
- **Reporting val mAP as the result.** Val chose the best epoch, so it is
  optimistic. Report test.
- **Random rather than stratified splitting.** Makes small-split metrics partly
  a measure of luck.
- **Augmenting hue and saturation on grayscale images.** No effect on the data;
  misleading in the report.
- **Leaving mosaic on to the last epoch.** The model finishes on artificial
  four-image composites rather than realistic frames.
- **Claiming small-object detection because defects sound small.** Ours have a
  median area of 11.8%. Measure before you claim.

---

## 9. Viva questions

**Q: Describe your dataset.**
NEU-DET: 1770 grayscale images of hot-rolled steel strip at 200x200, six defect
classes, 4126 bounding boxes in Pascal VOC XML. The commonly cited figure is
1800; the public mirror we used ships 1770, and we report what we have.

**Q: Is the dataset balanced?**
Balanced by image — roughly 295 per class — but not by box. Inclusion has 1001
boxes and pitted_surface 423, because an inclusion image usually contains
several separate streaks. The model trains on boxes, so that 2.4x imbalance is
the one that matters.

**Q: Why did you split the data 70/20/10 and stratify it?**
Train updates the weights, validation selects the best epoch, and test provides
the reported number. Stratifying per class ensures every split has the same
class mix, so validation mAP measures the model rather than the luck of the
split. Our val and test scores agree closely (0.732 and 0.711), which indicates
no selection overfitting.

**Q: Which augmentations did you use, and why those?**
Both flips at 0.5 because steel defects have no canonical orientation; ±10°
rotation because the strip is axis-aligned; ±30% scale for camera height
variation; brightness jitter at 0.3 to simulate the ring light dimming with age.
Hue and saturation are set to zero because the images are grayscale, and
perspective to zero because the camera looks straight down.

**Q: Why is crazing your worst class?**
Physics, not hyperparameters. Crazing is a network of hairline thermal cracks
with very low contrast against the mill finish, and its extent is genuinely
ambiguous, so annotators disagree about box boundaries. It is the lowest-AP
class in most published NEU-DET results too. Our recall is 0.222 and we report
it rather than hiding it in an average.

**Q: Will this work on a real production line?**
Not without adaptation, and we say so. NEU-DET is 200x200 grayscale under
laboratory lighting; the rig will have different resolution, lighting and
framing. The mitigations are to match the imaging geometry, fine-tune on a few
hundred rig images, and rebuild the anomaly bank from real good parts — the last
of which needs no labelling.

---

## 10. Summary

1. 1770 images, 6 classes, 4126 boxes, 200x200 grayscale, VOC XML — measured,
   not quoted.
2. Balanced by image, imbalanced 2.4x by box, and boxes are what train the model.
3. Median box area is 11.8%, so these are large diffuse regions and a
   small-object head is unjustified.
4. Stratified 70/20/10 with a fixed seed; report test, not val.
5. Augmentation choices follow the physics: both flips yes, hue jitter no.
6. The lab-to-factory domain gap is real, and the honest mitigation plan is
   matching the imaging conditions and fine-tuning on rig images.

**Next:** [10 — Training and hyperparameters](10_training_hyperparameters.md)
