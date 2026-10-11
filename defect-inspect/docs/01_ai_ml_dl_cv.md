# 01 — AI, Machine Learning, Deep Learning, Computer Vision

**Prerequisites:** none.

---

## 1. Concept

Four words get used interchangeably in project reports and they are not the
same thing. They nest inside each other:

```
┌──────────────────────────────────────────────────────┐
│ ARTIFICIAL INTELLIGENCE                              │
│ any machine doing something we'd call "intelligent"  │
│  ┌────────────────────────────────────────────────┐  │
│  │ MACHINE LEARNING                               │  │
│  │ the behaviour is learned from data, not coded  │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │ DEEP LEARNING                            │  │  │
│  │  │ learned by many-layered neural networks  │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘

COMPUTER VISION is not a layer in this stack. It is a *problem domain* —
"make a machine understand images" — that today is mostly solved using
deep learning, but was attempted for 40 years without it.
```

**Artificial Intelligence** is the umbrella. A chess program from 1985 with
hand-written rules is AI. It learns nothing.

**Machine Learning** narrows it: the program's behaviour comes from data.
You do not write the rule; you show examples and the rule is derived.

**Deep Learning** narrows it further: the model is a neural network with many
layers, and it learns not only the decision rule but also *what features to
look at*. That last part is the whole revolution.

**Computer Vision** is the goal — images in, understanding out. It overlaps the
other three but is not contained by them.

---

## 2. Why it is needed

Try to write, by hand, an `if` statement that detects a scratch on steel.

```python
# The classical approach. This is a real technique, and it really does fail.
def has_scratch(image):
    edges = detect_edges(image)
    long_thin_lines = [e for e in edges if e.length > 50 and e.width < 3]
    return len(long_thin_lines) > 0
```

Now the failures start arriving:

- The LED light dims after 400 hours. Every edge weakens. Detection rate collapses.
- A different steel batch has a rougher mill finish. Now the *texture itself*
  produces thousands of long thin edges. Every part is rejected.
- A scratch runs diagonally across a curved surface, so it is not straight.
- A "rolled-in scale" defect has no edges at all — it is a smudge of different
  reflectance.
- Someone moves the camera 5 cm. Every threshold you tuned is now wrong.

You can patch each one. Each patch adds a constant that works in one factory,
one shift, one lighting condition. This is the "brittle rule-based vision"
problem the project's patent background section describes, and it is why
rule-based inspection systems have a reputation for needing an engineer on
call.

Machine learning inverts the work. Instead of describing a scratch, you collect
300 pictures of scratches and let an optimiser find the description. When the
steel batch changes, you add pictures — you do not rewrite logic.

---

## 3. Real-world example

**Manual inspection (the baseline this project replaces).**
A human at the end of a rolling line looks at each strip. Accuracy on shift 1
is high. By hour 7 it drops — this is measured and well documented in quality
engineering literature as inspector fatigue. Two inspectors also disagree with
each other, which is why factories quote an "inspector agreement rate" at all.
And a human cannot look at 30 parts per second no matter how alert.

**Rule-based machine vision (the 1990s answer).**
Fast and cheap when the product never changes and the lighting is sealed.
Still used today for tasks like "is the cap present, yes or no". Fails on
surface *texture* problems, which is exactly our task.

**Deep learning (this project).**
Learns the appearance of six defect families from 1770 labelled photographs,
then generalises to photographs it has never seen. Tolerates lighting drift and
texture variation because it saw variation during training.

---

## 4. How it works

### The machine learning loop

```
 labelled data ──► model makes a guess ──► compare with the true answer
        ▲                                            │
        │                                            ▼
        └──────── adjust the weights ◄──── measure the error (loss)
```

Repeat a few thousand times. That is the entire idea. Everything else in this
course is detail about *what* the model is, *how* the error is measured, and
*how* the adjustment is computed.

### Three kinds of machine learning

| Kind | You give it | It learns | Our use |
|------|-------------|-----------|---------|
| **Supervised** | Images **and** the correct labels | To reproduce the labels | YOLOv8 — we drew boxes on 1770 images |
| **Unsupervised** | Images only, no labels | The structure of "normal" | PatchCore — we show it only good parts |
| **Reinforcement** | A reward signal for actions taken | A policy of actions | Not used here |

Our project uses both of the first two, deliberately, and Module 14 explains
why one is not enough.

### Why deep learning specifically, for images

A 200x200 grayscale image is 40,000 numbers. A classical ML model (say a
support vector machine) needs those 40,000 numbers reduced to a handful of
meaningful *features* first — edge counts, texture energy, histogram moments.
A human has to invent those features. That human is the bottleneck.

A convolutional neural network learns the features itself, from pixels, as part
of the same optimisation that learns the decision. Early layers end up
detecting edges; middle layers detect textures; deep layers detect
defect-shaped arrangements. Nobody programmed that hierarchy — it emerged
because it minimised the error. Module 05 shows how.

---

## 5. Mathematical intuition

Strip everything away and a supervised model is a function with adjustable
knobs:

$$\hat{y} = f(x; \theta)$$

- $x$ is the input image
- $\theta$ (theta) is the set of weights — for YOLOv8n, about 3.2 million numbers
- $\hat{y}$ is the prediction
- $y$ is the truth

Training minimises a loss $L(y, \hat{y})$ over the dataset:

$$\theta^{*} = \arg\min_{\theta} \frac{1}{N}\sum_{i=1}^{N} L\big(y_i, f(x_i; \theta)\big)$$

Read that as: "find the knob settings that make the average mistake smallest
across all N training examples." The search is done by gradient descent, which
Module 07 explains. There is no magic in the equation — the magic is that a
function with 3.2 million knobs and the right *shape* (convolutional, layered)
can express "this texture is a rolled-in scale defect".

---

## 6. Software implementation

In this repo the split maps directly onto files:

| Concept | Where it lives |
|---------|---------------|
| Supervised learning | `scripts/train.py` |
| The learned weights $\theta$ | `models/best.pt`, then `models/model_int8.onnx` |
| Inference $f(x;\theta)$ | `src/inspector/detector.py` |
| Unsupervised anomaly detection | `src/inspector/anomaly.py` |
| The loss $L$ | inside ultralytics; described in Module 10 |

---

## 7. Hardware implementation

Training and inference have opposite hardware needs, and conflating them is a
classic project mistake.

| | Training | Inference |
|---|---------|-----------|
| Runs | Once, on a laptop | Continuously, on the Pi |
| Hardware | RTX 4060, 8 GB VRAM | 4x Cortex-A76 CPU, no GPU |
| Duration | Minutes to hours | ~40 ms per frame, forever |
| Needs PyTorch | Yes | **No** — ONNX Runtime only |
| Power | 100+ W | ~7 W |

This is why the project exports to ONNX at all (Module 12). A Raspberry Pi can
technically run PyTorch; it would just be slow, huge, and pointless, because
the Pi never needs to compute a gradient.

---

## 8. Common mistakes

- **Calling the project "AI" in the report and never saying which kind.**
  Say "supervised deep learning for object detection, plus unsupervised anomaly
  detection". Precision reads as competence.
- **Believing deep learning removes the need to understand the data.**
  It does not. Module 09 shows the class imbalance in NEU-DET that will shape
  our results whether or not we look at it.
- **Assuming more layers is always better.** On a Pi, more layers is fewer
  frames per second. Model choice is an engineering trade, not a ranking.
- **Confusing Computer Vision with Deep Learning.** Our preprocessing
  (`src/inspector/preprocess.py`) is classical computer vision with no learning
  in it at all, and the system would not work without it.

---

## 9. Viva questions

**Q: Is your project AI, ML, or DL?**
All three, nested. It is AI because the machine makes an inspection judgement.
It is ML because that judgement is learned from data rather than programmed.
It is DL because the learner is a deep convolutional network, YOLOv8n.

**Q: Why not just use image processing with thresholds?**
Because surface defects vary in texture, not just intensity. Threshold-based
rules need retuning for every lighting change, steel grade and camera position.
We showed this concretely: "rolled-in scale" has no strong edges, so an
edge-based rule cannot see it at all.

**Q: What is the difference between supervised and unsupervised learning, in
your project?**
YOLOv8 is supervised: we gave it 1770 images with hand-drawn boxes and class
names. PatchCore is unsupervised: it is shown only defect-free parts and learns
what normal looks like, so it can flag a defect type we never labelled.

**Q: Why is computer vision hard for a computer but easy for a human?**
A human sees objects and materials; a computer receives a grid of brightness
numbers. There is no line in the array that says "scratch". Every bit of
meaning has to be reconstructed from spatial patterns of numbers, and those
patterns change with light, angle, scale and material.

**Q: Could you have used a classical machine learning model like SVM?**
Yes, with hand-engineered texture features such as Local Binary Patterns or a
Gabor filter bank, and published work does exactly that on NEU-DET. It gets
respectable classification accuracy but does not localise defects, and the
feature engineering has to be redone for a new defect type. We chose a detector
so the output is a box, which is what the ejector logic needs.

---

## 10. Summary

1. AI contains ML contains DL. Computer Vision is the problem, not a layer.
2. Rule-based vision fails on surface texture because it needs constants that
   depend on lighting, material and camera position.
3. Supervised learning replaces "describe the defect" with "show 300 examples".
4. Deep learning additionally learns *which features to look at*, removing the
   human feature engineer.
5. Training and inference have opposite hardware requirements; the whole
   deployment design in this project follows from that one fact.

**Next:** [02 — Python and tooling](02_python_env_tooling.md)
