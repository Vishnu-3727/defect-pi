# 05 — Convolutional Neural Networks

**Prerequisites:** [03 — NumPy and images](03_numpy_images.md).

---

## 1. Concept

A Convolutional Neural Network is a stack of layers that slides small learned
filters over an image. Early layers respond to edges. Later layers, seeing the
outputs of earlier ones, respond to textures. Deeper still, to arrangements of
textures that constitute a defect.

Nobody designs those filters. They start random and are shaped by the training
data. That is the entire difference between a CNN and the classical edge
detector from Module 01: the filter coefficients are learned rather than
written.

---

## 2. Why it is needed

Consider connecting every pixel of a 320x320x3 image to every neuron in a
1000-neuron layer, which is what a plain "fully connected" network does:

$$ 320 \times 320 \times 3 \times 1000 = 307{,}200{,}000 \text{ weights} $$

307 million weights in **one layer**. YOLOv8n has 3.2 million weights in the
*entire network*. The fully connected version is unusable: too large to store,
too slow to compute, and it would overfit 1236 training images instantly.

It is also *wrong in principle*, for two reasons:

**A scratch is a scratch anywhere in the frame.** A fully connected layer
learns a separate weight for a scratch at position (10, 10) and another for a
scratch at (200, 150). It cannot transfer knowledge between locations. A
convolution applies the *same* filter everywhere, so learning what a scratch
looks like once makes it detectable everywhere. This property is called
**translation equivariance**, and it is the reason CNNs work on images.

**Pixels near each other are related; distant pixels mostly are not.** A
convolution only looks at a small neighbourhood at a time, encoding that
assumption directly into the architecture. This is called **locality**.

Those two assumptions — locality and weight sharing — cut 307 million
parameters down to a few thousand per layer with no loss of useful capacity.

---

## 3. Real-world example

The hierarchy that emerges in a CNN trained on our steel dataset:

| Layer depth | What the filters respond to | Recognisable as |
|-------------|----------------------------|-----------------|
| 1 | Oriented intensity edges, blobs | "there is a boundary here, running at 30 degrees" |
| 2-3 | Combinations of edges: corners, parallel lines, granular texture | "this region is rough" / "these are parallel grooves" |
| 4-6 | Arrangements of textures | "a bright elongated streak on darker rough background" |
| final | Class and box | "scratch, confidence 0.87, at these coordinates" |

This mirrors what a human inspector does — notice a discontinuity, judge the
texture, classify the family — except it was derived by minimising an error
function rather than by apprenticeship.

---

## 4. How it works

### Convolution

A **kernel** is a small grid of weights, typically 3x3. It slides across the
image. At each position it multiplies overlapping values and sums them, giving
one output number.

```
 input patch          kernel          output
 ┌───┬───┬───┐     ┌────┬───┬────┐
 │ 10│ 10│ 10│     │ -1 │ 0 │ +1 │
 ├───┼───┼───┤  *  ├────┼───┼────┤  =  (10*-1 + 10*0 + 10*1) +
 │ 10│200│200│     │ -2 │ 0 │ +2 │     (10*-2 + 200*0 + 200*2) +
 ├───┼───┼───┤     ├────┼───┼────┤     (10*-1 + 200*0 + 200*1)
 │ 10│200│200│     │ -1 │ 0 │ +1 │  =  770
 └───┴───┴───┘     └────┴───┴────┘
```

That particular kernel is a vertical edge detector — it produces a large value
where brightness rises left to right, zero on flat regions. It happens to be
the classical Sobel operator. In a CNN, those nine numbers are *learned*, and
the network typically discovers something Sobel-like on its own in layer 1
because edges are genuinely useful.

Sliding it over the whole image produces a **feature map**: a new grid saying
"how strongly does this pattern appear, here?"

Key parameters:

| Parameter | Meaning | Typical |
|-----------|---------|---------|
| **Kernel size** | Side of the window | 3x3 |
| **Stride** | Step between positions | 1, or 2 to halve the size |
| **Padding** | Border pixels added so edges get processed | 1, to keep the size |
| **Channels out** | How many different filters are learned at this layer | 16 to 512 |

A layer with 64 output channels learns 64 different 3x3 filters and produces 64
feature maps. That is what "the network has 64 channels here" means.

### Activation

A convolution is a weighted sum, which is a linear operation. Stacking linear
operations gives another linear operation — a hundred-layer purely
convolutional network would have exactly the same expressive power as a single
layer. Something non-linear must sit between them.

That is the **activation function**. The common choice is ReLU:

$$ \text{ReLU}(x) = \max(0, x) $$

Negative values become zero; positive values pass through. It is almost free to
compute, and its gradient is either 0 or 1, which avoids the vanishing-gradient
problem that plagued earlier activations like sigmoid.

YOLOv8 uses **SiLU** (also called Swish), $x \cdot \sigma(x)$, a smooth version
of ReLU that performs slightly better and costs slightly more.

### Pooling

**Max pooling** takes a 2x2 window and keeps only the largest value, halving
width and height:

```
 ┌────┬────┬────┬────┐
 │ 12 │  8 │  3 │  5 │        ┌────┬────┐
 ├────┼────┼────┼────┤        │ 12 │  9 │
 │  4 │  2 │  9 │  1 │   ->   ├────┼────┤
 ├────┼────┼────┼────┤        │ 20 │ 15 │
 │ 20 │  6 │  7 │ 15 │        └────┴────┘
 ├────┼────┼────┼────┤
 │ 11 │  3 │ 14 │  2 │
 └────┴────┴────┴────┘
```

Three things happen at once:

1. **Compute drops fourfold** for every subsequent layer.
2. **The receptive field grows.** After pooling, a 3x3 kernel covers 6x6 of the
   original image. Stack enough and a single deep neuron sees the whole frame —
   which is how a network built from 3x3 windows can recognise a defect
   spanning 100 pixels.
3. **Small translations stop mattering.** The maximum in a window is the same
   whether the feature sits at its left or right edge.

The cost is precision of location. Pool too aggressively and you know a defect
is present but not exactly where — which matters for us, because the output is
a bounding box. This tension is exactly what the "neck" of a detector resolves,
and Module 08 covers it.

Modern architectures including YOLOv8 often use **strided convolution**
(stride 2) instead of explicit pooling: it downsamples and learns at the same
time.

### Fully connected layers

After the convolutional stack, a classifier network flattens the feature maps
into a vector and passes it through ordinary dense layers to produce class
scores.

YOLOv8 has **no fully connected layers at all**. Its head is convolutional,
which is what lets it output a spatial grid of predictions rather than one
label for the whole image. Worth knowing, because exam questions still assume
the older CNN diagram ending in dense layers.

### Putting it together

```
 input 320x320x3
      │
   [ conv 3x3, 16 filters ] ──► 320x320x16   edges
      │  SiLU
   [ conv 3x3, stride 2 ]   ──► 160x160x32   textures, wider view
      │  SiLU
   [ conv 3x3, stride 2 ]   ──►  80x80x64
      │  SiLU
   [ conv 3x3, stride 2 ]   ──►  40x40x128
      │  SiLU
   [ conv 3x3, stride 2 ]   ──►  20x20x256   defect-scale arrangements
      │
   [ detection head ]       ──►  boxes + classes
```

Spatial size shrinks, channel count grows. The network trades "where" for
"what", progressively. That single sentence is the shape of nearly every CNN
ever built.

---

## 5. Mathematical intuition

Discrete 2-D convolution of image $I$ with kernel $K$ of size $(2a{+}1)\times(2b{+}1)$:

$$ S(y,x) \;=\; \sum_{i=-a}^{a}\ \sum_{j=-b}^{b} I(y+i,\; x+j)\, K(i,j) \;+\; b_0 $$

With $C_{in}$ input channels the sum also runs over channels, and the layer
repeats this for each of $C_{out}$ filters:

$$ S_c(y,x) \;=\; \sum_{k=1}^{C_{in}} \sum_{i,j} I_k(y+i,\,x+j)\, K_{c,k}(i,j) \;+\; b_c $$

**Parameter count** for one convolutional layer:

$$ \text{params} = C_{out} \times (C_{in} \times k_h \times k_w + 1) $$

For 64 filters of 3x3 over 32 input channels: $64 \times (32\times3\times3+1) =
18{,}496$ weights. Compare with the 307 million from the fully connected layer
at the start of this module. That ratio — roughly 16,000 to 1 — is why CNNs
exist.

**Output size** for input $W$, kernel $k$, padding $p$, stride $s$:

$$ W_{out} = \left\lfloor \frac{W - k + 2p}{s} \right\rfloor + 1 $$

This is the formula that tells you why the network input must be a multiple of
32: five stride-2 stages divide the size by $2^5 = 32$, and a non-divisible
input leaves a fractional feature map. Our `config.py` enforces it:

```python
if cfg.detector.imgsz % 32 != 0:
    raise ValueError(...)
```

---

## 6. Software implementation

We do not write convolutions — ultralytics defines the architecture and ONNX
Runtime executes it. But the concepts appear directly in what we control:

| Concept | Where it shows up |
|---------|------------------|
| Input must be divisible by 32 | `config.py` validation, `imgsz: 320` |
| Channels-first layout | `preprocess.to_tensor` transposing HWC to CHW |
| Three input channels | Why we keep grayscale images as 3-channel |
| Receptive field vs box precision | Why the neck exists — Module 08 |
| Feature maps as the anomaly signal | PatchCore uses intermediate feature maps directly — Module 14 |

That last row is the most interesting: PatchCore does not classify anything. It
takes the *feature maps* from the middle of a pretrained CNN and compares them
against stored normal ones. The features learned in the hierarchy above are
useful on their own, independent of the classifier bolted on top.

---

## 7. Hardware implementation

A convolution is a multiply-accumulate loop, and that is the entire reason edge
AI hardware looks the way it does.

**Cost of one layer**, in multiply-accumulate operations:

$$ \text{MACs} = H_{out} \times W_{out} \times C_{out} \times C_{in} \times k_h \times k_w $$

For a 40x40 output, 128 filters, 64 input channels, 3x3 kernel:
$40 \times 40 \times 128 \times 64 \times 9 \approx 118$ million MACs — for one
layer of one frame.

YOLOv8n at 320x320 totals roughly 2.2 GFLOPs per frame. A Cortex-A76 core at
2.4 GHz with NEON manages several GFLOP/s, and there are four cores. That
arithmetic is where "about 20 to 30 FPS on a Pi 5" comes from, and Module 13
replaces the estimate with measurements.

Three hardware facts follow:

- **Convolution is memory-bound as often as compute-bound.** Weights and feature
  maps must move through cache. This is why INT8 quantisation gives more than
  the 4x you would expect from arithmetic alone — it also quarters the bytes moved.
- **NEON gives 4 to 8 lanes of parallelism per core**, applied automatically by
  ONNX Runtime's kernels. You get it for free, but only if the tensor layout
  matches what the kernel expects, which is another reason to hand it
  contiguous NCHW data.
- **Thread count should equal core count.** Four A76 cores means
  `intra_op_num_threads = 4`. More threads means context switching, not speed.

---

## 8. Common mistakes

- **Thinking pooling and stride are interchangeable.** Both downsample; only
  strided convolution learns while doing it.
- **Forgetting the activation.** A stack of convolutions with no non-linearity
  collapses mathematically into a single convolution.
- **Assuming more layers is better.** Deeper networks have larger receptive
  fields and more capacity, but on 1236 training images they overfit, and on a
  Pi they miss the frame deadline.
- **Feeding an input size not divisible by 32.** Either an error, or silent
  internal resizing that shifts every coordinate.
- **Believing the first layer detects "defects".** It detects edges. The defect
  concept only exists near the output.
- **Quoting parameter count as if it were speed.** A 3.2 M parameter model can
  be slower than a 5 M one if its feature maps are larger. FLOPs and memory
  traffic matter more than parameter count.

---

## 9. Viva questions

**Q: What is a convolution?**
A small learned kernel slid across the image; at each position it computes a
weighted sum of the pixels under it. The result is a feature map showing where
that pattern occurs. The same kernel is applied everywhere, which is weight
sharing.

**Q: Why not use a fully connected network for images?**
Parameter explosion and the wrong inductive bias. One dense layer on a 320x320x3
input needs 307 million weights, versus 3.2 million for all of YOLOv8n. It also
cannot share what it learns between image locations, so a defect at the top left
and the same defect at the bottom right are learned independently.

**Q: What does pooling do, and what does it cost?**
It downsamples, cutting compute fourfold per stage, enlarging the receptive
field, and adding tolerance to small shifts. The cost is spatial precision,
which matters for bounding-box regression — the reason detectors add a neck to
recombine fine and coarse feature maps.

**Q: Why is an activation function necessary?**
Without it the network is a composition of linear maps, which is itself linear.
Depth would add no expressive power. ReLU or SiLU introduces the non-linearity
that lets layers build on each other.

**Q: Why must the input be a multiple of 32?**
The backbone downsamples five times by a factor of two, so the final feature
map is the input divided by 32. A non-divisible size gives a fractional map,
which frameworks resolve by silently rounding — shifting every predicted
coordinate. Our config validation rejects it outright.

**Q: What is the receptive field?**
The region of the original image that influences one neuron's output. It grows
with depth and with each downsampling stage. A defect can only be recognised if
some neuron's receptive field covers it, which is why detecting large defects
needs depth and detecting small ones needs high-resolution feature maps.

---

## 10. Summary

1. A CNN slides learned kernels over the image; early layers find edges, deep
   layers find defect-shaped arrangements.
2. Weight sharing and locality cut a 307-million-parameter dense layer to a few
   thousand, and give translation equivariance for free.
3. Non-linear activation between layers is what makes depth mean anything.
4. Downsampling trades spatial precision for receptive field and speed — the
   central trade-off of every detector architecture.
5. On the Pi, a convolution is a multiply-accumulate loop that is as often
   memory-bound as compute-bound, which is why INT8 helps more than 4x.

**Next:** [06 — Classification, detection, segmentation](06_cv_task_taxonomy.md)
