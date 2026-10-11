# 07 — PyTorch

**Prerequisites:** [05 — CNN fundamentals](05_cnn_fundamentals.md).

---

## 1. Concept

PyTorch is the framework that trains the model. It provides three things:

1. **Tensors** — NumPy arrays that can live on a GPU.
2. **Autograd** — automatic differentiation. You write the forward computation;
   PyTorch works out the gradients.
3. **Modules and optimisers** — the building blocks of a network and the code
   that updates its weights.

In this project you never write PyTorch directly. Ultralytics wraps it.
But every number scrolling past during training comes from these three
mechanisms, and you cannot debug or defend a training run without knowing them.

---

## 2. Why it is needed

Return to the CNN of Module 05. It has 3 million weights. Training means
finding values for all of them that minimise a loss.

To move a weight in the right direction you need to know how the loss changes
when that weight changes — the partial derivative $\partial L / \partial w$.
For 3 million weights, through 73 layers, computed by hand, once per batch,
for 2000 batches.

That is not a task for a person. **Autograd is the reason deep learning is
practical**: you write only the forward pass, and the derivative of every
operation with respect to every parameter comes free.

---

## 3. Real-world example

Consider deciding how far to open a valve to hit a target temperature. You
nudge it, observe the error, nudge again. Gradient descent is that loop, with
two differences: there are 3 million valves, and the derivative tells you
which way to nudge each one *without trying it first*.

That last part is the whole trick. Testing 3 million weights one at a time
would need 3 million forward passes per step. Backpropagation computes all
3 million derivatives in roughly the cost of *one* extra forward pass.

---

## 4. How it works

### Tensors

```python
import torch

x = torch.zeros(1, 3, 320, 320)      # a batch of one 320x320 RGB image
print(x.shape, x.dtype, x.device)    # torch.Size([1, 3, 320, 320]) torch.float32 cpu

x = x.to("cuda")                     # move it to the GPU
```

A tensor is a NumPy array plus two abilities: it can live on a GPU, and it can
remember the operations performed on it so gradients can flow back through them.

Conversion is cheap and lossless in both directions:

```python
array  = tensor.cpu().numpy()
tensor = torch.from_numpy(array)
```

`.cpu()` is required before `.numpy()` — a GPU tensor has no NumPy
representation, and forgetting this is one of the most common PyTorch errors.

### Autograd

```python
w = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0])

y = w * x          # y = 6
loss = (y - 10) ** 2   # squared error against a target of 10, so loss = 16

loss.backward()    # compute dL/dw
print(w.grad)      # tensor([-24.])
```

Check that by hand: $L = (wx - 10)^2$, so
$\frac{dL}{dw} = 2(wx-10)\cdot x = 2(6-10)(3) = -24$.
The gradient is negative, meaning increasing $w$ decreases the loss — so the
optimiser will increase it. PyTorch built a graph of the operations as they
ran, then walked it backwards applying the chain rule.

### The training loop

Everything ultralytics does reduces to this:

```python
for epoch in range(epochs):
    for images, targets in dataloader:
        predictions = model(images)          # 1. forward
        loss = loss_fn(predictions, targets) # 2. how wrong
        optimizer.zero_grad()                # 3. clear old gradients
        loss.backward()                      # 4. backward: fill .grad
        optimizer.step()                     # 5. nudge every weight
```

Step 3 is not optional and not obvious: PyTorch **accumulates** gradients
rather than replacing them. Omit `zero_grad()` and every batch adds to the
previous batch's gradients, so the model takes wilder and wilder steps and the
loss diverges. It is deliberate — accumulation is how you simulate a large
batch on a small GPU — but as a default it catches everyone once.

### Backpropagation, concretely

For a two-layer chain $L \leftarrow y_2 \leftarrow y_1 \leftarrow w_1$, the
chain rule gives:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y_2} \cdot \frac{\partial y_2}{\partial y_1} \cdot \frac{\partial y_1}{\partial w_1}$$

Backpropagation is this applied layer by layer, from the loss backwards, with
each layer reusing the gradient the layer above already computed. The reuse is
what makes it cheap: no term is calculated twice.

### Transfer learning, and why it matters here

```python
model = YOLO("yolov8n.pt")     # weights pretrained on COCO, 118,000 images
model.train(data="neu.yaml", epochs=100)
```

`yolov8n.pt` was trained on COCO — people, cars, dogs. Nothing to do with
steel. So why start from it?

Because the *early layers* learned edges, textures and gradients, and those are
universal. A layer-1 filter that responds to a diagonal edge is equally useful
for a dog's ear and a scratch on steel. Only the deep layers are
COCO-specific, and fine-tuning reshapes those.

The alternative is starting from random weights. On 1236 training images that
is a losing proposition — the network would overfit long before learning
general features. Published comparisons on small datasets routinely show a
10 to 20 point mAP gap in favour of pretrained initialisation.

**This is the single highest-value line in `train.py`**, and a likely viva
question.

### GPU versus CPU for training

Our run: 100 epochs in 10.4 minutes on an RTX 4060. The same run on a CPU would
take hours. A GPU has thousands of simple cores, and a convolution is thousands
of independent multiply-accumulates — an almost perfect match.

Inference is different. One image at a time on a Pi has no batch to parallelise,
and the GPU's advantage largely evaporates. That asymmetry is why this project
trains on a GPU and deploys on a CPU.

---

## 5. Mathematical intuition

Gradient descent updates every weight by:

$$ w \leftarrow w - \eta \frac{\partial L}{\partial w} $$

where $\eta$ is the learning rate. The gradient points uphill, so the minus
sign steps downhill.

Plain gradient descent is rarely used. **AdamW**, which ultralytics selected
automatically for our run, keeps two running averages per weight — of the
gradient ($m$) and of its square ($v$):

$$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t $$
$$ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 $$
$$ w \leftarrow w - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

The intuition: dividing by $\sqrt{v}$ gives each weight its own effective
learning rate. A weight with consistently large gradients gets smaller steps; a
weight that rarely matters gets larger ones. On a small, imbalanced dataset like
ours this adapts far better than one global rate.

The `W` in AdamW is decoupled weight decay — a small pull of every weight
towards zero, applied separately from the gradient. It is a regulariser, and on
1236 images regularisation matters.

---

## 6. Software implementation

Where PyTorch appears in this repo, and where it deliberately does not:

| File | Uses PyTorch | Why |
|------|--------------|-----|
| `scripts/train.py` | yes, via ultralytics | training needs gradients |
| `scripts/evaluate.py` | yes | runs the `.pt` model |
| `scripts/export_onnx.py` | yes | reads `.pt`, writes ONNX |
| `scripts/build_patchcore.py` | yes | ResNet-18 weights, then exports to ONNX |
| `src/inspector/*` | **never** | this is what runs on the Pi |

That last row is enforced by a test:

```python
def test_runtime_has_no_training_dependency():
    banned = {"torch", "ultralytics", "torchvision"}
    for name in MODULES:
        source = stdlib_inspect.getsource(importlib.import_module(name))
        for word in banned:
            assert f"import {word}" not in source
```

The invariant is easy to break with one convenient import, and the failure
would only surface on the Pi. Testing it makes the break immediate.

---

## 7. Hardware implementation

| | Training (laptop) | Inference (Pi 5) |
|---|---|---|
| Framework | PyTorch + CUDA | ONNX Runtime |
| Install size | ~2.6 GB | ~50 MB |
| Hardware | RTX 4060, 8 GB VRAM | 4x Cortex-A76 |
| Our measurement | 100 epochs, 10.4 min | ~20 ms/frame |

PyTorch does run on a Pi. It is simply the wrong tool there: forty times the
install, a slower cold start, more RAM, and every one of those costs buys
gradient machinery that inference never uses.

---

## 8. Common mistakes

- **Forgetting `optimizer.zero_grad()`.** Gradients accumulate; the loss diverges.
- **Calling `.numpy()` on a GPU tensor.** Needs `.cpu()` first.
- **Leaving the model in training mode during evaluation.** `model.eval()`
  switches dropout off and batch-norm to running statistics. Forgetting it makes
  validation numbers noisy and wrong.
- **Not using `torch.no_grad()` for inference.** Builds a graph nobody will use,
  wasting memory and time.
- **Training from scratch on a small dataset.** Always start from pretrained
  weights unless there is a specific reason not to.
- **Assuming the GPU helps inference as much as training.** Batch size one on an
  edge device is a different problem.

---

## 9. Viva questions

**Q: What is a tensor?**
A multi-dimensional array that can live on a GPU and can record the operations
applied to it so gradients can be computed automatically. An image is a 3-D
tensor; a batch of images is 4-D.

**Q: Explain backpropagation.**
After the forward pass, the loss is known. Backpropagation applies the chain
rule from the loss backwards through the network, computing each layer's
gradient from the one after it. Reusing those intermediate gradients means all
3 million partial derivatives cost about one extra forward pass, instead of
3 million.

**Q: What is transfer learning and why did you use it?**
Starting from weights pretrained on a large dataset instead of random ones. The
early layers of a CNN learn edges and textures, which are useful for any image.
We fine-tuned COCO-pretrained YOLOv8n on 1236 steel images; training from
scratch on that little data would overfit badly.

**Q: Which optimiser did you use and why?**
AdamW, selected automatically by ultralytics with a learning rate of 0.001.
It maintains per-weight adaptive learning rates from running averages of the
gradient and its square, which suits a small imbalanced dataset better than a
single global rate. The decoupled weight decay adds regularisation, which
matters at 1236 images.

**Q: Why is PyTorch not installed on the Raspberry Pi?**
Because the Pi only runs inference, which needs no gradients. ONNX Runtime does
the same forward pass in about 50 MB instead of 2.6 GB, starts faster and uses
less RAM. A test in `tests/test_pipeline.py` asserts no runtime module imports
torch, so the invariant cannot rot.

**Q: What is the difference between `model.train()` and `model.eval()`?**
They switch layers that behave differently in the two phases — dropout is
active only in training, and batch normalisation uses batch statistics in
training but stored running statistics at evaluation. Evaluating in train mode
gives wrong and unstable numbers.

---

## 10. Summary

1. PyTorch gives tensors, autograd and optimisers; ultralytics wraps all three.
2. Autograd is why deep learning is practical — you write only the forward pass.
3. The five-line training loop is the whole of training; `zero_grad()` is the
   line everyone forgets.
4. Transfer learning from COCO weights is the highest-value decision in
   `train.py`, worth 10 to 20 mAP points on a dataset this size.
5. Training needs PyTorch and a GPU. Inference needs neither, which is the
   entire basis of this project's deployment design.

**Next:** [08 — YOLO, v1 to v8](08_yolo_v1_to_v8.md)
