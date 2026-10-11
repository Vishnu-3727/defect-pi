# 03 — NumPy and How a Computer Stores an Image

**Prerequisites:** [02 — Python and tooling](02_python_env_tooling.md).

---

## 1. Concept

An image is a grid of numbers. Nothing else. NumPy is the Python library for
holding and operating on grids of numbers efficiently. Therefore, in this
project, *an image is a NumPy array* — from the moment the camera returns it to
the moment it enters the neural network.

---

## 2. Why it is needed

Two reasons, and the second is the one people forget.

**Reason 1: speed.** A Python list of lists holding 40,000 pixel values, looped
over in Python, takes milliseconds per operation. The same operation on a NumPy
array takes microseconds, because NumPy stores the data as one contiguous block
of memory and processes it in compiled C with vector instructions.

Concretely, for our 200x200 images:

```python
# Python loop: about 8 ms
for y in range(200):
    for x in range(200):
        img[y][x] = img[y][x] / 255.0

# NumPy: about 0.02 ms, and it is one line
img = img / 255.0
```

At a 40 ms frame budget, the first version eats a fifth of it doing arithmetic
that should be free.

**Reason 2: it is the common language.** OpenCV returns NumPy arrays. ONNX
Runtime accepts NumPy arrays. PyTorch converts to and from them. NumPy is the
interchange format that lets four separate libraries hand data to each other
with no conversion cost.

---

## 3. Real-world example

A line-scan camera in a rolling mill produces 4096 pixels per line at 10 kHz.
That is 40 million numbers per second arriving as raw bytes. Nobody moves that
through Python objects. It lands in a preallocated NumPy buffer, and the
inspection code operates on slices of that buffer. The same principle applies
at our scale; only the numbers differ.

---

## 4. How it works

### An array has a shape, a dtype, and a memory layout

```python
import numpy as np
import cv2

img = cv2.imread("data/yolo/train/images/scratches_1.jpg")
print(img.shape)   # (200, 200, 3)
print(img.dtype)   # uint8
print(img.min(), img.max())   # 0 255
```

**Shape `(200, 200, 3)`** reads as `(height, width, channels)`. Note the order:
**height first**. This trips up everybody, because we say "200 by 200" and
think "x by y", but arrays index rows before columns.

**dtype `uint8`** means unsigned 8-bit integer: each value 0 to 255. That is
one byte per channel per pixel. Our image is `200 * 200 * 3 = 120,000` bytes.

### Indexing

```python
img[0, 0]         # the top-left pixel, as [B, G, R]
img[100, :]       # row 100 — a horizontal line across the image
img[:, 100]       # column 100
img[50:150, 50:150]   # a centre crop, 100x100
img[..., 0]       # the blue channel only, shape (200, 200)
```

Slicing returns a **view**, not a copy. Writing to `img[50:150, 50:150]` writes
into the original. This is fast and occasionally surprising — Module 19 lists
the bug it causes.

### NEU-DET is grayscale stored as colour

Our dataset images are grayscale, but they are saved as 3-channel JPEGs where
R = G = B. `cv2.imread` returns shape `(200, 200, 3)`. We keep the three
channels because YOLOv8's first convolution expects three input channels — it
was pretrained on colour images. Feeding one channel would require rebuilding
that layer and discarding pretrained weights.

### The four transformations between camera and model

```
   camera frame                     model input
 (480, 640, 3) uint8   ───────►   (1, 3, 320, 320) float32
        BGR                              RGB
      0 .. 255                        0.0 .. 1.0
   height, width, ch              batch, ch, height, width
```

Every one of those four changes is a separate, mandatory step:

| Step | From | To | Why |
|------|------|-----|-----|
| Resize | 480x640 | 320x320 | The network has a fixed input size |
| Colour order | BGR | RGB | OpenCV loads BGR; the model was trained on RGB |
| Scale | 0..255 uint8 | 0..1 float32 | Networks train on small floating-point values |
| Transpose | HWC | CHW, plus a batch axis | The convolution kernels expect channels first |

Get any one wrong and the model still runs, produces numbers, and detects
nothing useful. There is no error message. This is the single most common
silent bug in deployment, and Module 19 gives the test that catches it.

---

## 5. Mathematical intuition

A grayscale image is a function from position to brightness:

$$I : \{0,\dots,H-1\} \times \{0,\dots,W-1\} \rightarrow \{0,\dots,255\}$$

A colour image is three such functions stacked. Every operation we perform is a
function on that function:

- **Normalisation** is a scalar map: $I'(y,x) = I(y,x) / 255$
- **Resizing** is resampling: $I'(y,x) = I(\alpha y, \beta x)$ with interpolation
- **Convolution** (Module 05) is a weighted local sum:
  $$ (I * K)(y,x) = \sum_{i}\sum_{j} I(y+i,\; x+j)\, K(i,j) $$

Seeing an image as a function makes the next module's convolution formula
obvious rather than arbitrary.

---

## 6. Software implementation

From `src/inspector/preprocess.py` — the letterbox resize:

```python
def letterbox(img, size):
    """Resize preserving aspect ratio, pad the remainder with grey."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, (left, top)
```

Three things worth reading closely:

- `np.full(..., 114, ...)` — 114 is the grey value ultralytics pads with during
  training. Matching it means the model sees at inference exactly the padding
  distribution it saw while learning.
- The function returns `scale` and `(left, top)`. Without them you cannot map a
  box detected in the 320x320 letterboxed image back to a pixel in the original
  frame. Forgetting to return them is how boxes end up drawn in the wrong place.
- `canvas[top:top+nh, left:left+nw] = resized` is a slice assignment: NumPy
  writes the block in one compiled memcpy.

And the tensor conversion:

```python
def to_tensor(img):
    x = img[:, :, ::-1]                 # BGR -> RGB, a reversed view
    x = x.astype(np.float32) / 255.0    # uint8 0..255 -> float32 0..1
    x = x.transpose(2, 0, 1)            # HWC -> CHW, a view again
    x = np.ascontiguousarray(x)         # force a real contiguous copy
    return x[np.newaxis, ...]           # add the batch axis -> NCHW
```

**The order of those lines is not cosmetic, and we got it wrong first.**

The original version called `np.ascontiguousarray` immediately after the BGR to
RGB reversal, which reads sensibly — that reversal is what creates the negative
stride. But `transpose` afterwards creates a *new* non-contiguous view, undoing
the fix. The array handed to ONNX Runtime was still non-contiguous. The
self-check in `preprocess.demo()` asserts `tensor.flags["C_CONTIGUOUS"]` and
failed on the first run, which is exactly why that assertion exists. The copy
has to be the **last** operation before adding the batch axis.

This is a good illustration of a wider point: every operation that only
rearranges *how you look at* memory — slicing, reversing, transposing — returns
a view. Only operations that compute new values, like `astype`, materialise a
real array.

---

## 7. Hardware implementation

On the Pi 5, NumPy's compiled kernels use the ARM **NEON** SIMD instruction
set: one instruction operating on 16 bytes at once. The `astype(np.float32) /
255.0` line above is executed 16 pixels at a time without you doing anything.

The practical consequence is a memory rule: allocate once, reuse. Each frame
allocating a fresh 320x320x3 float32 array is 1.2 MB of allocation at 25 FPS,
which is 30 MB/s of pressure on an 8 GB shared-memory device. Module 13 covers
the buffer reuse this motivates.

---

## 8. Common mistakes

- **Assuming `shape` is `(width, height)`.** It is `(height, width, channels)`.
  A 640x480 camera frame has shape `(480, 640, 3)`.
- **Forgetting BGR vs RGB.** The model still runs. Accuracy quietly drops. On
  our grayscale dataset R=G=B, so this bug is *invisible in training data* and
  appears only on a real colour camera — the worst kind.
- **Dividing a `uint8` array by 255 and expecting floats.** `img / 255` promotes
  to float64, which is twice the memory and slower. Cast to `float32` explicitly.
- **Mutating a slice and being surprised the original changed.** Slices are views.
  Use `.copy()` when you mean a copy.
- **Looping over pixels in Python.** If you have written a `for y in range(h)`
  loop over an image, there is a NumPy one-liner for it.

---

## 9. Viva questions

**Q: How is an image represented in memory?**
As a contiguous block of bytes viewed as a NumPy array of shape
`(height, width, channels)` with dtype `uint8`. Our 200x200 colour image is
120,000 bytes: one byte per channel per pixel.

**Q: Why do you convert the image to float and divide by 255?**
Neural networks are trained with inputs in a small range around zero because
gradients behave badly otherwise. The pretrained YOLOv8 weights expect inputs
in 0 to 1, so inference must reproduce the exact scaling used in training.

**Q: What is NCHW and why not NHWC?**
Batch, Channels, Height, Width. It is the layout PyTorch and ONNX use because
convolution kernels iterate over channels innermost, so channels-first gives
better cache behaviour on CPU. TensorFlow historically preferred NHWC. The
model dictates the choice; we just have to match it.

**Q: What is letterboxing and why not just resize?**
Letterboxing scales the image to fit while preserving aspect ratio, then pads
the remainder. A plain resize to a square would stretch a 640x480 frame,
distorting defect shapes the model learned undistorted. The padding offsets are
returned so detections can be mapped back to original coordinates.

**Q: Why is NumPy fast if Python is slow?**
Because the loop is not in Python. NumPy stores data contiguously and dispatches
one call into compiled C that uses SIMD instructions — NEON on the Pi's ARM
cores, AVX on x86.

---

## 10. Summary

1. An image is a NumPy array of shape `(H, W, C)`, dtype `uint8`, values 0-255.
2. Height comes first in `shape`. This is the most common indexing mistake.
3. Camera to model requires four conversions: resize, BGR to RGB, scale to
   0-1 float32, and transpose to NCHW. All four are mandatory and all four fail
   silently.
4. Slices are views, not copies; `np.ascontiguousarray` when a real copy matters.
5. Never loop over pixels in Python. There is always an array operation.

**Next:** [04 — OpenCV](04_opencv.md)
