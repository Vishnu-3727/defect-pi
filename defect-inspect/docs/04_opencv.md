# 04 — OpenCV

**Prerequisites:** [03 — NumPy and images](03_numpy_images.md).

---

## 1. Concept

OpenCV (`cv2`) is a C++ computer vision library with a Python interface. In this
project it does everything to the image *except* the neural network: capture it
from the camera, resize it, convert colour, draw the result, and encode it for
the dashboard.

It contains no learning. This is worth stating plainly, because "we used
OpenCV" is sometimes written in reports as though it were the AI. It is not.
OpenCV is the plumbing on both sides of the model.

---

## 2. Why it is needed

The neural network is fussy: it accepts exactly one input size, one channel
order, one value range. The camera produces none of those. Something has to
bridge the gap at 25 frames per second without becoming the bottleneck.

You could write that bridge yourself in NumPy. Resizing alone would be a
hundred lines of interpolation, and it would be ten times slower than OpenCV's,
which is hand-optimised with SIMD for exactly this. Reuse is the whole
argument.

OpenCV also owns the parts of the pipeline that touch the operating system —
opening a camera device, decoding a video file, encoding JPEG — which are
platform-specific and not worth reimplementing.

---

## 3. Real-world example

Every machine-vision vendor stack — Cognex, Keyence, Basler's pylon — has an
equivalent of OpenCV underneath: an image acquisition layer plus a library of
fast pixel operations. When a factory writes a custom inspection application,
OpenCV is the open-source default, and it is deployed on production lines
worldwide. Using it here is not a student shortcut; it is the industry choice.

---

## 4. How it works

### The pieces we actually use

| Function | Purpose in this project |
|----------|------------------------|
| `cv2.VideoCapture` | Open a camera, a video file, or a stream |
| `cv2.imread` / `cv2.imwrite` | Load and save single images |
| `cv2.resize` | Scale a frame to the network's input size |
| `cv2.cvtColor` | Convert between BGR, RGB and grayscale |
| `cv2.rectangle` / `cv2.putText` | Draw detections for the operator |
| `cv2.imencode` | JPEG-encode a frame for the web dashboard |

That is six functions. OpenCV has about two thousand; ignore the rest until you
need them.

### Video capture

```python
cap = cv2.VideoCapture(0)                  # 0 = first camera device
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ok, frame = cap.read()
    if not ok:
        break                              # camera unplugged, or file ended
    ...
cap.release()
```

Two things that bite:

- **`cap.set` is a request, not a command.** The driver may ignore it. Always
  read back the actual size with `cap.get(cv2.CAP_PROP_FRAME_WIDTH)` rather
  than assuming.
- **`cap.read()` returns a success flag.** Ignoring it means that when the
  camera disconnects you process `None` and crash somewhere unrelated, three
  functions away from the real cause.

### The interpolation choice, which is not cosmetic

`cv2.resize` takes an `interpolation` argument, and the correct value depends
on whether you are shrinking or growing:

| Flag | Use when | What it does |
|------|----------|--------------|
| `INTER_AREA` | **Shrinking** | Averages all source pixels falling into each destination pixel |
| `INTER_LINEAR` | Growing | Blends the four nearest source pixels |
| `INTER_NEAREST` | Masks, labels | Copies the single closest pixel; no blending |

Shrinking with `INTER_LINEAR` point-samples: it looks at four pixels and
ignores the rest. A scratch two pixels wide, downscaled by 4x, can land between
sample points and **disappear entirely**. `INTER_AREA` averages, so the scratch
survives as a fainter line. For a defect detector this is the difference
between a detection and a miss, and it is one line of code.

Our `preprocess.letterbox` picks the flag from the scale factor for exactly
this reason.

### Colour spaces

| Space | Channels | Where it appears here |
|-------|----------|----------------------|
| **BGR** | Blue, Green, Red | What `cv2.imread` and `cap.read()` return |
| **RGB** | Red, Green, Blue | What the model expects |
| **Grayscale** | one intensity channel | What the NEU-DET images physically are |
| **HSV** | Hue, Saturation, Value | Useful when colour, not texture, defines the defect |

OpenCV returns BGR for a historical reason: the camera hardware Intel used in
1999 delivered bytes in that order. It has never been changed because too much
code depends on it. Every OpenCV-to-model boundary in the world contains a
BGR-to-RGB conversion, and forgetting it is the classic silent bug from Module 03.

HSV deserves a note even though we do not use it: if the defect were rust
(a colour anomaly) rather than a scratch (a texture anomaly), thresholding the
Hue channel would be a legitimate, near-free alternative to a neural network.
Choosing the neural network is justified by our defects being textural.

### Drawing

```python
cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
```

`cv2.rectangle` modifies `frame` **in place** and returns nothing useful. Draw
on a copy if you also need the clean frame — for instance to save the original
image of a failed part alongside the annotated one.

Coordinates must be integers. Passing the float coordinates that come out of
the model raises a cryptic error about overload resolution.

---

## 5. Mathematical intuition

Resizing is resampling a discrete function onto a new grid. For a scale factor
$s < 1$ (shrinking), destination pixel $(y,x)$ covers a source region of
$1/s \times 1/s$ pixels. `INTER_AREA` computes the mean over that region:

$$ I'(y,x) = \frac{1}{|R|}\sum_{(u,v) \in R} I(u,v) $$

This is a box filter followed by decimation, and the averaging is precisely
what preserves thin features. `INTER_LINEAR` instead evaluates

$$ I'(y,x) = \sum_{k=1}^{4} w_k \, I(u_k, v_k) $$

over only the four nearest neighbours, with weights from the fractional
position. Four samples cannot represent a region of sixteen pixels, so
information is discarded — the aliasing that erases the thin scratch.

---

## 6. Software implementation

From `src/inspector/capture.py`, the source abstraction:

```python
def open_source(cfg):
    """Return an iterator of frames, whatever the source is."""
    if cfg.source == "video" or cfg.source == "folder":
        ...
    cap = cv2.VideoCapture(cfg.device_index)
```

One function hides four different frame sources behind one interface, so
`pipeline.py` contains no `if` statement about where frames come from. This is
what lets the same pipeline run against a demo video on a laptop and a Camera
Module 3 on the Pi with a one-line config change.

The relevant OpenCV calls in `preprocess.py` were shown in Module 03; the
interpolation selection is the line worth re-reading:

```python
interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
```

---

## 7. Hardware implementation

### Camera options on the Pi 5

| Option | Interface | OpenCV support | Notes |
|--------|-----------|----------------|-------|
| **Camera Module 3** | CSI ribbon | Through `picamera2`, not `VideoCapture` | Best latency and image quality; autofocus |
| **USB webcam** | USB 2.0/3.0 | `cv2.VideoCapture(0)` directly | Simplest; adds USB transfer latency |
| **Industrial GigE camera** | Ethernet | Vendor SDK | What a real factory uses; out of budget |

The important gotcha: on Raspberry Pi OS Bookworm, the legacy camera stack was
removed. `cv2.VideoCapture(0)` **does not open a CSI camera module**. You must
use the `picamera2` library and hand the resulting NumPy array to OpenCV
yourself. Code written against a USB webcam on a laptop will therefore fail on
the Pi with the ribbon camera, and the failure looks like "camera not found".

Our `capture.py` handles both, choosing at runtime, which is why `source: auto`
exists in the config.

### Lighting is part of the vision system

The LED ring light is not an accessory. Surface defects are visible because
they scatter light differently from the surrounding metal. Change the angle and
a scratch can vanish.

- **Ring light, camera in the centre**: even, shadowless. Good general choice
  and what we specify.
- **Low-angle / dark-field**: light grazes the surface almost parallel, so flat
  areas reflect away from the lens and only raised or recessed features glow.
  Dramatically better for scratches and pitting.
- **Diffuse dome**: kills specular highlights on shiny metal.

A poorly lit image cannot be rescued by a better model. This is the most
common reason a working lab prototype fails on a factory floor.

---

## 8. Common mistakes

- **Using `INTER_LINEAR` to downscale.** Thin defects disappear. Use `INTER_AREA`.
- **Ignoring the return flag of `cap.read()`.** Produces a crash far from the cause.
- **Expecting `cv2.VideoCapture` to open a CSI camera on Bookworm.** It will not.
- **Forgetting BGR to RGB.** Invisible on grayscale training data, wrong on a
  colour camera.
- **Passing float coordinates to `cv2.rectangle`.** Cast to `int`.
- **Drawing on the frame you also want to save clean.** `cv2.rectangle` mutates.
- **Not calling `cap.release()`.** The camera device stays locked and the next
  run cannot open it.
- **Installing both `opencv-python` and `opencv-python-headless`.** They
  conflict; pick one. Headless is correct on a Pi with no display.

---

## 9. Viva questions

**Q: What is OpenCV used for in your project?**
Everything around the model: opening the camera, letterbox resizing to the
network input size, BGR to RGB conversion, drawing detection boxes for the
operator, and JPEG-encoding frames for the dashboard. It performs no learning.

**Q: Why does OpenCV use BGR instead of RGB?**
Historical: the byte order of the camera hardware Intel used when the library
was created. It was never changed for backwards compatibility. Since our model
expects RGB, we convert explicitly in `preprocess.to_tensor`.

**Q: Which interpolation method do you use for resizing, and why?**
`INTER_AREA` when shrinking, because it averages every source pixel in the
destination region. `INTER_LINEAR` samples only four neighbours, so a
two-pixel-wide scratch can fall between the sample points and vanish. For a
defect detector, that is a missed defect caused by one wrong argument.

**Q: How do you connect the Raspberry Pi camera?**
The Camera Module 3 uses the CSI ribbon connector and the `picamera2` library
on Bookworm, because the legacy stack that `cv2.VideoCapture` relied on was
removed. A USB webcam works with `VideoCapture` directly. Our `capture.py`
detects and supports both.

**Q: Why does lighting matter so much?**
Defects are visible through differential light scattering, not through colour.
Uniform ring lighting gives repeatable images, which is what the model was
trained on. Uneven or drifting light changes the appearance of every defect and
degrades accuracy without any code changing — the classic reason a lab
prototype fails in a factory.

---

## 10. Summary

1. OpenCV is the plumbing on both sides of the model; it does no learning.
2. `INTER_AREA` for shrinking. Using `INTER_LINEAR` deletes thin defects.
3. OpenCV gives BGR; the model wants RGB. Convert explicitly, every time.
4. Always check the return flag from `cap.read()`, and always `release()`.
5. On a Pi 5 with Bookworm, the CSI camera needs `picamera2`, not
   `cv2.VideoCapture`. Lighting design is part of the vision system, not an
   accessory.

**Next:** [05 — CNN fundamentals](05_cnn_fundamentals.md)
