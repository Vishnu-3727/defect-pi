# 18 — System Architecture

**Prerequisites:** modules [12](12_onnx_and_runtime.md) through
[17](17_gpio_arduino_relay.md).

---

## 1. Concept

Every previous module described one component. This one describes the **arrows
between them**: what crosses each boundary, in what representation, and what
happens when that boundary fails.

A block diagram with unlabelled arrows is decoration. A block diagram where
every arrow names its data type and its failure mode is a design.

---

## 2. Why it is needed

Integration is where projects fail. Each component can be individually correct
while the system is broken, because two modules disagree about a convention
neither documented — BGR versus RGB, pixels versus normalised coordinates,
letterboxed versus original frame.

These bugs share a signature: **no exception is raised.** The array has the
right shape and dtype, so everything runs and the output is quietly wrong.

This module is also the answer to the viva question *"explain your system"*.

---

## 3. Real-world example

An industrial vision system is normally specified as an **interface document**
before any code: the camera vendor guarantees frames in a stated format, the
PLC guarantees a digital input responds within a stated time, the vision
supplier guarantees a verdict within a stated latency.

Nobody can debug the system without knowing which contract was broken. This
module is that document for our project.

---

## 4. How it works

### The full data path

```
 [1] Camera / video / folder
      | numpy uint8 BGR (H,W,3), original resolution
      v
 [2] preprocess.letterbox + to_tensor
      | numpy float32 NCHW (1,3,320,320), 0..1, C-contiguous
      v
 [3] detector.Detector (ONNX Runtime, INT8)
      | numpy float32 (1, 10, 2100) raw output
      v
 [4] detector._decode  (threshold -> NMS -> unletterbox)
      | list[Detection]  xyxy in ORIGINAL pixels, class_id, confidence
      |
      +---> [5] anomaly.maybe_score  --> float | None
      |
      v
 [6] decision.decide
      | Verdict(passed: bool, reason: str)
      v
 [7a] storage.insert      -> SQLite row
 [7b] actuator.eject      -> GPIO / serial / mock
 [7c] pipeline annotate   -> JPEG bytes -> Flask /stream
```

### Arrow [1] → [2]: camera to preprocessing

| | |
|---|---|
| Type | `np.ndarray`, `uint8`, shape `(H, W, 3)`, **BGR** |
| Contract | OpenCV convention; `picamera2` delivers RGB and `PiCameraSource` converts |
| Failure | Camera absent, or unconverted RGB |

**Failure mode.** An unconverted RGB frame raises nothing. The model still
outputs boxes, just worse ones, because red and blue are swapped relative to
training. Accuracy degrades by a few points with no error anywhere — the classic
silent bug, and the reason the conversion lives inside `PiCameraSource` rather
than in calling code.

### Arrow [2] → [3]: preprocessing to the model

| | |
|---|---|
| Type | `float32`, `(1, 3, 320, 320)`, range 0–1, **C-contiguous** |
| Contract | letterboxed with pad value 114, RGB, NCHW |
| Failure | wrong scale, non-contiguous memory, wrong size |

Four transformations happen in a fixed order, and the order is load-bearing:

```python
x = img[:, :, ::-1]                 # BGR->RGB: a VIEW, negative stride
x = x.astype(np.float32) / 255.0    # materialises a real array
x = x.transpose(2, 0, 1)            # HWC->CHW: a VIEW again, non-contiguous
x = np.ascontiguousarray(x)         # ORT requires contiguous memory
return x[np.newaxis, ...]           # -> NCHW
```

`ascontiguousarray` **must come after the transpose**. Placed before it, the
transpose re-creates a non-contiguous view and the copy is wasted. This bug was
present in an early version and was caught only by an assertion in
`preprocess.demo()` checking `tensor.flags["C_CONTIGUOUS"]` — nothing else
failed.

**Failure mode.** Non-contiguous input makes ONNX Runtime copy internally
(slower) or misread strides (garbage). Wrong normalisation — 0–255 instead of
0–1 — produces confident nonsense.

### Arrow [3] → [4]: raw tensor to detections

| | |
|---|---|
| Type | `float32 (1, 10, 2100)` |
| Layout | 10 rows = 4 box + 6 class scores; 2100 = anchor positions |
| Failure | class count mismatch, INT8 range violations |

The tensor is transposed to `(2100, 10)`, class scores are argmaxed, candidates
below `conf_threshold` are discarded, and `cv2.dnn.NMSBoxesBatched` suppresses
overlaps **per class**.

Two details:

**NMS runs in letterboxed space, before coordinate mapping.** IoU is a ratio, so
it is invariant to the affine transform back to original pixels. Doing it first
means suppressing on a smaller candidate set.

**Confidences are clipped to [0, 1].** The INT8 model emitted values up to
**1.28** from a sigmoid output that is mathematically bounded by 1. Clipping is
a safety net, not the fix — the fix was excluding the head from quantisation.
Clipping preserves ordering, so NMS and thresholding are unaffected.

### Arrow [4] → [6]: detections to verdict

| | |
|---|---|
| Type | `list[Detection]`, boxes in **original image pixels**, xyxy |
| Failure | boxes still in letterboxed coordinates |

**Failure mode.** Forgetting `unletterbox_boxes` gives boxes that are plausible
but wrong — offset and scaled. They draw fine, they store fine, and every
downstream area calculation is wrong. `parity.py` exists to catch this: it
matched all 132 FP32 boxes at IoU 1.0000 against ultralytics, which proves the
whole coordinate chain.

### Arrow [4] → [5]: the anomaly gate

| | |
|---|---|
| Type in | frame + detections |
| Type out | `float` score, or **`None`** |
| Failure | bank missing, gate always firing |

`None` is a first-class value meaning *"not evaluated"*, and it propagates all
the way to a `NULL` in the database. It is not zero. Zero would claim PatchCore
ran and found nothing unusual.

**Failure mode.** A missing bank makes `build_anomaly_detector` return `None`
and the system runs detector-only. Deliberate — the line should not stop — but
it is a *silent* downgrade, which is why the artefacts are now versioned.

### Arrow [6]: the fusion rule

```python
fail = bool(significant_detections) or (anomaly is not None
                                        and anomaly > threshold)
```

**OR, not AND.** Either subsystem alone can reject a part.

This is the safety-critical decision in the project. AND would require both to
agree, which sounds rigorous and is exactly wrong: it means a defect the
detector recognises perfectly is passed because PatchCore, whose separation is
weak, disagreed. OR accepts more false rejects in exchange for fewer escapes,
which is the correct trade when a missed defect reaches a customer and a false
reject costs one part.

An area filter (`min_defect_area_px: 40`) suppresses tiny high-confidence boxes
that are almost always sensor noise.

### Arrows [7a/b/c]: acting on the verdict

| Arrow | Contract | Failure mode |
|---|---|---|
| `storage.insert` | FAIL commits immediately, PASS batches | disk full; no retention policy exists |
| `actuator.eject` | returns bool; never raises | falls back to mock, logs loudly |
| annotate → JPEG | publish-latest, never a queue | slow browser drops frames, never blocks |

The ordering in `pipeline.process` is deliberate: **save the image, then eject,
then record.** If power fails between ejecting and recording, a part is in the
reject bin with no row — which is why the FAIL commit is immediate and why the
image is written before the physical action.

---

## 5. Mathematical intuition

### The latency budget

Measured on the Pi at 4 threads, INT8:

| stage | ms |
|---|---:|
| preprocess | 2.0 |
| inference | 23.2 |
| decode + NMS | 0.4 |
| **detector subtotal** | **27.3** |
| anomaly, when it fires (10%) | 90 × 0.10 = 9.0 |
| storage + JPEG + actuator | ~22 |
| **end-to-end measured** | **~58 (17.1 FPS)** |

The gap between 36.6 FPS for the detector alone and 17.1 FPS end to end is not
overhead to be optimised away — it is the rest of the system doing its job. The
figure to quote for the *system* is 17.1 FPS; 36.6 is the *model*.

### Throughput versus latency

$$\text{parts per second} \le \frac{1}{t_{\text{frame}}}
\qquad
t_{\text{decision}} = t_{\text{frame}} + t_{\text{delay}}$$

Throughput sets how many parts can be inspected; latency plus conveyor delay
sets whether the ejector still reaches the right one. Improving the first does
not improve the second — which is why the p95 tail matters more than the median
for a physical system.

---

## 6. Software implementation

`src/inspector/pipeline.py` is the only module that knows the order. Every other
module is independently testable, which is what makes the eleven self-checks
possible.

```python
def process(self, frame):
    t0 = time.perf_counter()
    dets = self.detector(frame)
    score = self.anomaly.maybe_score(frame, dets) if self.anomaly else None
    verdict = decide(dets, score, self.min_area, self.anomaly_threshold)
    path = self._save_failure(frame, dets) if not verdict.passed else None
    if not verdict.passed:
        self.actuator.eject()
    self.storage.insert(Record(...))
    self._publish(frame, dets, verdict)
```

Dependencies point inward: `pipeline` imports every module; no module imports
`pipeline`. That acyclic structure is why `tests/test_pipeline.py` can run each
`demo()` in dependency order and know that a failure is local.

---

## 7. Hardware implementation

```
                     +-------------------+
    ring light ----> |                   |
                     |   steel part on   |
    CSI camera ----> |     conveyor      |
         |           +-------------------+
         v                        ^
   +-----------+                  |
   |  Pi 5 8GB |                  | push
   |  4x A76   |            +-----------+
   +-----------+            | solenoid  |
     |      |               +-----------+
     |      | USB serial          ^
     |      v                     | 12-24 V, separate PSU
     |  +---------+          +----------+
     |  | Arduino |--------> |  relay   |
     |  |  Nano   |   5 V    | (opto)   |
     |  +---------+          +----------+
     v
   Ethernet / WiFi --> dashboard on any browser
```

Physical constraints that the software cannot fix:

- **Lighting must be constant.** PatchCore measures distance from normal; change
  the light and everything is anomalous.
- **Working distance must be fixed.** Patch descriptors are scale-sensitive.
- **The solenoid needs its own supply.** Measured rail sag is already 228 mV
  under CPU load alone.
- **Ejector distance sets `eject_delay_s`** as $d/v$, measured on the real belt.

---

## 8. Common mistakes

**Undocumented colour order.** The most common cross-module bug in computer
vision, and it never raises.

**Coordinates in the wrong space.** Letterboxed boxes look right and are wrong.

**Treating `None` as `0.0`.** Loses the distinction between "not measured" and
"measured as zero".

**AND instead of OR in fusion.** Sounds rigorous, causes escapes.

**A queue between pipeline and dashboard.** Back-pressure makes the line run at
browser speed.

**Optimising the wrong number.** 36.6 FPS is the model; 17.1 FPS is the system.
Quoting the first for the second is the mistake this whole project was built to
avoid.

**Circular imports.** The moment a module imports `pipeline`, isolated testing
ends.

---

## 9. Interview questions

**Q: Walk me through your system.**
Frame in as BGR uint8. Letterbox to 320×320, convert to RGB float32 NCHW
contiguous. ONNX Runtime INT8 inference gives a `(1, 10, 2100)` tensor —
4 box parameters plus 6 class scores at 2100 anchor positions. Threshold,
per-class NMS in letterboxed space, then map coordinates back to the original
image. If the top confidence is below 0.55, PatchCore scores the frame, capped
at 3 calls per second. Fuse with OR. On failure: save the image, fire the
actuator, commit the row immediately. Publish an annotated JPEG for the
dashboard. 27.3 ms for detection, ~58 ms end to end.

**Q: Where is the system most likely to break silently?**
Colour order and coordinate space. Both produce correctly-shaped arrays with
wrong contents, so nothing raises. That is why `parity.py` compares our ONNX
output box-for-box against ultralytics — the FP32 row matched all 132 boxes at
IoU 1.0000, which validates the entire preprocessing and decode chain.

**Q: Why OR and not AND?**
Because a missed defect costs more than a false reject. AND would let a defect
the detector recognises perfectly through, because a weak anomaly module
disagreed.

**Q: Why is the end-to-end rate half the model rate?**
Because the system does more than inference: JPEG encoding, SQLite writes, image
saving on failure, and the anomaly module on 10% of frames. 36.6 FPS is a model
benchmark; 17.1 FPS is the system, and the system number is the honest one.

**Q: What happens if the camera disconnects mid-run?**
The frame source raises, the pipeline loop exits, and the service restarts —
systemd `Restart=on-failure` with five attempts in five minutes, then stop. That
limit is deliberate: a permanent hardware failure should surface as a stopped
service, not an infinite restart loop burying the error.

**Q: How would you add a seventh defect class?**
Retrain with the new class, re-export, re-quantise excluding the head, update
`classes` in the config, and re-run parity. The runtime needs no code change:
the detector validates its output row count against the configured class list
at construction, so a mismatch fails loudly at startup rather than silently
mislabelling.

---

## 10. Summary

- Every arrow has a **type, a contract and a failure mode**. Most integration
  bugs are contract violations that raise nothing.
- BGR in, RGB to the model. Conversion lives inside the camera source.
- `ascontiguousarray` **after** the transpose, never before.
- NMS runs in **letterboxed space**; IoU is scale-invariant, so mapping happens
  after.
- `None` from the anomaly gate means **not evaluated** and becomes SQL `NULL`.
- Fusion is **OR**. A missed defect costs more than a false reject.
- Order on failure: **save image, eject, record** — and FAIL rows commit
  immediately.
- **36.6 FPS is the model; 17.1 FPS is the system.** Quote the second.
- Dependencies point inward to `pipeline`; nothing imports it back. That is what
  makes 11 isolated self-checks possible.

Next: [19 — Debugging playbook](19_debugging_playbook.md).
