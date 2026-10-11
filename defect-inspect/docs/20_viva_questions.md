# 20 — Viva Questions

**Prerequisites:** all previous modules.

---

## 1. Concept

A viva is not a memory test. Examiners are checking whether you understand the
system you built well enough to have made its decisions yourself.

The reliable signal they look for is **whether you can state a trade-off**.
Anyone can say "we used YOLOv8n". Only someone who understands the system can
say "we used YOLOv8n because the Pi 5 has no usable GPU and a 3.2 MB model at
320 px hits 36.6 FPS, where YOLOv8s would have been more accurate and roughly
three times slower".

---

## 2. Why it is needed

You will be asked about parts of this project you have not thought about in
weeks, in front of people, under time pressure. The material below is what you
already know, arranged for recall.

**The three rules:**

1. **Answer with a number where one exists.** "Fast" is worthless; "36.6 FPS,
   measured on the device" ends the question.
2. **Never bluff.** "I haven't measured that, but here's how I would" is a good
   answer. An invented number is fatal, because the follow-up exposes it.
3. **Volunteer the limitation.** Saying "crazing recall is 0.222 and here's why"
   before you are asked converts a weakness into evidence of rigour.

---

## 3. Real-world example

In a design review, an engineer who says "it works" gets three follow-up
questions. One who says "it works at 36.6 FPS with a 51.5 ms p95, and here is
the case where it does not" gets none — because they have already asked
themselves the follow-ups.

---

## 4. The numbers to have memorised

| Quantity | Value |
|---|---|
| Dataset | NEU-DET, 1,800 images, 6 classes, 200×200 grey |
| Split | 70/20/10 stratified — 1,260 / 361 / 181 |
| Model | YOLOv8n, 320 px, 100 epochs |
| Test mAP@50 | **0.711** |
| Test mAP@50-95 | 0.407 |
| Worst class | crazing, recall **0.222**, mAP@50 0.385 |
| Best class | patches, mAP@50 0.911 |
| Detector on Pi 5 | **36.6 FPS**, 27.3 ms median, 51.5 ms p95 |
| Full system on Pi 5 | **17.1 FPS** |
| INT8 vs FP32 on Pi | 36.6 vs 22.4 FPS (1.64×) |
| Peak RSS | 122 MB |
| Model size | 12.1 MB FP32, 5.6 MB INT8 |
| Quantisation cost | 0.3 mAP points (0.7354 → 0.7324) |
| PatchCore separation | 2.402 defect vs 2.261 normal (**+0.141**) |
| Anomaly threshold | 2.77 (95th percentile of normal) |
| Load temperature | 57.6 °C, throttle point 80 °C |

---

## 5. Review 2 — likely questions

**Q: What does your project do?**
It inspects steel surfaces for defects in real time on a Raspberry Pi 5. A
camera views the part, a YOLOv8n detector identifies six defect types, an
unsupervised anomaly module provides a second opinion when the detector is
unsure, and defective parts are ejected by a solenoid. Every inspection is
logged to SQLite and shown on a web dashboard. 36.6 FPS for detection, 17.1 FPS
end to end, measured on the device.

**Q: Why YOLOv8n and not a larger model?**
The Pi 5 has no usable GPU, so everything runs on four Cortex-A76 cores.
YOLOv8n is the smallest variant, 3.2 M parameters. At 320 px it gives 36.6 FPS
against a 20 FPS requirement, with mAP@50 of 0.711. YOLOv8s would gain a few
points of mAP and cost roughly 3× the compute, which would put us under
target. The constraint chose the model.

**Q: Why 320 pixels?**
NEU-DET images are natively 200×200. 320 is the nearest multiple of 32 above
that — the backbone downsamples by 32, so the input must be divisible by it.
Training at 640 would be upsampling, inventing no information and costing 4×
the compute.

**Q: What is mAP@50?**
Mean Average Precision at an IoU threshold of 0.50. For each class you compute
the area under its precision–recall curve, counting a prediction correct if it
overlaps a ground-truth box by at least 50%, then average across classes.
mAP@50-95 averages that over IoU thresholds from 0.50 to 0.95 in steps of 0.05,
so it also measures how *precisely* the box is placed.

**Q: Why is crazing so bad?**
0.222 recall — it misses about four in five. Crazing is a fine network of
hairline thermal cracks with very low contrast against the mill finish, and its
box boundaries are genuinely ambiguous; annotators disagree about where one
region ends. It is the lowest-AP class in most published NEU-DET results too, so
this is a property of the problem rather than only our training. Remedies in
order of expected value: a per-class confidence threshold, CLAHE contrast
enhancement in preprocessing, oversampling during training, and PatchCore, which
does not need to recognise crazing to notice the surface is abnormal.

**Q: How do you know your ONNX pipeline is correct?**
`scripts/parity.py` compares our hand-written decode against ultralytics'
predictions from the same weights on 60 images. The FP32 model matched **all 132
boxes at IoU 1.0000**. That is not a similarity score, it is an exact match, and
it proves letterboxing, tensor layout, output decode, NMS and inverse coordinate
mapping are all correct — none of which raises an error when wrong.

---

## 6. Deployment and optimisation

**Q: What is ONNX and why export to it?**
Open Neural Network Exchange, a framework-independent graph format. Exporting
lets us run inference with ONNX Runtime instead of PyTorch. That matters because
torch on ARM is around 2 GB and slow, while ONNX Runtime is tens of megabytes
and faster on CPU. It also enables graph optimisations — constant folding and
operator fusion, notably folding BatchNorm into the preceding convolution, which
is free at inference because both are linear.

**Q: What is INT8 quantisation?**
Storing weights and activations as 8-bit integers instead of 32-bit floats, with
a scale and zero-point per tensor or per channel. Four times smaller, and
integer SIMD processes four times as many values per instruction. It needs
calibration: run representative images through the model, record the range each
activation takes, and choose scales from that.

**Q: What went wrong when you quantised?**
Quantising the whole graph dropped mAP@50 from 0.735 to 0.408. The diagnosis
came from the direction of the error — precision *rose* to 0.763 while recall
collapsed to 0.417, so the model was going silent rather than becoming confused,
which points at the confidence computation. That is the detection head:
`model.22`, 66 nodes. It contains a DFL softmax over 16 bins per box edge, a
sigmoid where one INT8 step near the threshold is the difference between a
detection and a miss, and box arithmetic on coordinates spanning 0–320, a much
wider range than backbone activations. Excluding those 66 nodes recovered mAP to
0.7324 at a 25% throughput cost. A corroborating symptom: the fully quantised
model emitted confidences up to **1.28** from a sigmoid bounded by 1.

**Q: Why opset 13?**
Per-channel quantisation requires it. At opset 12 the export succeeds and the
load fails with "Unrecognized attribute: axis for operator DequantizeLinear".
Per-channel means one scale per output channel rather than one for the whole
tensor, which matters when channels have very different dynamic ranges.

**Q: You benchmarked on a laptop first. Is that valid?**
Only for ratios, and even then loosely. The laptop is x86 with AVX-512 and 64
INT8 lanes per instruction at 5 GHz; the Pi is ARM with NEON, 16 lanes, at
2.4 GHz. Absolute FPS carries no information across that boundary. We projected
16–25 FPS from the A76 roofline and the device measured **36.6** — the estimate
was 47% low. Even the ratios shifted: INT8-over-FP32 was 1.21× on x86 and 1.64×
on ARM. Every number in this report is now from the device.

**Q: How do you know the benchmark was not thermally throttled?**
It was bracketed. 46.6 °C before, 57.6 °C after, against an 80 °C soft-throttle
point, and `vcgencmd get_throttled` was unchanged across the run. Two bits are
set, 16 and 18, but those are sticky "has occurred" flags from a brief
under-voltage at boot; no current-throttle bit was ever set. The number also
reproduced at 36.6–36.7 across two reboots and two fan curves.

---

## 7. Anomaly detection

**Q: Why add PatchCore when you already have a detector?**
Because YOLOv8 can only find the six classes it was trained on. A defect type it
has never seen produces an empty prediction, which is indistinguishable from a
good part. PatchCore is trained only on normal surfaces and reports distance
from normal, so it can flag a defect nobody labelled.

**Q: How does it work?**
Extract mid-level ResNet-18 features — `layer2` and `layer3`, which carry
texture rather than semantics — giving 256 patch descriptors of 384 dimensions.
Store descriptors from defect-free images in a memory bank, shrunk to about 1%
by a greedy k-centre coreset. At inference, each patch's distance to its nearest
bank neighbour is computed, and the image score is the **maximum** over patches.

**Q: Why maximum and not mean?**
A defect covers a small fraction of the image. With 4 anomalous patches at 8.0
and 252 normal at 2.0, the mean is 2.09 against a normal value of 2.0 — the
signal is diluted by the patches that are working correctly. Maximum is the only
aggregation that survives.

**Q: How does it run fast enough?**
The naive nearest-neighbour builds a 7.9 GB difference tensor. Expanding
$\|a-b\|^2 = \|a\|^2 - 2ab + \|b\|^2$ turns it into one matrix multiply with a
20 MB peak, since the bank norms are precomputed. And it is gated — it only runs
when the detector's top confidence is under 0.55, capped at 3 calls per second.
About 10% of frames in practice.

**Q: How well does it work?**
Honestly, weakly. Normal median 2.261, defect median 2.402, separation +0.141
with overlapping distributions. The reason is that NEU-DET contains **no
defect-free images at all**, so the bank was built from regions outside dilated
bounding boxes — 41.3% of image area. That contaminates the bank with
unannotated defects and omits defect-adjacent context. It is a backstop for
unknown defect types, not a second accurate classifier, and the fix is
photographing 200 known-good parts on the real rig, not a different algorithm.

---

## 8. Embedded and hardware

**Q: Why not connect the relay directly to a GPIO pin?**
A relay coil needs 70–90 mA; a Pi pin supplies about 16 mA safely. The relay
module has a transistor that draws coil current from the 5 V rail while the GPIO
only drives its base, and an opto-isolator so there is no electrical path
between the Pi and the coil at all.

**Q: What is a flyback diode?**
A diode reverse-biased across an inductive load. Interrupting current through an
inductor generates $V = -L\,di/dt$, which for a solenoid is hundreds of volts.
The diode gives that current a circulating path to decay as heat. Without one it
arcs across the relay contacts, pits them, and radiates interference that can
reset microcontrollers. The relay module includes one for its own coil; the
solenoid needs its own.

**Q: Why an Arduino when the Pi has GPIO?**
Three reasons: Linux is not real-time, so a pulse width can stretch under load,
while an Arduino's is crystal-accurate; a fault destroys a ₹300 board rather
than the ₹6,000 one holding the models; and most relay modules are 5 V devices,
which the Arduino drives natively while the Pi's 3.3 V sits at the margin.

**Q: Why does `RPi.GPIO` not work on the Pi 5?**
GPIO moved out of the main SoC into the RP1 south bridge on PCIe. `RPi.GPIO`
writes the old memory-mapped registers, which no longer control the pins. The
supported path is the kernel character device via `lgpio`. On this image
`/dev/gpiochip4` is a symlink to `gpiochip0`, and that should be verified with
`gpioinfo` rather than assumed, because the numbering has changed across kernel
versions.

**Q: Is your power supply adequate?**
It is a 100 W GaN charger, not the official brick, and it negotiates 5 V/3 A —
15 W. The rail reads 4.835 V idle and 4.607 V under four-core load, just below
the ~4.63 V warning threshold, so `dmesg` shows under-voltage messages. But
`get_throttled` gained no new bits across the benchmark, so it warns and never
throttles. The 228 mV sag is about 190 mΩ of cable resistance, not insufficient
capacity. Worth knowing: a bigger GaN charger would not help, because the Pi 5
only ever negotiates 5 V — only a 5 V/**5 A** source changes anything, which is
what makes the official supply different.

---

## 9. Harder questions, and how to handle them

**Q: Your test mAP is 0.711. Is that good?**
It is mid-range for NEU-DET; published YOLO results on this dataset cluster
around 0.70–0.78 mAP@50. It is honest in that it is measured on a test split
never used for model selection, and val was 0.732, so the close agreement says
there is no selection overfitting. The number is limited by crazing more than by
anything else.

**Q: What is the weakest part of your project?**
Two things. Crazing recall at 0.222, which is a genuine detection failure on one
of six classes. And PatchCore's separation of +0.141, which is weak because the
dataset has no defect-free images. Both are data problems rather than code
problems, and both have identified fixes that need the physical rig.

**Q: You have no conveyor. Is this a real system?**
The software stack is complete and measured on the target hardware: detection,
anomaly, decision, storage, dashboard, actuator, all running as a service that
survives reboot. What is not built is the mechanical rig. The actuator layer has
three interchangeable backends chosen by one config string, and GPIO17 has been
verified toggling at 250.1 ms, so connecting real hardware changes configuration
rather than code.

**Q: What would you do differently?**
Collect defect-free images from the start. Every PatchCore limitation traces
back to NEU-DET not having any, and no amount of algorithm work fixes it. I
would also have benchmarked on the target device before making performance
claims — the roofline projection was 47% low.

**Q: How would you improve accuracy?**
In order of expected value: per-class confidence thresholds, since crazing needs
a lower one than patches; CLAHE contrast enhancement, because crazing is a
low-contrast failure; oversampling crazing during training; and only then a
larger model, which costs the frame-rate headroom.

**Q: Something you do not know the answer to.**
"I haven't measured that. Here's how I would: [method]." Then stop. The examiner
is testing whether you know the boundary of your own evidence, and inventing a
number is the one unrecoverable answer.

Three genuine open items in this project, worth naming before being asked:

- The FP32-head cost on ARM is unmeasured — only the head-excluded model was
  deployed, so there is no whole-graph INT8 figure from the Pi.
- The 250.1 ms pulse width was measured on an idle Pi, not under inference load.
- `ACTIVE_LOW` in the Arduino firmware has never been verified against a real
  relay.

---

## 10. Summary

- Answer with **numbers**. 36.6 FPS, 0.711 mAP@50, 0.222 crazing recall, 122 MB.
- State the **trade-off**, not just the choice. Every design decision here has
  one, and naming it is what distinguishes understanding from recall.
- **Volunteer limitations.** Crazing, PatchCore separation, no conveyor, three
  unmeasured items. Saying them first converts weakness into rigour.
- **Never invent a number.** "I haven't measured that, here's how I would" is a
  strong answer; a fabricated figure ends badly on the follow-up.
- The three headline claims, all measured on the device: **36.6 FPS detection,
  17.1 FPS end to end, 0.711 test mAP@50.**

Next: [21 — Bill of materials, cost and alternatives](21_bom_cost_alternatives.md).
