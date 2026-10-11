# 21 — Bill of Materials, Cost and Alternatives

**Prerequisites:** [17 — GPIO, Arduino and the relay](17_gpio_arduino_relay.md),
[18 — System architecture](18_system_architecture.md).

---

## 1. Concept

A bill of materials is not a shopping list. It is a record of **decisions**:
what each component does, why it was chosen over the alternatives, and what
would have to change for a different choice to win.

Prices below are indicative Indian retail, mid-2026, and will drift. The
*ratios* are the durable part.

---

## 2. Why it is needed

Two reasons, one academic and one practical.

**Academic.** "Why a Raspberry Pi and not a Jetson?" is a certainty at viva. The
answer must be a comparison, not a preference.

**Practical.** This project is a demonstrator that could become a real machine.
The gap between the two is mostly cost and industrial hardening, and knowing
where that gap lies is part of understanding what you have built.

---

## 3. Real-world example

A commercial machine-vision inspection station — Cognex In-Sight or Keyence
CV-X class — costs roughly ₹4–12 lakh per station, including camera, controller,
lighting and software licence.

What that money buys is not better mathematics. It buys an IP67 housing, a
five-year supply guarantee, certified lighting with documented stability, a
support contract, and validated software. **The algorithm in this project is
comparable; the industrialisation is not.** That is an honest and useful thing
to be able to say.

---

## 4. The bill of materials

### Core compute — already owned

| Item | Cost | Why this |
|---|---:|---|
| Raspberry Pi 5, 8 GB | ₹8,500 | Four Cortex-A76 at 2.4 GHz. 36.6 FPS measured. |
| Active Cooler | ₹500 | **Not optional.** Sustained inference reaches 57.6 °C with it. |
| microSD 64 GB A2 | ₹700 | A2 rating matters for SQLite's random writes. |
| USB-C supply | ₹0 | Existing 100 W GaN. Measured adequate: 15 W, no throttling. |
| micro-HDMI cable | ₹250 | The Pi 5 uses micro-HDMI, not full size. |
| **Subtotal** | **₹9,950** | |

**8 GB and not 4 GB?** Peak RSS is 122 MB, so 4 GB would be ample for the
runtime. 8 GB is only justified if you ever want to train or experiment on the
device itself. **If buying today for this workload, 4 GB saves ₹2,000 and costs
nothing.**

### Imaging — required for a real demo

| Item | Cost | Why this |
|---|---:|---|
| Camera Module 3, 12 MP AF | ₹2,500 | CSI, not USB — the USB budget is clamped to 600 mA |
| LED ring light, 12 V | ₹800 | Constant, diffuse, shadow-free |
| 12 V 1 A supply for light | ₹250 | **Never** power the light from the Pi |
| Enclosure / shroud | ₹500 | Excludes ambient light |
| **Subtotal** | **₹4,050** | |

The camera and light are the highest-value remaining purchase. Without them
there is no live demo, and PatchCore's bank cannot be rebuilt from real good
parts — which is the single fix for its weak separation.

**Why CSI over a USB webcam?** Three reasons. The measured supply negotiates
5 V/3 A, so firmware clamped total USB current to 600 mA. CSI is a dedicated
bus, so it does not compete with anything. And `picamera2` exposes manual
exposure, gain and white balance — a USB webcam's auto-exposure will silently
change the image between the day you build the PatchCore bank and the day you
run inspection, which breaks the module.

### Actuation

| Item | Cost | Why this |
|---|---:|---|
| Arduino Nano (clone) | ₹300 | Deterministic timing, isolation, 5 V domain |
| 2-channel relay module | ₹150 | Opto-isolated, transistor-driven |
| 12 V push solenoid | ₹600 | The ejector |
| 12 V 2 A supply | ₹300 | Separate rail; shares only ground |
| 1N4007 diodes (×10) | ₹20 | **Mandatory** across the solenoid |
| Breadboard + jumpers | ₹300 | |
| **Subtotal** | **₹1,670** | |

**Total for a complete working demonstrator: ₹15,670**, of which ₹5,720 is not
yet purchased.

### Not costed

A conveyor. A belt, motor, speed controller and frame is ₹8,000–20,000 and is
mechanical engineering rather than this project. A turntable or a manual slide
demonstrates the same pipeline for a fraction of it.

---

## 5. Cost intuition

### Where the money goes

```
Compute      ₹9,950   64%
Imaging      ₹4,050   26%
Actuation    ₹1,670   10%
```

Two thirds is compute, and the compute is over-specified: 8 GB against a 122 MB
footprint. **The imaging is under-specified relative to its importance** —
lighting quality affects accuracy more than any hyperparameter, and it is 5% of
the budget.

If you had ₹3,000 more to spend, none of it should go on compute.

### Cost per inspection

At 17.1 FPS, eight hours a day, 250 days a year:

$$17.1 \times 3600 \times 8 \times 250 \approx 1.23 \times 10^8\ \text{parts/year}$$

Even at 1% duty cycle that is over a million parts. Amortised hardware cost is
therefore **effectively zero per part**, which is the real argument for edge
inspection: the capital cost is trivial against throughput, and the recurring
cost is electricity — roughly 10 W, about ₹80 a year.

The comparison that matters is against a human inspector at ₹15,000–25,000 a
month, who cannot sustain attention at 17 parts per second in any case.

---

## 6. Software cost

| Component | Licence | Cost |
|---|---|---|
| Raspberry Pi OS | Debian, mostly GPL | ₹0 |
| Python, NumPy, Flask, SQLite | PSF / BSD / public domain | ₹0 |
| OpenCV | Apache 2.0 | ₹0 |
| ONNX Runtime | MIT | ₹0 |
| **Ultralytics YOLOv8** | **AGPL-3.0** | **₹0, with a condition** |

**The AGPL point matters and is a fair viva question.** Ultralytics is AGPL-3.0:
if you distribute the software, or provide it as a network service, you must
release your source under a compatible licence. For academic work this is fine.
For a commercial product it is not, and Ultralytics sells a commercial licence
for exactly that reason.

Worth noting: **only training uses Ultralytics.** The runtime is ONNX Runtime
under MIT. The trained weights themselves are not clearly a derivative work, but
the safe reading is that a commercial deployment needs either the commercial
licence or a permissively-licensed training framework.

---

## 7. Alternatives considered

### Compute platform

| Platform | ~Cost | Verdict |
|---|---:|---|
| **Raspberry Pi 5 8 GB** | ₹8,500 | **Chosen.** 36.6 FPS measured, huge community, GPIO, no vendor lock-in |
| Pi 5 4 GB | ₹6,500 | Better value for this workload. 122 MB peak RSS |
| Pi 4 4 GB | ₹4,500 | Cortex-A72 at 1.5 GHz, roughly 2.5× slower — around 14 FPS. Viable, marginal |
| Jetson Orin Nano | ₹35,000 | 40 TOPS, would run YOLOv8m comfortably. **4× the cost for accuracy we do not need**, and TensorRT is a steeper toolchain |
| Coral USB TPU + Pi | ₹8,000 + Pi | 4 TOPS INT8, very fast. But it needs full-integer TFLite, and **quantising the YOLOv8 head is exactly what destroyed our accuracy** — the head cannot stay FP32 on a TPU |
| ESP32-CAM | ₹600 | Cannot run YOLOv8 in any form. Suitable for classification of tiny images only |
| Industrial smart camera | ₹4–12 lakh | What this replaces |

The Coral entry is the interesting one. It looks ideal on paper — 4 TOPS for
₹8,000 — and our own quantisation study is the reason it fails: the Edge TPU
requires the entire graph in INT8, and we measured that quantising the detection
head costs 32 points of mAP. Running the head on the host CPU is possible but
gives back much of the speed.

### Camera

| Option | ~Cost | Verdict |
|---|---:|---|
| **Camera Module 3** | ₹2,500 | **Chosen.** CSI bus, autofocus, manual exposure control |
| Camera Module 3 Wide | ₹3,000 | Only if the field of view demands it |
| USB webcam | ₹1,000 | Competes for a clamped USB budget; auto-exposure breaks PatchCore |
| Global-shutter camera | ₹4,500 | **Correct for a fast conveyor.** Rolling shutter skews moving parts |
| Industrial GigE | ₹25,000+ | Overkill |

Global shutter deserves a note: a rolling-shutter sensor exposes rows
sequentially, so a part moving during exposure appears skewed. At 0.5 m/s and a
10 ms exposure that is 5 mm of smear. **If the conveyor is ever built, this
becomes the right purchase**, and until then a strobed light or short exposure
mitigates it.

### Actuation

| Option | ~Cost | Verdict |
|---|---:|---|
| **Nano + relay + solenoid** | ₹1,050 | **Chosen.** Deterministic, isolated, cheap |
| Pi GPIO + relay directly | ₹150 | Works — `GpioActuator` — but timing is at the scheduler's mercy |
| Pneumatic cylinder + valve | ₹3,500 | Industry standard. Faster and stronger; needs compressed air |
| Stepper-driven arm | ₹1,500 | Slower, more complex, no advantage here |

### Model

| Option | Verdict |
|---|---|
| **YOLOv8n** | **Chosen.** 36.6 FPS, 0.711 mAP@50 |
| YOLOv8s | ~3× compute, a few points more mAP. Would fall under the FPS target |
| YOLOv5n | Comparable; v8 is anchor-free and simpler to decode |
| MobileNet-SSD | Faster, notably weaker on small defects |
| Faster R-CNN | Two-stage, far more accurate, far too slow for edge CPU |
| Pure PatchCore | No classification, and our separation is only +0.141 |

---

## 8. Common mistakes

**Buying compute first.** The instinct is a faster board. The measurements say
compute is 64% of the budget and already 84% above target, while lighting — 5%
of the budget — is the largest untested variable.

**Powering the light or solenoid from the Pi.** The USB budget is clamped to
600 mA and the rail already sags 228 mV under CPU load. Both need their own
supply.

**Skipping the Active Cooler.** ₹500 against thermal throttling that makes every
benchmark unreproducible.

**Buying an official PSU reflexively.** Measured first: 15 W, warnings but no
throttling, benchmark valid. And a bigger GaN charger would not help — the Pi 5
only negotiates 5 V, so only a 5 V/**5 A** source changes anything.

**Genuine Arduino at ₹1,800.** A ₹300 clone is electrically identical for this.
Spend the difference on lighting.

**Ignoring AGPL.** Fine academically, a real constraint commercially.

**Forgetting micro-HDMI.** ₹250, and easy to discover an hour before a review.

---

## 9. Interview questions

**Q: Why a Raspberry Pi and not a Jetson?**
Cost against requirement. The requirement was 20 FPS; the Pi delivers 36.6 at
₹8,500. A Jetson Orin Nano is ₹35,000 and would allow a larger model, but our
accuracy ceiling is set by crazing being physically hard to see, not by model
capacity — so 4× the cost buys very little. The Pi also has GPIO, so no separate
interface board is needed.

**Q: Would a Coral TPU be faster?**
Yes, and we would not use it. The Edge TPU requires the whole graph in
full-integer INT8, and we measured that quantising the YOLOv8 detection head
drops mAP@50 from 0.735 to 0.408. The 32 points we recovered by keeping the head
in FP32 cannot be recovered on a TPU without splitting the model and running the
head on the host, which gives back much of the speed advantage.

**Q: What is the most under-specified part of your system?**
Lighting. It is 5% of the budget and affects accuracy more than any
hyperparameter, and PatchCore in particular measures distance from a bank
recorded under specific illumination. If the light changes, every part becomes
anomalous.

**Q: What licence is your project under, and can it be sold?**
The runtime stack is permissive — MIT, BSD, Apache. Training uses Ultralytics
YOLOv8, which is AGPL-3.0, so distributing or network-serving it requires
releasing source under a compatible licence or buying Ultralytics' commercial
licence. Academically fine; commercially a real decision.

**Q: How much would a commercial equivalent cost?**
A Cognex or Keyence station is roughly ₹4–12 lakh. The difference is not the
algorithm — it is IP67 housing, certified stable lighting, a supply guarantee,
support and validation. Ours is a demonstrator at 1–3% of the cost with
comparable detection performance and none of the industrialisation.

**Q: What is the running cost?**
About 10 W, roughly ₹80 a year of electricity. Amortised hardware is effectively
zero per part at any realistic throughput. The comparison is a human inspector
at ₹15,000–25,000 a month who cannot sustain 17 parts per second.

---

## 10. Summary

- **₹15,670** for a complete demonstrator; **₹5,720** still to buy, almost all
  of it imaging and actuation.
- **64% of spend is compute, and it is over-specified** — 8 GB against a 122 MB
  footprint. 4 GB would have been the better buy.
- **Imaging is the highest-value remaining purchase.** Lighting affects accuracy
  more than any hyperparameter and is 5% of the budget.
- **CSI camera, not USB**: the USB budget is clamped to 600 mA, and a webcam's
  auto-exposure silently breaks PatchCore.
- **Global shutter becomes correct once a conveyor exists** — rolling shutter
  skews moving parts.
- A **Coral TPU is rejected by our own measurement**: it needs a fully INT8
  graph, and the INT8 head costs 32 points of mAP.
- **Ultralytics is AGPL-3.0.** Training only; the runtime is MIT. It matters
  commercially, not academically.
- Commercial equivalents are ₹4–12 lakh. The gap is industrialisation, not
  algorithms.

---

**End of the course.** Back to [00 — Start here](00_start_here.md).
