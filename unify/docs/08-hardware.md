# Section 9 — Hardware

## 9.0 Bill of materials and role

| Component | Role | Key spec | Cost (approx, INR) |
|---|---|---|---|
| Laptop + RTX GPU | Training | CUDA, ≥6 GB VRAM | – |
| Raspberry Pi 4 (8 GB) | Edge inference, API, dashboard | Cortex-A72 ×4 @1.5 GHz, ARMv8-A + NEON | 6,000–8,000 |
| 32 GB microSD (or USB SSD) | OS + model + DB | Class 10 / A2 | 500 |
| 5 V / 3 A USB-C PSU | Powers the Pi | must be 3 A | 700 |
| ESP32 DevKit V1 | Acquisition + Wi-Fi | Xtensa LX6 ×2 @240 MHz, 520 KB SRAM | 400 |
| MPU-6050 (GY-521) | Vibration sensing | 3-axis accel + 3-axis gyro, I2C | 150 |
| 12 V DC motor | Machine under test | geared or plain BO motor | 300 |
| L298N module | Motor driver | dual H-bridge, 2 A/ch | 150 |
| 12 V / 2 A adapter | Motor supply | separate from Pi supply | 400 |
| Breadboard + jumpers | Prototyping | – | 200 |

---

## 9.1 Laptop with RTX GPU — the training machine

**Why a GPU.** Training is dominated by dense matrix multiplies. A CPU has ~8 powerful cores; an RTX GPU has thousands of simple cores plus dedicated Tensor Cores, and 5–20× the memory bandwidth. For your workload expect **20–60× wall-clock speedup**, turning a 6-hour experiment into 10 minutes — which matters because you will run 30 experiments, not one.

**Practicals.**

- Verify with `torch.cuda.is_available()` → must print `True`. If `False`, you installed the CPU wheel; reinstall from the CUDA index URL.
- Your model is tiny (44k params), so you will be **data-loader-bound**, not GPU-bound. Set `num_workers=4, pin_memory=True, persistent_workers=True`. If GPU utilisation (`nvidia-smi`) sits below 30 %, the bottleneck is your Python data path, not the GPU. Cache preprocessed tensors as `.npy` and the problem disappears.
- Mixed precision (`torch.autocast`) is barely worth it at this size. Skip it.

**Honest note for the viva:** for a model this small you could train on a laptop CPU in an hour or two. The GPU is a convenience, not a requirement. Say that if asked — overclaiming hardware necessity is a common student error.

---

## 9.2 Raspberry Pi 4 (8 GB) — the edge node

### SoC: Broadcom BCM2711

| Spec | Value | Consequence for you |
|---|---|---|
| CPU | 4× Cortex-A72 @ 1.5 GHz | Enough for ~2 ms inference |
| Architecture | ARMv8-A, 64-bit | Install **64-bit** Raspberry Pi OS |
| Pipeline | Out-of-order, 3-wide superscalar | Real application core, not an MCU |
| SIMD | NEON, 128-bit | 4× fp32 or 16× int8 lanes — the reason ORT is fast |
| L1 cache | 32 KB I + 32 KB D per core | – |
| L2 cache | 1 MB shared | **Keep your model under ~500 KB and it stays resident** |
| RAM | 8 GB LPDDR4-3200 | Vastly more than you need |
| GPU | VideoCore VI | No usable ONNX EP; ignore it |
| Networking | Gigabit Ethernet, 802.11ac Wi-Fi, BT 5.0 | ESP32 link |
| GPIO | 40-pin: I2C ×2, SPI ×2, UART, PWM | Could read sensors directly if needed |
| Power | ~3 W idle, 6–7 W loaded | USB-C, needs a genuine 3 A supply |
| Thermal | Throttles ~80–85 °C | Fit a heatsink |

### Why the A72 is genuinely adequate

Rough arithmetic for your model: dominated by the CNN. Per sensor, per window:

```
Conv1: 30 positions × 32 out-ch × (1 in-ch × 5) ≈    4,800 MACs
Conv2: 30 × 64 × (32 × 3)                       ≈  184,000 MACs
Conv3: 15 × 64 × (64 × 3)                       ≈  184,000 MACs
                                        per sensor ≈  0.37 MMAC
× 16 sensors                                       ≈  6 MMAC
+ attention & head                                 ≈  negligible
                              ≈ 12 MFLOP per inference
```

The A72 with NEON does roughly 4–8 GFLOP/s per core on well-vectorised fp32. Even at 10 % efficiency on one core, that is **~2–3 ms**. Measure it; expect single-digit milliseconds, and expect the SD card and the network to dominate your end-to-end latency, not the maths.

### Practical setup

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev sqlite3
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install numpy onnxruntime fastapi uvicorn
vcgencmd measure_temp          # watch this during load
vcgencmd get_throttled         # 0x0 = never throttled; anything else = you need cooling
```

**Limitations to state honestly:** no hardware NPU; no ECC memory; SD cards fail (use a USB SSD or at minimum a good A2 card and back up); USB and Ethernet share bandwidth on the Pi 4 less problematically than on the Pi 3 but still not infinite; power supply quality is the number-one cause of mysterious instability — under-voltage shows up as random corruption, not as an error message.

---

## 9.3 ESP32 — the sensing endpoint

| Spec | Value | Note |
|---|---|---|
| CPU | Xtensa LX6 dual-core @ 240 MHz | Core 0 usually runs the Wi-Fi stack; put sampling on core 1 |
| SRAM | 520 KB | Your ring buffer must fit here |
| Flash | 4 MB typical | Firmware |
| ADC | 12-bit SAR ×2 | Not used — MPU-6050 has its own ADC and gives digital output over I2C |
| Wi-Fi | 802.11 b/g/n, 2.4 GHz only | **Not 5 GHz** — your router/hotspot must expose 2.4 GHz |
| I2C | 2 controllers, up to 400 kHz (1 MHz possible) | Sensor link |
| FPU | Single-precision hardware FPU | Feature maths in float is fine |
| Power | 80–260 mA active with Wi-Fi | Wi-Fi TX spikes to ~500 mA; a weak supply causes brownout resets |

**Why an ESP32 rather than reading I2C from the Pi directly?**

1. **Physical separation.** The sensor sits on a vibrating motor; the Pi sits on a desk. I2C is a ~1 m, unshielded, ground-referenced bus — it is not a field wiring standard. Wi-Fi bridges the distance properly.
2. **Deterministic timing.** A bare-metal loop on the ESP32 samples at a much steadier rate than Linux user-space on the Pi, which can be preempted for tens of milliseconds. Sampling jitter directly corrupts frequency-domain features.
3. **It mirrors industrial architecture** — sensor nodes at the machine, one gateway per line.
4. **It demonstrates edge preprocessing** — 1000 samples become 8 features before transmission.

**Dual-core tip.** Pin the sampling task to core 1 with `xTaskCreatePinnedToCore(..., 1)` and leave core 0 for Wi-Fi. Otherwise the Wi-Fi stack's interrupts inject jitter into your sampling loop. This is the single most valuable ESP32 implementation detail in the project.

---

## 9.4 MPU-6050 — and its honest limitations

### What it is

InvenSense 6-axis MEMS IMU: 3-axis accelerometer + 3-axis gyroscope, 16-bit ADCs, I2C interface, on-board Digital Motion Processor (unused here).

| Parameter | Value |
|---|---|
| I2C address | 0x68 (AD0 low), 0x69 (AD0 high) |
| Accel ranges | ±2 / ±4 / ±8 / ±16 g |
| Accel sensitivity | 16384 / 8192 / 4096 / 2048 LSB per g |
| Gyro ranges | ±250 / 500 / 1000 / 2000 °/s |
| Output data rate | up to 1 kHz for accel (8 kHz gyro internal) |
| DLPF | configurable, ~5 Hz to ~260 Hz bandwidth (`DLPF_CFG` 0–6) |
| Supply | 3.3 V (GY-521 boards have an on-board regulator, so 5 V-tolerant VCC) |
| Noise density | ~400 µg/√Hz |

### Configuration for vibration work

```cpp
// MPU6050 register setup for maximum useful bandwidth
writeReg(0x6B, 0x00);   // PWR_MGMT_1: wake up
writeReg(0x1A, 0x00);   // CONFIG: DLPF_CFG=0 → accel bandwidth 260 Hz, 1 kHz sample rate
writeReg(0x19, 0x00);   // SMPLRT_DIV=0 → 1 kHz output rate
writeReg(0x1C, 0x10);   // ACCEL_CONFIG: ±8 g (0x18 = ±16 g)
```

Range choice: a healthy small DC motor produces well under 1 g of vibration, so ±2 g gives the best resolution. But a failing bearing produces impulses — start at ±4 g or ±8 g and check for clipping (`|raw| == 32767`). Log a clipping counter; if it is non-zero, increase the range. That check is a five-line addition that separates a careful engineer from a hopeful one.

### The bandwidth problem — say this before your examiner does

```
FEMTO / XJTU sampling:   25,600 Hz   → usable spectrum to 12,800 Hz
MPU-6050 maximum:         1,000 Hz   → usable spectrum to    500 Hz
                                        (DLPF limits it to ~260 Hz anyway)

                         ≈ 25× lower sampling rate, ≈ 50× less usable bandwidth
```

Bearing defect signatures live partly in the **kHz** region: defect impulses excite structural resonances typically between 1 kHz and 20 kHz, which is precisely why the datasets sample at 25.6 kHz and why envelope analysis band-passes around those resonances.

**The MPU-6050 physically cannot see them.**

What it *can* see: shaft rotation (a 3000 rpm motor = 50 Hz), imbalance at 1× rotation, misalignment at 2×, looseness harmonics, and gross overall vibration level (RMS trend). These are real, useful condition indicators — they are what ISO 10816 overall-velocity monitoring uses — but they are **low-frequency machine faults, not incipient bearing spalls**.

### What to do about it — three options, ranked

1. **Reframe the rig's purpose (recommended, zero cost).** The motor rig demonstrates the *live pipeline* — acquisition, features, wireless transport, edge inference, dashboard, latency, alerting — and it detects gross faults you can induce (add an unbalanced mass to the shaft, loosen a mount, vary the load). Validated RUL accuracy comes from the datasets. Two claims, clearly separated, both honest.

2. **Induce the faults the MPU-6050 can actually see.** Tape a small weight to the motor shaft to create imbalance → RMS and the 1× component rise measurably. Progressive load via the L298N PWM duty → measurable change. This gives you a genuine, reproducible "degradation" demo that is visible within the sensor's bandwidth, and you can *label* it because you created it.

3. **Upgrade the sensor (if budget allows, ~₹1,500–3,000).** An **ADXL345** (up to 3200 Hz ODR, ±16 g) is a modest step up. An **ADXL1002/ADXL1005** (analogue, 11–21 kHz bandwidth, ±50 g) sampled by the ESP32's ADC or an external ADC is a real vibration sensor and would put you in the same bandwidth class as the datasets. If you want the project to be genuinely publishable, this is the upgrade.

Whichever you choose, **put the bandwidth comparison table in your report.** An examiner who spots the limitation and finds you already documented it will grade you higher, not lower. One who spots it and finds you silent will conclude you did not understand your own system.

### Mounting matters more than you expect

A sensor taped to a motor with double-sided foam measures the tape, not the motor. Mounting method determines usable bandwidth:

| Mounting | Usable to |
|---|---|
| Stud-mounted (screwed) | ~10 kHz+ |
| Adhesive (cyanoacrylate/epoxy) | ~5 kHz |
| Magnet | ~2 kHz |
| Handheld probe | ~1 kHz |
| **Double-sided foam tape** | **~few hundred Hz** ← don't |

Use epoxy or a screwed bracket, and mount **radially on the bearing housing**, as close to the load zone as possible — not on the plastic case. Keep the orientation constant across all experiments, and record it. A mounting change mid-experiment invalidates every comparison.

---

## 9.5 DC motor, L298N driver and power

### The motor

A 12 V brushed DC motor stands in for a rotating machine. Brushed motors are electrically noisy (commutator arcing) — a feature, not a bug, since real machines are noisy too, but it does mean you need decoupling and separated grounds.

Speed control is by PWM duty cycle. Speed is roughly proportional to applied voltage under constant load; measure actual rpm if you want a real shaft-frequency reference (a ₹60 IR photo-interrupter and a slotted disc gives you tachometer pulses, and knowing `f_r` lets you compute the true 1× and 2× components — a strong, cheap addition).

### L298N — and why it is a mediocre choice you can still defend

Dual H-bridge, based on **bipolar** transistors. Each channel: up to 2 A, 46 V.

**Its flaw:** a saturation voltage drop of roughly **1.4–2.0 V per side**, so ~2–3 V total is lost as heat inside the chip. From a 12 V supply the motor sees ~9–10 V, and at 1 A the L298N dissipates 2–3 W — it needs its heatsink and will still get hot. Modern MOSFET drivers (**TB6612FNG**, ~0.1 Ω on-resistance; **DRV8871**; **BTS7960** for high current) are far more efficient.

**Why it is nonetheless fine here:** it works, it is ubiquitous, it is ₹150, and motor efficiency is not what you are measuring. If asked, give exactly that answer plus the TB6612FNG alternative — knowing *why* your cheap part is cheap is the mark of an engineer.

**Wiring:**

```
12 V adapter (+) ──── L298N  +12V
12 V adapter (−) ──── L298N  GND ──┬── ESP32 GND        ← COMMON GROUND, mandatory
                                    └── Pi GND (if co-located)
L298N  ENA  ←── ESP32 GPIO (PWM, ledcWrite)
L298N  IN1  ←── ESP32 GPIO
L298N  IN2  ←── ESP32 GPIO
L298N  OUT1/OUT2 ──── motor terminals
L298N  5V   ──→ (on-board regulator; leave the 5V-EN jumper ON only if Vin ≤ 12 V,
                 and do NOT also feed 5 V in from elsewhere)
```

### Power — the section that saves your demo

Three rules:

1. **Separate supplies for logic and motor.** The Pi gets its own 5 V/3 A USB-C; the motor gets its own 12 V/2 A. Powering the motor off the Pi's 5 V rail will brown out the Pi at the worst possible moment.
2. **Common ground, always.** All grounds tied together, or I2C and PWM signals have no reference and behave randomly. This is the single most common wiring failure in student projects.
3. **Suppress motor noise.** Brushed motors inject huge spikes onto the supply and radiate EMI that corrupts I2C:
   - 0.1 µF ceramic capacitors across the motor terminals, and from each terminal to the motor case
   - 100–470 µF electrolytic across the L298N's 12 V input
   - Keep the MPU-6050's I2C wires short (< 20 cm) and away from motor leads; twist SDA/SCL with ground if you can
   - A flyback/TVS is already inside the L298N, but keep the motor leads twisted

**Symptom to recognise:** if the MPU-6050 starts returning `0` or `-1` or the I2C bus hangs when the motor spins up, you have a noise or grounding problem, not a software bug. Add a Wire timeout (`Wire.setTimeOut(50)`) and a bus-recovery routine (re-init on repeated failure) so the firmware survives it rather than freezing.

---

## 9.6 Full system wiring

```
   ┌──────────── 12 V / 2 A ────────────┐
   │                                     │
   ▼                                     ▼
┌─────────┐   OUT1/2   ┌─────────┐   ┌────────┐
│  L298N  │───────────▶│ 12 V DC │   │  MPU   │
│  driver │            │  motor  │◀──│  6050  │ (mounted radially on the housing,
└────┬────┘            └─────────┘   └───┬────┘  epoxy or bracket, NOT foam tape)
     │ ENA,IN1,IN2                        │ SDA (GPIO21)
     │ (PWM + dir)                        │ SCL (GPIO22)
     │                                    │ VCC 3.3 V, GND
     └────────────┬───────────────────────┘
                  ▼
            ┌───────────┐
            │  ESP32    │  samples 1 kHz, computes features every 1 s
            │ DevKit V1 │
            └─────┬─────┘
                  │ Wi-Fi 2.4 GHz, HTTP POST JSON
                  ▼
            ┌───────────┐
            │  Pi 4 8GB │  ONNX Runtime + FastAPI + SQLite + dashboard
            └─────┬─────┘
                  │ HTTP :8000
                  ▼
              browser / projector
```

All grounds common. Pi on 5 V/3 A USB-C. Motor on its own 12 V.

---

## Summary

- Laptop+GPU trains (a convenience, not a necessity at 44k parameters); Pi 4 infers.
- Pi 4: Cortex-A72, NEON SIMD, 1 MB L2 — keep the model small and it stays cached; ~2–5 ms inference expected.
- ESP32 acquires and preprocesses; pin sampling to core 1 away from the Wi-Fi stack.
- MPU-6050 tops out at 1 kHz / ~260 Hz DLPF bandwidth versus the datasets' 25.6 kHz. Document this limitation explicitly and reframe the rig as a pipeline demonstrator plus low-frequency fault detector.
- L298N is lossy but adequate; separate supplies, common ground, capacitors across the motor.

## Key takeaways

1. The bandwidth gap between MPU-6050 and the bearing datasets is the most important honest caveat in your entire project. Own it on a slide.
2. Sensor mounting determines usable bandwidth — foam tape throws away everything above a few hundred Hz.
3. Common ground and separate supplies prevent 80 % of hardware demo failures.
4. Model size (≤500 KB) is chosen against the Pi's 1 MB L2 cache, not arbitrarily.

## Common interview questions

1. Why does a GPU accelerate training more than inference?
2. What is SIMD and why does it matter for CPU inference?
3. Why 3.3 V vs 5 V logic, and is the MPU-6050 5 V-tolerant? (The bare chip is 3.3 V; GY-521 boards add a regulator on VCC — but the I2C lines are still 3.3 V logic, which is fine for the ESP32.)
4. Explain PWM speed control of a DC motor.
5. Why does an H-bridge allow reverse rotation?
6. What causes voltage drop in an L298N and how would you avoid it?

## Common viva questions

1. "Your sensor samples at 1 kHz but your training data is 25.6 kHz. How is that valid?" → The full honest answer from §9.4. Have it rehearsed. Lead with it before they ask.
2. "Why not read the sensor directly from the Pi's I2C?" → Physical separation, sampling jitter under Linux, architectural realism.
3. "How do you know the vibration you measure is from the bearing and not from the table?" → Mounting location, and a baseline recording with the motor off. Do record that baseline — it is 30 seconds of work and a perfect answer.
4. "What is the current draw of your system?" → Measure it. Pi ~0.6–1.4 A at 5 V, ESP32 ~0.1–0.3 A, motor depends on load.
5. "What happens if the motor's EMI corrupts I2C?" → Timeout, retry, bus recovery in firmware; capacitors and grounding in hardware. Show the code.

## Practical implementation tips

- Record a **10-minute baseline with the motor off**, then **10 minutes healthy running**, then **10 minutes with an added imbalance mass**. Three labelled recordings = a real, if small, experiment you can plot. Do this in week 1; it is the cheapest data you will ever collect.
- Log a **clipping counter** and a **sample-interval jitter histogram** from the ESP32. If jitter exceeds ~5 % of the sample period, your features are unreliable — move to a hardware timer ISR.
- Buy a **spare ESP32 and a spare SD card**. Both fail, always the night before a review.
- Photograph your wiring once it works. Rebuilding from a photo beats rebuilding from memory.

## Suggested improvements for this project

1. **Add a tachometer** (IR photo-interrupter + slotted disc, ~₹60). Knowing shaft frequency `f_r` lets you compute order-normalised features (1×, 2×, 3×), which is what real vibration analysis does and which makes your features speed-invariant.
2. **Upgrade to ADXL345 or ADXL1002** if budget permits — closes the bandwidth gap and turns a caveat into a capability.
3. **Add a current sensor** (ACS712 or INA219) on the motor supply. Motor current is a genuinely independent degradation channel, gives your model a second real sensor, and costs ₹150. Strongly recommended: it makes the live rig multi-sensor, which better exercises your multi-sensor architecture.
4. **Add a DS18B20 or use the MPU-6050's internal temperature register** for a third channel. FEMTO includes temperature; matching that is easy and makes the live sensor set look more like the training sensor set.
5. **Use a hardware timer ISR** on the ESP32 for sampling instead of `delay()` in `loop()`. Removes jitter, which directly improves feature quality.
