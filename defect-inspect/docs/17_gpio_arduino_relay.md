# 17 — GPIO, Arduino and the Relay

**Prerequisites:** [13 — CPU optimisation for the Pi 5](13_cpu_optimization_pi5.md).

---

## 1. Concept

Everything so far has been information. This module is where the system touches
the physical world: a decision becomes a voltage, the voltage closes a relay,
the relay energises a solenoid, and the solenoid pushes a defective part off the
conveyor.

Three electrical domains, deliberately isolated from each other:

```
  Pi 5             Arduino Nano        Relay module      Solenoid
  3.3 V logic      5 V logic           opto-isolated     12-24 V, amps
  GPIO17  ------>  D2 (serial 'E') --> IN pin  ------>   COM/NO contacts
                                          |
                                     separate supply ----+
```

---

## 2. Why it is needed

A defect detected and not acted upon has changed nothing. The value of the whole
pipeline is realised in the 250 ms that a solenoid fires.

It is also the part that can destroy hardware. Software mistakes throw
exceptions; electrical mistakes release smoke. **The Pi's GPIO pins have no
protection whatsoever** — no series resistance, no clamping, no current limit
beyond what the SoC can survive. A wiring error kills the RP1 south bridge, and
that is not a replaceable component.

---

## 3. Real-world example

A pneumatic reject arm on a bottling line. The vision system sits above the
conveyor; 400 mm downstream an air cylinder waits. When a bottle is rejected,
the PLC waits for the bottle to travel that distance and then fires a 24 V
solenoid valve, which releases compressed air, which extends the cylinder, which
pushes the bottle into a chute.

Two details that transfer directly to this project:

**The delay is a real parameter.** The part is not at the ejector when the
decision is made. `actuator.eject_delay_s` exists for exactly this, and its
value is `conveyor_distance / belt_speed` — measured, not guessed.

**The solenoid never shares a supply with the controller.** Its inrush current
would brown out the logic.

---

## 4. How it works

### Pi 5 GPIO electrical limits

| Property | Value |
|---|---|
| Logic high | 3.3 V (**not** 5 V tolerant) |
| Safe current per pin | ~16 mA |
| Total across all pins | ~50 mA |

A 5 V signal applied to a Pi input can damage the pin permanently. A typical
relay coil needs 70–90 mA — **five times what a pin can supply.** Wiring a bare
relay coil to a GPIO pin is the single most common way beginners destroy a Pi.

### The RP1 change on the Pi 5

On earlier Pis, GPIO was memory-mapped in the main SoC and `RPi.GPIO` poked
registers directly. On the Pi 5, GPIO moved into a separate **RP1 south bridge**
chip on PCIe. `RPi.GPIO` therefore does not work on a Pi 5 — it fails, or worse,
silently does nothing.

The supported paths are the kernel character device `/dev/gpiochip*` via
**`lgpio`**, or `gpiozero` layered on top.

**Measured on this Pi:**

```
/dev/gpiochip0 -> pinctrl-rp1 (54 lines)
/dev/gpiochip4 -> symlink to gpiochip0
gpiochip0 line 17: "GPIO17"  unused  input  active-high
```

`gpiochip4` is a compatibility symlink to `gpiochip0`, so
`lgpio.gpiochip_open(4)` in `actuator.py` reaches the right controller. **Verify
this on any new image** — the numbering has changed across kernel versions:

```bash
gpioinfo | grep GPIO17
```

### Why an Arduino at all

The Pi can drive a relay module directly, and `GpioActuator` does exactly that.
The Arduino exists for three reasons.

**1. Timing determinism.** Linux is not a real-time operating system. When
`time.sleep(0.25)` returns is at the scheduler's discretion, and with four cores
saturated by inference the pulse can stretch. An Arduino runs one loop with no
OS; its timing is crystal-accurate.

Measured on an *idle* Pi: 250.1 ms, five times running, no measurable jitter.
That is better than expected — but it is the idle case, and the honest statement
is that it must be re-measured under full inference load before being trusted.

**2. Electrical isolation.** A fault on the actuator side reaches an Arduino
that costs ₹300, not a Pi that costs ₹6,000 and holds the models.

**3. Voltage domain.** Most relay modules are 5 V. The Arduino is a 5 V device
and drives them natively; the Pi is 3.3 V and sits at the margin of what some
modules recognise as logic high.

### The relay

A relay is a switch operated by an electromagnet. The important property is
**galvanic isolation**: the coil circuit and the contact circuit share no
electrical connection. 24 V and several amps switch on one side while 5 V logic
sits safely on the other.

A relay *module* adds three things to the bare relay:

- a **transistor** to supply coil current the microcontroller cannot,
- an **opto-isolator** so even the coil-drive signal is optically coupled,
- a **flyback diode** across the coil.

Contacts: **COM** (common), **NO** (normally open, closed only when energised),
**NC** (normally closed, opens when energised). Use **NO** for an ejector — power
loss then means "do not fire", which is the safe state.

### Flyback: the non-negotiable diode

A solenoid is an inductor. Inductors resist changes in current, and

$$V = -L\frac{di}{dt}$$

When the switch opens, $di/dt$ is enormous — current goes from amps to zero in
microseconds. A 100 mH coil dropping 1 A in 1 µs generates:

$$V = 0.1 \times \frac{1}{10^{-6}} = 100{,}000\ \text{V}$$

Real voltages are limited by parasitic capacitance and arcing to hundreds of
volts, which is still far beyond anything on the board. It arcs across the relay
contacts, pits them, and radiates interference that resets microcontrollers.

A **1N4007 diode reverse-biased across the solenoid coil** gives that current a
path to circulate and decay harmlessly. Cathode (banded end) to the positive
supply. It costs ₹2 and is mandatory.

Note the distinction: relay modules include a flyback diode for *their own coil*.
The **solenoid needs its own**, and no module provides it.

### The serial protocol

Deliberately minimal — one byte:

```
Pi -> Arduino:  'E'      fire the ejector
Arduino -> Pi:  "OK\n"   acknowledgement
```

No framing, no checksum, no length field. On a 30 cm USB cable in a lab, the
error rate is negligible and complexity is the bigger risk. A 10 m cable in a
plant with VFDs on the same rack would need framing and a CRC — the protocol
would be wrong there, and the reason is distance and noise, not principle.

---

## 5. Mathematical intuition

### Base resistor sizing, if driving a transistor directly

Suppose a relay coil needs 80 mA at 5 V, switched by a 2N2222 with $h_{FE}
\approx 100$.

Base current for saturation, with a safety factor of 5 (transistors are much
worse at saturation than $h_{FE}$ suggests):

$$I_B = \frac{I_C}{h_{FE}} \times 5 = \frac{0.08}{100} \times 5 = 4\ \text{mA}$$

$$R_B = \frac{V_{GPIO} - V_{BE}}{I_B} = \frac{3.3 - 0.7}{0.004} = 650\ \Omega$$

Use 560 Ω or 470 Ω from the E12 series. Note 4 mA is well inside the Pi's 16 mA
budget — **that is the point of the transistor.**

A relay module does all of this for you, which is why buying one is correct.

### Ejection timing

$$t_{\text{delay}} = \frac{d}{v}$$

For a 400 mm gap and a 0.5 m/s belt:

$$t = \frac{0.4}{0.5} = 0.8\ \text{s}$$

The window in which the part is within the ejector's reach:

$$t_{\text{window}} = \frac{w_{\text{part}} + w_{\text{ejector}}}{v}$$

For a 50 mm part and 30 mm pusher at 0.5 m/s, that is 160 ms. **The 250 ms pulse
must start inside that window**, so the delay accuracy budget is about ±80 ms.
Comfortable for an Arduino, tight for a loaded Linux scheduler — which is the
quantitative version of reason 1 for the Arduino.

---

## 6. Software implementation

`src/inspector/actuator.py` defines a Protocol and three backends chosen by one
config string:

```yaml
actuator:
  backend: mock      # mock | serial | gpio
  pin: 17
  pulse_ms: 250
  eject_delay_s: 0.0
```

Nothing in the pipeline knows which backend is active.

### The startup glitch, and why the claim order matters

```python
self.chip = lgpio.gpiochip_open(4 if self._is_pi5() else 0)
lgpio.gpio_claim_output(self.chip, pin, 0)   # claim ALREADY LOW
```

Claiming an output without specifying the initial level can leave the pin
momentarily high — which fires the solenoid at every boot. The Arduino sketch
does the same in the other language:

```c
digitalWrite(RELAY_PIN, ACTIVE_LOW ? HIGH : LOW);  // set level FIRST
pinMode(RELAY_PIN, OUTPUT);                        // then enable the driver
```

Reversing those two lines produces a pulse on every reset. **Set the level, then
enable the output.** In both languages, in that order.

### Degrading rather than stopping

```python
def build_actuator(cfg):
    try:
        ...
    except Exception as exc:
        log.error("actuator %s failed (%s); falling back to mock", name, exc)
        return MockActuator()
```

An unplugged Arduino must not stop inspection. The system keeps detecting,
keeps recording, and logs loudly that ejection is not happening. Refusing to run
would be the worse failure: no data at all instead of data plus a warning.

### The serial reset

```python
self.link = serial.Serial(port, baud, timeout=0.2)
time.sleep(2.0)          # DTR assertion resets the Nano
```

Opening a serial port asserts DTR, which resets an Arduino Nano. The bootloader
then runs for about two seconds, discarding anything sent. Without the sleep,
the first eject command vanishes — and only the first, which makes it look
random.

### Arduino side

`firmware/ejector/ejector.ino`:

```c
if (millis() - lastFire < MIN_INTERVAL_MS) return;   // overflow-safe
```

Written as a subtraction, not `millis() < lastFire + MIN_INTERVAL`. `millis()`
overflows after 49.7 days; unsigned subtraction still yields the correct
interval across the wrap, while the addition form does not.

`MIN_INTERVAL_MS = 300` prevents a burst of detections from chattering the
relay — mechanically destructive and electrically noisy.

---

## 7. Hardware implementation

### Wiring

```
Pi GPIO17 (pin 11) ------> Relay module IN
Pi 5V     (pin 2)  ------> Relay module VCC     [see note]
Pi GND    (pin 6)  ------> Relay module GND
                           Relay COM  ------> Solenoid +
                           Relay NO   ------> PSU +12V
                           Solenoid - ------> PSU GND
                           1N4007 across solenoid, cathode to +
```

**Grounds must be common** between the Pi/Arduino and the relay logic side, or
the logic signal has no reference. The **solenoid supply must be separate** —
share only ground, never the positive rail.

### Verify ACTIVE_LOW before connecting anything

Most cheap relay modules are **active low**: the relay energises when the input
is pulled to 0 V. Get this backwards and the relay is energised whenever the
system is idle, and de-energises to eject — meaning power-on fires the solenoid
and holds it.

`ejector.ino` has `ACTIVE_LOW = true`, which has **never been verified against
real hardware**. The test, with the solenoid disconnected:

1. Power the relay module from the Arduino, nothing on the contacts.
2. Watch the module's LED and listen for the click.
3. Idle should be silent and unlit. If it clicks on power-up and releases when
   ejecting, flip the constant.

Only then connect the solenoid.

### Power budget

Measured on this Pi: the supply negotiates 5 V/3 A and the USB peripheral budget
is clamped to 600 mA. The relay module's coil draw is small, but **the solenoid
must never be powered from the Pi.** Its inrush is amps, and the rail already
sags 228 mV under CPU load alone.

---

## 8. Common mistakes

**Connecting a relay coil directly to a GPIO pin.** 80 mA demanded from a 16 mA
pin. Destroys the RP1 controller.

**Omitting the solenoid's flyback diode.** Hundreds of volts across the
contacts, pitted relays, resets, and eventual failure. The relay module's
built-in diode protects *its own* coil, not your solenoid.

**Using `RPi.GPIO` on a Pi 5.** Does not work. Use `lgpio` or `gpiozero`.

**Assuming `gpiochip4`.** Correct on this image, wrong on others. Run `gpioinfo`.

**Getting `ACTIVE_LOW` backwards.** Solenoid energised at idle. Test with the
load disconnected.

**Enabling the output before setting its level.** Fires on every boot and reset.

**Forgetting the 2 s wait after opening the serial port.** The first command
disappears, intermittently.

**Sharing a supply between solenoid and logic.** Inrush browns out the
controller; the Pi reboots mid-inspection.

**Believing the pulse width.** 250.1 ms measured on an idle Pi is not 250 ms on
a Pi running four inference threads. Re-measure under load.

**Ejecting immediately.** The part is downstream. `eject_delay_s` defaults to
0.0 because there is no conveyor yet; with one, it is $d/v$ and must be measured.

---

## 9. Interview questions

**Q: Why not connect the relay straight to a GPIO pin?**
A coil needs 70–90 mA; a Pi pin supplies about 16 mA safely. The relay module's
transistor supplies coil current from the 5 V rail while the GPIO only drives
the transistor's base, a few milliamps. The opto-isolator additionally means no
electrical path exists between the Pi and the coil at all.

**Q: What is a flyback diode and what happens without one?**
A diode reverse-biased across an inductive load. When current is interrupted,
$V = -L\,di/dt$ generates a large reverse voltage — hundreds of volts in
practice. The diode gives that current a circulating path so it decays as heat.
Without it, the spike arcs across the relay contacts, pits them, and radiates
interference that can reset microcontrollers.

**Q: Why use an Arduino when the Pi has GPIO?**
Timing determinism, electrical isolation, and voltage domain. Linux is not
real-time, so a pulse width can stretch under load; the Arduino's is
crystal-accurate. A fault destroys a ₹300 board instead of a ₹6,000 one. And 5 V
relay modules are driven natively rather than at the edge of 3.3 V logic.

**Q: Why does `RPi.GPIO` not work on a Pi 5?**
GPIO moved from the main SoC into the RP1 south bridge on PCIe. `RPi.GPIO` pokes
the old memory-mapped registers, which no longer control the pins. The supported
interface is the kernel character device via `lgpio`.

**Q: What is `ACTIVE_LOW` and why does it matter?**
Most relay modules energise when the input is pulled low. If the firmware
assumes active-high, the relay is energised whenever the system is idle and
releases to eject — so power-on fires and holds the solenoid. Verify with the
load disconnected before wiring anything.

**Q: How do you know when to fire, if the part has moved?**
$t = d/v$, the ejector distance divided by belt speed. The part must still be
within reach when the pulse starts, giving a window of
$(w_{part} + w_{ejector})/v$ — about 160 ms for a 50 mm part at 0.5 m/s, so the
timing budget is roughly ±80 ms.

**Q: What happens if the Arduino is unplugged?**
`build_actuator` logs the failure and returns a mock. Detection, recording and
the dashboard continue; ejection does not, and the log says so. Stopping
inspection would lose data as well as ejection.

---

## 10. Summary

- The Pi's GPIO is **3.3 V, ~16 mA, unprotected**. A relay coil needs 5× that.
  Always drive relays through a module with a transistor and opto-isolator.
- On a Pi 5, GPIO lives in the **RP1 south bridge**. `RPi.GPIO` does not work;
  use `lgpio`. `gpiochip4` is a symlink to `gpiochip0` on this image — verify
  with `gpioinfo`, do not assume.
- The Arduino buys **deterministic timing, isolation, and a 5 V domain**.
  Measured 250.1 ms with no jitter on an idle Pi; re-measure under load.
- **A flyback diode across the solenoid is mandatory.** $V = -L\,di/dt$ produces
  hundreds of volts. The relay module's diode protects only its own coil.
- **Set the pin level before enabling the output**, in both Python and C, or the
  solenoid fires on every boot.
- **Verify `ACTIVE_LOW` with the load disconnected.** Backwards means energised
  at idle.
- Solenoid gets **its own supply**, sharing only ground.
- A missing actuator **degrades** the system to detection-only; it never stops
  the line.
- Ejection is delayed by $d/v$, measured on the real conveyor.

Next: [18 — System architecture](18_system_architecture.md).
