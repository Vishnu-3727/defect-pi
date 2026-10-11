# Section 10 — Communication

## 10.0 Where each protocol lives in your system

```
MPU-6050 ──I2C (400 kHz, ~10 cm)──▶ ESP32 ──Wi-Fi 802.11n──▶ Pi ──HTTP/WebSocket──▶ browser
                                       │                      │
                                  (SPI/UART unused,           └── SQLite (local file, no protocol)
                                   explained below)
```

Three questions decide every protocol choice: **how far**, **how fast**, and **how many devices**.

---

## 10.1 I2C — Inter-Integrated Circuit

**Plain language.** A two-wire bus where one boss (controller) talks to several workers (targets), each with a numeric address. Like a classroom: the teacher says "Rahul, answer", and only Rahul speaks.

**Technical.**

| Property | Value |
|---|---|
| Wires | 2 — `SDA` (data), `SCL` (clock), plus common ground |
| Topology | Multi-drop bus, up to 127 addresses (7-bit) |
| Speeds | 100 kHz standard, **400 kHz fast**, 1 MHz fast+, 3.4 MHz high-speed |
| Direction | Half-duplex, bidirectional |
| Electrical | **Open-drain** — devices pull low, resistors pull high |
| Pull-ups | 4.7 kΩ typical (GY-521 boards include them) |
| Distance | ~30 cm practical, limited by bus capacitance (400 pF max) |
| Clocking | Synchronous — the controller supplies SCL |

**Transaction anatomy:**

```
START │ ADDR(7) │ R/W │ ACK │ DATA(8) │ ACK │ … │ STOP
```

To read the MPU-6050's accelerometer you write the starting register address (0x3B), then read 6 bytes (or 14 to include temperature and gyro) in one burst:

```cpp
Wire.beginTransmission(0x68);
Wire.write(0x3B);                 // ACCEL_XOUT_H
Wire.endTransmission(false);      // repeated START, keeps the bus
Wire.requestFrom(0x68, 14, true); // ax,ay,az, temp, gx,gy,gz  — all 16-bit big-endian
int16_t ax = (Wire.read() << 8) | Wire.read();
```

**Why burst-read all 14 bytes at once?** Six separate transactions cost six address phases and, worse, sample the registers at different instants. One burst is atomic with respect to the sensor's internal sample update — the readings are time-coherent. At 400 kHz, 14 bytes ≈ 350 µs, comfortably inside a 1 ms sampling period.

**Why I2C here.** Two wires, the MPU-6050 speaks it natively, and the distance is ~10 cm. Nothing else is needed.

**Limitations.** Short range; slow relative to SPI; open-drain means it is susceptible to noise (motor EMI!); a stuck target can hang the bus (recover by clocking SCL nine times manually, or by re-initialising).

---

## 10.2 SPI — Serial Peripheral Interface

| Property | Value |
|---|---|
| Wires | 4 — MOSI, MISO, SCK, CS (plus one CS per device) |
| Speed | 1–50+ MHz — **10–100× faster than I2C** |
| Duplex | Full-duplex (send and receive simultaneously) |
| Addressing | None — a dedicated chip-select line per device |
| Distance | ~10–20 cm |

**Why you are not using it.** The GY-521 breakout exposes only I2C. (The MPU-6000 variant supports SPI; the MPU-6050 does not.) I2C's 400 kHz is ample for 1 kHz × 14 bytes.

**When you *would* need it.** If you upgrade to a high-rate accelerometer — an ADXL345 at 3200 Hz, or an ADXL355 — SPI becomes necessary because I2C cannot keep up. Mention this in future scope; it shows you understand the coupling between sensor bandwidth and bus choice.

---

## 10.3 UART — Universal Asynchronous Receiver/Transmitter

| Property | Value |
|---|---|
| Wires | 2 — TX, RX (crossed), plus ground |
| Speed | Common 9600 – 921600 baud |
| Clock | **Asynchronous** — both sides must agree on baud rate beforehand |
| Devices | Point-to-point only (two devices) |
| Framing | start bit, 8 data bits, optional parity, stop bit |

**Where you actually use it.** Flashing the ESP32 and reading `Serial.print()` debug output over USB. The USB port on the DevKit is a CP2102/CH340 USB-to-UART bridge — so you use UART every day without thinking about it.

**Where you could use it.** As a wired fallback link between ESP32 and Pi (ESP32 TX → Pi GPIO15 RX, common ground, **level-shift or verify 3.3 V compatibility** — both are 3.3 V, so direct connection is fine). Worth implementing as a backup path: if the Wi-Fi fails during your demo, you flip a flag and run over the wire. Cheap insurance.

**Why not as the primary link.** It requires a physical cable, defeating the point of a wireless sensor node, and it is point-to-point so you could never add a second sensor node.

---

## 10.4 Wi-Fi (IEEE 802.11)

**Plain language.** Radio Ethernet. Devices join an access point and get an IP address; from then on they are ordinary network hosts.

**Technical.**

| Property | ESP32 |
|---|---|
| Standards | 802.11 b/g/n |
| Band | **2.4 GHz only** — not 5 GHz |
| Throughput | ~20 Mbps realistic (150 Mbps nominal) |
| Range | 30–50 m indoors |
| Latency | 2–20 ms typical, occasional 100 ms+ spikes |
| Power | 80–260 mA active; TX bursts to ~500 mA |
| Security | WPA2-PSK, WPA2-Enterprise |

**Why Wi-Fi for this project — the comparison table:**

| Option | Range | Rate | Power | Infra | Verdict |
|---|---|---|---|---|---|
| **Wi-Fi** | 30–50 m | 20 Mbps | High | Router/hotspot needed | ✔ Built into the ESP32; IP networking means HTTP/MQTT just work |
| BLE | 10 m | ~1 Mbps | Very low | None (direct) | Fine for battery nodes; more work to bridge to IP |
| LoRa | 2–10 km | 0.3–50 kbps | Very low | Gateway + extra module | Great for remote assets, far too slow for 1 Hz feature vectors? (actually adequate) but needs extra hardware |
| Zigbee | 100 m mesh | 250 kbps | Low | Coordinator | Common in industrial sensing; extra module |
| Ethernet | 100 m | 1 Gbps | – | Cable | Most reliable; defeats "wireless node" |
| 4G/5G | km | high | High | SIM + cost | Overkill |

Wi-Fi wins on: already on the chip (₹0 extra), gives you IP so every tool works, fast enough for future raw-data streaming, and easy to demo. Its cost is power — irrelevant since your node is mains-powered.

**Say this in viva:** "For a mains-powered node inside a building with an existing network, Wi-Fi is the right choice. For a battery-powered node in a remote pump station, I would use LoRaWAN or NB-IoT, accept a 1-minute reporting interval, and do more inference at the node."

**Practical warnings:**

- 2.4 GHz only. If your phone's hotspot defaults to 5 GHz, the ESP32 will silently never connect. This wastes an evening at least once per project.
- College Wi-Fi with a captive portal will not work. **Bring a phone hotspot or a small travel router to the demo**, or better, run the Pi as its own access point (`hostapd` + `dnsmasq`) so the system is entirely self-contained. That last option is the professional answer and makes your offline demo bulletproof.
- Give the Pi a static IP or use mDNS (`raspberrypi.local`) so the ESP32 does not need reflashing when DHCP changes.

---

## 10.5 The application-layer choices: HTTP vs MQTT vs UDP

This is a real design decision with a defensible answer either way.

### HTTP (over TCP)

Request/response. The ESP32 POSTs JSON; the Pi replies.

```cpp
HTTPClient http;
http.begin("http://192.168.1.50:8000/ingest");
http.addHeader("Content-Type", "application/json");
int code = http.POST(payload);          // returns 200 on success
http.end();
```

- ✔ Dead simple, debuggable with `curl` and a browser, works with FastAPI out of the box, delivery confirmed by the status code.
- ✖ Connection setup overhead per request (mitigate with `http.setReuse(true)`), request/response only — the server cannot push.

### MQTT (over TCP)

Publish/subscribe through a broker (`mosquitto` on the Pi).

```
ESP32 ──publish──▶ [broker: topic "plant/motor1/features"] ──subscribe──▶ Pi service
                                                            ──subscribe──▶ dashboard
                                                            ──subscribe──▶ logger
```

- ✔ Decoupled: adding a second sensor node or a second consumer requires no code changes. QoS levels 0/1/2. Retained messages and Last Will and Testament (the broker announces when a node dies — genuinely useful for staleness detection). **This is the actual IIoT standard.**
- ✖ One more moving part (the broker) to install, configure and explain.

### UDP

Fire-and-forget datagrams, no connection, no delivery guarantee.

- ✔ Lowest latency and overhead; a lost packet just means one missing sample, which for a 30-sample window is harmless.
- ✖ No delivery guarantee, no ordering, you must implement your own sequencing.

### Recommendation

**Start with HTTP POST.** It is the fewest moving parts, it debugs from a browser, and at 1 packet/second the overhead is irrelevant. Get the whole pipeline working with it.

**Then, if you have time, add MQTT** and present the comparison as a design study — "we implemented both and measured latency and CPU; MQTT scales to N nodes, HTTP was simpler for one". That comparison is worth more marks than either implementation alone, and it directly answers "how would this scale to a factory?"

Do not use raw UDP unless you later stream raw waveforms.

---

## 10.6 JSON — the payload format

**What.** Human-readable, text-based, self-describing data format.

```json
{
  "node": "motor1",
  "seq": 10432,
  "ts_ms": 8123456,
  "fs_hz": 1000,
  "n": 1000,
  "feat": {
    "ax_rms": 0.0412, "ax_kurt": 3.21, "ax_crest": 3.9,
    "ay_rms": 0.0387, "ay_kurt": 3.05, "ay_crest": 3.7,
    "az_rms": 0.1120, "az_kurt": 2.98, "az_crest": 3.4,
    "temp_c": 38.2
  },
  "clip": 0,
  "jitter_us": 42
}
```

- ✔ Readable during debugging, parsed by every language, schema-flexible, validated free by Pydantic on the FastAPI side.
- ✖ Verbose (~300 bytes here vs ~40 bytes packed binary), and float parsing costs CPU.

**At 1 Hz, JSON's overhead is irrelevant — use it.** If you ever stream raw waveforms at 1 kHz, switch to a packed binary struct or MessagePack/CBOR. Knowing *when* JSON stops being appropriate is the interview-worthy part.

**Include these fields, always:** a sequence number (detect loss), a timestamp (detect staleness), the sampling rate and sample count (so the Pi can validate assumptions), and a clipping/jitter diagnostic. Eight extra bytes; hours of debugging saved.

---

## 10.7 The full ESP32 → Pi flow

```
┌─ ESP32, core 1 (sampling task) ──────────────────────────────┐
│  hardware timer ISR @ 1 kHz                                   │
│    → burst-read 14 bytes over I2C                             │
│    → convert to g, push into ring buffer (1000 samples × 3)   │
│  every 1000 samples (1 s):                                    │
│    → compute RMS, std, kurtosis, skew, crest, p2p per axis    │
│    → hand the feature struct to the network task via a queue  │
└───────────────────────────────────────────────────────────────┘
                    │  FreeRTOS queue
┌─ ESP32, core 0 (network task) ───────────────────────────────┐
│  serialise to JSON (ArduinoJson, StaticJsonDocument)          │
│  HTTP POST → http://<pi>:8000/ingest                          │
│  on failure: retry ×3, then buffer up to N packets in RAM     │
└───────────────────────────────────────────────────────────────┘
                    │  Wi-Fi 2.4 GHz
┌─ Raspberry Pi (FastAPI) ─────────────────────────────────────┐
│  validate with Pydantic → reject malformed                    │
│  check seq for gaps → log them                                │
│  append features to a deque(maxlen=30)                        │
│  if full: z-score with saved stats → (1,S,30) → ORT.run()     │
│  smooth (EWMA), evaluate the alarm rule (N consecutive)       │
│  write to SQLite, push to the dashboard over WebSocket        │
│  reply {"rul": …} so the ESP32 can blink an LED locally       │
└───────────────────────────────────────────────────────────────┘
```

**Why the two-task split.** Blocking on an HTTP POST (which can take 200 ms if the network hiccups) inside the sampling loop destroys your sample timing. Separate tasks with a queue between them is the standard embedded pattern, and describing it correctly is a strong embedded-systems answer.

**Reply with the RUL.** It costs nothing, and it lets the ESP32 drive a local LED — green/amber/red — so the machine has a status indicator even if the network or dashboard dies. Small feature, very good demo moment.

---

## 10.8 Reliability details that matter

| Problem | Detection | Handling |
|---|---|---|
| Wi-Fi drops | `WiFi.status() != WL_CONNECTED` | Non-blocking reconnect with backoff; keep sampling and buffering |
| Pi unreachable | POST returns < 0 | Retry ×3, then buffer in RAM (a few hundred packets fit in 520 KB), flush on recovery |
| Packet lost | Pi sees a `seq` gap | Log it; if the gap is ≥ L, mark the window invalid and restart warm-up |
| Duplicate packet | Repeated `seq` | Ignore (idempotent ingest) |
| Clock skew | ESP32 `millis()` vs Pi wall clock | Use ESP32 `millis()` for intervals, Pi time for logging; optionally sync via NTP or via the POST response |
| Stale data | Pi: `now − last_packet > 3 s` | Dashboard shows "SENSOR OFFLINE"; suppress predictions rather than predicting on stale input |
| Malformed JSON | Pydantic validation error | Return HTTP 422, count it, do not crash |

That table is a slide. Robustness engineering is rarely shown in student projects and always noticed.

---

## Summary

- I2C (2 wires, 400 kHz, ~10 cm) connects MPU-6050 to ESP32; burst-read 14 bytes for time-coherent samples.
- SPI is faster but unavailable on the GY-521; UART is used for flashing/debug and is a good wired fallback.
- Wi-Fi 2.4 GHz carries features from ESP32 to Pi — chosen because it is on-chip, gives IP, and the node is mains-powered.
- HTTP POST + JSON is the right starting point; MQTT is the scaling answer and worth implementing as a comparison.
- Always include seq, timestamp, sample rate, and diagnostics in the payload.

## Key takeaways

1. Protocol choice = distance × rate × device count (× power, for battery nodes).
2. Never block the sampling loop on network I/O — use two tasks and a queue.
3. 2.4 GHz only on the ESP32. Plan your demo network in advance; Pi-as-access-point is the safest option.
4. Sequence numbers and timestamps cost 8 bytes and are worth hours.

## Common interview questions

1. Compare I2C, SPI and UART across wires, speed, duplex, addressing and distance.
2. Why does I2C use open-drain outputs and pull-up resistors?
3. What is a repeated START condition and why does the MPU-6050 read need one?
4. TCP vs UDP — when would you choose each?
5. What is MQTT's publish/subscribe model and what are QoS 0/1/2?
6. How do you detect packet loss in a stream you designed yourself?
7. Why is JSON a poor choice for high-rate telemetry, and what would you use instead?

## Common viva questions

1. "Why Wi-Fi and not Bluetooth/LoRa?" → Mains-powered, indoor, needs IP, already on-chip. Give the alternative for a battery-powered remote asset to show you understand the trade-off.
2. "What is the I2C address of your sensor?" → 0x68 (0x69 if AD0 is pulled high). Know it.
3. "What happens if two ESP32 nodes send at once?" → Nothing bad — TCP/IP handles multiplexing; the Pi keys buffers by `node` ID. Then note this is where MQTT would become clearly superior.
4. "How much data does one packet carry?" → ~300 bytes of JSON at 1 Hz. Know the number and the raw-vs-features reduction ratio.
5. "What if the network is congested?" → Buffer on the ESP32, detect staleness on the Pi, degrade gracefully. Never predict on stale data.

## Practical implementation tips

- Test the endpoint with `curl` before writing any firmware:
  ```bash
  curl -X POST http://<pi>:8000/ingest -H "Content-Type: application/json" \
       -d '{"seq":1,"ts_ms":0,"features":[0.1,0.2]}'
  ```
- Use `http.setReuse(true)` on the ESP32 to keep the TCP connection alive between posts — noticeably lower latency and less heap churn.
- Use `StaticJsonDocument` (stack-allocated) rather than `DynamicJsonDocument` on the ESP32 to avoid heap fragmentation over long runs.
- Print the ESP32's IP and the Pi's IP to Serial on boot. You will need them constantly.
- Set a static IP for the Pi on your demo network, and hard-code it in firmware only as a fallback after mDNS lookup fails.

## Suggested improvements for this project

1. **Run the Pi as a Wi-Fi access point** (`hostapd`/`NetworkManager` AP mode). The ESP32 joins the Pi directly, no router, no college network, no captive portal. Your demo becomes a self-contained box you can carry anywhere. Highest-value change in this section.
2. **Add MQTT alongside HTTP** and present the latency/CPU comparison as a design study.
3. **Add the LED status indicator** on the ESP32 driven by the POST response.
4. **Implement store-and-forward buffering** on the ESP32 and demonstrate it by unplugging the Pi mid-demo, then plugging it back in and showing the backlog flush. Very memorable.
5. **Add TLS or at least a shared-secret header** on the ingest endpoint. Right now anyone on the network can inject fake sensor readings into your maintenance system. Mentioning this unprompted — the trust boundary — is a genuinely strong point in a viva, and a `X-Node-Key` header check is five lines.
