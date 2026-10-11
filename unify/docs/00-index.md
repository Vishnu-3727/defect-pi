# UNIFY-RUL — Mentor Curriculum

Universal Remaining Useful Life Prediction — Any Dataset, Any Hardware
Chennai Institute of Technology · ECE · PBL · Vishnu Vardhan K S (24EC0251), VS Kamlesh (24EC0243)
Mentor: Mr. Prem Sangeeth

## How to use these notes

Read in order. Every file follows the same shape:

1. Plain-language explanation (no jargon)
2. Technical explanation (precise, with math / tensor shapes)
3. Why it exists **in this project specifically**
4. Summary + key takeaways
5. Interview questions
6. Viva questions (what your faculty will actually ask)
7. Practical implementation tips
8. Suggested improvements

Do not skip Section 05 (preprocessing). Most RUL projects fail there, not in the model.

## Files

| File | Covers | Original section(s) |
|---|---|---|
| `01-predictive-maintenance-and-rul.md` | Maintenance strategies, RUL vs everything else | 1 |
| `02-edge-ai.md` | Edge vs cloud, why the Pi | 2 |
| `03-system-architecture.md` | Every block, full data flow | 3 |
| `04-datasets.md` | C-MAPSS, FEMTO, XJTU-SY | 4 |
| `05-preprocessing.md` | Normalisation, windowing, RUL capping, the unit problem | 5 |
| `06-model.md` | CNN encoder, embeddings, cross-attention, head, losses, shapes | 6, 7, 16 |
| `07-software-stack.md` | Python → PyTorch → ONNX → Flask/FastAPI → SQLite | 8 |
| `08-hardware.md` | Pi 4, ESP32, MPU-6050, L298N, motor, power | 9 |
| `09-communication.md` | I2C, UART, SPI, Wi-Fi, MQTT/HTTP/UDP, JSON | 10 |
| `10-onnx-and-deployment.md` | ONNX format, dynamic axes, ORT on ARM, quantisation, latency | 11, 12 |
| `11-realtime-pipeline.md` | Motor → sensor → ESP32 → Wi-Fi → Pi → dashboard | 13 |
| `12-implementation-and-roadmap.md` | Folder structure, every file, 12-week plan | 14, 15 |
| `13-defense-and-industry.md` | Viva bank, common mistakes, industry comparison | 17, 18 |

## The one thing to internalise before you start

Your project has **two truths that must not be confused**:

- **Truth A — the datasets.** C-MAPSS, FEMTO and XJTU-SY are run-to-failure datasets. They have ground-truth RUL labels. This is where your model learns and where your accuracy numbers (RMSE, NASA score) come from.
- **Truth B — your motor rig.** Your 12 V DC motor + MPU-6050 has **no failure label**. You will never run it to destruction in a semester, and even if you did, one bearing's death is one sample.

So the rig is a **live inference and latency demonstrator**, not a source of validated RUL accuracy. State this openly in your report. Faculty respect a student who knows the boundary of their own claim; they eat alive a student who claims "my model predicts my motor's RUL to ±5 cycles."

Handled correctly (see `11-realtime-pipeline.md`, "the honesty layer"), this becomes a strength of the presentation, not a hole in it.
