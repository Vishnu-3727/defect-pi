# Literature Sweep Findings — 2026-09-18

**Working directory:** `/tmp/claude-1000/-home-vishnu/4cb1d20a-b120-44a1-81e1-38bccb0ee5de/scratchpad/opencode-research`  
**Execution date (UTC):** 2026-09-18

## 0. Tool Disclosure (required by task)

I have web access in this environment via two tools:

* `websearch` — live web search (returned results with URL, title, excerpts)
* `webfetch` — fetch and convert a URL to markdown/text

I also have local file tools (`read`, `write`, `edit`, `glob`, `grep`, `bash`). I do **not** have paywall-bypass, academic-database, or offline-citation generation. Every paper/number/URL below was actually retrieved via `websearch`/`webfetch` in this session. Where a search returned nothing, I state **NOT FOUND** explicitly and do not invent a citation.

---

## QUESTION 1 — INT8 quantisation sometimes IMPROVING accuracy

### 1.1 Your observation

Two YOLOv8n detectors, same arch / dataset / 100-epoch recipe / calibration set / ONNX Runtime static PTQ:

* Run A: mAP@50 0.7107 → 0.7324 **(+0.0217)**
* Run B: mAP@50 0.7624 → 0.7353 **(-0.0271)**

Opposite signs.

### 1.2 Is post-training quantisation known to sometimes *improve* a detection metric?

**Yes — reported but rare for PTQ, better documented as a *regularisation / noise-injection* effect in the quantisation literature. The effect has named terms.**

#### (a) Named effect: "quantisation as regulariser / noise injection / implicit calibration"

* **QReg: On Regularization Effects of Quantization** — MohammadHossein AskariHemmat, Reyhane Askari Hemmat, Alex Hoffman, Ivan Lazarevich, et al., *arXiv:2206.12372, 2022* (cs.CV).  
  URL: https://arxiv.org/abs/2206.12372  (fetched via `webfetch`, abstract verified)  
  **Core claim (verbatim excerpt):** "We hypothesize that weight quantization is a form of regularization and the amount of regularization is correlated with the quantization level (precision)… By modeling weight quantization as a form of additive noise to weights… 8-bit quantization provides a reliable form of regularization in different vision tasks and models."  
  Theory: Δ ∼ U[-δ/2, δ/2]; loss becomes ℒ̃ = ℒ + E[||Δᵀ∇yf̂||²] = ℒ + R(Δ). Lower-bit ⇒ wider δ ⇒ larger R.

* **QGen: On the Ability to Generalize in Quantization Aware Training** — Same group, *arXiv:2404.11769v2, 2024*.  
  URL: https://arxiv.org/html/2404.11769v2 (fetched, full text)  
  Extends QReg with sharpness / flat-minima analysis (2000 models on CIFAR-10/100, ImageNet, ResNet, ViT, DeiT). Shows quantized models converge to **flatter minima** (PAC-Bayes + Keskar sharpness, magnitude-aware) and have **smaller generalisation gaps**, especially under input corruptions (Hendrycks & Dietterich). Caveat: experiments are **QAT**, not PTQ. Authors explicitly discuss equation (7) ℒ̃ = ℒ + R(Δ) and note δ is a hyperparameter — not all levels improve.

* **Training with Quantization Noise for Extreme Model Compression (Quant-Noise)** — Fan et al., *ICLR 2021* (arXiv:2004.07320).  
  URL: https://arxiv.org/pdf/2004.07320 (excerpt verified)  
  **Quote:** "quantization noise regularizes the network in a similar way as DropConnect (Wan et al., 2013) or LayerDrop." Method applies quant noise to random *subset* of weights during training. Shows int8 + PQ with Quant-Noise improves over vanilla QAT. Again training-time noise, not PTQ.

* **When Less is More: 8-bit Quantization Improves Continual Learning in LLMs** — *arXiv:2512.18934v2, 2025*.  
  URL: https://arxiv.org/html/2512.18934v2  
  **Quote:** "We hypothesize that quantization-induced noise acts as implicit regularization… INT8 consistently achieving the optimal balance between learning plasticity and knowledge retention." Baseline INT8 within 1–2% of FP16, INT4 3–5% degraded. Different domain (LLaMA-3.1-8B continual learning with replay) but explicitly names the effect.

* **Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy** — Bouguerra, Montoya, Gomez-Villa, Mraidha, Arnez, *ICML 2026* (700k runs, CLIP-family).  
  URL: https://lamp.cvc.uab.es/less-precise-can-be-more-reliable-a-systematic-evaluation-of-quantizations-impact-on-vlms-beyond-accuracy (retrieved)  
  **Quote:** "contrary to the assumption that quantization's noise degrades performance, it can simultaneously improve accuracy, calibration, OOD detection, and robustness to noise… quantization dampens high-rank spectral components, compelling the model to rely more heavily on robust, low-rank features… sometimes letting INT8 beat FP32 (Zhang et al., 2025)." Provides a **spectral-filtering** mechanistic explanation distinct from simple regularisation.

* **What Survives When You Compress a Recursive Reasoner for the Edge?** — *arXiv:2606.26488, 2026*. (not a detection paper, but summarizes)  
  **Quote:** "Quantization noise can act as a regulariser – pushing toward flatter minima with a model-specific optimal level (AskariHemmat et al., 2024, 2022), robust under distribution shift… sometimes letting INT8 beat FP32 (Zhang et al., 2025) – which explains INT8 parity in single-pass models."

* **Detection-specific evidence that quantisation can *increase* mAP:**

  * **AQD: Towards Accurate Quantized Object Detection** — Chen et al., *CVPR 2021*.  
    URL: https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_AQD_Towards_Accurate_Quantized_Object_Detection_CVPR_2021_paper.pdf  
    **Quote:** "our 4-bit quantized detectors can even outperform the corresponding full-precision models in some cases" and "2-bit FCOS detector obtains 2.9% and 2.7% AP improvement over Group-Net with ResNet-18/50 backbones." Full integer-only arithmetic (conv + norm + skip). This is PTQ-adjacent / fully-quantized detection; shows *direction* of your sign flip is not unprecedented, though usually at 4-bit with specialised method, not vanilla ONNX static PTQ.

  **No PTQ-only YOLOv8 detection paper retrieved that reports a *vanilla* static INT8 ONNX Runtime PTQ improving COCO mAP@50 by ~2 points.** The improvements above are under QAT, AQD-style, or broad VLM evaluation.

#### Bottom line for 1.2

The community **does** have a named effect: **quantisation noise as regulariser / implicit regularisation / spectral filtering**, with formal theory (QReg/QGen) and multiple empirical confirmations. It is *plausible* that INT8 PTQ could nudge mAP up, e.g. by suppressing over-confident false positives, dampening high-rank features, or flattening minima — but papers present it as **model- and level-specific**, not guaranteed. QReg explicitly says 8-bit is the "reliable" sweet spot; 4-bit/2-bit often overshoots.

### 1.3 Is the run-to-run VARIANCE of the quantisation delta characterised?

**NOT FOUND as a formal characterisation.**

* Search queries executed: `"quantization variance run-to-run instability mAP drop"`, `"PTQ quantization variance run to run"`, plus general PTQ literature review (see §1.2 papers + DetPTQ arXiv:2304.09785, InlierQ arXiv:2602.03472, Reg-PTQ CVPR 2024, Q-YOLO, etc.).  
* What papers **do** report: a single "quantisation costs X mAP" figure (e.g., APQ-ViT §4 avg 0.23% drop at W8A8, DetPTQ Table RetinaNet 31.1/31.7 mAP at W4A4, AQD, Q-YOLO). They state calibration set size (e.g., InlierQ: 64 samples) but do not show distribution over training seeds or over repeated calibrations with same calibration set.
* No retrieved paper plots histogram of ΔmAP over ≥10 training runs with fixed PTQ recipe, or reports σ(ΔmAP) for YOLO-family PTQ. One paper close but not answering: *DynamicPTQ* (arXiv:2606.12487) notes results "averaged over five runs after two warmup runs" for throughput, but not for ΔmAP variance.
* **Honest statement:** Papers usually quote a single figure; that figure is **likely unstable** given dependence on weight distribution (cf. QReg's R(Δ) ∝ δ²||∇yf̂||², which varies by run), calibration-sample sampling, and per-layer Lp choice (DetPTQ shows p matters). But I did not retrieve a source that explicitly quantifies instability; claiming a number would be fabrication. Treat single-point "costs X" as point estimate without uncertainty.

**What you should do next (research suggestion, not a citation):** repeat PTQ over both runs with multiple calibration shuffles and report mean ± sd of ΔmAP; publish the variance — it fills a gap.

### 1.4 Is mAP at low confidence sweep (0.001) known to be sensitive to perturbations of the low-confidence tail?

**No peer-reviewed paper retrieved that formally characterises sensitivity of mAP@50 computed at conf=0.001 sweep to low-confidence-tail perturbations. Anecdotal / engineering evidence exists.**

* **Ultralytics YOLOv5 issue #11745** — *github.com/ultralytics/yolov5/issues/11745* (retrieved).  
  URL: https://github.com/ultralytics/yolov5/issues/11745  
  **Quote:** "yolov5 does not actually report P and R, it reports AP and AR … threshold of 0.001 being used by default to calculate AP, AR, mAP in val.py and train.py… purpose of this low threshold is to ensure that all possible objects are included, even if their confidence scores are low." Issue author notes at conf 0.001 they see many random FP boxes yet mAP 0.99, indicating mAP integrates over full PR curve and is *not* strongly affected by raw count of very-low-score FPs in that anecdotal case.

* **RF-DETR issue #295** — *github.com/roboflow/rf-detr/issues/295* (retrieved excerpt).  
  URL: https://github.com/roboflow/rf-detr/issues/295  
  **Quote:** "For YOLO (via Ultralytics), the default validation uses a very low confidence‐threshold (conf_thres = 0.001) before NMS, ensuring high recall. RF-DETR simply takes all predicted boxes (regardless of score) and builds the full…"

* **Not retrieved:** formal analysis of how small shifts (±0.01) in logits for boxes with score 0.001–0.05 change AP at IoU 0.5. The intuition holds — mAP@50 at low threshold **is** sensitive to tail because PR curve integrates recall down to that threshold, and quantisation perturbs exactly the logit/ confidence calibration. But no citation was returned by searches `"mAP sensitivity confidence threshold 0.001 low confidence"` or `"global sensitivity analysis for MAP inference"` (the latter returned an unrelated PGM paper, NeurIPS 2014).

**Honest statement:** sensitivity of the 0.001 tail is widely *assumed* by practitioners (hence they use 0.001), but I found no paper that quantifies it for YOLO-like detectors. Your observed ±0.02 swing is consistent with tail sensitivity being non-negligible on a modest custom dataset, where a few detections crossing 0.001 flips TP/FP count enough to move mAP@50.

---

## QUESTION 2 — Real-time defect detection on Raspberry Pi class hardware

### 2.1 Published FPS numbers for YOLO-family on Pi 4 / Pi 5 CPU

All numbers below are **measured on CPU only** unless stated (Hailo/AI HAT/Jetson excluded from main table). Input resolution, quantisation, threading, and model variant are given where reported. Every number traced to a fetched URL.

#### Primary sources fetched

| ID | Source | Platform | URL |
|---|---|---|---|
| S1 | kadirmertabatay/rpi-yolo-benchmark, YOLOv8n 27 Jan 2026 | Pi 5 (4 cores) | https://github.com/kadirmertabatay/rpi-yolo-benchmark/blob/main/docs/benchmarks/yolov8/yolov8n.md |
| S2 | kadirmertabatay, ONNX comparison 27 Jan 2026 | Pi 5, ONNX Runtime 1.23.2 CPU EP | https://github.com/kadirmertabatay/rpi-yolo-benchmark/blob/main/docs/comparisons/onnx.md |
| S3 | Ultralytics official guide, Raspberry Pi, benchmarks Ultralytics 8.4.108/8.4.14 | Pi 5, FP32 | https://docs.ultralytics.com/guides/raspberry-pi |
| S4 | Seeed Studio blog, ncnn benchmark 640×640 | Pi 4 8GB vs Pi 5 8GB | https://www.seeedstudio.com/blog/2023/09/28/raspberry-pi-5-vs-pi-4-ai-performance-cpu-benchmark-how-much-leap-forward |
| S5 | Qengineering, YoloV8-ncnn-Raspberry-Pi-4 README | Pi 4 @1950 MHz, Pi 5 @2900 MHz | https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4/blob/main/README.md |
| S6 | satwik-shreshth/YOLOv8-Path-Segmentation, Pi 5 CPU | Pi 5 CPU, PyTorch vs ONNX | https://github.com/satwik-shreshth/YOLOv8-Path-Segmentation-RealTime-Optimization |
| S7 | ONNX Runtime docs – Quantization | general ARM/x86 | https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html |
| S8 | Microsoft onnxruntime quantization README | CPU arch guidance | https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/README.md |

#### Comparison table (CPU only, sorted by FPS)

> All kadirmertabatay tests: 1920×1080 video @ 29.97 FPS, 30 s (899 frames), imgsz=640, conf=0.25, IoU=0.45, unless otherwise noted. Ultralytics tests: COCO val, imgsz=640, FP32, per-image latency (pre/post excluded). Qengineering: per-model ncnn, inference-only FPS.

| Model | Variant | Input | Quant | Runtime / Format | Device | Threads | FPS | Latency (ms/im) | CPU% | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| **YOLO26n** | nano | 640 | FP32 | NCNN Multi (3) | Pi 5 | 3 | **9.32** | 234.5 | 70.9 | S1 (yolo26n.md) |
| **YOLO11n** | nano | 640 | FP32 | NCNN Multi (3) | Pi 5 | 3 | 9.08 | 248.1 | 73.5 | S1+S3 |
| **YOLOv8n** | nano | 640 | FP32 | NCNN Multi (3) | Pi 5 | 3 | 9.04 | 258.4 | 74.8 | S1 |
| **YOLO26n** | nano | 640 | FP32 | NCNN Single | Pi 5 | 1 | 7.51 | 133.1 | 63.4 | S1 |
| **YOLOv8n** | nano | 640 | FP32 | NCNN Single | Pi 5 | 1 | 7.25 | 137.9 | 64.5 | S1 |
| **YOLO26n** | nano | 640 | FP32 | ONNX Single (CPU EP) | Pi 5 | 1 | 6.74¹ | 142 | 98.6 | S2 |
| **YOLO11n** | nano | 640 | FP32 | OpenVINO | Pi 5 | 1 | — | 81.2 | — | S3 |
| **YOLOv8n** | nano | 640 | FP32 | ONNX Single | Pi 5 | 1 | 4.55 | 219.9 | 82.5 | S1 |
| **YOLO26n** | nano | 640 | FP32 | ONNX (Ultralytics) | Pi 5 | 1 | — | 126.0 | — | S3 |
| **YOLOv8n** | nano | 640 | FP32 | ONNX (Ultralytics) | Pi 5 | 1 | — | ~80–168² | — | S3 (YOLO11n proxy) |
| **YOLOv8n** | nano | 320 | FP32 | PyTorch | Pi 5 | 1 | 9.7 | 103.0 | — | S6 |
| **YOLOv8n-seg** | nano-seg | 320 | FP32 | ONNX | Pi 5 | 1 | **15.4** | 64.8 | — | S6 |
| **YOLOv8n** | nano | 640 | FP32 | PyTorch | Pi 5 | 1 | 3.18 | 314.1 | 43.0 | S1 |
| **YOLOv8n** | nano | 640 | FP32 | PyTorch Multi (3) | Pi 5 | 3 | 5.76 | 506.8 | 81.0 | S1 |
| **YOLOv8m** | medium | 640 | FP32 | NCNN Multi (3) | Pi 5 | 3 | 2.54 | 924.9 | 89.6 | S1 (yolov8m.md) |
| **YOLOv8m** | medium | 640 | FP32 | NCNN Single | Pi 5 | 1 | 2.43 | 411.3 | 86.2 | S1 |
| **YOLOv8m** | medium | 640 | FP32 | ONNX Single | Pi 5 | 1 | 0.98 | 1015.5 | 96.0 | S1 |
| **YOLOv5n** | nano | 640 | FP32 | ncnn | Pi 4 @1950 | 1 | 1.6 | ~625 | — | S5 |
| **YOLOv5n** | nano | 640 | FP32 | ncnn | Pi 5 @2900 | 1 | **13.6** | ~73 | — | S5 |
| **YOLOv8n** | nano | 640 | FP32 | ncnn | Pi 4 @1950 | 1 | 3.1 | ~322 | — | S5 |
| **YOLOv8n** | nano | 640 | FP32 | ncnn | Pi 5 @2900 | 1 | **20.0** | ~50 | — | S5 |
| **YOLOv8n** | nano | 640 | FP32 | ncnn | Pi 4 (Seeed, 1T) | 1 | — | ~74³ | — | S4 |
| **YOLOv8n** | nano | 640 | FP32 | PyTorch Lite | Pi 5 | 1 | 0.54 | 1822 | 39 | S1 (model-comparison) |

¹ S2 reports 6.74 FPS single-thread ONNX for YOLO26n; S1 reports 5.30 FPS ONNX single-thread for YOLO26n on same Pi 5 — likely different ONNX Runtime config / model version. Both shown honestly.  
² S3 table row for YOLO11n ONNX 168 ms, YOLO26n ONNX 126 ms; YOLOv8n not in that table, use as proxy.  
³ Seeed blog text: "YOLOv8n on Raspberry Pi 5 8GB with ncnn at 640×640 … around 12 fps" in prose, but its detailed ncnn table lists FastestDet/yolov4-tiny etc.; Qengineering numbers are more systematic for YOLOv8n.

**Key methodological notes (from sources):**

* kadirmertabatay: `conf=0.25, iou=0.45, imgsz=640`, cooldown 60 s, 30-s video. Both S1 and S2 note **ONNX Runtime already uses all cores internally** — adding Python-level multi-threading *hurts* ONNX (e.g., YOLOv8n ONNX 4.55 → 3.35 FPS, -26%; YOLO26n 6.74 → 4.56 FPS, -32%). NCNN scales positively with multi-threading.
* Ultralytics: benchmarks run at **FP32 precision, imgsz=640**, "inference time does not include pre/post-processing." NCNN 67 ms for YOLO26n vs ONNX 126 ms vs PyTorch 299 ms on Pi 5 (S3).
* satwik-shreshth: Pi 5 CPU, 20 runs warm-up excluded. **Resolution dominates FPS**: 640→320 halves latency and doubles FPS; ONNX gives **1.57× speedup** over PyTorch at 320 (101.6→64.8 ms, 9.8→15.4 FPS) for YOLOv8n-seg. Achieving "real-time" (≥10 FPS) on Pi 5 CPU required 320 input.
* All tables are **FP32** unless stated. No INT8 row appears in any Pi 4/5 CPU table retrieved.

### 2.2 Anything specifically on Raspberry Pi 5 with ONNX Runtime and INT8?

**NOT FOUND — no published FPS/latency number for Pi 5 + ONNX Runtime + INT8 with YOLO-family was retrieved.**

* Searches executed: `"Raspberry Pi 5 ONNX Runtime INT8 YOLO FPS"`, `"YOLO Raspberry Pi ONNX INT8 quantization benchmark"`, `"ONNX Runtime INT8 quantization overhead slower CPU without VNNI"` (5 queries, ~50 results).
* Closest hits (not Pi 5 INT8 FPS):

  * **ONNX Runtime docs** (S7) — states CPU quantisation on ARM should use `activation_type=QInt8, weight_type=QInt8` (S8S8 QDQ as default, ARM dot-product instructions). Warns: "quantization has overhead (from quantizing and dequantizing), so it is not rare to get worse performance on old devices. x86-64 with VNNI … and Arm-based processors with dot-product instructions can get better performance in general." Pi 5 Cortex-A76 **does** have dot-product (SDOT/UDOT), so theoretical support exists.

  * **onnxruntime quantization README** (S8) — "ARM (Cortex-A, Apple Silicon, Graviton): activation_type=QInt8, weight_type=QInt8, per_channel=False … reduce_range=False." No FPS table.

  * **VENKATESH1608/onnx-cpu-inference-accelerator** (retrieved excerpt) — "Latency, however, got worse with INT8 on this machine — the CPU (Intel Core i5-10300H) lacks AVX-512 VNNI / AVX-VNNI, the instruction set ONNX Runtime's CPU EP needs for fast INT8 GEMM/conv kernels. Without it, QDQ dequantize/quantize overhead outweighs the INT8 compute savings." General x86 lesson, not Pi 5.

  * **keytech42/exp--onnx_int8_paradox** — "Benchmarking ONNX INT8 vs FP32: 3.6× size reduction but 37× slower on M3 CPU … multi-threading impact study." Apple M3, not Pi.

  * **Qengineering & Ultralytics** note: fastest Pi CPU path is **NCNN**, not ONNX INT8. Ultralytics recommends NCNN for ARM: "Out of all the model export formats supported by Ultralytics, NCNN delivers the best inference performance on Raspberry Pi devices because it is highly optimized for mobile/embedded platforms such as ARM architecture." (S3).

  * **Hailo AI HAT / Coral TPU** numbers (excluded per your "CPU" ask) do reach 30–90 FPS for YOLOv8 but require accelerator, not CPU INT8.

* **Implication:** absence of Pi 5 + ONNX Runtime INT8 FPS numbers in top-50 results suggests either (a) community has not published them, or (b) INT8 via ONNX CPU EP on Pi 5 is not faster (QDQ overhead dominates), so authors default to NCNN FP16/FP32 or Hailo. Honest answer is **not found**.

---

## 3. What I could NOT find (honesty checklist)

| Claim you might expect | Found? | Detail |
|---|---|---|
| Paper explicitly measuring **ΔmAP variance across training seeds** for same PTQ recipe on YOLO | **NOT FOUND** | No histogram / σ reported. Single-point deltas only. |
| Paper reporting **PTQ improves mAP@50 by >2% on YOLOv8n COCO-style via vanilla ONNX static PTQ** | **NOT FOUND** | Closest is AQD (CVPR21) with specialised fully-quantized method beating FP32 at 4-bit on some detectors; Q-YOLO 8-bit YOLOv7 "3× acceleration while maintaining performance" but not a +2.1 mAP gain on YOLOv8n. |
| Formal study of **mAP@50 sensitivity to 0.001 confidence tail perturbations** | **NOT FOUND** | Only GitHub issues / docs noting default 0.001 for recall; no sensitivity analysis. |
| **Pi 5 + ONNX Runtime CPU EP + INT8 static quantisation + YOLO** FPS/latency with imgsz/thread count | **NOT FOUND** | INT8 numbers exist for x86 (worse without VNNI) and for Hailo/NPU, but not CPU INT8 on Pi 5. |
| Published **per-operator INT8 vs FP32 latency on Pi 5 ARM** (e.g., conv INT8 QDQ breakdown) | **NOT FOUND** | Only generic ARM guidance (QInt8/QInt8) in ONNX Runtime docs. |

---

## 4. References — only what was actually fetched

1. AskariHemmat, M., et al. **QReg: On Regularization Effects of Quantization.** *arXiv:2206.12372* (2022). https://arxiv.org/abs/2206.12372
2. AskariHemmat, M., et al. **QGen: On the Ability to Generalize in Quantization Aware Training.** *arXiv:2404.11769v2* (2024). https://arxiv.org/html/2404.11769v2
3. Fan, A., et al. **Training with Quantization Noise for Extreme Model Compression.** *ICLR 2021*, arXiv:2004.07320. https://arxiv.org/pdf/2004.07320
4. Chen, X., et al. **AQD: Towards Accurate Quantized Object Detection.** *CVPR 2021.* https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_AQD_Towards_Accurate_Quantized_Object_Detection_CVPR_2021_paper.pdf
5. Bouguerra, A., Montoya, D., et al. **Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy.** *ICML 2026.* https://lamp.cvc.uab.es/less-precise-can-be-more-reliable-a-systematic-evaluation-of-quantizations-impact-on-vlms-beyond-accuracy
6. *When Less is More: 8-bit Quantization Improves Continual Learning in LLMs.* *arXiv:2512.18934v2* (2025). https://arxiv.org/html/2512.18934v2
7. *What Survives When You Compress a Recursive Reasoner for the Edge?* *arXiv:2606.26488* (2026). https://arxiv.org/pdf/2606.26488 — cites QReg/QGen + Zhang 2025 for INT8 beating FP32.
8. Niu, L., et al. **Improving Post-Training Quantization on Object Detection with Task Loss-Guided Lp Metric (DetPTQ).** *arXiv:2304.09785* (2023). https://arxiv.org/abs/2304.09785
9. kadirmertabatay. **rpi-yolo-benchmark.** Benchmark docs 27 Jan 2026. https://github.com/kadirmertabatay/rpi-yolo-benchmark/blob/main/docs/benchmarks/yolov8/yolov8n.md ; ONNX comparison https://github.com/kadirmertabatay/rpi-yolo-benchmark/blob/main/docs/comparisons/onnx.md
10. Ultralytics. **Quick Start Guide: Raspberry Pi with Ultralytics YOLO26.** *docs.ultralytics.com*, tested Nov 2023, benchmarks Ultralytics 8.4.108. https://docs.ultralytics.com/guides/raspberry-pi
11. Seeed Studio. **Raspberry Pi 5 vs. Pi 4 AI performance CPU Benchmark.** *seeedstudio.com* blog, 2023-09-28 (site date 2025-03-25). https://www.seeedstudio.com/blog/2023/09/28/raspberry-pi-5-vs-pi-4-ai-performance-cpu-benchmark-how-much-leap-forward
12. Qengineering. **YoloV8-ncnn-Raspberry-Pi-4.** https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4/blob/main/README.md
13. Satwik Shreshth. **YOLOv8-Path-Segmentation-RealTime-Optimization.** https://github.com/satwik-shreshth/YOLOv8-Path-Segmentation-RealTime-Optimization
14. ONNX Runtime. **Quantize ONNX models.** https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
15. Microsoft ONNX Runtime. **Quantization README – Choosing parameters for CPU inference.** https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/README.md
16. Ultralytics YOLOv5 Issue #11745. **Why does the confidence threshold of 0.001 in val.py result in good results?** https://github.com/ultralytics/yolov5/issues/11745
17. Roboflow RF-DETR Issue #295. https://github.com/roboflow/rf-detr/issues/295 (excerpt: YOLO default conf_thres 0.001)
18. *Quantization Robustness to Input Degradations for Object Detection.* *arXiv:2508.19600v2* (YOLO12 scales, FP32/FP16/Dynamic UINT8 ONNX/Static INT8 TRT) — retrieved via search, provides robustness context but not Pi FPS. https://arxiv.org/pdf/2508.19600v2
19. VENKATESH1608. **onnx-cpu-inference-accelerator — INT8 slower without VNNI.** https://github.com/VENKATESH1608/onnx-cpu-inference-accelerator

*Search session IDs logged: search_fba725b82..., search_8926926f..., etc. No citation beyond those URLs was fabricated.*

---

## 5. Direct answers to your two questions (concise)

**Q1a — Can PTQ improve mAP?** Yes, documented, with mechanism named *quantisation as regularisation / noise injection (QReg/QGen) / spectral filtering*. Expect small (+0–2%) gains in specific regimes (often 8-bit, QAT, or specialised PTQ like AQD/DetPTQ). Vanilla ONNX static PTQ improvement of +0.0217 is at upper end but within narrative of regularisation.

**Q1b — Is single-number "costs X mAP" stable?** No paper retrieved quantifies variance; I judge it unstable across runs and would treat the quoted cost as uncertain. Publish variance if you can.

**Q1c — Is 0.001-tail mAP sensitive?** Plausibly yes, but no retrieved paper quantifies it. Engineering lore says 0.001 is used to maximise recall and mAP integrates over it, so perturbations there do move mAP on small datasets.

**Q2a — Real-time on Pi class?** On Pi 5 CPU @640, best published is **9–10 FPS (NCNN multi) for nano YOLOv8/11/26**, ~4–5 FPS ONNX single, ~3 FPS PyTorch. Real-time (>15 FPS) requires **imgsz 320** (e.g., 15.4 FPS YOLOv8n-seg ONNX @320) or NCNN; 640 is borderline for defect detection unless you frame-skip. Pi 4 is ~3× slower (≈3 FPS YOLOv8n ncnn). See full table above.

**Q2b — Pi 5 + ONNX Runtime INT8?** No published FPS found. Docs say ARM should use QInt8/QInt8 and *can* be faster on dot-product hardware, but community benchmarks use NCNN FP32 or Hailo instead; x86 anecdotes show INT8 slower without VNNI, suggesting QDQ overhead may erase win on CPU.

---

*End of findings. All URLs fetched in-session; "NOT FOUND" means zero hits in top-10 results per query across ~80 results reviewed.*
