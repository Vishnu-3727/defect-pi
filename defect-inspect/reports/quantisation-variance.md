# The INT8 delta has a sign, and nobody reports its variance

> **PREMISE WITHDRAWN 2026-09-19.** The opposite-signed deltas this report is
> built on were an artefact of comparing PyTorch FP32 against INT8 ONNX. Measured
> within one runtime, all ten checkpoints lose ~0.007 mAP@50. The calibration-draw
> sweeps below stand as measurements; the framing does not. See
> `quantisation-corrected.md`.


## The observation

Two YOLOv8n detectors, same architecture, same dataset, same 100-epoch recipe,
same ONNX Runtime static quantisation with the same calibration set. Only the
training run differs.

| model | FP32 mAP@50 | INT8 mAP@50 | Δ |
|---|--:|--:|--:|
| v1 (Aug 2026 run) | 0.7107 | 0.7324 | **+0.0217** |
| v2 (Sep 2026 run) | 0.7624 | 0.7353 | **−0.0271** |

Identical procedure, opposite signs, and the magnitudes are comparable.

`README.md` and `benchmark.md` carried the line "INT8 costs 0.3 mAP" as though
it were a property of the quantisation. It is a property of *one run*.

## The effect is established; its variance is not

**Quantisation improving accuracy is a documented phenomenon with a name.**

- **QReg: On Regularization Effects of Quantization** — AskariHemmat, Askari
  Hemmat, Hoffman, Lazarevich, Saboori, Mastropietro, Sah, Savaria, David.
  arXiv:2206.12372, 2022. <https://arxiv.org/abs/2206.12372>
  Models weight quantisation as additive noise Δ ∼ U[−δ/2, δ/2], giving an
  effective loss ℒ̃ = ℒ + R(Δ). States that "weight quantization is a form of
  regularization and the amount of regularization is correlated with the
  quantization level", and that "8-bit quantization provides a reliable form of
  regularization in different vision tasks and models". *(Verified directly.)*

- **AQD: Towards Accurate Quantized Object Detection** — Chen, Liu, Zhuang, Tan,
  Shen. CVPR 2021, pp. 104–113.
  <https://openaccess.thecvf.com/content/CVPR2021/html/Chen_AQD_Towards_Accurate_Quantized_Object_Detection_CVPR_2021_paper.html>
  Detection-specific, and the closest prior result: 4-bit quantised RetinaNet
  **surpasses the full-precision model by 1.8% AP** (ResNet-18) and 0.8%
  (ResNet-34). *(Verified directly.)*

- **Training with Quantization Noise for Extreme Model Compression** — Fan et al.,
  ICLR 2021, arXiv:2004.07320. States quantisation noise "regularizes the network
  in a similar way as DropConnect or LayerDrop". Training-time noise, not PTQ.

- **QGen: On the Ability to Generalize in Quantization Aware Training** —
  arXiv:2404.11769, 2024. Extends QReg with flat-minima analysis over ~2000
  models; quantised models converge to flatter minima with smaller generalisation
  gaps. QAT rather than PTQ.

So a positive Δ is not anomalous, and the mechanism has a literature.

**What is missing is the variance.** A targeted search found no paper
characterising the run-to-run distribution of the post-training quantisation
delta across training seeds or calibration draws. Every source quotes a single
figure — "INT8 costs X mAP" — with no interval. AQD reports the improvement as a
result of *its method*; nothing retrieved treats the sign as something that can
flip between two runs of an unchanged recipe.

## Why this is worth a paragraph in the paper

The two deltas here differ by 0.049 mAP@50 — larger than most of the
improvements this project has otherwise measured, and larger than the gap that
motivated its resolution hypothesis. If the quantisation delta can swing that
far between training runs of the same recipe, then:

1. A single-point "quantisation costs X mAP" claim, quoted without a seed count
   or an interval, is not a measurement. Both of this project's own reports
   made that claim.
2. Any comparison between two quantised models that differ by less than ~0.05
   mAP@50 is inside the noise of the quantisation step alone, before considering
   training variance.

The honest experiment is cheap and not yet run: quantise one FP32 model N times
with different calibration subsets, and quantise K independently trained models,
then report the distribution. That would turn an anecdote into a number and is
a defensible small contribution in its own right.

### The likely mechanism, stated as a hypothesis

mAP is swept from a confidence floor of 0.001, so it is dominated by the
low-confidence tail. Quantisation perturbs that tail. Whether the perturbation
promotes marginal true positives above marginal false positives, or the reverse,
is not obviously constrained to one direction.

Consistent with this, the two models move in the *same* direction on the
stricter metric — v1 −0.0165 and v2 −0.0264 on mAP@50-95 — where localisation
precision is graded and noise cannot help. Only the loose-IoU metric flips sign.
That is a testable prediction, not a demonstrated mechanism: no retrieved source
analyses the sensitivity of mAP-at-0.001 to low-confidence perturbation.

## Deployment note from the same sweep

No published Raspberry Pi 5 figure for ONNX Runtime with INT8 was found, so
`benchmark.md`'s 36.7 FPS appears to be an uncontested data point rather than a
comparable one. Published Pi 5 CPU figures retrieved for context: YOLOv8n ONNX
at 640 runs 4.55–6.74 FPS, NCNN 9.0–9.3 FPS, and one report gives 15.4 FPS for
ONNX at 320 input.

One caveat from the ONNX Runtime documentation is worth recording: QDQ INT8 can
be *slower* than FP32 on CPUs lacking dot-product or VNNI instructions. The Pi
5's Cortex-A76 implements ARMv8.2 with DOTPROD, which is consistent with the
measured INT8 speedup (22.4 → 36.7 FPS) rather than a slowdown — but on a
Cortex-A72 (Pi 4) the same model could regress.

Raw sweep output, with the searches that returned nothing, is in
`paper/lit/quantisation-and-pi-benchmarks.md`.
