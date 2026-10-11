# Paper outline — defect-inspect

> **SEPARATION RESTATED 2026-09-19.** The "median defect 2.40 vs median
> normal 2.26, separation +0.141" quoted from this project's calibration was
> measured on 46 regions per group. On 669 regions the same statistic reads
> **+0.273 with AUROC 0.819**. PatchCore separates better than this project has
> been assuming, and "weak separation" should not be used as a premise until
> re-measured. See `../reports/patchcore-denoise.md`.


Working title: **Real-Time Surface Defect Inspection on a CPU-Only Raspberry Pi 5:
An INT8-Quantised Detector with an Unsupervised Backstop**

Status: outline. Every number below is sourced from this repository. Anything not
sourced is marked **[GAP]** and must be measured before it goes in a draft.

---

## 1. What the contribution actually is

Be honest about this in the paper itself, because reviewers will be.

There is **no novel architecture here.** YOLOv8n is off the shelf, NEU-DET is a
public benchmark, PatchCore is a published method, and INT8 post-training
quantisation is standard. A paper claiming novelty in any of those will be
rejected on the first pass.

What is defensible, and genuinely uncommon in the student/industrial-deployment
literature:

1. **An end-to-end deployed system measured on the target hardware**, not on a
   development GPU, with the development host's numbers explicitly excluded from
   the results.
2. **Measurement hygiene that is stated and checked** — thermal bracketing, warm-up
   discards, p95 rather than mean, and a numerical parity check proving the
   deployed ONNX graph computes what the training framework computed.
3. **A characterised failure mode**: the anomaly backstop is reported as *weak*
   with the numbers that show it, rather than being omitted or oversold.

That is an engineering/measurement contribution. It fits an applied venue, a
systems track, or a conference on industrial informatics — not a vision venue
looking for architectural novelty.

## 2. Suggested structure

| § | Content | Evidence available |
|---|---|---|
| 1 | Introduction — inspection on the line, why CPU-only, why the edge | narrative |
| 2 | Related work — NEU-DET results, YOLO on edge devices, PatchCore | **[GAP]** literature survey not written |
| 3 | Dataset and preparation — NEU-DET, VOC→YOLO, stratified split | sourced |
| 4 | Method — detector, quantisation, gated anomaly backstop, decision fusion | sourced |
| 5 | Experimental setup — hardware, runtime, protocol | sourced |
| 6 | Accuracy results | sourced |
| 7 | Latency and quantisation study on the Pi 5 | sourced, **needs re-measurement** |
| 8 | System-level throughput | sourced |
| 9 | Limitations | sourced, and unusually strong |
| 10 | Conclusion | — |

## 3. Numbers this repository can already support

### Dataset (§3)
- NEU-DET: 1770 grayscale images, 200×200, six classes, Pascal-VOC boxes.
- Stratified split 1236 train / 353 val / 181 test (70/20/10), seed fixed.
- Median defect box covers 11.8% of image area; only 1.2% of boxes fall below 1%.
  This is the justification for using stock detection heads rather than adding a
  high-resolution small-object head — state it, it is a design decision a reviewer
  will otherwise ask about.

### Accuracy, test split, 320×320 (§6) — `reports/metrics.md`
mAP@50 **0.7106**, mAP@50-95 **0.4066**, mean P 0.6274, mean R 0.6623.

| Class | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| crazing | 0.5489 | 0.2222 | 0.3855 | 0.1354 |
| inclusion | 0.7100 | 0.7619 | 0.8711 | 0.4563 |
| patches | 0.7505 | 0.8953 | 0.9111 | 0.5664 |
| pitted_surface | 0.5603 | 0.7281 | 0.7451 | 0.5437 |
| rolled-in_scale | 0.5457 | 0.5068 | 0.5138 | 0.2509 |
| scratches | 0.6489 | 0.8596 | 0.8370 | 0.4870 |

The crazing row is the paper's most interesting result and must not be buried:
**recall 0.2222** means roughly four in five crazing defects escape. Crazing is
low-contrast hairline cracking whose boundary is ambiguous even to a human
annotator. Report it, explain it, and let it motivate the backstop in §4.

### Quantisation and latency (§7) — `reports/benchmark.md`
INT8 costs **0.3 mAP** and halves the model: 12.1 MB → 5.6 MB.

Previously measured on the Pi 5 (60 class-balanced test images, 3 repeats,
10 warm-up frames discarded): FP32 22.4 FPS and INT8 **36.7 FPS** at 4 threads,
against a 20 FPS design target. Peak RSS 122 MB, 2.22 boxes/frame after NMS.

> **These numbers must be re-measured before submission — see §5 below.**

### Resolution sensitivity (§7) — `reports/resolution-study.md`
Measured 2026-09-18. Same weights (trained at 320), same test split:

| imgsz | P | R | mAP@50 | mAP@50-95 | ms/img |
|---|---|---|---|---|---|
| **320** (native) | 0.627 | 0.662 | **0.7107** | **0.4063** | 22.1 |
| 640 | 0.302 | 0.338 | 0.3091 | 0.1436 | 90.2 |

The 320 row reproduces `metrics.md` to four decimals (0.7107 vs 0.7106), which
validates the measurement environment.

Running at twice the training resolution costs **57% of mAP@50** for 4.1× the
compute. Frame this as a deployment hazard in the paper: `imgsz` sits in the
config beside thread count, but thread count trades speed for nothing while
`imgsz` silently invalidates the model — no error, boxes still appear, the
dashboard still shows a plausible reject rate.

### Controlled retrain (§7) — `reports/resolution-retrain.md`
The cross-project hypothesis was tested properly on Kaggle (T4, both resolutions
trained in one environment, split pinned by manifest). **It is refuted**, and the
negative result is the more interesting section:

| model | eval 320 | eval 640 |
|---|---|---|
| trained at 320 | **0.7624** | 0.3438 |
| trained at 640 | 0.7277 | **0.7629** |

- Train-640/deploy-320 **costs** 0.035 mAP@50; it does not gain 0.033.
- Each resolution is equally accurate at its native size (0.7624 vs 0.7629,
  noise on 427 instances), so 640 buys nothing for 4.2× the compute on 200×200
  source images.
- The mismatch asymmetry is real and larger than estimated: −0.419 evaluating
  above the training resolution, −0.035 below. Twelve times the penalty.

**The control is the methodological point worth publishing.** Retraining at 320
in the current stack scores 0.7624 against the deployed 0.7107 — **+0.052 from
the training environment alone**, larger than the 0.033 the original hypothesis
rested on. Had the new 640 model been compared against the old 320 weights, it
would have read +0.017 and "confirmed" a hypothesis the internal comparison
refutes. A paper that reports resolution effects without holding the training
stack fixed is reporting its library version.

Actionable side effect: `best_trained320.pt` at 0.7624 is a genuine +0.052 over
the deployed model at identical inference cost. Adopting it needs ONNX parity
re-run, INT8 recalibration, and a re-check of `anomaly.score_threshold`, which is
tuned to the old detector's confidence distribution.

### Parity (§5)
ONNX pipeline vs ultralytics: 132/132 boxes matched at **IoU 1.0000**. This is the
check that lets §6's accuracy numbers, measured through ultralytics, be claimed for
the ONNX graph that actually runs. Without it the two sections are about different
programs. Keep it — reviewers rarely see it and it is cheap to state.

### System throughput (§8)
Full pipeline — capture, detect, gate, decide, store, eject — **17.1 FPS**
end to end, against 36.7 FPS for inference alone. The gap is dominated by SQLite
commits and JPEG encoding of failure frames. Report the system number as the
headline and the inference number as a component; the reverse is the standard way
these papers overstate themselves.

### Anomaly backstop (§4, §9) — `reports/anomaly.md`
Motivation is definitional, and it is the cleanest argument in the project: a
detector can only report classes it was trained on, so an unseen seventh defect
type produces no detection and the part passes. PatchCore has no classes and
measures distance from normal instead, so it cannot have that blind spot — and
cannot name what it finds. Fused with OR, not AND.

Honest result: separation is **weak** on this dataset — median defect score 2.40
against median normal 2.26, threshold 2.77 set at the 95th percentile of
defect-free regions. Publish this as a negative result with the distributions
plotted. NEU-DET contains no clean defect-free images, so the "normal" bank is
built from defect-free *regions* of defect images, which is the likely cause. That
is a real finding about the dataset, not just about the method.

## 4. Limitations to state plainly (§9)

1. Crazing recall 0.22 — the system is not fit for crazing-critical inspection.
2. The anomaly backstop is weakly separating and is a backstop, not a classifier.
3. No physical actuation has been validated: the ejector firmware has never run on
   hardware, and results are for the mock backend. **[GAP]**
4. Single-camera, fixed lighting, no motion blur: NEU-DET images are static
   captures, not frames from a moving strip. Generalisation to a real line is
   unproven. **[GAP]**
5. No comparison against a baseline detector at the same budget. **[GAP]**

## 5. Blocking issues before any draft

**(a) The benchmark numbers are currently invalid.** On 2026-09-18 this Pi was
measured under-volting: `dmesg` reports `hwmon3: Undervoltage detected!`,
`vcgencmd get_throttled` flickers to `0x50005` (under-voltage *and* throttled, live
bits, not the sticky ones), and the ARM clock collapses from 2400 MHz to 1000 MHz.
Live p95 latency read 157 ms against the 51 ms in `reports/benchmark.md`.

Every FPS and latency figure in §7 and §8 must be re-measured on an adequate
supply — the official 27 W USB-C PD unit — with `get_throttled` verified as
`0x0` or sticky-only *before and after* the run and reported in the paper. A
reviewer who knows Pis will ask, and a throttled measurement is not reproducible.

**(b) No related-work section exists.** This is the single largest gap. NEU-DET has
a substantial published baseline table; the paper needs to sit in it.

**(c) No baseline comparison.** At minimum: YOLOv8n at another input size, or
YOLOv5n/YOLOv8s at equal latency. Without one, §6 is a single unanchored number.

**(d) Reproducibility statement.** State the commit hash, the fixed seed, the exact
split, onnxruntime version and thread count. Most of this is already in the repo;
it just needs collecting into a table.

## 6. Figures worth making

1. Per-class P/R bars with the crazing row visibly worst — the honest headline.
2. Latency vs thread count, FP32 against INT8, with p95 whiskers not just medians.
3. PatchCore score distributions, defect against normal, overlapping, with the
   threshold drawn — the figure that makes the negative result legible.
4. The pipeline diagram from `docs/18_system_architecture.md`.
5. Confusion matrix (already generated in `reports/`).

---

## Findings added 2026-09-18/19 — these now outrank the original framing

The measurement work below displaced most of the contributions this outline was
written around. Each links to a committed report with its own limits section.

| finding | report | strength |
|---|---|---|
| **Prevalence decides the operating point.** On KolektorSDD2 (real defect-free images, 11% prevalence) the cost-optimal threshold is 0.35; at prevalence 1.00 — the NEU-DET regime — it collapses to 0.01 *for every cost ratio*, including 1:1. Reproduced by sweeping p on data that has real negatives. | `ksdd2-prevalence.md` | strongest |
| **F1 tuning is provably degenerate** on defect-only benchmarks: precision = 1 at every threshold, so F1 = 2R/(1+R) is monotone in recall and always picks the lowest threshold. Costs 3.1× more per part at p=0.05, R=30. | `operating-point-method.md` | formal result |
| **`val(imgsz=320)` feeds the network 352.** rect batching pads to a stride multiple; exported models have fixed input and cannot. Bias is model-dependent (+0.021, −0.025, +0.016) and **reverses a model ranking**. | `imgsz-rect-trap.md` | verified in library source |
| **Evaluation settings span 0.0254 mAP@50** — more than the v1→v2 model gap (0.0057), the quantisation cost (0.0067), or most increments in the NEU-DET literature. | `evaluation-settings.md` | strong |
| **Parity at the operating point ≠ parity at the sweep.** INT8 box agreement is 0.884 at conf 0.35 and 0.684 at 0.001, where mAP integrates. | `imgsz-rect-trap.md` | new method |
| **The detector's blind spot is real**: 74% of parts carrying an unseen defect type pass silently at the deployed threshold. Lowering it does not fix that. | `holdout-ood.md` | solid |
| **Classification error is absent**, 0.00 on all six classes across two independent models. A low mAP here does not mean class confusion. | `tide-cross-system.md` | two-system |

### Claims withdrawn after testing

| withdrawn | why |
|---|---|
| "Lower the decision threshold to 0.10" | overkill rises 23%→60%; break-even ratio 84–437 at realistic prevalence |
| "INT8 imposes a ceiling" | slope 1.025, R² 0.997 across ten checkpoints — my artefact |
| "The INT8 delta flips sign between models" | compared PyTorch FP32 against ONNX INT8 |
| "Crazing's box partition is arbitrary" | region-level scoring does not rescue it |
| "Crazing is distinctively Loc/Bkg-dominated" | rolled-in_scale is worse on both axes |
| "PatchCore separates weakly (+0.141)" | artefact of n=46; AUROC is 0.819 |
| "PatchCore is redundant" | true only for in-distribution defects |
| "Crazing's Bkg is missed annotation" | only 3% of unmatched boxes exceed conf 0.5 |

Eight hypotheses tested and killed. **[GAP]** The paper's framing should lead with
measurement validity rather than with the detector, because that is where the
defensible contributions are.
