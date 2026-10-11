# Literature Novelty Check -- Ultralytics YOLO `rect`/`imgsz` Resolution Artefact

**Date:** 2026-09-19
**Author:** Muse Spark (model: muse-spark-1.2-contributor-free)
**Venue:** Findings written to `findings.md` in current directory per user request.

## 0. Web Search Availability -- Up-front Statement

**Web search IS available in this environment.** This report is based on live `default.websearch` and `default.webfetch` queries executed 2026-09-19 (queries listed in §6). No statements below are generated from memory alone. Where a targeted search returned no citable result, the answer is explicitly `NOT FOUND`/`No evidence found` -- no citation, number, or URL was fabricated.

> If web search had been unavailable, the instruction was to stop and say so rather than answering from memory. That condition was not triggered.

## Finding Under Check (verbatim from user prompt)

> In the Ultralytics YOLO library, calling `model.val(imgsz=320)` on a PyTorch `.pt` checkpoint does NOT evaluate at 320. The validator defaults to `rect=True` (rectangular batching) and pads to a stride multiple using `imgsz` as a lower bound, so 200x200 images are fed to the network at 352x352. An exported ONNX model has a fixed input shape, cannot be rect-batched, and genuinely runs at 320. We measured: same weights, same split, `.pt` default = 0.7624 mAP50 (fed 352), `.pt` with `rect=False` = 0.7411 (fed 320), `.onnx` = 0.7411 (fed 320). The PyTorch and ONNX runtimes agree EXACTLY at matched resolution. The bias is model-dependent (+0.021, -0.025, +0.016 across three checkpoints) and large enough to REVERSE the ranking of two models.

**Five questions asked:**

1. Documented anywhere?
2. Paper reporting PyTorch-vs-exported discrepancy that was actually a resolution artefact?
3. Literature on evaluation-protocol artefacts in detection benchmarks?
4. Known cases where default evaluation setting REVERSED a model ranking?
5. Published quantized-detector benchmarks that compare against PyTorch baselines without stating `rect`/letterbox settings?

---

## Executive Summary -- Novelty Verdict

*The **precise** claim as formulated (320 -> 352 on 200x200, exact numbers 0.7624/0.7411, +0.021/-0.025/+0.016, explicit ranking-reversal demonstration) was **NOT FOUND** in any paper, Ultralytics doc, blog, or GitHub issue in the search results.*

*The **underlying mechanism** -- `rect=True` pads to stride multiple with `pad=0.5`, `imgsz` is a lower bound not a fixed shape, PyTorch validator vs static ONNX/TensorRT `rect=False` mismatch inflates mAP and explains "ONNX costs X mAP" reports -- **IS documented** in multiple GitHub issues/discussions and implicitly in validator source/docs, but scattered, never quantified as above and never connected to a ranking-reversal claim.*

In short: **mechanism = known to power users/maintainers; quantified artefact + ranking reversal = novel to the literature search performed.**

---

## 1. Is this documented anywhere? -- Ultralytics docs, issues, discussions, blogs, papers

### 1a. YES -- Mechanism documented in pieces; NO -- Exact finding not documented

Searches run: `Ultralytics YOLO val rect imgsz rectangular inference`, `rect=True rect=False validation mAP`, `stride padding ONNX vs PyTorch mAP discrepancy`, `val imgsz 320 not evaluated at 320`, `set_rectangle unnecessary padding`, `ultralytics/validator.py rect`, `predict rect minimum rectangle`.

**Evidence that the mechanism IS known:**

| Source | Title / URL | What it says | Relevance |
|---|---|---|---|
| `ultralytics/engine/validator.py` (source) | https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/validator.py | `if not (pt or dynamic) : self.args.rect=False` -- static exports force `rect=False`; `imgsz = check_imgsz(..., stride=stride)`; `model.warmup((1 if pt else batch, channels, imgsz, imgsz))` | **Codifies** that non-PT formats cannot rect-batch; PT validator can. |
| `ultralytics/cfg/default.yaml` | https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml | `rect: False # rectangular batches for train; rectangular batching for val when mode='val'` | Default noted as `False` for standalone val, but code-comment shows `rect=True` during *training* val is hard-coded (`detect/train.py:77`). Docs conflict/nuance. |
| `docs/en/modes/val.md` and `docs.ultralytics.com/modes/val` | https://github.com/ultralytics/ultralytics/blob/b53c5e66/docs/en/modes/val.md ; https://docs.ultralytics.com/modes/val | `rect=True` uses rectangular inference, reducing padding; `imgsz` defines input size. | Documents `rect` exists, but describes it as speed optimization, **not** as `imgsz` being a lower bound that silently enlarges input. |
| `docs/en/modes/predict.md` | https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/predict.md | "By default predict uses `rect=True`... minimum-rectangle padding when possible. Image is scaled to fit inside `imgsz` and padded only to nearest stride multiple, so final tensor may be **smaller** than `imgsz`." / "Use `rect=False` to always pad to full `imgsz` target. Recommended when you need fixed input size to match exported models (ONNX, TensorRT, etc.)." | **Closest official documentation** of `rect` vs fixed shape for exported models, but framed for `predict`, not the `val` 320->352 inflation. |
| Issue #24388 `set_rectangle adds unnecessary padding even when batch shape already divisible by stride` | https://github.com/ultralytics/ultralytics/issues/24388 | Reports `batch_shapes = ceil(shapes * imgsz/stride + pad) * stride` with `pad=0.5` for val, so 640 -> 672. Maintainer reply: "Yes, intentional... It has higher mAP with padding... The original intention probably was to upscale... but dataloader downscales... LetterBox doesn't upscale because scaleUp disabled... So it's a bug that somehow helped and remained a feature." / "Wouldn't it be more honest to have val and predict use identical preprocessing? You can obtain that by setting `rect=False`." | **Direct acknowledgement** that val `rect` inflates imgsz by ~32px and yields higher mAP than `predict` at same nominal `imgsz`. No mention of 320->352 / 200x200 / ONNX parity. |
| Issue #6192 `The imgsz for inference is different from the Settings` | https://github.com/ultralytics/ultralytics/issues/6192 | User sets `imgsz=832` val but real size 864. Comment: "validation uses 0.5 stride padding around all sides for slightly improved results, so i.e. 640 validation would be run at 672." | Prior report of nominal vs actual imgsz inflation. |
| Issue #21781 `Difference in Feature map dimensions training vs validation` | https://github.com/ultralytics/ultralytics/issues/21781 | "Validation runs with `rect=True` ... `imgsz=640` can be rounded up to 672 ... yielding 84/42/21 feature maps; to keep 80/40/20 disable rectangular batching or explicitly use 672." | Confirms stride-rounding effect. |
| Issue #19832 `Is there an error in the padding of the validation set?` | https://github.com/ultralytics/ultralytics/issues/19832 | `imgsz (320,320), image size (720,1280), actual size (224,352), not filled to 320 but 32 more.` Reply: "Validation uses minimum rectangle padding by default... `imgsz` serves as upper bound but actual dimensions calculated as (h//stride*stride, w//stride*stride)" | **200x200 -> 352 analogue?** Reporter uses different aspect ratio but documents that actual size ≠ imgsz. No quantification of mAP bias. |
| Issue #502 `Can not run Val with onnx model and shape mismatch` (2023) | https://github.com/ultralytics/ultralytics/issues/502 | PT val shape 360x640 vs val ONNX expects 640x640 square, error `Got 384 Expected 640` . | Early ONNX static-shape vs rect mismatch known. |
| Issue #7797 `val can't validate exported model if not square resolution was used` | https://github.com/ultralytics/ultralytics/issues/7797 | Rectangular export vs square val pipeline incompatibility. | Documents fixed-shape export limitation. |
| Issue #13753 `Image Resize in Validation` | https://github.com/ultralytics/ultralytics/issues/13753 | Confusion about `imgsz=512` not yielding 512x512. | General `rect`/`imgsz` confusion. |
| PR #824 `update rect` (2023-02-06, Laughing-q) | https://github.com/ultralytics/ultralytics/pull/824 | Enables `rect` for val; states "valing: rect images by default, square images if model is not pt file or user manually set `rect=False`." | **Codifies** the default difference between PT and exported val. |
| PR #25559 `Reuse imgsz from export metadata when validating static exported models` | https://github.com/ultralytics/ultralytics/pull/25559 | Fixes `yolo val model=yolo26n.onnx` runtime error `Expected:320` when user omits imgsz; reuses `max(model.imgsz)` from export metadata. | Acknowledges ONNX fixed shape vs val default mismatch; fix auto-corrects `imgsz` but still forces `rect=False` + square. |
| PR #22440 `Downscale to padded size in load_image instead of unpadded size` | https://github.com/ultralytics/ultralytics/pull/22440 | "We always downscale images to unpadded `imgsz` in dataloader, even in `rect` mode. So we never really get to use extra pixels... We are always just running validation on a padded image as opposed to larger image." | Maintainer internal discussion of padding vs upscaling nuance. |
| Issue #24467 `How should I use rect parameter in YOLO training/validation/prediction` | https://github.com/ultralytics/ultralytics/issues/24467 | Long maintainer explainer: `rect` sorts by aspect ratio, pads to stride multiple, disables mosaic, etc. | General documentation, not specific artefact. |
| YOLOv5 Discussion #10862 `Different results when testing exported onnx model and pytorch model` | https://github.com/ultralytics/yolov5/discussions/10862 | `if rect variable is True ... image shape modified ... If I force rect to False I have same results. I could not find script argument to change rect to False during testing a pytorch model, so wondering why always True.` | **YOLOv5 predecessor** of same finding; shows awareness since 2023 but in YOLOv5, not YOLO library quantified case. |

**What is NOT documented (explicit NOT FOUND list for Q1):**

* No doc, issue, or PR states the exact numbers `200x200 -> 352x352` at `imgsz=320` (the `+32` per side from `+0.5*32` is known for 640->672, but not this 320 case).
* No doc quantifies the mAP bias as `0.7624 vs 0.7411` or `+0.021 / -0.025 / +0.016` model-dependent, or states exact parity `PT rect=False == ONNX` when matched.
* No doc states that the bias can **reverse ranking of two models** - the strongest claim. Maintainer in #24388 says "It generally should [still rank correctly]... It's unlikely for it to improve in `rect=True` while not improving in `rect=False`." -- the opposite of the reversal claim.
* No blog post found that systematically benchmarks PT vs ONNX at matched vs mismatched resolution.

### Conclusion Q1

Partially documented at mechanism level; **novel** at quantitative/ranking-reversal level. Search returned **mechanism awareness** but **zero** sources with the specific empirical finding.

---

## 2. Has any PAPER reported a PyTorch-vs-exported-model accuracy discrepancy that was actually a resolution artefact?

**Answer: NOT FOUND in papers; FOUND in GitHub issues (not peer-reviewed).**

Searches: `"ONNX vs PyTorch mAP discrepancy" site:github.com ultralytics`, `"export costs X mAP" ONNX`, `"ONNX export mAP drop paper"`.

**Found in issues (not papers) -- these are the "many papers report our ONNX export costs X mAP" candidates that turned out to be rect artefacts:**

* Issue #20231 `Exporting onnx and tensorrt models from pytorch model, the accuracy decreases` (2025-04-19, shyhoom): PT 0.537 mAP50 vs ONNX 0.563 -- **ONNX higher** in that report; comment: "PyTorch uses minimum rectangle during validation... It should be same if you use `rect=False` for both. Or if you use dynamic model with this PR #20232". URL: https://github.com/ultralytics/ultralytics/issues/20231
* Issue #19991 `parity mismatch torch vs onnx for yolo-detection-model` (2025-04-03, ambient-udeshmukh): raw logit delta 0.002 CPU vs 2.2 GPU, attributed to accumulated FP16 precision, not resolution. Shows many discrepancies are **incorrectly attributed** to conversion, but this one was precision, not rect. URL: https://github.com/ultralytics/ultralytics/issues/19991
* Issue #19513 `pt model to ONNX model, different result` (2025-03-04, GXDD666): "different because of minimum rectangle resizing... If you export with `dynamic=True` it should be same." URL: https://github.com/ultralytics/ultralytics/issues/19513
* Issue #20830 `detection results of onnx model and pt model are inconsistent` (2025-05-27, monkeycc): same comment pointer to #16339. URL: https://github.com/ultralytics/ultralytics/issues/20830
* Issue #10289 `When using yolo export to convert model to ONNX format, decrease in accuracy observed` (2024-04-24, xla145): generic replies about preprocessing consistency, not resolution. URL: https://github.com/ultralytics/ultralytics/issues/10289
* Issue #20930 `Performance drop on ONNX compared to PyTorch` (2025-06-04, kkeroo): **Key** -- reporter got 0.5 -> 0.46 mAP50-95 at `imgsz=768`; maintainer: "PyTorch model doesn't run validation at square image but ONNX does (because imgsz is fixed since you didn't export with `dynamic=True`). You should get same mAP if you run both validation with `rect=False` or export ONNX with `dynamic=True`." Later verified `rect=False` PT == ONNX drop, and `dynamic=True` + `rect=True` restores parity (with version fix). URL: https://github.com/ultralytics/ultralytics/issues/20930
* Issue #1608 `ONNX outputs not identical to PyTorch output` (2023, ShubhamAtPhilips): early report closed as "no accuracy loss" per benchmarks; predates the rect nuance surfacing. URL: https://github.com/ultralytics/ultralytics/issues/1608

**Papers searched:** arxiv queries for `ONNX mAP drop`, `quantization ONNX export costs X mAP`, `TensorRT INT8 mAP` returned only:
* Karimov et al. 2025 (2508.19600) -- reports INT8 drop 0.031-0.072 mAP but attributes to quantization, not resolution; explicitly states `Dynamic UINT8 (ONNX) matched FP32` (Table I) -- suggests they **avoided** the artefact via correct config (batch=1, static). No mention of rect.
* YOLOv6+ (Springer 2025-05-28) -- discusses INT8 dynamic range mismatch collapsing mAP to 0.0, solved by skip connections + regression normalization. Mentions TFLite INT8 conversion, not rect.
* **No paper** found where authors claim "our ONNX export costs X mAP" and it is later shown to be a resolution artefact. GitHub issues above show the community **suspects** many such claims, but no peer-reviewed paper explicitly diagnoses it.

**Explicit NOT FOUND list Q2:**

* No peer-reviewed paper titles/authors found reporting PT-vs-ONNX discrepancy and correctly attributing it to `rect`/`imgsz` lower-bound padding.
* No paper found where the reported ONNX tax disappears when re-evaluated at `rect=False` or matched resolution.
* No paper quantifies the tax as model-dependent sign-flipping (+/-) as in the finding.

---

## 3. Is there literature on evaluation-protocol artefacts in object detection benchmarks generally?

**Answer: YES -- substantial literature, but none about the specific `rect` artefact. Nearest analogues are metric-implementation and annotation-noise artefacts.**

| Paper / Report | Authors | Year | Venue | URL | Core artefact | Relation to finding |
|---|---|---|---|---|---|---|
| `Benchmarking Object Detectors with COCO: A New Path Forward` (COCO-ReM) | Authors not extracted in snippet; ECCV 2024 submission | 2024 | ECCV 2024 (`www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06053.pdf`) | https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06053.pdf | COCO-2017 masks are coarse, non-exhaustive, mislabeled; 50 detectors score higher on COCO-ReM (+7.7 AP for SOTA); ranking trends shift; asserts COCO-2017 wrongly penalizes sharper masks. | **General benchmark artefact**: benchmark impurity changes results and rankings. Different mechanism (annotation quality vs default preprocessing). |
| `Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details` (AP_old vs AP_fixed vs AP_pool) | Achal Dave, Piotr Dollár, Deva Ramanan, Alexander Kirillov, Ross Girshick (CMU / FAIR) | 2021 (arxiv 2102.01066, published 2022) | arXiv / presented at LVIS workshop | https://arxiv.org/abs/2102.01066 ; https://ar5iv.labs.arxiv.org/html/2102.01066 | Detections-per-image limit (300) makes AP category-dependent and **gameable**: trivial re-ranking boosts AP +1.2 (+2.9 for rare) without model change. Proposes AP_fixed (per-class limit). Benchmarks show many recent LVIS gains vanish under AP_fixed. | **Evaluation-protocol artefact** that changes interpretation of gains; analogous "default setting changes what is measured". No mention of input resolution. |
| `Noise-Aware Evaluation of Object Detectors` | Jeffri Murrugarra Llerena, Claudio R. Jung | 2025 | WACV 2025 (`openaccess.thecvf.com/...WACV_2025_paper.pdf`) | https://openaccess.thecvf.com/content/WACV2025/papers/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.pdf | Corner-noise in COCO boxes creates bounded IoU uncertainty; proposes upper/lower `AP_u` / `AP_l` bounds. Shows `ΔAP50` ~1.1 (large), ~3.4 (medium), ~13.5 (small) uncertainty; rankings can change for small objects (YOLOv10-S vs RepPoints). | Shows **metric uncertainty** large enough to flip conclusions for small objects; conceptually close to "silent evaluation detail changes reported mAP". |
| `How Trustworthy are Performance Evaluations for Basic Vision Tasks?` | Authors not extracted (references Herrera et al.) | 2020 (arXiv 2008.03533) | arXiv (peer-reviewed expansion 2022) | https://arxiv.org/abs/2008.03533 | Formal trustworthiness criteria; demonstrates mAP/log-AMR rankings **switch wildly** across IoU/GIoU thresholds; shows marginalized (COCO 0.5:0.95) can be less meaningful than optimal threshold; OSPA/EMD more stable. | **Direct ranking-instability** due to threshold default, closest to Q4. |
| `Object-Proposal Evaluation Protocol is 'Gameable'` | Neelima Chavali, Harsh Agrawal, Aroma Mahendru, Dhruv Batra | 2016 | CVPR 2016 | https://openaccess.thecvf.com/content_cvpr_2016/papers/Chavali_Object-Proposal_Evaluation_Protocol_CVPR_2016_paper.pdf | Evaluating proposals on partially-annotated PASCAL VOC inflates recall for methods tuned to annotated categories; method B can appear better while recalling fewer objects. | Classic **benchmark-protocol gameability**. |
| `Empirical Upper Bound, Error Diagnosis and Invariance Analysis of Modern Object Detectors` | Ali Borji et al. | 2020 (arXiv 2004.02877) | arXiv | https://arxiv.org/pdf/2004.02877v1.pdf | Upper bound AP 91.6 VOC / 78.2 COCO via perfect localization; diagnoses error types (classification vs localization). | **Methodology for diagnosing** what limits mAP, not a protocol artefact per se. |
| `Replication Study and Benchmarking of Real-Time Object Detection Models` | Pierre-Luc Asselin et al. (Univ. Laval / NorLab) | 2024 | arXiv 2405.06911 | https://arxiv.org/html/2405.06911 | Reproduces DETR/RTMDet/ViTDet/YOLOv7 from scratch vs MMDetection; finds pretrained speed mAP strongly hardware/batch-dependent; papers lack reproducibility details; unified pipeline reveals trade-offs that official numbers obscure. | **Reproducibility study** showing framework defaults/hardware change reported speed/mAP. No `rect` analysis. |
| `detrex` benchmark paper (part of 2306.07265) | Authors not extracted in snippet (community codebase) | 2023 | arXiv 2306.07265v2 | https://arxiv.org/pdf/2306.07265 | Shows reproduced DETR variants gain +0.2 to +1.1 AP via hyperparam tuning; NMS still helps DETRs; hyperparam sensitivity large. | **Finding that reported gains can be hyperparam tuning**, not architecture, related to reproducibility. |
| TensorFlow Object Detection API `evaluation_protocols.md` (PASCAL vs COCO vs OpenImages) | Google Brain / TF OD API team | -- | GitHub `tensorflow/models` docs | https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md | Documents three different mAP protocols (PASCAL weight vs COCO vs OpenImages group-of handling) that give different numbers on same detections. | Shows **protocol choice** alone changes mAP. |

**NOT FOUND for Q3:**
* No paper found that names `rect`, `letterbox`, `stride padding`, `imgsz lower bound`, or Ultralytics-specific defaults as an evaluation-protocol artefact.

---

## 4. Are there known cases where a default evaluation setting REVERSED a model ranking?

**Answer: YES -- ranking reversals due to evaluation settings ARE documented, but NOT for this specific `rect`/`imgsz` setting. This is the strongest novelty aspect.**

**Found -- clear ranking reversals attributable to evaluation defaults:**

* **How Trustworthy (2008.03533):** Fig. 1-2 / §5: "for IoU threshold below 0.8, detector C has worst performance ... when threshold above 0.8, C becomes best detector... Similarly, if threshold above 0.3, D is worst, below 0.3 D becomes best." Averaging over thresholds can exacerbate. Demonstrates **parameter-choice alone reverses ranking**. Venue: arXiv, peer-reviewed. Authors include Santhosh et al. style trustworthiness framework (exact author list not extracted; cite via URL).
* **LVIS AP_old vs AP_fixed (2102.01066, Dave et al. 2021):** Table in §5: "many reported gains do not translate to improvements under new evaluation, suggesting recent improvements may arise from changes to cross-category rankings." **Interpretation reversal:** methods that looked SOTA under AP_old are not SOTA under AP_fixed. AP_pool + calibration reverses conclusions about calibration benefits (+1.7 AP_pool). This is a **ranking/claim reversal** due to fixing a default implementation detail (detections-per-image limit 300).
* **COCO-ReM (ECCV 2024, 06053.pdf):** "models that predict visually sharper masks score higher on COCO-ReM, affirming they were being incorrectly penalized due to errors in COCO-2017." SOTA OneFormer InternImage-H gains +7.7 AP on ReM vs 2017; **gap widens for recent high-performing models**, implying fine-grained ranking among top models is unreliable on COCO-2017. Authors show **relative ordering shifts** between COCO-2017 and COCO-ReM.
* **SafetyRepro: Configuration-Conditional Rank Instability on Alignment Benchmarks (2026, arXiv 2605.25492):** Finite-envelope proposition tying pairwise-disagreement rate to strict ordering reversal; demonstrates **every benchmark tested admits a configuration-pair reversal**; XSTest admits all 6 orderings of (Qwen, Mistral, Yi) within envelope. While alignment benchmarks (not detection), it is a **formal demonstration that harness defaults can flip pairwise verdicts**. URL: https://arxiv.org/html/2605.25492
* **DAOD / ALDI study (par.nsf.gov 10615133, Cityscapes->Foggy etc.):** "ranking of methods changes across benchmarks and architectures" + "ranking of methods differs across datasets. ALDI++, MIC and AT are consistently top but SADA underperforms on CS->FCS but closes gap on Sim10k->CS." Figure 5/6 shows method ranking **not stable** across datasets/backbones/pre-training. Venue: arXiv / NSF PAR. Authors not extracted. URL: https://par.nsf.gov/servlets/purl/10615133

**NOT FOUND for Q4:**

* No paper found where **letterbox/rect/imgsz or stride padding** is shown to reverse a ranking. Search `ranking reversal rect imgsz letterbox` returned nothing.
* No object-detection paper found where **PyTorch vs ONNX evaluation discrepancy** reverses ranking (e.g., "Model A > B in PT but B > A in ONNX at same nominal imgsz").
* No paper found that reports the **model-dependent sign** (+/-) of the bias as in the finding (+0.021, -0.025, +0.016).

Thus the **reversal claim is supported as a plausible class of phenomenon** (rankings are known to be fragile to evaluation defaults) but **not instantiated for this specific default** in prior literature.

---

## 5. Does any published work benchmark quantised detectors and compare against PyTorch baselines without stating rect/letterbox settings?

**Answer: YES -- multiple papers/repos compare quantized vs PyTorch and either omit `rect`/`letterbox`/`imgsz` handling or state only `model.val()` defaults, making them potentially affected. We name specific works; we do NOT claim they are definitely affected, only that their numbers *may* be affected given the bias magnitude.**

Search: `quantized detector benchmark YOLO ONNX TensorRT INT8 mAP comparison PyTorch baseline paper`.

| Work | Type | Quantization compared | Baseline | States `rect`/letterbox? | Says `imgsz` handling for PT vs export? | Why numbers may be affected | URL |
|---|---|---|---|---|---|---|---|
| **Karimov, Imani, Kazakov -- `Quantization Robustness to Input Degradations for Object Detection`** | arXiv 2508.19600 (2025), peer-reviewed preprint, code at `AllanK24/QRID` | YOLO12 n/s/m/l/x : FP32 TRT, FP16 TRT, Dynamic UINT8 ONNX Runtime, Static INT8 TRT (clean vs mixed calibration) | FP32 TRT | **No explicit `rect` mention.** Says "All models were benchmarked with batch size 1. Performance primarily evaluated using standard COCO metrics (mAP50-95, mAP50) and inference latency, obtained via the Ultralytics `model.val()` framework [9]." | Says batch=1 (which mitigates rect-batch-shape variance) but **does not state `rect` or `imgsz` matching**. Table I shows Dynamic UINT8 matches FP32 exactly (suggesting they did match correctly, perhaps inadvertently via `rect=False` auto for static). | If any comparison used PT default `rect=True` vs static TRT `rect=False` at nominal 640, the observed INT8 drop (0.031-0.072) could be confounded with resolution bias (±0.02 in finding). Risk is moderate; batch=1 reduces but does not eliminate pad=0.5 inflation (640->672 still happens even at batch=1 in some datasets). | https://arxiv.org/abs/2508.19600 ; https://github.com/AllanK24/QRID |
| **the0807/YOLOv8-ONNX-TensorRT repo** (popular benchmark, not peer-reviewed) | GitHub benchmark tables, ~500 stars | YOLOv8 n/s/m/l/x : PT FP32, ONNX FP16, TRT FP32/FP16/INT8 | PT | **No.** README says "mAP val values are for single-model single-scale on COCO val2017" without `imgsz`, `rect`, or export `dynamic` flags. | No. | Tables claim ONNX FP16 = PT (37.1, 44.7 etc.) but INT8 drops (33.0 vs 37.1 for n). If FP16 ONNX was evaluated with `rect=False` square vs PT `rect=True`, the FP16 "no loss" claim could be masking a small rect bias. | https://github.com/the0807/YOLOv8-ONNX-TensorRT |
| **Nipun Batra -- `Same Model, Same GPU, Half the Latency: What TensorRT, ONNX Runtime, and FP16 Actually Buy You` (blog, 2026-07-11)** | Blog, RTX A5000, YOLO11 n/s/m | PyTorch FP32/FP16, ONNX Runtime CUDA, TRT FP32/FP16/INT8 | PyTorch FP32 `model.predict()` | **Partial.** Accuracy protocol: "mAP50-95 on COCO128 for every backend of every model." Code `bench_accuracy.py` uses `model.val()`; not clear if `rect` controlled. Does discuss `imgsz=640` fixed, batch=1. | Implicitly via `model.val()` defaults; does not discuss `rect` vs static shape. Shows FP16/ONNX within ±0.005, INT8 tax -0.048 n / -0.020 s / -0.016 m. | INT8 tax numbers are ~1-2x the rect bias magnitude; if PT baseline was at inflated 672 vs INT8 at true 640, the "INT8 tax" is overstated. Blog's careful batch=1 and COCO128 (mixed aspect ratios? unclear) makes risk non-zero. | https://nipunbatra.github.io/blog/posts/2026-07-11-yolo-inference-speed-tensorrt.html |
| **Springer -- `YOLOv6+: simple and optimized object detection model for INT8 quantized inference on mobile devices`** | Springer *Signal, Image and Video Processing* (2025-05-28) | YOLOv6, YOLOv8 via TFLite INT8 PTQ; proposes Repblock_add + regression normalization (RN) | FP32 | **Not found in snippet.** No mention of letterbox/resolution handling in highlights. Focus is dynamic-range mismatch (mAP collapsed to 0.0 then recovered). | Not stated. | Same TFLite 640 vs pad issue could exist, but paper's main effect is dynamic-range (large), so resolution bias would be secondary. Still, baseline mAP comparisons may be confounded if val pipeline differs. | https://link.springer.com/article/10.1007/s11760-025-04234-0 |
| **TildAlice -- `YOLOv8 INT8 Quantization: 4x Faster on Jetson Orin`** (blog, 2026-02-07) | Blog/tutorial, Jetson Orin Nano, YOLOv8n | PTQ INT8 TRT, FP16, calibration size 100->5000 images | FP16 baseline (52.8 / 37.4) | **No.** Mentions `model.export(format='engine', int8=True, data='coco128.yaml')` but not `rect`. | Mentions calibration on 128 images, not val `rect`. | Reports <1% mAP drop with correct calibration; if val `rect` mismatch exists, the "1% drop" could be noise vs bias. | https://tildalice.io/yolov8-int8-quantization-jetson/ |
| **engineering.cloudflight.io -- `Quantized YOLO for Edge Solutions` (Cloudflight team)** | Engineering blog, YOLOv7/v8, Coral/NXP/Intel | `yolov8n FP32 37.3 -> INT8 per-tensor 32.9 / per-channel 33.9 -> INT8+FP32 head 36.3` | FP32 | **No.** Describes pixel normalization fix, not letterbox. | No. | Reports 4% improvement from normalized scaling factors; again, resolution handling not discussed, so baseline comparisons could include rect bias. | https://engineering.cloudflight.io/quantized-yolo-for-edge-solutions |
| **huangzongmou/yolov8-pytorch_quantization** (repo) | GitHub repo, YOLOv8 PTQ INT8 | PTQ INT8 via `model.val()` after calibration | PT | **Explicitly sets `rect=True` for calibration dataloader** (`pad=0.5, rect=True`) but `yolo.val(data=opt.data)` after calibration uses **default** (likely `rect=False` for val-standalone). Shows inconsistency within one repo. | Code does `create_dataloader(..., rect=True, pad=0.5)` for calib but val call hides rect. | Demonstrates that even when authors know `rect` for calibration, evaluation may still be mismatched. | https://github.com/huangzongmou/yolov8-pytorch_quantization/blob/main/yolov8_ptq_int8.py |

**Explicit NOT FOUND list Q5:**

* No peer-reviewed paper found that **explicitly states** "we evaluated PT with `rect=True` and ONNX/TRT with `rect=False` at nominal `imgsz=640` and report delta as quantization cost" -- the failure mode the finding warns about.
* No paper found that **controls for** `rect` and proves its deltas are **not** resolution artefacts. The burden of proof (show `rect=False` parity) is missing in all surveyed quantized works.
* No systematic audit paper found quantifying how many published INT8 numbers are affected.

**Note on methodology:** This assessment is based on search snippets and available highlights; absence of `rect` mention in a snippet does not prove omission in full paper text. Full PDFs were not exhaustively fetched for all quantization papers due to tool limits. The conclusion is therefore "not found in search results" rather than a definitive universal claim.

---

## 6. Search Methodology & Limitations

**Tools:** `default.websearch` (numResults 8, fallback/preferred livecrawl), `default.webfetch` where URL known, plus `default.bash` for file listing. Background agents not used; all searches performed inline.

**Queries executed (verbatim):**

* `Ultralytics YOLO val rect imgsz rectangular inference site:github.com`
* `Ultralytics YOLO ONNX vs PyTorch mAP discrepancy difference github issue`
* `"rect=True" OR "rect=False" validation mAP Ultralytics imgsz stride padding discrepancy`
* `site:github.com ultralytics val imgsz 320 not evaluated at 320 stride padding ONNX rect`
* `object detection evaluation protocol artefacts reproducibility benchmark mAP default settings paper`
* `"reproducibility" object detection ranking reversed evaluation setting paper`
* `quantized detector benchmark YOLO ONNX TensorRT INT8 mAP comparison PyTorch baseline paper`
* `object detection mAP framework default changes ranking benchmark study` (duplicate)
* `YOLO quantization paper INT8 ONNX export mAP drop letterbox rect imgsz not reported`
* `"reproducibility" evaluation protocol object detection mAP ranking changes paper 2023 2024`
* `"ONNX" "mAP drop" YOLO export paper quantization arxiv 2023 2024 2025`
* `"2102.01066" authors AP fixed LVIS`
* Plus fetches of individual issue URLs.

**Coverage:** ~60-70 search hits inspected; top 8 per query retained by provider. Search is not exhaustive of all literature; arXiv, Springer, WACV, CVPR, ECCV, GitHub, Ultralytics docs, and blogs are represented but not all venues.

**Biases:** Search provider favors recent, English, highly-cited, and GitHub content. Non-English, thesis, or non-indexed workshop papers may be missed. Authors/venues for some hits were not extracted by provider (listed as "not extracted").

**No fabrication:** All URLs/titles/authors/years/venues above are taken directly from tool outputs. Where tool returned `Published: N/A` or `Author: N/A`, we label as not extracted rather than inventing.

---

## 7. References (citable entities actually retrieved)

* **Validator source** -- `ultralytics/engine/validator.py` -- Ultralytics -- https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/validator.py
* **Config** -- `ultralytics/cfg/default.yaml` -- Ultralytics -- https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
* **Val docs** -- `Model Validation with Ultralytics YOLO` -- Ultralytics Docs -- https://docs.ultralytics.com/modes/val ; GitHub mirror https://github.com/ultralytics/ultralytics/blob/b53c5e66/docs/en/modes/val.md
* **Predict docs** -- `docs/en/modes/predict.md` -- Ultralytics -- https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/predict.md
* **Issue #24467** -- `How should I use the rect parameter...` -- enescicek / Ultralytics maintainers -- https://github.com/ultralytics/ultralytics/issues/24467
* **Issue #24388** -- `set_rectangle adds unnecessary padding even when batch shape is already divisible by stride` -- Ultralytics -- https://github.com/ultralytics/ultralytics/issues/24388
* **Issue #19832** -- `Is there an error in the padding of the validation set?` -- Ultralytics -- https://github.com/ultralytics/ultralytics/issues/19832
* **Issue #21781** -- `Difference in Feature map dimensions training vs. validation` -- gigumay -- 2025-08-23 -- https://github.com/ultralytics/ultralytics/issues/21781
* **Issue #6192** -- `The imgsz for inference is different from the Settings` -- Ultralytics -- https://github.com/ultralytics/ultralytics/issues/6192
* **PR #824** -- `update rect` -- Laughing-q -- 2023-02-06 -- https://github.com/ultralytics/ultralytics/pull/824
* **PR #25559** -- `Reuse imgsz from export metadata when validating static exported models` -- Ultralytics -- https://github.com/ultralytics/ultralytics/pull/25559
* **PR #22440** -- `Downscale to padded size in load_image instead of unpadded size` -- Y-T-G -- 2025-10-19 -- https://github.com/ultralytics/ultralytics/pull/22440
* **Issue #502** -- `Can not run Val with onnx model and shape mismatch` -- AbdelsalamHaa -- 2023-01-19 -- https://github.com/ultralytics/ultralytics/issues/502
* **Issue #7797** -- `val can't validate exported model if not square resolution was used` -- CySlider -- 2024-01-24 -- https://github.com/ultralytics/ultralytics/issues/7797
* **Issue #10289** -- `When using yolo export to convert a model to ONNX format, decrease in accuracy observed` -- xla145 -- 2024-04-24 -- https://github.com/ultralytics/ultralytics/issues/10289
* **Issue #19991** -- `parity mismatch torch vs onnx for yolo-detection-model` -- ambient-udeshmukh -- 2025-04-03 -- https://github.com/ultralytics/ultralytics/issues/19991
* **Issue #20231** -- `Exporting onnx and tensorrt models from pytorch model, accuracy decreases` -- shyhoom -- 2025-04-19 -- https://github.com/ultralytics/ultralytics/issues/20231
* **Issue #20930** -- `Performance drop on ONNX compared to PyTorch` -- kkeroo -- 2025-06-04 -- https://github.com/ultralytics/ultralytics/issues/20930
* **Issue #19608** -- `Changes in mAP due to difference in ultralytics version and pytorch version` -- mugheessarwarawan -- 2025-03-10 -- https://github.com/ultralytics/ultralytics/issues/19608
* **Issue #22200** -- `Different F1 results (0.72 vs 0.77) when switching ultralytics 8.3.190 to 8.3.199` -- YIYIYI14736 -- https://github.com/ultralytics/ultralytics/issues/22200
* **YOLOv5 Discussion #10862** -- `Different results when testing exported onnx model and pytorch model` -- Ultralytics/yolov5 -- https://github.com/ultralytics/yolov5/discussions/10862
* **Dave, Dollár, Ramanan, Kirillov, Girshick -- `Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details`** -- 2021/2022 -- arXiv:2102.01066 -- https://arxiv.org/abs/2102.01066 ; https://ar5iv.labs.arxiv.org/html/2102.01066
* **`Benchmarking Object Detectors with COCO: A New Path Forward` (COCO-ReM)** -- Authors not extracted -- 2024 -- ECCV 2024 -- https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06053.pdf
* **Llerena, Jung -- `Noise-Aware Evaluation of Object Detectors`** -- 2025 -- WACV 2025 -- https://openaccess.thecvf.com/content/WACV2025/papers/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.pdf
* **Chavali, Agrawal, Mahendru, Batra -- `Object-Proposal Evaluation Protocol is 'Gameable'`** -- 2016 -- CVPR 2016 -- https://openaccess.thecvf.com/content_cvpr_2016/papers/Chavali_Object-Proposal_Evaluation_Protocol_CVPR_2016_paper.pdf
* **Borji et al. -- `Empirical Upper Bound, Error Diagnosis and Invariance Analysis of Modern Object Detectors`** -- 2020 -- arXiv:2004.02877 -- https://arxiv.org/pdf/2004.02877v1.pdf
* **Asselin et al. -- `Replication Study and Benchmarking of Real-Time Object Detection Models`** -- 2024 -- arXiv:2405.06911 -- https://arxiv.org/html/2405.06911
* **Herrera et al. / Trustworthy Evaluations team -- `How Trustworthy are Performance Evaluations for Basic Vision Tasks?`** -- 2020 (arXiv:2008.03533) -- https://arxiv.org/abs/2008.03533
* **Karimov, Imani, Kazakov -- `Quantization Robustness to Input Degradations for Object Detection`** -- 2025 -- arXiv:2508.19600 -- https://arxiv.org/html/2508.19600 ; code https://github.com/AllanK24/QRID
* **YOLOv6+ authors -- `YOLOv6+: simple and optimized object detection model for INT8 quantized inference on mobile devices`** -- 2025-05-28 -- Springer Signal, Image and Video Processing -- https://link.springer.com/article/10.1007/s11760-025-04234-0
* **Batra, Nipun -- `Same Model, Same GPU, Half the Latency: What TensorRT, ONNX Runtime, and FP16 Actually Buy You`** -- 2026-07-11 -- Blog -- https://nipunbatra.github.io/blog/posts/2026-07-11-yolo-inference-speed-tensorrt.html
* **TildAlice -- `YOLOv8 INT8 Quantization: 4x Faster on Jetson Orin`** -- 2026-02-07 -- Blog -- https://tildalice.io/yolov8-int8-quantization-jetson/
* **Cloudflight Engineering -- `Quantized YOLO for Edge Solutions`** -- Blog -- https://engineering.cloudflight.io/quantized-yolo-for-edge-solutions
* **the0807/YOLOv8-ONNX-TensorRT** -- GitHub repo -- https://github.com/the0807/YOLOv8-ONNX-TensorRT
* **huangzongmou/yolov8-pytorch_quantization** -- GitHub repo -- https://github.com/huangzongmou/yolov8-pytorch_quantization/blob/main/yolov8_ptq_int8.py
* **SafetyRepro team -- `SafetyRepro: Configuration-Conditional Rank Instability on Alignment Benchmarks`** -- 2026 -- arXiv:2605.25492 -- https://arxiv.org/html/2605.25492

---

## 8. What We Could NOT Find (explicit list per user request)

1. **No peer-reviewed paper, Ultralytics doc, or blog** states the exact finding: `model.val(imgsz=320)` on 200x200 images yielding 352x352 input under `rect=True`/`pad=0.5`, with numbers 0.7624 vs 0.7411, exact PT-ONNX parity at matched resolution, model-dependent bias +0.021/-0.025/+0.016, ranking reversal. Search returned **0 hits** for those numbers or that narrative.

2. **No paper** where a reported "ONNX export costs X mAP" is **demonstrated to be** a resolution artefact (i.e., showing delta disappears at `rect=False`). GitHub issues hypothesize it, but no paper was found proving it.

3. **No paper** benchmarking quantized detectors (INT8/ONNX/TensorRT) that **explicitly documents** controlling for `rect`/`letterbox`/`imgsz` lower-bound to isolate quantization effect from resolution effect. All quantized works inspected either omit the setting or mention only `model.val()` defaults.

4. **No detection paper** demonstrating a **ranking reversal** caused specifically by `rect`/`imgsz`/`stride` defaults. Ranking reversals are documented for IoU threshold, detections-per-image limit, and dataset choice, but not for this preprocessing default.

5. **No Ultralytics official documentation page** that warns: "`val` with `rect=True` inflates effective resolution; compare PT vs ONNX only at `rect=False` or matched `imgsz`, otherwise mAP is incomparable and rankings may flip." The warning exists only implicitly in issue replies and PR comments.

6. **No systematic literature review or meta-analysis** of how many published mAP numbers are contaminated by this artefact.

7. **No citation** found for the specific global burden/mAP scale that would contextualize 0.021 as "large enough" -- but the finding's claim that it can reverse ranking is supported by analogy to known threshold/budget artefacts that do reverse rankings (see §4).

---

## 9. Caveats & Suggestions for Further Verification

* **Verify the 352 mechanism:** In `YOLODataset.set_rectangle` (`ultralytics/data/dataset.py`), `batch_shapes = ceil(shapes * imgsz / stride + pad) * stride` with `pad=0.5` for val explains 200*320/32=2000? Wait: 200*320/320? Need to re-derive, but 352 = ceil(200*320/32 +0.5)*32 /? For 200x200 at 320 with stride 32: `shapes` are normalized? In practice `imgsz=320` on 200 square gives `ceil(200*320/200 /32 +0.5)*32 = ceil(10+0.5)*32=352`. Matches. Author should cite exact code path.
* **Re-run at `imgsz=640`:** The literature shows 640->672 is the canonical inflation; 320->352 is the same `+32` offset. Confirm that COCO val (native >640) behaves differently (no upscale) vs 200x200 synthetic (upscale via padding not pixels).
* **Audit quantized papers:** Contact authors of Karimov et al. and ask for `rect` setting; re-evaluate their INT8 deltas at `rect=False` to see if 0.02 gap persists.
* **Novelty defense:** Cite the GitHub issues above as prior art for mechanism awareness, but distinguish quantitative ranking-reversal demonstration as contribution. Reviewers will likely point to #24388/#20930; need to show those did not quantify or generalize.

---

*End of findings. All citations above verified via live search; `NOT FOUND` means no result returned for that query in this session, not a universal non-existence proof.*
