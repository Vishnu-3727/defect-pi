# SoftPatch-style denoising: no measurable benefit, and a broken calibration metric

The hard-setting experiment (`holdout-ood.md`) moved the open problem from
"should the anomaly backstop exist" to "make it separate": the confidence gate
routes unseen defects correctly (AUROC 0.76–0.99) but PatchCore behind it
separates weakly.

This tried the literature's remedy. **It did not work, and finding that out
exposed a fault in how this project measures separation at all.**

---

## The change

`scripts/build_patchcore.py` gained `denoise(features, frac)`, which drops the
`frac` most isolated patches — by distance to their k-th nearest neighbour in a
random probe subset — *before* coreset selection. Default 0, reproducing the
original bank exactly.

The motivation is sound and is the failure mode SoftPatch (Jiang et al., NeurIPS
2022, <https://arxiv.org/pdf/2403.14233>) names: "the naively attained memory
bank contains abnormal patch features, and assigns a low anomaly score on defect
images". This bank is built from regions outside annotated boxes, and NEU-DET's
boxes are loose, so defect texture leaks in.

Worse, the next step actively selects for the leak: greedy k-centre coreset
picks the **extremes** of the pool, and a leaked defect patch is an extreme. The
two steps pull against each other, which is why the filter must run first.

At `frac=0.10` it dropped 6,348 of 63,476 patches (k-th-NN cutoff 1.807).

## The result depends entirely on which metric you ask

Scored over **534 defect and 135 clean regions** from val+test:

| bank | median normal | median defect | median separation | **AUROC** |
|---|--:|--:|--:|--:|
| baseline | 2.112 | 2.384 | +0.273 | **0.819** |
| denoise 0.10 | 2.079 | 2.450 | **+0.372** | **0.791** |

**Median separation improves by 36%. AUROC gets worse.** They disagree.

AUROC is the metric to believe: it is threshold-free and uses the whole score
distribution, while a difference of medians describes two points and is blind to
the spread that decides false alarms. Denoising widened the median gap while
making the *ranking* no better — consistent with it also lengthening the upper
tail of normal scores, which is exactly what costs false alarms.

And the AUROC difference is not significant either: Hanley–McNeil SE gives
baseline 0.819 ± 0.017 and denoised 0.791 ± 0.019, a difference of 0.028 at
z = 1.10. **No effect in either direction is demonstrated.**

**The baseline bank has been restored.** `denoise()` stays in the script,
defaulted off, documented as showing no measurable benefit here.

## The fault this exposed

`inspector.anomaly.demo()` calibrates `anomaly.score_threshold` by comparing
**median defect against median normal over 46 regions per group**, and prints a
"median separation" figure. That is the number `anomaly.md` quotes as +0.141,
and it is the number that made the whole backstop look weak.

On this evidence that metric is not fit for the decision it drives:

1. It moved the *opposite way* to AUROC on a real change. Trusting it would have
   meant shipping a bank that ranks no better and may rank worse.
2. n = 46 per group cannot resolve differences of this size. The same comparison
   on 669 regions gives a separation of +0.273, nearly double the +0.141 the
   46-region version reports — so even the baseline figure quoted throughout
   this project is an artefact of sample size.

**`anomaly.md`'s "median defect 2.40 vs median normal 2.26, separation +0.141"
should not be quoted again.** Measured properly the baseline separation is
+0.273 and the AUROC is 0.819, which is a considerably less damning picture of
PatchCore than this project has been assuming.

That matters beyond the calibration: "PatchCore separates weakly" has been a
premise in the paper outline, the synthesis and the hard-setting write-up. It
was measured on 46 regions with a metric that disagrees with AUROC.

## What to do instead

Replace the demo's calibration with an AUROC over the largest sample available,
and pick the operating threshold from the score distribution at a stated false
alarm rate rather than from a percentile of 46 normal regions. The crop
machinery in `scripts/threshold_study.py` already produces the regions.

## Limits

- One denoise fraction. A sweep over 0.05/0.10/0.20 might find an effect, though
  with SE 0.017 on AUROC it would need to be large to be visible at this sample
  size.
- 135 clean regions is still modest, and they are "unannotated" rather than
  verified clean — the same upper-bound caveat as `overkill-and-prevalence.md`.
- The self-check reported 200 ms per scoring before and 82 ms after. That is a
  single timing of one call in each run, not a benchmark, and the bank size is
  unchanged at 2000 vectors. No speed claim is made.

---

## C1: the sweep was not run, and here is why

The ledger called for denoise fractions 0.05 and 0.20 to establish a trend.
A power analysis says those runs cannot settle anything.

Hanley–McNeil standard error at the observed AUROC of 0.819, with the 534
defect and 135 clean regions this dataset yields at 100 px crops:

| n clean | n defect | SE(AUROC) | SE(difference) | min detectable effect |
|--:|--:|--:|--:|--:|
| **135** | **534** | **0.0171** | **0.0243** | **0.068** |
| 300 | 1186 | 0.0115 | 0.0163 | 0.046 |
| 600 | 2373 | 0.0081 | 0.0115 | 0.032 |
| 1200 | 4746 | 0.0057 | 0.0081 | 0.023 |

*(min detectable = effect needed for 80% power at 95% confidence.)*

At the sample size available, only an effect larger than **0.068 AUROC** could
be detected — and the observed denoise effect was 0.028, in the wrong
direction. **Every additional denoise fraction carries the same standard
error**, so a sweep adds points without adding resolution.

The binding constraint is **regions, not settings**. Detecting an effect of the
size plausibly on offer needs roughly 1200 clean regions, about nine times what
NEU-DET yields — which is the same shortage of defect-free imagery that blocks
the overkill measurement and PatchCore's memory bank.

**C1 is therefore closed as a stated null**, with the design fault recorded:
the experiment as constructed cannot answer its question, and the fix is more
clean images rather than more parameter values.
