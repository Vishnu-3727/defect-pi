# The other half of the trade: overkill, and why prevalence decides everything

`threshold-decoupling.md` measured that lowering the decision threshold from
0.35 to 0.10 cuts the escape rate from 8.8% to 0.6% and recommended it, noting
that the overkill rate could not be measured because NEU-DET contains no
defect-free images.

It can be measured, from defect-free **regions**. Doing so reverses the
recommendation for any realistic production line.

---

## Measuring overkill without clean images

Every NEU-DET image has substantial unannotated area — mean defect coverage is
39.5%, median 32.2%. The largest defect-free rectangle was found per image
(histogram method), and a 100×100 crop taken from its centre, asserted not to
touch any box. A matching 100×100 crop centred on the largest defect box gives
a positive control processed identically.

**Scale confound, found and corrected.** A 100 px crop upscales 3.2× to reach
the 320 px input, against 1.6× for a full 200 px frame, and the detector
false-fires more on magnified texture. Tiling the crop to 200×200 preserves
texture scale exactly. It matters:

| group | 0.35 | 0.25 | 0.15 | 0.10 | 0.05 |
|---|--:|--:|--:|--:|--:|
| clean, raw 100px (3.2× upscale) | 0.426 | 0.596 | 0.681 | 0.809 | 0.872 |
| **clean, tiled to 200 (scale preserved)** | **0.234** | 0.319 | 0.468 | **0.596** | 0.702 |
| defect, raw 100px | 0.845 | 0.967 | 1.000 | 1.000 | 1.000 |
| defect, tiled to 200 | 0.884 | 0.939 | 0.994 | 0.994 | 1.000 |

Defect crops barely move, clean crops nearly halve. All figures below use the
scale-preserved row. n = 47 clean, 181 defect, test split only.

## Both error rates, finally

| threshold | escape rate | overkill rate |
|---|--:|--:|
| **0.35 (deployed)** | 0.088 | **0.234** |
| 0.25 | 0.044 | 0.319 |
| 0.15 | 0.017 | 0.468 |
| **0.10 (proposed)** | **0.006** | **0.596** |
| 0.05 | 0.000 | 0.702 |

Escapes fall 15×. **Overkill rises 2.5×, to 60% of good parts rejected.**

## Prevalence decides, and NEU-DET hides it

Expected cost per part is `p·escape(t)·C_e + (1−p)·overkill(t)·C_o`, where `p`
is defect prevalence and `R = C_e/C_o` is how much worse an escape is than a
false reject. Optimal threshold:

| prevalence | R=1 | R=10 | R=30 | R=100 | R=300 | R=1000 |
|---|--:|--:|--:|--:|--:|--:|
| 0.01 | 0.35 | 0.35 | 0.35 | 0.35 | 0.25 | 0.15 |
| 0.02 | 0.35 | 0.35 | 0.35 | 0.25 | 0.15 | 0.05 |
| 0.05 | 0.35 | 0.35 | 0.35 | 0.25 | 0.10 | 0.05 |
| 0.10 | 0.35 | 0.35 | 0.25 | 0.15 | 0.05 | 0.05 |
| 0.50 | 0.35 | 0.15 | 0.05 | 0.05 | 0.05 | 0.05 |

Break-even ratio above which 0.10 beats 0.35:

| prevalence | 1% | 2% | 5% | 10% | 30% | 50% |
|---|--:|--:|--:|--:|--:|--:|
| **R\*** | **437** | 216 | 84 | 40 | 10 | 4.4 |

**At a realistic line prevalence of 1–5%, dropping to 0.10 pays only if one
escape costs 84 to 437 false rejects.** For steel strip that is implausible —
scrapping or re-inspecting a good section is cheap and immediate.

**The deployed 0.35 is almost certainly correct, and the earlier
recommendation was wrong.**

## It replicates on the independent system

The same measurement on `defectvision` — separate codebase, model trained at 640,
different split, 45 clean crops from 182 test images:

| threshold | escape (dv) | overkill (dv) | escape (di) | overkill (di) |
|---|--:|--:|--:|--:|
| 0.35 | 0.154 | 0.133 | 0.088 | 0.234 |
| 0.10 | 0.005 | 0.444 | 0.006 | 0.596 |

Break-even ratio for 0.10 to beat 0.35:

| prevalence | 1% | 2% | 5% | 10% | 50% |
|---|--:|--:|--:|--:|--:|
| defect-inspect | 437 | 216 | 84 | 40 | 4.4 |
| **defectvision** | **207** | **102** | **40** | 19 | 2.1 |

The absolute numbers differ and the direction does not. `defectvision`'s bar is
about half as high because it starts with nearly twice the escape rate (15.4%
against 8.8%) and so has more to gain — itself a consequence of its model being
trained at 640 and run at 320, which shifts every confidence down and therefore
moves *both* error rates at a fixed threshold.

**At 1–5% prevalence both systems need an escape to cost 40–437 false rejects
before lowering the threshold pays.** The conclusion is a property of the
problem, not of one checkpoint.

## Why the error was structural, not careless

**NEU-DET has 100% defect prevalence.** Every image contains a defect, so the
overkill term is multiplied by `(1 − p) = 0`, and *any* threshold reduction
looks free. The recommendation to lower it was not a misreading of the data —
it is what the data says, because the dataset encodes a production line where
every part is bad.

That generalises beyond this project: **a benchmark with 100% positive
prevalence systematically biases operating-point selection toward recall**, and
nothing in mAP, precision, recall or F1 exposes the bias, since all are computed
conditional on the labelled set rather than on a realistic mix. A paper that
tunes a threshold on NEU-DET, MVTec's defect split, or any defect-only
benchmark and calls the result an operating point is reporting the dataset's
prevalence as if it were the factory's.

This is the more defensible form of the original finding and connects directly
to the cost-sensitive operating-point literature (UAI 2024,
<https://arxiv.org/abs/2404.17427>; Algorithms 19(5):409, 2026), using the
canonical escape/overkill vocabulary (IJAMT 2011,
<https://doi.org/10.1007/s00170-011-3342-2>).

## What survives from the original claim

The **structural** point is untouched: one constant should not set both what is
drawn and what is ejected, because those are different jobs with different
costs. That remains true and unpublished.

What changes is the prescription. The decision threshold should be **derived**
from prevalence and the cost ratio, not assumed low. On this evidence a line at
2% prevalence with R = 30 should run the decision at 0.35 — the same value as
the display — and the two thresholds only diverge once R or prevalence is high
enough to justify it. The contribution is the method for choosing, plus the
measurement of both error rates, not "use 0.10".

## Limits

1. **n = 47 clean crops.** Each is worth ~2 points of rate. The trend across
   thresholds is monotone and steep so the direction is safe, but the exact
   break-even ratios are not.
2. **"Defect-free" means "unannotated".** NEU-DET's annotation of subtle defects
   is documented as incomplete, so some clean crops likely contain real defects
   and the measured overkill is an **upper bound**. The conservative direction —
   but it means 0.234 at the deployed threshold should not be quoted as a true
   false-alarm rate.
3. **Tiled crops are not real parts.** Tiling repeats texture and creates seams.
   50 genuine defect-free images from the production rig would replace this
   entire proxy, and remain the single highest-value missing input.
4. Prevalence and cost ratio for the actual application are unknown. The tables
   above are a method, not an answer; the answer needs two numbers from whoever
   owns the line.
