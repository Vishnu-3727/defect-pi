# Related work for the threshold-decoupling claim

Searched 2026-09-18 across arXiv, IEEE Xplore, ScienceDirect, MDPI, Springer,
USPTO/Google Patents, OpenAlex. Two sources were unavailable and limit coverage:
MDPI was in scheduled maintenance, and the Semantic Scholar API returned HTTP 429
on every attempt. Citations below marked *verified* were fetched and checked
directly; the rest come from the sweep and should be re-checked before use.

## Verdict

The specific claim — **use one detector's score at two simultaneous thresholds,
a low one governing the eject decision and a high one governing what is drawn
and logged, because the actuator and the operator are different consumers with
opposite error costs** — returned **zero hits** in peer-reviewed literature.

Every ingredient is published separately. Nobody assembles them, and nobody
names the single-threshold conflation as a defect.

**But the idea is one sentence from folklore, and the paper must not pretend
otherwise.** A hostile reviewer will call it obvious, and they will have a
point. The defensible contribution is therefore the **measurement and the
framing**, not the idea:

- escape rate 8.8% → 0.6% (and 15.4% → 0.5% on an independent system), zero
  added compute, no retraining;
- the demonstration that no existing metric reports the loss — mAP is swept at
  conf 0.001 and never sees the operating threshold, while the dashboard reports
  rates *after* the cut;
- the observation that the constant is not transferable between models.

## Use the established vocabulary. Do not coin terms.

**escape rate / overkill rate** is canonical: escape = NOK sample passed as OK;
overkill = OK sample rejected as NOK. "Overkill increases production costs,
while escape is a source of customer dissatisfaction."

- *Measurement system escape and overkill rate analysis*, International Journal
  of Advanced Manufacturing Technology, 2011.
  <https://doi.org/10.1007/s00170-011-3342-2> — the formal treatment; cite this
  for the metric.
- AOI vendor literature uses **False Acceptance / False Rejection** for the same
  pair, with "defect escape" as an explicit synonym for FA.

This project's reports already used "escapes" informally; that happens to match
the standard term, and the reports should cite IJAMT 2011 rather than define it.

## What to position against

- **Kassem Sbeyti et al., *Cost-Sensitive Uncertainty-Based Failure Recognition
  for Object Detection*, UAI 2024.** <https://arxiv.org/abs/2404.17427>
  A cost-sensitive framework tailored to user budgets on missed and false
  detections, automating threshold selection on the ROC. *Verified by the sweep
  as optimising a single threshold, not two.*
- **Lee, Hong, Kim, Song, Lee, *Managing Cost–Stability Trade-Offs in Industrial
  Object Detection: A Unified Decision Support Framework*, Algorithms (MDPI)
  19(5):409, 2026.** <https://doi.org/10.3390/a19050409> — the closest competing
  framing. *Full abstract retrieved via OpenAlex 2026-09-19; MDPI itself returns
  403 to automated fetches and there is no open-access PDF, so the full text
  remains unread.*

  Its premise is ours, verbatim: "while mean average precision (mAP) averages the
  performance over all confidence thresholds, real-world deployment demands
  committing to a single operating threshold under score imprecision,
  distribution shifts, and asymmetric — often only approximately known — error
  costs."

  It proposes **DSR-LRP**, a three-phase framework encoding practitioner
  preferences as three parameters (k for asymmetric error cost, αIoU for
  tolerable localisation imprecision, β for expected perturbation), and yields
  "the operating threshold, its DSR-LRP score, and visual evidence". Evaluated on
  blood-cell screening, wildfire smoke, pothole detection and semiconductor
  sensor inspection.

  **Three differences that preserve our contribution:**
  1. It selects **one** threshold. Our structural point — that the display and
     the eject decision are different consumers needing different thresholds —
     is untouched.
  2. Its cost asymmetry parameter `k` is our `R`, but **the abstract makes no
     mention of prevalence or base rate.** Our finding is that prevalence, not
     cost asymmetry alone, dominates the choice — at 1% prevalence the
     break-even ratio is 437, at 50% it is 4.4. A framework parameterised on cost
     alone cannot express that.
  3. Its concern is **robustness under distribution shift** (an LRP-optimal
     threshold degrading 15.6% versus DSR-LRP's 1.8%), not the base-rate bias of
     the benchmark the threshold was selected on.

  Positioning: *DSR-LRP chooses one threshold robustly given costs; we show the
  choice is dominated by a quantity no defect-only benchmark can supply, and that
  the system has two consumers rather than one.*

The honest one-line positioning: *these select one operating point well; we
argue the system has two consumers and therefore needs two.*

## Nearby patterns, none of which is the claim

1. **Three-tier confidence routing** (auto-accept / human review / discard) is
   common practice — e.g. Roboflow's guidance to accept above 0.70, review
   0.40–0.70, ignore below 0.40; and semiconductor ADC patents partitioning into
   high-confidence / marginal / nuisance (US10043264, US9715723). All three tiers
   feed the *same* consumer. The split is by confidence band, not by purpose.
2. **Cascade thresholds** in inspection: a loose first threshold generates
   candidates, a strict second confirms them (US6898305, US8934704). This is the
   **opposite polarity** to our claim — there the strict stage owns the verdict.
3. **Per-class thresholds by severity** — industry practice; split is per-class.
4. **Per-distance thresholds** — US11900706, low threshold for far objects, high
   for near; split is per-spatial-bin.
5. **The exact distinction exists as vendor folklore in another domain.** A
   Garrett metal-detector product page distinguishes the *detection* threshold
   ("what the machine senses") from the *alarm* threshold ("when it alerts
   you"). No citation value, and strong evidence the idea is obvious once named.
   Cite it anyway in a footnote: it is more honest than pretending nobody has
   thought this.

## Established elsewhere, so claim nothing here

- **Image-level verdicts from box output.** max-over-boxes is the de facto
  standard and appears as patent prior art; anomaly detection uses the identical
  convention (max over the anomaly map as the image-level score). Our use of it
  is conventional.
- **mAP is a poor proxy for deployed decisions.** Otani et al., *Optimal
  Correction Cost for Object Detection Evaluation*, CVPR 2022
  (<https://arxiv.org/abs/2203.14438>); Hall et al., *Probabilistic Object
  Detection* (<https://arxiv.org/abs/1811.10800>). Well-trodden — cite, do not
  re-argue.
- **MIL / weak supervision for defect inspection** is mature (Božič, Tabernik &
  Skočaj, *Mixed supervision for surface-defect detection*, Computers in
  Industry 2021, and others). Those concern **training** from image-level
  labels; ours concerns **inference-time threshold policy**. Different problem,
  worth one sentence to distinguish.

## Withdrawn citation

The sweep surfaced a snippet attributing to **Mezher & Marble, arXiv:2305.09407**
a method of "mapping object detection scores to image-level probabilities by
assigning them equal to the lowest probability predicted for any bounding box".
**Checked directly: the paper does not support this.** Its contribution is
out-of-context training for robustness to novel defects. Do not cite it for
aggregation.

## Not found, stated plainly

- No paper naming a *display* threshold and a *decision* threshold as a pair.
- No before/after escape-rate number attributable to a threshold change.
- Nothing in AOI/PCB/semiconductor deep-learning literature framing
  single-threshold conflation as a bug.
