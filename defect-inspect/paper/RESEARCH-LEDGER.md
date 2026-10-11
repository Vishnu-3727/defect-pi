# Research ledger

Working queue for the research → audit → implement → test loop. Every item has
an owner-visible exit test. An item leaves the queue only when its test has been
**run and its result committed**, including when the result is negative.

Updated 2026-09-19.

---

## Exit criteria for the loop as a whole

Stop when all of:

1. No **P0** or **P1** item remains actionable without hardware or user input.
2. Every published number in `reports/` states the resolution and runtime it was
   measured at.
3. Every claim in `paper/outline.md` is either backed by a committed experiment
   or explicitly marked as a gap.

Stop early and hand back if: an experiment needs the Pi, clean images, GPU
budget beyond Kaggle's quota, or a decision only the user can make.

---

## P0 — the headline numbers are wrong until these are done

| # | item | exit test | status |
|---|---|---|---|
| A1 | Re-measure every `.pt` figure with `rect=False` | `metrics.md` states an evaluated input size matching the deployed 320 | **CLOSED** — metrics.md now 0.7411 at a true 320 |
| A2 | `scripts/evaluate.py`: pass `rect=False` | report cannot disagree with its own title | **CLOSED** |
| A3 | Same audit on `defectvision` | both projects' numbers comparable | **CLOSED** — 0.744 = NMS 0.45 + rect; deployed config gives 0.7360; 320>640 conclusion survives |
| A4 | Extend `parity.py` to also compare at conf 0.001 | the tail a swept metric integrates over is checked | **CLOSED** — INT8 recall 0.884 at 0.35 vs 0.684 at 0.001 |

## P1 — novelty, in descending value

| # | item | exit test | status |
|---|---|---|---|
| B1 | Validate the prevalence framework on real defect-free images | a true overkill rate, not an upper bound | **CLOSED** — KSDD2: at p=1.00 optimal threshold is 0.01 for every cost ratio; at the real p=0.11 it is 0.35. Training on defect-only data gives a 3x worse escape-overkill AUC |
| B2 | Audit NEU-DET annotation completeness | a number for the mAP any model can reach | **PARTIAL** — 'Bkg is missed annotation' refuted (only 3% of unmatched boxes exceed conf 0.5); crazing's confidence carries near-zero information about annotation agreement (ratio 0.67 vs 0.14-0.34). A true ceiling needs repeated annotation — **needs a person, not a GPU** |
| B3 | Turn the prevalence critique into a method | a procedure + what it recommends vs mAP-tuning | **CLOSED** — `scripts/operating_point.py`; F1 is provably degenerate on defect-only data (precision=1 so F1 is monotone in recall) and costs 3.1x more at p=0.05, R=30 |
| B4 | Re-measure PatchCore separation properly and restate `anomaly.md` | "weak separation" either substantiated or withdrawn | **CLOSED** — AUROC 0.819 over 669 regions; "weak separation" withdrawn |

## P2 — worth doing, not blocking

| # | item | exit test | status |
|---|---|---|---|
| C1 | Denoise sweep | a trend, or a stated null | **CLOSED — stated null.** Underpowered by design: min detectable effect 0.068 AUROC at n=135 clean regions; observed effect 0.028. More fractions carry the same SE. Needs ~1200 clean regions, not more settings |
| C2 | TIDE on `defectvision` | does Cls≈0 hold on an independent model | **CLOSED** — Cls = 0.00 on all six classes. Crazing's Bkg dominance does NOT replicate (rolled-in 28.34 > crazing 19.95) |
| C3 | Re-check the resolution-retrain axis labels | conclusions restated at true resolutions | **CLOSED** — defectvision's 320>640 holds, margin grows +0.0095 -> +0.0135 |

## Blocked — do not attempt

| item | blocked on |
|---|---|
| Any latency or FPS number | Pi is off and under-volting; needs a 27 W supply |
| True overkill rate on *this* product | ~50 defect-free images from the rig |
| Camera / actuator integration | hardware not connected |
| GitHub push | `gh` not installed, no credentials |

---

## Closed this session

| item | result |
|---|---|
| Overkill measurable from defect-free regions | done — reversed the threshold recommendation |
| Hard-setting OOD (6 folds) | done — blind spot real, 74% of unseen defects pass |
| TIDE per-class | done — Cls≈0 everywhere, crazing is Bkg-dominated |
| Quantisation ceiling | **refuted** — slope 1.025, R² 0.997 |
| PyTorch↔ONNX gap | **root-caused** — `rect=True` feeds 352, not 320 |
| SoftPatch denoising | no significant effect; exposed the n=46 calibration fault |
| Crazing partition mechanism | falsified |
