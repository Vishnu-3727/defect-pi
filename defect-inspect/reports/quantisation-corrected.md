# Correction: there is no quantisation ceiling, and there was no sign flip

Two earlier reports — `quantisation-variance.md` and `quantisation-ceiling.md` —
are **withdrawn**. Both rested on an artefact I introduced. This report records
what is actually true and how the error happened.

---

## What was claimed

That the same INT8 procedure gave **+0.0217 mAP@50 on v1 and −0.0271 on v2**,
that the sign was a property of the trained model, and that INT8 imposed a
ceiling near 0.734 which destroyed 91% of any FP32 advantage.

## What is actually true

Ten FP32 checkpoints quantised identically — same export script, same
calibration seed, same test split, and critically **FP32 ONNX compared against
INT8 ONNX**, one runtime throughout:

| model | FP32 ONNX | INT8 ONNX | Δ |
|---|--:|--:|--:|
| v2_current | 0.7411 | 0.7353 | −0.0058 |
| kaggle_t320 | 0.7411 | 0.7353 | −0.0058 |
| v1_aug | 0.7354 | 0.7341 | −0.0013 |
| kaggle_t640 | 0.7121 | 0.7055 | −0.0066 |
| no_crazing | 0.6824 | 0.6804 | −0.0020 |
| no_rolled | 0.6502 | 0.6387 | −0.0115 |
| no_inclusion | 0.6139 | 0.6044 | −0.0095 |
| no_pitted | 0.5824 | 0.5795 | −0.0029 |
| no_patches | 0.5795 | 0.5642 | −0.0153 |
| no_scratches | 0.5502 | 0.5438 | −0.0064 |

Regression over an FP32 range of 0.550–0.741:

```
INT8 = -0.0230 + 1.0247 * FP32
slope 1.025 +/- 0.019 (1 s.e.)   95% CI [0.987, 1.062]   R^2 0.997
```

- **Slope is 1 within confidence.** Quantisation preserves model ranking.
- **INT8 spread is 100% of FP32 spread.** No compression, no ceiling.
- **All ten deltas are negative**: mean −0.0067, sd 0.0042.

So post-training INT8 quantisation on this architecture costs a **small, roughly
constant 0.007 mAP@50** and otherwise leaves the ordering intact. That is the
boring, useful answer, and it is the opposite of both withdrawn claims.

## How the error happened

The "+0.0217 on v1" compared **v1's PyTorch mAP (0.7107)** against **v1's INT8
ONNX mAP (0.7324)**. Two changes at once — runtime *and* precision — attributed
entirely to precision. Measured within ONNX, v1's quantisation delta is
**−0.0013**, in line with every other model.

The cross-runtime gap is real and large, and it does not even have a consistent
sign:

| | PyTorch | FP32 ONNX | difference |
|---|--:|--:|--:|
| v1 | 0.7107 | 0.7354 | **+0.0247** |
| v2 | 0.7624 | 0.7411 | **−0.0213** |

A likely mechanism is visible in the evaluation logs: the ONNX path reports
**27.0 ms preprocessing per image against 0.8 ms** for the PyTorch path. Those
are different preprocessing implementations — plausibly differing in letterbox
padding, resize interpolation or normalisation — so the two numbers are not
measurements of the same computation. **This is not established**, only
indicated; isolating it would need a direct comparison of the preprocessed
tensors, which has not been run.

## What this invalidates elsewhere

- **`quantisation-variance.md`** — its premise (opposite-signed deltas) is false.
  The calibration-draw sweeps in it remain valid as measurements: within a
  model and a runtime the delta is stable, sd 0.0027–0.0036. Only the framing
  is wrong.
- **`quantisation-ceiling.md`** — withdrawn entirely.
- **`model-v2.md`** — its explanation ("quantisation moves the two models in
  opposite directions") is wrong. The corrected picture: v2's advantage over v1
  is **+0.0517 in PyTorch, +0.0057 in FP32 ONNX, +0.0029 in INT8**. Almost all
  of it disappears at the PyTorch→ONNX boundary, not at quantisation. The
  report's *conclusion* — that the deployed gain is marginal and only mAP@50-95
  is defensible — still holds, for a different reason.

## The lesson worth keeping

**Never attribute a difference to one step when two changed.** Every accuracy
comparison in this project should state its runtime, and comparisons that cross
runtimes should be treated as uninterpretable until the preprocessing is shown
to match. The repository's existing `scripts/parity.py` exists precisely to
check this and reports perfect box agreement at the operating threshold
(121/121, IoU 1.0000) — but it compares at conf 0.35, while mAP sweeps from
0.001, so it never exercised the low-confidence tail where these differences
live.

That gap between what the parity check validates and what the metric measures
is the concrete, reusable finding here, and it is worth a paragraph in the
paper: a parity check at the operating point does not license carrying a
threshold-swept metric across the same boundary.

## Limits

- Ten models, one architecture, one dataset, one quantisation scheme.
- The hold-out models were trained on five classes, so their absolute mAP is not
  comparable to v1/v2. Each still contributes a valid FP32/INT8 *pair*, which is
  all the regression uses.
- The PyTorch↔ONNX mechanism is indicated, not demonstrated.
