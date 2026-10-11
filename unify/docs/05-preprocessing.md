# Section 5 — Preprocessing

> Rule of thumb in prognostics: **preprocessing decides 70 % of your result, architecture decides 30 %.** Papers publish architectures because preprocessing is unglamorous. Do not be fooled.

## 5.0 Why preprocessing matters at all

Neural networks assume, implicitly, that:

- inputs are roughly zero-mean and unit-variance (otherwise gradients explode or vanish per-feature),
- input size is fixed,
- the target is on a sane scale,
- samples are IID-ish within a batch.

None of that is true of raw prognostics data. Raw C-MAPSS has a temperature at 1400 °R sitting next to a ratio at 0.03 — a 46,000× scale difference. Raw FEMTO is a 25.6 kHz waveform. Raw RUL ranges 0–360 in one dataset and 0–2500 in another. Preprocessing is the adapter layer between physics and gradient descent.

---

## 5.1 The pipeline, in order

Order matters. Doing these in the wrong sequence introduces bugs that never raise an exception.

```
1. Parse            → per-unit arrays in the common schema
2. Sensor selection → drop constant / uninformative channels
3. Feature extraction (BEARINGS ONLY) → 25.6 kHz waveform → per-snapshot features
4. Split by UNIT    → train / val / test, before any statistics are computed
5. Fit normalisation on TRAIN ONLY → save mean/std to disk
6. Apply normalisation to all splits
7. Compute RUL labels + cap
8. Normalise RUL to [0,1] per dataset
9. Sliding window   → (N, S, L) tensors
10. Cache to disk
```

**Step 4 before step 5 is non-negotiable.** Computing normalisation statistics over the full dataset before splitting leaks test information into training. It is the most common silent error in student ML projects, and an examiner who knows ML will ask exactly this.

---

## 5.2 Normalisation

### Plain language

Put every sensor on the same scale so the network judges them fairly. Otherwise the biggest-numbered sensor dominates purely because it is big.

### Z-score (standardisation) — what you use

```
        x − μ
z  =  ─────────        μ = mean of that sensor over the training set
          σ            σ = standard deviation of that sensor over the training set
```

Result: mean 0, std 1. Outliers are preserved (not clipped), which matters — in prognostics, the outliers *are* the degradation.

```python
# fit on train only
mu    = train_signals.mean(axis=0)          # (S,)
sigma = train_signals.std(axis=0) + 1e-8    # (S,) — the epsilon prevents div-by-zero
                                            # on the constant C-MAPSS sensors
z = (signals - mu) / sigma
json.dump({"mean": mu.tolist(), "std": sigma.tolist()}, open("norm_stats.json","w"))
```

### Alternatives, and why not

| Method | Formula | Range | Use when | Why not here |
|---|---|---|---|---|
| **Z-score** | (x−μ)/σ | unbounded | Gaussian-ish, outliers meaningful | ✔ your choice |
| Min-max | (x−min)/(max−min) | [0,1] | Bounded sensors, known limits | A single spike sets `max` and squashes everything else into ~0. Fragile. And test data can exceed the training range. |
| Robust scaler | (x−median)/IQR | unbounded | Heavy outliers to be suppressed | Suppresses exactly the spikes that signal failure |
| Max-abs | x/max\|x\| | [−1,1] | Sparse data | Same fragility as min-max |
| Log then z-score | z(log(1+x)) | unbounded | Exponentially growing features (bearing RMS!) | **Worth trying** for bearing energy features — see improvements |

### Per-condition normalisation (C-MAPSS FD002/FD004)

In FD002 and FD004 the engine runs under 6 different operating conditions. The condition changes sensor readings more than degradation does. Global z-scoring therefore buries the signal in condition-driven variance.

Standard fix: cluster the 3 operational-setting columns into 6 clusters (they are literally 6 discrete points — `KMeans(n_clusters=6)` or just `round()` and `groupby`), then z-score **within each condition**:

```python
cond = KMeans(6).fit_predict(df[["op1","op2","op3"]])
for c in range(6):
    m = cond == c
    df.loc[m, sensors] = (df.loc[m, sensors] - mu[c]) / sigma[c]
```

Reported effect in the literature: FD002 RMSE improves from roughly the high-20s to the mid-teens. It is the single highest-leverage preprocessing trick for C-MAPSS, and worth a slide of its own.

**Analogy:** measuring a person's weight loss while they keep changing clothes. Subtract the clothes (condition) first, then look at the trend.

---

## 5.3 Feature extraction for bearings (the step most students skip)

FEMTO gives you 2560 raw samples per snapshot; XJTU gives 32768. You cannot use those as window timesteps. Convert each snapshot into a small feature vector. This step also makes your MPU-6050 rig compatible with the datasets, because the ESP32 computes the *same* features.

### Time-domain features (compute these on the ESP32 too)

| Feature | Formula | What it detects |
|---|---|---|
| RMS | √(1/N Σx²) | Overall vibration energy — the workhorse trend feature |
| Peak | max\|x\| | Impulsive shocks |
| Peak-to-peak | max−min | Amplitude swing |
| Std dev | σ | Energy excluding DC |
| **Kurtosis** | E[(x−μ)⁴]/σ⁴ | "Spikiness". ≈3 for Gaussian (healthy), rises sharply at early spalling |
| Skewness | E[(x−μ)³]/σ³ | Asymmetry |
| **Crest factor** | Peak/RMS | Early fault indicator; rises then falls as the fault spreads |
| Shape factor | RMS/mean(\|x\|) | Waveform shape |
| Impulse factor | Peak/mean(\|x\|) | Impulsiveness |
| Clearance factor | Peak/(mean√\|x\|)² | Very sensitive early indicator |
| Entropy | −Σp log p over amplitude bins | Signal complexity |

**Kurtosis is the classic early-warning feature for bearings.** Healthy bearing vibration is roughly Gaussian → kurtosis ≈ 3. When a spall forms, each pass over the defect produces an impulse → kurtosis jumps to 5–20. As the defect spreads and the whole surface roughens, the signal becomes Gaussian again and kurtosis *falls back toward 3* while RMS keeps climbing. So kurtosis alone is non-monotonic — you need it **together with** RMS. That two-feature story is an excellent thing to say in viva; it shows you understand the physics, not just the code.

### Frequency-domain features (optional, higher value)

Bearing defects excite specific frequencies determined by geometry and speed:

```
BPFO (outer race)  = (n/2)·f_r·(1 − (d/D)cos φ)
BPFI (inner race)  = (n/2)·f_r·(1 + (d/D)cos φ)
BSF  (ball)        = (D/2d)·f_r·(1 − ((d/D)cos φ)²)
FTF  (cage)        = (1/2)·f_r·(1 − (d/D)cos φ)

n = number of rolling elements, d = ball diameter, D = pitch diameter,
φ = contact angle, f_r = shaft rotation frequency
```

Band energy around these frequencies (and their harmonics) is a physically-motivated feature. Better still is **envelope analysis / Hilbert transform demodulation**, the industry-standard bearing diagnostic: band-pass around the resonance, take the envelope, FFT the envelope, look for BPFO/BPFI peaks.

For a semester project, time-domain features + a few band energies is the right effort/reward point. Mention envelope analysis in future scope — it will impress.

### The resulting shape

```
FEMTO Bearing1_1, 2803 snapshots, 2 accelerometers, 8 features each:
raw:      2803 × 2560 × 2   =  14.3 M numbers
features: 2803 × 16                    ← this is your (T, S) array, S = 16
```

Now a window of L=30 covers 30 snapshots = 300 seconds of real time, and the tensor is a sane size.

**Design decision to record:** do you treat "horizontal RMS" and "vertical RMS" as 2 sensors of 8 features (S=2, richer per-sensor input), or as 16 independent scalar sensors (S=16)? For the shared-encoder architecture, **S=16 scalar channels is the cleaner choice** — every channel is a 1-D time series of length L, exactly like a C-MAPSS sensor, and the sensor embedding tells the model which is which. Uniformity across datasets is the whole point.

---

## 5.4 Sliding window

### Plain language

Slide a fixed-length viewing frame along the time series; each frame position is one training sample, labelled by the RUL at the frame's right edge.

### Parameters

```
L      window length   (how many timesteps the model sees)
stride step between consecutive windows
```

```
T = 200 timesteps, L = 30, stride = 1  →  N = (200 − 30)/1 + 1 = 171 windows
T = 200 timesteps, L = 30, stride = 5  →  N = (200 − 30)/5 + 1 =  35 windows
```

General: `N = floor((T − L)/stride) + 1`

### Choosing L

| L too small | L too large |
|---|---|
| Model sees only noise, no trend | Units shorter than L produce zero windows (C-MAPSS test engines can be 31 cycles!) |
| Cannot distinguish degradation from fluctuation | Slower inference, more memory |
| | Longer warm-up before the Pi can predict |

Standard values: **C-MAPSS L=30** (the field's convention; some use 50 with FD001). Bearings: L=20–50 snapshots. Since your model is sensor-agnostic but **not** length-agnostic, pick one L for all datasets. L=30 is the sane default.

Note the real-time cost: with L=30 and 1 snapshot/second from the ESP32, the Pi needs **30 seconds** after boot before its first valid prediction. Design the dashboard to say "warming up 12/30" rather than showing garbage.

### Choosing stride

- **Training: stride = 1.** Maximum data from a tiny dataset. Yes, adjacent windows are 97 % identical — that's acceptable augmentation here, though it does mean your effective sample count is much lower than N suggests (relevant when you report "18,000 training samples" — be ready to note they come from only 100 engines).
- **Validation/test: stride = 1** for dense evaluation, or evaluate only the final window per unit to mirror the C-MAPSS protocol.
- **Inference on the Pi: stride = 1 at your update rate** — a rolling buffer, one new sample in, one out, predict every second.

```python
from collections import deque
buf = deque(maxlen=L)          # the entire "sliding window" on the Pi is one deque
buf.append(new_feature_vector)
if len(buf) == L:
    x = np.stack(buf, axis=-1)[None]   # (1, S, L)
    rul = session.run(None, {"x": x})[0]
```

### The unit-boundary trap

**Never let a window span two units.** A window containing the last 20 cycles of engine 7 and the first 10 cycles of engine 8 is physically meaningless and will corrupt training. Window each unit separately, then concatenate:

```python
X = np.concatenate([window_one_unit(u) for u in units], axis=0)
```

If your window count comes out suspiciously round, or exactly `T_total − L + 1`, you have made this mistake.

---

## 5.5 RUL capping (piecewise-linear RUL)

### The problem

Raw RUL is linear from the first cycle: engine 1 starts at RUL 191, then 190, 189… But a **healthy engine at cycle 1 looks identical to a healthy engine at cycle 40.** The sensors have not moved. Yet you are asking the model to output 191 for one and 151 for the other from indistinguishable inputs. That is not a learnable function — it is noise, and the model wastes capacity trying to fit it, hurting the accuracy in the region you actually care about (near failure).

### The fix

Cap the label:

```
RUL_capped(t) = min(RUL_true(t), R_max)          R_max = 125 for C-MAPSS (field standard)
```

```
Raw label                             Capped label (piecewise-linear)
RUL                                   RUL
 ▲                                     ▲
 │╲                                    │────────╲
 │ ╲                                   │ 125     ╲
 │  ╲                                  │          ╲
 │   ╲                                 │           ╲
 └────╲──▶ t                           └────────────╲──▶ t
      failure                                     failure
"degrades from cycle 1"               "healthy until it isn't, then degrades"
```

This encodes a physical truth: degradation is not detectable during the healthy phase. R_max ≈ 125–130 is the near-universal choice for C-MAPSS (originating with Heimes 2008); values from 115 to 135 make little difference.

### Effect on results

Uncapped FD001 RMSE typically lands around 25–35. Capped: 12–16. **Half your accuracy comes from this one line of code.** Also: any paper reporting RMSE ~13 on C-MAPSS is using the cap, so if you compare against them without capping, you are comparing different problems.

### For bearings

The same argument applies but the right cap differs — bearing lives vary from 40 min to 42 h, so a fixed cap in minutes is wrong. Two options:

- Cap as a **fraction of total life**: `RUL_norm = min(RUL/T_total, 1.0)` with a knee at e.g. 0.7 of life.
- Or cap in absolute time per operating condition.

Recommended: normalise first (next section), then cap in normalised space. One rule for all datasets.

---

## 5.6 The unit problem — the core design decision of UNIFY-RUL

This is the part not covered by any of your three base papers, and therefore the part most worth getting right and talking about.

### The problem, stated

Your model has one output neuron. What does the number mean?

- C-MAPSS wants flight cycles (0–125).
- FEMTO wants seconds (0–25,000).
- XJTU wants minutes (0–2,500).

Train a single head on all three in native units, and the loss is dominated by whichever dataset has the biggest numbers. FEMTO errors of 500 seconds swamp C-MAPSS errors of 10 cycles. The model becomes a FEMTO model that occasionally guesses about engines.

### Three possible designs

**Option A — normalised RUL (recommended).**

Predict the *fraction of life remaining*, a dimensionless number in [0,1]:

```
y_norm = RUL_capped / R_max_dataset
```

Then at inference, denormalise using the per-dataset scale stored in `norm_stats.json`:

```
RUL_physical = y_norm_hat × R_max_dataset      (in that dataset's native unit)
```

- ✔ One loss scale for all datasets; every dataset contributes equally to the gradient.
- ✔ Output is bounded → you can use a sigmoid and never emit a negative RUL.
- ✔ The health-index interpretation is free: `1 − y_norm` is a 0–1 degradation bar, which is exactly what your dashboard should show for the unlabeled motor.
- ✖ Requires knowing which dataset a sample belongs to at inference in order to report physical units. On the Pi you always know, so this costs nothing.

**Option B — per-dataset heads.** Shared trunk, three output heads, each in its own unit. Works, but adds a dataset-ID input at inference and mildly undermines the "one universal model" story.

**Option C — log-space RUL.** `y = log(1+RUL)` compresses the dynamic range so a single head in native units is trainable. Elegant, but the error metric becomes hard to interpret and the NASA score no longer applies cleanly.

**Take Option A.** State it explicitly in your report as a contribution: *"UNIFY-RUL predicts normalised remaining life, making the objective dimensionless and enabling joint multi-domain training; physical units are recovered per-dataset at inference."* That sentence is publishable-sounding and it is true.

### Reporting metrics

Always report **in physical units per dataset** (denormalise before computing RMSE), because that is what is comparable to literature. Reporting "normalised RMSE 0.09" is meaningless to a reviewer. Compute both, report both, but headline the physical one.

---

## 5.7 Handling variable sensor count in a batch

If one batch contains a C-MAPSS window (S=14) and an XJTU window (S=16), you cannot stack them into one tensor without padding.

```python
S_max = 21
x    = zeros(B, S_max, L)     # padded with zeros
mask = zeros(B, S_max, dtype=bool)
x[b, :S_b, :]  = window_b
mask[b, :S_b]  = True         # True = real sensor, False = padding
```

The mask travels with the batch and is applied inside the attention (padded positions get `−inf` before softmax). Without it, the model attends to zero-padding and learns nonsense that varies with batch composition — a bug that shows up as "my validation loss depends on batch size", which is baffling if you don't know to look for it.

Simplest correct approach for a custom collate function: pad to `S_max` of the batch (not the global max), and return the mask. See `06-model.md` §6.6 for how the mask is consumed.

---

## 5.8 Train/val/test splitting

**Split by unit. Always.**

```python
# CORRECT
train_units, test_units = train_test_split(unit_ids, test_size=0.2, random_state=42)

# CATASTROPHICALLY WRONG — leaks
train_X, test_X = train_test_split(all_windows, test_size=0.2)
```

Windows from the same engine overlap by 97 %. Random window splitting puts near-identical samples in train and test, and your RMSE drops to ~3, which you might mistake for success. It is not success; it is leakage. If your numbers look far better than published state of the art, you have a bug, not a breakthrough.

For C-MAPSS use the provided train/test files. For bearings with 15–17 units, use **leave-one-bearing-out** cross-validation and report mean ± std.

---

## Summary

- Order: parse → select → extract features → **split by unit** → fit stats on train → apply → label → cap → normalise RUL → window → cache.
- Z-score with train-only statistics, saved to disk and shipped to the Pi.
- Per-condition normalisation is essential for C-MAPSS FD002/FD004.
- Bearings need feature extraction (RMS, kurtosis, crest factor…) to reduce 25.6 kHz snapshots to per-snapshot vectors.
- Sliding window L=30, stride 1, never spanning unit boundaries.
- RUL cap at 125 encodes "healthy is undetectable" and roughly halves RMSE.
- Predict **normalised** RUL so all three datasets share one loss scale; denormalise per dataset for reporting.
- Pad + mask to batch different sensor counts together.

## Key takeaways

1. Preprocessing outperforms architecture in this field. Budget your time accordingly.
2. Two lines — the RUL cap and per-condition normalisation — account for most of the achievable accuracy on C-MAPSS.
3. Normalised RUL is the design decision that makes multi-dataset training actually work.
4. Split by unit or your results are fiction.

## Common interview questions

1. Why z-score and not min-max here?
2. What is the RUL cap and why does it improve results? Draw the piecewise-linear label.
3. What is data leakage in a time-series context, and how do you avoid it?
4. Why can't you compute normalisation statistics on the fly at inference?
5. How do you batch sequences with different numbers of channels?
6. Why is kurtosis non-monotonic over a bearing's life?

## Common viva questions

1. "How do you handle three different time units?" → Normalised RUL, Option A. Have the one-sentence version memorised.
2. "Show me the line where you save the normalisation statistics." → Have it ready; if it does not exist, you have a bug waiting.
3. "Why L=30?" → Convention on C-MAPSS, short enough that test engines with 31 cycles still yield a window, and 30 s warm-up on the Pi is acceptable. Say you benchmarked 20/30/50 — then actually benchmark them.
4. "What if a sensor fails during operation?" → With the mask, you can drop that sensor at inference without retraining. This is a genuine selling point of your architecture — say it loudly.
5. "Did you use the operational settings columns?" → Yes, for per-condition normalisation on FD002/FD004; not as direct model inputs (or as 3 extra channels — decide, and justify).

## Practical implementation tips

- Write `preprocess.py` so it emits a `manifest.json` recording L, stride, cap, sensor list, split seed and dataset versions. Six months later this is the difference between reproducible and lost.
- Assert invariants after preprocessing: `rul.max() <= cap`, `abs(z.mean()) < 0.1`, `0.9 < z.std() < 1.1`, `not np.isnan(X).any()`. Cheap asserts catch expensive bugs.
- Cache windowed tensors as `.npy` with `np.save`; loading is instant compared to re-windowing.
- Plot 3 random windows per dataset as line charts and eyeball them before training. If they look flat or spiky-in-the-wrong-way, fix that before touching the model.

## Suggested improvements for this project

1. **Log-transform bearing energy features** before z-scoring (`log1p(RMS)`). Bearing vibration grows roughly exponentially; log makes the trend linear and easier for a small CNN to fit. Cheap experiment, often a real gain.
2. **Add exponentially-weighted moving average smoothing** of features with a short span (e.g. α=0.3) as *extra channels* alongside the raw ones. Gives the model an explicit trend signal.
3. **Try per-condition normalisation on the bearing datasets too** (3 speed/load conditions each) — the same argument applies and almost nobody does it.
4. **Add envelope-spectrum band energies** at BPFO/BPFI as extra sensor channels for the bearing datasets. Physics-informed features usually beat learned ones in small-data regimes, and it makes your project sound like mechanical engineering rather than just curve fitting.
5. **Ablate every preprocessing step** (no cap / no per-condition norm / no feature extraction) and put the resulting table in your report. An ablation table is the fastest way to look like a researcher.
