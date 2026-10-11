# Section 4 — Datasets

Three datasets, deliberately chosen to be as *different* as possible. If they were similar, "universal" would mean nothing.

| | C-MAPSS | FEMTO (PRONOSTIA) | XJTU-SY |
|---|---|---|---|
| Machine | Turbofan aircraft engine (simulated) | Rolling-element bearing (real rig) | Rolling-element bearing (real rig) |
| Origin | NASA Ames, 2008 | FEMTO-ST Institute, France, 2012 | Xi'an Jiaotong Univ. + Changxing Sumyoung, 2018 |
| Data type | Multivariate low-rate sensor readings | High-rate raw vibration waveform | High-rate raw vibration waveform |
| Sensors | 21 (+3 op settings) | 2 accelerometers (+temp on some) | 2 accelerometers |
| Sampling | 1 sample per flight cycle | 25.6 kHz, 0.1 s snapshot every 10 s | 25.6 kHz, 1.28 s snapshot every 1 min |
| Units | flight cycles | seconds | minutes |
| # run-to-failure units | 709 engines across 4 subsets | 17 bearings | 15 bearings |
| Failure definition | end of provided series | vibration amplitude > 20 g | vibration > 10× baseline (horizontal) |
| Physics | thermodynamic degradation | fatigue spalling | fatigue spalling |
| Why included | slow, many sensors, simulated, huge | fast, 2 sensors, real, noisy, tiny | fast, 2 sensors, real, clean, labelled fault type |

That table is your Section-4 slide.

---

## 4.1 NASA C-MAPSS

### What it is

**C-MAPSS** = Commercial Modular Aero-Propulsion System Simulation, a MATLAB/Simulink model of a ~90,000 lb-thrust turbofan engine. NASA ran it thousands of times, each run starting from a slightly different initial wear state and injecting a progressive fault until the engine's health fell below an operating limit. Published as the **PHM08 / Turbofan Engine Degradation Simulation** dataset (NASA Prognostics Data Repository).

**It is simulated, not measured.** Say this openly — an examiner who knows the field will ask. The value is that it is large, clean, fully labelled, and the de-facto benchmark: every RUL paper since 2008 reports on it, so you can compare against literature.

### Structure — four subsets

| Subset | Train units | Test units | Operating conditions | Fault modes | Difficulty |
|---|---|---|---|---|---|
| FD001 | 100 | 100 | 1 (sea level) | 1 (HPC degradation) | easiest |
| FD002 | 260 | 259 | 6 | 1 (HPC) | medium |
| FD003 | 100 | 100 | 1 | 2 (HPC + fan) | medium |
| FD004 | 248 | 249 | 6 | 2 (HPC + fan) | hardest |

HPC = High-Pressure Compressor.

Multiple operating conditions matter: in FD002/FD004 the sensor readings jump around because the engine is at different altitudes/Mach/throttle settings, and that variation is *larger* than the degradation signal. A model that does not handle it learns nothing. This is why per-condition normalisation is standard practice on FD002/FD004 (see `05-preprocessing.md` §5.4).

### File format

Space-separated text, 26 columns, no header:

```
col  1     unit number          (engine id, 1..100)
col  2     time, in cycles      (1, 2, 3, ... until failure)
col  3–5   operational settings 1–3   (altitude, Mach, throttle resolver angle)
col  6–26  sensor measurements 1–21
```

Files per subset: `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`.

### The 21 sensors

| # | Symbol | Description | Unit | Constant in FD001? |
|---|---|---|---|---|
| 1 | T2 | Total temperature at fan inlet | °R | ✔ constant |
| 2 | T24 | Total temperature at LPC outlet | °R | |
| 3 | T30 | Total temperature at HPC outlet | °R | |
| 4 | T50 | Total temperature at LPT outlet | °R | |
| 5 | P2 | Pressure at fan inlet | psia | ✔ constant |
| 6 | P15 | Total pressure in bypass duct | psia | |
| 7 | P30 | Total pressure at HPC outlet | psia | |
| 8 | Nf | Physical fan speed | rpm | |
| 9 | Nc | Physical core speed | rpm | |
| 10 | epr | Engine pressure ratio (P50/P2) | – | ✔ constant |
| 11 | Ps30 | Static pressure at HPC outlet | psia | |
| 12 | phi | Fuel flow / Ps30 | pps/psi | |
| 13 | NRf | Corrected fan speed | rpm | |
| 14 | NRc | Corrected core speed | rpm | |
| 15 | BPR | Bypass ratio | – | |
| 16 | farB | Burner fuel-air ratio | – | ✔ constant |
| 17 | htBleed | Bleed enthalpy | – | |
| 18 | Nf_dmd | Demanded fan speed | rpm | ✔ constant |
| 19 | PCNfR_dmd | Demanded corrected fan speed | rpm | ✔ constant |
| 20 | W31 | HPT coolant bleed | lbm/s | |
| 21 | W32 | LPT coolant bleed | lbm/s | |

In FD001, sensors **1, 5, 10, 16, 18, 19** have zero variance — they carry no information and are dropped by essentially every published method, leaving **14 usable sensors** (2,3,4,7,8,9,11,12,13,14,15,17,20,21). Some authors also drop 6 (nearly constant). Do the variance check in code rather than hard-coding the list — that way FD002–FD004 are handled automatically. And say in your report *why* you dropped them; "I ran a variance filter" is a much better answer than "the paper said 14".

### What RUL means here

- **Train files:** each engine runs to failure. RUL at cycle `t` = `max_cycle_of_that_engine − t`. You compute it yourself.
- **Test files:** each engine's series is **truncated at a random point before failure**. `RUL_FD00X.txt` gives one integer per test engine: the true RUL at its last recorded cycle. You predict for the last window only.

```python
# train label construction — 3 lines, get them right
max_cycle = df.groupby("unit")["cycle"].transform("max")
df["RUL"] = max_cycle - df["cycle"]
df["RUL"] = df["RUL"].clip(upper=125)      # the piecewise-linear cap, see §5.5
```

### Benchmark numbers (so you know if your model is sane)

Published FD001 test-set results, roughly:

| Method | RMSE | NASA score |
|---|---|---|
| MLP (2016 baseline) | ~37 | ~18000 |
| Deep CNN (Li et al. 2018) | 12.6 | 274 |
| CNN-LSTM (Ma & Mao 2021) | ~11.8 | ~230 |
| DAST Transformer (Zhang et al. 2022) | 11.4 | 203 |
| Best published (varies) | ~11–12 | ~190–230 |

If your single-dataset FD001 RMSE lands in **12–16**, you have a working model. If it is under 11, check for a label leak. If it is over 25, something is wrong with preprocessing (usually the cap or the normalisation).

**Expect your multi-dataset model to be slightly worse on FD001 than a FD001-only model.** That is the honest, expected cost of universality, and saying so proactively is a strong viva move: you traded ~1–2 RMSE for the ability to run on three datasets and any sensor count.

---

## 4.2 FEMTO / PRONOSTIA

### What it is

Generated on the **PRONOSTIA** accelerated-life test bench at FEMTO-ST (Besançon, France), released for the **IEEE PHM 2012 Prognostic Data Challenge**. Real bearings, run under overload until they self-destruct — typically in 1 to 7 hours instead of the years they would last normally.

### Rig and conditions

A motor spins a shaft; a pneumatic jack applies a radial load far above the bearing's rated dynamic load. Two accelerometers (horizontal + vertical) sit on the outer race; a PT100 temperature probe is present for some bearings.

| Condition | Speed | Radial load | Bearings |
|---|---|---|---|
| 1 | 1800 rpm | 4000 N | 7 |
| 2 | 1650 rpm | 4200 N | 7 |
| 3 | 1500 rpm | 5000 N | 3 |

Total 17 bearings: **6 in the learning set** (full run-to-failure), **11 in the test set** (truncated).

### Sampling scheme — this is the part people get wrong

```
Every 10 seconds:   record 2560 samples at 25.6 kHz  →  a 0.1 s snapshot
                    (temperature: 600 samples at 10 Hz, i.e. 60 s, when present)

So the data is a SEQUENCE OF SNAPSHOTS, not a continuous waveform.
Timeline:  |■         |■         |■         |■   ...
           0.1s of data, then 9.9 s of nothing, repeat
```

Each snapshot is one CSV: `acc_00001.csv`, `acc_00002.csv`, … Columns: hour, minute, second, microsecond, horizontal acceleration (g), vertical acceleration (g).

**Consequence for your pipeline:** you cannot feed 2560 raw points per sensor into your window — that would make L=2560 and blow up the model. You **extract features per snapshot**, turning each snapshot into a handful of numbers, and then your "timestep" is one snapshot (10 s). See `05-preprocessing.md` §5.3. This is also exactly what your ESP32 does with the MPU-6050, which is not a coincidence — it is what makes the live rig compatible with the bearing datasets.

### Failure definition and RUL

Experiments stop when the vibration amplitude exceeds **20 g** (a safety limit to avoid destroying the rig). So:

```
RUL(snapshot k) = (K_total − k) × 10 seconds      for learning-set bearings
```

For the 11 test bearings the true RUL at truncation is published separately.

Characteristics: **extremely noisy, highly variable lives.** Bearings under identical conditions have lifetimes differing by 5–20×. Some show a long flat period then a sudden spike; others degrade gradually. There is no clean "health index" — this is the dataset that teaches humility. Published RUL errors on FEMTO are often 20–50 %, versus ~10 % on C-MAPSS. Know this, so you are not shocked when your FEMTO numbers look bad. They look bad for everyone.

---

## 4.3 XJTU-SY

### What it is

Released 2018 by Xi'an Jiaotong University's Institute of Design Science and Basic Component, with Changxing Sumyoung Technology. Fifteen LDK UER204 rolling-element bearings run to complete failure. Widely regarded as the cleaner, better-documented cousin of FEMTO.

### Rig and conditions

| Condition | Speed | Radial force | Bearings | Folder |
|---|---|---|---|---|
| 1 | 2100 rpm (35 Hz) | 12 kN | 5 | `35Hz12kN` |
| 2 | 2250 rpm (37.5 Hz) | 11 kN | 5 | `37.5Hz11kN` |
| 3 | 2400 rpm (40 Hz) | 10 kN | 5 | `40Hz10kN` |

Two PCB 352C33 accelerometers, horizontal and vertical, on the housing.

### Sampling scheme

```
Every 1 minute:   record 32768 samples at 25.6 kHz  →  a 1.28 s snapshot
```

One CSV per minute: `1.csv`, `2.csv`, … with columns `Horizontal_vibration_signals`, `Vertical_vibration_signals`. Bearing lives range from ~40 minutes to ~42 hours (bearing 1_3 is short; 2_1 is long).

### What makes it valuable

1. **Complete run-to-failure** — all 15 bearings run until actual failure, not a 20 g cut-off.
2. **Fault type is documented** per bearing (outer race, inner race, cage, ball, or combinations). That enables an auxiliary classification task if you want one.
3. **Longer snapshots** (1.28 s vs 0.1 s) → far better frequency resolution if you do FFT-based features (Δf = 25600/32768 ≈ 0.78 Hz).
4. **Cleaner degradation trends** than FEMTO — a good sanity dataset. If your model can't learn XJTU, it can't learn anything.

### RUL definition

```
RUL(minute k) = (K_total − k) minutes
```

Straightforward, because every bearing ran to real failure.

---

## 4.4 Why these three, specifically

Argue it along four axes — this is the "justification of dataset selection" paragraph your report needs:

1. **Sensor-count diversity.** 14–21 vs 2 vs 2. If your model works on both without architectural change, the "any sensor count" claim is demonstrated, not asserted.
2. **Time-scale diversity.** 1 sample/cycle vs 1 snapshot/10 s vs 1 snapshot/60 s; lives measured in cycles, seconds and minutes. Forces the unit-normalisation design (§5.6) that makes the model genuinely universal.
3. **Domain diversity.** Thermodynamic degradation of a gas turbine vs mechanical fatigue spalling of a bearing. Completely different physics. Shared features across these two domains is a real result.
4. **Simulated vs real.** C-MAPSS is clean and synthetic; FEMTO and XJTU are noisy and physical. Training on both prevents the model from over-fitting to simulator artefacts.

And the practical reason: **all three are free, public, and standard**, so your results are comparable to published literature. Say that too.

### Optional fourth dataset (strong extension)

If you want to strengthen the universality claim with modest effort, add one of:

- **NASA IMS bearing dataset** (Center for Intelligent Maintenance Systems) — 4 bearings, 20 kHz, 3 test-to-failure runs. Same feature pipeline as FEMTO/XJTU, so it costs you one parser.
- **NASA Li-ion battery dataset** — capacity fade, completely different domain, 1–2 "sensors". Would make the universality claim dramatic. Higher risk.
- **MFPT / CWRU bearing** — these are *fault classification* datasets, **not** run-to-failure. Do not use them for RUL. Knowing this distinction is worth a viva mark.

---

## 4.5 Practical: getting and organising the data

```
data/
  raw/
    cmapss/   train_FD001.txt … RUL_FD004.txt
    femto/    Learning_set/Bearing1_1/acc_00001.csv …
              Full_Test_Set/Bearing1_3/…
    xjtu/     35Hz12kN/Bearing1_1/1.csv …
  interim/    per-unit .npz after parsing (signals, rul, sensor_ids)
  processed/  windowed tensors, ready for the DataLoader
```

**Cache aggressively.** FEMTO is ~5 GB of tiny CSVs; XJTU is ~10 GB. Parsing them on every training run wastes hours. Parse once → save `.npz` per unit → the training script loads only `interim/`. Your slide says the download step is done; make the parse-and-cache step done next.

Rough sizes: C-MAPSS ~12 MB (trivial), FEMTO ~2–5 GB, XJTU ~10 GB. Budget SD-card/disk space accordingly — and note the Pi never needs any of this, only the model.

---

## Summary

- C-MAPSS: simulated turbofan, 21 sensors, 1 sample/cycle, 4 subsets of increasing difficulty, the field's benchmark, RUL in cycles, cap at 125.
- FEMTO: real bearings, 2 accelerometers, 0.1 s snapshots every 10 s at 25.6 kHz, ends at 20 g, very noisy, 17 bearings.
- XJTU-SY: real bearings, 2 accelerometers, 1.28 s snapshots every minute, true run-to-failure, fault types labelled, 15 bearings.
- Chosen for diversity in sensor count, time scale, physics, and simulated-vs-real.

## Key takeaways

1. C-MAPSS is per-cycle values; the bearing datasets are raw waveform snapshots. They enter your pipeline through *different front doors* and meet at the common schema.
2. Six C-MAPSS sensors are constant in FD001 — detect that with a variance filter, don't hard-code.
3. FEMTO is genuinely hard; poor numbers there are normal and publishable.
4. CWRU/MFPT are classification datasets, not RUL datasets.

## Common interview questions

1. Is C-MAPSS real or simulated, and why does that matter?
2. What is the difference between FD001 and FD004?
3. How is RUL labelled in the C-MAPSS test set vs the training set?
4. Why can't you feed FEMTO's raw 25.6 kHz signal directly into the network?
5. What is the failure criterion in FEMTO, and how does it differ from XJTU?
6. How many run-to-failure units exist across all three datasets, and what does that imply about overfitting? (≈740, but only 32 bearings — small-data regime, so regularise hard and use unit-level splits.)

## Common viva questions

1. "Why three datasets and not one?" → The universality claim is unfalsifiable with one dataset. Diversity in sensor count, timescale and physics is the experiment.
2. "Your model is worse than the best C-MAPSS paper. Why?" → Those models are FD001-specialists with a hard-coded 14-sensor input. Yours handles 2 and 21 sensors and three domains with one weight set. Quote both numbers and own the trade-off.
3. "How did you split train/test?" → **By unit, never by window.** If windows from the same engine appear in both splits, you have leaked and your RMSE is fiction. Be ready to say this unprompted.
4. "What's the sampling rate of your data versus your MPU-6050?" → 25.6 kHz vs ~1 kHz max. Have the honest answer from `08-hardware.md` ready; this is the sharpest question you will get.
5. "What does one training sample look like?" → "A 30-timestep window over S sensors from one unit, labelled with that unit's RUL at the final timestep."

## Practical implementation tips

- Write one parser per dataset, each returning the common schema. Unit-test each with an `assert` that RUL is monotonically decreasing and ends at 0.
- Verify total counts against the papers: 100 train engines in FD001, 6 FEMTO learning bearings, 15 XJTU bearings. If your loader finds 14 bearings, you have a path bug.
- Plot RMS vs time for every bearing before modelling. You will immediately see which bearings have the classic hockey-stick and which are pathological (XJTU 1_3 fails fast; some FEMTO bearings barely trend at all).
- Fix a random seed and record it. Prognostics results move a lot run-to-run because of the tiny number of units.

## Suggested improvements for this project

1. **Add the IMS bearing dataset** as a fourth source, reusing the FEMTO feature pipeline. Strengthens "Any Dataset" for maybe 80 lines of code.
2. **Leave-one-bearing-out cross-validation** for FEMTO/XJTU instead of a single split. With 15 bearings, a single split is statistically meaningless; LOBO gives you an error bar. This is the single most credible-looking improvement you can make to your results section.
3. **Report per-dataset *and* pooled metrics** in one table, plus a specialist-vs-universal comparison column. That table is your headline result.
4. **Document the fault type** from XJTU in your logs, then check whether the model's attention weights differ by fault type. If they do, that is a genuinely interesting figure for the report.
