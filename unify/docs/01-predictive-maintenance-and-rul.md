# Section 1 — Predictive Maintenance and RUL

## 1.1 What is maintenance, really?

**Plain language.** Every machine wears out. A bearing loses grease, a turbine blade erodes, a belt stretches. Maintenance is the set of decisions about *when to intervene*. There are only three possible answers: after it breaks, on a fixed calendar, or when the machine itself tells you it is about to break.

**Technical.** Maintenance strategy is a decision policy that trades three costs:

- `C_f` — cost of unplanned failure (downtime, secondary damage, safety, lost production)
- `C_p` — cost of planned intervention (labour, spare part, planned stop)
- `C_w` — cost of wasted remaining life (you replaced a part that had 60 % of its life left)

Reactive minimises `C_p` and `C_w`, pays `C_f` in full.
Preventive minimises `C_f`, pays `C_w` heavily.
Predictive tries to minimise all three by intervening *just before* failure.

### The three strategies

| | Reactive (run-to-failure) | Preventive (time/usage-based) | Predictive (condition-based) |
|---|---|---|---|
| Trigger | Machine stopped working | Calendar or hour-meter | Sensor-derived health estimate |
| Also called | Breakdown, corrective | Scheduled, planned | PdM, CBM, prognostics |
| Data needed | None | Failure statistics (MTBF) | Continuous sensor data + model |
| Typical rule | "Fix it when it dies" | "Change oil every 5000 km" | "Bearing has ~40 h left, schedule Friday" |
| Wasted life | 0 % | 30–60 % typical | 5–15 % |
| Risk of surprise failure | 100 % | Moderate (early failures still happen) | Low, if the model is good |
| Cost of setup | Zero | Low | High (sensors, network, model, skills) |
| When it is *correct* | Cheap, redundant, non-critical parts (a ₹40 light bulb) | Consumables with tight, well-known wear laws (oil, filters) | Expensive, critical, or hard-to-access assets (turbine, wind farm gearbox, aircraft engine) |

**Key insight your faculty will probe:** predictive maintenance is *not* universally better. It is better only when `C_f` is large relative to the cost of instrumentation. Nobody puts a vibration sensor on a ceiling fan. This is why the datasets you chose are aircraft engines and industrial bearings — high `C_f`, high value of an accurate estimate.

### Real numbers worth quoting

- Unplanned downtime in industrial manufacturing is commonly cited at **$50k–$260k per hour** depending on sector; automotive assembly lines sit at the top of that range.
- Preventive-to-predictive migrations are typically reported to cut maintenance cost 15–30 % and unplanned downtime 30–50 %.
- Treat all such figures as vendor-sourced and say so — quoting them as hard science is a classic viva trap.

### An analogy that always lands

- **Reactive** — you drive until the car stops on the highway.
- **Preventive** — you change the timing belt every 100,000 km whether it needs it or not. Sometimes you throw away a perfectly good belt; sometimes it snaps at 92,000 km anyway.
- **Predictive** — the car listens to its own belt, notices the tension signature drifting, and says "you have about 6,000 km left."

RUL is that last number.

---

## 1.2 What is Remaining Useful Life (RUL)?

**Plain language.** RUL is the answer to "how much longer will this thing work before it fails?" — expressed in whatever unit the machine's life is measured in: flight cycles, hours, minutes, revolutions, kilometres.

**Technical.** For a system with a degradation state and a failure threshold, at inspection time `t`:

```
RUL(t) = T_f − t          where T_f = first time the system crosses the failure threshold
```

`T_f` is a random variable, so strictly RUL is a random variable too, and the honest output is a distribution:

```
p(RUL | x_{1:t})          conditional on all sensor history up to now
```

Most deep-learning papers (and your baseline model) collapse this to a point estimate:

```
RUL_hat(t) = E[RUL | x_{1:t}] ≈ f_θ(window of recent sensor data)
```

This collapse is exactly where your slide's **problem P4 (false alarm risk)** comes from, and Section 16 shows how to get the distribution back with quantile regression.

### Units differ per dataset — this matters enormously

| Dataset | Unit of RUL | Typical range |
|---|---|---|
| C-MAPSS | flight cycles (integer) | 0–360, usually capped at 125 |
| FEMTO | seconds (or 10 s samples) | 0–7 h |
| XJTU-SY | minutes (1 sample = 1 min) | 0–~40 h |

A single model that outputs "127" cannot mean cycles for one dataset and minutes for another. Handling this is a core design decision for UNIFY-RUL — see `05-preprocessing.md` §5.6. If you only take one thing from Section 5, take that.

---

## 1.3 RUL vs the rest of the diagnostics family

There is a standard ladder here, formalised in **ISO 13374 / OSA-CBM**. Learn the ladder; it is the cleanest way to answer "how is your work different from anomaly detection?"

```
Layer 1  Data Acquisition        raw volts from the accelerometer
Layer 2  Data Manipulation       filtering, FFT, RMS, kurtosis  ── signal processing
Layer 3  State Detection         "is this normal?"              ── anomaly detection
Layer 4  Health Assessment       "how bad, on a 0–1 scale?"     ── health monitoring
Layer 5  Prognostics             "how long until failure?"      ── ★ RUL — YOUR PROJECT
Layer 6  Advisory Generation     "replace bearing on Friday"    ── your dashboard alert
Layer 7  Presentation            the human-facing screen
```

Your project spans layers 1–7, but its *novelty* sits at layer 5.

| Task | Question answered | Output type | Labels needed | Difficulty |
|---|---|---|---|---|
| **Condition monitoring** | What is the machine doing right now? | Continuous measurements, trends | None | Low — it's instrumentation, not ML |
| **Anomaly detection** | Is this different from normal? | Binary / anomaly score | Only healthy data (unsupervised) | Low–medium |
| **Fault detection** | Is there a fault? | Binary (fault / no fault) | Healthy + faulty examples | Medium |
| **Fault classification** | *Which* fault is it? | Class label (inner race / outer race / cage / imbalance) | Labelled fault types | Medium |
| **Health monitoring / HI** | How degraded is it? | Scalar 1.0 → 0.0 | Often synthesised, sometimes arbitrary | Medium–hard |
| **RUL prognosis** | How much life is left? | Continuous, in time units | **Full run-to-failure histories** | **Hard** |

### Why RUL is strictly harder than fault detection — six reasons

Memorise these; this is the single most likely viva question in the whole project.

1. **Labels are ruinously expensive.** A fault-detection dataset needs a machine with a fault. A RUL dataset needs a machine destroyed from healthy to dead, with every intermediate moment recorded. XJTU-SY has **15** such bearings. Fifteen. That's the whole dataset. Compare with ImageNet's 1.2 million images.
2. **The label is not observable at the time it is needed.** When the sensor reading arrives, nobody knows the true RUL — it is only knowable *retrospectively*, after failure. Fault presence is observable now.
3. **It is regression, not classification.** No softmax to hide behind. The model must produce a calibrated number, and an error of 20 cycles is a different kind of wrong than an error of 2.
4. **The target is non-stationary and non-linear in time.** Degradation is usually flat then suddenly exponential (the "hockey stick"). A model must be accurate in both regimes. Early on, the signal barely moves while the label changes every cycle — the model is being asked to distinguish two nearly identical windows whose labels differ by 50.
5. **Errors are asymmetric.** Predicting failure too late is dangerous (engine fails in flight). Predicting too early is merely wasteful. Symmetric losses like MSE do not encode this — which is exactly why the NASA scoring function exists (`06-model.md` §6.8).
6. **The future depends on unknown future operating conditions.** RUL under gentle load ≠ RUL under heavy load. Any point-estimate model implicitly assumes the future looks like the past.

### The concrete example to use in your presentation

Same bearing, four models, one moment in time:

- **Condition monitor:** "Vibration RMS = 1.8 g, temperature 61 °C."
- **Anomaly detector:** "This is 4.2σ from the healthy baseline. Unusual."
- **Fault classifier:** "Outer-race defect, 88 % confidence."
- **Health monitor:** "Health index 0.34."
- **RUL model (you):** "≈ 47 minutes of useful life remaining. Schedule the swap at the shift change in 30 minutes."

Only the last one lets a plant manager make a *scheduling* decision. That is the business value, and it is why prognostics is the top of the ladder.

---

## 1.4 Why this project chose RUL — mapping to your own problem statements

From your Review-1 slide:

| Your problem | What it really means technically | Where it is solved |
|---|---|---|
| P1 — Dataset dependency | Model input layer is `nn.Linear(21, …)`; feeding 2 sensors is a shape error | Shared per-sensor encoder + set-based attention (`06-model.md`) |
| P2 — Poor generalisation | Trained on one machine's failure physics; features don't transfer | Joint multi-dataset training + sensor embeddings |
| P3 — Limited edge deployment | Models shipped as `.pt` requiring full PyTorch (≈ 800 MB on ARM) | ONNX + ONNX Runtime (`10-onnx-and-deployment.md`) |
| P4 — False alarm risk | Point estimate with no uncertainty; a noisy prediction trips the alarm | **Currently unsolved in your design.** Fix with quantile regression + temporal smoothing (`06-model.md` §6.12) |

Be aware: P4 is listed on your slide but not addressed by anything in your architecture as presented. A sharp examiner will notice. Fixing it costs about 30 lines of code and turns a weakness into your best "future scope" slide.

---

## Summary

- Maintenance strategies: reactive (cheapest to set up, most expensive to suffer), preventive (safe but wasteful), predictive (best when failure cost is high).
- RUL = time remaining until a failure threshold is crossed. Properly a distribution; usually predicted as a point.
- The diagnostics ladder (ISO 13374 / OSA-CBM): acquisition → processing → detection → health → **prognostics** → advisory → presentation.
- RUL is harder than detection because labels require destroying machines, the target is unobservable at inference time, degradation is non-linear, and errors are asymmetric.

## Key takeaways

1. Predictive maintenance is an economic decision, not a technology fad.
2. RUL sits at OSA-CBM layer 5 — above anomaly detection and fault classification.
3. Data scarcity is the defining constraint of prognostics: 15 bearings, 100 engines. This is why generalisation across datasets (your project's core idea) is a genuinely useful contribution.
4. Asymmetric error cost is the reason a special scoring function exists.

## Common interview questions

1. Explain reactive vs preventive vs predictive maintenance with a cost argument.
2. When is predictive maintenance *not* worth doing?
3. Define RUL formally. Why is it a random variable?
4. Difference between diagnostics and prognostics.
5. Why can't you just train a classifier to say "will fail in the next 10 cycles"? (You can — it's called a binary fault-prediction / TTF-window classifier, and it is easier but far less useful for scheduling. Good answer: it discards the magnitude information a planner needs.)
6. What is MTBF and why is it insufficient for a single machine? (It is a fleet-average; a specific unit's remaining life can differ by an order of magnitude.)

## Common viva questions

1. "Why did you pick RUL prediction instead of fault classification?" → Higher business value, harder problem, top of the OSA-CBM ladder, and it is what P1–P3 in the literature are stuck on.
2. "What is the unit of your output?" → Be ready with the multi-dataset unit answer from §1.2. Do not say "cycles" if you also train on bearings.
3. "How do you know the true RUL of your motor?" → You don't. Say so, then explain the honesty layer (`11-realtime-pipeline.md`).
4. "Is RUL supervised or unsupervised?" → Supervised regression, and the supervision is the expensive part.
5. "What happens if the machine is operated differently in future?" → Point estimate assumes stationary future operating conditions; this is a known limitation and a stated future-scope item.

## Practical implementation tips

- Before writing any model code, write a script that prints, for one engine and one bearing, a plot of `RUL vs time` and 3 raw sensor traces. If you cannot explain the shape of those plots, you are not ready to model them.
- Keep a `NOTES.md` with the unit and failure definition of each dataset written out in one line each. You will forget, and mixing them up silently produces plausible-but-wrong results.

## Suggested improvements for this project

1. Add a **health index (HI) auxiliary output** alongside RUL. It costs one extra linear layer, gives the dashboard a 0–1 bar that is meaningful even on your unlabeled motor, and multi-task learning usually improves the RUL head slightly.
2. Address **P4 explicitly** with quantile regression (see §6.12). It is the difference between "we listed four problems and solved three" and "we solved all four."
3. In the report, include the OSA-CBM ladder diagram and mark which layers your system implements. Examiners love a standards reference.
