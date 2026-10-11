# 19 — Debugging Playbook

**Prerequisites:** [18 — System architecture](18_system_architecture.md).

---

## 1. Concept

Debugging is not staring at code until the answer appears. It is **binary search
over a chain of assumptions**: find the midpoint of the pipeline, verify whether
the data is correct there, and discard half the system.

Every module in this project ships a `demo()` that asserts its own contract.
That is not test theatre — it is the instrument that makes binary search
possible. When something is wrong, `uv run python tests/test_pipeline.py` tells
you *which half* in about twenty seconds.

---

## 2. Why it is needed

Machine-vision bugs have a property that makes them unusually expensive: **most
of them raise no exception.** The array has the right shape and dtype, so the
program runs happily and produces subtly wrong answers.

You cannot debug those by reading a traceback, because there isn't one. You
debug them by **checking invariants at boundaries**.

---

## 3. Real-world example

A production line ran for three weeks with a vision system that had been
performing well, then reject rates tripled overnight with no software change.
The cause was a fluorescent tube at the end of its life, shifting colour
temperature.

The lesson generalises: **when a vision system degrades without a code change,
suspect the physical world first.** Lighting, focus, camera position, a
fingerprint on the lens. In this project, the equivalent is PatchCore, which
measures distance from a bank recorded under specific lighting.

---

## 4. How it works: the general procedure

1. **Reproduce deterministically.** `--source <folder> --loop` replays the same
   frames indefinitely. A bug you cannot reproduce cannot be fixed.
2. **Run the self-checks.** `uv run python tests/test_pipeline.py`. Eleven
   checks, dependency-ordered — the first failure localises the fault.
3. **Bisect the chain.** Pick the midpoint (usually detections after decode) and
   verify. Correct there means the fault is downstream.
4. **Check the invariant, not the value.** "Is this box plausible?" is weak.
   "Are all coordinates inside the image, and does area match the class median?"
   is a test.
5. **Change one thing.** Two simultaneous changes make attribution impossible.

---

## 5. Mathematical intuition: read the direction of the error

The single most useful diagnostic skill is inferring cause from *which* metric
moved.

| Symptom | Almost always means |
|---|---|
| Precision up, recall down | Model going **silent** — threshold or confidence path |
| Precision down, recall up | Threshold too low, or duplicate boxes surviving NMS |
| Both down | Features broken — preprocessing, colour order, normalisation |
| mAP@50 fine, mAP@50-95 poor | Boxes roughly right, coordinates imprecise |
| Train good, val poor | Overfitting or a leaked split |
| Val good, test poor | Selection overfitting on val |

**This is not theory.** It solved the largest bug in this project. Whole-graph
INT8 quantisation dropped mAP@50 from 0.735 to 0.408, and the instinct was to
suspect the features. But precision *rose* (0.643 → 0.763) while recall
collapsed (0.681 → 0.417). The model was not confused, it was going silent —
which points at the confidence computation, not the backbone. That pointed
directly at the detection head, and excluding its 66 nodes recovered 32 points.

Reading the direction saved days of staring at the backbone.

---

## 6. Software: bugs by module

### `config.py`

| Symptom | Cause | Isolation |
|---|---|---|
| `KeyError` on a key you added | Key not in `_build()` | Every key is named explicitly; add it there |
| Silent wrong behaviour | Typo in YAML key | `_build()` rejects unknown keys — that is why |
| `imgsz must be divisible by 32` | 300 instead of 320 | Stride-32 backbone requirement |

### `preprocess.py`

| Symptom | Cause | Isolation |
|---|---|---|
| Detections slightly off | `ascontiguousarray` before transpose | `assert tensor.flags["C_CONTIGUOUS"]` |
| Accuracy a few points low, no error | BGR not converted to RGB | Print `x[0,0,0]` vs `frame[0,0,2]` |
| Boxes stretched | Resize without letterbox | Check the aspect ratio of the padded image |

**Worked example — the contiguity bug.** An early version placed the copy before
the transpose:

```python
x = np.ascontiguousarray(x)     # copies
x = x.transpose(2, 0, 1)        # ...and immediately makes it a view again
```

Nothing raised. ONNX Runtime accepted it. The only thing that caught it was the
assertion in `demo()`.

### `detector.py`

| Symptom | Cause | Isolation |
|---|---|---|
| Zero detections everywhere | Threshold too high, or a broken decode | Print `confidences.max()` before thresholding |
| Confidence > 1.0 | INT8 head quantisation | Measured 1.28; clip, and exclude the head |
| Boxes offset | `unletterbox_boxes` skipped | `parity.py` — FP32 must match at IoU 1.0 |
| Model loads, output shape wrong | Class count mismatch | Constructor validates rows against config |

**Worked example — the sampling trap.** The detector self-check reported zero
detections and looked like a broken decoder. Raw FP32 and INT8 outputs were
checked and both were fine (9 and 7 candidates above threshold). NMS was tested
directly and returned `[1]`.

The actual cause: `sorted(paths)[:20]` selected only `crazing_*` files —
alphabetically first, and the class with **0.222 recall**. The code was correct;
the sample was pathological. Now it samples four images per class.

**Lesson: when the code looks right, question the data you fed it.**

### `anomaly.py`

| Symptom | Cause | Isolation |
|---|---|---|
| Every part anomalous | Lighting changed since the bank was built | Rebuild the bank |
| Score is `nan` | Negative from the distance expansion | `np.maximum(d2, 0.0)` before `sqrt` |
| Module does nothing | Bank file missing | Logged at startup; check for the warning |
| FPS collapses | Gate not firing, PatchCore on every frame | Check `max_calls_per_second` |

The `nan` case deserves attention. $\|a\|^2 - 2ab + \|b\|^2$ is algebraically
non-negative but numerically a difference of similar magnitudes, giving `-1e-7`
for identical vectors. `sqrt` gives `nan`, and one `nan` propagates through
`min` and silently disables the module.

### `decision.py`

| Symptom | Cause | Isolation |
|---|---|---|
| Everything fails | Threshold too low, or noise boxes | Check `min_defect_area_px` |
| Known defects pass | AND instead of OR | It is OR; verify you did not change it |

### `storage.py`

| Symptom | Cause | Isolation |
|---|---|---|
| `database is locked` | Multi-thread access without the lock | `check_same_thread=False` **requires** the lock |
| Dashboard slow | Missing index | `EXPLAIN QUERY PLAN` should show an index |
| Rows lost on power cut | PASS batching (expected) | FAIL rows commit immediately by design |
| Card fills up | No retention policy | Known gap |

### `actuator.py`

| Symptom | Cause | Isolation |
|---|---|---|
| Solenoid fires at boot | Output enabled before level set | Set level first, then `pinMode`/claim |
| Nothing happens, no error | Fell back to mock | Look for the fallback line in the log |
| First eject ignored | Serial DTR reset | 2 s sleep after opening the port |
| Relay energised at idle | `ACTIVE_LOW` backwards | Test with the load disconnected |
| `gpiochip_open` fails | Wrong chip number | `gpioinfo \| grep GPIO17` |

### `pipeline.py` / dashboard

| Symptom | Cause | Isolation |
|---|---|---|
| `image_path` is `NULL` on FAIL | `relative_to` raised | Caught; falls back to absolute |
| Two of everything | `use_reloader=True` | Two pipelines, one camera |
| Video live, stats frozen | `threaded=False` | Stream monopolises the handler |
| FPS drops when a browser connects | Encoding in the generator | Encode once in the pipeline |

**Worked example — the `relative_to` bug.** Images were being written but
`image_path` recorded as `None`, because `Path.relative_to` raises `ValueError`
when the target is outside the base. Fixed with a fallback to the absolute path,
plus a new assertion in `demo()` that **every FAIL row carries a usable
`image_path`** — the assertion matters more than the fix, because it prevents
regression.

---

## 7. Hardware: bugs the software cannot cause

| Symptom | Check |
|---|---|
| `Undervoltage detected!` | `vcgencmd get_throttled`; bits 0–3 now, 16–19 since boot |
| FPS drops after minutes | `vcgencmd measure_temp` — thermal throttling above 80 °C |
| Camera not found | `rpicam-hello --list-cameras`; ribbon orientation |
| Everything anomalous | Lighting changed; rebuild the bank |
| Blurry images | Focus, or motion blur — shorten exposure and add light |
| Pi unreachable | Wrong SSID band. This Pi cannot see 5 GHz channel 149 under regdom IN |
| Relay chatters | `MIN_INTERVAL_MS` too low, or no flyback diode |

**Worked example — the WiFi failure.** The Pi never joined the network after
first boot. The evidence chain:

1. `custom.toml` was gone and `init=...firstboot` had been stripped from
   `cmdline.txt` — so first boot **did** run and parse the config.
2. `cfg80211.ieee80211_regdom=IN` had been *added* to `cmdline.txt`, which could
   only come from the `[wlan]` section. So the WiFi block was read.
3. `nmcli device wifi list` on the Pi showed `Keybee_Home` on **channel 4**, and
   no sign of `Keybee_Home_5`.

The configured SSID was the 5 GHz one, which the Pi's radio cannot use on
channel 149 under regulatory domain IN. **Each step eliminated a hypothesis
using evidence already on the device**, rather than guessing and re-flashing.

**Worked example — the power question.** `dmesg` showed under-voltage warnings.
Rather than buy a new supply, the actual numbers were read:

```
max_current = 3000          # 5 V/3 A, 15 W ceiling
EXT5V idle  = 4.835 V
EXT5V load  = 4.607 V       # 228 mV sag
get_throttled unchanged across the benchmark
```

Conclusion: it warns, it never throttles, the benchmark is valid, and the sag is
~190 mΩ of cable resistance rather than insufficient capacity. **Measure before
spending.**

---

## 8. Common mistakes in debugging itself

**Changing two things at once.** If it works, you do not know why.

**Trusting a metric without checking the sample.** The zero-detections
investigation was 80% wasted because the sample was all crazing.

**Assuming the code is wrong.** Sometimes the data, the lighting or the cable is
wrong. In this project the cable, the SSID and the sample were each the culprit
at least once.

**Fixing without adding an assertion.** A fix without a check is a bug waiting
to return. Every real bug here gained a `demo()` assertion.

**Reading logs instead of measuring.** `dmesg` said under-voltage; the ADC said
it never throttled. Both were true and only one mattered.

**Debugging on the wrong machine.** x86 numbers do not predict ARM. This cost a
16–25 FPS projection that turned out to be 36.6.

---

## 9. Interview questions

**Q: A model that worked yesterday is failing today. No code changed. What do
you check?**
The physical world first: lighting, camera position, focus, lens cleanliness.
Then whether the input pipeline changed — a camera firmware update altering
default white balance would do it. Only then the model. Vision systems are
coupled to their environment far more tightly than ordinary software.

**Q: mAP dropped after quantisation. How do you find out why?**
Look at the direction. If precision rises while recall falls, the model is going
silent — the confidence path, which means the head. If both fall, features are
damaged, which means the backbone. Here it was the former: 0.643→0.763
precision, 0.681→0.417 recall, and excluding the 66 head nodes recovered 32
points of mAP.

**Q: The system reports zero detections. Walk me through it.**
Confirm the model loads and the output shape matches the class count. Print the
maximum raw confidence before thresholding — if it is high, the bug is in NMS or
coordinate mapping; if it is low, in preprocessing or the model. Then check the
input sample: in this project the "bug" was that the first twenty files
alphabetically were all crazing, the class with 0.222 recall.

**Q: How do you debug something with no exception?**
Assert invariants at boundaries. Contiguity, colour order, coordinate range,
value range. Each module's `demo()` does this, so `tests/test_pipeline.py`
localises a fault to one module in twenty seconds.

**Q: Your Pi is unreachable after a headless first boot. No monitor. What now?**
Put the card back in a laptop. The FAT boot partition is readable from Windows
even though the ext4 root is not. Whether `custom.toml` was consumed and whether
`init=...firstboot` still appears in `cmdline.txt` tells you if first boot ran
and whether the config was parsed — turning a guess into evidence.

---

## 10. Summary

- Debugging is **binary search over assumptions**, not code-reading. Eleven
  `demo()` self-checks make the search fast.
- Most vision bugs **raise nothing**. Assert invariants at boundaries:
  contiguity, colour order, coordinate space, value range.
- **Read the direction of the error.** Precision up with recall down means going
  silent, not confused. That one heuristic solved the INT8 collapse.
- **Question the sample, not just the code.** Alphabetical file order once made
  a working detector look broken.
- **Suspect the physical world** when nothing in software changed. Lighting,
  focus, cable, SSID band, supply voltage.
- **Measure before spending.** The under-voltage warnings were real and the
  throttling never happened.
- Every fix gets an **assertion**, or it will come back.

Next: [20 — Viva questions](20_viva_questions.md).
