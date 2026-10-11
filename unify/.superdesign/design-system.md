# Design system — UNIFY-RUL showcase site

## Product context

UNIFY-RUL is an ECE Project-Based-Learning submission at Chennai Institute of
Technology (AY 2026-27). Team: Vishnu Vardhan K S (24EC0251), VS Kamlesh
(24EC0243). Mentor: Mr. Prem Sangeeth.

**One sentence:** one 44,417-parameter model, trained jointly across datasets
with different sensor counts and different time units, exported to ONNX with a
dynamic sensor axis, predicting Remaining Useful Life in 0.42 ms on a CPU-only
Raspberry Pi 5.

**Title claim:** "Universal Remaining Useful Life Prediction — Any Dataset, Any
Hardware."

### Job to be done

A review panel has fifteen minutes and has seen a hundred decks that assert
results. This site has to make a measured claim checkable at a glance and
survive a hostile question. It is a *showcase*, not the live dashboard — that is
a separate page (`edge/static/index.html`) served from the Pi.

### Audience and posture

Primary audience is the **review panel and mentor**, so the page is
**evidence-first**: the numbers and the conditions that produced them lead, and
the limitations are stated on the page rather than waiting to be discovered.
The visual treatment is bold, dark, technical-marketing — but the boldness is in
the typography and layout, never in the claims.

**Non-negotiable content rule:** this project's whole character is that it
reports what it measured and names what it did not. A design that softens or
omits the honest parts is wrong, however good it looks. Specifically, the page
MUST carry:

- The RandomForest baseline **tying/beating** the model on RMSE (14.41 vs 14.51).
- That FEMTO and XJTU are **not downloaded**, so "any dataset" is demonstrated on
  C-MAPSS + synthetic, not on all three.
- That the live rig has **no ground-truth RUL** and the MPU-6050 (1 kHz) cannot
  see the bearing datasets' 25.6 kHz band.
- That the Pi runs the **synthetic build**, not the C-MAPSS one whose accuracy is
  reported.

These are not disclaimers in fine print. They are a section with the same
typographic weight as the results.

## Key sections (in order)

1. **Hero** — the title claim, the four headline numbers (44,417 params · 190 KB ·
   0.42 ms p50 on a Pi 5 · 14.51 RMSE), and one line of provenance.
2. **The problem** — three datasets, three sensor counts (15/16/21), three time
   units (cycles/seconds/minutes). Why one model normally cannot span them.
3. **How it works** — CNN encoder + masked cross-attention over a sensor-id
   embedding, so the sensor axis is data, not architecture. Predicts a normalised
   fraction of life in [0,1], denormalised per dataset by `life_scale`.
4. **Measured results** — the accuracy table, `fig2_pred_vs_true`,
   `fig4_model_vs_baseline`, and the honest reading of the baseline tie.
5. **On the target hardware** — Pi 5 latency table, `fig5_pi_latency`,
   `fig9_footprint`, thermals (46.1 °C, throttled 0x0), and the "one core is
   enough / the tail is flat" observations.
6. **What this is not** — the limitations listed above, given full weight.
7. **Reproduce it** — the five-minute `make smoke` path, no downloads.
8. **Footer** — institution, team, mentor, review date.

## Branding & styling

Inherit the instrument palette from the live dashboard (`edge/static/index.html`)
and push the type scale, not the hue count.

### Color

| Token | Value | Role |
|---|---|---|
| `--bg` | `#0b0b0c` | page ground, one step darker than the dashboard's `#111` for more contrast range |
| `--surface` | `#111113` | card / panel fill |
| `--fg` | `#eeeeee` | primary text |
| `--dim` | `#888888` | labels, captions, secondary values |
| `--line` | `#2a2a2e` | hairline borders — the only separator device |
| `--ok` | `#4ade80` | healthy, pass, the primary accent |
| `--warn` | `#fbbf24` | caution, "measured but qualified" |
| `--bad` | `#f87171` | limitation callouts, failure states |
| `--accent` | `#60a5fa` | attention/metric emphasis, links |

Status colors carry meaning and must not be used decoratively. Green is the
primary accent; blue is for emphasis and links; amber and red belong to the
limitations section. No hue outside this set.

Gradients are permitted **only** in the hero, and only as a subtle
`--bg → --surface` wash or a thin green-to-blue rule. No multi-stop marketing
gradients, no glow, no glassmorphism.

### Type

- **Body, data, tables, labels:** `ui-monospace, "SF Mono", Menlo, Consolas, monospace`.
  Monospace is the project's identity — every number the panel checks should sit
  in a column that lines up.
- **Display headings:** a tight geometric sans is allowed here (this is the one
  place the showcase departs from the dashboard), falling back to
  `system-ui, -apple-system, "Segoe UI", sans-serif`. **No webfont may be
  required** — the page must render correctly with zero network requests.
- Scale: hero `clamp(2.5rem, 6vw, 4.5rem)` / 700, section `1.75rem` / 600,
  metric `2.5rem` / 600 / `line-height 1.05`, body `0.9375rem` / 1.6,
  label `0.6875rem` uppercase `letter-spacing .08em`, caption `0.75rem` dim.
- Numbers use tabular figures (`font-variant-numeric: tabular-nums`) everywhere.

### Spacing, radius, depth

- 4px base scale: 4 / 8 / 12 / 16 / 24 / 32 / 48 / 80.
- Section rhythm: `80px` vertical between major sections, `24px` inside cards.
- Radius: `8px` cards, `4px` chips and meters. Nothing fully rounded.
- **No shadows.** Depth is hairline borders and background steps only — same as
  the dashboard. This is the strongest single rule in the system.
- Content max-width `1100px`, centered, `24px` gutter.

### Layout

- Breakpoint-free where possible: `repeat(auto-fit, minmax(240px, 1fr))` for
  metric rows, mirroring the dashboard's grid.
- Figures sit full-bleed inside their section container with a `--line` border
  and a `--dim` caption below stating what produced them.
- Tables are bordered top and bottom only, `12–13px`, right-aligned numerics.

### Motion

Restrained. A single fade-and-rise on section entry (`opacity` + `translateY(8px)`,
`300ms ease-out`), and the dashboard's `width/background .3s` on any meter.
`prefers-reduced-motion: reduce` must disable both. No parallax, no scroll-jacking,
no counters that animate numbers — an animated measurement reads as decoration.

## Project requirements

- **Self-contained.** One HTML file, inline CSS and JS, no CDN, no build step, no
  webfont. The repo has no `package.json` and the target device may be offline.
- **Real figures.** `reports/figures/*.png` are generated by
  `scripts/make_report_figures.py` from actual runs. Use the uploaded ones; never
  draw a fake chart.
- **Every number on the page must exist in `README.md` or `results/metrics.json`.**
  No rounding up, no invented benchmarks, no "up to". If a number needs a
  condition to be honest (thread count, which build, which split), the condition
  goes next to it.
- Dark only. No light theme, no toggle — matches `color-scheme: dark`.
- Accessible: ≥4.5:1 body contrast, visible focus rings, real heading order,
  figure alt text describing the finding rather than the file name.
