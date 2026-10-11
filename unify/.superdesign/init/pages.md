# Pages — UNIFY-RUL

One page exists. There are no imports to trace: the file inlines its own CSS and
JS and pulls nothing from the filesystem or a CDN.

## `/` — Live dashboard
Entry: `edge/static/index.html`
Dependencies: **none** (self-contained: inline `<style>`, inline `<script>`, no `<link>`, no `<script src>`)

Served by:
- `edge/app.py` — FastAPI; mounts the static file and owns `/health`, `/metrics`, `/api/recent`, `POST /ingest`
  - `edge/inference.py` — ONNX Runtime session, EWMA + N-consecutive alarm hysteresis, latency ring buffer
  - `edge/artifacts/unify_rul.onnx` — 190 KB fp32 model
  - `edge/artifacts/norm_stats.json` — per-dataset z-score tables
  - `edge/artifacts/sensor_vocab.json` — sensor-id embedding vocabulary
  - `config/live_map.*.json` — maps live channels onto trained sensor ids

Renders six cards: Health index, Remaining useful life, Inference latency,
Attention, Health over time (sparkline), Features — plus an alarm banner and a
one-line provenance footer.

## Context selection for THIS task

The task is a **new showcase site**, not a redesign of the page above. The
context bundle that matters is therefore not a dependency tree but the project's
substance and visual language:

- `.superdesign/init/theme.md` — the palette and monospace type it must inherit
- `.superdesign/init/components.md` — card, label, metric, meter patterns
- `README.md` — every measured number the site is allowed to claim
- `reports/figures/*.png` — nine generated charts available as real imagery

Do **not** pass `edge/app.py` or the ONNX artifacts as design context; they carry
no UI information.
