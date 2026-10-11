# Routes — UNIFY-RUL

No client-side router. No file-based routing. One static page plus a JSON API,
all served by FastAPI from `edge/app.py`.

| URL | Serves | Notes |
|---|---|---|
| `/` | `edge/static/index.html` | the only page; the live dashboard |
| `/health` | JSON | `stale`, `alarm`, `window`, `last_packet_age_s`, `gaps`, `rejected`, `model`, `model_note`, `uptime_s`, `features` |
| `/metrics` | JSON | `p50_ms`, `p95_ms`, `p99_ms`, `n` — inference latency |
| `/api/recent?n=240` | JSON | `rows[]` (`health`, `rul`, `rul_lo`, `rul_hi`, `feat[]`, `attn[]`) + `features[]` |
| `POST /ingest` | — | sensor node posts feature vectors here |

The page polls all three GET endpoints every 1000 ms with `Promise.all` and
repaints. No websockets, no SSE, no framework.

## Route being designed in this task

**None of the above.** The request is a *new* project showcase site — a public
page that explains what UNIFY-RUL is and reports its measured results. It is not
the live dashboard and does not consume these endpoints. The dashboard is
relevant only as the source of the visual language (see `theme.md`) and as a
thing the site should show a picture of.
