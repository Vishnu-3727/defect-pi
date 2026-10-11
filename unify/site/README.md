# site/ — the demo site

Two pages. Everything is static; there is no server-side code.

| Page | What it is |
|---|---|
| `index.html` | The demo. Runs the **real exported ONNX model in the visitor's browser** over real C-MAPSS test engines. |
| `results.html` | The measured results, figures and limitations. |

## Serve it

The page fetches the model and the engine data, and instantiates a WASM runtime.
Both are blocked under `file://`, so it must be served over HTTP:

```bash
cd site && python -m http.server 8777
# then open http://127.0.0.1:8777/
```

## Rebuild the data

```bash
python scripts/build_demo_site.py          # export engines + copy the model
python scripts/build_demo_site.py --demo   # self-check only, writes nothing
```

The exporter refuses to write a payload it cannot verify. It runs the exported
tensors through onnxruntime in Python and asserts they reproduce the full-precision
prediction to within 0.05 cycles — measured worst case is **0.0018**. If the
per-condition normalisation or the rounding were ever wrong, the build fails
before the page can show a confident wrong number.

## What is real, and what is not

**Real:** the weights (`unify_rul.onnx`, the same 190 KB file the Pi runs), the
inputs (held-out official C-MAPSS test units), the normalisation (the shipped
training statistics, looked up per operating condition), and every prediction —
computed client-side at view time, not replayed.

**Not the Pi:** the latency readout is *this browser on single-threaded WASM*.
It lands around 3 ms. The measured Pi 5 figure is **0.423 ms p50** at 4 threads.
A WASM runtime is not a native one and an x86 laptop is not an ARM proxy — the
page says so where the number appears.

## Two details that are easy to get wrong

1. **Scoring is against the capped label.** C-MAPSS RUL is capped at 125 cycles
   during training (`config/datasets.yaml`, the Heimes 2008 standard), so the
   model *cannot* output more. Comparing against the uncapped label invents a
   50-cycle "error" on every healthy engine. The page scores against
   `true_rul_capped` and shows the uncapped value beside it. The cap is read from
   config, not hardcoded, so retraining at a different cap cannot silently leave
   the page scoring against the old one.

2. **`ort.env.wasm.wasmPaths` must be a fully resolved URL.** The runtime
   dynamically imports `ort-wasm-simd-threaded.mjs`; a bare relative path like
   `"vendor/"` fails module resolution with *"no available backend found"*.

## Offline

`vendor/` holds onnxruntime-web (1.20.1) and the Tailwind play script, so neither
page needs the network. That is deliberate: a demo that dies on a bad venue
connection is a demo that fails in front of an audience. Single-threaded WASM is
also deliberate — multi-threading needs `SharedArrayBuffer`, which needs COOP/COEP
headers that a plain static file server does not send.

`vendor/` is ~12 MB, almost all of it the WASM binary.
