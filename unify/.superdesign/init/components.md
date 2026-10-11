# Components — UNIFY-RUL

**There is no component library.** No `package.json`, no npm dependency tree, no
React/Vue/Svelte, no Tailwind, no build step. The entire frontend of this repo is
one hand-written file: `edge/static/index.html` (137 lines, HTML + CSS + vanilla
JS inline).

That is deliberate and load-bearing. The file is served by a FastAPI app on a
Raspberry Pi 5 that may have no internet, so it can have no CDN, no bundler, and
no runtime dependency. Any design produced for this project must be able to ship
the same way: **one self-contained file, no external requests.**

## The "components" that exist

They are CSS classes, not modules. Reproduced verbatim so a design can match them.

### `.card` — the only container primitive
```css
.card { border:1px solid var(--line); border-radius:6px; padding:12px; }
.wide { grid-column:1/-1; }
```

### `.label` — small caps section header, used above every metric
```css
.label { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
```

### `.big` — the metric readout
```css
.big { font-size:32px; font-weight:600; line-height:1.1; }
```

### `.bar` / `.abar` — progress and attention meters
```css
.bar { height:10px; background:#2a2a2a; border-radius:5px; overflow:hidden; margin-top:6px; }
.bar > div { height:100%; background:var(--ok); transition:width .3s,background .3s; }
.abar { height:8px; background:#2a2a2a; border-radius:4px; overflow:hidden; }
.abar > div { height:100%; background:#60a5fa; }
```
The health bar recolors by threshold in JS, and the thresholds are the design's
semantics, not decoration: `<30% bad, <60% warn, else ok`.

### `.note` — the qualifier line under a metric
```css
.note { color:var(--dim); font-size:11px; margin-top:4px; }
```
Used for the honesty captions this project insists on (e.g. "point estimate — no
uncertainty interval").

### Sparkline — inline SVG, no chart library
```html
<svg viewBox="0 0 600 90" preserveAspectRatio="none" height="90"><polyline id="spark"/></svg>
```
```css
svg { width:100%; display:block; }
polyline { fill:none; stroke:var(--ok); stroke-width:1.5; }
```
Plotted on a **fixed 0..1 scale** so the shape carries meaning across reloads —
an autoscaled trace would make every trajectory look identical.

### Alarm banner
```css
#alarm { display:none; margin-top:12px; padding:10px 12px; border-radius:6px;
         background:#3b1214; border:1px solid var(--bad); color:var(--bad); font-weight:600; }
.stale { color:var(--bad); font-weight:600; }
```

### Data table
```css
table { border-collapse:collapse; width:100%; font-size:12px; }
td { padding:2px 6px 2px 0; white-space:nowrap; }
td.v { text-align:right; color:var(--dim); }
td.b { width:99%; }
```
