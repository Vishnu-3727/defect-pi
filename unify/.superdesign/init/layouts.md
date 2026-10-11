# Layouts — UNIFY-RUL

**No layout components exist.** There is no app shell, nav bar, sidebar, header,
footer, or router. The one page is a bare document with an `<h1>` and a single
responsive grid — no chrome at all, because it is a wall-mounted status readout
next to a motor rig, not an application someone navigates.

## The only layout: an auto-fit grid

```css
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }
.wide { grid-column:1/-1; }
body { background:var(--bg); color:var(--fg); font:14px/1.5 ui-monospace,Consolas,monospace; margin:0; padding:16px; }
```

`auto-fit` + `minmax(230px,1fr)` is the whole responsive strategy — no
breakpoints are declared anywhere in the project. Cards reflow from four columns
to one by themselves. A design for this project should keep that property: it has
to be legible on a phone held next to the rig and on a projector at a review.

## Page skeleton, verbatim

```html
<h1>UNIFY-RUL &middot; <span id="live">connecting…</span></h1>

<div class="grid">
  <div class="card">      <!-- Health index      : .label + .big + .bar   -->
  <div class="card">      <!-- Remaining useful life : .label + .big + .note -->
  <div class="card">      <!-- Inference latency  : .label + .big + .note  -->
  <div class="card">      <!-- Attention          : .label + table         -->
  <div class="card wide"> <!-- Health over time   : .label + svg sparkline -->
  <div class="card wide"> <!-- Features           : .label + table         -->
</div>

<div id="alarm">⚠ MAINTENANCE ADVISED — health below threshold</div>
<div id="status">—</div>
```

The `<h1>` doubles as the connection indicator (`LIVE` / `SENSOR OFFLINE` /
`SERVER UNREACHABLE`), and `#status` is a single dense line of provenance —
window fill, packet age, gaps, rejected count, model name, model note, uptime.
Both are worth preserving in any redesign: this project's whole posture is that
a number is worthless without the conditions that produced it.
