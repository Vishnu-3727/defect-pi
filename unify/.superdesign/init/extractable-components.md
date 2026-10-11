# Extractable components — UNIFY-RUL

Thin by construction: one page, no framework, no component modules. What follows
are CSS-level patterns worth extracting as Superdesign DraftComponents so the
showcase site and the dashboard stay visually one family.

## Layout components

### StatusHeader
- Source: `edge/static/index.html` (`<h1>` + `#live`)
- Category: layout
- Description: Wordmark with an inline connection-state indicator
- Extractable props: `state` (string, one of `LIVE` | `SENSOR OFFLINE` | `SERVER UNREACHABLE` | `connecting…`, default `LIVE`)
- Hardcoded: the `UNIFY-RUL ·` wordmark, monospace stack, 16px/600, `.stale` red styling

### MetricGrid
- Source: `edge/static/index.html` (`.grid`)
- Category: layout
- Description: Breakpoint-free auto-fit card grid, 230px minimum track
- Extractable props: `minTrack` (string, default `230px`), `gap` (string, default `12px`)
- Hardcoded: `repeat(auto-fit, minmax(...,1fr))`, the `.wide` full-row modifier

### ProvenanceBar
- Source: `edge/static/index.html` (`#status`)
- Category: layout
- Description: Single dense line of run conditions under the content — window fill, packet age, gaps, rejected, model, model note, uptime
- Extractable props: `text` (string)
- Hardcoded: 12px, `--dim`, 12px top margin

## Basic components

### MetricCard
- Source: `edge/static/index.html` (`.card` + `.label` + `.big` + `.note`)
- Category: basic
- Description: Bordered card with an uppercase label, a large value, and a qualifier caption
- Extractable props: `label` (string), `value` (string), `note` (string, default empty), `wide` (boolean, default false)
- Hardcoded: 1px `--line` border, 6px radius, 12px padding, 32px/600 value, 11px uppercase label

### ThresholdMeter
- Source: `edge/static/index.html` (`.bar`)
- Category: basic
- Description: Horizontal fill bar that recolors by value — the health readout
- Extractable props: `percent` (number, default 100)
- Hardcoded: `#2a2a2a` track, 10px height, the 30/60 threshold colors (`--bad`/`--warn`/`--ok`), `width/background .3s` transition

### AttentionBar
- Source: `edge/static/index.html` (`.abar`)
- Category: basic
- Description: Slim blue bar in a table row, showing per-channel attention weight
- Extractable props: `percent` (number, default 0)
- Hardcoded: `#60a5fa` fill, `#2a2a2a` track, 8px height

### Sparkline
- Source: `edge/static/index.html` (`svg` + `polyline#spark`)
- Category: basic
- Description: Dependency-free SVG trace on a fixed 0..1 scale
- Extractable props: `points` (string)
- Hardcoded: `viewBox="0 0 600 90"`, `preserveAspectRatio="none"`, `--ok` stroke at 1.5, no fill

### DataTable
- Source: `edge/static/index.html` (`table`, `td.v`, `td.b`)
- Category: basic
- Description: Compact two/three-column readout — name, optional bar, right-aligned dim value
- Extractable props: `rows` (array)
- Hardcoded: 12px, collapsed borders, `white-space:nowrap`, right-aligned `--dim` value column

### AlarmBanner
- Source: `edge/static/index.html` (`#alarm`)
- Category: basic
- Description: Full-width red-on-maroon warning strip
- Extractable props: `visible` (boolean, default false), `text` (string, default `⚠ MAINTENANCE ADVISED — health below threshold`)
- Hardcoded: `#3b1214` background, `--bad` border and text, 6px radius, 600 weight
