# Theme — UNIFY-RUL

## Part 1 — Token summary

No Tailwind, no theme provider, no token file. Everything is eight CSS custom
properties in one `:root` block in `edge/static/index.html`, plus a handful of
literals used inline.

### Color

| Token | Value | Role |
|---|---|---|
| `--bg` | `#111` | page background |
| `--fg` | `#eee` | primary text |
| `--dim` | `#888` | labels, secondary values, captions |
| `--line` | `#333` | card borders, hairlines |
| `--ok` | `#4ade80` | healthy state, sparkline stroke, default meter fill |
| `--warn` | `#fbbf24` | health 30–60% |
| `--bad` | `#f87171` | health <30%, stale sensor, alarm |
| *(literal)* | `#60a5fa` | attention bars — the only blue, and the only non-status hue |
| *(literal)* | `#2a2a2a` | empty meter track |
| *(literal)* | `#3b1214` | alarm banner background |

`color-scheme: dark` is declared. **There is no light theme** and no toggle.

The palette is deliberately status-first: green/amber/red carry meaning, blue is
the one decorative-adjacent hue, and everything else is greyscale. A showcase
site should stay inside this set rather than introduce a brand accent.

### Type

| | |
|---|---|
| Family | `ui-monospace, Consolas, monospace` — **everything, including headings** |
| Base | `14px / 1.5` |
| `h1` | `16px / 600` |
| `.big` (metric) | `32px / 600 / line-height 1.1` |
| `.label` | `11px`, uppercase, `letter-spacing .05em` |
| `.note`, `#status` | `11–12px`, `--dim` |
| table | `12px` |

No webfont is loaded — no network request is possible on the target device. Any
design must fall back cleanly to a system monospace stack.

### Spacing, radius, motion

- Spacing: `12px` gap and card padding, `16px` body padding, `6px` label offsets. A 4px-ish scale, informally.
- Radius: `6px` cards and banner, `5px`/`4px` meters.
- Shadows: **none anywhere.** Depth is borders only.
- Motion: one transition — `width .3s, background .3s` on the health bar. Nothing else animates.
- Breakpoints: **none declared.** Responsiveness is `auto-fit minmax(230px,1fr)`.

## Part 2 — Raw source

Full `:root` and stylesheet from `edge/static/index.html`:

```css
:root { color-scheme: dark; --bg:#111; --fg:#eee; --dim:#888; --line:#333;
        --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; }
body { background:var(--bg); color:var(--fg); font:14px/1.5 ui-monospace,Consolas,monospace; margin:0; padding:16px; }
h1 { font-size:16px; margin:0 0 12px; font-weight:600; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }
.card { border:1px solid var(--line); border-radius:6px; padding:12px; }
.wide { grid-column:1/-1; }
.label { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
.big { font-size:32px; font-weight:600; line-height:1.1; }
.bar { height:10px; background:#2a2a2a; border-radius:5px; overflow:hidden; margin-top:6px; }
.bar > div { height:100%; background:var(--ok); transition:width .3s,background .3s; }
svg { width:100%; display:block; }
polyline { fill:none; stroke:var(--ok); stroke-width:1.5; }
table { border-collapse:collapse; width:100%; font-size:12px; }
td { padding:2px 6px 2px 0; white-space:nowrap; }
td.v { text-align:right; color:var(--dim); }
td.b { width:99%; }
.abar { height:8px; background:#2a2a2a; border-radius:4px; overflow:hidden; }
.abar > div { height:100%; background:#60a5fa; }
#status { color:var(--dim); font-size:12px; margin-top:12px; }
.stale { color:var(--bad); font-weight:600; }
#alarm { display:none; margin-top:12px; padding:10px 12px; border-radius:6px;
         background:#3b1214; border:1px solid var(--bad); color:var(--bad); font-weight:600; }
.note { color:var(--dim); font-size:11px; margin-top:4px; }
```

There is no `tailwind.config`, no `globals.css`, and no second theme block in
the repository.
