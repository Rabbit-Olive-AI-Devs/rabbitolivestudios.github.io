# Fix Yellow Legibility on `/color/weather`

**Date:** 2026-05-18
**Target version:** 3.11.3
**Display:** reTerminal E1002 (Spectra 6 color, 800×480)

## Problem

On the E1002 Spectra 6 color display, `/color/weather` renders several elements
in yellow against the white page background. Yellow has insufficient contrast on
white, so yellow text and thin yellow strokes are effectively unreadable on the
e-ink panel.

The Spectra 6 palette is black, white, red, yellow, green, blue. Of these, only
**black, red, green, and blue** are legible as foreground (text / thin strokes)
on a white background. Yellow is usable only as a background fill or inside large
dithered image regions — never as foreground on white.

## Affected elements (`src/pages/color-weather.ts`)

1. **`tempColor()`** — returns `var(--s6-yellow)` for the 77–95°F warm tier. Used
   as text color in 4 places: current temp, feels-like temp, 5-day forecast
   highs/lows, hourly temps.
2. **`batteryIcon()`** — fills the battery level rect with yellow when
   `level <= 50`.
3. **Moon icon** — `moonPhaseHTML()` is called with a yellow lit surface
   (`rgb(239,222,68)`).

## Out of scope (intentionally unchanged)

- **Alert banner** — non-severe alerts use a yellow *background* with black text;
  readable, untouched.
- **Headlines `regulatory` badge** — yellow *background* with black text;
  readable, untouched.
- **AI image pipelines** (color moment, skyline) — yellow there is part of the
  dithered artwork.
- **Mono `/weather` page** — unaffected.

## Design

### 1. Temperature color tiers: 4-tier → 3-tier

`tempColor(tempC)` becomes:

| Range (°F) | Range (°C)   | Color              | Meaning              |
|------------|--------------|--------------------|----------------------|
| ≤ 32       | ≤ 0          | `var(--s6-blue)`   | cold / freezing      |
| 33–86      | 0 < t ≤ 30   | `#000` (black)     | comfortable — neutral |
| > 86       | > 30         | `var(--s6-red)`    | hot                  |

Concept: only the temperature extremes get a warning color; the comfortable
middle stays plain black, which is the most readable foreground color on white.
The previous green tier and yellow tier are both removed from temperature
coloring. Green remains in use elsewhere (battery fill, headlines `company`
badge) — only its use as a temperature color is dropped.

All 4 call sites inherit the change with no edits — they all call `tempColor()`.

### 2. Battery fill: 3-tier → 2-tier

`batteryIcon()` fill color:

| Level   | Color |
|---------|-------|
| ≤ 20%   | red   |
| > 20%   | green |

The yellow `≤ 50%` branch is removed. A battery indicator's purpose is "is it
low" — red/green covers that. No Spectra color works as a sensible mid-tier on
white.

### 3. Moon icon lit surface: yellow → white

The `moonPhaseHTML()` call in `renderHTML()` changes the lit color from
`rgb(239,222,68)` to `#fff`. The moon disc keeps its black outline and black
shadow path, so the phase shape stays crisp at every phase. This matches the
mono E1001 treatment (white lit / black shadow). Supersedes the "Color: yellow
lit" detail of DECISIONS.md #34.

## Versioning & cache

- Version bump **3.11.2 → 3.11.3** in `package.json` and the `VERSION` constant
  in `src/index.ts`.
- **No cache key change** — `/color/weather` is a live HTML page with no image
  cache.

## Documentation updates (same commit)

- **`DECISIONS.md`** — new entry: the yellow-on-white legibility rule; records the
  3-tier temp scheme, 2-tier battery, and white moon; notes that it supersedes
  the "Color: yellow lit" detail of #34.
- **`README.md`** — update if temp-color tiers or moon color are documented there
  (verify during implementation).
- **`MEMORY.md`** (Claude auto-memory) — update the DECISIONS #34 moon reference
  if present.

## Testing (browser at 800×480)

- `/color/weather?test-temp=-5` → current temp text **blue**.
- `/color/weather?test-temp=0` → current temp text **black** (0°C boundary).
- `/color/weather?test-temp=15` → current temp text **black**.
- `/color/weather?test-temp=30` → **black**; `?test-temp=31` → **red**
  (verify the 30°C boundary).
- Note: `?test-temp` overrides only `weather.current.temp_c` (the big current
  temp). Feels-like, 5-day, and hourly temps come from the live API — verify
  those visually against whatever the API returns, and by code review that they
  call the same `tempColor()`.
- `/color/weather?test-device` → battery green at 73%. `test-device` injects a
  fixed 73%, so the low-battery red state must be verified by code review or by
  temporarily lowering the injected value.
- `/color/weather?test-moon=0` … `?test-moon=7` → all 8 moon phases legible with
  white lit surface + black outline.
- `npm run typecheck`, `npm run test:utils`, `npm run dry-run` all pass.

## Done checklist

- [ ] `tempColor()` returns blue / black / red on the new thresholds; no yellow,
      no green.
- [ ] `batteryIcon()` fill is red ≤ 20%, green > 20%; no yellow.
- [ ] Moon lit surface is white; shape reads at all 8 phases.
- [ ] Version is 3.11.3 in `package.json` and `src/index.ts`.
- [ ] `DECISIONS.md` (and `README.md` / `MEMORY.md` as needed) updated in the
      same commit.
- [ ] Build verification commands pass.
- [ ] Visually verified at 800×480 for the temp boundaries, battery, and moon.
