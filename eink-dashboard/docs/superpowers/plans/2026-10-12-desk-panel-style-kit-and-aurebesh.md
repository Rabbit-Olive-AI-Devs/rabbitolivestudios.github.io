# Desk Panel: Style Kit + Aurebesh Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the new desk E1001 a Worker-drawn, pure 1-bit "targeting HUD" style kit, a calibration card at `/desk/test`, and the Aurebesh word-of-the-day page at `/desk/aurebesh`.

**Architecture:** The Worker draws each page into an 800x480 1-bit pixel buffer with its own primitives (lines, rectangles, dots, bitmap text, stroke text), encodes it with the existing `encodePNG1Bit`, and serves one HTML document with the PNG inlined as a `data:` URI. Pages are pure functions of the Chicago date: no KV, no AI, no cron.

**Tech Stack:** TypeScript on Cloudflare Workers, no new dependencies. Tests are plain JS under `node --test` against the `tsc` build in `/tmp/eink-dashboard-tests`.

**Spec:** `docs/superpowers/specs/2026-10-12-desk-panel-design.md` (sections 5–8). Read it and `CLAUDE.md` before starting.

## Global Constraints

- Output is exactly **800x480**, **pure black and white**: buffer values are `0` = black, `1` = white, as `encodePNG1Bit` expects. No grey anywhere.
- **One request per panel page**: the PNG is inlined as a `data:` URI (DECISIONS #65). No `<img src="/...">` on a panel page.
- **No JavaScript, no emoji** in any HTML returned to the panel.
- **No KV reads or writes, no AI calls, no cron changes** in this plan. Do not touch `wrangler.toml`.
- **No new npm dependencies.**
- The repo is **public**: commit no font files, no film artwork, no personal data.
- Do not modify any existing image pipeline (`image.ts`, `skyline-image.ts`, `convert-1bit.ts`, `pages/color-moment.ts`). New code lives in `src/hud/` and `src/pages/desk.ts`.
- All commands run from `eink-dashboard/`. Before every commit: `npm run typecheck && npm run test:utils`.
- Commit messages: short summary, blank line, body explaining why, ending with the attribution trailer the session provides. Push after committing. Never force-push.
- **Version bump and deploy need the owner's explicit approval** (Tasks 5 and 9).

## Review Focus

1. **Drawing outside the canvas** (negative or oversized coordinates, a word wider than the panel): must clip silently, never throw and never wrap onto another row. Tested in Task 1.
2. **`?word=` with lowercase, digits, symbols, emoji, an empty value or 40 letters**: must reduce to A–Z, cap at 9 letters, and fall back to the daily word when nothing usable is left. Tested in Task 8.
3. **`?date=` that is malformed or impossible** (`2026-13-45`, `yesterday`): must fall back to today's Chicago date, not produce `NaN` or crash. Tested in Task 7.
4. **Leap day and year boundaries** (Dec 31, Feb 29): the word picker must return a valid word for day 366. Tested in Task 7.
5. **A draw function that throws**: the panel must get the plain text page, never a 500 or a broken-image glyph. Tested in Task 4.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/hud/canvas.ts` (new) | 1-bit buffer and primitives. Knows nothing about the look |
| `src/hud/stroke.ts` (new) | Stroke-font engine (glyphs as polylines on a grid) and the Latin A–Z/0–9 stroke font |
| `src/hud/style.ts` (new) | The tunable constants chosen from the calibration card |
| `src/hud/parts.ts` (new) | Reusable pieces of the look: frame, brackets, crosshair, ruler, hatch, dot fill, title bar, accent block, big number |
| `src/hud/page.ts` (new) | Encode a canvas, wrap it in the one-request HTML page, turn a failure into the text page |
| `src/hud/test-card.ts` (new) | Draws the calibration card |
| `src/hud/aurebesh.ts` (new) | Aurebesh stroke font and the 26-letter check sheet |
| `src/hud/aurebesh-words.ts` (new) | Word list, input cleaning, date-to-word picker |
| `src/hud/aurebesh-page.ts` (new) | Draws the Aurebesh page |
| `src/pages/desk.ts` (new) | Route handlers for `/desk/*` |
| `src/index.ts` (modify) | Two imports, two `case` lines, the 404 endpoint list |
| `tests/hud.test.js` (new) | All tests for the above |

---

### Task 1: Canvas primitives

**Files:**
- Create: `src/hud/canvas.ts`
- Test: `tests/hud.test.js`

**Interfaces:**
- Consumes: `drawText(buf, bufWidth, bufHeight, x, y, text, scale)` and `measureText(text, scale)` from `src/font.ts`.
- Produces:
  - `PANEL_W = 800`, `PANEL_H = 480`
  - `interface Canvas { w: number; h: number; px: Uint8Array }`
  - `createCanvas(w?, h?): Canvas` (all white)
  - `setPixel(c, x, y, v = 0): void`
  - `fillRect(c, x, y, w, h, v = 0): void`
  - `hline(c, x, y, len, t = 1): void`, `vline(c, x, y, len, t = 1): void`
  - `strokeRect(c, x, y, w, h, t = 1): void`
  - `invertRect(c, x, y, w, h): void`
  - `line(c, x0, y0, x1, y1, t = 1): void`
  - `circle(c, cx, cy, r, t = 1): void`
  - `text(c, x, y, s, scale = 1): void`, `textWidth(s, scale = 1): number`
  - `countBlack(c): number`

- [ ] **Step 1: Write the failing tests**

Create `tests/hud.test.js`:

```js
const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const cv = fromBuild("src/hud/canvas.js");

const at = (c, x, y) => c.px[y * c.w + x];

test("canvas starts all white at panel size", () => {
  const c = cv.createCanvas();
  assert.equal(c.w, 800);
  assert.equal(c.h, 480);
  assert.equal(c.px.length, 800 * 480);
  assert.equal(cv.countBlack(c), 0);
});

test("fillRect paints exactly its area", () => {
  const c = cv.createCanvas(20, 10);
  cv.fillRect(c, 2, 3, 4, 5);
  assert.equal(cv.countBlack(c), 20);
  assert.equal(at(c, 2, 3), 0);
  assert.equal(at(c, 5, 7), 0);
  assert.equal(at(c, 6, 3), 1);
  assert.equal(at(c, 2, 8), 1);
});

test("drawing outside the canvas clips and never wraps to another row", () => {
  const c = cv.createCanvas(10, 10);
  cv.fillRect(c, -5, -5, 8, 8);          // only the 3x3 corner is inside
  assert.equal(cv.countBlack(c), 9);
  cv.fillRect(c, 8, 4, 50, 1);            // runs off the right edge
  assert.equal(at(c, 9, 4), 0);
  assert.equal(at(c, 0, 5), 1, "must not wrap onto the next row");
  assert.doesNotThrow(() => cv.line(c, -100, -100, 500, 500, 3));
  assert.doesNotThrow(() => cv.circle(c, 5, 5, 400, 2));
  assert.doesNotThrow(() => cv.text(c, -20, 9999, "OUT", 3));
});

test("strokeRect draws a hollow box of the given thickness", () => {
  const c = cv.createCanvas(20, 20);
  cv.strokeRect(c, 0, 0, 10, 10, 2);
  assert.equal(at(c, 0, 0), 0);
  assert.equal(at(c, 9, 9), 0);
  assert.equal(at(c, 1, 5), 0);
  assert.equal(at(c, 2, 5), 1, "inside stays white");
  assert.equal(cv.countBlack(c), 10 * 10 - 6 * 6);
});

test("invertRect flips black and white inside the area only", () => {
  const c = cv.createCanvas(10, 10);
  cv.fillRect(c, 0, 0, 5, 10);
  cv.invertRect(c, 0, 0, 10, 10);
  assert.equal(at(c, 0, 0), 1);
  assert.equal(at(c, 9, 9), 0);
  assert.equal(cv.countBlack(c), 50);
});

test("line joins its endpoints and thickness widens it", () => {
  const c = cv.createCanvas(20, 20);
  cv.line(c, 2, 2, 12, 2);
  assert.equal(cv.countBlack(c), 11);
  const d = cv.createCanvas(20, 20);
  cv.line(d, 2, 5, 12, 5, 3);
  assert.equal(at(d, 7, 4), 0);
  assert.equal(at(d, 7, 6), 0);
  const e = cv.createCanvas(20, 20);
  cv.line(e, 0, 0, 9, 9);
  for (let i = 0; i < 10; i++) assert.equal(at(e, i, i), 0);
});

test("circle is symmetric and hollow", () => {
  const c = cv.createCanvas(41, 41);
  cv.circle(c, 20, 20, 10);
  assert.equal(at(c, 30, 20), 0);
  assert.equal(at(c, 10, 20), 0);
  assert.equal(at(c, 20, 30), 0);
  assert.equal(at(c, 20, 10), 0);
  assert.equal(at(c, 20, 20), 1);
});

test("text draws black pixels and textWidth matches the font metrics", () => {
  const c = cv.createCanvas(100, 20);
  cv.text(c, 0, 0, "A", 2);
  assert.ok(cv.countBlack(c) > 0);
  assert.equal(cv.textWidth("AB", 2), 2 * 16 + 2);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm run test:utils 2>&1 | tail -20`
Expected: FAIL, `Cannot find module '/tmp/eink-dashboard-tests/src/hud/canvas.js'`

- [ ] **Step 3: Write the implementation**

Create `src/hud/canvas.ts`:

```ts
/**
 * 1-bit drawing surface for the desk panel pages (DECISIONS #66).
 *
 * Values match encodePNG1Bit: 0 = black, 1 = white. Every primitive clips to
 * the buffer, so callers never need to bounds-check and a miscalculated layout
 * degrades to a cut-off shape instead of an exception or a wrapped row.
 */

import { drawText, measureText } from "../font";

export const PANEL_W = 800;
export const PANEL_H = 480;

export interface Canvas {
  w: number;
  h: number;
  px: Uint8Array;
}

export function createCanvas(w: number = PANEL_W, h: number = PANEL_H): Canvas {
  return { w, h, px: new Uint8Array(w * h).fill(1) };
}

export function setPixel(c: Canvas, x: number, y: number, v: number = 0): void {
  const xi = Math.round(x);
  const yi = Math.round(y);
  if (xi < 0 || yi < 0 || xi >= c.w || yi >= c.h) return;
  c.px[yi * c.w + xi] = v;
}

export function fillRect(c: Canvas, x: number, y: number, w: number, h: number, v: number = 0): void {
  const x0 = Math.max(0, Math.round(x));
  const y0 = Math.max(0, Math.round(y));
  const x1 = Math.min(c.w, Math.round(x + w));
  const y1 = Math.min(c.h, Math.round(y + h));
  for (let yy = y0; yy < y1; yy++) {
    for (let xx = x0; xx < x1; xx++) c.px[yy * c.w + xx] = v;
  }
}

export function hline(c: Canvas, x: number, y: number, len: number, t: number = 1): void {
  fillRect(c, x, y, len, t);
}

export function vline(c: Canvas, x: number, y: number, len: number, t: number = 1): void {
  fillRect(c, x, y, t, len);
}

export function strokeRect(c: Canvas, x: number, y: number, w: number, h: number, t: number = 1): void {
  hline(c, x, y, w, t);
  hline(c, x, y + h - t, w, t);
  vline(c, x, y, h, t);
  vline(c, x + w - t, y, h, t);
}

export function invertRect(c: Canvas, x: number, y: number, w: number, h: number): void {
  const x0 = Math.max(0, Math.round(x));
  const y0 = Math.max(0, Math.round(y));
  const x1 = Math.min(c.w, Math.round(x + w));
  const y1 = Math.min(c.h, Math.round(y + h));
  for (let yy = y0; yy < y1; yy++) {
    for (let xx = x0; xx < x1; xx++) c.px[yy * c.w + xx] ^= 1;
  }
}

/** Stamp a t×t square centred on (x, y); the building block for thick lines. */
function stamp(c: Canvas, x: number, y: number, t: number): void {
  const off = Math.floor((t - 1) / 2);
  fillRect(c, x - off, y - off, t, t);
}

/** Bresenham line. Steps are capped so a wild coordinate cannot spin the CPU. */
export function line(c: Canvas, x0: number, y0: number, x1: number, y1: number, t: number = 1): void {
  let x = Math.round(x0);
  let y = Math.round(y0);
  const xe = Math.round(x1);
  const ye = Math.round(y1);
  const dx = Math.abs(xe - x);
  const dy = -Math.abs(ye - y);
  const sx = x < xe ? 1 : -1;
  const sy = y < ye ? 1 : -1;
  let err = dx + dy;
  for (let i = 0; i < 4000; i++) {
    stamp(c, x, y, t);
    if (x === xe && y === ye) break;
    const e2 = 2 * err;
    if (e2 >= dy) { err += dy; x += sx; }
    if (e2 <= dx) { err += dx; y += sy; }
  }
}

/** Midpoint circle outline. */
export function circle(c: Canvas, cx: number, cy: number, r: number, t: number = 1): void {
  let x = Math.round(r);
  let y = 0;
  let err = 1 - x;
  while (x >= y) {
    stamp(c, cx + x, cy + y, t); stamp(c, cx - x, cy + y, t);
    stamp(c, cx + x, cy - y, t); stamp(c, cx - x, cy - y, t);
    stamp(c, cx + y, cy + x, t); stamp(c, cx - y, cy + x, t);
    stamp(c, cx + y, cy - x, t); stamp(c, cx - y, cy - x, t);
    y++;
    if (err < 0) err += 2 * y + 1;
    else { x--; err += 2 * (y - x) + 1; }
  }
}

/** 8x8 bitmap text. `scale` is an integer: 1 = 8px tall, 2 = 16px, 3 = 24px. */
export function text(c: Canvas, x: number, y: number, s: string, scale: number = 1): void {
  drawText(c.px, c.w, c.h, Math.round(x), Math.round(y), s, scale);
}

export function textWidth(s: string, scale: number = 1): number {
  return measureText(s, scale);
}

export function countBlack(c: Canvas): number {
  let n = 0;
  for (let i = 0; i < c.px.length; i++) if (c.px[i] === 0) n++;
  return n;
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm run typecheck && npm run test:utils 2>&1 | tail -8`
Expected: typecheck clean; `# pass 120`, `# fail 0` (112 existing + 8 new).

- [ ] **Step 5: Commit**

```bash
git add src/hud/canvas.ts tests/hud.test.js
git commit -m "Add a 1-bit drawing canvas for the desk panel pages"
```

---

### Task 2: Stroke-font engine and the Latin stroke font

Big numerals scaled up from the 8x8 font look crude. Glyphs defined as polylines on a small grid can be drawn at any size with a chosen line thickness, and the same engine will draw Aurebesh in Task 6.

**Files:**
- Create: `src/hud/stroke.ts`
- Test: `tests/hud.test.js` (append)

**Interfaces:**
- Consumes: `Canvas`, `line`, `fillRect` from Task 1.
- Produces:
  - `interface StrokeFont { w: number; h: number; gap: number; glyphs: Record<string, string> }`
  - `parseGlyph(g: string): number[][][]` (list of polylines, each a list of `[x, y]`)
  - `strokeTextWidth(font, s, cell): number`
  - `drawStrokeText(c, font, x, y, s, cell, t): void`
  - `LATIN: StrokeFont` (grid 4 wide, 6 tall, gap 2; keys `A`–`Z`, `0`–`9`, `-`, `.`, `:`, `/`, `%`)

A glyph string is polylines separated by `|`; each polyline is `x,y` points separated by spaces; `y` grows downward. A polyline with one point is a dot.

- [ ] **Step 1: Write the failing tests**

Append to `tests/hud.test.js`:

```js
const st = fromBuild("src/hud/stroke.js");

test("parseGlyph splits polylines and points", () => {
  assert.deepEqual(st.parseGlyph("0,0 4,0|2,0 2,6"), [[[0, 0], [4, 0]], [[2, 0], [2, 6]]]);
  assert.deepEqual(st.parseGlyph("2,6"), [[[2, 6]]]);
});

test("LATIN covers A-Z and 0-9 with points inside its grid", () => {
  const need = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.:/%".split("");
  for (const ch of need) {
    const g = st.LATIN.glyphs[ch];
    assert.ok(g, `missing glyph ${ch}`);
    for (const poly of st.parseGlyph(g)) {
      for (const [x, y] of poly) {
        assert.ok(x >= 0 && x <= st.LATIN.w, `${ch} x=${x}`);
        assert.ok(y >= 0 && y <= st.LATIN.h, `${ch} y=${y}`);
      }
    }
  }
});

test("strokeTextWidth counts advances without a trailing gap", () => {
  // 3 glyphs of width 4 plus 2 gaps of 2 = 16 units
  assert.equal(st.strokeTextWidth(st.LATIN, "ABC", 5), 16 * 5);
  assert.equal(st.strokeTextWidth(st.LATIN, "", 5), 0);
});

test("drawStrokeText draws ink, skips unknown characters, and stays in its box", () => {
  const c = cv.createCanvas(200, 100);
  st.drawStrokeText(c, st.LATIN, 10, 10, "L", 10, 2);
  assert.ok(cv.countBlack(c) > 0);
  // L is "0,0 0,6 4,6": nothing above y=10 or left of x=10 (minus stamp offset 0 for t=2)
  for (let x = 0; x < 200; x++) assert.equal(c.px[5 * 200 + x], 1);
  const d = cv.createCanvas(200, 100);
  assert.doesNotThrow(() => st.drawStrokeText(d, st.LATIN, 10, 10, "é~", 10, 2));
  assert.equal(cv.countBlack(d), 0, "unknown characters draw nothing");
});

test("a space advances without drawing", () => {
  const c = cv.createCanvas(300, 100);
  st.drawStrokeText(c, st.LATIN, 0, 0, "I I", 10, 1);
  const cols = new Set();
  for (let y = 0; y < 100; y++) for (let x = 0; x < 300; x++) if (c.px[y * 300 + x] === 0) cols.add(x);
  assert.ok([...cols].some((x) => x < 50));
  assert.ok([...cols].some((x) => x >= 120), "second I sits two advances to the right");
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm run test:utils 2>&1 | tail -20`
Expected: FAIL, `Cannot find module '.../src/hud/stroke.js'`

- [ ] **Step 3: Write the implementation**

Create `src/hud/stroke.ts`:

```ts
/**
 * Stroke fonts: glyphs as polylines on a small grid, drawn with the canvas line
 * primitive. They scale to any size with a chosen thickness, need no font file,
 * and contain no grey — the properties the mono panel needs (DECISIONS #48, #66).
 *
 * Glyph syntax: polylines separated by "|", points as "x,y" separated by spaces,
 * y growing downward. A single-point polyline is a dot.
 */

import type { Canvas } from "./canvas";
import { line } from "./canvas";

export interface StrokeFont {
  /** Grid width and height of one glyph, in grid units. */
  w: number;
  h: number;
  /** Space between glyphs, in grid units. */
  gap: number;
  glyphs: Record<string, string>;
}

export function parseGlyph(g: string): number[][][] {
  return g.split("|").map((poly) =>
    poly.trim().split(/\s+/).map((p) => p.split(",").map(Number)));
}

export function strokeTextWidth(font: StrokeFont, s: string, cell: number): number {
  if (s.length === 0) return 0;
  return (s.length * font.w + (s.length - 1) * font.gap) * cell;
}

/**
 * Draw `s` with its top-left at (x, y). `cell` is pixels per grid unit and `t`
 * the line thickness in pixels. Characters with no glyph (including the space)
 * advance without drawing.
 */
export function drawStrokeText(
  c: Canvas, font: StrokeFont, x: number, y: number, s: string, cell: number, t: number,
): void {
  let gx = x;
  for (const ch of s) {
    const g = font.glyphs[ch];
    if (g) {
      for (const poly of parseGlyph(g)) {
        if (poly.length === 1) {
          line(c, gx + poly[0][0] * cell, y + poly[0][1] * cell, gx + poly[0][0] * cell, y + poly[0][1] * cell, t + 1);
          continue;
        }
        for (let i = 0; i < poly.length - 1; i++) {
          line(c, gx + poly[i][0] * cell, y + poly[i][1] * cell,
            gx + poly[i + 1][0] * cell, y + poly[i + 1][1] * cell, t);
        }
      }
    }
    gx += (font.w + font.gap) * cell;
  }
}

const O = "1,0 3,0 4,1 4,5 3,6 1,6 0,5 0,1 1,0";
const P = "0,6 0,0 3,0 4,1 4,2 3,3 0,3";

/** Angular capitals and digits on a 4x6 grid. Curves are chamfered corners. */
export const LATIN: StrokeFont = {
  w: 4,
  h: 6,
  gap: 2,
  glyphs: {
    A: "0,6 0,2 2,0 4,2 4,6|0,4 4,4",
    B: "0,0 0,6 3,6 4,5 4,4 3,3 0,3|3,3 4,2 4,1 3,0 0,0",
    C: "4,1 3,0 1,0 0,1 0,5 1,6 3,6 4,5",
    D: "0,0 0,6 3,6 4,5 4,1 3,0 0,0",
    E: "4,0 0,0 0,6 4,6|0,3 3,3",
    F: "4,0 0,0 0,6|0,3 3,3",
    G: "4,1 3,0 1,0 0,1 0,5 1,6 3,6 4,5 4,3 2,3",
    H: "0,0 0,6|4,0 4,6|0,3 4,3",
    I: "1,0 3,0|2,0 2,6|1,6 3,6",
    J: "4,0 4,5 3,6 1,6 0,5",
    K: "0,0 0,6|4,0 0,3 4,6",
    L: "0,0 0,6 4,6",
    M: "0,6 0,0 2,3 4,0 4,6",
    N: "0,6 0,0 4,6 4,0",
    O,
    P,
    Q: `${O}|2,4 4,6`,
    R: `${P}|2,3 4,6`,
    S: "4,1 3,0 1,0 0,1 0,2 1,3 3,3 4,4 4,5 3,6 1,6 0,5",
    T: "0,0 4,0|2,0 2,6",
    U: "0,0 0,5 1,6 3,6 4,5 4,0",
    V: "0,0 2,6 4,0",
    W: "0,0 1,6 2,3 3,6 4,0",
    X: "0,0 4,6|4,0 0,6",
    Y: "0,0 2,3 4,0|2,3 2,6",
    Z: "0,0 4,0 0,6 4,6",
    "0": `${O}|1,5 3,1`,
    "1": "1,1 2,0 2,6|1,6 3,6",
    "2": "0,1 1,0 3,0 4,1 4,2 0,6 4,6",
    "3": "0,1 1,0 3,0 4,1 4,2 3,3 4,4 4,5 3,6 1,6 0,5|2,3 3,3",
    "4": "3,6 3,0 0,4 4,4",
    "5": "4,0 0,0 0,3 3,3 4,4 4,5 3,6 1,6 0,5",
    "6": "4,1 3,0 1,0 0,1 0,5 1,6 3,6 4,5 4,4 3,3 0,3",
    "7": "0,0 4,0 1,6",
    "8": "1,3 0,2 0,1 1,0 3,0 4,1 4,2 3,3 1,3 0,4 0,5 1,6 3,6 4,5 4,4 3,3",
    "9": "4,3 1,3 0,2 0,1 1,0 3,0 4,1 4,5 3,6 1,6 0,5",
    "-": "1,3 3,3",
    ".": "2,6",
    ":": "2,2|2,5",
    "/": "0,6 4,0",
    "%": "0,6 4,0|0,0 1,0 1,1 0,1 0,0|3,5 4,5 4,6 3,6 3,5",
  },
};
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm run typecheck && npm run test:utils 2>&1 | tail -8`
Expected: `# pass 125`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add src/hud/stroke.ts tests/hud.test.js
git commit -m "Add a stroke-font engine and angular Latin capitals for the desk pages"
```

---

### Task 3: Style constants and the reusable parts of the look

**Files:**
- Create: `src/hud/style.ts`, `src/hud/parts.ts`
- Test: `tests/hud.test.js` (append)

**Interfaces:**
- Consumes: everything from Tasks 1 and 2.
- Produces, from `style.ts`: `MARGIN = 8`, `RULE = 1`, `LABEL_SCALE = 2`, `SMALL_SCALE = 1`, `STROKE_BIG = 4`, `STROKE_MED = 3`.
- Produces, from `parts.ts`:
  - `plus(c, x, y, arm = 4): void`
  - `frame(c): void` — border inset by `MARGIN`, with `+` marks inside each corner
  - `brackets(c, x, y, w, h, len = 18, t = 2): void`
  - `crosshair(c, cx, cy, r, gap = 6): void`
  - `ruler(c, x, y, len, step = 10, majorEvery = 5): void` — horizontal ticks rising from `y`
  - `hatch(c, x, y, w, h, step = 6): void`
  - `dotFill(c, x, y, w, h, level): void` — `level` 0 (white) to 1 (black), ordered 4x4 dither
  - `titleBar(c, left, right): void`
  - `accentBlock(c, x, y, label, scale = LABEL_SCALE): number` — returns the block width
  - `bigNumber(c, x, y, value, unit, cell = 8): void`

- [ ] **Step 1: Write the failing tests**

Append to `tests/hud.test.js`:

```js
const parts = fromBuild("src/hud/parts.js");
const style = fromBuild("src/hud/style.js");

test("frame draws a border inset by the margin and leaves the centre white", () => {
  const c = cv.createCanvas();
  parts.frame(c);
  const m = style.MARGIN;
  assert.equal(at(c, m, m), 0);
  assert.equal(at(c, 800 - m - 1, 480 - m - 1), 0);
  assert.equal(at(c, 400, m), 0);
  assert.equal(at(c, 0, 0), 1, "outside the margin stays white");
  assert.equal(at(c, 400, 240), 1);
});

test("brackets mark only the corners", () => {
  const c = cv.createCanvas(200, 200);
  parts.brackets(c, 20, 20, 100, 80, 10, 2);
  assert.equal(at(c, 20, 20), 0);
  assert.equal(at(c, 119, 99), 0);
  assert.equal(at(c, 70, 20), 1, "the middle of an edge is open");
  assert.equal(at(c, 20, 60), 1);
});

test("dotFill density follows the level and stays in its box", () => {
  const box = 40 * 40;
  const fill = (level) => {
    const c = cv.createCanvas(60, 60);
    parts.dotFill(c, 10, 10, 40, 40, level);
    assert.equal(at(c, 5, 5), 1);
    assert.equal(at(c, 55, 55), 1);
    return cv.countBlack(c) / box;
  };
  assert.equal(fill(0), 0);
  assert.equal(fill(1), 1);
  assert.ok(Math.abs(fill(0.5) - 0.5) < 0.07);
  assert.ok(fill(0.25) < fill(0.75));
});

test("hatch draws diagonal stripes inside its box only", () => {
  const c = cv.createCanvas(60, 40);
  parts.hatch(c, 10, 10, 40, 20, 6);
  const black = cv.countBlack(c);
  assert.ok(black > 0 && black < 40 * 20);
  assert.equal(at(c, 5, 15), 1);
  assert.equal(at(c, 55, 15), 1);
});

test("accentBlock is a solid block with white text and reports its width", () => {
  const c = cv.createCanvas(300, 60);
  const w = parts.accentBlock(c, 10, 10, "KEY", 2);
  assert.equal(w, cv.textWidth("KEY", 2) + 12);
  assert.equal(at(c, 10, 10), 0, "block corner is black");
  let white = 0;
  for (let y = 10; y < 10 + 24; y++) for (let x = 10; x < 10 + w; x++) if (at(c, x, y) === 1) white++;
  assert.ok(white > 0, "text shows as white inside the block");
});

test("ruler, crosshair, titleBar and bigNumber all draw without throwing", () => {
  const c = cv.createCanvas();
  parts.ruler(c, 40, 100, 300);
  parts.crosshair(c, 400, 240, 60);
  parts.titleBar(c, "LEFT / TITLE", "2026-10-12");
  parts.bigNumber(c, 40, 300, "14", "VSL");
  assert.ok(cv.countBlack(c) > 500);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm run test:utils 2>&1 | tail -20`
Expected: FAIL, `Cannot find module '.../src/hud/parts.js'`

- [ ] **Step 3: Write the implementation**

Create `src/hud/style.ts`:

```ts
/**
 * Tunable constants for the desk panel look. These are the values the owner
 * settles from the calibration card at /desk/test on the real panel — change
 * them here, not in the pages.
 */

/** Gap between the panel edge and the outer frame, in pixels. */
export const MARGIN = 8;
/** Thickness of a hairline rule. */
export const RULE = 1;
/** 8x8 font scale for labels (2 = 16px tall). */
export const LABEL_SCALE = 2;
/** 8x8 font scale for the smallest captions (1 = 8px tall). */
export const SMALL_SCALE = 1;
/** Stroke thickness for large stroke text. */
export const STROKE_BIG = 4;
/** Stroke thickness for medium stroke text. */
export const STROKE_MED = 3;
```

Create `src/hud/parts.ts`:

```ts
/**
 * The reusable pieces of the desk panel's "targeting HUD" look. Every function
 * draws black on the white canvas; nothing here knows about any specific page.
 */

import type { Canvas } from "./canvas";
import {
  circle, fillRect, hline, invertRect, setPixel, strokeRect, text, textWidth, vline,
} from "./canvas";
import { drawStrokeText, LATIN, strokeTextWidth } from "./stroke";
import { LABEL_SCALE, MARGIN, RULE, SMALL_SCALE, STROKE_BIG } from "./style";

/** A "+" registration mark centred on (x, y). */
export function plus(c: Canvas, x: number, y: number, arm: number = 4): void {
  hline(c, x - arm, y, arm * 2 + 1);
  vline(c, x, y - arm, arm * 2 + 1);
}

/** Outer border inset by MARGIN, with a "+" inside each corner. */
export function frame(c: Canvas): void {
  strokeRect(c, MARGIN, MARGIN, c.w - MARGIN * 2, c.h - MARGIN * 2, RULE);
  const inset = MARGIN + 12;
  plus(c, inset, inset);
  plus(c, c.w - inset - 1, inset);
  plus(c, inset, c.h - inset - 1);
  plus(c, c.w - inset - 1, c.h - inset - 1);
}

/** Corner brackets around a box, leaving the middle of each edge open. */
export function brackets(
  c: Canvas, x: number, y: number, w: number, h: number, len: number = 18, t: number = 2,
): void {
  hline(c, x, y, len, t); vline(c, x, y, len, t);
  hline(c, x + w - len, y, len, t); vline(c, x + w - t, y, len, t);
  hline(c, x, y + h - t, len, t); vline(c, x, y + h - len, len, t);
  hline(c, x + w - len, y + h - t, len, t); vline(c, x + w - t, y + h - len, len, t);
}

/** A circle with four tick lines that stop `gap` pixels short of the centre. */
export function crosshair(c: Canvas, cx: number, cy: number, r: number, gap: number = 6): void {
  circle(c, cx, cy, r);
  hline(c, cx - r - 10, cy, r + 10 - gap);
  hline(c, cx + gap + 1, cy, r + 10 - gap);
  vline(c, cx, cy - r - 10, r + 10 - gap);
  vline(c, cx, cy + gap + 1, r + 10 - gap);
}

/** Horizontal ruler: a baseline at `y` with ticks rising above it. */
export function ruler(
  c: Canvas, x: number, y: number, len: number, step: number = 10, majorEvery: number = 5,
): void {
  hline(c, x, y, len);
  for (let i = 0, n = 0; i <= len; i += step, n++) {
    const tick = n % majorEvery === 0 ? 8 : 4;
    vline(c, x + i, y - tick, tick);
  }
}

/** Diagonal stripes, one pixel wide, every `step` pixels. */
export function hatch(c: Canvas, x: number, y: number, w: number, h: number, step: number = 6): void {
  for (let yy = 0; yy < h; yy++) {
    for (let xx = 0; xx < w; xx++) {
      if ((xx + yy) % step === 0) setPixel(c, x + xx, y + yy);
    }
  }
}

const BAYER4 = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
];

/** Ordered-dither fill. `level` runs from 0 (white) to 1 (solid black). */
export function dotFill(c: Canvas, x: number, y: number, w: number, h: number, level: number): void {
  const threshold = Math.max(0, Math.min(1, level)) * 16;
  for (let yy = 0; yy < h; yy++) {
    for (let xx = 0; xx < w; xx++) {
      if (BAYER4[(y + yy) & 3][(x + xx) & 3] < threshold) setPixel(c, x + xx, y + yy);
    }
  }
}

/** Page title on the left, a short stamp on the right, and a rule beneath. */
export function titleBar(c: Canvas, left: string, right: string): void {
  const x = MARGIN + 28;
  const y = MARGIN + 16;
  text(c, x, y, left, LABEL_SCALE);
  text(c, c.w - x - textWidth(right, LABEL_SCALE), y, right, LABEL_SCALE);
  hline(c, x, y + 8 * LABEL_SCALE + 8, c.w - x * 2, RULE);
}

/**
 * The one accent this look has on a mono panel: a solid black block with white
 * text. Returns the block's width so callers can place what follows.
 */
export function accentBlock(c: Canvas, x: number, y: number, label: string, scale: number = LABEL_SCALE): number {
  const w = textWidth(label, scale) + 12;
  const h = 8 * scale + 8;
  fillRect(c, x, y, w, h, 1);
  text(c, x + 6, y + 4, label, scale);
  invertRect(c, x, y, w, h);
  return w;
}

/** A large stroke-drawn value with a small unit caption to its right. */
export function bigNumber(c: Canvas, x: number, y: number, value: string, unit: string, cell: number = 8): void {
  drawStrokeText(c, LATIN, x, y, value, cell, STROKE_BIG);
  text(c, x + strokeTextWidth(LATIN, value, cell) + 8, y, unit, SMALL_SCALE);
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm run typecheck && npm run test:utils 2>&1 | tail -8`
Expected: `# pass 131`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add src/hud/style.ts src/hud/parts.ts tests/hud.test.js
git commit -m "Add the reusable parts of the desk panel's HUD look"
```

---

### Task 4: Page wrapper, calibration card and the `/desk/test` route

**Files:**
- Create: `src/hud/page.ts`, `src/hud/test-card.ts`, `src/pages/desk.ts`
- Modify: `src/index.ts` (import near line 11; `case` before `case "/health":` near line 1275; endpoint list near line 1284)
- Test: `tests/hud.test.js` (append)

**Interfaces:**
- Consumes: Tasks 1–3; `encodePNG1Bit(pixels, width, height): Promise<Uint8Array>` and `pngToBase64(png): string` from `src/png.ts`; `htmlResponse(html, cacheControl)` from `src/response.ts`; `unavailableHTML(title, detail)` from `src/pages/unavailable.ts`.
- Produces:
  - `canvasToPng(c: Canvas): Promise<Uint8Array>`
  - `hudPageHTML(title: string, base64: string): string`
  - `hudPageResponse(title: string, draw: () => Canvas): Promise<Response>`
  - `drawTestCard(): Canvas`
  - `handleDeskTest(): Promise<Response>`

- [ ] **Step 1: Write the failing tests**

Append to `tests/hud.test.js`:

```js
const page = fromBuild("src/hud/page.js");
const card = fromBuild("src/hud/test-card.js");
const desk = fromBuild("src/pages/desk.js");

/** Width, height and bit depth from a PNG's IHDR chunk. */
function pngHeader(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return { sig: [...bytes.slice(0, 4)], w: dv.getUint32(16), h: dv.getUint32(20), depth: bytes[24] };
}

const inlinedPng = (html) => {
  const m = html.match(/src="data:image\/png;base64,([^"]+)"/);
  assert.ok(m, "page must inline a PNG data URI");
  return new Uint8Array(Buffer.from(m[1], "base64"));
};

test("canvasToPng produces an 800x480 1-bit PNG", async () => {
  const png = await page.canvasToPng(cv.createCanvas());
  const h = pngHeader(png);
  assert.deepEqual(h.sig, [137, 80, 78, 71]);
  assert.equal(h.w, 800);
  assert.equal(h.h, 480);
  assert.equal(h.depth, 1);
});

test("hudPageResponse is one request: inline image, no script, no other URLs", async () => {
  const res = await page.hudPageResponse("Desk Test", () => cv.createCanvas());
  assert.equal(res.status, 200);
  assert.match(res.headers.get("Content-Type"), /^text\/html/);
  assert.equal(res.headers.get("Cache-Control"), "no-store");
  const html = await res.text();
  assert.equal(pngHeader(inlinedPng(html)).w, 800);
  assert.doesNotMatch(html, /<script/i);
  assert.doesNotMatch(html, /src="\//, "no second request to the Worker");
  assert.doesNotMatch(html, /https?:\/\//, "no external URL");
});

test("a draw function that throws becomes the text page, not an error", async () => {
  const res = await page.hudPageResponse("Desk Test", () => { throw new Error("boom"); });
  assert.equal(res.status, 200);
  const html = await res.text();
  assert.match(html, /Desk Test/);
  assert.doesNotMatch(html, /<img/i);
  assert.doesNotMatch(html, /boom/, "the error text is not shown on the panel");
});

test("the calibration card fills the panel with a moderate amount of ink", () => {
  const c = card.drawTestCard();
  assert.equal(c.w, 800);
  assert.equal(c.h, 480);
  const ratio = cv.countBlack(c) / (800 * 480);
  assert.ok(ratio > 0.02 && ratio < 0.5, `black ratio ${ratio}`);
});

test("/desk/test serves the card as an inlined page", async () => {
  const res = await desk.handleDeskTest();
  assert.equal(res.status, 200);
  assert.equal(pngHeader(inlinedPng(await res.text())).h, 480);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm run test:utils 2>&1 | tail -20`
Expected: FAIL, `Cannot find module '.../src/hud/page.js'`

- [ ] **Step 3: Write the implementation**

Create `src/hud/page.ts`:

```ts
/**
 * Turn a drawn canvas into what the panel loads: one HTML document with the
 * 1-bit PNG inlined as a data URI. SenseCraft screenshots about a second after
 * the document arrives and cancels any sub-resource still loading, so a panel
 * page must be a single request (DECISIONS #65).
 */

import type { Canvas } from "./canvas";
import { encodePNG1Bit, pngToBase64 } from "../png";
import { htmlResponse } from "../response";
import { unavailableHTML } from "../pages/unavailable";

export function canvasToPng(c: Canvas): Promise<Uint8Array> {
  return encodePNG1Bit(c.px, c.w, c.h);
}

export function hudPageHTML(title: string, base64: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800">
<title>${title}</title>
<style>
  * { margin: 0; padding: 0; }
  html, body { width: 800px; height: 480px; overflow: hidden; background: #fff; }
  img { display: block; width: 800px; height: 480px; image-rendering: pixelated; }
</style>
</head>
<body>
  <img src="data:image/png;base64,${base64}" width="800" height="480" alt="${title}">
</body>
</html>`;
}

/**
 * Draw, encode and wrap. `title` must be a fixed string from our own code, never
 * request input: it is placed in the HTML unescaped. If drawing or encoding
 * fails, the panel gets the plain text page rather than a 500 (DECISIONS #58).
 */
export async function hudPageResponse(title: string, draw: () => Canvas): Promise<Response> {
  try {
    const png = await canvasToPng(draw());
    return htmlResponse(hudPageHTML(title, pngToBase64(png)), "no-store");
  } catch (err) {
    console.error(`${title}: draw failed:`, err);
    return htmlResponse(
      unavailableHTML(title, "This page could not be drawn. It will try again at the next refresh."),
      "no-store",
    );
  }
}
```

Create `src/hud/test-card.ts`:

```ts
/**
 * Calibration card for the desk panel (/desk/test). It puts the open style
 * questions side by side so they can be judged on the real panel, where
 * SenseCraft's cloud render and the e-ink itself decide what is legible
 * (DECISIONS #48). Keep it: later pages are tuned against it.
 */

import type { Canvas } from "./canvas";
import { createCanvas, fillRect, hline, text, vline } from "./canvas";
import { accentBlock, bigNumber, brackets, crosshair, dotFill, frame, hatch, ruler, titleBar } from "./parts";
import { drawStrokeText, LATIN } from "./stroke";

export function drawTestCard(): Canvas {
  const c = createCanvas();
  frame(c);
  titleBar(c, "DESK PANEL / CALIBRATION", "CARD 1");

  // Column 1: the 8x8 bitmap font at three sizes.
  text(c, 36, 66, "A  8X8 FONT", 1);
  text(c, 36, 82, "1X  STORAGE 512 GB / 89%", 1);
  text(c, 36, 98, "2X  STORAGE 89%", 2);
  text(c, 36, 122, "3X  STORAGE", 3);

  // Column 2: stroke text at three sizes and thicknesses.
  text(c, 420, 66, "B  STROKE FONT", 1);
  drawStrokeText(c, LATIN, 420, 82, "ORION 0123456789", 3, 2);
  drawStrokeText(c, LATIN, 420, 110, "ORION 89%", 5, 3);
  drawStrokeText(c, LATIN, 420, 150, "14:30", 9, 4);

  hline(c, 36, 214, 728);

  // Row 2: rule thickness.
  text(c, 36, 224, "C  RULES 1 / 2 / 3 PX", 1);
  hline(c, 36, 242, 150, 1);
  hline(c, 36, 252, 150, 2);
  hline(c, 36, 264, 150, 3);
  vline(c, 200, 240, 28, 1);
  vline(c, 210, 240, 28, 2);
  vline(c, 222, 240, 28, 3);

  // Row 2: dot fills from light to solid.
  text(c, 260, 224, "D  DOT FILL 12 / 25 / 50 / 75 / 100", 1);
  const levels = [0.125, 0.25, 0.5, 0.75, 1];
  levels.forEach((lv, i) => dotFill(c, 260 + i * 60, 240, 52, 30, lv));

  // Row 2: hatch band and accent block.
  text(c, 580, 224, "E  HATCH + ACCENT", 1);
  hatch(c, 580, 240, 90, 30, 6);
  accentBlock(c, 684, 240, "CRIT", 2);

  hline(c, 36, 284, 728);

  // Row 3: the composed look.
  text(c, 36, 294, "F  COMPOSED", 1);
  brackets(c, 36, 312, 220, 130);
  crosshair(c, 146, 377, 34);
  dotFill(c, 124, 355, 44, 44, 0.5);

  bigNumber(c, 290, 316, "14", "VSL");
  bigNumber(c, 290, 384, "89", "PCT");
  ruler(c, 420, 440, 330);
  text(c, 420, 316, "TARGET INSPECTOR", 2);
  text(c, 420, 344, "VESSEL  MV ORION", 1);
  text(c, 420, 358, "SPEED   9.4 KN", 1);
  text(c, 420, 372, "HDG     011 T", 1);
  fillRect(c, 420, 392, 60, 6);
  accentBlock(c, 620, 340, "GAP 18M", 2);

  return c;
}
```

Create `src/pages/desk.ts`:

```ts
/**
 * Desk panel pages (third display: a mono E1001 on the home-office desk).
 *
 * /desk/test — calibration card for judging type, rules and fills on the panel
 *
 * Every page here is a Worker-drawn 1-bit PNG inlined in one HTML document, a
 * pure function of the date: no KV, no AI, no cron (DECISIONS #66).
 */

import { hudPageResponse } from "../hud/page";
import { drawTestCard } from "../hud/test-card";

export function handleDeskTest(): Promise<Response> {
  return hudPageResponse("Desk Panel Calibration", drawTestCard);
}
```

In `src/index.ts`, add the import after the skyline pages import (line 11):

```ts
import { handleDeskTest } from "./pages/desk";
```

Add the route immediately before `case "/health":`:

```ts
      case "/desk/test":
        return handleDeskTest();
```

In the 404 `endpoints` array, change the skyline line to:

```ts
              "/skyline", "/skyline-bw", "/skyline.png", "/desk/test",
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm run typecheck && npm run test:utils 2>&1 | tail -8 && npm run dry-run 2>&1 | tail -5`
Expected: `# pass 136`, `# fail 0`; dry-run prints the bundle size with no error.

- [ ] **Step 5: Look at the card in a browser at 800x480**

```bash
lsof -ti:8790 || true
CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c npx wrangler dev --port 8790
```

Open `http://localhost:8790/desk/test` at an 800x480 viewport and take a screenshot. Check: nothing is cut off at the right or bottom edge, no two blocks overlap, every label is readable. Adjust coordinates in `test-card.ts` until that holds, re-running the tests after each change. Check the network panel shows exactly one request. Stop the dev server when done.

- [ ] **Step 6: Commit**

```bash
git add src/hud/page.ts src/hud/test-card.ts src/pages/desk.ts src/index.ts tests/hud.test.js
git commit -m "Add the desk panel calibration card at /desk/test"
```

---

### Task 5: Checkpoint — the owner judges the card on the real panel

No code. This is the gate the spec sets before any real page is built.

- [ ] **Step 1: Ask the owner for approval to deploy.** Say what is being deployed (one new route, `/desk/test`, no behaviour change to existing routes) and wait for a yes.

- [ ] **Step 2: Deploy and confirm**

```bash
CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c npx wrangler deploy
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" https://eink-dashboard.thiago-oliveira77.workers.dev/desk/test
curl -s https://eink-dashboard.thiago-oliveira77.workers.dev/health
```

Expected: `200 text/html; charset=utf-8`, and `/health` still reports `ok`. If the deploy returns `Authentication error [code: 10000]`, the owner must run `npx wrangler logout && npx wrangler login` themselves; do not ask for a token.

- [ ] **Step 3: Give the owner these steps, in plain words**
  1. Pair the new E1001 in SenseCraft HMI (Workspace, Add Device, enter the pair code on its screen).
  2. Add a page of type Web Function with the URL `https://eink-dashboard.thiago-oliveira77.workers.dev/desk/test`.
  3. Put it in a pagelist, set the interval to 15 minutes, press Deploy.
  4. Take a clear photo of the panel and send it.

- [ ] **Step 4: Decide from the photo, with the owner.** For each lettered block on the card: which bitmap size is the smallest that reads cleanly (block A), whether stroke text is sharp and at which thickness (B), the thinnest rule that survives (C), which dot levels look like distinct tones (D), whether hatch and the inverted block are clean (E).

- [ ] **Step 5: Record the choices in `src/hud/style.ts`** by changing only the constant values (`LABEL_SCALE`, `SMALL_SCALE`, `RULE`, `STROKE_BIG`, `STROKE_MED`). Run `npm run typecheck && npm run test:utils`, then commit:

```bash
git add src/hud/style.ts
git commit -m "Set desk panel style constants from the on-device calibration card"
```

If the panel shows that Worker-drawn type is not good enough at any size, **stop**: that invalidates spec section 5 and the owner must choose again between approaches A, B and C before Task 6.

---

### Task 6: Aurebesh stroke font and the 26-letter check sheet

The glyph data cannot be written from memory with confidence, so this task authors it against a reference and proves it by eye. That is why the stroke strings are not printed in this plan.

**Files:**
- Create: `src/hud/aurebesh.ts`
- Modify: `src/pages/desk.ts`, `src/index.ts`
- Test: `tests/hud.test.js` (append)

**Interfaces:**
- Consumes: `StrokeFont`, `parseGlyph`, `drawStrokeText`, `strokeTextWidth`, `LATIN` from Task 2; `createCanvas`, `text` from Task 1; `frame`, `titleBar` from Task 3; `hudPageResponse` from Task 4.
- Produces:
  - `AUREBESH: StrokeFont` with `w: 6`, `h: 6`, `gap: 2` and exactly the keys `A`–`Z`
  - `drawAurebeshSheet(): Canvas`
  - `handleDeskAurebesh(url: URL): Promise<Response>` in `src/pages/desk.ts` (serves the sheet for every request; Task 8 adds the word page and the `?sheet` switch)

- [ ] **Step 1: Write the failing tests**

Append to `tests/hud.test.js`:

```js
const aur = fromBuild("src/hud/aurebesh.js");
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

test("AUREBESH has exactly A-Z, each with real strokes inside the grid", () => {
  assert.deepEqual(Object.keys(aur.AUREBESH.glyphs).sort(), LETTERS);
  for (const ch of LETTERS) {
    const polys = st.parseGlyph(aur.AUREBESH.glyphs[ch]);
    const points = polys.flat();
    assert.ok(points.length >= 3, `${ch} needs at least three points`);
    for (const [x, y] of points) {
      assert.ok(Number.isFinite(x) && x >= 0 && x <= aur.AUREBESH.w, `${ch} x=${x}`);
      assert.ok(Number.isFinite(y) && y >= 0 && y <= aur.AUREBESH.h, `${ch} y=${y}`);
    }
  }
});

test("no two Aurebesh letters render identically", () => {
  const seen = new Map();
  for (const ch of LETTERS) {
    const c = cv.createCanvas(80, 80);
    st.drawStrokeText(c, aur.AUREBESH, 8, 8, ch, 10, 2);
    const key = Buffer.from(c.px).toString("base64");
    assert.ok(!seen.has(key), `${ch} looks the same as ${seen.get(key)}`);
    seen.set(key, ch);
  }
});

test("the check sheet shows all 26 letters on one panel", () => {
  const c = aur.drawAurebeshSheet();
  assert.equal(c.w, 800);
  assert.equal(c.h, 480);
  assert.ok(cv.countBlack(c) > 3000);
});

test("/desk/aurebesh?sheet serves the check sheet", async () => {
  const res = await desk.handleDeskAurebesh(new URL("https://x/desk/aurebesh?sheet"));
  assert.equal(res.status, 200);
  assert.equal(pngHeader(inlinedPng(await res.text())).w, 800);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm run test:utils 2>&1 | tail -20`
Expected: FAIL, `Cannot find module '.../src/hud/aurebesh.js'`

- [ ] **Step 3: Get a reference chart, outside the repo**

Download a chart of the Aurebesh alphabet from Wikimedia Commons (search Commons for "Aurebesh") into the session scratchpad directory. **Do not put it in the repository.** View it. Confirm it shows the 26 single letters with their Latin equivalents.

- [ ] **Step 4: Write the module with the sheet, then author the glyphs**

Create `src/hud/aurebesh.ts` with this structure. Fill `glyphs` with one stroke string per letter, authored while looking at the reference, using the same syntax as `LATIN` on a grid 6 wide and 6 tall:

```ts
/**
 * Aurebesh, the Star Wars alphabet, as stroke data for the desk panel.
 *
 * The 26 single letters only; the digraph letters (CH, SH, TH and others) are
 * left out. The letterforms originate from Star Wars. The stroke data below was
 * drawn for this project against a public reference chart; no font file or film
 * artwork is in the repository, and the use is personal and non-commercial.
 */

import type { Canvas } from "./canvas";
import { createCanvas, text } from "./canvas";
import { frame, titleBar } from "./parts";
import type { StrokeFont } from "./stroke";
import { drawStrokeText } from "./stroke";

export const AUREBESH: StrokeFont = {
  w: 6,
  h: 6,
  gap: 2,
  glyphs: {
    // One entry per letter A–Z, authored in Step 4 against the reference chart.
  },
};

/** All 26 letters with their Latin equivalents, for checking against a chart. */
export function drawAurebeshSheet(): Canvas {
  const c = createCanvas();
  frame(c);
  titleBar(c, "AUREBESH / CHECK SHEET", "A-Z");
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const cols = 9;
  const cellW = 82;
  const cellH = 130;
  for (let i = 0; i < letters.length; i++) {
    const x = 40 + (i % cols) * cellW;
    const y = 70 + Math.floor(i / cols) * cellH;
    drawStrokeText(c, AUREBESH, x, y, letters[i], 9, 3);
    text(c, x + 20, y + 66, letters[i], 2);
  }
  return c;
}
```

In `src/pages/desk.ts`, add the import and handler, and update the header comment's route list:

```ts
import { drawAurebeshSheet } from "../hud/aurebesh";

/** /desk/aurebesh — `?sheet` shows all 26 letters for checking against a chart. */
export function handleDeskAurebesh(_url: URL): Promise<Response> {
  return hudPageResponse("Aurebesh Check Sheet", drawAurebeshSheet);
}
```

(Until Task 8 adds the word page, every request to this route shows the sheet. Task 8 replaces this function and uses the `url` parameter.)

In `src/index.ts`, change the import to `import { handleDeskTest, handleDeskAurebesh } from "./pages/desk";`, add before `case "/health":`:

```ts
      case "/desk/aurebesh":
        return handleDeskAurebesh(url);
```

and add `"/desk/aurebesh"` after `"/desk/test"` in the 404 endpoint list.

- [ ] **Step 5: Verify by eye, letter by letter**

Run the dev server (`CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c npx wrangler dev --port 8790`), open `http://localhost:8790/desk/aurebesh?sheet` at 800x480, screenshot it, and compare each of the 26 glyphs with the reference chart. Fix any glyph that differs and repeat until all 26 match. Stop the dev server. Write down any letter you are not sure about.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `npm run typecheck && npm run test:utils 2>&1 | tail -8`
Expected: `# pass 140`, `# fail 0`.

- [ ] **Step 7: Commit**

```bash
git add src/hud/aurebesh.ts src/pages/desk.ts src/index.ts tests/hud.test.js
git commit -m "Add Aurebesh stroke glyphs and a 26-letter check sheet"
```

---

### Task 7: Word list, input cleaning and the date-to-word picker

**Files:**
- Create: `src/hud/aurebesh-words.ts`
- Test: `tests/hud.test.js` (append)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `WORDS: readonly string[]`
  - `MAX_WORD_LEN = 9`
  - `cleanWord(raw: string | null): string | null` — uppercase A–Z only, at most 9 letters; `null` if nothing is left
  - `parseDateParam(raw: string | null): string | null` — a valid `YYYY-MM-DD` or `null`
  - `dayOfYear(dateStr: string): number` — 1 to 366
  - `pickWord(dateStr: string): string`

- [ ] **Step 1: Write the failing tests**

Append to `tests/hud.test.js`:

```js
const words = fromBuild("src/hud/aurebesh-words.js");

test("every word is 3-9 capital letters and the list has no duplicates", () => {
  assert.ok(words.WORDS.length >= 100);
  for (const w of words.WORDS) assert.match(w, /^[A-Z]{3,9}$/, w);
  assert.equal(new Set(words.WORDS).size, words.WORDS.length);
});

test("dayOfYear handles the ends of the year and leap day", () => {
  assert.equal(words.dayOfYear("2026-01-01"), 1);
  assert.equal(words.dayOfYear("2026-12-31"), 365);
  assert.equal(words.dayOfYear("2028-02-29"), 60);
  assert.equal(words.dayOfYear("2028-12-31"), 366);
});

test("pickWord is stable per date, valid on day 366, and changes day to day", () => {
  assert.equal(words.pickWord("2026-10-12"), words.pickWord("2026-10-12"));
  assert.ok(words.WORDS.includes(words.pickWord("2028-12-31")));
  assert.notEqual(words.pickWord("2026-10-12"), words.pickWord("2026-10-13"));
});

test("parseDateParam accepts only real calendar dates", () => {
  assert.equal(words.parseDateParam("2026-10-12"), "2026-10-12");
  assert.equal(words.parseDateParam("2028-02-29"), "2028-02-29");
  for (const bad of [null, "", "yesterday", "2026-13-45", "2026-02-30", "2026-1-5", "20261012", "2026-10-12T00:00"]) {
    assert.equal(words.parseDateParam(bad), null, String(bad));
  }
});

test("cleanWord reduces any input to A-Z, capped at 9 letters", () => {
  assert.equal(words.cleanWord("hello"), "HELLO");
  assert.equal(words.cleanWord("R2-D2 droid!"), "RDDROID");
  assert.equal(words.cleanWord("supercalifragilistic"), "SUPERCALI");
  assert.equal(words.cleanWord("café \u{1F680}"), "CAF");
  for (const empty of [null, "", "   ", "1234", "\u{1F680}"]) assert.equal(words.cleanWord(empty), null);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm run test:utils 2>&1 | tail -20`
Expected: FAIL, `Cannot find module '.../src/hud/aurebesh-words.js'`

- [ ] **Step 3: Write the implementation**

Create `src/hud/aurebesh-words.ts`:

```ts
/**
 * Word of the day for the Aurebesh page: the list, the picker, and the cleaning
 * of the free test parameters. English, A–Z only, 3 to 9 letters so a word fits
 * one line of large glyphs.
 */

export const MAX_WORD_LEN = 9;

export const WORDS: readonly string[] = [
  "FORCE", "JEDI", "REBEL", "EMPIRE", "GALAXY", "PLANET", "DROID", "PILOT", "SABER", "MASTER",
  "PADAWAN", "KNIGHT", "TEMPLE", "COUNCIL", "SENATE", "ALLIANCE", "CRUISER", "FIGHTER", "FALCON", "SHUTTLE",
  "HANGAR", "CANTINA", "SMUGGLER", "BOUNTY", "HUNTER", "WOOKIEE", "EWOK", "JAWA", "TUSKEN", "BANTHA",
  "TAUNTAUN", "RANCOR", "SARLACC", "KRAYT", "PORG", "LOTHCAT", "ENDOR", "HOTH", "DAGOBAH", "TATOOINE",
  "NABOO", "BESPIN", "JAKKU", "ALDERAAN", "KASHYYYK", "MUSTAFAR", "SCARIF", "CRAIT", "YAVIN", "KESSEL",
  "CORELLIA", "MANDALORE", "KAMINO", "GEONOSIS", "UTAPAU", "FELUCIA", "LOTHAL", "DATHOMIR", "ILUM", "EXEGOL",
  "HOPE", "COURAGE", "PATIENCE", "TRUST", "BALANCE", "DESTINY", "WISDOM", "HONOR", "LOYALTY", "FREEDOM",
  "PEACE", "JUSTICE", "KINDNESS", "FRIEND", "FAMILY", "BROTHER", "SISTER", "FATHER", "MOTHER", "CHILD",
  "STAR", "MOON", "COMET", "NEBULA", "ORBIT", "ROCKET", "ENGINE", "SHIELD", "LASER", "BLASTER",
  "CRYSTAL", "KYBER", "HOLOCRON", "HOLOGRAM", "BEACON", "SIGNAL", "CIPHER", "SECRET", "MISSION", "SQUADRON",
  "COMMAND", "CAPTAIN", "ADMIRAL", "GENERAL", "TROOPER", "SCOUT", "RANGER", "GUARDIAN", "SENTINEL", "WARRIOR",
  "DESERT", "FOREST", "SWAMP", "CANYON", "GLACIER", "VOLCANO", "ASTEROID", "HORIZON", "SUNSET", "TWILIGHT",
];

/** Uppercase A–Z only, at most MAX_WORD_LEN letters; null when nothing usable remains. */
export function cleanWord(raw: string | null): string | null {
  if (!raw) return null;
  const letters = raw.toUpperCase().replace(/[^A-Z]/g, "").slice(0, MAX_WORD_LEN);
  return letters.length > 0 ? letters : null;
}

/** A real calendar date as YYYY-MM-DD, or null. */
export function parseDateParam(raw: string | null): string | null {
  if (!raw || !/^\d{4}-\d{2}-\d{2}$/.test(raw)) return null;
  const [y, m, d] = raw.split("-").map(Number);
  const t = new Date(Date.UTC(y, m - 1, d));
  const real = t.getUTCFullYear() === y && t.getUTCMonth() === m - 1 && t.getUTCDate() === d;
  return real ? raw : null;
}

/** 1 for January 1st through 365, or 366 in a leap year. Expects a valid YYYY-MM-DD. */
export function dayOfYear(dateStr: string): number {
  const [y, m, d] = dateStr.split("-").map(Number);
  return Math.round((Date.UTC(y, m - 1, d) - Date.UTC(y, 0, 1)) / 86400000) + 1;
}

/** The same date always gives the same word; the year offsets the cycle so years differ. */
export function pickWord(dateStr: string): string {
  const year = Number(dateStr.slice(0, 4));
  return WORDS[(dayOfYear(dateStr) + year * 7) % WORDS.length];
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm run typecheck && npm run test:utils 2>&1 | tail -8`
Expected: `# pass 145`, `# fail 0`. If the duplicate test fails, remove the repeated word and add a new one that matches `^[A-Z]{3,9}$`.

- [ ] **Step 5: Commit**

```bash
git add src/hud/aurebesh-words.ts tests/hud.test.js
git commit -m "Add the Aurebesh word list, picker and input cleaning"
```

---

### Task 8: The Aurebesh word page

**Files:**
- Create: `src/hud/aurebesh-page.ts`
- Modify: `src/pages/desk.ts`
- Test: `tests/hud.test.js` (append)

**Interfaces:**
- Consumes: `AUREBESH` (Task 6); `cleanWord`, `parseDateParam`, `pickWord` (Task 7); `LATIN`, `drawStrokeText`, `strokeTextWidth` (Task 2); parts and style (Task 3); `hudPageResponse` (Task 4); `getChicagoDateParts()` from `src/date-utils.ts`.
- Produces:
  - `fitCell(word: string): number` — pixels per grid unit so the word fits 620px, capped at 22
  - `drawAurebeshPage(word: string, dateStr: string): Canvas`
  - `handleDeskAurebesh(url)` now serves the word page unless `?sheet` is present

Layout (y positions in pixels): title bar at the top; the Aurebesh word centred inside brackets spanning x 60–740, y 78–268; the English word in stroke capitals centred at y 288; a rule at y 348; a `KEY` accent block at y 360; the key strip at y 366 showing each distinct letter's glyph with its Latin letter beneath.

- [ ] **Step 1: Write the failing tests**

Append to `tests/hud.test.js`:

```js
const ap = fromBuild("src/hud/aurebesh-page.js");

test("fitCell shrinks long words to fit and caps short ones", () => {
  assert.equal(ap.fitCell("JEDI"), 20);          // 4*6 + 3*2 = 30 units -> floor(620/30) = 20
  assert.equal(ap.fitCell("EWO"), 22);           // 22 units -> 28, capped at 22
  const nine = ap.fitCell("MANDALORE");          // 70 units -> 8
  assert.equal(nine, 8);
  assert.ok(st.strokeTextWidth(aur.AUREBESH, "MANDALORE", nine) <= 620);
});

test("the word page keeps all ink inside the frame for the shortest and longest words", () => {
  for (const w of ["EWO", "MANDALORE", "WWWWWWWWW"]) {
    const c = ap.drawAurebeshPage(w, "2026-10-12");
    for (let x = 0; x < 800; x++) {
      for (const y of [0, 1, 2, 3, 4, 5, 6, 7, 472, 473, 474, 475, 476, 477, 478, 479]) {
        assert.equal(c.px[y * 800 + x], 1, `${w}: ink in the margin at ${x},${y}`);
      }
    }
    for (let y = 0; y < 480; y++) {
      for (const x of [0, 1, 2, 3, 4, 5, 6, 7, 792, 793, 794, 795, 796, 797, 798, 799]) {
        assert.equal(c.px[y * 800 + x], 1, `${w}: ink in the margin at ${x},${y}`);
      }
    }
    assert.ok(cv.countBlack(c) > 4000);
  }
});

test("different words draw different pages", () => {
  const a = ap.drawAurebeshPage("JEDI", "2026-10-12");
  const b = ap.drawAurebeshPage("REBEL", "2026-10-12");
  assert.notEqual(Buffer.from(a.px).toString("base64"), Buffer.from(b.px).toString("base64"));
});

const pageBytes = async (qs) => {
  const res = await desk.handleDeskAurebesh(new URL(`https://x/desk/aurebesh${qs}`));
  assert.equal(res.status, 200);
  return Buffer.from(inlinedPng(await res.text())).toString("base64");
};

test("?word and ?date select the page; bad input falls back instead of failing", async () => {
  const jedi = await pageBytes("?word=jedi&date=2026-10-12");
  assert.equal(await pageBytes("?word=JEDI!!&date=2026-10-12"), jedi, "cleaned to the same word");
  assert.notEqual(await pageBytes("?word=rebel&date=2026-10-12"), jedi);
  const daily = await pageBytes("?date=2026-10-12");
  assert.equal(await pageBytes("?word=1234&date=2026-10-12"), daily, "unusable word -> daily word");
  assert.equal(await pageBytes("?word=&date=2026-10-12"), daily);
  assert.notEqual(await pageBytes("?date=2026-10-13"), daily);
  await pageBytes("?date=2026-13-45");            // must not throw
  await pageBytes("?date=yesterday&word=%F0%9F%9A%80");
  assert.notEqual(await pageBytes("?sheet"), daily, "?sheet still shows the check sheet");
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm run test:utils 2>&1 | tail -20`
Expected: FAIL, `Cannot find module '.../src/hud/aurebesh-page.js'`

- [ ] **Step 3: Write the implementation**

Create `src/hud/aurebesh-page.ts`:

```ts
/**
 * The Aurebesh word-of-the-day page: the word in large Aurebesh glyphs, its
 * English spelling beneath, and a key strip that maps each letter used to its
 * glyph so the word can be decoded by eye.
 */

import type { Canvas } from "./canvas";
import { createCanvas, hline, text } from "./canvas";
import { AUREBESH } from "./aurebesh";
import { accentBlock, brackets, frame, titleBar } from "./parts";
import { drawStrokeText, LATIN, strokeTextWidth } from "./stroke";
import { RULE, STROKE_BIG, STROKE_MED } from "./style";

const WORD_BOX_X = 60;
const WORD_BOX_Y = 78;
const WORD_BOX_W = 680;
const WORD_BOX_H = 190;
const WORD_MAX_W = 620;
const MAX_CELL = 22;

/** Pixels per grid unit so the Aurebesh word fits its box, capped for short words. */
export function fitCell(word: string): number {
  const units = word.length * AUREBESH.w + (word.length - 1) * AUREBESH.gap;
  return Math.min(MAX_CELL, Math.floor(WORD_MAX_W / units));
}

export function drawAurebeshPage(word: string, dateStr: string): Canvas {
  const c = createCanvas();
  frame(c);
  titleBar(c, "AUREBESH / WORD OF THE DAY", dateStr);

  brackets(c, WORD_BOX_X, WORD_BOX_Y, WORD_BOX_W, WORD_BOX_H);
  const cell = fitCell(word);
  const wordW = strokeTextWidth(AUREBESH, word, cell);
  const wordH = AUREBESH.h * cell;
  drawStrokeText(
    c, AUREBESH,
    Math.round((c.w - wordW) / 2),
    WORD_BOX_Y + Math.round((WORD_BOX_H - wordH) / 2),
    word, cell, cell >= 14 ? STROKE_BIG : STROKE_MED,
  );

  const latinCell = 7;
  drawStrokeText(
    c, LATIN, Math.round((c.w - strokeTextWidth(LATIN, word, latinCell)) / 2), 288,
    word, latinCell, STROKE_MED,
  );

  hline(c, 36, 348, c.w - 72, RULE);
  const keyX = 36 + accentBlock(c, 36, 360, "KEY") + 20;
  const letters = [...new Set(word.split(""))];
  const slot = Math.min(72, Math.floor((c.w - 36 - keyX) / letters.length));
  letters.forEach((ch, i) => {
    const x = keyX + i * slot;
    drawStrokeText(c, AUREBESH, x, 366, ch, 6, 2);
    text(c, x + 10, 412, ch, 2);
  });

  return c;
}
```

Replace `handleDeskAurebesh` in `src/pages/desk.ts`, add the imports, and update the header comment to list `/desk/aurebesh` with its `?date=`, `?word=` and `?sheet` parameters:

```ts
import { getChicagoDateParts } from "../date-utils";
import { drawAurebeshPage } from "../hud/aurebesh-page";
import { cleanWord, parseDateParam, pickWord } from "../hud/aurebesh-words";

/**
 * /desk/aurebesh — word of the day. Free test parameters, no auth, no cost:
 * `?date=YYYY-MM-DD`, `?word=HELLO`, and `?sheet` for the 26-letter check sheet.
 * Unusable input falls back to today's date and the daily word.
 */
export function handleDeskAurebesh(url: URL): Promise<Response> {
  if (url.searchParams.has("sheet")) {
    return hudPageResponse("Aurebesh Check Sheet", drawAurebeshSheet);
  }
  const dateStr = parseDateParam(url.searchParams.get("date")) ?? getChicagoDateParts().dateStr;
  const word = cleanWord(url.searchParams.get("word")) ?? pickWord(dateStr);
  return hudPageResponse("Aurebesh Word of the Day", () => drawAurebeshPage(word, dateStr));
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm run typecheck && npm run test:utils 2>&1 | tail -8 && npm run dry-run 2>&1 | tail -5`
Expected: `# pass 149`, `# fail 0`; dry-run succeeds.

- [ ] **Step 5: Look at it in a browser at 800x480**

Run the dev server on port 8790 and screenshot, at an 800x480 viewport: `/desk/aurebesh?word=EWO`, `/desk/aurebesh?word=MANDALORE`, `/desk/aurebesh?word=WWWWWWWWW`, and `/desk/aurebesh` with no parameters. Check on each: the word is centred inside the brackets and touches none of them; the English word does not collide with the rule below it; the key strip stays inside the frame; exactly one network request. Adjust the layout constants until all four hold, re-running the tests. Stop the dev server.

- [ ] **Step 6: Commit**

```bash
git add src/hud/aurebesh-page.ts src/pages/desk.ts tests/hud.test.js
git commit -m "Add the Aurebesh word-of-the-day page at /desk/aurebesh"
```

---

### Task 9: Documentation sweep, version, deploy, on-panel check

**Files:**
- Modify: `DECISIONS.md`, `README.md`, `CLAUDE.md`, `HANDOVER.md`, `docs/superpowers/specs/2026-10-12-desk-panel-design.md`, and, only with the owner's approval, `package.json`, `package-lock.json`, `src/index.ts` (`VERSION`)

- [ ] **Step 1: Ask the owner two things and wait for the answers:** approval to bump the version to **3.17.0**, and approval to deploy.

- [ ] **Step 2: Add DECISIONS #66** at the end of `DECISIONS.md`, titled `## 66. Desk Panel Pages Are Drawn by the Worker as 1-Bit Images (v3.17.0, <date>)`. It must state: the symptom being avoided (cloud-render fog on mono, #48); the decision (own canvas, bitmap and stroke type, PNG inlined in one document, no KV/AI/cron); the style constants chosen in Task 5 and what the owner saw on the card; the rejected alternatives with reasons (live HTML; Browser Rendering screenshot; TRMNL; custom firmware); what was verified, with the final test count; and that the Aurebesh strokes are our own data checked against a public chart. Do not edit older records.

- [ ] **Step 3: Update the other docs**
  - `README.md`: add a "Desk Panel (E1001 #2)" group to the endpoints table with `/desk/aurebesh` (and its `?date=`, `?word=`, `?sheet` parameters) and `/desk/test`, cache column `none`; mention the third panel in the opening paragraph.
  - `CLAUDE.md`: add the `src/hud/*.ts` files and `src/pages/desk.ts` to the file structure with one-line descriptions; add an obligations row: "Adding a desk panel page | Draw it with `src/hud/`, never live HTML text; tune sizes in `src/hud/style.ts` and check `/desk/test` on the panel (DECISIONS #66)"; add the two routes to the test endpoints block.
  - `HANDOVER.md`: update the "Describes" line to the new version and commit; add the third panel to section 2 and the routes to section 3; update the test count on the `npm run test:utils` line; replace the "Third panel planned" paragraph in section 15 with what is built and what remains (planet, TARS, Flamengo).
  - The spec: change its Status line to say steps 1 and 2 are built, and tick off the font question in section 9 with the answer.

- [ ] **Step 4: Bump the version (only if approved)** in `package.json` and the `VERSION` constant in `src/index.ts` to `3.17.0`, then run `npm install --package-lock-only` so `package-lock.json` matches.

- [ ] **Step 5: Verify, commit, push**

```bash
npm run typecheck && npm run test:utils 2>&1 | tail -6 && npm run dry-run 2>&1 | tail -5
git add DECISIONS.md README.md CLAUDE.md HANDOVER.md docs/superpowers/specs/2026-10-12-desk-panel-design.md package.json package-lock.json src/index.ts
git status --short
git commit -m "Document the desk panel style kit and Aurebesh page (v3.17.0, #66)"
git push
```

`git status --short` must show nothing left unstaged or untracked; if it does, find out why before committing.

- [ ] **Step 6: Deploy (only if approved) and confirm**

```bash
CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c npx wrangler deploy
curl -s https://eink-dashboard.thiago-oliveira77.workers.dev/health
curl -s -o /dev/null -w "%{http_code} %{content_type} %{size_download}\n" https://eink-dashboard.thiago-oliveira77.workers.dev/desk/aurebesh
curl -s https://eink-dashboard.thiago-oliveira77.workers.dev/health-detailed | head -40
```

Expected: `/health` shows `3.17.0` within about 30 seconds; the page returns `200 text/html`; `/health-detailed` still shows every daily image cached and `ai_budget.blocked: false`, proving the existing panels are untouched.

- [ ] **Step 7: On the panel.** Ask the owner to change the desk panel's page URL to `/desk/aurebesh`, redeploy the pagelist in SenseCraft HMI, and send a photo. Also ask them to compare `/desk/aurebesh?sheet` in a browser with any Aurebesh chart they trust and to name any letter that looks wrong. Fix reported letters in `src/hud/aurebesh.ts`, re-run the tests, commit, and redeploy with approval.

- [ ] **Step 8: Report** with the project's change summary (what changed, why, how to test, docs updated, risks and follow-ups), stating plainly which checks were done by tests, which in a browser, and which only the owner has seen on the panel.
