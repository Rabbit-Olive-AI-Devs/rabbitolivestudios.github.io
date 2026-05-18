# Color-Weather Yellow Legibility Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all unreadable yellow foreground uses on `/color/weather` (temperature text, battery fill, moon lit surface) on the E1002 Spectra 6 display.

**Architecture:** Three small, independent edits in `src/pages/color-weather.ts`: `tempColor()` becomes a 3-tier blue/black/red scheme, `batteryIcon()` becomes a 2-tier red/green scheme, and the moon's lit surface changes from yellow to white. `tempColor()` and `batteryIcon()` are pure-enough functions; they get exported and unit-tested in the existing `tests/utils.test.js` Node suite. The moon change is a single argument change verified visually.

**Tech Stack:** Cloudflare Workers, TypeScript, Node `node:test` (via `npm run test:utils`), wrangler.

**Branch:** `fix/color-weather-yellow-legibility` (already checked out; the design spec is already committed there).

**Spec:** `docs/superpowers/specs/2026-05-18-color-weather-yellow-legibility-design.md`

---

### Task 1: Temperature color tiers — 4-tier → 3-tier

**Files:**
- Modify: `src/pages/color-weather.ts:129-136` (`tempColor`)
- Test: `tests/utils.test.js`

- [ ] **Step 1: Add the failing test**

In `tests/utils.test.js`, add this import line immediately after the existing `fromBuild(...)` import block (after the `cache-keys.js` destructure that ends around line 22):

```js
const { tempColor, batteryIcon } = fromBuild("src/pages/color-weather.js");
```

Then append this test at the end of the file:

```js
test("tempColor: blue when cold, black when comfortable, red when hot", () => {
  // <= 32F (<= 0C) -> blue
  assert.equal(tempColor(-10), "var(--s6-blue)");
  assert.equal(tempColor(0), "var(--s6-blue)"); // 0C == 32F, boundary -> blue
  // 0C < t <= 30C -> black
  assert.equal(tempColor(1), "#000");
  assert.equal(tempColor(20), "#000");
  assert.equal(tempColor(30), "#000"); // 30C == 86F, boundary -> black
  // > 30C -> red
  assert.equal(tempColor(31), "var(--s6-red)");
  assert.equal(tempColor(40), "var(--s6-red)");
  // yellow and green are never returned as a temperature color
  for (let t = -20; t <= 50; t++) {
    const c = tempColor(t);
    assert.notEqual(c, "var(--s6-yellow)");
    assert.notEqual(c, "var(--s6-green)");
  }
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm run test:utils`
Expected: FAIL — `tempColor` is not exported from `color-weather.ts` yet, so the destructured value is `undefined` and the new test throws `TypeError: tempColor is not a function`.

- [ ] **Step 3: Change `tempColor` to the 3-tier scheme and export it**

In `src/pages/color-weather.ts`, replace the whole function (lines 129-136):

```ts
/** Get temperature color class based on Fahrenheit thresholds. */
function tempColor(tempC: number): string {
  const f = tempC * 9 / 5 + 32;
  if (f <= 32) return "var(--s6-blue)";
  if (f <= 77) return "var(--s6-green)";
  if (f <= 95) return "var(--s6-yellow)";
  return "var(--s6-red)";
}
```

with:

```ts
/**
 * Temperature text color. Only the extremes get a warning color; the
 * comfortable middle is plain black (the most readable foreground on white).
 * Yellow is never used as foreground — see DECISIONS.md #40.
 */
export function tempColor(tempC: number): string {
  const f = tempC * 9 / 5 + 32;
  if (f <= 32) return "var(--s6-blue)"; // cold / freezing
  if (f <= 86) return "#000";           // comfortable — neutral
  return "var(--s6-red)";               // hot
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm run test:utils`
Expected: PASS — the new `tempColor` test passes and all existing tests still pass.

- [ ] **Step 5: Typecheck**

Run: `npm run typecheck`
Expected: no output (success). `export` on `tempColor` does not break any caller — the 4 call sites in the same file are unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/pages/color-weather.ts tests/utils.test.js
git commit -m "$(cat <<'EOF'
Fix unreadable yellow temperature text on /color/weather

Yellow text has almost no contrast on the Spectra 6 white background.
tempColor() now uses a 3-tier blue/black/red scheme: only cold and hot
extremes get a warning color, the comfortable middle stays plain black.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Battery fill color — 3-tier → 2-tier

**Files:**
- Modify: `src/pages/color-weather.ts:111-127` (`batteryIcon`)
- Test: `tests/utils.test.js`

- [ ] **Step 1: Add the failing test**

`batteryIcon` is already added to the import line in Task 1. Append this test at the end of `tests/utils.test.js`:

```js
test("batteryIcon: red at or below 20%, green above, never yellow", () => {
  const red = "rgb(178,19,24)";
  const green = "rgb(18,95,32)";
  const yellow = "rgb(239,222,68)";
  assert.ok(batteryIcon(10, false, 20).includes(red));
  assert.ok(batteryIcon(20, false, 20).includes(red)); // <= 20 -> red
  assert.ok(batteryIcon(21, false, 20).includes(green));
  assert.ok(batteryIcon(50, false, 20).includes(green)); // was yellow
  assert.ok(batteryIcon(90, false, 20).includes(green));
  // a mid-range level no longer renders yellow
  assert.ok(!batteryIcon(35, false, 20).includes(yellow));
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm run test:utils`
Expected: FAIL — `batteryIcon` is not exported from `color-weather.ts` yet, so the destructured value is `undefined` and the new test throws `TypeError: batteryIcon is not a function`. (The yellow-regression assertion in this test only becomes meaningful once the function is exported in Step 3.)

- [ ] **Step 3: Change the fill color to the 2-tier scheme and export the function**

In `src/pages/color-weather.ts`, change the function signature on line 111 from:

```ts
function batteryIcon(level: number, charging: boolean, size: number): string {
```

to:

```ts
export function batteryIcon(level: number, charging: boolean, size: number): string {
```

Then replace line 117:

```ts
  const fillColor = level <= 20 ? "rgb(178,19,24)" : level <= 50 ? "rgb(239,222,68)" : "rgb(18,95,32)";
```

with:

```ts
  // red when low, green otherwise — no yellow mid-tier (DECISIONS.md #40)
  const fillColor = level <= 20 ? "rgb(178,19,24)" : "rgb(18,95,32)";
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm run test:utils`
Expected: PASS — the battery test passes and all earlier tests still pass.

- [ ] **Step 5: Typecheck**

Run: `npm run typecheck`
Expected: no output (success).

- [ ] **Step 6: Commit**

```bash
git add src/pages/color-weather.ts tests/utils.test.js
git commit -m "$(cat <<'EOF'
Drop yellow battery mid-tier on /color/weather

The yellow battery fill is hard to read on the Spectra 6 white
background. batteryIcon() now fills red at/below 20% and green above —
a battery indicator only needs to signal "is it low".

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Moon lit surface — yellow → white

**Files:**
- Modify: `src/pages/color-weather.ts:156-157` (moon phase call inside `renderHTML`)

This is a single-argument change with no unit test — `renderHTML` needs a full `WeatherResponse` object. It is verified visually in Task 5.

- [ ] **Step 1: Change the lit color and update the comment**

In `src/pages/color-weather.ts`, replace lines 156-157:

```ts
  // Moon phase — yellow lit surface on color display
  const moonStr = moonPhaseHTML("rgb(239,222,68)", "#000", 22, now, moonOverride);
```

with:

```ts
  // Moon phase — white lit surface; yellow is unreadable on white (DECISIONS.md #40)
  const moonStr = moonPhaseHTML("#fff", "#000", 22, now, moonOverride);
```

The moon disc keeps its black outline and black shadow path, so the phase shape stays crisp at every phase. This matches the mono E1001 treatment.

- [ ] **Step 2: Typecheck**

Run: `npm run typecheck`
Expected: no output (success).

- [ ] **Step 3: Dry-run build**

Run: `npm run dry-run`
Expected: wrangler reports a successful build (`Total Upload` line, no errors).

- [ ] **Step 4: Commit**

```bash
git add src/pages/color-weather.ts
git commit -m "$(cat <<'EOF'
Use white moon lit surface on /color/weather

The yellow lit surface was faint on the Spectra 6 white background.
White lit + black outline + black shadow matches the mono E1001 moon
and keeps the phase shape crisp.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Version bump and documentation

**Files:**
- Modify: `package.json:3` (`version`)
- Modify: `src/index.ts:53` (`VERSION` constant)
- Modify: `DECISIONS.md` (append new entry #40)
- Verify: `README.md` (no change expected)

- [ ] **Step 1: Bump the version in `package.json`**

In `package.json`, change line 3 from:

```json
  "version": "3.11.2",
```

to:

```json
  "version": "3.11.3",
```

- [ ] **Step 2: Bump the `VERSION` constant in `src/index.ts`**

In `src/index.ts`, change line 53 from:

```ts
const VERSION = "3.11.2";
```

to:

```ts
const VERSION = "3.11.3";
```

- [ ] **Step 3: Append DECISIONS.md entry #40**

Append the following to the end of `DECISIONS.md` (after entry #39, which ends with the "Notes" list):

```markdown

---

## 40. Yellow Is Not a Foreground Color on White (v3.11.3, 2026-05-18)

### Decision: Drop yellow from all foreground uses on /color/weather

**Problem:** On the E1002 Spectra 6 display, `/color/weather` rendered temperature text, the battery fill, and the moon's lit surface in yellow. Against the white page background, yellow has almost no contrast — yellow text and thin yellow strokes are unreadable on the e-ink panel.

**Rule:** Of the Spectra 6 palette (black, white, red, yellow, green, blue), only **black, red, green, and blue** are legible as foreground (text / thin strokes) on a white background. Yellow is usable only as a *background* fill (e.g. the non-severe alert banner, the headlines `regulatory` badge) or inside large dithered image regions — never as foreground on white.

**Changes:**

1. **Temperature colors: 4-tier → 3-tier.** `tempColor()` now returns blue for ≤ 0°C, black (`#000`) for 0–30°C, red for > 30°C. Only the extremes get a warning color; the comfortable middle is plain black (the most readable foreground). The old green and yellow tiers are gone.

2. **Battery fill: 3-tier → 2-tier.** `batteryIcon()` fills red for ≤ 20%, green above. The yellow mid-tier is removed — no Spectra color works as a sensible mid value on white.

3. **Moon lit surface: yellow → white.** The `/color/weather` moon now uses a white lit surface with a black outline and black shadow, matching the mono E1001 treatment. This supersedes the "Color: yellow lit" detail of #34.

**Not changed:** The alert banner (yellow *background*, black text) and the headlines `regulatory` badge (yellow *background*, black text) stay — yellow as a background is fine. The AI image pipelines are unaffected.

**Tests:** `tempColor()` and `batteryIcon()` were exported and given boundary tests in `tests/utils.test.js`.
```

- [ ] **Step 4: Verify README.md needs no change**

Run: `grep -n -iE "tempColor|temperature color|yellow|moon" README.md`
Expected: matches only describe `/color/weather` generically ("Spectra 6 palette accents, moon phase") — no specific temperature-tier or color documentation. If a match documents the old tiers or yellow specifically, update it to match the new scheme; otherwise leave `README.md` unchanged.

- [ ] **Step 5: Typecheck**

Run: `npm run typecheck`
Expected: no output (success).

- [ ] **Step 6: Commit**

```bash
git add package.json src/index.ts DECISIONS.md
git commit -m "$(cat <<'EOF'
Bump to v3.11.3 and document yellow-legibility decision

DECISIONS.md #40 records the rule: yellow is never a foreground color
on the Spectra 6 white background, only a background fill.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

If Step 4 required a `README.md` edit, add `README.md` to the `git add` above.

---

### Task 5: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full build-check suite**

Run: `npm run typecheck && npm run test:utils && npm run dry-run`
Expected: typecheck silent, all Node tests pass, wrangler dry-run reports a successful build.

- [ ] **Step 2: Start the local dev server**

Run: `lsof -ti:8790` (expect no output — port free). Then start the server:
Run: `npx wrangler dev --port 8790` (run in background)

- [ ] **Step 3: Visually verify temperature tiers at 800×480**

Open each URL in a browser sized to exactly 800×480 and confirm the big current-temperature color:
- `http://localhost:8790/color/weather?test-temp=-5` → temperature text is **blue**.
- `http://localhost:8790/color/weather?test-temp=0` → **blue** (0°C boundary).
- `http://localhost:8790/color/weather?test-temp=15` → **black**.
- `http://localhost:8790/color/weather?test-temp=30` → **black** (30°C boundary).
- `http://localhost:8790/color/weather?test-temp=31` → **red**.
Confirm no temperature anywhere on the page renders in yellow or green.

- [ ] **Step 4: Visually verify battery and moon**

- `http://localhost:8790/color/weather?test-device` → battery icon fill is **green** (injected level is 73%).
- `http://localhost:8790/color/weather?test-moon=0` through `?test-moon=7` → each of the 8 moon phases reads clearly: white lit surface, black outline, black shadow. The phase shape is unambiguous at every value.
- Code-review check for low battery: confirm `batteryIcon` returns the red fill for `level <= 20` (covered by the Task 2 unit test; `?test-device` cannot inject a low level).

- [ ] **Step 5: Stop the dev server**

Stop the background `wrangler dev` process. Confirm port 8790 is free again: `lsof -ti:8790` (expect no output).

- [ ] **Step 6: Confirm the done checklist**

Verify every item in the spec's "Done checklist" is satisfied. Report the result.

---

## Notes for the implementer

- All work stays on branch `fix/color-weather-yellow-legibility`. Do not deploy (`npx wrangler deploy`) — deployment is the user's decision after review.
- After the branch is verified, the user decides how to integrate it (merge / PR) — that is outside this plan.
- The Claude auto-memory (`MEMORY.md`, external to this checkout) references DECISIONS #34's "Color: yellow lit" moon detail; that is updated as part of the session wrap-up, not in a repo commit.
