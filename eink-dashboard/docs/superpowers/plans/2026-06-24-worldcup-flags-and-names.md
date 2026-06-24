# World Cup Flags + Full Team Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show full country names (both displays) and Spectra-6 flags (color display only) on the World Cup dashboard, replacing 3-letter codes.

**Architecture:** Flags are pre-rendered offline from flag-icons SVG → Floyd–Steinberg-dithered indexed PNGs (reusing the project's existing color image path) → committed as a base64 map in `src/worldcup-flags.ts`. At runtime, `worldcup-ui.ts` renders full names via a new `teamLabel()` helper and looks flags up through an optional `WcTheme.flag` hook (color theme provides it, mono omits it).

**Tech Stack:** Cloudflare Workers (TypeScript); offline: Node + `@resvg/resvg-js` (SVG raster) + `flag-icons` (source art), both devDependencies.

## Global Constraints

- E-ink 800×480; mono display = pure black only (no flags); color display = Spectra-6 palette only (black/white/red/yellow/green/blue).
- No emoji, no JS in HTML. Flags are inline `<img>` data-URIs (base64 indexed PNG).
- Only emit exact Spectra-6 colors on the color page — flags are pre-dithered to the palette offline so the device renders them as-is.
- Reuse existing `ditherFloydSteinberg` + `encodePNGIndexed` + `pngToBase64`; do not write new image code.
- All escaping via `escapeHTML`. Version bump → 3.14.0. No KV cache-key change (presentation only).

---

### Task 1: Offline flag asset pipeline → `src/worldcup-flags.ts`

**Files:**
- Create: `scripts/generate-flags.mjs`
- Create (generated, committed): `src/worldcup-flags.ts`
- Modify: `package.json` (devDependencies + `flags` script)

**Interfaces:**
- Produces: `export const FLAGS: Record<string, string>` — FIFA TLA (e.g. `"BRA"`) → base64 indexed-PNG string (no `data:` prefix). Missing codes simply absent.

- [ ] **Step 1: Add devDependencies**

```bash
npm i -D @resvg/resvg-js flag-icons
```

- [ ] **Step 2: Confirm flag-icons SVG path + a participant team list**

Flags live at `node_modules/flag-icons/flags/4x3/<iso2>.svg`. Pull the 48 participant TLAs from the live API to size the FIFA→ISO2 table:

```bash
curl -s -H "X-Auth-Token: $FOOTBALL_DATA_KEY" "https://api.football-data.org/v4/competitions/WC/teams" | python3 -c 'import sys,json;[print(t["tla"], t["name"]) for t in json.load(sys.stdin).get("teams",[])]'
```

- [ ] **Step 3: Write `scripts/generate-flags.mjs`**

Compiles the project to the test build dir (same as `test:utils`), then rasterizes + dithers each flag and writes the generated TS. FIFA→ISO2 table covers all 48 participants.

```js
import { execSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { Resvg } from "@resvg/resvg-js";
import path from "node:path";

const BUILD = "/tmp/eink-dashboard-tests";
execSync(`npx tsc -p tsconfig.test.json --outDir ${BUILD}`, { stdio: "inherit" });
const { ditherFloydSteinberg } = await import(path.join(BUILD, "src/dither-spectra6.js"));
const { encodePNGIndexed, pngToBase64 } = await import(path.join(BUILD, "src/png.js"));
const { SPECTRA6_PALETTE } = await import(path.join(BUILD, "src/spectra6.js"));

const W = 36, H = 27;
// FIFA TLA -> ISO-3166 alpha-2 (lowercase) for the 48 participants. Verified against /v4/competitions/WC/teams.
const MAP = {
  BRA:"br", ARG:"ar", FRA:"fr", ESP:"es", ENG:"gb-eng", POR:"pt", NED:"nl", GER:"de",
  BEL:"be", CRO:"hr", URU:"uy", COL:"co", USA:"us", MEX:"mx", CAN:"ca", JPN:"jp",
  KOR:"kr", AUS:"au", IRN:"ir", KSA:"sa", QAT:"qa", MAR:"ma", SEN:"sn", TUN:"tn",
  EGY:"eg", GHA:"gh", CIV:"ci", CMR:"cm", NGA:"ng", RSA:"za", DZA:"dz", SUI:"ch",
  SRB:"rs", POL:"pl", DEN:"dk", AUT:"at", SCO:"gb-sct", WAL:"gb-wls", NOR:"no",
  TUR:"tr", UKR:"ua", ECU:"ec", PER:"pe", PAR:"py", CHI:"cl", CRC:"cr", PAN:"pa",
  HON:"hn", JAM:"jm", NZL:"nz", UZB:"uz", JOR:"jo", CPV:"cv",
};

const out = {};
for (const [tla, iso] of Object.entries(MAP)) {
  const svgPath = `node_modules/flag-icons/flags/4x3/${iso}.svg`;
  if (!existsSync(svgPath)) { console.warn("MISSING svg:", tla, iso); continue; }
  const svg = readFileSync(svgPath, "utf8");
  const img = new Resvg(svg, { fitTo: { mode: "width", value: W } }).render();
  const rgba = img.pixels; // RGBA, img.width x img.height
  const iw = img.width, ih = img.height;
  const rgb = new Uint8Array(iw * ih * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    // composite over white (flags have no transparency, but be safe)
    const a = rgba[i + 3] / 255;
    rgb[j]   = Math.round(rgba[i]   * a + 255 * (1 - a));
    rgb[j+1] = Math.round(rgba[i+1] * a + 255 * (1 - a));
    rgb[j+2] = Math.round(rgba[i+2] * a + 255 * (1 - a));
  }
  const indices = ditherFloydSteinberg(rgb, iw, ih, SPECTRA6_PALETTE);
  const png = await encodePNGIndexed(indices, iw, ih, SPECTRA6_PALETTE);
  out[tla] = `${iw}x${ih}|` + pngToBase64(png);
}

// Store width/height alongside base64 so the renderer can set intrinsic size.
const body = Object.entries(out).map(([k, v]) => `  ${k}: ${JSON.stringify(v)},`).join("\n");
writeFileSync("src/worldcup-flags.ts",
`// GENERATED by scripts/generate-flags.mjs — do not edit by hand.
// FIFA TLA -> "WxH|<base64 indexed PNG>". Spectra-6 dithered, offline.
export const FLAGS: Record<string, string> = {\n${body}\n};\n`);
console.log("Wrote", Object.keys(out).length, "flags");
```

- [ ] **Step 4: Add `flags` npm script**

In `package.json` scripts: `"flags": "node scripts/generate-flags.mjs"`.

- [ ] **Step 5: Generate + sanity-check**

Run: `FOOTBALL_DATA_KEY=… npm run flags` (key only needed if re-pulling the team list in Step 2; generation itself needs no key).
Expected: `Wrote 48 flags`, `src/worldcup-flags.ts` created, no `MISSING` warnings. If any MISSING, fix that TLA's ISO2 in `MAP`.

- [ ] **Step 6: Commit**

```bash
git add package.json package-lock.json scripts/generate-flags.mjs src/worldcup-flags.ts
git commit -m "Add offline flag-icons -> Spectra-6 generator + generated FLAGS map"
```

---

### Task 2: `teamLabel()` truncation helper

**Files:**
- Modify: `src/worldcup-ui.ts`
- Test: `tests/worldcup.test.js`

**Interfaces:**
- Produces: `export function teamLabel(team: { name: string; code: string }, maxChars: number): string` — escaped full `team.name`; if `name.length > maxChars`, truncates to `maxChars - 1` chars + `…` (then escaped). Empty/whitespace name → `"TBD"`. (`teamCode` stays, still used by `isFav`.)

- [ ] **Step 1: Write the failing test**

```js
const { teamLabel } = fromBuild("src/worldcup-ui.js");

test("teamLabel returns full name within budget", () => {
  assert.equal(teamLabel({ name: "Brazil", code: "BRA" }, 14), "Brazil");
});
test("teamLabel truncates with ellipsis past budget", () => {
  assert.equal(teamLabel({ name: "United States", code: "USA" }, 9), "United S…");
});
test("teamLabel returns TBD for empty name", () => {
  assert.equal(teamLabel({ name: "", code: "" }, 14), "TBD");
});
test("teamLabel escapes HTML", () => {
  assert.equal(teamLabel({ name: "A&B", code: "AB" }, 14), "A&amp;B");
});
```

- [ ] **Step 2: Run, verify it fails** — `npm run test:utils` → FAIL (`teamLabel is not a function`).

- [ ] **Step 3: Implement in `src/worldcup-ui.ts`** (near `teamCode`)

```ts
/** Full team name, escaped, truncated to maxChars with an ellipsis. "TBD" if empty. */
export function teamLabel(team: { name: string; code: string }, maxChars: number): string {
  const name = (team.name ?? "").trim();
  if (!name) return "TBD";
  const shown = name.length > maxChars ? name.slice(0, Math.max(1, maxChars - 1)) + "…" : name;
  return escapeHTML(shown);
}
```

- [ ] **Step 4: Run, verify pass** — `npm run test:utils` → all pass.

- [ ] **Step 5: Commit** — `git commit -am "Add teamLabel() full-name truncation helper"`

---

### Task 3: Flag theme hook + render full names & flags in every layout

**Files:**
- Modify: `src/worldcup-ui.ts` (WcTheme, render fns, CSS)
- Modify: `src/pages/color-worldcup.ts` (provide `flag` hook)
- Modify: `src/pages/worldcup.ts` (mono: no flag hook — verify only)

**Interfaces:**
- Consumes: `FLAGS` (Task 1), `teamLabel` (Task 2).
- Produces: `WcTheme.flag?: (code: string) => string`.

- [ ] **Step 1: Extend `WcTheme`** — add `flag?: (code: string) => string;` to the interface.

- [ ] **Step 2: Add a flag helper + swap codes→names in render fns.** In `worldcup-ui.ts`:

```ts
const flagImg = (theme: WcTheme, code: string) => (theme.flag ? theme.flag(code) : "");
```

`matchRow` (budget 14): replace `${escapeHTML(h)}` / `${escapeHTML(a)}` with `${flagImg(theme, teamCode(mm.home))}${teamLabel(mm.home, 14)}` (and away). Keep the `isFav(teamCode(...))` accent + `▶` star logic on the **code**.

`groupTable` (budget 12): team cell becomes `${flagImg(theme, code)}${teamLabel(r.team, 12)}${mark}` where `code = teamCode(r.team)`.

`bracketColumn` (budget 9): `side()` becomes `<div class="wc-bteam" ...>${flagImg(theme, code)}${teamLabel(team, 9)}</div>` — pass the full `team` object and `code` into `side`.

`championLayout`: big flag above the name — `${theme.flag ? theme.flag(teamCode(data.champion)).replace('wc-flag','wc-flag-big') : ""}` then `teamLabel(data.champion, 20)`; final line uses `teamLabel(final.home,20)` / `teamLabel(final.away,20)`.

- [ ] **Step 3: Add `.wc-flag` CSS** to the shared `<style>` in `renderWorldCupHTML`:

```css
.wc-flag { height: 13px; width: auto; vertical-align: middle; margin-right: 6px; border: 1px solid #000; }
.wc-flag-big { height: 110px; width: auto; border: 2px solid #000; image-rendering: pixelated; }
.wc-bteam .wc-flag { height: 11px; margin-right: 4px; }
```

- [ ] **Step 4: Provide the flag hook in `color-worldcup.ts`.** Import `FLAGS`; add to `COLOR_THEME`:

```ts
import { FLAGS } from "../worldcup-flags";
// in COLOR_THEME:
flag: (code: string) => {
  const v = FLAGS[code];
  if (!v) return "";
  const [dim, b64] = v.split("|");
  const [w, h] = dim.split("x");
  return `<img class="wc-flag" width="${w}" height="${h}" alt="" src="data:image/png;base64,${b64}">`;
},
```

Mono `MONO_THEME` in `worldcup.ts`: leave `flag` unset (text only).

- [ ] **Step 5: typecheck + render check**

Run: `npm run typecheck` → clean.
Run locally (wrangler dev) and verify `/color/worldcup` rows show flag + full name, `/worldcup` shows full name only, both still HTTP 200 across `?test-phase=group|r32|knockout|champion`.

- [ ] **Step 6: Commit** — `git commit -am "Render full team names + color-display flags in all WC layouts"`

---

### Task 4: FLAGS coverage test, version bump, docs, deploy

**Files:**
- Test: `tests/worldcup.test.js`
- Modify: `package.json`, `src/index.ts`, `README.md`, `DECISIONS.md`, external `MEMORY.md`

- [ ] **Step 1: Coverage test** — every team in canned `testWorldCupData("group")` whose code is non-empty has a `FLAGS` entry:

```js
const { FLAGS } = fromBuild("src/worldcup-flags.js");
test("FLAGS map covers favorite + is non-empty", () => {
  assert.ok(Object.keys(FLAGS).length >= 40);
  assert.ok(FLAGS["BRA"] && FLAGS["BRA"].includes("|"));
});
```

- [ ] **Step 2: Run full check** — `npm run typecheck && npm run test:utils && npm run dry-run` → all clean.
- [ ] **Step 3: Version bump** — `package.json` 3.13.0 → 3.14.0; `src/index.ts` `VERSION = "3.14.0"`.
- [ ] **Step 4: Docs** — README (World Cup section: full names both displays, flags color-only, offline flag pipeline + `npm run flags`); DECISIONS #45 (flags+names rationale, Option A, mono text-only); MEMORY NEXT SESSION update.
- [ ] **Step 5: Commit** — `git commit -am "World Cup full names + flags (v3.14.0): docs + version"`
- [ ] **Step 6: Deploy** — `npx wrangler deploy` (CLOUDFLARE_API_TOKEN env), verify `/health` = 3.14.0 and `/color/worldcup` HTTP 200.

---

## Self-Review

- **Spec coverage:** §3 pipeline → Task 1; §4 teamLabel+flag hook → Tasks 2–3; §5 layouts → Task 3; §6 edge cases (missing flag → `""`, empty name → `"TBD"`) → Tasks 1–2; §7 tests → Tasks 2,4; §8 files → all; §9 version → Task 4. Covered.
- **Placeholders:** none — all steps have concrete code/commands.
- **Type consistency:** `teamLabel(team, maxChars)`, `WcTheme.flag?(code)`, `FLAGS: Record<string,string>` used identically across tasks. Flag value format `"WxH|b64"` defined in Task 1, parsed in Task 3 Step 4.
