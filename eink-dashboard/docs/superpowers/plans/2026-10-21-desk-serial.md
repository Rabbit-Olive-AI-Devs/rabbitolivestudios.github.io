# The Desk Serial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `/desk/castaway` into a weekly-written serial: each beat is a page built offline on the Mac mini (Jarvis's painting + text) and served from R2, with the current engine as a routine-only fallback.

**Architecture:** Day schedules and finished pages live in the existing private R2 bucket `eink-series` under `serial/`. The Worker reads today's (and yesterday's) schedule, serves the current beat's stored HTML, and falls back to the engine. Pages are drawn offline by compiling the Worker's own drawing kit (`src/hud/`) in Node, the way `scripts/storybook.mjs` does.

**Tech Stack:** TypeScript (Workers), Node scripts (`.mjs`), wrangler R2, launchd on the Mac mini, the Dropbox series mailbox.

**Spec:** `docs/superpowers/specs/2026-10-21-desk-serial-design.md`

## Phasing

- **Phase 1 (today, first beat 13:30 Chicago, 2026-10-21):** Tasks 1–3, plus a manual run of the writer (background helper, already started) and of the build script once Jarvis's pictures land. This airs today's episode.
- **Phase 2 (following days):** Tasks 4–9 automate it: watcher, weekly writer, review gate, live rebuilds, alerts, story book, docs.

## Global Constraints

- Spoilers: no story text, scene notes, pictures or season plan in the repo, commits, docs, logs, alert text or the main conversation. Private files: `~/desk-serial-work/s01/`; Jarvis's: `Dropbox/00_JARVIS_SPOILERS/desk-serial/<date>/`.
- Worker: no AI, no new KV writes or lists, one request per page (picture inlined as `data:` PNG), no drawing on the request path for serial pages.
- Fonts: real fonts only, nothing under `plex11m`; text and lines pure black and white; the painting in full colour (`setColor`, #81).
- Layout: top bar to row 36 (`TOP_BAR_BOTTOM`), picture band rows 37–351 (800×315), rule at row 352, story text at `TEXT_X=20` within `TEXT_W=548`, "EARLIER" log at `LOG_X=592` with 2 entries.
- Every external read degrades to the engine fallback, never to an error page.
- Build: `npm run typecheck`, `npm run test:utils`, `npm run dry-run` pass before every commit; deploy with `CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c npx wrangler deploy`.

## Review Focus

1. **Before today's first beat** (00:00–13:29 on day 1) there is no beat and no yesterday schedule: expect the engine page, not an error. (Task 2 test.)
2. **Overnight hold:** at 03:00 the latest beat is yesterday's last one. (Task 2 test.)
3. **A schedule that lists a page missing from R2:** expect the engine page. (Task 2 test.)
4. **A picture in the wrong shape** (portrait, tiny, JPEG named .png, transparent PNG): the build crops to fill, flattens alpha, and refuses blank or tiny (<800 px wide) pictures, leaving the beat out of the schedule. (Task 3 test.)
5. **Text that overflows** (a 5th line, an over-wide log line): the build refuses the beat instead of clipping. (Task 3 test.)

---

### Task 1: Serial page layout

**Files:**
- Modify: `src/hud/castaway-page.ts`
- Test: `tests/castaway.test.js`

**Interfaces:**
- Produces: `export interface SerialPageView { day: number; time: string; chapter: string; text: string[]; log: { time: string; text: string }[] }` and `export function drawSerialPage(v: SerialPageView, picture: RGBImage): Canvas` (`RGBImage` from `src/series/art.ts`, exactly 800×315). Also `export function storyTextFits(chapter: string, text: string[], log: {time: string; text: string}[]): string | null` (null when it fits, else the reason).

- [ ] **Step 1: Failing tests** in `tests/castaway.test.js`: `drawSerialPage` with a solid red 800×315 picture gives a canvas whose pixel (400, 200) is red via `colorAt`, whose rows 352–353 are black, and whose top bar draws (some black in rows 0–36); `storyTextFits` returns null for 4 short lines and a reason for 5 lines, for a line wider than `TEXT_W` in `plex15m`, and for a log line wider than `800-LOG_X-16` in `plex12m`.
- [ ] **Step 2:** Run `npm run test:utils`; expect those tests to fail.
- [ ] **Step 3:** Extract the text half of `drawCastawayPage` (chapter, story lines, rule, log) into a private `drawStoryText(c, chapter, lines, log)` used by both pages. `drawSerialPage` draws `topBar(c, ["DESK", CASTAWAY_TITLE], [v.time], \`DAY ${v.day}\`)`, copies the picture into rows 37–351 with `setColor`, then `drawStoryText`. The serial text is pre-wrapped lines (no wrapping at draw time), drawn in `plex15m`.
- [ ] **Step 4:** Run `npm run test:utils`; expect all to pass, and the existing castaway tests unchanged.
- [ ] **Step 5:** Commit `Desk story: a page layout for painted serial beats`.

### Task 2: Serve the current beat from R2

**Files:**
- Create: `src/serial.ts`
- Modify: `src/pages/desk.ts` (`handleDeskCastaway`)
- Test: `tests/serial.test.js`

**Interfaces:**
- Consumes: `env.SERIES` (R2), `chicagoParts` from `src/date-utils.ts` (or what `castaway.ts` uses), `shiftDateStr`.
- Produces: `export interface SerialBeat { id: string; start: string /* "HH:MM" Chicago */; key: string }`, `export interface SerialDay { date: string; day: number; beats: SerialBeat[] }`, `export const serialDayKey = (date: string) => \`serial/days/${date}.json\``, `export const serialPageKey = (id: string) => \`serial/pages/${id}.html\``, `export function currentBeat(today: SerialDay | null, yesterday: SerialDay | null, hhmm: string): SerialBeat | null`, `export async function serialPage(env: Env, now: Date): Promise<string | null>` (HTML or null).

- [ ] **Step 1: Failing tests:** `currentBeat` picks the beat with the latest `start <= hhmm` (beats 13:30/16:00: at 15:59 → 13:30, at 16:00 → 16:00); before the first beat it returns yesterday's last beat; with no yesterday it returns null; unsorted beats still work. `serialPage` with a fake bucket returns the stored HTML; returns null when the day file is missing, malformed JSON, or the page object is missing; never throws when `get` rejects.
- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement. `serialPage` reads today's and, only when needed, yesterday's day file. `handleDeskCastaway` tries `serialPage(env, now)` first (including with `?at=`) and returns `htmlResponse(html, "no-store")`; otherwise the engine as now.
- [ ] **Step 4:** Run tests, typecheck and dry-run; expect pass. Local check: `npx wrangler dev --port 8790`, `curl -s localhost:8790/desk/castaway | grep -c data:image/png` gives 1 (engine path).
- [ ] **Step 5:** Commit `Desk story: serve the current serial beat from R2, engine as fallback`, deploy, then check that the live `/desk/castaway` still returns the engine page (HTTP 200 with one inline PNG).

### Task 3: Offline build and upload

**Files:**
- Create: `scripts/serial-build.mjs`
- Test: `tests/serial-build.test.js`

**Interfaces:**
- Consumes: Task 1's `drawSerialPage`, `storyTextFits`; `cropToFill`, `flattenAlpha`, `isNearlyBlank` (`src/series/art.ts`); `decodePNG` (`src/png-decode.ts`); `canvasToPng`, `hudPageHTML` (`src/hud/page.ts`); Task 2's key helpers.
- Produces: CLI `node scripts/serial-build.mjs <day.json> <pictures-dir> [--upload]`. The day file is `{date, day, beats: [{id, start, title, text: string[], log, scene: {fx, fy}}]}`. Exported `buildBeat(beat, day, pngBytes) -> {html} | {error}` for the test.

- [ ] **Step 1: Failing tests:** `buildBeat` with a 1600×630 PNG gives HTML containing exactly one `data:image/png;base64,`; returns an error for a 5-line text, a blank picture, a picture under 800 px wide, and a non-PNG file; a transparent PNG is flattened, not refused.
- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement. Compile the repo like `storybook.mjs`'s `engine()`. Convert non-PNG input with `sips -s format png` into a temp file first. Focus = `[fx*w, fy*h]`. The log shows the two previous beats of the day (time + their `log` line). With `--upload`, run `npx wrangler r2 object put eink-series/<key> --remote --file …` per page (content-type `text/html`), then upload the day file listing **only the beats that built**, so a late picture means its beat is simply absent and the previous beat holds. Retry each upload once after 10 s (as `series-mailbox.mjs`'s `wrangler()`). Print counts only, never text.
- [ ] **Step 4:** Run tests; then run the CLI on the first delivered picture with `--upload` and check the live page shows a different (serial) image after the beat's start (by response size/hash only, never by looking at it in the main conversation).
- [ ] **Step 5:** Commit `Desk story: offline build and upload of serial beat pages`.

*Phase 1 ends here: today's beats air as Jarvis's pictures arrive; until 13:30 and for any missing picture, the engine page or the previous beat shows.*

### Task 4: Routine-only fallback

**Files:** Modify `src/castaway.ts`, `src/pages/desk.ts`; Test `tests/castaway.test.js`.
**Interfaces:** Produces `export const SERIAL_START = "2026-10-21"` and a `routineOnly` flag on the view builder: on dates `>= SERIAL_START`, `castawayView` and `castawayMoment` use no storyline moments (routine, calendar and live moments only).
- [ ] Failing test: for 2026-10-22, no moment of the day comes from a storyline; for 2026-10-20, unchanged. Implement, pass, commit `Desk story: routine-only engine once the serial runs`.

### Task 5: Mailbox: deliveries build themselves

**Files:** Modify `scripts/series-mailbox.mjs`; Test `tests/series-mailbox.test.js` (or the existing mailbox tests).
**Interfaces:** On each `run`, scan `to-claude/serial/<date>/` for `<beat-id>.png` files whose beat has no built page yet (compare with the day file in R2 or a local `~/desk-serial-work/s01/built.json`), call `serial-build.mjs` for that day, and log `serial <date>: N built, M waiting` (counts only). No `.md` messages are needed from Jarvis.
- [ ] Failing test for the "which beats need building" function; implement; pass; commit `Series mailbox: build serial beats as Jarvis's pictures land`.

### Task 6: The weekly writer job

**Files:** Create `scripts/series-agent/serial-writer.md` (prompt, spoiler rules, format, inputs), modify `scripts/series-mailbox.mjs` (`serial-weekly` command), create `launchd/com.eink.serial-weekly.plist`.
**Interfaces:** Wednesdays 09:00 Chicago, `claude -p` (WebSearch allowed for light inputs) writes the 7 day files for the week starting 12 days later into `~/desk-serial-work/s01/days/`, the briefs into the spoiler folder, a spoiler-free `SERIAL BRIEF` note, and appends to `lines.txt`; it refuses lines already in `lines.txt` or `src/castaway-story.ts`. During the first weeks, it also fills the gap up to that week.
- [ ] Test the line-uniqueness and fit checker as a pure function; add the job; dry-run it once with output to a temp folder; commit.

### Task 7: Review gate and live rebuilds

**Files:** Modify `scripts/serial-build.mjs`, `scripts/series-mailbox.mjs`; prompt `scripts/series-agent/serial-review.md`.
**Interfaces:** A day file gains `"airable": true` only after a background review returns PASS for that day (text fits, pictures non-blank and matching their notes, no repeated line); `serialPage` (Task 2) serves only airable days; update its test. Beats may mark live slots (`{{match}}`, `{{sunset}}`); a nightly 21:30 run (and on match days, every 30 min while the Flamengo board changes) rebuilds the next/current day from `flamengo:v3:state` and `weather:60540:v2` read with `wrangler kv key get --remote` (reads only).
- [ ] Tests for `airable` serving and slot filling; implement; commit.

### Task 8: Alerts and health

**Files:** Modify `src/alert.ts`, the `/health-detailed` handler in `src/index.ts`; Test the alert tests.
**Interfaces:** During a season (today's day file or tomorrow's exists in the plan window), the alert fires when, at its 21:00+ Chicago check, tomorrow's day file is missing in R2, lists fewer than 4 beats, or is not airable; text is spoiler-free ("desk serial: tomorrow has 2 of 5 pages"). `/health-detailed` gains `serial: {today_beats, tomorrow_beats, airable}`.
- [ ] Failing tests; implement; pass; commit.

### Task 9: Story book and docs

**Files:** Modify `scripts/storybook.mjs` (moment = current serial beat id from the R2 day file when one serves, else the engine), `DECISIONS.md` (#96 the desk serial), `README.md` (route behaviour), `HANDOVER.md`, `CLAUDE.md` (file list, quick obligations: serial spoiler rule), memory.
- [ ] Test `needsFrame` with a beat moment; implement; full doc sweep; commit; push.
