# Claude Code — E-Ink Dashboard Project Guidelines

**Purpose**: Persistent instructions for Claude Code sessions on the E-Ink Dashboard. Read this file end-to-end at the start of every session before doing any work.

**Default behavior**: Prefer small, reviewable changes. Test visually before deploying. Strong documentation over large refactors.

**Critical**: Claude Code auto-memory (`MEMORY.md`) may exist outside this checkout. If it is present/auto-loaded, it defines current technical truth. If absent, prefer `DECISIONS.md`, then `README.md`, then source code and git history.

---

## Start of Session Checklist (mandatory)

Every session must begin with these steps:

1. Read this file (`CLAUDE.md`) end-to-end
2. Read `MEMORY.md` if present/auto-loaded — authoritative technical snapshot for Claude sessions
3. Read `DECISIONS.md` — understand standing decisions and failed approaches
4. Read `README.md` — understand current endpoints, pipelines, architecture
5. Run `git log --oneline -10` — understand recent changes
6. Run `git status` — check for uncommitted work
7. Check `package.json` version — know what version is current
8. Summarize the current project state in 3-5 bullets before proceeding
9. Ask the user what they want to work on
10. Confirm scope and define a "done checklist" before writing any code

---

## Quick Obligations

| Situation | Required action |
|-----------|----------------|
| Starting a session | Run the full Start of Session Checklist above |
| Before writing code | Read the files you plan to modify. Never propose changes to code you haven't read |
| After completing a feature or fix | Update all documentation in the same commit (see Documentation section) |
| Before deploying | `npx wrangler deploy --dry-run` to verify the build succeeds |
| Before committing | Verify build succeeds. Do not commit broken code |
| Ending a session | Update docs, commit and push, stop dev server if running |
| Ideas or features discussed but not implemented | Note in commit message or DECISIONS.md |
| Changing any pipeline or cache behavior | Bump the relevant cache key version |
| Bumping a cache key version | Safe for the stale-image fallback — it resolves keys by KV prefix, not by rebuilding them (DECISIONS #58). Never reintroduce a fallback that rebuilds keys from `*_CACHE_VERSION` |
| Adding an HTML page whose body is an `<img>` | Give it a text fallback so a failed image never renders as a broken-image glyph on the panel (DECISIONS #58) |
| Any image pipeline change | Never pass raw bytes to `env.IMAGES.input()` — it throws `undefined (reading 'font')` and burns the AI call before failing (DECISIONS #59) |
| An image cache looks empty | Check whether *writes* are failing (`wrangler kv key list --remote …`) before assuming reads are racing (DECISIONS #59). Without `--remote`, wrangler 4 reports the empty local store |
| Diagnosing what a panel actually received | `wrangler tail --format json` shows the renderer's requests (UA `HeadlessChrome`, from Azure US West via Cloudflare SJC). An `outcome: canceled` with no log line is the client hanging up — on SenseCraft that is a white frame. The tail can go silent while alive; `/health-detailed` cache ages are ground truth (DECISIONS #65) |
| Touching the series (`src/series/`, `/series` routes) | **Never commit season scripts, pictures, packages or build output**: the repo is public and they are spoilers; they live in Dropbox, `~/series-work/` and R2. Pages are built offline by `npm run series -- build`; never convert images at request time. The writer's picture goes to every panel untouched in full colour and each panel converts it itself: do not add a dithering step back without the owner seeing it on the panels (DECISIONS #73). A chapter airs only after `npm run series -- approve` (Jarvis approves; the owner watches unspoiled, so never show them a scene). DECISIONS #71 |
| A session starts | Read `from-claude/NEEDS-THIAGO.md` and `~/Library/Logs/eink-series/mailbox.log` in the Dropbox series mailbox: the watcher (DECISIONS #72) handles routine messages itself and flags only what needs a session |
| Touching World Cup code | It is **retired but preserved** for 2030 — unwired, not deleted. Keep the files and their tests green; see DECISIONS #61 before removing anything |
| Adding any per-request KV write or `list` | Cost it against the free tier's **1,000 writes/day** — a normal day already uses ~590 (DECISIONS #62). Reads are effectively free; writes and lists are not |
| Adding a new failure mode | Ask whether the alert check in `src/alert.ts` would catch it. Graceful degradation without alerting is indistinguishable from working (DECISIONS #60) |
| Adding visual changes to weather/fact pages | Test in browser at 800x480 before deploying, **including with an alert banner** (`?test-alert=tornado`) — that state has ~34px less room (DECISIONS #63) |
| Adding an AI image route, or anything slow, to a page the panel loads | Never generate on the request path when something older exists: serve a stale image and generate under `ctx.waitUntil` (`serveStaleWhileGenerating`, DECISIONS #64) |
| Adding or changing any page a panel loads | It must be **one request**: inline images as `data:` URIs (`renderSkylineInlineHTML`, `/color/moment`). SenseCraft screenshots ~1s after the document arrives and cancels sub-resources still in flight — a separate `<img src="/skyline.png">` went white on the E1002 even with a warm cache (DECISIONS #65). The document itself is waited for; a sub-resource is not |
| Touching the daily cron or `dailyWarmTargetDate` | The warm must fill the **next** Chicago date before it rolls (`35 4,5 * * *`). Warming "right after midnight" reopens the hole that blanked the skyline (DECISIONS #64) |
| Changing how a panel picture is converted (`/fact`, `/color/moment`, `/skyline*`, birthdays) | Since 2026-10-15 every one goes to the panel **as-is**: the AI image cropped to 800x480 in full colour, untouched, caption burned in as solid black and white (`burnCaption` in `src/image.ts`); the panel converts it (DECISIONS #81, after the series #73). Do not add our own quantise/dither step back without the owner seeing both on the panels. A page that inlines such a picture must stay one request (~0.6-1 MB PNG) |
| Adding or editing an art style or a scene prompt | The panels handle any style (owner, 2026-10-16). Do not cull styles or add palette, simplicity, contrast or composition limits "for the display"; every style is enabled (DECISIONS #84). Styles live in one shared library, `src/art-styles.ts`: add a style there and tag what it suits (#86). Pipeline B's 1-bit styles are the exception: its own conversion needs them |
| Spending AI on test generations (portraits, styles) | Read real usage from GraphQL `aiInferenceAdaptiveGroups` first, and until ~06:00 UTC **add the previous UTC day's total**: Cloudflare's daily reset lags, and a refusal arms the budget block for the whole day, which can cost the nightly warm (DECISIONS #88). Leave room for the warm (~2,000-2,800) |
| Touching the daily pictures, `src/art.ts` or `/art/brief` | Jarvis draws them from the brief, and Workers AI only fills what he did not deliver (DECISIONS #89). Keep the brief built from the Worker's own choices (event, `momentStyleFor`, skyline picks, `getBirthdayStyle`), so both sources agree. The warm takes R2 `art/<date>/<kind>` first. A refused event pick is never cached; the brief then lets Jarvis choose the event (`chooseEvent`), and his `title/year/place` lines become the caption. The alert emails when Jarvis missed a night (markers `art/used/<date>/<kind>`) or the Mac mini stops reporting to TARS (#90) |
| Adding or changing a holiday week | One entry in `src/holidays.ts` (DECISIONS #91): ASCII name (the skyline caption font), skyline dressing without snow (southern cities), a documented history event for the day itself. The test keeps weeks from overlapping. A theme change only affects dates not yet cached, so no cache-key bump is needed unless non-holiday prompts change |
| Touching Family Memory (`src/family.ts`, `/color/family`, `scripts/family-pick.py`) | The photo comes from the owner's album, then Favourites, then the album's closest day: wide only, at least 11 months old, once per round, no names on the panel (DECISIONS #92). Never write the album's share link anywhere; never commit photos, the shown log or `people.json`. Copy Photos.sqlite to the external drive, not the Mac's disk (short of space). No Workers AI: Jarvis or the photo itself |
| Refreshing the birthday reference photos | `scripts/pick-portraits-from-photos.py` on a read-only copy of the Photos database → review the crops' contact sheet → `npm run upload-photos` (needs `hints.json` so group photos crop the right face). Never commit photos or the people/hints files (DECISIONS #87) |
| Drawing text or bars into an image that will be Floyd–Steinberg dithered | Draw them after dithering or map them by nearest colour — diffused error turns a black bar into speckle (DECISIONS #64) |
| Adding anything to `/weather` or `/color/weather` | They carry only 5-7px of vertical slack and `.hourly` absorbs every deficit. `body.scrollHeight` reports 480 even when cards are clipped — measure the **deepest rendered element** instead (DECISIONS #63) |
| Adding or changing a desk panel page (`/desk/*`) | Draw it with `src/hud/` as an image, never with live HTML text: text and lines pure black and white, pictures (crest, planet) in full colour the panel converts (`setColor`, DECISIONS #81), so the same page suits the mono and the colour panels; a page with no colour stays a 1-bit PNG. **Text is real fonts only** (`drawType` with a face from `fonts-data.ts`); never the 8x8 bitmap font or hand-drawn letters, and **never smaller than `plex11m`**, the owner's minimum on the panel (DECISIONS #67). Need a new size? Add a face in `scripts/generate-desk-fonts.mjs` and run `npm run desk-fonts`. Show the owner a mockup before building a new look. Keep them free of AI, and of KV unless the page truly needs outside data: then one write a day, warmed by the cron, with an older value and a built-in fallback (`src/planet.ts`, DECISIONS #68). The Flamengo board is the costed exception: one entry refreshed by age, 4 to 12 writes a day (`src/flamengo.ts`, DECISIONS #70). Club crests are fetched at run time and never committed |

---

## Project Overview

- **Project**: E-Ink "Moment Before" Dashboard
- **Tech**: Cloudflare Workers (TypeScript)
- **Repo**: `Rabbit-Olive-AI-Devs/rabbitolivestudios.github.io` (moved from `rabbitolivestudios/…` on 2026-09-10; the old URL redirects)
- **Subdirectory**: `eink-dashboard/`
- **Live URL**: `https://eink-dashboard.thiago-oliveira77.workers.dev`
- **Displays**: reTerminal E1001 (mono, 7.5" ePaper, 800x480) + reTerminal E1002 (Spectra 6 color, 7.3", 800x480)
- **Third display (being set up)**: a second reTerminal E1001 (mono) on the home-office desk, showing the `/desk/*` pages (DECISIONS #66)
- **Device manager**: SenseCraft HMI (screenshots URLs; since v1.3.2 a URL is added through the HTML widget in the Canvas Editor at 800x480, formerly the "Web Function" page type)

---

## Build, Test, Deploy

```bash
# Build check (always run before committing)
npm run typecheck
npm run test:utils
npm run dry-run

# Local dev server (check port 8787/8790 availability first)
lsof -ti:8790
npx wrangler dev --port 8790

# NOTE: `wrangler dev --remote` under the REAL worker name inherits deployed secrets,
# so TEST_AUTH_KEY gates you out of the test routes (404). To exercise them against real
# bindings, run under a different name (-c a copy of wrangler.toml with `name` changed)
# and supply any needed values via .dev.vars (gitignored). Plain `wrangler dev` runs
# env.IMAGES in LOCAL mode, which does not reproduce edge Images faults (DECISIONS #59).

# Deploy to production
npx wrangler deploy

# Production KV and logs. wrangler 4 kv commands read the LOCAL store unless --remote
# is given; `wrangler tail` can stop delivering a few minutes after attaching while the
# process stays alive — re-attach in rounds and kill the node child directly (DECISIONS #65).
npx wrangler kv key list --remote --namespace-id de97776d35af4df08b13fd2158acebdc --prefix "skyline:v5:"
npx wrangler tail eink-dashboard --format json
pkill -f "bin/wrangler tail"

# Test endpoints (E1001) — no key needed in local dev
curl http://localhost:8790/weather?test-device
curl http://localhost:8790/weather?test-device&test-alert=tornado
curl http://localhost:8790/fact.png
curl "http://localhost:8790/test.png?m=10&d=31"
# In production (TEST_AUTH_KEY set): curl "https://URL/test.png?m=10&d=31&key=YOUR_KEY"

# Test endpoints (E1002 color)
curl http://localhost:8790/color/weather?test-device
curl http://localhost:8790/color/weather?test-device&test-alert=tornado
curl http://localhost:8790/color/headlines?test-headlines
curl "http://localhost:8790/color/test-moment?m=7&d=20"
curl http://localhost:8790/skyline
curl "http://localhost:8790/alert-test"          # run the failure check
curl "http://localhost:8790/alert-test?force=1"  # send a test alert email
curl "http://localhost:8790/skyline-test?date=2026-06-01"

# Test endpoints (desk panel) — free, no auth, no AI
curl http://localhost:8790/desk/test
curl "http://localhost:8790/desk/aurebesh?word=JEDI&date=2026-10-12"
curl "http://localhost:8790/desk/aurebesh?sheet"
curl "http://localhost:8790/desk/planet?date=2026-10-20"   # live NASA lookup, no cache write
curl http://localhost:8790/desk/flamengo                    # live ESPN lookup; writes the local KV entry
curl "http://localhost:8790/desk/series?at=2026-10-26T10:30" # series preview (needs the local R2 to hold pages)
npm run series -- build ~/series-work/<package> --out ~/series-work/out   # offline import; see docs/series/
# TARS locally: start dev with a throwaway key, then run the real reporter against it
#   npx wrangler dev --port 8790 --var TARS_KEY:localtestkey
#   TARS_URL=http://localhost:8790/desk/tars TARS_KEY=localtestkey ./scripts/tars-report.sh mini "Mac mini"
curl "http://localhost:8790/desk/tars?key=localtestkey"
```

Always build before committing. Do not commit code that doesn't compile.

---

## Image Pipelines — DO NOT Cross-Contaminate

This project has independent image pipelines. They share the LLM event selection (via `getOrGenerateMoment`) but diverge at style injection, image model, and post-processing. The one deliberate exception: the colour moment (D), skyline (E) and birthday portrait (C) pick styles from the shared library `src/art-styles.ts` (DECISIONS #86); their prompts, models and post-processing stay separate. Changes to one pipeline must not affect the others.

| | Pipeline A (`/fact.png`) | Pipeline B (`/fact1.png`) | Pipeline D (`/color/moment`) | Pipeline E (`/skyline`) |
|---|---|---|---|---|
| Model | FLUX.2 klein-9b | SDXL | FLUX.2 (fallback SDXL) | FLUX.2 + ref photo (fallback SDXL) |
| Style | Daily rotation (Woodcut/Pencil/Charcoal) | 6-style rotation (style-aware) | shared library `src/art-styles.ts`: 70 styles, random order, no repeat until all used (#86) | same library: 66 styles, never the moment's style that day (#86) |
| Output | Full colour as-is (panel converts, #81) | 1-bit (Bayer or threshold) | Full colour as-is (panel converts, #81) | Full colour as-is, both modes (#81) |
| Cache key | `fact4:v5:YYYY-MM-DD` | `fact1:v7:YYYY-MM-DD` | `color-moment:v4:YYYY-MM-DD:STYLE_ID` | default `skyline:v5:DATE:daily[:bw]`; rotate `skyline:v5:DATE:rN:bBUCKET[:bw]` |
| Display | none — on request only, not warmed (#84) | none — on request only, not warmed (#83) | E1002 + home E1001 (#84) | E1002 + home E1001 (`/skyline-bw` not warmed, #84) |

---

## E-Ink Display Constraints

These constraints are non-negotiable for any UI changes:

- **800x480px** — no scrolling, everything must fit
- **Pure black (#000) only** — grays are invisible on e-ink
- **No emoji** — ESP32-S3 renderer lacks emoji font support, use inline SVG
- **No JavaScript** — SenseCraft HMI screenshots static HTML
- **Test at 800x480** in browser before deploying visual changes
- **Test all alert states** — normal, tornado, winter, rain warning
- **Always bump cache key version** after pipeline changes

---

## Definition of Done

Before coding any task, write a short checklist:
- What must work
- What must be tested (browser at 800x480 + live device if applicable)
- What docs must be updated
- How success is verified
- Edge cases to consider (e.g., alert banners, no device data, API failures)

At the end, explicitly confirm each item.

---

## Documentation Requirements

Change logging is mandatory. Every meaningful change must update documentation **in the same commit**.

### Documentation Files

| File | Purpose | Update when |
|------|---------|-------------|
| `MEMORY.md` | Claude Code auto-memory when available; may be external to this checkout | Architecture changes, new patterns, critical bugs |
| `DECISIONS.md` | Why things are the way they are, failed approaches | Any tradeoff, threshold change, or pipeline change |
| `README.md` | User-facing: endpoints, pipelines, architecture, setup | New endpoints, changed behavior, version bumps |
| `HANDOVER.md` | One-file briefing for an agent starting cold: mental model, rules, budgets, process, runbook | Anything it states changes — routes, crons, cache keys, budgets, rules, open items. Update its "Describes" line to the new commit/version |
| `AGENTS.md` | Short entry point for agents that auto-load it; points at `HANDOVER.md` and `CLAUDE.md` | Only if the pointers or the handful of hard rules it repeats change |
| `package.json` | Version number | Feature releases |
| `src/index.ts` | `VERSION` constant | Feature releases |

### Mandatory Documentation Sweep

Every time you update documentation, sweep ALL files:

| # | File | Check |
|---|------|-------|
| 1 | `MEMORY.md` | If present, key architecture, cache keys, learnings — all accurate? |
| 2 | `DECISIONS.md` | Any new or changed decisions documented? |
| 3 | `README.md` | Endpoints table, architecture diagram, version — all accurate? |
| 4 | `HANDOVER.md` | "Describes" line, routes, crons, cache keys, budgets, test count, drift/open-items list — all still true? |
| 5 | `package.json` + `src/index.ts` | Version numbers match? |

Do not commit documentation updates until you have verified every file.

---

## Code Standards

- **Think before coding.** Follow this order:
  1. Think about the architecture
  2. Read the existing codebase
  3. Check `DECISIONS.md` and `MEMORY.md` if present for past learnings
  4. Implement, or ask about tradeoffs
- **Fix from first principles.** Don't apply bandaids. Find the root cause.
- **Keep it simple.** Write idiomatic TypeScript. Always ask: is this the simplest solution?
- **No dead code.** Delete unused functions, parameters, and files.
- **Avoid unnecessary changes.** Don't refactor or "improve" code that wasn't part of the task.
- **Do not mix refactors with features.** Separate commits.
- **Follow existing patterns.** `alerts.ts` is the template for cached API fetches. `image.ts` is the template for image pipelines.
- **Graceful degradation.** External API failures should return `null` or `[]`, never crash the page.

### TypeScript/Workers Specific

- Use `Env` type from `src/types.ts` for all bindings
- Always coerce LLM responses: `typeof raw === "string" ? raw : JSON.stringify(raw)`
- Chunk large arrays for `String.fromCharCode` (8192-byte slices) to avoid stack overflow
- SDXL does NOT support `negative_prompt` — embed negatives in positive prompt
- FLUX.2 requires multipart FormData, not JSON
- `env.IMAGES.input()` requires a **ReadableStream** — pass `bytesToImageStream()` from `src/images-input.ts`, never a `Uint8Array`/`ArrayBuffer`/`Blob` (DECISIONS.md #59)
- KV cache dates use America/Chicago timezone
- All external fetches must use `fetchWithTimeout()` from `src/fetch-timeout.ts`
- KV free tier: **1,000 writes/day, 1,000 lists/day**, 100,000 reads/day. Baseline is already ~590 writes — see DECISIONS #62 before adding a write path
- All KV `.put()` calls must include `expirationTtl` (86400 for ephemeral, 604800 for daily) — see DECISIONS.md #24 for why ephemeral must be >>soft TTL

---

## Testing Expectations

For visual changes (weather page, fact page):
- Test in browser at exactly 800x480 viewport
- Test all states: with/without device data, with/without alerts, with/without rain warning
- Test params: `?test-device`, `?test-alert=tornado|winter|flood`, `?test-rain`, `?test-temp=N`

For image pipeline changes:
- Use `/test.png?m=MM&d=DD` and `/test1.png?m=MM&d=DD` to test arbitrary dates
- Use `/test-birthday.png?name=KEY&style=N` for birthday portraits
- Verify both pipelines independently

For API/cache changes:
- Test with fresh dates to bypass KV cache
- Remember local KV persists between `wrangler dev` restarts

---

## Git Practices

- **Commit messages**: Short summary line, blank line, body explaining why (not what). End with the `Co-Authored-By:` and `Claude-Session:` attribution lines the current session provides (the model name changes between sessions; do not hardcode one)
- **Commit often**: One logical change per commit. Don't batch unrelated changes.
- **Never force push** to main.
- **Push after committing** unless told otherwise.
- **Version bumps**: Only when the user explicitly requests or approves.
- **Do not delete or rewrite history** in documentation files.

---

## Change Summary (mandatory output)

After completing any work, output a change summary:

```
## Change Summary
- **What changed**: (files and what was modified)
- **Why**: (reason for the change)
- **How to test**: (exact steps to verify)
- **Docs updated**: (list of docs updated)
- **Risks / follow-ups**: (anything that needs attention)
```

---

## Hard "Do Not" List

- Do not claim something is implemented unless it builds and works.
- Do not delete or rewrite history in documentation.
- Do not mix large refactors with feature work in one commit.
- Do not propose changes to code you haven't read.
- Do not commit code that doesn't compile (`--dry-run` first).
- Do not skip documentation updates.
- Do not cross-contaminate Pipeline A and Pipeline B.
- Do not deploy without testing visual changes at 800x480.
- Do not use grays, emoji, or JavaScript in HTML pages (e-ink constraints).
- Do not forget to bump cache key versions after pipeline changes.
- Do not leave dev servers running at end of session.

---

## Credential & Secret Protection

The SenseCraft API key is intentionally hardcoded (shared platform key, single-device personal project — documented in `DECISIONS.md`). For any other credentials:

1. **REFUSE credentials.** If the user shares passwords, private keys, or auth tokens, STOP and warn them.
2. **NEVER write credentials to any file** — not code, not documentation, not commit messages.
3. **NEVER commit credential files** (`.env`, `*.key`, `*.pem`, `*.secret`).

---

## File Structure

```
eink-dashboard/
  src/
    index.ts              — Main router, cron handler, VERSION constant
    types.ts              — All TypeScript interfaces
    date-utils.ts         — Chicago timezone date helpers, shiftDateStr(), dailyWarmTargetDate() (#64)
    weather.ts            — Open-Meteo weather fetch + KV cache
    weather-codes.ts      — WMO code → label/icon mapping, day/night overrides
    alerts.ts             — NWS weather alerts fetch + KV cache
    device.ts             — SenseCraft device data fetch + KV cache
    fact.ts               — Wikipedia "On This Day" fetch + KV cache
    moment.ts             — LLM event selection + scene prompt generation
    image.ts              — Pipeline A (FLUX.2, full colour as-is, #81) + Pipeline B (SDXL 1-bit); burnCaption() lays a caption over a colour picture in solid black/white
    image-color.ts        — jpegToRGB(): AI image → 800x480 full-colour RGB via Cloudflare Images; the SDXL/FLUX generate-and-decode helpers (#81)
    convert-1bit.ts       — 1-bit conversion engine (Bayer + threshold)
    styles-1bit.ts        — Pipeline B style table, picker, hash
    art.ts                — Jarvis's daily pictures: the brief (`/art/brief`), and the warm taking his R2 deliveries before Workers AI (DECISIONS #89)
    art-styles.ts         — The shared art-style library (71 styles, each tagged moment/skyline/portrait) + seeded daily pick (DECISIONS #86)
    family.ts             — Family Memory: the brief's family picture, the warm's store (Jarvis's repainting or the photo itself), caption (DECISIONS #92)
    holidays.ts           — Holiday weeks: the 7 days up to each holiday dress the skyline and add a mood to the moment; on the day the moment is the holiday's history (DECISIONS #91)
    birthday.ts           — Birthday data + detection
    tars.ts               — Desk panel TARS: cleans and merges the Macs' reports into KV `tars:v1:state`, builds the page's view (DECISIONS #69)
    flamengo.ts           — Desk panel Flamengo board: ESPN's public feeds (table, results, fixtures, crests) + FotMob's legend for the zone lines, in KV `flamengo:v3:state`, refreshed by age; builds the page's view (DECISIONS #70)
    series/               — The series (DECISIONS #71): pure TS used by the offline import and the routes
      types.ts            — package, schedule and slot types; plannedChapters()
      clock.ts            — Chicago story clock → slot; resolveSlot() for unapproved chapters; scene time labels
      keys.ts             — R2 key of every page and card; slotKey()
      templates.ts        — the five page templates and the strip
      art.ts              — flatten alpha, crop-to-fill around a focus point, blank check
      letter.ts           — captions and balloons: sized from text, placed by search, checked by pixel
      package.ts          — validatePackage(): structure, ASCII, banned names, a new picture each weekday
      banned.ts           — canon names the series must never print
      render.ts           — compose scene pages (+ replay) in full colour, untouched art (#73); teasers, before/after cards; encode PNGs
      build.ts            — one chapter: hold, crop, map anchors, render → R2 key → PNG
      watch.ts            — alert problems and /health-detailed snapshot for the series
      proof.ts            — proof export helpers: a chapter's 51 R2 keys, the PNG inside a page, SHA-256, the revision digest (DECISIONS #78)
      mailbox.ts          — parses mailbox first-line commands (APPROVE needs the export revision); dueClaudeChapter() for the weekly run (DECISIONS #72); CALENDAR + heartbeat() for the Monday note (#76)
    planet.ts             — Desk panel planet of the day: NASA Exoplanet Archive queries + daily KV cache `planet:v1:DATE` (DECISIONS #68)
    birthday-image.ts     — Birthday portrait generation (FLUX.2 + R2 photos)
    images-input.ts       — bytesToImageStream(): wraps bytes for env.IMAGES.input(), which rejects byte input (DECISIONS #59)
    escape.ts             — HTML escaping utility (escapeHTML)
    fetch-timeout.ts      — fetchWithTimeout() utility (AbortController-based)
    validate.ts           — Input validation (parseMonth, parseDay, parseStyleIdx)
    response.ts           — htmlResponse() with security headers
    alert.ts              — failure alerting: cache-state check + Email Routing sender (DECISIONS #60)
    cache-guard.ts        — AI budget marker, soft generation lock, serveStaleWhileGenerating() (DECISIONS #64)
    stale-cache.ts        — prefix-based lookup of the most recent cached entry; survives cache-key version bumps (#58). Listings memoised at `stale-idx:v1:<prefix>` 6h so an outage can't exhaust the KV list quota (#62)
    weather-ui.ts         — Shared weather page helpers (formatDate, formatTime, icon, etc.)
    headlines.ts          — Steel/trade RSS/HTML fetch + deterministic ranking (no LLM)
    moon.ts               — Pure moon phase calculator + parametric SVG icons
    skyline.ts            — World Skyline Series: city/style selection, prompt building
    skyline-image.ts      — Skyline image generation (SDXL) + BW/color post-processing
    spectra6.ts           — Spectra 6 palette constants + CSS variables
    dither-spectra6.ts    — Floyd-Steinberg dithering engine (no panel picture uses it since #81; kept for the World Cup flag generator)
    png.ts                — Pure JS PNG encoder (8-bit, 1-bit, indexed) + pngToBase64()
    png-decode.ts         — PNG decoder (RGB, RGBA, Gray, GrayAlpha)
    font.ts               — 8x8 bitmap font (CP437)
    # --- World Cup: RETIRED 2026-08-21, preserved for 2030 (DECISIONS #61). Unwired from
    # --- the router/cron; files and tests intentionally kept. Do not delete.
    worldcup.ts           — WC data layer (SWR/withBudget/KV) + finalize (phase, qualifying flags)
    worldcup-football-data.ts — football-data.org adapter (primary source)
    worldcup-openfootball.ts  — openfootball/worldcup.json adapter (no-key fallback)
    worldcup-ui.ts        — WC shared LOGIC (qualifiedFlags fixture-aware, computePhase, pickRotatingGroup) + HTML section builders + renderWorldCupHTML(data, theme)
    worldcup-bracket.ts   — HARDCODED official 2026 knockout tree (BRACKET_R32) + orderKnockout(): places KO matches by the real bracket, matched by teams (DECISIONS #52)
    worldcup-styles.ts    — SEPARATE per-display stylesheets: COLOR_STYLE (E1002, pristine) + MONO_STYLE_BASE (E1001 crisp). Never share CSS (DECISIONS #48)
    worldcup-browser-image.ts — B&W: Browser Rendering screenshot of /worldcup?variant=src → supersample → 1-bit PNG
    worldcup-fonts.ts     — GENERATED: inlined Inter 500/700 WOFF2 (base64) for the B&W image source
    worldcup-flags.ts     — GENERATED: Spectra-6 dithered country-flag PNGs (color page only; `npm run flags`)
    worldcup-testdata.ts  — canned fixtures for ?test-phase previews
    hud/                  — Desk panel drawing kit: Worker-drawn 1-bit pages (DECISIONS #66, #67)
      canvas.ts           — 800x480 1-bit buffer + clipping primitives (line, rect, invert)
      fonts-data.ts       — GENERATED (`npm run desk-fonts`): 1-bit glyph bitmaps of IBM Plex Mono (ASCII + accented capitals) + Aurebesh AF
      type.ts             — draws text from those glyphs: alignment, letter-spacing, white-on-black
      chrome.ts           — the top bar every desk page shares (breadcrumb, status, inverted pill)
      parts.ts            — pieces of the HUD look: brackets, ruler, hatch, dot grid, "+" marks; all bounded by the canvas
      page.ts             — canvas → 1-bit PNG → one-request HTML page; a draw failure becomes the text page
      test-card.ts        — the calibration card drawn at /desk/test
      aurebesh.ts         — the check sheet: 26 letters + 10 numerals
      aurebesh-words.ts   — word list (180), themes, letter names, date-to-word picker, special days, ?word=/?date= cleaning
      aurebesh-pronounce.ts — hand-written "how to say it" respelling for every word
      aurebesh-page.ts    — layout of the Aurebesh word-of-the-day page
      planet-draw.ts      — draws a planet from its data: class, surface, lighting, name-seeded pattern, in full colour by kind of world (DECISIONS #68, #81)
      planet-page.ts      — layout of the planet-of-the-day page + number formatting
      tars-page.ts        — layout of the TARS page (machines, quota table, cards)
      crest.ts            — a club crest PNG → 96x96 RGB (transparency as paper; the panel converts, #81), and drawing it
      flamengo-page.ts    — layout of the Flamengo board (table, next match, cards) + fitting names to their places
    pages/
      weather2.ts         — /weather HTML page (E1001 mono)
      fact.ts             — /fact HTML wrapper for fact.png (falls back to a text page)
      unavailable.ts      — 800x480 "no image to show" page (e-ink safe: #000 on #fff, no JS)
      # --- RETIRED 2026-08-21 with the rest of the World Cup code (DECISIONS #61). Unwired
      # --- from the router; kept for 2030. Do not delete.
      worldcup.ts         — /worldcup (E1001): default = pre-dithered image (SWR + cron warm), ?variant=src = HTML source
      color-worldcup.ts   — /color/worldcup (E1002 Spectra 6): live HTML + flags
      color-weather.ts    — /color/weather HTML page (E1002 Spectra 6)
      color-moment.ts     — /color/moment + color birthday + test endpoints
      color-headlines.ts  — /color/headlines HTML page
      skyline.ts          — /skyline + /skyline-bw (PNG inlined as a data URI, one request — DECISIONS #65) + /skyline-test HTML pages
      desk.ts             — /desk/test, /desk/aurebesh, /desk/planet, /desk/tars (view + report), /desk/flamengo handlers (DECISIONS #66–#70)
      family.ts           — /color/family: today's family picture inlined, else the latest earlier one, else the text page (#92)
      series.ts           — /desk/series, /series, /color/series (one R2 page inlined; before a premiere the "coming soon" card is drawn live with a countdown, #80) and /series/proof (DECISIONS #71)
  photos/
    portraits/            — Birthday reference photos ({key}_0..3.jpg) + people.json (key → Photos name), hints.json (face positions), picks.json (manual face picks); all gitignored (#85, #87)
    skylines/             — Skyline reference photos source (30 cities, 65 photos)
  scripts/
    upload-photos.sh      — Upload birthday photos to R2, prepared by prepare-portraits.py (#85)
    prepare-portraits.py  — Birthday references for FLUX.2: upright, cropped to the face (Vision via portrait-faces.swift), under 512px; picks.json for group photos; contact sheet (#85)
    portrait-faces.swift  — Prints the faces Apple's Vision finds in each image, as JSON (#85)
    family-pick.py        — Family Memory: picks the date's photo from a read-only Photos.sqlite copy (album, Favourites, closest day); tests in test_family_pick.py (#92)
    family-fetch.swift    — Family Memory: downloads one photo at 1600 px through PhotoKit; compiled to ~/Library/Application Support/eink-family/ (#92)
    pick-portraits-from-photos.py — Best recent face photos per family member from the Photos library's People album (read-only DB copy; iCloud originals exported via Photos) + face-position hints (#87)
    download-skyline-photos.sh — Download skyline photos from Unsplash
    upload-skyline-photos.sh   — Upload skyline photos to R2
    tars-report.sh             — Runs on each Mac every 30 min (launchd): posts disk/memory/uptime + CodexBar snapshot to /desk/tars; key from the Keychain (DECISIONS #69)
    tars-launchd.plist.example — launchd template for the reporter
    series-import.mjs          — Offline series import: build | upload | approve | export | templates | preview (`npm run series --`, DECISIONS #71). `upload` also writes the proof export for Jarvis to Dropbox `00_JARVIS_SPOILERS/`; `approve` needs the reviewed `--revision` (#78)
    materialize-exec.py        — Lets a launchd job read Dropbox online-only (dataless) files: sets the materialize I/O policy, then execs the real program (DECISIONS #72)
    series-teaser/             — `teaser.py`: the series teaser, frames and score drawn by code; `blender/lantern.py`: v2's 3D shots (the ship, the outpost, space), built and rendered in Blender (DECISIONS #77, docs/series/TEASER.md); pictures, video, audio and .blend files stay out of the repo
    series-paint/              — Claude's painting engine (DECISIONS #74): `painter.py` paints a code-drawn underpainting with textured brush strokes (gouache for the Prologue; oil and pixel for style tests), `page-png.py` pulls the picture out of a built page. Python 3 + numpy + PIL, run with `python3 -E -P`
    series-mailbox.mjs         — The mailbox watcher (DECISIONS #72): `run` handles Jarvis's APPROVE/DELIVERY/STYLE TEST by script, other messages via a read-only `claude -p`; `weekly` writes Claude's next due chapter; `review` (also run after every Season 1 upload) is Claude's independent PASS/REVISE, which `approve` requires for the showrunner's own season (#79). Prompts in `series-agent/`, LaunchAgents in `launchd/`
    series-timeline.mjs        — What the panels will show at every refresh change, from the live schedule through the real clock code, by page NAME only (safe in front of the owner), and whether R2 holds every page; `--approve 1,2` pretends. Run before each season (DECISIONS #80)
    series-set-key.sh          — Owner runs it in a real terminal: choose SERIES_KEY (hidden prompt) → ~/.config/eink/series-key (600) + Cloudflare secret, then checks the proof page
    generate-desk-fonts.mjs    — Offline: rasterise fonts/desk/ into src/hud/fonts-data.ts (`npm run desk-fonts`, DECISIONS #67)
  fonts/
    desk/                 — Source fonts for the desk panel + licences (IBM Plex Mono OFL, Aurebesh AF CC0); not shipped
  CLAUDE.md               — This file
  DECISIONS.md            — Architecture & design decisions
  README.md               — Project documentation
  HANDOVER.md             — One-file agent briefing; part of the documentation sweep
  AGENTS.md               — Entry point for agents that auto-load it; points at HANDOVER.md
  wrangler.toml           — Cloudflare Worker config + bindings
  package.json            — Dependencies + version
  tsconfig.json           — TypeScript config
```
