# E-Ink "Moment Before" Dashboard — Agent Handover

**Last verified:** 2026-10-15 · **Describes:** v3.21.0 at `b9466f6` (the proof export #78, Season 1 rules #79, the live countdown card #80, deployed 2026-10-15 as c85f6fce) (the series, DECISIONS #71–#77: full-colour pages converted by each panel, painted art, art direction, rhythm, teasers), which is what production runs.
**Audience:** an AI agent picking this project up cold. Read this once end to end, then use the linked files as reference.
**Maintenance:** this file is part of the mandatory documentation sweep in `CLAUDE.md`. When a change makes any statement here untrue, fix it in the same commit and update the line above.

This file is a map and a set of rules. It does not replace the three documents that hold the detail:

| File | What it holds | Trust it for |
|---|---|---|
| `CLAUDE.md` | Session rules, obligations table, file map | How to work here |
| `DECISIONS.md` | 81 numbered decision records, including every failed approach | *Why* things are the way they are |
| `README.md` | Endpoints, pipelines, setup, troubleshooting table | What exists |
| `INCIDENT-2026-08-20-neuron-budget-blowout.md` | Evidence, timeline and diagnosis runbook for the worst outage | What to do when AI pages fail |

`AGENTS.md` is a short entry point that sends auto-loading agents here.

When documents disagree about **current behaviour**, the source code and the live `/health-detailed` endpoint win. When the question is **why**, `DECISIONS.md` wins. Section 15 lists what is still unconfirmed.

---

## 1. What this project is

A single Cloudflare Worker (TypeScript, no framework, zero runtime dependencies) that renders content for three e-ink panels owned by one person. It is a personal project on Cloudflare's **free tier**, and staying on the free tier is a deliberate choice that shapes most of the engineering.

Since 2026-10-14 the panels also carry **The Long Drift**, a Star Wars graphic-novel series written and drawn ahead of time by AI writers, one season per quarter, coordinated by the owner's agent Jarvis and shown a scene at a time (#71, #72).

The headline feature is **"Moment Before"**: every day the Worker picks a famous historical event that happened on this date, has an image model illustrate it in a rotating art style, and converts the result to something an e-ink panel can show well. Alongside it the panels show a weather dashboard, a daily AI city skyline, and steel/trade headlines.

- **Live URL:** `https://eink-dashboard.thiago-oliveira77.workers.dev`
- **Repo:** `Rabbit-Olive-AI-Devs/rabbitolivestudios.github.io` (public; moved from `rabbitolivestudios/…` on 2026-09-10, old URL redirects). The project lives in the `eink-dashboard/` subdirectory; the repo root holds an unrelated website.
- **Branch:** `main`. Deploys are done from a local machine with `wrangler`, not by CI.

## 2. The panels, and how pixels reach them

| | E1001 | E1002 |
|---|---|---|
| Hardware | reTerminal E1001, 7.5" mono ePaper | reTerminal E1002, 7.3" E Ink Spectra 6 (6 colours) |
| Resolution | 800x480 | 800x480 |
| Location | Home, Naperville IL (zip 60540) | Office, Chicago IL (zip 60606) |
| SenseCraft device ID | `20225290` | `20225358` |
| Routes it uses | `/fact.png`, `/weather`, `/skyline-bw`, `/series` (added 2026-10-14) | `/color/moment`, `/color/weather`, `/skyline`, `/color/headlines`, `/color/series` (added 2026-10-14) |

The exact pagelists live in the SenseCraft HMI web app, not in this repo. (Since SenseCraft HMI v1.3.2, February 2026, there is no separate "Web Function" page type: a URL is shown through the **HTML widget** in the Canvas Editor, sized 800x480 at position 0,0. Older notes in this repo still say "Web Function"; they mean the same mechanism.) The routes above are what the renderer was seen requesting in `wrangler tail` on 2026-09-10; confirm with the owner before assuming a route is or is not on a panel.

A **third panel**, a second E1001 (mono), sits on the owner's home-office desk and shows the `/desk/*` pages. It was paired in SenseCraft HMI on 2026-10-12; its device ID is not recorded here and no telemetry is fetched for it. The owner added `/desk/series` to it on 2026-10-14.

**The mental model that explains most of this project's rules:** the panel never fetches anything itself. SenseCraft HMI's "Web Function" loads each URL in a **headless Chromium in Seeed's cloud**, screenshots it, quantises the screenshot to the panel's palette, and pushes a finished image to the device. Every page of a pagelist is rendered in one burst about every 15 minutes. Consequences:

- The Worker controls only the HTML/CSS/PNG it returns. It cannot control fonts, anti-aliasing, refresh mode or timing.
- The renderer waits for the **document**, screenshots roughly **one second** later, and cancels any sub-resource still loading. A page that needs a second request for its image can come out fully white (DECISIONS #65).
- A document that takes more than about 8–10 seconds fails outright and the panel keeps its previous frame (#42).
- Any non-200 or slow response is a visible outage. There is no error handling on the device, so serving something stale is nearly always better than serving a status code.
- Anti-aliased grey edges get quantised into speckle. Text crispness problems cannot be reproduced in a local browser; they need on-device verification (#48).

## 3. Routes

| Route | Panel | What it returns |
|---|---|---|
| `/weather` | E1001 | Mono HTML weather dashboard |
| `/color/weather` | E1002 | Colour HTML weather dashboard (Spectra 6 accents) |
| `/fact.png` | E1001 | Moment Before PNG, full colour as-is with a black caption bar; the panel converts it (#81). Birthday portrait on family birthdays |
| `/fact` | E1001 | HTML page with the `/fact.png` picture **inlined** (one request since #81), text fallback |
| `/fact1.png` | E1001 | 1-bit Moment Before PNG, 6 rotating styles |
| `/color/moment` | E1002 | HTML with the Spectra-6 dithered Moment image inlined as base64 |
| `/skyline`, `/skyline-bw` | E1002, E1001 | HTML with the daily skyline PNG inlined as a `data:` URI |
| `/skyline.png` | — | The skyline PNG itself (`?bw=1`, `?mode=daily\|rotate\|random`) |
| `/color/headlines` | E1002 | Steel/trade headlines, ranked without an LLM |
| `/desk/aurebesh` | Desk E1001 | Aurebesh word of the day as a Worker-drawn 1-bit PNG inlined in one document (`?date=`, `?word=`, `?sheet`) (#66) |
| `/desk/planet` | Desk E1001 | Planet of the day from NASA's Exoplanet Archive, drawn from its data; `?date=` (#68) |
| `/desk/tars` | Desk E1001 | TARS: the owner's Macs and AI allowances. `GET ?key=` shows the page, `POST` with a bearer key takes a report from a Mac (#69) |
| `/desk/flamengo` | Desk E1001 | Flamengo board: Serie A table, next match with crests, last result, form, countdown; ESPN's public feeds, zone lines from FotMob's table legend (#70) |
| `/desk/series` | Desk E1001 | The series: current scene from R2 `eink-series`, pre-built offline, full colour, converted by the panel (#71, #73) |
| `/series` | Home E1001 | The series: the same page as `/desk/series` (#71, #73) |
| `/color/series` | Office E1002 | The series: the same page again (#71, #73) |
| `/desk/series/test`, `/series/test`, `/color/series/test` | Any (temporary) | A writer's style-test picture, untouched and cropped to the screen (`?folder=&image=&key=`; the three addresses are the same page), uploaded by the watcher (#72) |
| `/series/proof` | Showrunner | A chapter's scenes and teaser for approval, each page framed from `/series/proof/page`; needs `SERIES_KEY` (#71) |
| `/desk/test` | Desk E1001 | Calibration card for judging type, rules and fills on the panel (#66) |
| `/clean` | either | Solid full-screen colour fills, rotating, to clear e-ink ghosting (#56) |
| `/health`, `/health-detailed` | — | Version; cache state, `next_day_images`, AI budget block, alerting config |
| `/weather.json`, `/fact.json`, `/fact-raw.jpg` | — | Data and debug outputs |
| `/color/apod` | — | 301 to `/skyline` (legacy) |

**Test routes** that cost AI calls are protected by the `TEST_AUTH_KEY` secret (`?key=…`; a wrong key returns 404 to hide the route; with no secret set, as in local dev, they are open): `/test.png`, `/test1.png`, `/test-birthday.png`, `/color/test-moment`, `/color/test-birthday`, `/skyline-test`, `/skyline-test.png`, `/alert-test`.

**Free test parameters** (no AI, no auth): `?test-device`, `?test-alert=tornado|winter|flood`, `?test-rain`, `?test-temp=N`, `?test-moon=0..7`, `?test-provider=nws|fail`, `?test-headlines`.

## 4. Platform

| Binding / secret | Service | Use |
|---|---|---|
| `env.AI` | Workers AI | Llama 3.3 70B (event selection), FLUX.2 klein-9b and SDXL (images) |
| `env.IMAGES` | Cloudflare Images | JPEG to PNG, centre-crop and resize to 800x480 |
| `env.CACHE` | KV namespace `de97776d35af4df08b13fd2158acebdc` | Every cache in the project |
| `env.PHOTOS` | R2 bucket `eink-birthday-photos` | `portraits/` (birthday references), `skylines/` (city references) |
| `env.SERIES` | R2 bucket `eink-series` (private) | The series: every finished page (`series/sNN/cNN/…html`, one-request documents) and `series/schedule.json` (#71) |
| `env.SEND_EMAIL` | Email Routing | Failure-alert email |
| `TEST_AUTH_KEY`, `ALERT_TO`, `ALERT_FROM` | Secrets | Test-route auth; alert addresses (secrets because the repo is public) |
| `TARS_KEY` | Secret | Lets the owner's Macs post to `/desk/tars` and unlocks that page |
| `SERIES_KEY` | Secret | Unlocks series previews (`?at=`) and the proof page; they reveal unreleased chapters. The owner sets it with `scripts/series-set-key.sh`; a copy lives in `~/.config/eink/series-key` on the Mac mini (not the Keychain, which locks in remote sessions) |
| `FOOTBALL_DATA_KEY` | Secret | Only for the retired World Cup code |

**Cron triggers** (`wrangler.toml`), all in UTC:

- `35 4,5 * * *` — the daily image warm. It fires *before* Chicago midnight and fills the keys for the date the panels are **about to** ask for (`dailyWarmTargetDate()`), so a panel poll is never the thing that triggers generation (#64). It is one expression with two fire times because the account is at the free plan's cap of 5 cron triggers.
- `5 0,6,12,18 * * *` — refreshes headlines, both weather locations and both devices' telemetry, warms the desk panel's planet for today and tomorrow, refreshes the Flamengo board if its entry has aged, then runs the failure-alert check.

The daily warm runs its pipelines **sequentially in priority order** (Pipeline A or birthday, Pipeline B, colour moment, skyline, skyline BW), skips any key that already exists, and stops at the first neuron-budget error.

**Off-Worker jobs on the Mac mini** (LaunchAgents, logs in `~/Library/Logs/eink-series/` and `/tmp`): `com.eink.tars-report` (TARS, every 30 min, #69); `com.eink.series-mailbox` (handles Jarvis's mailbox within minutes of a file arriving, #72); `com.eink.series-weekly` (Wednesdays 09:05: writes Claude's next due chapter, #72); `com.eink.series-heartbeat` (Mondays 08:00: a status note for Jarvis of what is late, due and on air, no AI, #76). The series jobs need the external SSD mounted, because Dropbox syncs to `/Volumes/SamsungSSD/.CloudStorage/Data/Dropbox`. They start through `scripts/materialize-exec.py`, because Dropbox keeps files online-only and a launchd job may not download them otherwise (reads fail with "Unknown system error -11").

**External data sources:** NASA Exoplanet Archive (desk planet, no key), Wikipedia "On This Day", Open-Meteo (weather, primary), NWS `api.weather.gov` (alerts, and keyless weather fallback), SenseCraft HMI API (battery, indoor temperature and humidity), Steel Industry News / SteelOrbis / Google News RSS (headlines).

## 5. Image pipelines

All pipelines share one LLM-selected event per day through `getOrGenerateMoment()` (KV `moment:v1:DATE`). The LLM writes a **scene-only** prompt; each pipeline prepends its own style. They diverge after that and **must not be cross-contaminated**: a change to one pipeline's style, model or post-processing must not leak into another.

| | A: `/fact.png` | B: `/fact1.png` | C: birthday | D: `/color/moment` | E: skyline |
|---|---|---|---|---|---|
| Model | FLUX.2 klein-9b, SDXL fallback | SDXL | FLUX.2 + up to 4 R2 photos | FLUX.2, SDXL fallback | FLUX.2 + R2 reference photo, SDXL fallback; BW variant is SDXL only |
| Style rotation | 3, by day of year | 6, by hash of date/title/location | 10, by year | 5, by day of year | 14 styles over 30 cities |
| Output | Full colour as-is (#81) | 1-bit PNG (Bayer or threshold per style) | Full colour as-is (#81) | Full colour as-is (#81) | Full colour as-is, both modes (#81) |
| Cache key | `fact4:v5:DATE` | `fact1:v7:DATE` | `birthday:v2:DATE`, `color-birthday:v2:DATE` | `color-moment:v3:DATE:STYLE` | `skyline:v4:DATE:daily[:bw]` |

Facts that have each cost real debugging time:

- The brand is "Moment Before" but the prompt depicts **the event itself** at its defining moment; calm pre-event scenes were unreadable on e-ink (#7).
- FLUX.2 takes multipart FormData, SDXL takes JSON. SDXL has no `negative_prompt`, so negatives go in the positive prompt, and it caps at 20 steps.
- **`env.IMAGES.input()` must be given a ReadableStream.** Raw bytes throw `undefined (reading 'font')` *after* the AI call has been billed. Always wrap with `bytesToImageStream()` from `src/images-input.ts` (#59).
- **One FLUX attempt, then fall back. No retry loops.** When changing a retry or budget pattern, grep every call site; a past fix missed four of five (#38).
- **Since 2026-10-15 the panels get every picture as-is** (#81): full colour, untouched, caption burned in as solid black and white by `burnCaption()`; each panel converts it, which the owner judged far better than our own 4-grey and Spectra 6 conversions (first seen with the series, #73). Do not add a conversion step back without the owner comparing on the panels. History: Floyd–Steinberg was used for the colour panel and Bayer/threshold for mono (#4, #14); captions were drawn after dithering so diffused error could not corrupt them (#64). Only `/fact1.png` (1-bit, unused by the panels) still converts.
- **The series is a separate, offline pipeline** (#71): pictures from the writers are cropped and lettered by `scripts/series-import.mjs` on the Mac mini and stored in R2 as finished one-request HTML pages, the picture untouched in full colour. Every panel gets the same page and converts it itself; the series does no dithering of its own (#73). The Worker only passes one stored page through per request. Never move that work onto the request path. Claude paints its own pictures with `scripts/series-paint/painter.py` (gouache over a code-drawn underpainting, #74); the owner approves every writer's look from story-free style tests.
- Workers cannot run native image libraries; the PNG encoder and decoder are hand-written (`png.ts`, `png-decode.ts`) and the caption font is an 8x8 bitmap.

## 6. Caching

Dates in cache keys are **America/Chicago** dates. Cache versions live in `src/cache-keys.ts`.

| Kind | Keys | Soft TTL (freshness) | KV `expirationTtl` (survival) |
|---|---|---|---|
| Daily images and data | `fact4:`, `fact1:`, `color-moment:`, `skyline:`, `birthday:`, `moment:`, `headlines:v3:DATE:PERIOD` | 24h (headlines 6h) | 604800 (7 days) |
| Weather | `weather:{zip}:v2` | 15 min | 86400 |
| Alerts, device telemetry | `alerts:{zip}:v1`, `device:{id}:v1` | 5 min | 86400 |
| Desk planet | `planet:v1:DATE` | 24h | 604800 (7 days) |
| Desk TARS | `tars:v1:state` (one document, rewritten by each Mac's report) | 75 min per machine, 3 h per provider | 604800 (7 days) |
| Desk Flamengo | `flamengo:v1:state` (one document: table, results, fixtures, two crests) | 6 h; 30 min from kickoff to four hours after | 604800 (7 days) |
| Guards and bookkeeping | `ai-budget:v1:block`, `gen-lock:v1:*`, `stale-idx:v1:<prefix>`, `alert:v1:last`, `nws-points:{zip}` | — | varies |

Rules:

1. **Soft TTL and hard TTL do different jobs.** The soft TTL decides when to refetch; the KV TTL decides how long a stale fallback survives. The hard TTL must be far larger than the soft one. Setting them close caused a production crash (#24), and a 24h TTL on a daily key left zero overlap and nothing to fall back to (#57).
2. **Every `.put()` carries an `expirationTtl`.**
3. **Bump the cache-key version after any pipeline change** so old output is not served. The stale fallback resolves keys by KV *prefix*, so a bump does not disable it. Never reintroduce a fallback that rebuilds keys from the current version constant (#58).
4. Local KV persists between `wrangler dev` restarts; use the test routes with a fresh date to bypass it.

## 7. The four budgets

Everything here runs inside hard free-tier limits. Cost any change against them before writing code.

| Budget | Limit | Normal use | Notes |
|---|---|---|---|
| Workers AI neurons | 10,000/day, resets 00:00 UTC | ~5,500 | FLUX.2 klein-9b costs ~1,364 per image and is essentially the whole bill. Both panels share one account-wide pool: either can blank the other. |
| KV writes | 1,000/day | ~590, plus ~96 once both Macs report to TARS, plus 4 to 12 for the Flamengo board | The tightest budget. Mostly weather/alerts/device caches refreshed by two panels polling every 15 minutes. About 400 spare (#62). |
| KV lists | 1,000/day | 0 on a healthy day | Only the stale fallback lists, and its listings are memoised for 6h. |
| R2 (series) | 10 GB stored, 10 M reads and 1 M writes a month | ~2 reads per panel poll; ~160 writes per chapter upload | Far inside the free tier (#71). |
| Cron triggers | 5 per account, shared by six Workers | 2 used here | A deploy that exceeds it fails with `code: 10072` but **still uploads the Worker** on the old schedule. |

A fifth budget is time: about one second for sub-resources and 8–10 seconds for the document (section 2).

Moving to Workers Paid ($5/month) would remove the neuron limit. It was offered and the owner chose to stay on the free tier and remove waste instead.

## 8. How the system degrades

The design goal is that a panel always shows *something reasonable*. In order:

1. **Warm cache.** The cron filled today's keys before midnight. This is the normal path.
2. **Stale-while-generating.** On a cold key with a healthy budget, AI routes return the most recent cached image immediately (`X-Stale-While-Generating: 1`, `no-store`) and generate under `ctx.waitUntil` (`serveStaleWhileGenerating` in `cache-guard.ts`, #64).
3. **Stale fallback.** If generation fails or the budget is blocked, routes serve the newest cached image from up to 7 days back, found by KV prefix (`stale-cache.ts`, #58).
4. **Cross fallback.** The colour skyline will serve the BW skyline rather than fail.
5. **Text page.** If AI is blocked *and* nothing cached remains, wrapper pages render a plain 800x480 black-on-white page with a Chicago timestamp, never a broken-image glyph (`pages/unavailable.ts`).
6. **503**, only when nothing else is possible.

Supporting mechanisms:

- **AI budget block** (`ai-budget:v1:block`): the first neuron-quota error arms a marker that lasts until the next 00:00 UTC. While armed, no new generation is attempted.
- **Generation lock** (`gen-lock:v1:*`): a best-effort single-flight guard. KV has no atomic put-if-absent, so it reduces duplicates but cannot prevent them. Do not design anything that depends on it being correct.
- **Weather** uses stale-while-revalidate, a 4-second Open-Meteo timeout, an NWS fallback, and a 5-second cap on the cold path (`with-budget.ts`, #42, #43).
- **All external fetches** go through `fetchWithTimeout()`; failures return `null` or `[]` and never crash a page.

**The series** degrades on its own path: an unapproved or missing chapter shows the last approved scene; a missing page or broken `schedule.json` gives the text page; between seasons a "coming soon" or "returns soon" card shows. The alert check also emails when a chapter airing within 7 days is not approved, when a season ends within 14 days with nothing after it, or when the page for now is missing (#71).

**Alerting** (`src/alert.ts`, #60): graceful degradation hides failures, and two multi-day outages were found only by looking at a panel. The 6-hourly cron now checks whether the day's five images are in KV and whether the budget block is armed, and emails on a *change* of state through Cloudflare Email Routing. A KV fingerprint limits it to one email per distinct problem, one reminder per 24h, and one recovery notice. It skips Chicago hours before 3 so it cannot race the warm. When you add a failure mode, ask whether this check would catch it.

## 9. Non-negotiable rules

**For anything a panel displays:**

- Exactly 800x480, no scrolling, nothing overflowing.
- Pure `#000` on `#fff`. Greys vanish on mono and dither into speckle on colour.
- No emoji (use inline SVG). No JavaScript.
- **One request per page.** Inline images as `data:` URIs.
- Never block a panel request on AI generation when something older can be served.
- On the colour panel, text is black. Colour comes only from large solid fills or pre-dithered images; of the six colours only black, red, green and blue are legible as foreground, and yellow is background-only (#40, #45).
- Font weight at most 700, body 500. Size rows to content with fixed spacing so baselines land on whole pixels. Never put text rows in a shrinkable flex container with `overflow: hidden` (#46, #48).
- `/weather` and `/color/weather` have only **5–7px of vertical slack**, and `.hourly` absorbs every deficit while `overflow: hidden` hides the clipping. `body.scrollHeight` reports 480 even when cards are cut. Measure the deepest rendered element instead, and test with a banner (`?test-alert=tornado`) (#63).
- An HTML page whose body is an image needs a text fallback.
- **The series** (#71, #72): never commit season scripts, pictures, packages or build output (public repo, spoilers); never show the owner an unaired scene, proof page or plot (he watches unspoiled), so write chapters through a background helper or the weekly unattended run, never in the main conversation; a chapter airs only after the showrunner's `APPROVE`.
- **Desk panel pages (`/desk/*`)** are drawn by the Worker as 1-bit images with `src/hud/`, never as live HTML text. Type is real fonts only, from the generated glyph faces, and never smaller than `plex11m` (11px medium), the owner's minimum on the panel. Show the owner a mockup before building a new page (#66, #67).

**For code:**

- Use the `Env` type from `src/types.ts`.
- Coerce LLM output: `typeof raw === "string" ? raw : JSON.stringify(raw)`.
- Chunk `String.fromCharCode` in 8192-byte slices (`pngToBase64`).
- Escape all dynamic HTML with `escapeHTML()`; return HTML through `htmlResponse()`.
- Follow the existing templates: `alerts.ts` for cached API fetches, `image.ts` for image pipelines.
- No dead code, with one named exception in section 13.
- Never write a credential to any file, commit message or document. The repo is public. The SenseCraft API key in `device.ts` is the single documented exception: it is a shared platform key published by Seeed (#17).

## 10. How the owner and the agent work together

This is the process the owner expects. It is written down in `CLAUDE.md`; this is the short form.

**Starting a session.** Read `CLAUDE.md`, `DECISIONS.md` and `README.md`. Run `git log --oneline -10` and `git status`, and check the version in `package.json`. Compare against `origin/main` and `/health`, because deploys happen from more than one place. Summarise the state in a few bullets, confirm scope, and write a short "definition of done" before touching code: what must work, what must be tested, which docs change, how success is verified, which edge cases apply.

**While working.**

- Small, reviewable changes. One logical change per commit. Refactors and features in separate commits.
- Read the code before proposing a change to it.
- Fix root causes. When the first explanation does not account for the numbers, keep going: the August outage was first "explained" by a real but irrelevant bug, and the true cause was found a day later (#57 corrected by #59).
- **Measure rather than reason.** Several fixes here looked complete and were less than half as effective as assumed until measured (#62).
- When a cache looks empty, check whether **writes** are failing before concluding that reads are racing (`wrangler kv key list --remote`).
- Visual changes are checked in a browser at exactly 800x480 in every state before deploying. Crispness and renderer-timing questions need the real device.
- Record rejected alternatives. Most decision records have a "Rejected" section, and it is often the most useful part.

**Finishing.**

- Run `npm run typecheck`, `npm run test:utils` and `npm run dry-run`. Never commit code that does not build.
- Update documentation **in the same commit**, sweeping all of `DECISIONS.md`, `README.md`, `HANDOVER.md`, `CLAUDE.md`, and the version in both `package.json` and the `VERSION` constant in `src/index.ts`.
- Never delete or rewrite history in the docs. When a decision turns out wrong, add a "superseded by #N" note to the old record and write a new one.
- Commit message: short summary, blank line, a body that explains *why*. End with the attribution trailer your session provides.
- Push after committing. Never force-push `main`.
- **Version bumps only with the owner's approval.**
- Stop any dev server you started.
- End with a change summary: what changed, why, how to test, docs updated, risks and follow-ups.

**Owner preferences worth knowing.**

- Free tier, no third-party services, no new accounts or API keys where avoidable. Alerting was built on Cloudflare Email Routing for exactly this reason.
- Do not ask the owner to paste an API token. A Cloudflare `10000` auth error means the stored OAuth login was revoked and they need to log in again interactively.
- Honest reporting. Say what was verified and what was not. An earlier agent reported commits, a pull request and a production check that never happened (#23); claims are checked against `git log` and the live site.
- Useful dormant code is kept, not deleted (section 13).

## 11. Commands

```bash
# Verify (always, before committing)
npm run typecheck
npm run test:utils        # compiles to /tmp, runs node --test; 284 tests
npm run dry-run
npm run desk-fonts        # only after changing fonts/desk or the face list; regenerates src/hud/fonts-data.ts

# Local dev. Port 8787 is often taken by another daemon; check, then use 8790.
lsof -ti:8790
CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c npx wrangler dev --port 8790
# add --test-scheduled to be able to hit /__scheduled?cron=35+4,5+*+*+*

# Deploy. The account ID is needed inline every time; it is not a secret.
CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c npx wrangler deploy
curl -s https://eink-dashboard.thiago-oliveira77.workers.dev/health   # ~20-30s to flip

# Production state. wrangler 4 KV commands read the LOCAL store unless --remote is given.
curl -s https://eink-dashboard.thiago-oliveira77.workers.dev/health-detailed
npx wrangler kv key list --remote --namespace-id de97776d35af4df08b13fd2158acebdc --prefix "fact4:"
npx wrangler kv key get "ai-budget:v1:block" --remote --namespace-id de97776d35af4df08b13fd2158acebdc
npx wrangler tail eink-dashboard --format json     # can go silent while alive; re-attach

# The series (offline, on the Mac mini; packages and output stay in ~/series-work, never the repo)
npm run series -- build <package.zip|dir> [--chapter N] --out ~/series-work/out-…
CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c npm run series -- upload ~/series-work/out-…   # pending until approved
CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c npm run series -- approve --season S --chapter C --revision R   # R = the reviewed proof export; rechecked against R2
CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c npm run series -- export --season S --chapter C   # proof export for Jarvis (upload runs it too), #78
npm run series -- preview <picture|folder> --out DIR    # a style test: desk, office and as-is versions
npm run series -- templates                             # regenerate docs/series/templates/*.png
curl -s "https://eink-dashboard.thiago-oliveira77.workers.dev/series/proof?season=0&chapter=1&key=$(cat ~/.config/eink/series-key)" | grep -c '<img'
launchctl kickstart gui/$(id -u)/com.eink.series-mailbox   # handle the mailbox now
tail ~/Library/Logs/eink-series/mailbox.log
```

Things that are not obvious:

- "Failed to retrieve account IDs" means the account ID was not passed inline. It is not an auth failure.
- `Authentication error [code: 10000]` on deploy means the OAuth login was revoked. The owner runs `npx wrangler logout && npx wrangler login` themselves.
- Plain `wrangler dev` runs `env.IMAGES` in local mode and does **not** reproduce edge Images faults. `wrangler dev --remote` under the real Worker name inherits production secrets, so the test routes return 404. To test against real bindings, run under a different name with a copy of `wrangler.toml` and a throwaway KV namespace, and put any needed values in `.dev.vars` (gitignored).
- `/test1.png` uses SDXL, which has billed 0 neurons, so it is the cheap way to exercise a pipeline end to end.
- After any deploy that touches `[triggers]`, confirm the live schedule through the Cloudflare API (`/accounts/{id}/workers/scripts/eink-dashboard/schedules`).
- Real neuron and KV usage come from the GraphQL analytics API (`aiInferenceAdaptiveGroups`, `kvOperationsAdaptiveGroups`); the exact queries are in the incident report and DECISIONS #62. Do not trust a `4006` error at face value: Cloudflare has returned false ones, and the midnight reset can lag by an hour.
- Browser screenshots must be taken over http(s), with a cache-busting query parameter, at an 800x480 viewport.
- The Cloudflare dashboard cannot be driven by browser automation here (not signed in). Give the owner the click path and verify the result through the API.
- **Sessions on the Mac mini run in tmux** (since 2026-10-15). The owner connects with `ssh -t mini bin/claude-eink` (`~/bin/claude-eink` attaches or starts the tmux session `claude-eink` in this repo and continues the last conversation), because their SSH through the Cloudflare tunnel drops every 20 minutes or so on their PC's side. A drop no longer ends the session; a reboot or `/exit` does. Scheduled checks a session sets up (CronCreate) live only as long as that session.
- **Before a season airs**, run `CLOUDFLARE_ACCOUNT_ID=… node scripts/series-timeline.mjs --from <date> --to <date> [--approve 1,2]`: what the panels show at every change, by page name only (safe in front of the owner), and whether R2 holds every page (#80).
- Jarvis reads Dropbox on another Mac (`/Users/quark/Dropbox`). If he reports empty files, check the server through the Dropbox API first: on 2026-10-15 they were online-only placeholders there, not an export fault (#78).

## 12. Symptom to first check

| Symptom on the panel | Look at first |
|---|---|
| Several AI pages fail together, or show an old image | `/health-detailed` → `config.ai_budget` and `daily_images`. Then list KV: if the newest image key predates today, generation is failing *after* the paid model call, most likely in `env.IMAGES` (#59). |
| A page is fully white, no text at all | The renderer screenshotted before a sub-resource arrived. Confirm the page is a single request (#65). Re-deploying the pagelist in SenseCraft HMI forces a re-render. |
| A large "could not be generated" text page | Expected degradation: AI blocked and nothing cached. Clears after 00:00 UTC. |
| Weather page blank or "Failed to load remote image" | The page took too long. Check `ephemeral.weather_*` age and `source` in `/health-detailed` (#42, #43). |
| Bottom row of weather cards cut off | Vertical overflow inside `.hourly`, usually with a banner showing (#63). |
| Smudged or foggy text | Grey or coloured text, weight above 700, or fractional pixel positions (#45, #46, #48). |
| Faint ghost of an old image | Image retention after a long static frame. Point the panel at `/clean` for several cycles (#56). |
| Cloudflare email about KV operations nearing the cap | Writes are normally ~59% of the cap. If `list` is elevated, AI is blocked and the stale fallback is running (#62). |
| An alert email arrives | It names the missing images, the armed budget block, or a series problem. Start at the first row of this table. |
| A series panel is stuck on one scene | The current chapter is not approved: `/health-detailed` → `series.slot`, the proof page status, and the mailbox (`from-claude/NEEDS-THIAGO.md`, `~/Library/Logs/eink-series/mailbox.log`). |
| Series page shows "has not started yet" | No `schedule.json` in R2, or it failed its shape check (#71). |

## 13. Dormant code: the World Cup dashboard

A FIFA World Cup 2026 dashboard was built for both panels and **retired on 2026-08-21 without being deleted** (#61). Only the wiring was removed (routes, a `*/15` cron, the `[browser]` binding). All `src/worldcup*.ts` files, `src/pages/worldcup.ts`, `src/pages/color-worldcup.ts`, their tests, and decision records #44–#55 are kept on purpose for the 2030 tournament.

**Do not remove it as dead code.** Keep it compiling and its tests green. `@cloudflare/puppeteer`, `flag-icons` and `@resvg/resvg-js` are devDependencies so that the preserved code typechecks without shipping. DECISIONS #61 has the revival checklist.

## 14. Short history

| When | What happened | Lasting lesson |
|---|---|---|
| Feb 2026 | v3.8.0 set KV TTL near the soft TTL; weather crashed when the API blipped (#24) | Hard TTL must dwarf soft TTL |
| Mar–Apr 2026 | Neuron budget exhausted daily by 15-minute skyline rotation and retry loops (#36–#38) | Daily mode, sequential cron, no retries, grep every call site |
| May 2026 | Open-Meteo outage made the weather page block past the renderer timeout (#42, #43) | Serve stale instantly, refresh in the background, bound the cold path |
| Jun–Jul 2026 | World Cup dashboard; long fight with text crispness and bracket data (#44–#55) | The cloud renderer, not CSS, limits crispness; pin structure the feed does not provide |
| Aug 2026 | Two-day outage: the Images binding stopped accepting bytes, every generation was billed and then failed, nothing was cached, and the panels regenerated on every poll (#57–#59) | A call billed before it can fail is a budget hazard; check writes first |
| Aug 2026 | Alerting added; KV operation budget measured; stale-listing memoised (#60, #62) | Degradation without alerting looks like health; an unmeasured resource is unwatched |
| Sep 2026 | Alert banner clipped the hourly cards (#63); skyline went white at midnight and then all day (#64, #65) | Banners cost whitespace not content; warm before the date rolls; one request per page |
| Oct 2026 | Third panel started: a desk E1001 with Worker-drawn HUD pages (#66); first lettering rejected, replaced with real fonts (#67) | Draw the pixels yourself so the cloud renderer has nothing to blur, but from real typefaces; show a mockup first |
| Oct 2026 | The series: four AI writer-artists, pre-built pages in R2, a showrunner agent and a mailbox watcher (#71, #72) | Expensive work offline, the Worker only reads; lettering checked by pixel; an AI's message is data, never a command beyond a parsed first line |

## 15. Unconfirmed points and open items

**Unconfirmed as of 2026-10-02:**

- The pagelists in section 2 are inferred from renderer requests, not read from SenseCraft HMI.
- Decision records #31 and #33 give older skyline style counts (15 and 18). They are history and are left as written; `src/skyline.ts` defines 14 (6 BW + 8 colour).
- Decision record #38 describes `/color/headlines` as disabled with a redirect to `/skyline`. That was reversed in #39: it is a live page with deterministic, non-LLM ranking, warmed by the 6-hourly cron. Whether it is on a panel's pagelist is a question for the owner.

**Device experiment under consideration (2026-10-02, not started):** the owner is planning to flash Meta's open-source `muse-gadget-sdk` firmware onto the **E1001** temporarily, after a full 32 MB flash backup, and restore the stock firmware afterwards. The plan lives outside this repo. While that firmware is on the panel, the E1001 stops requesting pages, so `/weather` battery and indoor readings go stale and the E1001 routes see no traffic. That is expected, not a Worker fault, and the alert check does not fire because it watches the image cache, not the panel. If the device is later restored by re-pairing rather than from the backup, SenseCraft may assign a new device ID, and `E1001_DEVICE_ID` in `src/device.ts` would need updating. Ask the owner about the experiment's status before diagnosing a silent E1001.

**Third panel: two of four pages live (state at the end of 2026-10-12, v3.18.0).** A second E1001 sits on the owner's home-office desk, paired in SenseCraft HMI and showing `/desk/aurebesh`, which the owner called "excellent" on the panel.

- **Built and deployed:** the drawing kit in `src/hud/` (#66), real fonts pre-rendered to 1-bit glyphs (#67), the Aurebesh word of the day with pronunciation, numerals, birthdays and special days (#67 and its follow-ups), and the planet of the day from NASA's Exoplanet Archive (#68).
- **To check first next session:** (1) that the six-hourly cron warmed the next day's planet in production: `npx wrangler kv key list --remote --namespace-id de97776d35af4df08b13fd2158acebdc --prefix "planet:v1:"` should list today and tomorrow (Chicago dates); (2) whether the owner added `/desk/planet` to the panel's pagelist, and how it looks there.
- **TARS (#69) is live** (v3.19.0, 2026-10-13). The Mac mini reports every 30 minutes through the LaunchAgent `com.eink.tars-report` (script in `~/Library/Application Support/eink-tars/`, log in `/tmp/eink-tars-report.log`, key in the login Keychain under the service name `eink-tars`). Both Macs report as of 2026-10-13 evening and the page is on the panel. **Open:** the MacBook's reports carry no good Codex reading (the Mac mini is the only source), so Codex shows `NO DATA` three hours after the Mac mini sleeps; **cause found 2026-10-13:** the MacBook's Codex sign-in had expired (`401 token_expired` from chatgpt.com in the `codex` entry's error), nothing wrong in TARS. The owner ran `codex login` on the MacBook and its 03:48 UTC report carried Codex (verified in KV): **resolved**. If Codex goes `NO DATA` again while the Mac mini sleeps, suspect an expired sign-in first. To see the page from the Mac mini: `curl "https://eink-dashboard.thiago-oliveira77.workers.dev/desk/tars?key=$(security find-generic-password -s eink-tars -w)"`. Never print or commit the key.
- **Flamengo board (#70) is live** (v3.20.0, deployed 2026-10-13 night): `/desk/flamengo`, `src/flamengo.ts`, `src/hud/flamengo-page.ts`, `src/hud/crest.ts`. Checked in production right after the deploy: the page drew, and the KV entry held 20 table rows, both crests and the zones `[4, 5, 11, 16]`, so FotMob does answer from Cloudflare's network. The owner still has to add the page to the panel (HTML widget, 800x480). Not yet seen: a match day (`LIVE`, the half-hour refresh), the off-season, the page on the panel.
- **The series, state at 2026-10-15 01:00 (newer than the paragraph below where they differ).** Every panel gets the same full-colour page and converts it (#73). The owner is the art director (#74, #75): Claude's Prologue is **painted in gouache** with `scripts/series-paint/painter.py`; chapters 1–3 were repainted and re-uploaded and are **pending Jarvis's approval** (chapter 1 by Sun 18 Oct). Tone is grown-up (no gore), original Force-users may appear as guests; writers may change look by story layer with a reason; every writer does the standard style test (now, not on the calendar). Jarvis owns the production rhythm; a Monday heartbeat note is his status board (#76). Credit names: Claude, Zoe (S1), Volt with Inky (S2), Pepper (S3). Teaser v1 in Dropbox `series/teaser/`; v2 with Blender is for the owner's MacBook (`docs/series/TEASER.md`, #77). **Open at the end of 2026-10-15:** Jarvis must APPROVE Prologue chapter 1 (with `revision=`) by Sun 18 Oct; if not, the alert mails the owner from Mon 19 Oct 07:05. Jarvis has not yet answered whether the writers' upload route works (`from-claude/…-question-writer-upload-route.md`); the owner's "Dropbox web sign-in" to-do depends on it. Volt (S2) and Pepper (S3) stay season writers (owner, 15 Oct). The posted teaser v2 still shows Zoe for Season 1; the next teaser swaps in Jarvis. **Pictures as-is (#81, 15 Oct):** every panel picture is now sent in full colour and converted by the panel; `/fact` inlines its picture. If the owner dislikes a page on its panel, revert that pipeline (commit in #81). **Countdown (#80, 15 Oct):** before a premiere the Worker draws the "coming soon" card live with the days left (no AI, no KV; falls back to the stored card). **Season 1 (#79, 15 Oct):** Jarvis writes and draws it (not Zoe) and Claude independently reviews each chapter; `approve` needs Claude's PASS for the exact revision. Looks are the writers' own; the owner approves none. **Proof export (#78, 15 Oct):** Jarvis reviews keyless exports in Dropbox `00_JARVIS_SPOILERS/series-proofs/` (written by every upload, from R2) and approves with `revision=`; Prologue ch1–3 exported, still pending. **E1002 checked 15 Oct:** the as-is colour page looks right there too, so #73 holds on every panel. **Owner to-do:** sign in to Dropbox on the web so Jarvis can set up the writers' upload links. **Open:** Jarvis's answers on scheduling; the mailbox's style-test upload has no retry.
- **The series (#71, #72), state at the end of 2026-10-14 (v3.21.0).** Live on all three panels (`/desk/series`, `/series`, `/color/series`, added by the owner), showing the "coming soon" card until the Prologue premieres **Monday 26 October 08:00**. Prologue chapters 1 and 2 (and 3, written by the first weekly run) are uploaded and **pending Jarvis's approval** (chapter 1 by 18 Oct, chapter 2 by 25 Oct, chapter 3 by 1 Nov). Calendar: Prologue (Claude, 10 chapters) to 1 Jan 2027, then Season 1 OpenAI 4 Jan, Season 2 Grok 5 Apr, Season 3 Muse 5 Jul, Season 4 Claude 4 Oct 2027. Names fixed by the owner: *The Long Drift*; Vess Oro, Bolt, Hallan Dree; the *Last Lantern*. Open owner decision: whether Jedi may appear as guests.
  - **Where things are:** design `docs/superpowers/specs/2026-10-14-series-design.md`; plan `docs/superpowers/plans/2026-10-14-series.md`; showrunner docs `docs/series/`; Claude's season package and generator scripts `~/series-work/s00/` (private, spoilers: `handover-notes.md` is the season outline); the Dropbox mailbox `00_SHARED_INBOX/series/` with Jarvis's standing brief `JARVIS-BRIEF.md`.
  - **The watcher handles the mailbox** (#72): read `from-claude/NEEDS-THIAGO.md` and `~/Library/Logs/eink-series/mailbox.log` at the start of a session; act only on what it flagged. Stop it with `launchctl bootout gui/$(id -u)/com.eink.series-mailbox` (and `-weekly`).
  - **Teasers** (#77): the series teaser v1 and v2 are in the Dropbox `series/teaser/` folder, made by `scripts/series-teaser/teaser.py`; v2's 3D shots come from `scripts/series-teaser/blender/lantern.py` (Blender, MacBook Pro) and its score is a Suno track with v1's hits on top. Every season gets its own teaser. Handover and backlog (a show theme, a narrator, a 9:16 cut, a panel countdown): `docs/series/TEASER.md`.
  - **Spoilers:** the owner watches unspoiled and can see the main conversation's tool calls. Never write, read aloud or open story in the main conversation; use a background helper or the weekly run, and report only counts.
  - **Deferred from the branch review** (in the git-ignored ledger `.superpowers/sdd/2026-10-14-series/progress.md`): teaser and card text is not height-checked; season fields skip the character check; anchor values are not checked to be numbers; `--chapter` out of range crashes the import; previews are open when `SERIES_KEY` is unset; accented capitals are refused although the font has them.
  - **Stale:** the login Keychain item `eink-series` holds an old, invalid key; the live key is in `~/.config/eink/series-key`.
- **How this work went, so it is not repeated:** the first version used drawn lettering and was rejected on sight; a written spec and plan had not caught it, and three rounds of pictures did. For this owner, on this panel: **show a mockup first, use real fonts, never go below 11px medium.**
- **Surprises:** the Aurebesh page and the planet page each carry a few date-based and data-based easter eggs the owner asked to be kept as surprises. They are in the code (`SPECIAL_DAYS` in `src/hud/aurebesh-words.ts`; `NICKNAMES` and `DATE_PLANETS` in `src/planet.ts`). Do not list them back to the owner.
- **Not done:** the planet page and the real-font rebuild had no independent code review (the first version did). The pronunciations are hand-written; only seven were confirmed against Wookieepedia. The panel's SenseCraft device ID is not in the repo, and no telemetry is fetched for it.

This is separate from the Muse firmware experiment above, which concerns the first E1001.

**Open items the owner has not chosen to act on:**

- The E1002 pagelist may effectively show the skyline twice if it still includes the legacy `/color/apod` entry.
- Headline freshness: the news shown can be stale; a better source or approach is unsolved.
- SDXL has billed 0 neurons for weeks. Unverified whether it is free or unreported, so no model switch has been made on that basis.
- No persistent logging of weather cold-path or budget events (#43).
- The alert check does not watch KV operation counts (#62).

**Ideas discussed, not started:** a combined F1/sports page; photo-referenced Moment Before images; an interactive test page for picking city and style; a weather sparkline in place of hourly cards; category icons on headlines; a fifth E1002 page.

## 16. File map

`CLAUDE.md` has the full annotated list. The files you will touch most:

| Area | Files |
|---|---|
| Router, cron, health | `src/index.ts` |
| Event selection | `src/fact.ts`, `src/moment.ts` |
| Mono image pipelines | `src/image.ts`, `src/convert-1bit.ts`, `src/styles-1bit.ts` |
| Colour pipeline | `src/pages/color-moment.ts`, `src/image-color.ts`, `src/dither-spectra6.ts`, `src/spectra6.ts` |
| Skyline | `src/skyline.ts`, `src/skyline-image.ts`, `src/pages/skyline.ts` |
| Birthday | `src/birthday.ts`, `src/birthday-image.ts` |
| Weather | `src/weather.ts`, `src/weather-nws.ts`, `src/alerts.ts`, `src/device.ts`, `src/weather-ui.ts`, `src/pages/weather2.ts`, `src/pages/color-weather.ts` |
| Caching and guards | `src/cache-keys.ts`, `src/cache-guard.ts`, `src/stale-cache.ts`, `src/with-budget.ts` |
| Alerting | `src/alert.ts` |
| Shared utilities | `src/date-utils.ts`, `src/fetch-timeout.ts`, `src/images-input.ts`, `src/png.ts`, `src/png-decode.ts`, `src/escape.ts`, `src/response.ts`, `src/validate.ts` |
| Desk panel | `src/hud/*.ts` (drawing kit, type, test card, Aurebesh), `src/pages/desk.ts`, `scripts/generate-desk-fonts.mjs`, `fonts/desk/` |
| Tests | `tests/*.test.js` (pure-function tests run with `node --test`) |
| Design notes for past features | `docs/superpowers/plans/`, `docs/superpowers/specs/` |
