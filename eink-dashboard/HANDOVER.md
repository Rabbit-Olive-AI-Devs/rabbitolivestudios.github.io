# E-Ink "Moment Before" Dashboard — Agent Handover

**Last verified:** 2026-10-02 · **Describes:** v3.16.5 (code as of `07c4427`), which is what production runs.
**Audience:** an AI agent picking this project up cold. Read this once end to end, then use the linked files as reference.
**Maintenance:** this file is part of the mandatory documentation sweep in `CLAUDE.md`. When a change makes any statement here untrue, fix it in the same commit and update the line above.

This file is a map and a set of rules. It does not replace the three documents that hold the detail:

| File | What it holds | Trust it for |
|---|---|---|
| `CLAUDE.md` | Session rules, obligations table, file map | How to work here |
| `DECISIONS.md` | 65 numbered decision records, including every failed approach | *Why* things are the way they are |
| `README.md` | Endpoints, pipelines, setup, troubleshooting table | What exists |
| `INCIDENT-2026-08-20-neuron-budget-blowout.md` | Evidence, timeline and diagnosis runbook for the worst outage | What to do when AI pages fail |

`AGENTS.md` is a short entry point that sends auto-loading agents here.

When documents disagree about **current behaviour**, the source code and the live `/health-detailed` endpoint win. When the question is **why**, `DECISIONS.md` wins. Section 15 lists what is still unconfirmed.

---

## 1. What this project is

A single Cloudflare Worker (TypeScript, no framework, zero runtime dependencies) that renders content for two e-ink panels owned by one person. It is a personal project on Cloudflare's **free tier**, and staying on the free tier is a deliberate choice that shapes most of the engineering.

The headline feature is **"Moment Before"**: every day the Worker picks a famous historical event that happened on this date, has an image model illustrate it in a rotating art style, and converts the result to something an e-ink panel can show well. Alongside it the panels show a weather dashboard, a daily AI city skyline, and steel/trade headlines.

- **Live URL:** `https://eink-dashboard.thiago-oliveira77.workers.dev`
- **Repo:** `Rabbit-Olive-AI-Devs/rabbitolivestudios.github.io` (public; moved from `rabbitolivestudios/…` on 2026-09-10, old URL redirects). The project lives in the `eink-dashboard/` subdirectory; the repo root holds an unrelated website.
- **Branch:** `main`. Deploys are done from a local machine with `wrangler`, not by CI.

## 2. The two panels, and how pixels reach them

| | E1001 | E1002 |
|---|---|---|
| Hardware | reTerminal E1001, 7.5" mono ePaper | reTerminal E1002, 7.3" E Ink Spectra 6 (6 colours) |
| Resolution | 800x480 | 800x480 |
| Location | Home, Naperville IL (zip 60540) | Office, Chicago IL (zip 60606) |
| SenseCraft device ID | `20225290` | `20225358` |
| Routes it uses | `/fact.png`, `/weather`, `/skyline-bw` | `/color/moment`, `/color/weather`, `/skyline`, `/color/headlines` |

The exact pagelists live in the SenseCraft HMI web app, not in this repo. The routes above are what the renderer was seen requesting in `wrangler tail` on 2026-09-10; confirm with the owner before assuming a route is or is not on a panel.

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
| `/fact.png` | E1001 | 4-level greyscale Moment Before PNG (birthday portrait on family birthdays) |
| `/fact` | E1001 | HTML wrapper around `/fact.png` with a text fallback |
| `/fact1.png` | E1001 | 1-bit Moment Before PNG, 6 rotating styles |
| `/color/moment` | E1002 | HTML with the Spectra-6 dithered Moment image inlined as base64 |
| `/skyline`, `/skyline-bw` | E1002, E1001 | HTML with the daily skyline PNG inlined as a `data:` URI |
| `/skyline.png` | — | The skyline PNG itself (`?bw=1`, `?mode=daily\|rotate\|random`) |
| `/color/headlines` | E1002 | Steel/trade headlines, ranked without an LLM |
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
| `env.SEND_EMAIL` | Email Routing | Failure-alert email |
| `TEST_AUTH_KEY`, `ALERT_TO`, `ALERT_FROM` | Secrets | Test-route auth; alert addresses (secrets because the repo is public) |
| `FOOTBALL_DATA_KEY` | Secret | Only for the retired World Cup code |

**Cron triggers** (`wrangler.toml`), all in UTC:

- `35 4,5 * * *` — the daily image warm. It fires *before* Chicago midnight and fills the keys for the date the panels are **about to** ask for (`dailyWarmTargetDate()`), so a panel poll is never the thing that triggers generation (#64). It is one expression with two fire times because the account is at the free plan's cap of 5 cron triggers.
- `5 0,6,12,18 * * *` — refreshes headlines, both weather locations and both devices' telemetry, then runs the failure-alert check.

The daily warm runs its pipelines **sequentially in priority order** (Pipeline A or birthday, Pipeline B, colour moment, skyline, skyline BW), skips any key that already exists, and stops at the first neuron-budget error.

**External data sources:** Wikipedia "On This Day", Open-Meteo (weather, primary), NWS `api.weather.gov` (alerts, and keyless weather fallback), SenseCraft HMI API (battery, indoor temperature and humidity), Steel Industry News / SteelOrbis / Google News RSS (headlines).

## 5. Image pipelines

All pipelines share one LLM-selected event per day through `getOrGenerateMoment()` (KV `moment:v1:DATE`). The LLM writes a **scene-only** prompt; each pipeline prepends its own style. They diverge after that and **must not be cross-contaminated**: a change to one pipeline's style, model or post-processing must not leak into another.

| | A: `/fact.png` | B: `/fact1.png` | C: birthday | D: `/color/moment` | E: skyline |
|---|---|---|---|---|---|
| Model | FLUX.2 klein-9b, SDXL fallback | SDXL | FLUX.2 + up to 4 R2 photos | FLUX.2, SDXL fallback | FLUX.2 + R2 reference photo, SDXL fallback; BW variant is SDXL only |
| Style rotation | 3, by day of year | 6, by hash of date/title/location | 10, by year | 5, by day of year | 14 styles over 30 cities |
| Output | 4-level greyscale PNG | 1-bit PNG (Bayer or threshold per style) | 4-level greyscale | Spectra-6 indexed PNG, Floyd–Steinberg | BW: 4-level; colour: Spectra-6 |
| Cache key | `fact4:v4:DATE` | `fact1:v7:DATE` | `birthday:v1:DATE`, `color-birthday:v1:DATE` | `color-moment:v2:DATE:STYLE` | `skyline:v3:DATE:daily[:bw]` |

Facts that have each cost real debugging time:

- The brand is "Moment Before" but the prompt depicts **the event itself** at its defining moment; calm pre-event scenes were unreadable on e-ink (#7).
- FLUX.2 takes multipart FormData, SDXL takes JSON. SDXL has no `negative_prompt`, so negatives go in the positive prompt, and it caps at 20 steps.
- **`env.IMAGES.input()` must be given a ReadableStream.** Raw bytes throw `undefined (reading 'font')` *after* the AI call has been billed. Always wrap with `bytesToImageStream()` from `src/images-input.ts` (#59).
- **One FLUX attempt, then fall back. No retry loops.** When changing a retry or budget pattern, grep every call site; a past fix missed four of five (#38).
- Floyd–Steinberg is right for the colour panel and wrong for the mono panel, where Bayer 8x8 or a histogram threshold is used (#4, #14).
- Captions are drawn **after** dithering, or mapped by nearest colour, so diffused error cannot corrupt them (#64).
- Workers cannot run native image libraries; the PNG encoder and decoder are hand-written (`png.ts`, `png-decode.ts`) and the caption font is an 8x8 bitmap.

## 6. Caching

Dates in cache keys are **America/Chicago** dates. Cache versions live in `src/cache-keys.ts`.

| Kind | Keys | Soft TTL (freshness) | KV `expirationTtl` (survival) |
|---|---|---|---|
| Daily images and data | `fact4:`, `fact1:`, `color-moment:`, `skyline:`, `birthday:`, `moment:`, `headlines:v3:DATE:PERIOD` | 24h (headlines 6h) | 604800 (7 days) |
| Weather | `weather:{zip}:v2` | 15 min | 86400 |
| Alerts, device telemetry | `alerts:{zip}:v1`, `device:{id}:v1` | 5 min | 86400 |
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
| KV writes | 1,000/day | ~590 | The tightest budget. Mostly weather/alerts/device caches refreshed by two panels polling every 15 minutes. About 400 spare (#62). |
| KV lists | 1,000/day | 0 on a healthy day | Only the stale fallback lists, and its listings are memoised for 6h. |
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
npm run test:utils        # compiles to /tmp, runs node --test; 112 tests at v3.16.5
npm run dry-run

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
| An alert email arrives | It names the missing images or the armed budget block. Start at the first row of this table. |

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

## 15. Unconfirmed points and open items

**Unconfirmed as of 2026-10-02:**

- The pagelists in section 2 are inferred from renderer requests, not read from SenseCraft HMI.
- Decision records #31 and #33 give older skyline style counts (15 and 18). They are history and are left as written; `src/skyline.ts` defines 14 (6 BW + 8 colour).
- Decision record #38 describes `/color/headlines` as disabled with a redirect to `/skyline`. That was reversed in #39: it is a live page with deterministic, non-LLM ranking, warmed by the 6-hourly cron. Whether it is on a panel's pagelist is a question for the owner.

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
| Tests | `tests/*.test.js` (pure-function tests run with `node --test`) |
| Design notes for past features | `docs/superpowers/plans/`, `docs/superpowers/specs/` |
