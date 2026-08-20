# Incident Report — Workers AI Neuron Budget Blowout

**Date:** 2026-08-20
**Version at fault:** v3.15.21 · **Fixed in:** v3.15.22, hardened in v3.15.23
**Decision record:** [DECISIONS.md #57](DECISIONS.md)
**Severity:** Both displays showed error screens instead of content for ~9 hours; one page (skyline) stayed blank for ~19 hours.

---

## 1. Summary

Every AI-generated page on both e-ink displays failed at once. The Worker had exhausted the
Cloudflare Workers AI **free allocation of 10,000 neurons/day** and was returning 503 on all
image routes.

The overspend was not caused by a pricing change or a traffic spike. It was caused by a
**daylight-saving mismatch between the cron schedule (UTC) and the cache keys (America/Chicago)**.
For 65 minutes every summer day, the day's cache keys were already cold but the warm-up cron had
not yet run — so the two display devices generated the images themselves on the request path,
through a lock that is best-effort by design, duplicating the work.

A separate flaw then made the outage self-perpetuating: the internal budget block expired on a
fixed 6-hour timer, while Cloudflare's quota only resets at 00:00 UTC. Each time the block lifted
early, the devices retried and burned more neurons that were never going to be granted.

---

## 2. Impact

| Surface | Route | Behaviour during incident |
|---|---|---|
| E1002 (color) | `/color/moment` | `503 Color moment temporarily unavailable` |
| E1002 (color) | `/skyline` | HTML rendered, `<img>` broken (`/skyline.png` → 503) |
| E1001 (mono) | `/fact.png` | `503 Failed to generate image` |
| E1001 (mono) | `/fact1.png` | `503 Failed to generate image` |
| Both | `/weather`, `/color/weather` | **Unaffected** (no AI dependency) |
| Both | `/clean`, `/health` | **Unaffected** |

The user noticed and reported the two color-display pages. The mono display had the identical
failure and was found during investigation.

---

## 3. Timeline (2026-08-20, UTC — Chicago is UTC−5 in CDT)

| UTC | Chicago | Event |
|---|---|---|
| 05:00 | 00:00 | Chicago date rolls to `2026-08-20`. Every daily cache key goes cold. |
| 05:00–06:05 | 00:00–01:05 | **The 65-minute hole.** Devices poll every 15 min, miss the cold keys, and generate on the request path. `withGenerationLock` fails to dedupe concurrent polls: **8 FLUX.2 calls, 9,636 neurons** (vs. the usual 4 / ~4,136). |
| 06:05 | 01:05 | Daily warm cron finally runs. Its FLUX.2 call is **rejected** (billed 0 neurons) — the remaining allowance can't cover another image. Cron arms the AI budget block. |
| 06:05–12:05 | 01:05–07:05 | Block active. All AI routes 503. Displays show error screens. |
| 12:05 | 07:05 | **6-hour block expires — but the real quota is still spent.** |
| ~12:05–12:15 | 07:05–07:15 | Devices immediately retry. A further **2,773 neurons** burned on generations that cannot succeed. |
| 12:15 | 07:15 | `/fact.png` request hits 4006 again; block re-armed until 18:15 UTC. |
| 13:48 | 08:48 | User reports the two color pages. Investigation begins. |
| ~14:16 | ~09:16 | v3.15.22 deployed. `/fact.png`, `/fact1.png`, `/color/moment` recover immediately via stale fallback. |
| 00:00 (Aug 21) | 19:00 | Cloudflare quota resets; generation becomes possible again. |

---

## 4. Evidence

### 4.1 The block marker (KV `ai-budget:v1:block`)

```json
{
  "source": "fact.png",
  "message": "4006: you have used up your daily free allocation of 10,000 neurons, please upgrade to Cloudflare's Workers Paid plan if you would like to continue usage.",
  "createdAt": 1787228110913,
  "blockUntil": 1787249710913
}
```

`createdAt` = 2026-08-20T12:15:10Z, `blockUntil` = 18:15:10Z. The `source` is `fact.png` — a
**request-path** call, not `cron ...` — which was the first clue that the devices, not the cron,
were driving generation.

Note `/health-detailed` deliberately does **not** expose `message`; reading the raw KV value is
the only way to see the upstream error text.

### 4.2 Daily neuron usage (GraphQL `aiInferenceAdaptiveGroups`)

| Date | FLUX.2 calls | FLUX.2 neurons | SDXL calls | LLM neurons | Day total |
|---|---|---|---|---|---|
| 2026-08-10 | 4 | 5500 | 2 | 55 | **5555** |
| 2026-08-11 | 4 | 5500 | 3 | 55 | **5555** |
| 2026-08-12 | 4 | 5500 | 2 | 53 | **5553** |
| 2026-08-13 | 4 | 5500 | 4 | 56 | **5556** |
| 2026-08-14 | 4 | 5500 | 2 | 55 | **5555** |
| 2026-08-15 | 4 | 5500 | 2 | 55 | **5555** |
| 2026-08-16 | 5 | 5500 | 3 | 55 | **5555** |
| 2026-08-17 | 4 | 5500 | 2 | 56 | **5556** |
| 2026-08-18 | 4 | 5500 | 3 | 56 | **5556** |
| 2026-08-19 | 5 | 5500 | 4 | 59 | **5559** |
| 2026-08-20 | **14** | **12409** | 5 | 56 | **12465 — OVER** |

Ten days of near-perfect stability at ~5,555, then a single day at 12,465.

> The 14 calls on 08-20 include several that were **rejected and billed 0 neurons** (the 06:00
> cron attempt, the post-fix retries, and two calls from `--test-scheduled` verification during
> the investigation at 14:00 UTC). Actual billed work was ~9 image generations.

### 4.3 Hourly breakdown — this is what identified the culprit

| Hour (UTC) | Model | Calls | Neurons | Reading |
|---|---|---|---|---|
| 2026-08-19 05:00 | flux-2-klein-9b | 4 | 4136 | normal day: devices generate in the hole |
| 2026-08-19 06:00 | flux-2-klein-9b | 1 | 1364 | cron fills the one remaining image |
| **2026-08-20 05:00** | **flux-2-klein-9b** | **8** | **9636** | **the anomaly — duplicated generation** |
| 2026-08-20 06:00 | flux-2-klein-9b | 1 | 0 | cron call rejected (4006) |
| 2026-08-20 12:00 | flux-2-klein-9b | 3 | 2773 | wasted retries after the block lifted early |

The decisive observation: **the calls cluster in the 05:00 UTC bucket, before the 06:05 cron.**
The cron was never the generator on any day — the devices always beat it. Aug 19 shows the same
shape at half the volume, which is why the problem stayed invisible until the race went badly.

### 4.4 KV cache state at time of diagnosis

```
color-moment:v2:2026-08-14 … 2026-08-19   (6 keys — 7-day TTL, fallbacks available)
fact4:v4:2026-08-14 … 2026-08-19          (6 keys — 7-day TTL, fallbacks available)
fact1:v7:2026-08-14 … 2026-08-19          (6 keys — 7-day TTL, fallbacks available)
skyline:*                                  ZERO KEYS  ← 24h TTL, nothing to fall back to
```

This explained why skyline could not be recovered even with a stale fallback, while the other
three could be.

### 4.5 Ruling out the alternative explanations

- **False-4006 bug.** Cloudflare's community forum carried numerous reports of 4006 being returned
  while dashboard usage showed 0/10k. Ruled out: the analytics API showed **12,409 real neurons**.
- **The 2026-08-18 Workers AI pricing change.** Ruled out: per-call cost was unchanged
  (~1,364 neurons/image before and after). The call **count** tripled, not the price.
- **A retired/renamed model.** Ruled out: the error text was the canonical quota message, and the
  same model IDs had worked hours earlier.

---

## 5. Root cause

```
Cron scheduled in UTC ("5 6 * * *")   +   cache keys dated in America/Chicago
                              │
      In CDT the Chicago date rolls at 05:00 UTC, but the cron runs at 06:05 UTC
                              │
                              ▼
        65-minute window daily where keys are cold and no cron has run
                              │
      2 devices × 15-min poll hit the cold keys → generate on the REQUEST PATH
                              │
      withGenerationLock is best-effort (KV has no atomic put-if-absent, and
      its reads are eventually consistent) → concurrent polls DUPLICATE work
                              │
                              ▼
          8 FLUX.2 generations instead of 4 → 9,636 neurons → 4006
                              │
                              ▼
              6h block expires before the 00:00 UTC quota reset
                              │
              → devices retry → burn 2,773 more → re-block → repeat
```

The `wrangler.toml` comment asserted the schedule was "after midnight Chicago in both CST/CDT".
That was true in CST (06:05 UTC = 00:05 Chicago) and false in CDT (06:05 UTC = 01:05 Chicago).
**The bug was live for roughly half of every year and only surfaced when a lock race went badly.**

---

## 6. Contributing factors

1. **The daily cron was not idempotent.** Pipelines A and B regenerated unconditionally; only the
   color moment and skyline steps checked for an existing key. Any second run doubled the spend.
2. **No graceful degradation on AI routes.** A generation failure returned a bare 503 even when a
   perfectly good day-old image was sitting in KV. The device renders a failed fetch as an error
   screen, so a recoverable condition became a visible outage.
3. **Skyline's TTL had zero overlap.** Written at ~05:00 UTC with `expirationTtl: 86400`, the entry
   expired at the exact moment the next day's key went cold. This violated the rule in DECISIONS #24
   (`expirationTtl` must be >> soft TTL) and left the existing stale-fallback code with nothing to find.
4. **The budget block's duration was unrelated to the actual quota cycle.** A fixed 6h timer against
   a 00:00 UTC reset guarantees at least one pointless retry storm per incident.
5. **No alerting.** The only signal was the user looking at a physical display.

---

## 7. Remediation (all shipped in v3.15.22)

| # | Fix | Files |
|---|---|---|
| 1 | Daily warm fires at **both 05:05 and 06:05 UTC** — one per Chicago UTC offset — so the cache is warm before devices poll. Shipped as a single `"5 5,6 * * *"` expression. | `wrangler.toml`, `src/index.ts` |
| 2 | Daily warm is **idempotent**: Pipelines A and B skip when the key already exists. | `src/index.ts` |
| 3 | Budget block expires at the **next 00:00 UTC** (`nextUtcMidnight()`), not a fixed 6h. | `src/cache-guard.ts` |
| 4 | AI routes **serve the most recent cached image** (up to 7 days back) instead of 503, with `no-store`. | `src/index.ts` (`findRecentDailyPNG`), `src/pages/color-moment.ts` (`findRecentColorMoment`) |
| 5 | Skyline daily **TTL 86400 → 604800**; lookback walks 7 days off the *Chicago* date and no longer probes expired rotate keys. | `src/index.ts` |
| — | `shiftDateStr()` — UTC-based, DST-proof calendar-day arithmetic used by the lookbacks. | `src/date-utils.ts` |

### Deploy note (cost us a round trip)

The first deploy attempt used two separate cron entries and was rejected:

```
code: 10072 — This account has reached the Workers Free limit of 5 cron triggers per account.
```

**This failure still uploads the Worker and only rejects the schedule update.** Production briefly
ran v3.15.22 code on the old 06:05-only schedule. Combining the fire times into one expression
(`5 5,6 * * *`) keeps the account's trigger count unchanged. Always confirm the live schedule after
a deploy that touches `[triggers]`:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  https://api.cloudflare.com/client/v4/accounts/$ACCOUNT/workers/scripts/eink-dashboard/schedules
```

---

## 8. Verification performed

| Check | Result |
|---|---|
| `npm run typecheck` | pass |
| `npm run test:utils` | **56/56** (52 before; +4 for `nextUtcMidnight` and `shiftDateStr`) |
| `npm run dry-run` | pass |
| `wrangler dev --remote` against **real KV** → `/color/moment` | 503 → **200**, log: `serving stale fallback from 2026-08-19 (1d back)` |
| Same → `/fact.png`, `/fact1.png` | 503 → **200**, stale fallback confirmed in logs |
| Served image integrity | `fact.png` 800×480 8-bit gray · `fact1.png` 800×480 1-bit · `color/moment` 800×480 indexed (Spectra-6) |
| Browser screenshot @ 800×480 | Correct render, caption `Paris, France · Liberation of Paris · Aug 19, 1944` |
| `wrangler dev --test-scheduled` → `?cron=5+5+*+*+*` | `Cron: daily image warm (5 5 * * *)` — new trigger dispatches as daily |
| Same → `?cron=5+0,6,12,18+*+*+*` | `Cron: periodic data refresh` — 6h trigger still skips images |
| Idempotency (seeded local KV, re-ran cron) | `Cron: Pipeline A already cached` / `Pipeline B already cached` |
| Skyline miss-path latency after read reduction | 13.1s → **1.1s** |
| Production after deploy | `/color/moment` **200**, `/fact.png` **200**, `/fact1.png` **200**, `/skyline.png` 503 (expected — no cache exists) |
| Live schedules | `5 5,6 * * *`, `5 0,6,12,18 * * *`, `*/15 * * * *` |

---

## 9. Residual risk / watch items

1. **`/skyline` stays blank until the quota resets** (00:00 UTC / 19:00 Chicago on 2026-08-20).
   Its cache was empty when the outage began, so there is nothing to fall back to. The 7-day TTL
   only helps from the next successful generation onward. **Self-heals; no action needed.**
2. **UTC-day 2026-08-21 will contain two generation sets.** Tonight's catch-up for Chicago Aug 20
   (at the 00:00 UTC reset) plus Aug 21's warm at 05:05 UTC both fall inside the same UTC quota
   day: roughly **8,200 neurons against 10,000**. It should fit now that duplicate generation is
   prevented, but this is the one day worth checking. Verify via `/health-detailed` →
   `config.ai_budget`, or the GraphQL query in §10.
3. **Structural headroom is thinner than it looks.** A healthy day is ~5,555 neurons — 55% of the
   cap — and a single duplicated FLUX.2 generation costs another 1,364. The margin absorbs about
   three accidental extra generations, no more.
4. **`withGenerationLock` remains best-effort.** KV cannot express atomic put-if-absent. The fix
   removes the *conditions* that made races likely (cold keys during a device-poll window); it does
   not make the lock correct. Any future path that leaves daily keys cold while devices poll will
   reintroduce this.
5. **Still no alerting.** Detection depended on the user looking at a display.

---

## 9b. Follow-up hardening (v3.15.23, DECISIONS #58)

A review of the #57 remediation found two gaps, both since closed.

**The fallback disabled itself on every cache-key version bump.** The lookback *rebuilt* past
keys via `fact4CacheKey(prevDate)` and friends, which embed the **current** version — so bumping
`FACT4_CACHE_VERSION` to `v5` made it search for `fact4:v5:…` keys that had never existed. Since
CLAUDE.md requires a version bump after any pipeline change, and a freshly changed pipeline is
what most often fails, the safety net switched itself off at the moment of maximum risk. Keys are
now resolved by **KV prefix** (`src/stale-cache.ts`), which needs no list of old version strings
and self-heals across bumps. Verified against real KV holding only `v4` keys with the code bumped
to `v5`: `/fact.png` still served `fact4:v4:2026-08-19`.

**The wrapper pages still emitted images known to be dead.** `/fact`, `/skyline` and
`/skyline-bw` are thin `<img>` pages; when the image route fails the panel shows a broken-image
glyph — exactly what was reported as "an HTML error". They now render a plain 800x480 text page
when generation is blocked *and* nothing cached remains. The check is conservative: a cold cache
on a healthy day still renders the image, because the route will simply generate it. The page
carries a Chicago timestamp, so a static error screen is distinguishable from a panel frozen on
an old frame — the failure that caused the image retention in #56.

Residual risks 1–5 above are unchanged; these were separate latent faults, not causes of this
incident.

## 10. Runbook — diagnosing this class of failure again

**Symptom: several/all AI pages fail simultaneously.**

1. **Check the block state** (fast, no auth):
   ```bash
   curl -s https://eink-dashboard.thiago-oliveira77.workers.dev/health-detailed | jq '.config.ai_budget, .daily_images'
   ```
2. **Read the actual upstream error** — `/health-detailed` does not expose it:
   ```bash
   npx wrangler kv key get "ai-budget:v1:block" \
     --namespace-id=de97776d35af4df08b13fd2158acebdc --remote
   ```
3. **Do not trust a 4006 at face value** — Cloudflare has a widely-reported false-4006 bug.
   Confirm real consumption:
   ```bash
   TOKEN=$(grep -m1 'oauth_token' ~/Library/Preferences/.wrangler/config/default.toml | sed 's/.*= *"//; s/"//')
   curl -s https://api.cloudflare.com/client/v4/graphql \
     -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     --data '{"query":"query { viewer { accounts(filter: {accountTag: \"f22a506dedde3bb3837157cd47d5fe5c\"}) { aiInferenceAdaptiveGroups(limit: 10000, filter: {datetimeHour_geq: \"2026-08-20T00:00:00Z\"}, orderBy: [datetimeHour_ASC]) { count sum { totalNeurons } dimensions { datetimeHour modelId } } } } }"}'
   ```
4. **Read the hourly buckets, not just the daily total.** They reveal *who* generated:
   the hour containing Chicago midnight = devices on the request path; the cron hour = the cron.
5. **Compare against baseline.** A healthy day is 4–5 FLUX.2 calls / ~5,555 neurons. Diverging
   *call count* points at duplicate generation; diverging *cost per call* points at a pricing or
   model change.

### Reference numbers

| Item | Value |
|---|---|
| Free allocation | **10,000 neurons/day**, resets **00:00 UTC** |
| FLUX.2 klein-9b | **~1,364 neurons** per 1024×1024 image — effectively 100% of the bill |
| SDXL | **0 neurons** observed across 11 consecutive days (unverified whether genuinely free or simply unreported) |
| Llama 3.3 70B | ~56 neurons/day total |
| Healthy daily total | ~5,555 neurons |
| Daily FLUX.2 generations | 3 unique (Pipeline A, color moment, color skyline); BW skyline is SDXL-only |

---

## 11. Lessons

1. **A cron scheduled in UTC that fills cache keyed by a DST-observing timezone is only correct for
   half the year.** Either derive the schedule from the same timezone as the keys, or cover both
   offsets and make the work idempotent. This generalizes to any UTC-scheduled job with
   locale-dated state.
2. **If a warm-up job is late, its clients become the generators.** Anything expensive behind a
   cache is implicitly protected by the warm-up running *first*. Measure the gap between when a key
   goes cold and when the job fills it — that gap is the exposure window.
3. **Back-off timers must track the resource's real reset cycle.** A fixed-duration block against a
   fixed-time quota reset guarantees wasted retries.
4. **A cache with a TTL equal to its refresh period has zero overlap.** Skyline's 24h TTL expired at
   the precise moment the next key went cold, silently disabling a fallback path that had been
   written, tested, and assumed to work. Retention must exceed the refresh interval by a wide margin.
5. **A display that screenshots a URL has no error handling.** Any non-200 is a visible outage.
   Serving stale content is almost always better than serving a status code.
6. **Instrument for the question "who did this?", not just "how much?"** The daily total said the
   budget was gone; only the hourly-by-model breakdown revealed that the devices, not the cron, had
   been generating the images all along — including on the days that looked healthy.
