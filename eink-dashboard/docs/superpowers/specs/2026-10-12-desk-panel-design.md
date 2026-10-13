# Desk Panel (third display) — Design

**Date:** 2026-10-12
**Status:** Sections 1–4 agreed with the owner. Sections 5–8 are a **proposal awaiting the owner's review**. Nothing here is built.
**Branch:** none yet

## 1. What this is

The owner has a third panel: a new **reTerminal E1001** (mono, 800x480). It will sit on the
desk in the home office. It must not repeat what the two working panels show (Moment Before,
weather, skyline, steel headlines, birthday portraits) or what the owner's Android photo frame
shows (family calendar, chores, lists, photos, travel time).

Success means: four pages that are worth a glance several times a day, that look like one
designed product, and that cost the free tier almost nothing.

## 2. Decisions already made

| Decision | Choice | Why |
|---|---|---|
| Pages | NASA planet of the day, TARS dashboard, Aurebesh word of the day, Flamengo board | Picked by the owner from a longer brainstorm |
| Visual style | "Chokepoint Watch" targeting-HUD look | Owner's reference: <https://x.com/adeebrq/status/2109079189236400244> |
| Platform | SenseCraft HMI + this Worker (same as the two panels) | Free, known, full control over every pixel |
| Build order | Style kit + Aurebesh, then planet, then TARS, then Flamengo | Prove the style on the easiest page, on the real panel, first |

**The style, translated for a 1-bit panel.** The reference uses stippled dot terrain, a crosshair
with corner brackets, tick-mark rulers, monospace capitals, big numerals with tiny unit labels,
hairline rules, hatched dividers, "+" registration marks and one orange accent on light grey.
On this panel: every grey becomes pure black on white, the accent becomes an inverted black
block with white text, the smallest labels grow, and each page carries one picture and two or
three data blocks instead of six.

### Rejected

- **TRMNL firmware + its Claude connector.** It runs on the E1001 (Seeed documents it), but
  needs a US$50 per-device licence that was listed as sold out on 2026-10-12, gives less control
  over the exact look, and drives only one button. Still a reasonable route for quick extra
  screens later.
- **Custom firmware.** The only way to use the buzzer and buttons, and by far the most work.
  Ideas that depended on it (goal buzzer, frost alarm, "on my way" button) were dropped.
- **Daily Star Wars opening crawl.** Needs motion; it belongs on the photo frame.

## 3. The four pages in one line each

1. **Aurebesh word of the day** — one word in the Star Wars alphabet, large, with its
   translation and a decoding key.
2. **NASA planet of the day** — one real exoplanet from NASA's public archive, drawn as a dotted
   sphere inside the crosshair, with a "target inspector" block of facts.
3. **TARS dashboard** — the owner's two Macs (storage, memory, uptime, last seen) and AI
   subscription quotas with reset countdowns, in TARS-style readouts.
4. **Flamengo board** — next match, last result, table position, countdown. No live score: the
   panel refreshes about every 15 minutes.

## 4. What is known about TARS data (for the later TARS spec)

- AI quotas come from **CodexBar** (<https://github.com/steipete/CodexBar>), installed on both
  Macs. `codexbar dashboard --identity redacted` prints one JSON snapshot (schema v1) in about
  12 seconds, headless, with per-provider plan, usage windows (`usedPercent`, `resetAt`), credits
  and estimated spend.
- Quotas are per account, so either Mac can report them. Spend for some providers is estimated
  from each machine's local logs, so it differs per machine and must not be blindly summed.
- One Mac is always on; the other sleeps during the day. Both run the same reporter; the Worker
  keeps the newest good reading **per provider** and shows a machine that stops reporting as
  "last seen HH:MM".
- Each report is one KV write. Two machines at 30-minute intervals is about 100 writes a day
  against roughly 400 spare (DECISIONS #62).
- The page shows plans and spending, so it must sit behind a secret key in its URL. The repo
  and the Worker URL are public.

## 5. Proposal: how the pages are drawn

**The Worker draws every pixel itself and serves a pure 1-bit PNG.**

DECISIONS #48 established that SenseCraft's cloud browser turns anti-aliased text edges into
fog on the mono panel, that no CSS can prevent it, that it cannot be reproduced in a local
browser, and that the only crisp result was an image we rendered to pure black and white
ourselves. A style made of hairlines, small capitals and dots is the worst case for that fog.
An image with no grey in it leaves the cloud renderer nothing to smudge.

| Approach | Verdict |
|---|---|
| **A. Worker-drawn 1-bit PNG** (own canvas, own bitmap type) | **Proposed.** Crisp by construction; no AI, no browser binding; pure functions that unit-test well. Costs a small drawing library and hand-placed layout |
| B. Live HTML/SVG like `/weather` | Fastest to write and finest typography, but fog risk is highest exactly for this style and can only be discovered on the device |
| C. Screenshot via Cloudflare Browser Rendering, then threshold | Proven for the World Cup page, but the `[browser]` binding was removed at retirement (#61) and it adds a heavy dependency for pages that are mostly lines and dots |

The known weakness of A is type: the existing 8x8 font was rejected for the World Cup page,
where it had to carry long proportional names. Here the style calls for short monospace
capitals, which suits a bitmap font, but that is a judgement to make on the real panel, not in
this document (see 6.3).

## 6. Proposal: step 1, the shared style kit

### 6.1 Module layout

```
src/hud/
  canvas.ts     — 800x480 1-bit buffer (0 = black, 1 = white, as encodePNG1Bit expects) and
                  primitives: pixel, hline, vline, line, rect, fillRect, invertRect, circle
  parts.ts      — the reusable pieces of the look: frame with "+" registration marks, corner
                  brackets, crosshair, tick ruler, hatched divider, title bar, inverted accent
                  block, big-number-with-unit block, dotted (ordered-dither) fill
  page.ts       — encode the buffer and wrap it for the panel
```

Each file has one job. `canvas.ts` knows nothing about the look; `parts.ts` knows nothing
about any page; a page module only composes parts.

### 6.2 How a page reaches the panel

A page is one HTML document with the PNG inlined as a `data:` URI and a text fallback, the
pattern `/skyline` uses (DECISIONS #58, #65). The image is a few kilobytes, mostly white.
Routes live under `/desk/`.

The pages in steps 1 and 2 are pure functions of the Chicago date, so they are drawn on each
request: **no KV reads or writes, no AI calls, no cron, nothing to warm and nothing to alert
on.** If drawing proves too slow for the free plan's CPU limit, a daily KV entry is the
fallback, at one write a day.

### 6.3 First deliverable: a calibration card

`/desk/test` renders one screen that puts the open questions side by side: the 8x8 font at 1x,
2x and 3x; a taller bitmap font candidate; 1px and 2px rules; dot fills at several densities;
an inverted block; a hatched band. The owner looks at it on the real panel and the choices are
made from that photo. It stays in the codebase as the reference card for later pages.

## 7. Proposal: step 2, the Aurebesh page

**Route:** `/desk/aurebesh`

**Layout:** title bar (`AUREBESH / WORD OF THE DAY`, date); the word in large Aurebesh glyphs
inside corner brackets; the English word beneath in monospace capitals; a key strip along the
bottom showing each letter used with its glyph.

**Glyphs:** the 26 single letters, each defined in `src/hud/aurebesh.ts` as strokes on a small
grid and drawn with the canvas line primitive, so they scale to any size with no font file.
Digraph letters (CH, SH, TH and others) are left out. `/desk/aurebesh?sheet` renders all 26
with their Latin letters so the owner can check them against a reference chart; they are drawn
from memory and must be verified, not assumed. The letterforms originate from Star Wars; the
stroke data is our own, no font or film artwork is committed, and the use is personal and
non-commercial.

**Words:** a curated list in `src/hud/aurebesh-words.ts`, English, letters A–Z only, 3 to 9
letters so the word fits one line. The day of the year picks the word.

**Test parameters:** `?date=YYYY-MM-DD` and `?word=HELLO`, free and unauthenticated like the
other no-AI test parameters.

## 8. Proposal: testing and done criteria for steps 1 and 2

- Unit tests (`tests/hud.test.js`): primitives stay inside the buffer; the frame is drawn where
  expected; every letter A–Z has stroke data; the word list is A–Z only and within length; the
  same date always picks the same word; the page output is an 800x480 1-bit PNG.
- `npm run typecheck`, `npm run test:utils`, `npm run dry-run` pass.
- Browser check at exactly 800x480: one network request, nothing clipped.
- **On the panel:** the owner pairs the new device in SenseCraft HMI, adds the page, and
  confirms the calibration card and the Aurebesh page are sharp. Only then is the style kit
  considered proven and step 3 started.
- Docs in the same commit: README endpoints, CLAUDE.md file map, HANDOVER, a DECISIONS record
  for the drawing approach. A version bump needs the owner's approval.

## 9. Still open

- Font choice and line weights: decided from the calibration card.
- Whether the free football feed covers the Libertadores as well as the Brazilian league.
- Which exoplanet fields NASA's archive returns reliably enough to show every day.
- Whether a locked (not sleeping) Mac still runs the reporter.
- The new panel's SenseCraft device ID, once paired.

Steps 3, 4 and 5 each get their own design before they are built.
