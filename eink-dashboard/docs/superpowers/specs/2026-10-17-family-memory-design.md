# Family Memory: a family photo from this day, redrawn as art (design)

**Status:** design for the owner (2026-10-17), not built. Builder check done; the owner answered its
two questions (wide photos only; favourites only, a curated album later).

## Intent

Each day, a new page shows one family photo from the owner's iPhone Photos library, taken on this
date in an earlier year, redrawn as art. It works like the iPhone's "memories", but as a painting
on the panel. The owner picked it from a list of ideas, and it is the family counterpart of the daily
moment and skyline.

## What the owner decided

- **A colour page only**, at a new route (`/color/family`), drawn like every panel picture since #81:
  the art cropped to 800x480 in full colour, untouched, and each panel converts it. The owner adds it
  to the **office E1002** and the **home E1001** (not the desk panel) in SenseCraft.
- **Family photos only:** photos where Photos has recognised at least one family member, meaning the
  nine people of the birthday list. The names map to Photos People through
  `photos/portraits/people.json` (gitignored), the same map as #87. In the library there are several
  `ZPERSON` rows for some names; all of them count.
- **This date in an earlier year.** If no family photo exists for the exact date in any earlier year,
  use the **closest date** that has one, by calendar distance within the year, looking either way and
  wrapping around New Year.
- **The caption shows the photo's real date and how long ago**, for example
  `15 Oct 2019 · 7 years ago` ("1 year ago" singular). **No names** in the caption.
- **Wide photos only (owner, 2026-10-17):** landscape photos (wider than tall, as displayed). They
  crop to the panel's 5:3 without cutting heads. Tall photos are never picked.
- **Favourites only, for now (owner):** only photos the owner has marked with the heart in Photos.
  Nobody reviews the photo before it airs, including on the office panel, so the owner curates by
  marking. **The owner will make a curated album later.** The source must be one setting (favourites,
  or a named album), so switching changes nothing else. Photos newly marked or added are picked up
  automatically on the next evening.
- **Repeats (owner accepted the proposal):** every photo in the source is shown once before any
  photo repeats. After that, the photo shown longest ago comes back first, still choosing by the
  closest date.
- **A different style:** the art style is never the same as that day's moment or skyline style.

## Library facts (checked 2026-10-17 on a read-only copy of Photos.sqlite)

- **All family photos:** 17,205 still photos with a family face (2011-2026), 8,231 with two or more
  family members.
- **The source, favourites that are wide:** 778 family favourites, of which **538 are wide**. They
  fall on only **164 days** of the year, so on most days "the closest date" is days or weeks away; the
  caption still gives the real date. At one a day the pool is used up in about 18 months, hence the
  repeat rule. (The width check used `ZWIDTH > ZHEIGHT`; the builder should confirm those columns are
  already display-oriented.)

## How it works

It reuses the Jarvis picture route (#89): the Worker decides, Jarvis draws, Workers AI is not
needed.

1. **Pick (Mac mini, evening, before the brief).** A new step runs before the 17:00 `art-brief` (for
   example as part of the same launchd job). It reads a fresh **read-only copy** of `Photos.sqlite`,
   as `scripts/pick-portraits-from-photos.py` does, and never touches the live database. It picks
   tomorrow's photo with these rules:
   - **Candidates:** still images (no videos or screenshots) from the source (favourites, or the
     curated album once set), wide, not hidden or trashed, with at least one family face, and **at
     least 11 months old** (Claude's call: a photo from 31 Dec can never appear on 1 Jan as "0 years
     ago"). "N years ago" is the whole years between the two dates, at least 1.
   - **Day:** among the photos not yet shown in the current round (all of them once every photo has
     been shown, ordered by longest since shown), the closest calendar day to the target.
   - **Ranking within that day:** more family faces first, then larger and sharper faces. Exact
     weights are the builder's call.
   - **The photo itself:** exports the original upright (EXIF applied) at a usable size. The long side
     is resized to about 1600 px; reject anything under 1024 px and move to the next candidate. The
     export uses the AppleScript route (#87) when the original is iCloud-only.
   - **Upload:** puts the JPEG to R2 `PHOTOS` at `family/<date>/photo.jpg`, with
     `family/<date>/photo.json` = `{ "taken": "YYYY-MM-DD" }`. Nothing else about the photo leaves the
     Mac: no names and no location. It uses the mini's own Cloudflare login (`wrangler r2 object put`),
     as the art delivery does.
   - **Log:** records the asset's UUID and the date shown in a local log on the mini (outside the
     repo, e.g. `~/.config/eink/family-shown.json`), used by the repeat rule.
   - **Timing:** it must finish before the 17:00 `art-brief`, because `cmdArtBrief` does not resend a
     brief that is already complete.

   It runs on the target date's eve, so a missed evening (the mini locked after a restart) means no
   photo for that date.
2. **Brief (Worker).** `buildArtBrief` adds a `family` picture when `family/<date>/photo.jpg` exists.
   That includes birthdays and holiday weeks: Family Memory is its own page and is never themed.
   - **Style:** a schedule pick from the shared library (`src/art-styles.ts`, portrait-friendly
     styles, since faces matter), with its own salt, skipping that date's moment style and skyline
     style.
   - **Prompt:** redraw the reference photograph faithfully in that style: the same people, faces,
     poses, clothing and composition, with the likeness kept. Only the medium changes. No text,
     borders or added people, plus the `ART_RULES`.
   - **Brief fields:** `references: 1`.
   - **The watcher's job:** `art-brief` copies the photo from R2 to `from-claude/art/<date>-family-ref.jpg`,
     as it does the birthday references, and lists `<date>-family.png` among the files to deliver.
3. **Draw (Jarvis).** Delivers `to-claude/art/<date>-family.png` with the other pictures. The
   watcher's `ART DELIVERY` handling accepts the new kind and puts it to `art/<date>/family`.
4. **Warm (Worker cron, the existing nightly warm for the next date).**
   - **Which picture:** Jarvis's picture from `art/<date>/family` if delivered, marked in
     `art/used/<date>/family`. Otherwise the original photo itself, cropped and captioned the same
     way, with no AI.
   - **Where it goes:** stored under a new KV key `family:v1:<date>`, 7-day TTL, one write a day. The
     R2 photo is then deleted, and so is Jarvis's file, as the other kinds do.
   - **No photo for the date:** nothing is stored, and the page falls back as below.
5. **Page (`/color/family`).**
   - **One request:** the PNG is inlined as a `data:` URI (#65). The caption bar is burned in as solid
     black and white like the skyline's: date left, "Family Memory" centre, "N years ago" right, ASCII
     only (the caption font is the 8x8 bitmap, so the `·` above becomes a plain separator).
   - **Never generated on the request path:** when today's key is empty, it serves the most recent
     earlier entry by prefix (`stale-cache.ts`). With none at all, it serves the e-ink text page
     (`pages/unavailable.ts`), never a broken image.

## Constraints

- **No Workers AI spend:** Jarvis draws it, and the fallback is the photo itself.
- **KV:** +1 write a day (DECISIONS #62 budget). No `list` calls on the request path beyond the
  memoised stale index.
- **Privacy:** the family photo goes to R2 (private bucket) and to Jarvis through the mailbox, the
  same as the birthday references the owner already approved. Photos, the shown log, and
  `people.json` are never committed. R2 copies are deleted after the warm.
- **The alert (#60/#90) watches it:** today's `family` image is missing, or Jarvis did not draw it. A
  day with no photo (the mini was down) shows as missing, with the mini-quiet alert as the likely
  cause.
- `/health-detailed` lists it, and `drawn_by` includes `family`.

## Left to the builder

- Ranking weights and the face-quality thresholds.
- Where the pick step lives (in `series-mailbox.mjs` or its own script), and its launchd wiring.
- The exact prompt wording, within the rules above.
- Test coverage. At least: the closest-date search with wraparound and the 11-month rule, the
  repeat rule, the style never equal to the moment's or the skyline's, the caption text, the brief
  with and without a photo, and the page fallbacks.

## Not doing

- No names, places or people tags on the panel.
- No holiday theming, and no change to birthdays.
- No mono (1-bit) version; the panels convert.
- No desk panel.
