# Family Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a daily `/color/family` page: a family photo from this date in an earlier year, redrawn by Jarvis, with the photo itself as the fallback.

**Architecture:**
- **The Mac mini picks.** The pick runs inside the existing `art-brief` job. It reads a read-only copy of Photos.sqlite, downloads the one photo through PhotoKit, and puts it in R2 at `family/<date>/photo.jpg` plus `photo.json`.
- **The Worker.**
  - The brief adds a `family` picture.
  - The watcher accepts `<date>-family.png`.
  - The nightly warm stores Jarvis's art, or the photo itself, captioned, under `family:v1:<date>`.
  - The page serves that key, falling back to the most recent earlier entry, then to the text page.

**Tech Stack:** TypeScript Worker (node --test), Python 3 (stdlib sqlite3, unittest), Swift (PhotoKit), Node mailbox script.

**Spec:** `docs/superpowers/specs/2026-10-17-family-memory-design.md`

## Global Constraints

- **Candidates:** wide only (`ZWIDTH > ZHEIGHT`), still images only (`ZKIND = 0`), not hidden or trashed, and at least 11 months old (taken at least 335 days before the target date).
- **Source order:** the album "Coelhos de Oliveira" on the exact day, then a Favourites family photo (family face from `people.json`) on the exact day, then the album's closest day (calendar distance within the year, wrapping around New Year).
- **Repeats:** a photo isn't picked again until every album photo has been shown. After that, the photo shown longest ago comes first.
- **Caption:** left `15 Oct 2019`, centre `Family Memory`, right `7 years ago` (`1 year ago` singular, at least 1). ASCII. No names.
- **Style:** from `stylesFor("portrait")`, never equal to that date's moment style or skyline style.
- **Storage:** KV `family:v1:<date>`, TTL 604800, one write a day. No Workers AI. Never generate on the request path.
- **Privacy:** never commit photos, the shown log or `people.json`. Never write the album's share link anywhere. Delete the R2 photo after the warm.

## Review Focus

- **The mini was down that evening, so no photo for the date:** the page shows yesterday's entry, and the alert reports `family` missing. Test in Task 2 (page fallback).
- **Jarvis delivers but the photo is gone:** the caption needs `taken`, so `photo.json` must outlive the photo until the warm. The warm deletes both only after storing. Test in Task 1.
- **The album photo can't be downloaded (offline, iCloud error):** the pick moves on to the next candidate and doesn't log a failed photo as shown. Test in Task 3.
- **The date is a family birthday or in a holiday week:** the family picture is still briefed, unthemed. Test in Task 1.
- **The target is in early January:** the closest-day search wraps to December, and the 11-month rule still keeps last December's photos out. Test in Task 3.

---

### Task 1: Worker — family picture, brief, warm store

**Files:**
- Create: `src/family.ts`
- Modify:
  - `src/cache-keys.ts`: add `familyCacheKey`.
  - `src/art.ts`: `ArtKind` gains `"family"`; `buildArtBrief` appends the family picture.
- Test: `tests/family.test.js`

**Interfaces:**
- **Produces:**
  - `familyCacheKey(date: string): string`, returning `family:v1:<date>`.
  - `familyStyleFor(date: string): ArtStyle`.
  - `yearsAgo(taken: string, date: string): number`.
  - `familyCaption(taken: string, date: string): {left, center, right}`.
  - `familyPicture(env, date): Promise<ArtPicture | null>`.
  - `storeFamily(env, date): Promise<"jarvis" | "photo" | null>`.
  - R2 keys: `family/<date>/photo.jpg`, `family/<date>/photo.json` (`{"taken":"YYYY-MM-DD"}`), and `art/<date>/family`.

- [ ] **Tests:**
  - `familyStyleFor` over 400 dates is never `momentStyleFor(d)` or the daily skyline style.
  - `yearsAgo("2019-10-15","2026-10-17") === 7`; `yearsAgo("2025-10-20","2026-10-17") === 1` (minimum 1).
  - Caption is `{left:"15 Oct 2019", center:"Family Memory", right:"7 years ago"}`, and `"1 year ago"` when singular.
  - The brief with a photo in a fake R2 has a `family` picture (`references: 1`, `file: <date>-family.png`).
  - The brief without one has no `family` picture.
  - On 2026-10-20 (Thiago's birthday) the brief has the birthday and family pictures, and the family prompt has no holiday words on 2026-10-31.
  - `storeFamily`:
    - Jarvis art present: it stores and returns `"jarvis"`, deletes `photo.jpg`, `photo.json` and `art/<date>/family`, and writes the `art/used` marker.
    - Only the photo: returns `"photo"`.
    - No `photo.json`: returns `null` and writes nothing.
- [ ] **Run:** `npm run test:utils`, which should fail on the missing module.
- [ ] **Implement.**
  - **Style:** `dailyPick(stylesFor("portrait"), date, 0x46414d)`, stepping to the next style in the pool while it equals the moment or the skyline style.
  - **Prompt:** "a faithful repainting of the reference photograph in <style>: the same people, faces, expressions, poses, clothing and composition, keeping each person's exact likeness; only the medium changes; no added people, no text, no borders".
  - **Store:** `jpegToRGB` (the art or the photo), then `finishSkylinePNG(rgb, familyCaption(...))`, then a KV put.
- [ ] **Run:** `npm run test:utils`, all pass. Typecheck.
- [ ] **Commit.**

### Task 2: Worker — page, warm, alert, health

**Files:**
- Create: `src/pages/family.ts`
- Modify:
  - `src/index.ts`: add the route `/color/family`. The warm calls `storeFamily(env, dateStr)` after the skyline, and it doesn't depend on the AI budget. `dailyImageKeys` gains `family`, `ALERT_WATCHED_IMAGES` gains `"family"`, `jarvisKinds` gains `"family"`, and `drawnBy` gains `family`.
  - `src/alert.ts`: the `jarvisKinds` type and the label mapping gain `family`.
- Test: `tests/family.test.js`, `tests/alert.test.js`

- [ ] **Tests:**
  - The page serves today's key.
  - With no entry for today, the page serves the most recent earlier `family:` entry.
  - With none at all, the page serves the unavailable text page and no `<img>`.
  - In the alert, `jarvisMissed` includes `family` when the family image exists without its marker.
- [ ] **Implement.** The page reuses an inline-image page in the style of `renderSkylineInlineHTML`, with the title "Family Memory". `Cache-Control: no-store`.
- [ ] **Run:** tests, typecheck, `npm run dry-run`.
- [ ] **Commit.**

### Task 3: Mac mini — pick and download

**Files:**
- Create:
  - `scripts/family-pick.py`, with a pure `choose(candidates, shown, target) -> candidate | None` and the DB reading. CLI: `--db COPY --date YYYY-MM-DD --people photos/portraits/people.json --shown ~/.config/eink/family-shown.json`. It prints the candidate list in order, as JSON lines `{uuid, taken, source}`.
  - `scripts/family-fetch.swift`: `family-fetch <localIdentifier> <out.jpg>`, a 1600 px long side through PhotoKit with network access. Exit code 0 means written. It is compiled to `~/Library/Application Support/eink-family/family-fetch` so the Photos permission sticks to one binary.
- Test: `scripts/test_family_pick.py` (`python3 -E -P -m unittest scripts/test_family_pick.py`)

- [ ] **Tests:**
  - Album exact day beats a Favourites exact day.
  - A Favourites exact day beats the album's closest day.
  - The closest day wraps: a target of 01-02 matches album 12-30.
  - The 11-month rule: a photo taken 2026-12-30 is rejected for 2027-01-02.
  - Shown photos are skipped until all album photos are shown, then the one shown longest ago comes first.
  - The order lists fallbacks after the best, so a failed download moves to the next.
- [ ] **Implement and run** the tests until they pass. Run the CLI against a real DB copy for tomorrow and look at the result (UUID and date only).
- [ ] **Commit.**

### Task 4: Mac mini — wire into the mailbox job

**Files:**
- Modify: `scripts/series-mailbox.mjs`
  - `cmdArtBrief`, before fetching the brief and once per date: copy the DB, run the pick, try `family-fetch` down the list until one works, check it's at least 1024 px wide (`sips`), then put `family/<date>/photo.jpg` and `photo.json` to R2. On success, append `{uuid, date, shownOn}` to the shown log. Delete the temporary files.
  - The brief copies `family/<date>/photo.jpg` to `from-claude/art/<date>-family-ref.jpg` and sets `referenceFiles`.
  - `artDelivery` kinds gain `"family"`.
- Test: `tests/series.test.js` if the mailbox parsing changed (it doesn't); otherwise a dry run.

- [ ] **Implement.**
- [ ] **Run on the mini:** `node scripts/series-mailbox.mjs art-brief <tomorrow>` with `ART_FORCE=1` (tomorrow's brief has already gone out). Check that R2 holds `family/<tomorrow>/photo.jpg`, and that the brief JSON lists `family` with a ref file.
- [ ] **Check from launchd:** `launchctl kickstart` the art-brief job. If PhotoKit is refused, the owner grants Photos access to `family-fetch` in System Settings → Privacy & Security → Photos.
- [ ] **Commit.**

### Task 5: Docs and release check

- [ ] **Docs:**
  - README: the endpoint and the family section.
  - HANDOVER: routes, KV key, open items, test count.
  - CLAUDE.md: file map rows plus an obligation row.
  - DECISIONS #92: mark it built.
  - Memory.
- [ ] **Deploy** (`CLOUDFLARE_ACCOUNT_ID=… npx wrangler deploy`). Then check `/color/family` (text page until the first warm) and `/health-detailed`.
- [ ] **Tell the owner** to add `/color/family` to the office E1002 and home E1001 pagelists after the first warm.
- [ ] **Commit and push.**
