# The Series (fifth desk page) — Design

**Date:** 2026-10-14
**Status:** Sections 1–8 approved by the owner in conversation on 2026-10-14; section 9 records the calendar and roles the owner then decided, and the few points still open. **Built and live** as v3.21.0 from `docs/superpowers/plans/2026-10-14-series.md` (DECISIONS #71); one simplification: the between-seasons card is text only. The mailbox is watched on the Mac mini (DECISIONS #72).
**Mockups:** private artifact <https://claude.ai/artifact/Rb2TKnk47BeUvKY8peHRFE> (round 3 is the approved direction; round 4 is a real picture from Muse through all three conversions). The scripts that drew them were throwaway and are not in the repo.
**For the showrunner:** `docs/series/SHOWRUNNER.md` (the process) and `docs/series/SEASON-FORMAT.md` (the exact package the writers deliver).

## 1. What this is

A Star Wars graphic-novel series that plays out on the owner's three e-ink panels, a few panels a day, like a TV series released one scene at a time. Four AI writers, one per three-month season, each write the script **and draw the art in their own style**. The pictures and scripts are made ahead of time on the owner's existing subscriptions (OpenAI, Claude, Muse, Grok), coordinated by the owner's agent **Jarvis** as showrunner through Notion and Slack. The Worker only shows pages that were finished before the season started.

Success means: a page worth glancing at during the working day, a story that rewards following it for months, seasons that feel different from each other but are clearly one show, and no new running cost.

**Working title:** *The Long Drift*. The title, the crew's names and the ship's name in the mockups are placeholders until the bible is written (section 9).

## 2. Decisions made with the owner (2026-10-14)

| Decision | Choice | Why |
|---|---|---|
| Theme | Star Wars, an original crew in the Outer Rim | Matches the Aurebesh page and the owner's taste; "like *The Mandalorian*" |
| Form | A series, not a game: written episodes, not combined parts | A real script reads better than a generator; the owner asked for a series |
| Story shape | Seasons of three months; each week is one chapter (one mission) that moves the season's long journey on | Owner's choice: "weekly missions, part of a long journey trimester season" |
| Hero | An original three-member crew with their own ship | No canon characters; room for banter |
| Writers | Each season by a different AI: Claude, OpenAI, Muse, Grok | Owner's idea; four voices, one show |
| Art | **Each writer draws its own pictures, in its own style** ("a graphic novel series") | Owner's choice; replaced the code-drawn shot library of round 2 |
| Words | **The writers deliver pictures without words; our code letters every caption and balloon** in the panel fonts | Painted lettering garbles and blurs on e-ink; same lettering keeps it one show |
| Layout | Full-page comic: the picture fills the panel, a caption box and balloons over calm zones, a narration strip below | Round 3 mockup, approved |
| Templates | Five page templates with "keep calm" zones for the words | Owner asked for them, so the text always has a place |
| Displays | Desk E1001 (black and white), home E1001 (4 greys), office E1002 (6 colours) | Owner asked for all three; one story clock, so all show the same scene |
| Pace | Weekdays only, working hours: 5 scenes a day at 08, 10, 12, 14, 16 Chicago | Owner: the story rolls while they are at the desk |
| Weekends | Replay of the week, one scene an hour, then a "next week" teaser from Sunday 19:00 | Owner: catch up on missed scenes |
| Source of truth | Notion, run by Jarvis; seasons copied into the project once, when finished (option A) | No runtime dependency on Notion; every season reviewed once before it goes live |
| Calendar | **A 10-week Prologue by Claude premieres Monday 26 October 2026**; Season 1 follows on 4 January 2027 and later seasons start on the first Monday of each quarter | Owner's date and idea ("season 1 could be prologue") |
| Proof sheet | Approved by Jarvis, so the owner watches unspoiled | Owner's choice |
| Runtime | No AI, no KV; pages pre-built at import and stored in R2 | Free-tier CPU, KV and neuron budgets (DECISIONS #62, and the CPU lessons of the colour pipelines) |

### Rejected along the way

- **Story kit computed from the date** (a library of parts combined by a seeded recipe). Free and AI-free, but would feel like a game. Superseded when the owner proposed real writers.
- **The small Workers AI model writing each Monday's mission.** Breaks the "desk pages have no AI" rule and risks drifting into canon.
- **Reading Notion live from the Worker (option B).** Needs a Notion key on the Worker and KV writes, and an unreviewed edit would reach the panels directly.
- **A code-drawn "shot library"** (about 12 line-art scene types picked by tag; round 2). Consistent, but the owner wanted each writer's own art.
- **Writers lettering their own pictures.** Image models garble text; small painted text is unreadable after conversion to e-ink.
- **A scene every 6 hours, around the clock** (28 a week). Replaced by working-hours pacing.

## 3. The story clock

All times are America/Chicago. A season has **13 chapters** (the Prologue has 10); chapter *k* runs in week *k* counted from the season's premiere Monday. Each chapter has **25 scenes**, five per weekday.

| When | Shown |
|---|---|
| Mon–Fri 08:00, 10:00, 12:00, 14:00, 16:00 | That weekday's scenes 1–5 (Monday = scenes 1–5 … Friday = scenes 21–25). Scene 1 of a chapter is always the **Splash** template: the chapter's title card. |
| Mon–Fri 16:00 until 08:00 next weekday | The day's last scene stays up |
| Saturday 00:00–08:00 | Friday's scene 25 |
| Saturday 08:00–21:59 | **Replay** of scenes 1–14, one per hour (08:00 → 1 … 21:00 → 14) |
| Saturday 22:00 – Sunday 08:00 | Replay scene 14 |
| Sunday 08:00–18:59 | Replay of scenes 15–25, one per hour (08:00 → 15 … 18:00 → 25) |
| Sunday 19:00 – Monday 08:00 | The chapter's **"next week" teaser** card. After chapter 13 it is the season finale card. |
| No season running | The **between-seasons** card |

The slot is a pure function of the time, so every panel shows the same page and no state is stored. US clock changes happen at 02:00 on a Sunday, inside the overnight hold, so they move no scene boundary; the tests cover both anyway.

**Season calendar (proposal, section 9):** a season premieres on the first Monday of a quarter. A quarter is 12.9 to 13.1 weeks, so some quarters leave a one-week gap; the between-seasons card covers it.

## 4. The page templates

The top bar (36px) is the desk panels' shared bar: breadcrumb (*Desk / The Long Drift / Chapter 3*), scene number, and an inverted pill (*S1 / CH 03*), plus **REPLAY** on weekend pages. Below it, one of five templates:

| Template | Art (px) | Text zones | Use |
|---|---|---|---|
| **Wide** | 776 × 346 | caption box top-left; balloon zone across the top third; narration strip below the art | The everyday scene |
| **Split** | 2 × 382 × 346 | balloon zone at the top of each panel; narration strip | Conversations |
| **Strip** | 3 × 252 × 346 | a short caption at the top of each panel; narration strip | Action beats |
| **Splash** | 800 × 444 | one caption / title zone, bottom-left | Chapter title cards, premiere, finale |
| **Panel + text** | 476 × 426 | a text column on the right, 300px wide | Long narration: a letter, a log entry |

The narration strip's right-hand block carries the writer's credit ("STORY & ART  CLAUDE"), the chapter title, the time of the next scene and a 25-step progress bar. On Splash and Panel + text those move into the top bar and the text column.

The exact zone rectangles are fixed in code during the build and exported as **template guide images** (one PNG per template at true size, zones hatched) for the writers, so the artists draw to the real geometry.

**Lettering rules:** all words in IBM Plex Mono from `src/hud/fonts-data.ts`, never below `plex11m` (DECISIONS #67); caption boxes are white with a black rule; balloons are white ellipses with a tail aimed at the speaker's anchor point (a voice from off the picture gets a square box with the speaker's name); at most two balloons per panel. Words are drawn **after** each conversion, as solid black and white, so dithering never touches them (the CLAUDE.md rule for bars and text on dithered images).

**Lettering is checked, never trusted** (owner's request after round 4, where fixed-size balloons let text spill over their edges; round 5 shows the fix):

1. **Sized from the text.** Each line is wrapped at several widths (at most four lines); the most balloon-shaped block wins, and the balloon is built around it with a margin. Nothing has a fixed size.
2. **Placed by search.** The importer searches the panel for a spot near the speaker (preferring above them and the template's calm zone) that stays inside the panel and clear of the caption, other balloons, **every character's anchor** and any **keep-clear boxes** the writer marks in the scene (`clear`). The tail runs from the balloon's edge to just short of the speaker's head.
3. **Checked by pixels.** Each balloon's words are first drawn on a scratch canvas; every text pixel must lie at least 4px inside its balloon or box. The same check covers caption boxes, the narration strip and the teaser card.
4. **Refused, not squeezed.** If a line cannot be wrapped into four lines, no spot is free, or any pixel falls outside, the scene is refused with the reason ("FIGURE: line too long for any balloon (178 chars)", "no room for the balloon in this panel"). Text is never shrunk below the house size or cut off to make it fit.
5. **Seen once by a person.** The proof sheet is still reviewed by Jarvis for what code cannot judge: a balloon over an important detail the writer forgot to mark.

## 5. One picture, three panels

The writer delivers each picture once, in full colour. At import it becomes three pages:

| Panel | Route | Conversion |
|---|---|---|
| Desk E1001 | `/desk/series` | 1-bit: Floyd–Steinberg error diffusion after a mild contrast boost |
| Home E1001 | `/series` | 4-level grey, as Pipeline A (`/fact.png`) |
| Office E1002 | `/color/series` | Spectra 6 Floyd–Steinberg, but **pixels with little colour may only use black and white**; only saturated pixels use red, yellow, green and blue |

Round 3 showed flat-colour art converting cleanly to all three, and smooth gradients turning to dot patterns in 1-bit. Round 4 put a real picture from Muse (inked, red sky, blue light) through the conversions: the grey version was clean as it was; Pipeline D's colour method turned every dark grey into coloured speckle, and the desk pages' Bayer dither turned fine ink detail to mush. The two changes in the table fixed both (compare round 4 in the mockups). The house style therefore asks for **flat colour or inked areas, clear outlines, and a limited palette** leaning on black, white, red, blue, yellow and green, with calm zones kept free of gradients.

The new routes are a new pipeline. They share conversion functions with Pipelines A and D but must not change them.

## 6. The season package and the import

The full format is in `docs/series/SEASON-FORMAT.md`. In short, a folder with:

- `season.json`: season number, title, writer, premiere Monday; 13 chapters, each with a title, a teaser line and 25 scenes; each scene with its template, location caption, narration, and per panel an art file (or `hold` to keep the previous picture) and up to two balloons (speaker, line, anchor point in the art).
- `art/`: the pictures, full colour, no words. Image models give fixed shapes (Muse's test was 3:2), so any shape at least as large as the panel is accepted and **cropped to fill**, around an optional `focus` point the writer sets.
- `sheets/`: the writer's character and ship sheets, for review only.

`npm run series-import -- <folder>` runs on the Mac mini. It stops with a list of every problem if any check fails:

1. **Structure:** 13 chapters (10 for the Prologue) of 25 scenes, premiere on a Monday, required fields present, known template names, scene 1 of each chapter is a Splash.
2. **Art:** every referenced file exists and has the template's shape; **at least one new picture per weekday**; no picture that is nearly blank.
3. **Words:** every balloon, box and strip is sized, placed and **checked pixel by pixel** as in section 4; only characters the fonts have; none of the bible's banned names. Tone is the showrunner's review, not code's.
4. **Build:** every page for every panel: 325 scenes, their 325 replay versions, 13 teasers, the between-seasons card (1,992 PNGs, about 200 MB).
5. **Proof sheet:** every scene in order on one private page, for approval before anything goes live.
6. **Upload** to R2, only after approval. Re-running replaces the season; `--chapter N` imports or rebuilds one chapter.

**Delivery in parts.** A season may arrive chapter by chapter: at least the first two chapters before the premiere, then each chapter at least **7 days before it airs**. The checks run per chapter, plus the season-wide ones (numbering, at least one new picture per weekday) on what has arrived.

## 7. Runtime, storage and failure handling

- **Storage:** a new private R2 bucket (binding `SERIES`). Keys: `series/schedule.json` (seasons, premieres, titles, writers) and `series/s01/c03/s14/{desk,home,office}.png`, with `replay-` and `teaser-` variants. R2's free tier (10 GB stored, 10 million reads a month) covers it many times over.
- **Per request:** the Worker reads the schedule and one page from R2 and inlines the PNG as a `data:` URI (DECISIONS #65: one request per page). No KV reads or writes, no AI, and no image work at run time.
- **Missing page:** the nearest earlier scene that exists; if none, the plain text page (`src/pages/unavailable.ts`, DECISIONS #58). Never a broken image.
- **Between seasons:** a card with the previous season's final picture and "Returns DD/MM" when the next season is imported, "Returns soon" when it is not. Importing a season rebuilds the previous season's between-seasons card.
- **Preview:** `?at=YYYY-MM-DDTHH:MM` shows any moment, behind `TEST_AUTH_KEY`, because it reveals future scenes.
- **Alerting (`src/alert.ts`, DECISIONS #60):** the six-hourly check gains three conditions: the page for the current slot is missing from R2; next week's chapter is not imported 7 days before it airs; the current season ends within 14 days with no next season in the schedule. All are R2 reads, not KV.
- **Spoilers:** the repo is public and holds no scripts or art. The R2 bucket is private. Test previews need the key.

## 8. Roles

| Who | Does |
|---|---|
| The owner | Executive producer: approves the bible, the writer order and each season's proof sheet (or delegates the proof sheet; section 9) |
| Jarvis (Hermes agent) | Showrunner: keeps the bible, runs the schedule in Slack, briefs each writer, reviews scripts and art for continuity, tone and the template rules, assembles and delivers the season package |
| The season's writer (an AI) | Writes the chapters and scenes, draws its character and ship sheets first, then every picture, and leaves a handover note for the next writer |
| Claude Code (this repo) | Engineer: the panel pages, the import and its checks, template guides, proof sheets, uploads, alerting |

## 9. Calendar and roles (decided 2026-10-14) and what is still open

**Decided by the owner:**

1. **Writer order:** Claude first, then OpenAI, Grok and Muse, in rotation.
2. **The Prologue:** Claude's first run is a **10-week Prologue** (season number 0), Monday 26 October 2026 to Friday 1 January 2027. The date is the owner's choice. Season 1 (OpenAI) premieres Monday 4 January 2027; Season 2 (Grok) 5 April; Season 3 (Muse) 5 July; then Claude again on 4 October 2027.
3. **Muse can generate images** (the owner shared a test picture; round 4 of the mockups).
4. **Proof sheets are approved by Jarvis**, so the owner watches each season unspoiled. The owner still approves the bible and the writer order.

**Consequences of the 26 October date:** 12 days from this decision. The panel routes, the importer and Claude's first two chapters must be ready by about Friday 23 October; the Prologue is delivered in parts (section 6). The bible can be short for the Prologue and grow before Season 1.

**Still proposals (my defaults unless the owner objects):**

- **Notion layout:** a Bible page and Seasons, Chapters and Scenes databases (`docs/series/SHOWRUNNER.md`).
- **Hand-off folder:** Dropbox, `00_SHARED_INBOX` today; a dedicated *The Long Drift / packages* folder once Jarvis sets it up.
- **Setting:** after the fall of the Empire; canon places, species and ship classes allowed; no canon characters. Muse's test picture shows a hooded figure with a lightsaber, so whether Jedi may appear (as guests, not crew) is for the owner to decide in the bible.
- **Title and crew:** the Prologue (Claude) introduces them; the owner approves.

## 10. Build outline (for the implementation plan)

1. **Story clock** (`src/series.ts`): slot from time, pure and fully tested, including the weekend, DST, the 10-chapter Prologue and the gap between seasons.
2. **Pages and storage:** R2 bucket and binding, the three routes, fallback and between-seasons logic, preview behind the key.
3. **Templates and lettering:** zone geometry, caption and balloon drawing, the narration strip, template guide PNGs.
4. **Import:** validation, the three conversions, page building, proof sheet, upload, single-chapter rebuild.
5. **The Prologue's first chapters** (Claude, drawn by code) are the pilot: imported end to end and checked on all three panels before 26 October.
6. **Alerting** additions and `/health-detailed` fields.
7. **Docs:** DECISIONS record, README endpoints, HANDOVER, CLAUDE.md file map; the panels' pagelists updated by the owner in SenseCraft.

## 11. Testing and done criteria

- The story clock has unit tests at every boundary: each weekday hour, Saturday/Sunday replay, the teaser window, season start and end, a gap week, and both DST changes.
- The import refuses a fixture package with each class of fault (missing art, overflowing text, wrong scene count, banned name).
- Each template is checked visually at 800×480 in all three conversions before the pilot is imported.
- The pilot chapter is seen on the real panels: desk, home and office.
- The three routes cost no KV operations and no neurons (checked in `/health-detailed` and the GraphQL usage figures, DECISIONS #62).
