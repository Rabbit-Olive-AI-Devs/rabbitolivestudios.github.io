# The Series — Showrunner Handover

**For:** Jarvis (Hermes agent), showrunner of the series.
**From:** Claude Code, the engineer on this repo, on behalf of the owner (Thiago).
**Date:** 2026-10-14. **Status:** the design is agreed, the calendar decided, and the panel code and import tool are **built** (DECISIONS #71) for the Prologue's premiere on 26 October 2026. Items marked **(awaiting owner)** are proposals the owner has not confirmed; check them with the owner in Slack before acting on them.

Read this file end to end, then `docs/series/SEASON-FORMAT.md` (what the writers deliver). The full design and its reasons: `docs/superpowers/specs/2026-10-14-series-design.md`. You do not need the rest of the repo.

## 1. What you are running

A Star Wars **graphic-novel series** told on the owner's three e-ink panels: the desk panel in the home office, the original panel at home, and the colour panel at the office. Working title *The Long Drift*. An original crew of three (a pilot, a droid and one more) on their own ship in the Outer Rim. Think *The Mandalorian*: a found family, a job each week, a longer journey across the season.

- **A season lasts three months** and has **13 chapters**, one per week. Each chapter is one mission that moves the season's journey on.
- **Each season has one writer**, an AI that writes the script **and draws every picture in its own style**: Claude, OpenAI, Grok, Muse.
- **Scenes appear on weekdays only, during working hours:** five a day at 08:00, 10:00, 12:00, 14:00 and 16:00 Chicago time, 25 per chapter. Saturday and Sunday replay the week, one scene an hour, and from Sunday 19:00 a "next week" teaser shows until Monday.
- **The writers draw pictures without words.** Our code letters every caption, balloon and narration line in the panel's font, so the words are always sharp and every season looks like one show.
- **Everything is made in advance.** A finished season is delivered as one package, checked and lettered by the engineer, approved on a proof sheet, then switched on. Nothing is generated while the panels run.

Your job: keep the show coherent and on schedule. The writers' job: their season. The engineer's job: get it onto the panels.

## 2. Roles

| Who | Responsible for |
|---|---|
| **Owner** | Executive producer and **art director**. Approves the bible, the writer order and **each writer's drawing style**, from style tests (practice pictures with no story in them, so the owner stays unspoiled). Never sees scenes: **you approve the proof sheets** |
| **Jarvis** | Showrunner. The bible, the schedule, briefing writers, reviewing every script and picture (including that every page **matches the style the owner approved**; send back any chapter that drifts), assembling and delivering the package, the continuity log |
| **The season's writer** | Character and ship sheets, all 13 chapters (script and pictures), the handover note to the next writer |
| **Claude Code** | The panel pages, the import and its checks, template guide images, the proof sheet, uploading, alerts when something is missing |

Coordination with the writers happens in the owner's Slack channel, where the agents already talk, and stays spoiler-free there. **The engineer cannot read Slack**, and deliveries, approvals and style tests never go through it: talk to it through the Dropbox mailbox `00_SHARED_INBOX/series/` (`to-claude/` for your messages, `from-claude/` for its replies and `STATUS.md`, `packages/` for zips, `style-tests/` for a writer's sample pictures). One Markdown file per message, named `YYYY-MM-DD-HHMM-topic.md`, starting with `DELIVERY`, `APPROVE`, `STYLE TEST`, `QUESTION` or `FYI`. A watcher on the Mac mini (DECISIONS #72) handles every message within minutes, at any hour: `APPROVE`, `DELIVERY` and `STYLE TEST` (first line `STYLE TEST folder=<name in style-tests/>`) by script, anything else by an unattended Claude that can only read and reply. When something needs Thiago, it adds a spoiler-free line to `from-claude/NEEDS-THIAGO.md`; please relay those lines to him in Slack. The full standing brief is `00_SHARED_INBOX/series/JARVIS-BRIEF.md`.

## 3. The bible (you own it)

The bible is the one place every writer reads before writing a word. Keep it in Notion. It must contain:

1. **Premise:** one paragraph. Who the crew are, what they want, what keeps them moving.
2. **Setting:** the Star Wars galaxy, Outer Rim, after the fall of the Empire. Canon places, species and ship classes are fine. **No canon characters.** **Original Force-users may appear as guests** (owner, 2026-10-14), not as crew: their own names, looks and blades; never a canon Jedi or Sith or a look-alike.
3. **Tone:** grown-up (owner, 2026-10-14, replacing "family-friendly"): real stakes, loss, moral grey, consequences that last, quiet and dark moments as well as humour and warmth. Still no gore, no sexual content and no cruelty for its own sake. Dry humour from the droid is a house trait.
4. **The crew:** for each member, name, role, how they talk (three sample lines), what they want, what they will never do. **Fixed by the owner on 2026-10-14:** the show is *The Long Drift*; the crew are Vess Oro (pilot, captain), Bolt (deadpan droid) and Hallan Dree (old mechanic); the ship is the *Last Lantern*.
5. **The ship:** name, layout, quirks, what breaks.
6. **Running state:** credits, cargo, damage, who is owed what. Updated at the end of every season from the writer's handover note.
7. **House rules for the page:** no words in pictures; readable at panel size; calm zones kept plain; plain ASCII text; the length guides. All are in `SEASON-FORMAT.md`; link it, do not copy it.
8. **Banned names:** canon characters and anything else that must never be printed. The import tool refuses any scene that uses one. Send the list to the engineer whenever it changes.
9. **Continuity log:** one page per season, filled from the writer's handover.

## 4. Notion layout (awaiting owner)

A suggested structure; adjust it if your tools prefer another:

- **The Long Drift** (top page)
  - **Bible** (page, with sub-pages for each section above)
  - **Seasons** (database): number, title, writer, premiere date, status (*planning, writing, drawing, review, delivered, live, done*)
  - **Chapters** (database, linked to Seasons): number, title, teaser, status
  - **Scenes** (database, linked to Chapters): number, weekday and time, template, caption, narration, balloons (speaker, line, anchor), art files, status, review notes
  - **Sheets** (page per season): the writer's character and ship sheets
  - **Continuity log** (page per season)

The Scenes database maps one-to-one onto `season.json`, so the export to the package can be mechanical.

## 5. Calendar (decided by the owner, 2026-10-14)

| Season | Writer (credit name) | Premiere (Monday) | Chapters |
|---|---|---|---|
| **Prologue** (season 0) | Claude | **26 October 2026** | 10, ending Friday 1 January 2027 |
| Season 1 | Zoe (OpenAI) | 4 January 2027 | 13 |
| Season 2 | Volt (Grok) | 5 April 2027 | 13 |
| Season 3 | Pepper (Muse) | 5 July 2027 | 13 |
| Season 4 | Claude | 4 October 2027 | 13 |

After the Prologue, a season premieres on the **first Monday of a quarter** and runs 13 weeks. Some quarters leave a one-week gap; the panels show a "returns on" card then. **Muse can generate images** and draws its own season like the others.

**The Prologue is tight:** it premieres 12 days after this plan. Claude writes and draws it inside the engineer's repo, chapters 1 and 2 by Friday 23 October (both were uploaded on 14 October), then each chapter at least 12 days before it airs, written by a weekly unattended run (DECISIONS #72). Your first jobs: set up Notion, start the bible from what the Prologue establishes (crew, ship, premise), and brief OpenAI for Season 1.

### Deadlines for each season after the Prologue, counted back from its premiere

| When | What |
|---|---|
| 9 weeks before | Writer briefed: bible, previous handover note, format, template guides |
| 8 weeks before | **Style test approved by the owner** (section 8 of SEASON-FORMAT.md); character and ship sheets reviewed and approved |
| 7 weeks before | Season outline (13 chapter titles, one paragraph each) approved |
| 4 weeks before | Chapters 1–4 scripted, drawn and reviewed |
| **3 weeks before** | **First delivery to the engineer:** at least chapters 1 and 2; and **6 to 8 story-free key frames for the season teaser** (`TEASER.md`) |
| Every chapter after | Delivered **at least 12 days before it airs**, approved **at least 8 days before** (the alert emails the owner from 7 days before an unapproved chapter airs) |

For Season 1: brief Zoe by 2 November 2026; first delivery by 14 December 2026. These are deadlines, not start dates: style tests, and after an approved look the sheets and outline, may start any time (owner, 2026-10-14).

**Production rhythm: you own it (owner, 2026-10-14).** As showrunner you set the cadence for every writer and keep them to it; what follows is the frame you work within, not a schedule imposed on you. Seasons are made in order, because each one starts where the last ended (the handover note), and a chapter a week reads better and spreads each writer's usage across the weeks:

- **Any time now:** every writer's standard style test; after the owner approves the look, character and ship sheets and the season outline.
- **About 4 weeks before the premiere:** chapters 1 to 4 drafted and reviewed.
- **During the season:** one chapter a week, about two weeks ahead of airing. A suggested week: Monday you nudge the writer, Thursday they deliver to their `writer-exchange` lane, Friday you review and send the DELIVERY, the weekend you approve.
- **At most two writers active at once:** the one on air and the next one preparing.
- **The heartbeat is your status board, not your boss:** every Monday 08:00 Chicago a note `…-heartbeat.md` arrives in `from-claude/` with what the engineering side knows: what is uploaded, approved, late, due and on air, and the milestones of the next two weeks (DECISIONS #76). Run your own schedule and nudge the writers on it; if you have your own scheduler and no longer need the note, say so and it is switched off.

## 6. Briefing a writer

Give each writer, in one message or Notion page:

1. The bible.
2. The previous writer's `handover.md` (none for Season 1).
3. `SEASON-FORMAT.md` and the template guide images.
4. Their season number, premiere date and deadlines.
5. The order of work: **sheets first**, then the outline, then chapters in order. Never draw scenes before the owner has approved the style and you have approved the sheets.
6. A reminder of the three rules that cause most rework: **no words in pictures**, **keep calm zones plain**, **readable at thumbnail size**. Give them the shared reference guides in `00_SHARED_INBOX/series/reference/`.

## 7. Reviewing (before you package)

For every chapter:

- **Story:** the chapter is one mission with a beginning, middle and end; Friday 16:00 ends on a hook or a resolution; it moves the season's journey on; the teaser does not spoil the next chapter.
- **Continuity:** names, ship state, credits, injuries and threads match the bible and earlier chapters.
- **Tone:** grown-up but never gratuitous; the crew sound like themselves.
- **Pictures:** on model with the sheets; no words or logos; calm zones plain; readable at panel size; any change of look has a stated reason; the right size and shape for the template; at least one new picture each weekday.
- **Words:** within the length guides; plain ASCII; balloon anchors on the speaker's head.

The import tool re-checks the mechanical rules and lists any failure by chapter and scene. Taste and continuity are yours.

## 8. Delivering a season

1. Export the Scenes database to `season.json` exactly as in `SEASON-FORMAT.md`, with the art in `art/`, sheets in `sheets/`, and the writer's `handover.md`.
2. Zip the folder as `season-NN-chapter-NN.zip`, put it in Dropbox `00_SHARED_INBOX/series/packages/`, and write a message in `to-claude/` whose first line is `DELIVERY package=season-NN-chapter-NN.zip`. The mailbox watcher on the Mac mini picks it up within minutes, without anyone opening a session. Seasons may be delivered in parts: see `SEASON-FORMAT.md` section 5.
3. The engineer runs the import. If it fails, you get the full list of problems; fix them in Notion and re-export.
4. When it passes, the engineer uploads the chapter as *pending* and sends you its **proof page**: `https://eink-dashboard.thiago-oliveira77.workers.dev/series/proof?season=S&chapter=C&key=KEY`, every scene in order. The owner gives you the key privately (it is set with `scripts/series-set-key.sh`); never post it in Slack or Notion. You can also preview any moment on the panels' own addresses with `?at=YYYY-MM-DDTHH:MM&key=KEY`. **You approve it** (the owner has chosen to watch unspoiled, so do not send them the proof sheet or spoilers).
5. Approve it with a message in `to-claude/` whose first line is `APPROVE season=S chapter=C` (or `chapters=C,D`); the watcher switches it on within minutes and confirms in `from-claude/`, and the chapter airs on its Monday at 08:00. A chapter that is not approved never airs: the panels keep showing the last approved scene, and the owner gets an alert seven days before.

### Fixes after a season is live

A typo or a bad picture can be fixed at any time: correct it, re-export, tell the engineer which chapter. One chapter is rebuilt; if it is already approved and on air, the engineer replaces it only after **you approve the correction** (the import skips approved chapters otherwise), and the panels pick it up on their next refresh.

## 9. What never to do

- **Never put scripts, pictures or spoilers in the GitHub repo.** It is public. Stories live in Notion and the private storage only.
- **No canon characters, no real people's likenesses, nothing unsuitable for children.**
- **No words in the art.**
- **No last-minute packages:** a late season means the panels show the "returns soon" card.
- Never share the owner's keys or secrets in Slack or Notion. The engineer never needs them from you.

## 10. Questions

- Story, tone, writers, dates: ask the owner in Slack.
- Format, templates, import errors, what the panels can show: ask the engineer (Claude Code).
