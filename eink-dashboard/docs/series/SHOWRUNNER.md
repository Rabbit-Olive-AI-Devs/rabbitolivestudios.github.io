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
- **Everything is made in advance.** A finished season is delivered as one package, checked and converted by the engineer, approved on a proof sheet, then switched on. Nothing is generated while the panels run.

Your job: keep the show coherent and on schedule. The writers' job: their season. The engineer's job: get it onto the panels.

## 2. Roles

| Who | Responsible for |
|---|---|
| **Owner** | Executive producer. Approves the bible and the writer order. Watches each season unspoiled: **you approve the proof sheets** |
| **Jarvis** | Showrunner. The bible, the schedule, briefing writers, reviewing every script and picture, assembling and delivering the package, the continuity log |
| **The season's writer** | Character and ship sheets, all 13 chapters (script and pictures), the handover note to the next writer |
| **Claude Code** | The panel pages, the import and its checks, template guide images, the proof sheet, uploading, alerts when something is missing |

Coordination with the writers happens in the owner's Slack channel, where the agents already talk. **The engineer cannot read Slack**: talk to it through the Dropbox mailbox `00_SHARED_INBOX/series/` (`to-claude/` for your messages, `from-claude/` for its replies and `STATUS.md`, `packages/` for zips, `style-tests/` for a writer's sample pictures). One Markdown file per message, named `YYYY-MM-DD-HHMM-topic.md`, starting with `DELIVERY`, `APPROVE`, `STYLE TEST`, `QUESTION` or `FYI`. The engineer works when the owner opens a session, so for anything urgent also ask the owner in Slack to have Claude check the series inbox. The full standing brief is `00_SHARED_INBOX/series/JARVIS-BRIEF.md`.

## 3. The bible (you own it)

The bible is the one place every writer reads before writing a word. Keep it in Notion. It must contain:

1. **Premise:** one paragraph. Who the crew are, what they want, what keeps them moving.
2. **Setting (awaiting owner):** the Star Wars galaxy, Outer Rim, after the fall of the Empire. Canon places, species and ship classes are fine. **No canon characters.** Whether Jedi may appear (as guests, not crew) is the owner's call.
3. **Tone:** family-friendly. Peril, humour, warmth; no gore, nothing a child at home should not read. Dry humour from the droid is a house trait.
4. **The crew:** for each member, name, role, how they talk (three sample lines), what they want, what they will never do. **Fixed by the owner on 2026-10-14:** the show is *The Long Drift*; the crew are Vess Oro (pilot, captain), Bolt (deadpan droid) and Hallan Dree (old mechanic); the ship is the *Last Lantern*.
5. **The ship:** name, layout, quirks, what breaks.
6. **Running state:** credits, cargo, damage, who is owed what. Updated at the end of every season from the writer's handover note.
7. **House rules for the page:** no words in pictures; flat colour and clear outlines; calm zones kept plain; plain ASCII text; the length guides. All are in `SEASON-FORMAT.md`; link it, do not copy it.
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

| Season | Writer | Premiere (Monday) | Chapters |
|---|---|---|---|
| **Prologue** (season 0) | Claude | **26 October 2026** | 10, ending Friday 1 January 2027 |
| Season 1 | OpenAI | 4 January 2027 | 13 |
| Season 2 | Grok | 5 April 2027 | 13 |
| Season 3 | Muse | 5 July 2027 | 13 |
| Season 4 | Claude | 4 October 2027 | 13 |

After the Prologue, a season premieres on the **first Monday of a quarter** and runs 13 weeks. Some quarters leave a one-week gap; the panels show a "returns on" card then. **Muse can generate images** and draws its own season like the others.

**The Prologue is tight:** it premieres 12 days after this plan. Claude writes and draws it inside the engineer's repo, chapters 1 and 2 by Friday 23 October, then a chapter at least 7 days before it airs. Your first jobs: set up Notion, start the bible from what the Prologue establishes (crew, ship, premise), and brief OpenAI for Season 1.

### Deadlines for each season after the Prologue, counted back from its premiere

| When | What |
|---|---|
| 9 weeks before | Writer briefed: bible, previous handover note, format, template guides |
| 8 weeks before | Character and ship sheets reviewed and approved |
| 7 weeks before | Season outline (13 chapter titles, one paragraph each) approved |
| 4 weeks before | Chapters 1–4 scripted, drawn and reviewed |
| **3 weeks before** | **First delivery to the engineer:** at least chapters 1 and 2 |
| Every chapter after | Delivered **at least 12 days before it airs**, approved **at least 8 days before** (the alert emails the owner from 7 days before an unapproved chapter airs) |

For Season 1: brief OpenAI by 2 November 2026; first delivery by 14 December 2026.

## 6. Briefing a writer

Give each writer, in one message or Notion page:

1. The bible.
2. The previous writer's `handover.md` (none for Season 1).
3. `SEASON-FORMAT.md` and the template guide images.
4. Their season number, premiere date and deadlines.
5. The order of work: **sheets first**, then the outline, then chapters in order. Never draw scenes before the sheets are approved.
6. A reminder of the three rules that cause most rework: **no words in pictures**, **keep calm zones plain**, **flat colour with clear outlines**.

## 7. Reviewing (before you package)

For every chapter:

- **Story:** the chapter is one mission with a beginning, middle and end; Friday 16:00 ends on a hook or a resolution; it moves the season's journey on; the teaser does not spoil the next chapter.
- **Continuity:** names, ship state, credits, injuries and threads match the bible and earlier chapters.
- **Tone:** family-friendly; the crew sound like themselves.
- **Pictures:** on model with the sheets; no words or logos; calm zones plain; flat colour; the right size and shape for the template; at least one new picture each weekday.
- **Words:** within the length guides; plain ASCII; balloon anchors on the speaker's head.

The import tool re-checks the mechanical rules and lists any failure by chapter and scene. Taste and continuity are yours.

## 8. Delivering a season

1. Export the Scenes database to `season.json` exactly as in `SEASON-FORMAT.md`, with the art in `art/`, sheets in `sheets/`, and the writer's `handover.md`.
2. Zip the folder as `season-NN-chapter-NN.zip` and put it in Dropbox `00_SHARED_INBOX/series/`, then tell the engineer in Slack. Seasons may be delivered in parts: see `SEASON-FORMAT.md` section 5.
3. The engineer runs the import. If it fails, you get the full list of problems; fix them in Notion and re-export.
4. When it passes, the engineer uploads the chapter as *pending* and sends you its **proof page**: `https://eink-dashboard.thiago-oliveira77.workers.dev/series/proof?season=S&chapter=C&key=KEY`, every scene in order. The owner gives you the key privately; never post it in Slack or Notion. You can also preview any moment on the panels' own addresses with `?at=YYYY-MM-DDTHH:MM&key=KEY`. **You approve it** (the owner has chosen to watch unspoiled, so do not send them the proof sheet or spoilers).
5. Approve it in Slack; the engineer runs the approve step and the chapter airs on its Monday at 08:00. A chapter that is not approved never airs: the panels keep showing the last approved scene, and the owner gets an alert seven days before.

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
