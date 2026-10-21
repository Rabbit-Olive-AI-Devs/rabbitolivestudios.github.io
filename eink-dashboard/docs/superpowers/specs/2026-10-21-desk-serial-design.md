# The Desk Serial — Design

**Date:** 2026-10-21 · **Status:** for the owner's approval (reviewed by a builder check, 2026-10-21) · **Builds on:** DECISIONS #93 (the desk story page), #95 (its story book), #89 (Jarvis's picture pipeline), #71–#80 (the series' offline build and mailbox)

> **Spoiler rule.** The desk story page is a surprise. This document describes format and mechanics only. Story, outlines, scripts, scene notes and pictures never appear here, in the repo, in commit messages or in the main conversation. They live in private folders (below) and in R2.

## 1. Intent

Today the Surprise page on the desk E1001 (`/desk/castaway`) is a clock-driven engine: each day draws one of 21 storylines in a shuffled cycle, plus routine moments from pools, over a flat 1-bit scene drawn from shapes. Option A (2026-10-21, `1eebde9`) cut the routine repetition. But the storylines still cycle about every 50 days, there is no arc and no ending, and the picture is always the same composition.

The owner wants **a little story that moves forward each day** (their words), read in **a few glances through the day**, with **pictures painted in full colour, like the series pages**. The pictures are **drawn by Jarvis in a brick-built diorama style**, and **Claude writes it weekly**. It runs in **seasons with proper endings**. The writing may **draw on current events, news, films and the owner's interests** (their suggestion, not a rule).

What they would wince at: **repeating the same lines and pictures over and over.**

## 2. Goals, non-goals, anti-goals

**Goals**
- A serial with continuity: each day is a short episode that moves the story forward, in seasons with planned arcs and real endings.
- 4–6 beats a day (owner's number), each with its own new text and **its own new painting**.
- Each glance shows the latest beat. The "EARLIER" column recaps the day's earlier beats, so a missed glance can be caught up.
- Every page is built offline on the **Mac mini** (owner, 2026-10-21: it never sleeps, `sleep 0`, `autorestart 1`, and already runs the series watcher, art brief and story book jobs; locked is fine) and served as stored HTML (option D). There is no drawing on the Worker's request path for serial pages, which also removes the desk page's CPU risk (#95 incident) for this route.
- The story book (#95) keeps working: one frame per beat, one chapter per day.

**Non-goals**
- Live weather, light or moon changes inside the painting. The pictures are painted ahead, so the beat's painting is the picture. *(Claude's call: a painted beat is worth more than live weather in a shape drawing.)*
- A news feed. Current events are echoes inside the story, never headlines.
- Owner approval of anything. The owner never sees content before it airs.

**Anti-goals** (failure even if it works)
- Repeated lines or pictures. A repeat is allowed only as a rare backup (section 7), and never a whole beat repeated.
- A blank or error page when something upstream is late.
- Spoilers reaching the owner through docs, commits, logs, alerts or chat.
- Pushing the free plan's limits: no AI on the Worker, no new per-request KV writes, no extra CPU per request.

## 3. Continuity *(Claude's call)*

The serial carries on from the story the owner has read so far (same world, same characters), so it feels like the story found its feet rather than being replaced. The existing engine stays as the **fallback** for days with no serial page (breaks between seasons, missed weeks). Once the serial has started, the fallback plays **routine moments only**: the 21 old storylines are retired, because some assume an earlier state of the story and would contradict the serial. *(Claude's call.)* This needs a routine-only mode in `src/castaway.ts`, with tests.

## 4. Shape of a season and a day

- **Season:** about 8–10 weeks, with a one-week break before the next *(Claude's call)*. Before week 1 is written, Claude writes a private season plan: the arc, the ending, and the turning points by week. Each week's writing follows the plan and may adjust it.
- **Break:** the page falls back to the routine engine (the "quiet in-between" state). Nothing from the season is rerun.
- **Day:** one episode of 4–6 beats with start times on the Chicago clock. *Claude's suggested default:* about 07:00, 10:00, 13:00, 16:00, 19:00 and 22:00, fewer on lighter days. Each beat shows until the next one; the last beat holds through the night until the first beat of the next day.
- **Every day, weekends included** *(Claude's call: the story book reads continuously; a lighter weekend uses 4 beats)*.
- **Beat content:** an episode title line (as the page has now), up to 4 lines of story text, an "EARLIER" log entry (short time + one line), and one painting.
- **"EARLIER" keeps its two most recent entries**, so the layout is unchanged. The full day is in the story book. *(Claude's call.)*
- **Jarvis's capacity:** 4–6 new paintings a day, every day, is confirmed by the owner ("Jarvis can do it. His imagegen is fast"). Every beat gets its own painting, and paintings are never shared between beats.

## 5. Page layout

The current layout stays: the shared top bar, the picture band (rows 37–351), a rule at row 352, story text on the left and the "EARLIER" log on the right. The picture band becomes **Jarvis's painting in full colour**, cropped to fill 800×315 around a focus point *(as the series does; the panels convert colour, #81)*. All text uses the real fonts (`plex11m` minimum, #67), in pure black and white.

- The top bar is baked into the page, showing the beat's own start time in the clock slot and the day number in the pill. Battery and similar live readings are dropped from the serial page, since stale values would be wrong. *(Claude's call.)*
- **Before building:** show the owner a mockup with **placeholder** text and a placeholder picture (no story content), per CLAUDE.md "show the owner a mockup before building a new look".
- The page must stay **one request** (#65): the painting inlined as a `data:` URI, ~0.6–1 MB.

## 6. The weekly pipeline

All of it runs on the Mac mini and the Dropbox series mailbox (#72), reusing its patterns.

1. **Inputs (weekly, before writing).** The weekly run gathers light inputs: notable news, new films and shows, Flamengo's results and fixtures, holidays and the season, and Jarvis's notes on the owner's tastes (`to-claude/2026-10-17-1816-reply-thiago-tastes.md`, private). These are optional seasoning. Nothing grim or political enters the story directly *(Claude's call: it is a light read at a desk)*.
2. **Writing (Claude, weekly).** A new weekly job writes the **next Monday–Sunday** of the serial: for every beat, its time, text, log line and a **scene note** for the painting (setting, who is in it, what is happening, mood, time of day, and the focus point for the crop). It writes on Wednesday for the week starting 12 days later, giving Jarvis a full week to paint. It runs as a background `claude -p` job like the series weekly run, so nothing enters the main conversation. Before handing off, it checks every line for fit, glyphs and **no repeat against everything already aired** (an aired-lines index kept privately).
3. **Brief to Jarvis.** The scene notes go to Jarvis's private spoiler folder (`00_JARVIS_SPOILERS/desk-serial/<week>/`), with a mailbox note `SERIAL BRIEF week=<date>`.
4. **Painting (Jarvis).** He paints 4–6 pictures a day (28–42 a week) in the brick-built diorama style, full colour, at least 800×315 at the crop. He delivers them to `to-claude/serial/<week>/` with `SERIAL DELIVERY week=<date>`. He also checks that each picture matches its note.
5. **Build (watcher on the Mac mini).** The watcher validates each picture (format, size, not blank), crops it, renders each beat's page (text, log, top bar) offline with the same drawing kit as the Worker, and uploads the pages and a **week schedule** (beat times → page keys) to R2. These are uploads, not KV writes. It replies `SERIAL RECEIVED`.
6. **Check before airing** *(Claude's call)*. Before a built week can air, a background Claude review (like the series' independent review, #79) checks the built pages: text fits, pictures not blank and matching their scene notes, no line repeating an aired one. It writes PASS or REVISE. The schedule marks a week airable only on PASS, and the Worker serves only airable weeks; otherwise it uses the fallback. Jarvis's delivery counts as his sign-off on his pictures.
7. **Nightly and live rebuilds** *(Claude's call)*. Some beats carry live slots that a script written 12 days ahead can't know: match days and results from the Flamengo board, dated surprises and calendar moments, and sunrise or sunset timing. The script marks these as slots, and the Mac mini rebuilds and re-uploads the next day's pages each night. On a match day it re-uploads the current page again when the board's data changes (R2 uploads, at most a few a day). The Worker still only serves stored pages.
8. **Serve (Worker).** `/desk/castaway` looks up today's schedule in R2, picks the latest beat at or before now, and returns that stored page. With no airable serial page for now (a break, a missed week, or a week without PASS), it falls back to the engine in routine-only mode. The serving path is a read and a passthrough, with no drawing.

**Season plan and private files:** the arc, scripts, scene notes, aired-lines index and pictures live in `~/desk-serial-work/` (private, never committed) and Jarvis's spoiler folder, mirroring the series (`~/series-work/`). The repo holds code, tests and spoiler-free docs only.

## 7. When something is late

| Missing | What the page shows | Who hears |
|---|---|---|
| One beat's painting | That beat with a painting from the **settings library**: places painted once at the start of a season, chosen by the beat's setting. Never another beat's painting. | Alert email if any of tomorrow's paintings are missing at 21:00 Chicago |
| A whole day's paintings | Library pictures for every beat; the text still airs | Same alert |
| Claude's week not written, or no PASS | The routine engine for that week; the serial resumes when the writing does (no skipping ahead) | Alert when next week's script is missing by Friday |
| R2 read fails | The routine engine | Existing `/health-detailed` and alert patterns |

Jarvis paints the **settings library** (about 10–15 places per season) in the week before a season starts. This is the only reuse, and it stays rare.

## 8. Story book

The book job (#95) keeps one frame per beat. With serial pages the moment key becomes the beat's id, read from the R2 schedule (a read, no Worker drawing). The EPUB builder's repeat filter stays.

## 9. Constraints (from the project rules)

- **Spoilers:** owner never sees content; docs, commits, logs, alerts and replies are spoiler-free; nothing creative in the public repo (CLAUDE.md, #93).
- **No AI on the Worker**, no new per-request KV writes or lists (#62), one request per page (#65), real fonts at `plex11m` minimum (#67), colour pictures as-is (#81).
- **Alerting:** every new failure mode must be caught by `src/alert.ts` (#60), and the alert text must be spoiler-free.
- **Jarvis's load:** 28–42 paintings a week on top of 3 daily pictures, and from January the Season 1 series. The owner confirmed he can do it. Tell him the format through the mailbox before season 1 is scheduled.
- **Mac mini restarts:** after a restart, FileVault waits for the password and nothing runs. The existing alert (#90) emails when the mini's TARS reports stop for 2 hours. Nightly rebuilds then fall back to the previous build (still airable) or to the routine engine.

## 10. Left to the builder

- Exact R2 key layout, schedule format, and how the Worker finds "today's" schedule.
- Whether the offline page renderer reuses the series build code (`src/series/`) or the desk drawing kit (`src/hud/`), or both.
- The watcher's message names and validation thresholds, following the art pipeline's (#89).
- How the weekly job gathers inputs (web search inside the background run, RSS), within the light-touch rule.
- Start date of season 1: the earliest Monday with a full week of pages in R2 and the settings library delivered. It should not launch the same week as the series premiere (Mon 26 Oct).
- Tests: schedule lookup, fallback paths, fit/glyph checks for every built page, no-repeat check, alert checks.
