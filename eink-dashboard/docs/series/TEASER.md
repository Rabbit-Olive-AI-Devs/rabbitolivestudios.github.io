# The Long Drift: teasers

**Status:** the series teaser v1 exists (2026-10-14, DECISIONS #77). Next: **v2 with real 3D shots made in
Blender on the MacBook Pro**, then one teaser per season. This file is the handover for whoever makes
them: a Claude session on the MacBook, on the Mac mini, or anyone else.

## Rules that never bend

- **No spoilers.** The owner (Thiago) watches the series unspoiled. A teaser shows only the setting,
  the crew's archetypes, the authors and their looks, the format and the dates. Never a scene, a plot
  point, a chapter title or anything from a season package. Never open `~/series-work/s*/`, the
  Dropbox `packages/` folder, or a proof page.
- **Pictures:** only story-free ones: the style tests (`style-tests/`) and the non-canon test scene
  `LD-SCENE-0DBEB0E8` (the desert bazaar every author drew). Copies are in the Dropbox series folder
  under `teaser/assets/`.
- **No canon characters, insignia or ships** (see SHOWRUNNER.md). Saga-like iconography is fine as
  treatment, not as content.
- **Never commit pictures, video or audio** to this repo (it is public). Code only.

## v1, what exists

`Dropbox/00_SHARED_INBOX/series/teaser/the-long-drift-teaser-v1.mp4`: 60 s, 1920x1080 at 24 fps,
2.39:1 letterbox, film grain, Optima titles with a soft glow and a dark halo for legibility. Made by
`scripts/series-teaser/teaser.py` (frames and score both in numpy + PIL), joined with ffmpeg.

| Time | Picture | Words | Sound |
|---|---|---|---|
| 0-10 s | Stars in three depths, a dim nebula | AFTER THE EMPIRE FELL / THE OUTER RIM BELONGED TO NO ONE | Low drone in D, wind; heartbeat thuds from 6 s |
| 10-15.5 s | Claude's oil painting of the freighter at dusk, slow push-in | ONE SHIP / THE LAST LANTERN | Thuds quicken |
| 15.5-19.5 s | Claude's gouache daylight version, close pan on the crew | A PILOT WITH DEBTS / A DROID WITH OPINIONS / A MECHANIC WITH A PAST | Riser |
| 19.6-23.5 s | Black, stars | **THE LONG DRIFT** (white flash on the hit) | Braam (detuned saws, filter sweep) + big thud |
| 23.5-27.5 s | Stars | ONE STORY. FIVE SEASONS. / FOUR AUTHORS. EACH IN THEIR OWN HAND. | Plucked ostinato in D minor starts |
| 27.5-43.5 s | 4 s per author: Claude (gouache), Zoe (her test scene), Volt & Inky (theirs), Pepper (Muse's rain-noir sample) | Name + season; "AND CLAUDE RETURNS · SEASON FOUR" | A hit on every cut |
| 43.6-48.3 s | Montage of every look, cuts quickening | EVERY SCENE ITS OWN STYLE | Ostinato double time, long riser |
| 48.4-52.4 s | Stars | THREE E-INK SCREENS / A NEW SCENE EVERY TWO HOURS | Near silence, a high tense tone |
| 52.5-60 s | Black, stars | THE LONG DRIFT / THE PROLOGUE / MONDAY 26 OCTOBER 2026 / the four season dates | Final braam, thuds, reverb tail |

Integrated loudness is about -16 LUFS. The score was checked by its levels, not by ear: the engineer
cannot listen. Ask the owner what he hears.

**The weak part of v1:** slow pans on still paintings. That is what v2 fixes.

## v2 with Blender (the MacBook Pro)

Keep v1's cut, words, dates and score; replace or add shots made in 3D:

1. **Cold open:** the *Last Lantern* (original design: a long hull, a raised cockpit on a forward
   neck, a dorsal hump with a dish, a rear engine with a blue glow, cargo pods underneath; see
   `teaser/assets/claude-gouache.png`) drifting past the camera through a starfield, engine glow
   lighting the hull.
2. **Arrival:** a slow descent over a desert outpost at dusk (two moons, domes, a tower with a red
   beacon), volumetric haze, the ship settling on the pad.
3. **Jump:** a hyperspace jump or a fast pull-away before the title hit (19.6 s).
4. **Optional:** a painterly pass over the 3D (Blender's Grease Pencil or a toon/painterly shader)
   so it belongs with the painted pictures, and a final wide shot under the date card.

Render at 1920x1080, 24 fps, the same 2.39:1 framing (picture area 1920x804), into
`TEASER_DIR/shots/`, then cut them into `teaser.py`'s timeline (each shot replaces a `picture(...)`
call for its time range) or edit them over the v1 score with ffmpeg. Deliver
`the-long-drift-teaser-v2.mp4` to the Dropbox `teaser/` folder and tell the owner.

## Starting a session on the MacBook

Claude's memory is per machine, so the MacBook session starts from this file, not from the
Mac mini's memory.

1. **Get the repo.** In Terminal: `cd` to the checkout and `git pull`, or clone it with
   `gh repo clone Rabbit-Olive-AI-Devs/rabbitolivestudios.github.io`.
2. **Check Dropbox has synced** `00_SHARED_INBOX/series/teaser/` (the assets, v1 and `score-v1.wav`).
   On the MacBook the Dropbox folder is likely `~/Library/CloudStorage/Dropbox/` or `~/Dropbox/`, not
   the Mac mini's external SSD.
3. **Tools:** `brew install ffmpeg` and `python3 -m pip install numpy pillow`. Blender open with its
   MCP add-on running, and the Blender MCP connected in Claude Code (`claude mcp list` shows it).
4. **Start Claude** in `eink-dashboard/` (`claude`) and say: *"Read docs/series/TEASER.md and make
   teaser v2 with Blender. Assets are in my Dropbox at …/series/teaser/assets."* The session reads
   CLAUDE.md on its own; the mailbox and LaunchAgent steps in it belong to the Mac mini and can be
   skipped there.
5. When it is done, commit only code (scripts or Blender `.py`), never the renders, and push.

## A teaser for every season

The owner wants one before each premiere (2026-10-14). Jarvis commissions it as part of the season:

- **About 3 weeks before the premiere**, the season's author delivers **6 to 8 story-free key frames**
  in their approved look (setting, mood, the crew standing, the ship; no scene from the season) to
  their `writer-exchange` lane, plus one line on the season's mood (not its plot).
- **Claude assembles the teaser** from the same template: their frames in the picture slots, the
  season's name and author, the show's theme in that author's variation, and the dates. 30 to 60 s.
- **Jarvis reviews it for spoilers** before it goes to the owner, and puts it in `teaser/`.

## Ideas the owner liked, not built yet (2026-10-14)

- **A theme for the show:** a short motif (four or five notes) used in every teaser, each author's
  season with its own variation (instrument, tempo, mode).
- **A narrator** for the opening lines (a voice-over makes it feel like a theatre trailer).
- **A vertical 9:16 cut** for sharing on a phone.
- **A countdown on the panels:** the "coming soon" card showing the days left to the premiere
  (a change to `renderCard` in `src/series/render.ts`; the card is a pre-built page, so the import
  would build one per day, or the Worker would draw the number).
