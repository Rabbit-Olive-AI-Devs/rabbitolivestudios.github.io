# The Long Drift: teasers

**Status:** v1 (2026-10-14, code-drawn) and **v2 (2026-10-15, with 3D shots made in Blender on the
MacBook Pro)** exist (DECISIONS #77). Next: one teaser per season. This file is the handover for
whoever makes them: a Claude session on the MacBook, on the Mac mini, or anyone else.

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

## v2, what exists (Blender, the MacBook Pro)

`Dropbox/00_SHARED_INBOX/series/teaser/the-long-drift-teaser-v2.mp4`: v1's cut, words and dates;
the first 19.6 s and the date card are new 3D shots, and the score is a Suno track with v1's hits
laid over it.

| Time | v2 picture | Sound |
|---|---|---|
| 0-10.5 s | **Cold open (3D):** the *Last Lantern* crosses close to camera at an angle, between the two opening lines (nearest at ~5 s), then drifts away to the right, engine toward us. Engine surges on 6.0 and 8.6 s | Suno track from its start |
| 9.2-18.6 s | **Arrival (3D):** a slow crane down over the dusk outpost (domes, beacon tower, two moons, haze); the ship descends, legs drop out of the belly at 12.9 s, touchdown at ~15.4 s, the ramp drops at 16 s and three crew silhouettes stand in the hatch light. "ONE SHIP / THE LAST LANTERN", then the crew lines (now low, over the sand) | Suno; v1's riser into the title |
| 18.6-19.75 s | **Jump (3D):** from behind, the engine flares, stars stretch into streaks, the ship goes to a point into the 19.6 s flash | v1's braam + thud on 19.6 |
| 19.6-52.4 s | As v1: title, premise, the four authors, montage, the screens card | Suno; v1's hits on 27.5/31.5/35.5/39.5 with the track dipped a breath before each; the track dipped under the screens card for v1's high tone and riser |
| 52.5-60 s | **Wide (3D)** under the date card: the ship small, heading for the desert world's sunlit limb | Suno's climax (its 62-70 s) under v1's final braam |

**The ship.** Our own design, built from primitives in code, after a concept sheet the owner made
with ChatGPT (2026-10-14): `Dropbox/00_SHARED_INBOX/series/teaser/assets/last-lantern-concept.png`
(Dropbox only; pictures never go in this repo). A faceted octagonal hull tapering to a blunt prow,
a raised cab with a window band on a short neck, a dorsal spine with a grille and a dish on a
mast, a heavy banded engine cowl with a finned drum, a blue glow and two swept spikes, three
mismatched pods (white, rust, teal) on a belly rack, four chunky legs that fold away in flight,
off-white plating with rust patches. The prompt that made the sheet, to make it again or a variant:

<details><summary>ChatGPT prompt for the concept sheet</summary>

```
Concept art turnaround sheet for an original spaceship called "The Last Lantern": a small,
battered independent cargo freighter, about 28 metres long, flown by a crew of three in a
lawless frontier region. ORIGINAL DESIGN ONLY: it must not resemble the Millennium Falcon,
the Razor Crest, Serenity, the Ghost or any existing film/game/franchise ship; no insignia,
logos or lettering.

SHAPE (the most important part): not a tube. Give it a strong, memorable silhouette that reads
from the side and from below:
- A long main hull with a flattened, faceted cross-section (wider than tall, slab sides that
  taper toward the nose, a chined keel line), slightly asymmetric: more mass to one side.
- A raised cockpit on a short forward neck, set above and a little behind the nose, with a
  wraparound band of small square windows. A blunt, scuffed nose like a tug's prow.
- A dorsal hump amidships (the crew quarters) carrying a small dish antenna on a mast.
- One big rear engine in a heavy armoured cowl, glowing pale blue, with heat fins.
- Three detachable cargo pods clamped underneath on visible rails: blocky, mismatched, one a
  different colour (replaced at some point).
- Landing gear: four chunky hydraulic legs with wide pads; show it both DEPLOYED (landed view)
  and RETRACTED into wells in the belly (flight views).

SURFACE: weathered, practical, lived-in. Off-white and warm grey hull panels with sun-bleached
patches, rust-orange primer showing through, scorch marks behind the engine, mismatched
replacement plates, rivets, hatches, handholds, pipes and vents along the flanks, a boarding
ramp under the neck. It should feel heavy, chunky and repaired many times, like a 1970s
model-kit ship: blocky greebles, no sleek curves.

SHEET LAYOUT (for a 3D modeller): one landscape image on a plain warm off-white background.
Clean orthographic views at the same scale, lined up: SIDE (left profile, gear deployed),
TOP, FRONT, REAR, BELOW (gear retracted, pods visible), plus one larger 3/4 hero view in flight
(gear retracted, engine glowing). Even, neutral lighting so the shapes are readable; a thin
scale bar; no text other than view labels.

STYLE: hand-painted gouache concept art with visible brush strokes and a slightly naive,
storybook quality, in the spirit of classic 1970s-80s space illustration. Muted palette:
off-white, warm grey, rust orange, a touch of teal; pale blue engine glow.
```
</details>

**How it was made.**
- `scripts/series-teaser/blender/lantern.py` builds every shot from primitives (no downloaded or
  generated models, no `.blend` file) and renders it: EEVEE, toon-banded shading through Shader
  to RGB (keeps some of the light's colour, so the engine glow tints the hull), a bounded haze
  volume at dusk (a world volume swallowed the sky), AgX Punchy, a glow and an anisotropic
  Kuwahara pass in the compositor. Frames are 1920x804, the picture area; `teaser.py` adds the
  bars. It was built live through the Blender MCP (Blender 4.5 LTS), then rendered headless.
- `teaser.py` uses the shots when `TEASER_DIR/shots/<shot>/f0001.png` exists for all four (v1's
  paintings otherwise), adds a painterly brush-dab texture that boils on twos so the renders sit
  with the paintings, then the titles, grain and letterbox as in v1.
- `teaser.py audio <bed.wav>` cuts the music track to the timeline (its 0-52.5 s, then its climax
  from 62 s under the final hit), dips it for the screens card and a breath before each cut hit,
  and lays v1's braams, thuds, risers and high tone on top -> `score-v2.wav`.
- The music: Suno (the owner's account, v6-mini, instrumental, 70 s), style prompt *"instrumental
  cinematic space trailer score, D minor, slow dark drone and wind intro, low heartbeat pulses,
  rising tension, huge braam hit, driving plucked string ostinato, taiko hits, epic brass swell,
  final braam, reverb tail, no vocals"*. The file: `teaser/long-drift-suno-A.m4a`.

**How to re-render** (TEASER_DIR holds `assets/`, `shots/`, the music):

```
export TEASER_DIR=~/series-work/teaser-v2
B=/Applications/Blender.app/Contents/MacOS/Blender
for s in cold arrival jump wide; do $B -b --factory-startup -P scripts/series-teaser/blender/lantern.py -- render $s; done
ffmpeg -i "$TEASER_DIR/long-drift-suno-A.m4a" -ar 48000 -ac 2 "$TEASER_DIR/suno-A.wav"
python3 -E -P scripts/series-teaser/teaser.py audio "$TEASER_DIR/suno-A.wav"     # -> score-v2.wav
python3 -E -P scripts/series-teaser/teaser.py frames                             # -> frames/
ffmpeg -framerate 24 -i "$TEASER_DIR/frames/f%05d.jpg" -i "$TEASER_DIR/score-v2.wav" -af loudnorm=I=-16:TP=-1.5 \
  -c:v libx264 -b:v 7M -maxrate 10M -bufsize 14M -pix_fmt yuv420p -c:a aac -b:a 256k -ar 48000 \
  -movflags +faststart -shortest the-long-drift-teaser-v2.mp4
```

A still of one shot frame: `... -P lantern.py -- still arrival 200`; a composed still at a time:
`teaser.py still 17.5`. Python needs numpy + Pillow (the MacBook uses a venv in TEASER_DIR).
Render time on the MacBook Pro: 24 min 21 s for the four Blender shots, headless (cold 5:07, jump 0:51, wide 3:40, arrival 14:43: 1.2 to 3.9 s a frame), then 16 s to compose the 1,440 frames and 44 s to encode.

**What did not work, and limits.**
- A world volume for the haze turned the sky black; a bounded box fixed it, but its floor showed
  as a pale ellipse until it went below the dunes. The engine light drew hard toon bands on the
  sand until the ground got a soft ramp.
- The Kuwahara pass smears single-pixel stars, so space shots use a small size and brighter stars.
- The first ship (cylinders) read as a tube; the owner asked for better, and the concept sheet
  gave the shape.
- Chrome on the MacBook would not save Suno's download from the automated tab, and the CDN files
  are encrypted for the web player; the owner downloaded the track by hand.
- The score is checked by levels (hits about 8 dB over the dip before them), not by ear. The crew
  are simple silhouettes. The look is stylised toon plus a brush texture, cleaner than the
  paintings.

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
