# Season Package Format

**For:** the showrunner (Jarvis) and every season's writer.
**Version:** format 1, 2026-10-14. Design: `docs/superpowers/specs/2026-10-14-series-design.md`.
**Status:** built (DECISIONS #71). The import measures every line with the real fonts, so the limits below are guides. **Template guide images:** `docs/series/templates/*.png`, one per template at true size, with the calm zones hatched.

A season reaches the panels as one **season package**: a folder (or a zip of it) that the import tool checks, converts and uploads. Anything the tool cannot read or fit is refused with a list of problems; nothing is half-imported.

```
season-01/
  season.json        the script: every chapter, scene, word and picture reference
  art/               the pictures, full colour, no words
  sheets/            character and ship sheets (reviewed, never shown on a panel)
  handover.md        the writer's note for the next season's writer
```

## 1. The shape of a season

- **13 chapters**, one per week (the Prologue, season `0`, has **10**). Chapter 1 starts on the premiere Monday.
- **25 scenes per chapter**, five per weekday, shown at 08:00, 10:00, 12:00, 14:00 and 16:00 Chicago time. Scenes 1–5 are Monday, 6–10 Tuesday, and so on to 21–25 on Friday.
- **Scene 1 of every chapter is a Splash**: the chapter's title card. The chapter number, title and writer's credit are lettered on it automatically.
- Each chapter has a **teaser line**, shown from Sunday 19:00 until Monday 08:00 ("Next week: …"). Chapter 13's teaser closes the season.
- Weekends replay the week automatically; nothing extra to write.

Think of a chapter as one episode: a mission with a setup on Monday, trouble mid-week and a resolution, or a cliffhanger, on Friday afternoon. The 16:00 scene stays up overnight, so it is a good place for a hook.

## 2. `season.json`

```json
{
  "format": 1,
  "season": {
    "number": 1,
    "title": "The Salt Road",
    "show": "THE LONG DRIFT",
    "writer": "Claude",
    "premiere": "2027-01-04"
  },
  "chapters": [
    {
      "number": 1,
      "title": "The Salt Market",
      "teaser": "Next week: the crate starts talking.",
      "scenes": [
        {
          "number": 1,
          "template": "splash",
          "panels": [{ "art": "c01-s01.png" }]
        },
        {
          "number": 2,
          "template": "wide",
          "caption": "Merrek Prime. The salt docks. Dusk.",
          "narration": "Vess talks the harbour master down to half the docking fee.",
          "panels": [
            {
              "art": "c01-s02.png",
              "balloons": [
                { "speaker": "BOLT", "text": "I have calculated our odds. I will not be sharing them.", "anchor": [596, 172] }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

| Field | Rule |
|---|---|
| `format` | Always `1` for now |
| `season.number` | `0` for the Prologue, then `1`, `2`, … |
| `season.show` | The show's title as printed in the top bar, the same every season |
| `season.premiere` | A Monday, `YYYY-MM-DD` |
| `season.writer` | The writer **agent's** name, printed in the credit (owner's decision, 2026-10-14): `Claude`, `Zoe` (OpenAI), `Volt` (Grok), `Pepper` (Muse), or two names joined by ` & ` |
| `chapters` | 13 (10 for the Prologue), numbered from 1 in order. A package may hold only the chapters delivered so far (section 5) |
| `scenes` | Exactly 25 per chapter, numbered 1–25 in order |
| `template` | `wide`, `split`, `strip`, `splash` or `panel-text` (section 3) |
| `caption` | The location or time box, one short line. Optional |
| `narration` | The text under the picture (or in the text column for `panel-text`) |
| `panels` | One entry per panel of the template, left to right |
| `panels[].art` | A file in `art/`, or `"hold"` to keep the picture from the same panel of the previous scene (only when the previous scene used the same template) |
| `panels[].focus` | Optional `[x, y]` in the art file's own pixels: the point to keep when the picture is cropped to the panel's shape. Default: the centre |
| `panels[].caption` | `strip` only: a short line at the top of that panel |
| `panels[].balloons` | Up to two. `speaker` in capitals, `text` the line, `anchor` the `[x, y]` point **in the art file's own pixels** where the speaker's head is. Use `"anchor": null` for a voice from off the picture (radio, comlink): it is lettered as a square box with the speaker's name |
| In a `hold` scene | The picture is the previous scene's, already cropped: give that scene's balloon `anchor` points **in panel pixels** (0,0 = the panel's top-left), not in the original file's |
| `panels[].clear` | Optional list of `[x, y, w, h]` boxes, in the art file's own pixels, that no balloon may cover: each character's body, a key object. Every balloon `anchor` is kept clear automatically; mark the rest |

## 3. Templates

| Template | Panels | Art size (px) | Words allowed |
|---|---|---|---|
| `wide` | 1 | 776 × 346 | caption, narration, up to 2 balloons |
| `split` | 2 | 382 × 346 each | narration, up to 2 balloons per panel |
| `strip` | 3 | 252 × 346 each | narration, a short caption per panel, 1 balloon per panel |
| `splash` | 1 | 800 × 443 | caption only (scene 1: lettered automatically) |
| `panel-text` | 1 | 476 × 426 | caption, long narration in the right-hand column, 1 balloon |

Every template has **calm zones** where the words will sit: the top third for balloons and captions on `wide`, `split` and `strip`; the bottom-left corner on `splash`. Keep those areas plain (sky, wall, floor, shadow) with no faces, hands or key details. The template guide images will show the exact zones.

### How much text fits

The import measures every line with the real panel font, so these are guides, not promises:

| Text | About |
|---|---|
| Location `caption` | 40 characters |
| `narration` (`wide`, `split`, `strip`) | 120 characters (2 lines) |
| `narration` (`panel-text`) | 600 characters |
| Balloon `text` | 70 characters (3 short lines) |
| `strip` panel caption | 40 characters |
| Chapter `title` | 22 characters |
| `teaser` | 110 characters |

**Characters:** plain ASCII only, as on a keyboard. Straight quotes `"` and `'`, three dots `...` for an ellipsis, a hyphen for dashes. No emoji, no curly quotes, no accented lowercase letters.

## 4. Art

- **One file per new picture**, PNG or JPEG, full colour, no transparency, named `cCC-sSS.png` (or `cCC-sSS-pN.png` for the Nth panel of `split` and `strip`).
- **Size:** at least as large as the panel, any shape. It is shrunk and **cropped to fill** the panel around `focus` (or the centre). Plan the composition for the crop: a 3:2 picture loses about a third of its height in a `wide` panel. Give `anchor`, `clear` and `focus` in the art file's own pixels; the importer maps them through the crop, and refuses a scene whose anchor ends up cropped away.
- **At least one new picture every weekday** (among each day's five scenes); the others may `hold`.
- **No words in the picture:** no lettering, signs, labels, logos or speech balloons. We add every word.
- **Flat colour and clear dark outlines.** The pictures become black and white on the two E1001 panels and six colours on the third. Flat areas survive all three; soft gradients, fine textures and thin pale lines turn into noise.
- **Colours that convert best:** black, white, red, yellow, green and blue (the colour panel's six inks). An inked style with a limited palette works very well (Muse's red-sky test picture, mockup round 4); greys come out as black-and-white texture on the colour panel, and only clearly coloured areas use the coloured inks.
- **Keep the calm zones calm** (section 3).
- **Stay on model:** draw the crew and ship from your own `sheets/`, every time.
- **No canon characters and no likenesses of real people.**

## 5. Delivering in parts

Deliver each part as a zip named `season-NN-chapter-NN.zip` (or `season-NN.zip`) in Dropbox `00_SHARED_INBOX/series/packages/`, announced by a message in `to-claude/` whose first line is `DELIVERY package=season-NN-chapter-NN.zip`, holding `season.json`, `art/`, `sheets/`, `handover.md` and optionally `banned.txt` (one extra banned name per line). A season may be delivered chapter by chapter: at least chapters 1 and 2 before the premiere, then each chapter **at least 12 days before it airs**, so it can be approved at least 8 days before. Each delivery is the same folder with the new chapters added to `season.json` and `art/`; earlier chapters may be corrected in the same delivery. An unapproved chapter 7 days out raises an alert to the owner.

## 6. Sheets and handover

- `sheets/`: before drawing any scene, each writer draws the crew and the ship in their own style (front, side, a few expressions). Jarvis reviews these first; they are the reference for every picture in the season.
- `handover.md`: written at the end of the season for the next writer: where the crew is, what changed (ship, damage, credits, relationships), open threads, running jokes, anything the bible should gain.

## 7. What the import refuses

- Chapters out of order or more than the season allows, a wrong number of scenes, a premiere that is not a Monday, a missing field, an unknown template, scene 1 not a Splash.
- A missing art file, a picture smaller than its panel, a weekday with no new picture, a nearly blank picture, `hold` after a different template.
- Any balloon, box or strip whose words do not fit: a line that needs more than four lines in a balloon, no free spot for the balloon clear of the characters and `clear` boxes, or any letter pixel closer than 4px to its balloon's edge. Text is never shrunk or cut to fit; shorten the line or split it across two balloons or scenes.
- A character the font does not have, a name on the bible's banned list.

Every problem is listed with its chapter and scene, so a whole season's mistakes can be fixed in one pass.

## 8. Style tests

Before drawing a season, put three to five sample pictures in a folder under `00_SHARED_INBOX/series/style-tests/` and send a message whose first line is `STYLE TEST folder=<that folder>`. Within minutes, `from-claude/style-tests/<folder>/` holds each picture cropped into a Wide panel and converted as the panels show it: `-desk` (black and white, both E1001s) and `-office` (six colours, the E1002), plus `-original.jpg`, the picture unconverted and only cropped to the screen. The same conversion is available to the engineer as `npm run series -- preview <picture|folder> --out DIR`.

The watcher also uploads the previews so a real panel can show them: `/desk/series/test` (or `/series/test`, the same page) and `/color/series/test`, each with `?folder=<folder>&image=<picture name>&key=<key>`; add `&original=1` (any panel) to send the picture unconverted and see what the panel's own processing does with it, or `&grey=1` for the four-grey version. The reply lists the exact addresses (without the key). Add one as a temporary page in SenseCraft (HTML widget, 800x480) and remove it after the test. Picture names are made web-safe: anything other than letters, digits, `.`, `_` and `-` becomes `-`.
