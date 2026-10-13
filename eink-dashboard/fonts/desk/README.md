# Desk panel fonts

Source fonts for the desk panel pages. They are **not** shipped to the Worker as font files:
`npm run desk-fonts` rasterises them into pure 1-bit glyph bitmaps in `src/hud/fonts-data.ts`
(DECISIONS #67).

| File | Font | Licence |
|---|---|---|
| `IBMPlexMono-Regular.ttf`, `-Medium.ttf`, `-SemiBold.ttf` | IBM Plex Mono, from <https://github.com/google/fonts/tree/main/ofl/ibmplexmono> | SIL Open Font License 1.1, see `IBMPlexMono-OFL.txt` |
| `AurebeshAF-Canon.otf` | Aurebesh AF (Canon) by AurekFonts, from <https://aurekfonts.github.io/?font=AurebeshAF> | The font's own name table reads "CC-0 public domain, created by AurekFonts" |

Aurebesh is the Star Wars alphabet; the use here is personal and non-commercial.
