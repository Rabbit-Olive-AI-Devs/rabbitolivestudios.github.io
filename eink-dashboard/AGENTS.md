# E-Ink Dashboard — AGENTS.md

> Place this at `~/projects/rabbitolivestudios.github.io/eink-dashboard/AGENTS.md`.

## What This Is

"Moment Before" — a Cloudflare Workers backend for e-paper displays. Every day it generates an AI illustration depicting a famous historical event at its most iconic, dramatic moment.

**Devices:**
- reTerminal E1001: ESP32-S3, 7.5" ePaper, 800x480, monochrome (4-level grayscale)
- reTerminal E1002: ESP32-S3, 7.3" E Ink Spectra 6, 800x480, 6-color

**Stack:** Cloudflare Workers, Hono framework
**AI Models:** Llama 3.3 70B (event selection), FLUX.2 + SDXL (image generation)
**Location:** Naperville, IL (for weather data)

## Daily Pipeline

1. Fetch historical events for today from Wikipedia
2. LLM picks the most visually dramatic event
3. Image model generates illustration with daily rotating art style
4. Two versions: 4-level grayscale PNG (FLUX.2) + 1-bit PNG (SDXL, 6 rotating styles)

## Also Serves

- Weather data for Naperville, IL
- Steel/trade headlines
- World Skyline Series
- Daily "On This Day" historical fact

## Project Structure

```
eink-dashboard/
├── CLAUDE.md           # Session guidelines
├── MEMORY.md           # Authoritative technical truth
├── DECISIONS.md        # Standing decisions + failed approaches
├── README.md           # Architecture, endpoints, pipelines
├── src/                # Cloudflare Workers source
├── scripts/            # Utility scripts
└── photos/             # Device photos
```

## Rules

- Before deploying: `npx wrangler deploy --dry-run`
- Visual changes: test in browser at 800x480 before deploying
- Pipeline/cache changes: bump relevant cache key version
- `MEMORY.md` is authoritative. When sources conflict: MEMORY.md > DECISIONS.md > README.md > chat logs
