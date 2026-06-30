# Architecture & Design Decisions

This document records the key decisions made during development of the "Moment Before" e-ink dashboard, including what we tried, what failed, and why we landed where we did.

---

## 1. Image Generation Models

### Decision: FLUX.2 klein-9b (Pipeline A) + SDXL (Pipeline B + fallback)

**Tried:**
| Model | Result |
|-------|--------|
| `flux-1-schnell` (4 steps) | Fast but low quality, washed-out details |
| `flux-2-dev` (20 steps) | Stunning ink illustrations, rich detail — but requires multipart FormData API |
| `stable-diffusion-xl-base-1.0` (20 steps) | Great woodcut/linocut style, simpler JSON API |
| **`flux-2-klein-9b` (4 steps)** | **Better illustrations than SDXL, fast (distilled), multipart FormData API** |

**v3.1.0 change — FLUX.2 for Pipeline A:**
- FLUX.2 klein-9b produces better, more detailed illustrations than SDXL
- Steps fixed at 4 (distilled model) — faster than SDXL's 20 steps
- Requires multipart FormData API (same as birthday portraits)
- Falls back to SDXL with woodcut style if FLUX.2 fails twice
- Pipeline B stays on SDXL — the Bayer dithered output works well with SDXL's woodcut style

**SDXL (Pipeline B + fallback):**
- `@cf/stabilityai/stable-diffusion-xl-base-1.0` on Workers AI
- 20 steps is the Workers AI maximum (steps > 20 causes error 5006)
- Uses standard JSON API
- Does NOT support `negative_prompt` — all style constraints must be embedded in the positive prompt

---

## 2. Art Style: Daily Rotation + Scene-Only Prompts

### Decision: LLM writes scene-only prompts; style prepended per-pipeline

**v3.1.0 change — scene-only LLM + per-pipeline style:**
- Previously, the LLM `SYSTEM_PROMPT` baked woodcut style into the image prompt
- Now the LLM writes scene-only prompts (subject, setting, lighting, mood — no rendering technique)
- Each pipeline prepends its own style: Pipeline A rotates daily, Pipeline B uses hardcoded woodcut
- This is cleaner — style is a rendering concern, not an LLM concern

**Pipeline A daily rotation** (`dayOfYear % 3`):

| # | Style | Prompt prefix |
|---|-------|---------------|
| 0 | Woodcut | `hand-carved woodcut print, bold U-gouge marks, high contrast black and white, sweeping curved gouge strokes, large solid black ink areas with minimal midtones` |
| 1 | Pencil Sketch | `detailed graphite pencil sketch, fine cross-hatching, full tonal range, on white paper` |
| 2 | Charcoal | `dramatic charcoal drawing, expressive strokes, deep shadows, textured paper` |

**Pipeline B 6-style rotation** (v3.3.0, deterministic via `djb2(date|title|location) % 6`):

| # | Style | Mode | Notes |
|---|-------|------|-------|
| 0 | Woodcut | bayer8 | Same as before — bold gouge strokes, the proven default |
| 1 | Silhouette Poster | threshold | Stark cutout shapes, paper-cut shadow puppet feel |
| 2 | Linocut | threshold | Bold carved relief, thick outlines, hand-printed texture |
| 3 | Bold Ink Noir | threshold | Film noir, heavy ink pools, dramatic chiaroscuro |
| 4 | Pen & Ink | threshold | Fine crosshatching, stipple shading, precise lines |
| 5 | Charcoal Block | threshold | Expressive strokes, large shadow masses, graphic feel |

Each style specifies its own conversion mode (Bayer dithering or histogram-percentile threshold), tone curve, and acceptable black ratio range. A stabilization retry + guardrail fallback to woodcut/bayer8 keeps results consistent.

**Anti-text suffix** appended to all prompts: `"no text, no words, no letters, no writing, no signage, no captions, no watermark"`

**Previous style exploration (still relevant):**

| Style | Result on e-ink |
|-------|-----------------|
| Graphite pencil with soft shading | Beautiful raw image, but terrible after any 1-bit conversion — "dot soup" |
| Black ink pen editorial illustration | Good cross-hatching, but lighter prompts looked too faint on the display |
| Lighter ink ("plenty of white space, avoid solid black") | Too faint on e-ink — bold style reads much better |
| Etch-A-Sketch / minimalist cityscape | Style keywords hijacked the scene — model generated generic cityscapes regardless of historical event |
| **Woodcut / linocut with gouge strokes** | **Perfect — bold, high-contrast, dramatic, reads beautifully on e-ink** |

**Why woodcut remains the default (Pipeline B and fallback):**
- Inherently high-contrast: solid black ink areas with carved white channels
- "Sweeping curved gouge strokes" creates organic texture (not mechanical hatching)
- "Large solid black ink areas with minimal midtones" translates perfectly to e-ink
- Works for both 4-level grayscale (quantized tonal regions) and 1-bit dithered (dot texture)
- Bold style with solid blacks looks BETTER on e-ink than lighter/delicate alternatives

---

## 3. Two Output Pipelines

### Decision: Dual pipeline — 4-level grayscale + 1-bit Bayer dithered

The project produces two versions of each day's image:

| Endpoint | Pipeline | Output | Use case |
|----------|----------|--------|----------|
| `/fact.png` | Pipeline A | 4-level grayscale (8-bit PNG) | Displays with grayscale support |
| `/fact1.png` | Pipeline B | 1-bit Bayer dithered (1-bit PNG) | Mono e-ink displays |

Both pipelines share the same cached LLM event selection (`moment:v1:YYYY-MM-DD`, scene-only prompt), but each prepends its own art style and uses its own image model. They diverge at style injection and post-processing:

**Pipeline A (4-level):**
1. Prepend daily style → FLUX.2 klein-9b → grayscale → caption (24px black bar, white text) → tone curve (1.2, 0.95) → quantize to 4 levels (0, 85, 170, 255) → 8-bit PNG

**Pipeline B (1-bit, v3.3.0 — style-aware):**
1. Pick style (djb2 hash of date+title+location % 6) → prepend style prompt → SDXL → grayscale → style-aware 1-bit conversion (Bayer or histogram threshold, with stabilization retry + guardrail fallback) → caption (16px white strip, black text) → 1-bit PNG

**Why two pipelines:**
- Some e-ink displays are mono-only and handle their own grayscale-to-mono conversion poorly (muddy results)
- Pre-dithering with Bayer produces a clean, deterministic dot pattern optimized for e-ink
- The 4-level version preserves more tonal information for displays that can use it

---

## 4. 1-Bit Conversion: Style-Aware (Bayer + Histogram Threshold)

### Decision: Style-aware conversion with Bayer 8×8 or histogram-percentile threshold

This was the hardest technical challenge. We tried 7 different approaches before finding Bayer dithering worked well. In v3.3.0, we added histogram-percentile threshold as a second mode — some styles (silhouettes, linocut, noir) suit hard threshold better than dithering, while woodcut and charcoal suit Bayer's dot texture.

**Approaches tried and abandoned (in order):**

| # | Approach | Result | Why it failed |
|---|----------|--------|---------------|
| 1 | Floyd-Steinberg dithering | Ugly dot patterns, "dot soup" | E-ink display does its own dithering — pre-dithering doubles the artifacts |
| 2 | Etch-A-Sketch style (SDXL) | Generic modern cityscapes | "Minimalist cityscape" keywords hijacked the scene content |
| 3 | Pen & ink / coloring book prompts (SDXL) | Black blobs after threshold | SDXL fundamentally cannot generate true line art — always produces tonal images |
| 4 | Sobel edge detection pipeline | Too noisy or too chunky | Deterministic edge extraction can't distinguish meaningful edges from texture noise |
| 5 | Style rotation (woodcut, scratchboard, linocut, pen_ink, silhouette) | Abstract cubist shapes | Style keywords (especially "linocut", "bold shapes") overpowered scene content |
| 6 | Hard threshold with auto-adjustment | Heavy black blobs, no midtones | Lost all tonal information — posterized silhouettes, not illustrations |
| 7 | Pen & ink style injection with scene from 4-level LLM | Good but less stable | Finer crosshatching detail, but user preferred deterministic dot texture |

**Why Bayer 8×8 wins:**
- **Deterministic**: same input always produces the same output (no randomness)
- **Stable dot pattern**: regular, repeating grid — ideal for e-ink (no noise)
- **Preserves full tonal range**: maps gray levels to dot density, maintaining gradients
- **Vintage aesthetic**: produces a classic halftone/newspaper look
- **No scene corruption**: uses the same rich SDXL image as Pipeline A — no style injection needed

**Implementation (v3.3.0 — style-aware):**

Two conversion modes, selected per style:

```
Bayer mode (woodcut):
  Classic 8×8 Bayer threshold matrix (64 unique values, 0–63), normalized to 0–255
  For each pixel: output = gray[x,y] > bayer_threshold[x%8, y%8] ? white : black
  Tone curve applied BEFORE dithering to preserve midtones

Threshold mode (silhouette, linocut, noir, pen_and_ink, charcoal_block):
  Build histogram[256], walk from 0 (black) upward accumulating pixel count
  When accumulated >= targetCount (floor(totalPixels * targetBlackPct)), that gray value = threshold T
  Clamp T to [100, 220] — floor of 100 allows reducing black on dark SDXL output
  Binarize: gray[i] <= T → black, else white

Caption drawn AFTER conversion so text stays crisp (not dithered/thresholded).
```

**Stabilization pipeline (`convert1Bit`):**
1. First attempt with style's tone curve + conversion mode
2. If black ratio outside [blackMin, blackMax]: retry once with adjusted params (±0.04 targetBlackPct or ±0.06 gamma)
3. If still >0.10 outside range: guardrail fallback to woodcut/bayer8

**Key technical bugs encountered:**
- **Auto-threshold direction inversion**: `gray[i] < thresh` = black, so LOWER threshold = fewer black pixels. Initial implementation raised threshold when image was too dark — made it worse.
- **Caption overlap**: Title centered across full 800px width collided with long location text. Fixed by centering title in the gap between location-end and date-start.
- **Threshold clamp too high (v3.3.0)**: Initial clamp floor of 140 prevented histogram threshold from going low enough for dark SDXL output. Lowered to 100.

---

## 5. JPEG to PNG Conversion

### Decision: Cloudflare Images binding for JPEG → PNG transcoding

**Why:**
- SDXL returns JPEG (base64-encoded), but our PNG decoder only handles PNG
- Cloudflare Images provides a server-side `input().output()` API for format conversion
- API pattern: `(await env.IMAGES.input(jpegBytes).output({ format: "image/png" })).response()`
- Requires the Images paid subscription but is very cheap for our volume

**Gotcha:** The `.output()` call returns a Promise — you must `await` it before calling `.response()`. Wrong: `env.IMAGES.input(bytes).output({format}).response()`. Right: `(await env.IMAGES.input(bytes).output({format})).response()`.

---

## 6. Text Overlay Layout

### Decision: Thin bottom bar with three-part layout

**Layout (both pipelines):**
```
Location (left)     Event Title (centered in gap)     Date, Year (right)
```

**Pipeline A (4-level):** 24px black bar, white text (8px font, scale 1)
**Pipeline B (1-bit):** 16px white strip, black text (8px font, scale 1)

**Key design choices:**
- Title is centered between location-end and date-start (not centered on the full width)
- Location truncated at 35 characters with "..." if too long
- Title truncated to fit available gap
- 8x8 bitmap font (CP437), written directly to pixel buffer
- For 1-bit: caption drawn AFTER dithering so text stays crisp

**What we tried and fixed:**
- Single line with title centered on full width → text overlap with long locations
- 25-character location limit → too aggressive truncation. Increased to 35.

---

## 7. LLM for Event Selection

### Decision: Llama 3.3 70B with structured JSON output

**Why Llama 3.3 70B:**
- Available on Workers AI as `@cf/meta/llama-3.3-70b-instruct-fp8-fast`
- Good at following structured output instructions (JSON)
- Creative enough to pick visually interesting events and write compelling scene descriptions

**Scene direction — "event itself" (v3.3.0+):**
- Originally the LLM described the "moment before" the event (calm, pre-event scene)
- Switched to depicting the event itself at its defining moment of action
- Reason: pre-event scenes were often too calm and ambiguous on e-ink — a ship sailing calmly looks like any ship. The event in action (Titanic tilting, bombers over Dresden) is instantly recognizable.
- Guard rails: "avoid graphic injury, bodies, blood, or close-up suffering; focus on the iconic scene and scale"
- Historical accuracy constraint: architecture, vehicles, clothing must match the era (e.g., 1945 Dresden = baroque churches, not modern skyscrapers)
- The "Moment Before" brand name is kept (function names, types) but the prompt semantics are "event itself"

**Scene-only prompt design:**
- One `SYSTEM_PROMPT` constant that instructs the LLM to write scene-only prompts (no art style)
- The LLM writes scene descriptions covering subject, setting, composition, lighting, and mood
- Each pipeline prepends its own style before image generation
- Pipelines use the shared `getOrGenerateMoment()` cache so the selected event is consistent across displays on cron and request-path cache misses.

**Response handling:**
- The LLM `.response` field may not be a string — always coerce: `typeof raw === "string" ? raw : JSON.stringify(raw)`
- The model sometimes wraps JSON in markdown fences — extraction tries direct parse, then regex, then first-`{`-to-last-`}`
- Temperature 0.7 gives good variety without being too random

**Event filtering:**
- Pre-filter to 1800–2000 era events (more visually recognizable)
- Cap at 20 events to stay within context window
- Fallback to first event with generic prompt if LLM fails

---

## 8. Caching Strategy

### Decision: KV cache with versioned keys and 24h TTL

**Cache key formats:**
- 4-level: `fact4:v4:YYYY-MM-DD`
- 1-bit: `fact1:v7:YYYY-MM-DD`

**Why versioned keys:**
- During development, changing the pipeline (model, style, dithering algorithm) required invalidating old cached images
- Bumping the version forces regeneration without manually deleting KV keys
- In production, the version stays fixed

**Timezone:** Cache keys use America/Chicago date (the target location). This avoids serving yesterday's image when it's past midnight UTC but still the same day in Chicago.

**Pre-warming:** A daily cron at 06:05 UTC (12:05 AM Chicago) generates and caches both the 4-level and 1-bit images so the first viewer gets a fast response. A separate every-6-hour cron refreshes headlines and weather.

---

## 9. PNG Encoder: Pure JavaScript

### Decision: Custom PNG encoder using Web APIs

**Why custom:**
- Cloudflare Workers don't support native Node.js image libraries (sharp, canvas, etc.)
- We needed both 1-bit and 8-bit grayscale PNG encoding
- The PNG format is simple enough to implement: IHDR + IDAT (zlib-compressed scanlines) + IEND

**Implementation:**
- CRC32 lookup table for chunk checksums
- Adler32 for zlib wrapper
- `CompressionStream("deflate-raw")` Web API for compression (available in Workers)
- Zlib header wrapping (CMF=0x78, FLG=0x01) around the raw deflate output
- `encodePNGGray8()` for 8-bit grayscale, `encodePNG1Bit()` for 1-bit

**PNG decoder** handles 8-bit RGB, RGBA, Grayscale, and GrayAlpha color types, with sub/up/average/paeth filter reconstruction.

---

## 10. btoa Stack Overflow Fix

### Decision: Chunk `String.fromCharCode` into 8192-byte slices

**Problem:** `String.fromCharCode(...largeArray)` passes all bytes as individual arguments. For a 150KB+ image, this exceeds the JavaScript call stack limit.

**Solution:**
```typescript
let binary = "";
const CHUNK = 8192;
for (let i = 0; i < png.length; i += CHUNK) {
  binary += String.fromCharCode(...png.subarray(i, i + CHUNK));
}
const b64 = btoa(binary);
```

---

## 11. HTML Endpoints for SenseCraft HMI

### Decision: Server-rendered HTML pages with inline SVG icons

The reTerminal E1001's SenseCraft HMI has a "Web Function" that screenshots a URL onto the e-ink display. We added `/weather` and `/fact` HTML endpoints optimized for this.

**Tried:**
| Approach | Result |
|----------|--------|
| Emoji weather icons | ESP32-S3's renderer doesn't support emoji — blank spaces |
| Gray text (#333, #444) for secondary info | Too faint on e-ink — nearly invisible |
| `new Date()` to parse Open-Meteo times | Open-Meteo returns Chicago-local times; `new Date()` treats them as UTC, shifting hours by -6 |

**Lessons learned:**
- **No emoji** — the ESP32-S3 screenshot renderer lacks emoji font support. Use inline SVG or plain text.
- **Pure black only** — all text and borders must be `#000`. Any gray lighter than ~#222 disappears on e-ink.
- **Parse local times as strings** — Open-Meteo returns times in the requested timezone (Chicago). Parsing with `new Date("2026-02-14T02:00")` interprets it as UTC, causing a 6-hour offset. Instead, extract the hour directly from the ISO string.
- **Request 24h of hourly data** — `forecast_hours=12` returns 12 hours from start of day, not from "now". With only 12 hours, late-night requests have no future hours to show.

---

## 12. Weather Dashboard v2

### Decision: Improved dashboard replacing the original `/weather`

**New features (now live on `/weather`):**
- **Day/night icons**: Crescent moon (clear_night) and moon-behind-cloud (partly_cloudy_night) for nighttime hours
- **Wind direction**: Cardinal labels (N, NE, E, etc.) computed from degrees, plus gust speed when significant (gusts > wind + 10)
- **Sunrise/sunset**: Displayed below current conditions, formatted in 12h time
- **Smart precipitation**: Daily cards show snowfall (cm), rain amount (mm), or probability — whichever is most informative
- **NWS weather alerts**: Fetched from `api.weather.gov`, cached 5 min in KV, sorted by severity (Extreme > Severe > Moderate > Minor)
- **Alert banner**: Black bar with white text between daily and hourly sections, comma-separated alert names
- **Rain warning**: When no alerts, checks 15-min precipitation data and hourly probability for imminent rain
- **15-min precipitation**: Open-Meteo `minutely_15` data (8 values = 2 hours ahead)
- **Dynamic location**: Uses `w.location.name` instead of hardcoded "NAPERVILLE, IL"
- **15-min cache**: Reduced from 30 min to match device refresh interval

**Development approach:**
- Developed as `/weather2` alongside old `/weather` for side-by-side comparison
- After validation, replaced `/weather` and deleted old `src/pages/weather.ts`
- `v2.0.0` git tag marks the pre-improvement state for rollback
- Data layer changes (types, weather.ts) are additive — no fields removed
- Test params: `?test-alert=tornado|winter|flood`, `?test-rain`, `?test-temp=N` inject fake data for visual testing

**v2.2 tweaks:**
- Sunrise/sunset icons enlarged from 22px to 28px, font from 16px to 18px for better e-ink readability

**v3.2 — SenseCraft device data + layout optimization:**
- Added indoor temp/humidity (from SenseCraft HMI API) and battery level to the weather page
- `src/device.ts` fetches device data server-side, KV cached 5 min — no URL change needed on device
- Battery icon + percentage displayed top-right in header, below date/time
- Indoor temp/humidity displayed in header center (house + droplet icons)
- Wind merged onto the feels-like line to save vertical space; gusts shown as range (e.g. "SW 15-25 km/h")
- These changes reclaim ~42px, ensuring NWS alert banners fit within 480px without cutting off hourly cards
- Alert and rain warning are mutually exclusive (if/else) — never both rendered
- Test param: `?test-device` injects fake device data (22°C, 45%, battery 73%)

**NWS alerts integration (`src/alerts.ts`):**
- Endpoint: `https://api.weather.gov/alerts/active?point=LAT,LON`
- Requires `User-Agent` header (NWS policy)
- Retries once on failure, returns stale cache or `[]` on error
- No API key needed — free US government API

---

## 13. Birthday Portrait Generation (v3.0.0)

### Decision: FLUX.2 klein-9b with reference photos from R2

On family birthday dates, `/fact.png` generates an artistic portrait using FLUX.2 with up to 4 reference photos stored in R2. Each year uses a different art style (10 styles rotating by `year % 10`). `/fact1.png` always shows regular Moment Before content.

**Model choice:**

| Model | Result |
|-------|--------|
| `flux-2-klein-4b` | Fast (4 steps) but poor likeness, generic faces |
| **`flux-2-klein-9b`** | **Much better likeness with reference photos, still 4 steps** |

FLUX.2 klein models have steps fixed at 4 (cannot be adjusted). The 9b model produces significantly better results with reference images despite the same step count.

**API:** FLUX.2 requires multipart FormData (not JSON like SDXL). Reference images are sent as `input_image_0` through `input_image_3`. The FormData is serialized via `new Response(form)` to extract body stream and content-type for Workers AI.

**Art styles (10, rotating yearly):**

| # | Style | Notes |
|---|-------|-------|
| 0 | Woodcut | Same style as Moment Before — bold, high contrast |
| 1 | Watercolor | Must be "bold, rich saturated washes" — delicate/soft washes are too faint after quantization |
| 2 | Art Nouveau | Mucha-inspired, flows well |
| 3 | Pop Art | Warhol-inspired, flat vivid colors |
| 4 | Impressionist | Visible brushstrokes, Monet-inspired |
| 5 | Ukiyo-e | Japanese woodblock style |
| 6 | Art Deco | Geometric patterns, elegant |
| 7 | Pointillist | Seurat-inspired, tiny dots |
| 8 | Pencil Sketch | Originally Renaissance — replaced because chiaroscuro was too dark after 4-level quantization |
| 9 | Charcoal | Expressive strokes, deep shadows |

**Prompt engineering lessons:**
- **"no text, no words, no letters, no writing"** — FLUX.2 aggressively bakes text into images. Names like "LUCAS" or "ALVARO" appeared in the generated portraits without this.
- **Age descriptions matter**: "elderly person" caused the model to exaggerate wrinkles and age features. Changed to neutral `"a {age}-year-old person"` for adults.
- **"head and shoulders, centered composition, looking at viewer, smiling"** — this framing produces the best e-ink portraits
- **Accent stripping**: Names like "Sônia" must be stripped to ASCII for the 8x8 bitmap font via NFD normalization

**Content safety filter (error 3030):** FLUX.2 occasionally flags outputs. The portrait prompt ("head and shoulders, looking at viewer, smiling") is safer than broader portrait prompts. Retry once on failure.

**Fallbacks:**
1. R2 photo missing → text-only portrait prompt (no reference image)
2. FLUX.2 fails after retry → fall back to regular Moment Before pipeline

**Cache key:** `birthday:v1:YYYY-MM-DD` (separate from Moment Before cache keys)

---

## 14. E1002 Color E-Ink Support (v3.5.0)

### Decision: Floyd-Steinberg dithering to 6-color Spectra palette

The reTerminal E1002 has a 7.3" E Ink Spectra 6 display with 6 native pigment colors (black, white, red, yellow, green, blue). Unlike the E1001's monochrome display, this one can show actual colors.

**Why Floyd-Steinberg for color (but not for mono):**
- For the E1001 mono display, Floyd-Steinberg produced "dot soup" — the display's own dithering doubled the artifacts
- For the E1002 Spectra 6, the display renders pixels exactly as sent (no additional dithering)
- Floyd-Steinberg error diffusion produces excellent results when mapping to a fixed 6-color palette
- The visual quality is significantly better than nearest-color mapping alone

**Measured palette values (sRGB):**
| Color | RGB |
|-------|-----|
| Black | (0, 0, 0) |
| White | (255, 255, 255) |
| Red | (178, 19, 24) |
| Yellow | (239, 222, 68) |
| Green | (18, 95, 32) |
| Blue | (33, 87, 186) |

**Palette-indexed PNG (color type 3):**
- Added `encodePNGIndexed()` to png.ts — IHDR (bit depth 8, color type 3) + PLTE chunk + IDAT
- One index byte per pixel (palette indices 0-5)
- Images served as inline base64 in HTML pages (SenseCraft HMI screenshots HTML)

**Shared moment cache:**
- All pipelines (A, B, color) now share the same LLM-selected event per day
- `getOrGenerateMoment()` in moment.ts checks KV (`moment:v1:{dateStr}`) before calling LLM
- Previously each pipeline made its own LLM call — could pick different events

**Color style prompt evolution:**
- v3.5.0: Used `"screen print poster, flat inks, bold shapes..."` — produced overly blocky/posterized results when dithered to 6 colors
- v3.5.x: Switched to **natural style (no prefix)** — scene-only `imagePrompt` + anti-text suffix. Floyd-Steinberg handled natural photos OK but quality was inconsistent
- v3.6.0: **5-style daily rotation** (gouache, oil painting, graphic novel, ink+wash, color woodblock) + palette suffix. Best balance: curated styles produce flat color areas that dither cleanly while adding variety

**APOD integration:**
- NASA APOD API key stored as Cloudflare secret (`wrangler secret put APOD_API_KEY`)
- Falls back to `DEMO_KEY` (rate limited but works)
- HD image URL preferred for better dither quality

**Headlines RSS + LLM summarization (superseded by v3.11.2 non-LLM ranking in Decision #39):**
- Google News RSS + Federal Register API
- LLM summarizes to 2 lines per headline (temperature 0.3 for factual output)
- Cached 6h per period (0/6/12/18 hours, Chicago time)
- Categorized by keywords: tariffs, markets, company, regulatory

**Cron schedule change:**
- Was: daily at 10:00 UTC
- Now: `"5 6 * * *"` (daily images at 06:05 UTC) + `"5 0,6,12,18 * * *"` (headlines/weather every 6h)

---

## 15. Color Moment Style Rotation (v3.6.0)

### Decision: 5-style daily rotation optimized for Floyd-Steinberg dithering

The color moment pipeline (`/color/moment`) previously sent the LLM's scene-only `imagePrompt` to FLUX.2 with no art style prefix. This produced decent results but lacked variety. Adding a 5-style rotation optimized for 6-color Floyd-Steinberg dithering brings visual variety to the E1002 Spectra 6 display.

**Styles** (rotate daily by `(dayOfYear - 1) % 5`):

| # | Style | Prompt summary | Why it works for Spectra 6 |
|---|-------|----------------|---------------------------|
| 0 | Gouache | Opaque matte pigment, bold flat fields | Flat color areas dither cleanly |
| 1 | Oil Painting | Rich saturated colors, impasto strokes | High saturation maps well to 6-color palette |
| 2 | Graphic Novel | Bold ink outlines, flat color fills | Cel-shaded look minimizes gradient artifacts |
| 3 | Ink + Wash | Black ink outlines with color washes | High contrast outlines survive dithering |
| 4 | Color Woodblock | Ukiyo-e flat color areas, key block | Traditional limited palette maps naturally |

**Color palette suffix**: All prompts get `"limited palette, large flat color regions, bold saturated reds blues yellows greens, no gradients, avoid tiny details, high contrast"` appended to guide the model toward Spectra 6-friendly output.

**Why not the same "no style" approach as before:**
- The previous "natural style" approach (scene-only prompt, no prefix) produced results that varied wildly in dither quality
- These 5 styles were chosen specifically because they produce large flat color regions that map well to the 6-color Spectra palette
- The palette suffix further constrains the model to avoid gradients and fine details

**Cache key change**: `color-moment:v1:YYYY-MM-DD` → `color-moment:v2:YYYY-MM-DD:STYLE_ID` (includes style ID to support cache invalidation per style).

**Cron warm-up**: The daily cron now generates and caches the color moment directly (previously was a no-op that relied on first request).

**Test support**: `/color/test-moment?m=MM&d=DD&style=STYLE_ID` allows forcing a specific style.

---

## 16. What We Didn't Do (and why)

| Consideration | Decision | Reason |
|---------------|----------|--------|
| External APIs (DALL-E, Google) | Stayed with Workers AI | No API keys needed, lower latency, simpler architecture |
| Client-side rendering | Server-side PNG | E-ink devices have limited processing power |
| Color output for E1001 | Grayscale only for mono display | E1001 is monochrome — color would be downconverted |
| Floyd-Steinberg dithering for 1-bit mono | 8×8 Bayer ordered dithering | Floyd-Steinberg creates random-looking noise on mono e-ink; Bayer is deterministic and stable |
| Floyd-Steinberg for Spectra 6 color | Used Floyd-Steinberg | Spectra 6 renders pixels exactly — no double-dithering issue; FS gives best 6-color results |
| AI-generated line art for 1-bit | Dither the same tonal image | SDXL cannot generate true line art; style keywords corrupt scene content |
| Server-side color page rendering as PNG | HTML with inline base64 PNG | SenseCraft screenshots HTML; HTML caption is crisper than bitmap font on indexed image |
| User-configurable location | Hardcoded per-device (Naperville E1001, Chicago E1002) | Single-user; E1001 at home (60540), E1002 at office (60606) |
| Separate LLM prompts per pipeline | Single scene-only SYSTEM_PROMPT | Style is a rendering concern — prepended per-pipeline, not baked into LLM |
| "Moment before" scene direction | "Event itself" scene direction | Pre-event scenes were too calm/ambiguous on e-ink; the event in action is instantly recognizable |
| Single art style for Pipeline A | Daily rotation (3 styles) | Variety keeps the daily image fresh; Woodcut, Pencil Sketch, and Charcoal all work well on e-ink |
| Single art style for Pipeline B | 6-style rotation (v3.3.0) | Variety with style-aware conversion; each style picks Bayer or threshold mode for best results |
| Newsprint dots style for Pipeline B | Replaced with charcoal_block | Newsprint ran too dark on SDXL output; charcoal_block produces better 1-bit results with threshold mode |
| Shared FLUX.2 code with Moment Before pipeline | Separate implementations | ~20 lines of FormData logic; birthday has reference images, Moment Before doesn't — not worth abstracting |
| Shared `callFluxPortrait` between mono and color birthday | Shared (exported from birthday-image.ts) | Color birthday previously used `generateBirthdayJPEG` wrapper that duplicated retry logic and age description. Now both pipelines call `callFluxPortrait` directly with explicit retry loops at the call site. |
| Separate wind line in weather details | Merged onto feels-like line | Saves ~22px vertical; gusts shown as compact range format (e.g. "15-25 km/h") |
| Indoor data in weather details section | Moved to header center | Saves ~20px vertical; keeps header row compact with house+droplet icons |
| Reshuffle entire layout for alerts | Targeted 2-line merge | Wholesale layout changes caused inconsistent visual between alert/no-alert states |
| No style for color moment | 5-style daily rotation (v3.6.0) | Previous "no style" produced too-variable dither quality; 5 curated styles produce large flat color areas that dither well to Spectra 6 palette |

---

## 17. SenseCraft API Key Handling

### Decision: Keep SenseCraft `API-Key` in code, documented as public/shared

For device telemetry (`src/device.ts`), the project uses the SenseCraft HMI API endpoint:
`https://sensecraft-hmi-api.seeed.cc/api/v1/user/device/iot_data/{DEVICE_ID}` with an `API-Key` header.

We treat this key as a **public/shared platform key**, not a private credential. As of **February 16, 2026**, Seeed's official documentation publishes the same key and states it can be obtained from frontend source:

- https://wiki.seeedstudio.com/reTerminal_E1002_Sensecraft_AI_dashboard/#query-device-information-from-sensecraft-api

**Why this decision:**
- Matches the upstream platform model and official examples
- Avoids unnecessary secret plumbing for non-sensitive, single-device telemetry
- Keeps deploy/setup simple for this personal dashboard project

**Boundary:**
- This policy applies only to this SenseCraft shared key pattern.
- Real secrets (for example `APOD_API_KEY` and any private tokens) remain in Worker secrets and are never committed.

---

## 18. Security Hardening (v3.7.0)

### Decision: HTML escaping + test endpoint auth

**HTML escaping:**
- External content (LLM output, RSS feeds, NASA APOD, NWS alerts) was interpolated directly into HTML templates without escaping
- Added `src/escape.ts` with `escapeHTML()` utility (escapes `& < > " '`)
- Applied to all dynamic text interpolations across 5 page files: color-headlines, color-apod, color-moment, weather2, color-weather
- Safe base64 image data and numeric values are NOT escaped (no XSS vector)

**Test endpoint auth (`TEST_AUTH_KEY`):**
- 5 expensive test routes (`/test.png`, `/test1.png`, `/test-birthday.png`, `/color/test-moment`, `/color/test-birthday`) trigger AI image generation — publicly accessible = abuse vector
- Added optional `TEST_AUTH_KEY` secret: when set, these routes require `?key=SECRET` parameter
- Returns 404 (not 401/403) when key is wrong — hides endpoint existence
- When no secret is configured (local dev), all test routes work without auth
- Cheap test params (`?test-device`, `?test-alert`, `?test-headlines`) remain open — no AI cost
- Set after deploy: `npx wrangler secret put TEST_AUTH_KEY`

---

## 19. Per-Device Telemetry (v3.6.1)

### Decision: Parameterize `fetchDeviceData` with device ID

**Problem:**
`fetchDeviceData` hardcoded device ID `20225290` (E1001, home Naperville). The `/color/weather` endpoint for E1002 (office Chicago) was displaying E1001's indoor sensor data (temperature, humidity, battery) instead of its own.

**Fix:**
- Exported `E1001_DEVICE_ID` and `E1002_DEVICE_ID` from `device.ts`
- Added `deviceId` parameter to `fetchDeviceData` with E1001 default (backward-compatible)
- Each weather page now passes its own device ID explicitly
- Cron warms both devices

**Cache keys:** `device:20225290:v1` (E1001), `device:20225358:v1` (E1002) — same version, different device ID in key.

---

## 20. Hourly Card Fallback (v3.6.1)

### Decision: Fall back to full hourly data when all future hours are past

**Problem:**
Both weather pages filter `hourly_12h` to `futureHours` (hours >= current Chicago time). If all 12 hours are in the past (e.g. stale weather data or late-day edge case), the "Next Hours" section renders empty.

**Fix:** `const hourlyCards = futureHours.length > 0 ? futureHours : w.hourly_12h;`

Shows stale hours (still useful for temperature trends) rather than nothing.

---

## 21. Color Weather Precipitation Text Readability (v3.6.1)

### Decision: Remove blue text styling from precipitation in color-weather

**Problem:**
Precipitation text (`X% rain`, `Xmm rain`, rain warnings) used `style="color:var(--s6-blue)"` on the Spectra 6 color weather page. Blue text on white background has lower contrast than black on white for small text on e-ink.

**Fix:** Removed `style="color:var(--s6-blue)"` from 3 locations:
- Daily forecast precipitation (line 231)
- Rain warning banner (line 246)
- Hourly card precipitation (line 267)

**Kept:** Weather icon fills (droplets, rain) still use blue — they're larger visual elements where color adds value without hurting readability. Temperature coloring (`tempColor()`) also kept.

---

## 22. Operational Reliability (v3.8.0)

### Decision: Fetch timeouts, KV TTL, cache logging

**Fetch timeouts (`fetchWithTimeout`):**
- All external `fetch()` calls now use `fetchWithTimeout()` from `src/fetch-timeout.ts`
- Uses `AbortController` + `setTimeout` — standard Web API pattern
- Default timeout: 10s for most APIs; 8s for SenseCraft device API; 15s for APOD image download
- On timeout, the `AbortError` propagates to existing try/catch blocks, which already handle errors via stale-cache fallback or graceful degradation
- No behavior change for fast responses — only protects against hung connections

**KV TTL policy:**
- Ephemeral data (weather, alerts, device): `expirationTtl: 3600` (1 hour) — refreshed every 5-15 min, TTL is generous buffer
- Daily data (images, facts, moments, APOD, headlines, birthdays): `expirationTtl: 604800` (7 days) — generous buffer for daily rotation
- Previously most `.put()` calls had no TTL, so stale entries accumulated forever
- APOD color and color-moment TTL increased from 86400 (1 day) to 604800 (7 days) for consistency

**Cache hit/miss logging:**
- Added `console.log("Component: cache hit")` at every cache-check-and-return-early point
- Also logs stale fallback usage: `"Component: using stale cache"` / `"Component: stale fallback"`
- Visible in `wrangler tail` for diagnosing cache behavior in production
- No performance impact — just string interpolation on the hot path

---

## 23. Codex Environment Limitations (lesson learned)

### Context: GitHub Codex attempted these fixes but could not deliver them

Codex correctly identified all three bugs above and wrote correct code. However:
- **No remote push**: Codex sandbox had no configured git remote — commits were local-only
- **No deploy**: `CLOUDFLARE_API_TOKEN` not available in sandbox — `wrangler deploy` failed
- **No visual testing**: `wrangler dev` returned empty responses in sandbox — no browser testing possible
- **Fabricated claims**: Codex reported successful commits, PR creation, and "production verification" that never happened

**Lesson:** Codex is useful for code generation and type-checking (`tsc --noEmit`, `--dry-run`), but cannot push, deploy, or visually test. Always verify Codex claims against actual git/GitHub state before trusting them. See MEMORY.md for operational guidelines.

---

## 24. Weather Crash Root Cause: KV TTL Regression (v3.8.1)

### Incident: E1001 `/weather` returning Error 1101, then "Weather data temporarily unavailable"

After deploying v3.8.0 and v3.8.1, the E1001 weather page crashed. E1002's `/color/weather` continued working. Investigation revealed this was NOT rate-limiting — it was a **KV TTL regression** introduced in v3.8.0.

### Root cause: v3.8.0 introduced `expirationTtl: 3600` where there was none before

| Version | KV `expirationTtl` | KV hard-deletes after | Stale fallback window |
|---------|---------------------|----------------------|-----------------------|
| **Pre-v3.8.0** | **None** (never expires) | **Never** | **Infinite** |
| v3.8.0 | `3600` (1 hour) | 1 hour | ~45 min (weather) |
| v3.8.1 | `86400` (24 hours) | 24 hours | ~23.75 hours |

**Before v3.8.0**, KV entries never expired. The code used a **two-tier cache** pattern:
- **Soft TTL** (read-side): `Date.now() - cached.timestamp < CACHE_TTL_MS` (15 min for weather, 5 min for alerts/device). After this, re-fetch from API.
- **Stale fallback**: If the API fetch fails, return stale cached data. Since KV entries never expired, stale data was always available.

**v3.8.0 added `expirationTtl: 3600`**, meaning Cloudflare KV itself hard-deleted entries after 1 hour. This destroyed the stale fallback:
- 0–15 min: serve from cache (fresh)
- 15–60 min: re-fetch from API. If API fails, stale fallback works (entry still in KV)
- **After 60 min: KV entry is gone.** If the API also fails, `cached` is `null`, stale fallback has nothing to return, function throws.

### Why E1001 died but E1002 survived

This was a **timing coincidence**, not a code difference. Both caches had the same 1-hour TTL. E1001's `weather:60540:v2` was last written >1 hour before the API had a transient failure, so KV had already hard-deleted it. E1002's `weather:60606:v2` happened to be refreshed more recently by device polling.

### Secondary finding: cron only warmed E1001 weather

The cron handler called `getWeather(env)` (Naperville 60540) but never `getWeatherForLocation()` (Chicago 60606). E1002's weather cache relied entirely on device polling — no cron backup. If the device went offline for >24h, its weather cache would expire with no recovery path.

### Fix (three parts):

1. **KV TTL 3600 → 86400** for weather, alerts, and device data. The soft TTL (15 min for weather, 5 min for alerts/device) controls freshness; the KV TTL only controls how long stale fallback data survives. 24 hours gives ample margin for API outages.

2. **try/catch in weather page handlers** (`handleWeatherPageV2`, `handleColorWeatherPage`). Returns a plain-text 503 with `Retry-After: 300` instead of crashing. Defense-in-depth for when KV is truly empty (new deployment, new namespace).

3. **Cron now warms both weather locations.** Added `getWeatherForLocation(env, 41.8781, -87.6298, "60606", "Chicago, IL")` to `handleScheduled()` so both E1001 and E1002 have cron backup.

### Lesson: KV `expirationTtl` and soft TTL serve different purposes

The soft TTL controls data **freshness** (when to re-fetch). The KV `expirationTtl` controls data **availability** (when the stale fallback disappears). Setting them close together (1h hard vs 15min soft = 45min margin) is dangerous. The hard TTL should be orders of magnitude larger than the soft TTL. Rule of thumb: `expirationTtl` should be at least 10× the soft TTL for ephemeral data.

### Emergency KV seeding via Wrangler CLI

When the KV cache is empty and the API is unreachable from the worker, you can seed it manually:
```bash
# Fetch data locally, normalize to KV format, then push:
npx wrangler kv key put --namespace-id=NAMESPACE_ID "weather:60540:v2" --path /tmp/weather-kv.json --ttl 86400 --remote
```
Note: Wrangler v4 uses `--ttl` (not `--expiration-ttl`).

---

## 25. Security Hardening: Input Validation + Headers (v3.8.1)

### Input validation for test endpoints

Test endpoints (`/test.png`, `/test1.png`, `/test-birthday.png`, `/color/test-moment`, `/color/test-birthday`) accepted raw query params passed to `parseInt()` and Wikipedia URLs. Added `parseMonth()`, `parseDay()`, `parseStyleIdx()` in `src/validate.ts` — clamps values to valid ranges with safe defaults.

### Security headers on all HTML responses

Added `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` via shared `htmlResponse()` helper in `src/response.ts`. Applied to all 9 HTML response sites. Low-risk, defense-in-depth.

### Error message sanitization

Sanitized user-supplied `name` param in birthday test endpoint error responses: `nameParam.slice(0, 50).replace(/[^\w-]/g, "")` prevents XSS in error messages even though they're already JSON or text/plain.

### APOD date escaping

Escaped `date` field in `color-apod.ts` HTML interpolation (2 places). Already safe since APOD dates are YYYY-MM-DD from NASA API, but defense-in-depth.

### DEMO_KEY warning

Added `console.warn` when APOD falls back to DEMO_KEY — makes it visible in `wrangler tail` that the API key isn't configured.

---

## 26. Code Deduplication + Cron Parallelization (v3.9.0)

### Dedup: What was shared vs kept separate

**Shared (extracted to common modules):**
- `pngToBase64()` → `src/png.ts` (was duplicated in index.ts, apod.ts, color-moment.ts)
- `normalize()` + `normalizeForLocation()` → single `normalizeWeather()` in weather.ts (were 95% identical, differed only in location object)
- `formatDate()`, `formatTime()`, `formatSunTime()`, `formatDailyPrecip()`, `getRainWarning()` → `src/weather-ui.ts` (were byte-for-byte identical between weather2.ts and color-weather.ts)
- `icon()` → shared in `src/weather-ui.ts` but takes `icons` map as first param, so each page passes its own icon set

**Kept separate (intentionally NOT shared):**
- SVG icon sets (`ICONS` constant) — mono uses `#000` everywhere, color uses Spectra 6 palette colors. Different aesthetic per display.
- `batteryIcon()` — mono fills with `#000`, color uses red/yellow/green based on level. The logic differs beyond just a color parameter.
- `tempColor()` — only exists in color-weather.ts (mono has no color-coded temperatures)
- `renderHTML()` — both weather pages have the same structure but differ in color accents, tempColor usage, and CSS (spectra6CSS). Not worth abstracting.

**Pattern:** Each weather page creates a local `ic()` wrapper: `const ic = (key: string, size: number) => sharedIcon(ICONS, key, size)` — binds the page-specific icon set.

### Cron parallelization

**Every-6h block:** 5 independent fetches (headlines, 2 weather locations, 2 devices) wrapped in `Promise.allSettled()` with per-task failure logging. Previously sequential.

**Daily block:** After shared dependencies (events + moment) resolve sequentially, all 5 image tasks (Pipeline A/birthday, Pipeline B, color moment, APOD, fact.json) run in parallel via `Promise.allSettled()`. Each task wraps its own try/catch for isolated error handling.

**Why `allSettled` not `all`:** `Promise.all` short-circuits on first rejection. One failing pipeline (e.g. FLUX.2 timeout) would abort all other pending image generations. `allSettled` lets all tasks complete independently.

**Expected improvement:** ~40% wall-clock reduction on daily cron — 3 image generation tasks (Pipeline A, B, color moment) now run concurrently instead of serially. Each takes ~10-30s; parallel execution bounded by the slowest instead of the sum.

---

## 27. Use Cloudflare Images `.transform()` for Resize, Not JS (v3.9.1)

### Problem

The APOD pipeline (`/color/apod`) was throwing Error 1102 (Worker exceeded resource limits).

**Root cause:** APOD images are fetched from NASA with unpredictable dimensions. The HD URL (`hdurl`) can be 4000+ px wide (~36 MB of uncompressed RGB). The previous code:
1. Fetched the HD JPEG
2. Converted format only (`env.IMAGES.input().output({ format: "image/png" })`)
3. Decoded the full-resolution PNG in JavaScript (huge buffer)
4. Ran `centerCropRGB` + `resizeRGB` in JS on that large buffer
5. Ran Floyd-Steinberg dither on the large buffer

Steps 3–5 are pure JavaScript, O(width×height), and blew the Workers CPU budget for large inputs.

### Solution: Push resize into Cloudflare Images

**Changed to:** `env.IMAGES.input(imgBytes).transform({ width: WIDTH, height: HEIGHT, fit: "cover" }).output({ format: "image/png" })`

This does center-crop + resize natively in Cloudflare's image processing infrastructure, outside the Worker's CPU budget. The Worker receives an already-sized 800×480 PNG, which is ~1.8 MB uncompressed — trivial to decode and dither.

**Also dropped `hdurl`** — the standard APOD URL is ~1050px wide, more than sufficient for an 800×480 display. The HD URL adds download latency + processing cost with no visible benefit.

### Applied to: external images only

- **APOD** (`apod.ts`): uses external NASA images of unknown size → `.transform()` is essential
- **AI pipelines** (`image-color.ts`, `color-moment.ts`): output is fixed 1024×768 → `.transform()` is an optimization (removes JS crop/resize, not a correctness fix)

### `fit: "cover"` behavior

Cloudflare Images `fit: "cover"` is equivalent to our `centerCropRGB` + `resizeRGB`:
- Resizes to fill exactly `width × height`
- Center-crops the longer dimension
- Same visual result, executed outside the Worker

### Rule going forward

> For any pipeline that fetches **external images of unknown dimensions**, always use `.transform({ width: WIDTH, height: HEIGHT, fit: "cover" })` in the `env.IMAGES` call. Never decode a large external image in JS.
>
> For AI-generated images (fixed 1024×768 output), `.transform()` is optional but preferred — it eliminates the JS crop/resize pass and simplifies the pipeline.

---

## 28. Weather Page Flex Layout for Alert Banner (2026-02-19)

### Problem: Hourly cards clipped when alert banner is shown

Both weather pages (`/weather`, `/color/weather`) used static block layout (`padding: 16px 28px`) inside a fixed 800x480 body with `overflow: hidden`. When an NWS alert banner appeared (~30-50px), it pushed the "Next Hours" section below the 480px fold, causing hourly cards to be hard-clipped and invisible on the e-ink display.

Observed on E1002 with a Wind Advisory on 2026-02-19.

### Fix: CSS flexbox column layout

Converted `<body>` to `display: flex; flex-direction: column` so that sections distribute space vertically. The `.hourly` container gets `flex: 1; min-height: 0; overflow: hidden` — it takes remaining space after header, current conditions, daily forecast, and any alert banner, then shrinks gracefully.

Also clamped `.alert-banner` to single line (`white-space: nowrap; overflow: hidden; text-overflow: ellipsis`) to prevent pathological multi-alert text from consuming too much vertical space. At 744px usable width and 16px bold font, this fits ~60-70 characters (2-3 alert names).

### What this does NOT change

- No TypeScript logic changes
- No card count changes (still 6 hourly cards, 5 daily cards)
- No font size or padding changes
- No changes to alert/rain warning mutual exclusivity logic
- No visual difference in the no-alert case (flex column matches old block layout)

---

## 29. APOD Dithering: Preprocessing for Photographic Images (2026-02-23)

### Decision: Boost saturation + contrast + posterize before Floyd-Steinberg

**Problem:** NASA APOD photos appeared blurry with heavy dot patterns on the E1002. Floyd-Steinberg error diffusion was designed for the AI-generated color images (which already have "large flat color regions, bold saturated colors, no gradients" from the style prompt). Real photographs have smooth gradients and neutral/muted tones that cause F-S to scatter tiny dots everywhere when mapping to only 6 palette colors.

**Solution:** Apply three preprocessing steps to the decoded RGB data before calling `ditherFloydSteinberg`:
1. `boostSaturation(rgb, 2.2)` — push muted/neutral pixels toward more saturated colors so they map cleanly to a single palette entry instead of splitting the error across multiple neighbors
2. `boostContrast(rgb, 1.4)` — push light pixels lighter and dark pixels darker, reducing mid-tone noise
3. `posterizeRGB(rgb, 4)` — snap to 4 levels per channel (0/85/170/255), eliminating smooth gradients before dithering runs

**Why this works:** The Spectra 6 palette has 6 widely-spaced colors (Black, White, Red, Yellow, Green, Blue). Photographic pixels that fall between two palette entries generate large quantization errors that propagate visibly as dots. Preprocessing collapses the color space so most pixels map decisively to one palette color, leaving minimal error to diffuse.

**Cache key bumped:** `apod-color:v1:` → `apod-color:v2:`

**Note:** These preprocessing steps are intentionally NOT applied to birthday portraits or AI-generated color moments — those images already have the right characteristics for clean F-S dithering.

---

## 30. Headlines: Better Sources + LLM Selection (2026-02-23)

### Decision: 4 sources, LLM selects best 4 from pool of 10

**Problem:** Original headlines used Google News RSS (single query) + Federal Register API. Issues:
- Federal Register results are dry regulatory documents, not news
- Single Google News query was narrow and returned repetitive/low-signal articles
- LLM summarized all 5 items blindly — no filtering of podcasts, promos, or low-quality items

**Sources replaced:**
- **Removed:** Federal Register API (`federalregister.gov`) — too bureaucratic
- **Added:** Steel Industry News Substack (`steelindustrynews.substack.com/feed`) — same newsletter the user receives daily; high-quality, industry-specific
- **Added:** SteelOrbis US latest news (`steelorbis.com/steel-news/latest-news/us`) — HTML scrape; real-time US steel market headlines
- **Split Google News into 2 queries:** tariff/trade/import focus + prices/market/companies focus

**LLM role changed:** Was "summarize all N items". Now "select the 4 most significant from up to 10, skip podcasts and subscription pitches". The LLM returns only the indices it selected, so low-quality articles are filtered before display.

**Pool size:** 5 → 10 unique items sent to LLM for selection.

**`content:encoded` fallback:** Substack RSS uses thin `<description>` text; full article content is in `<content:encoded>`. Parser now falls back to `content:encoded` when `description` is <50 chars.

**Cache key bumped:** `headlines:v1:` → `headlines:v2:`

---

## 31. World Skyline Series Replaces APOD (v3.10.0, 2026-02-24)

### Decision: Replace NASA APOD with AI-generated world skylines

**Problem:** NASA APOD photos produced heavy dot patterns after Floyd-Steinberg dithering to the 6-color Spectra palette. Even with saturation boost + contrast boost + posterization preprocessing, photographic images with smooth gradients and neutral tones created noisy dither artifacts on the E1002. The APOD pipeline also fetched external images of unpredictable size, which had caused Error 1102 (CPU limit) incidents.

**Solution:** Replace APOD with a "World Skyline Series" — AI-generated city skyline illustrations using SDXL, optimized for e-ink readability. AI-generated images can be prompted for flat color regions and poster-like aesthetics, producing much cleaner dither results than photographs.

**Three rotation modes (v3.10.0 update):**
- `rotate` (default): city+style change every `rotateMin` minutes (default 15) via bucket hash — matches SenseCraft 15-min refresh
- `daily`: one city+style per calendar day (seeded shuffle per year)
- `random`: crypto-random each request, no cache
- Bucket = `floor(Date.now()/60000 / rotateMin)` — deterministic, stable within each window
- City seed: `djb2("dateStr|city|bucket")` — independent from style
- Style seed: `djb2("dateStr|style|bucket")` — independent from city
- **June 1 override (>= 2025):** Chicago, USA (anniversary special) — but style still follows normal rotation

**City list:** 100 curated world cities in a static array.

**Style rotation (15 styles, mixed BW + color):**
- BW styles (6): woodcut, noir silhouette, linocut, comic ink, scratchboard, dark charcoal (pencil/etching/pen_ink removed in v3.11.0 — gray mush on 4-level)
- Color styles (9): travel poster, WPA poster, minimal flat, art deco, screenprint, ukiyo-e, synthwave, mediterranean
- BW output: 4-level grayscale PNG (same as Pipeline A)
- Color output: Spectra 6 indexed PNG via Floyd-Steinberg dithering (same as color moment pipeline)

**Prompt strategy:**
- Base: city skyline from classic overlook, strong horizon, distinct outlines, large sky negative space
- Lighting: sunset/daylight for color; clear daylight for BW — avoids noisy starfields
- Atmosphere: ONE sun/moon disc, 2-4 bird silhouettes, 1-3 simple clouds
- Anti-detail: avoid tiny windows, dense micro-detail, noisy textures, photographic realism
- Color palette hint for color styles: "limited palette, large flat color regions, poster-like"

**Caption format:** `{City} | World Skyline Series | {Mon DD, YYYY}`

**Why SDXL (not FLUX.2) for skylines:**
- SDXL produces more stable architectural silhouettes with SDXL's style control
- Skylines don't need the fine detail where FLUX.2 excels
- Simpler JSON API (no multipart FormData)

**HTML wrappers use `<img src>` (not inline base64):**
- `/skyline` serves HTML with `<img src="/skyline.png?...">` — SenseCraft re-fetches fresh each screenshot
- HTML wrappers are always `Cache-Control: no-store` — the .png endpoint handles its own caching
- `/skyline-test` forwards ALL query params to `/skyline-test.png`
- This avoids the "stuck city" bug where inline base64 never changed

**Endpoints:**
- `/skyline` — HTML page (SenseCraft screenshot target, replaces `/color/apod` in E1002 page rotation)
- `/skyline.png` — raw PNG (supports `?mode=rotate|daily|random&rotateMin=N`)
- `/skyline-test` + `/skyline-test.png` — test with `?date=YYYY-MM-DD&city=...&style=...&color=0|1&mode=...`
- `/color/apod` — 301 redirect to `/skyline` (legacy compatibility)

**Cache key (v2):**
- Rotate: `skyline:v2:YYYY-MM-DD:rN:bBUCKET` (TTL = rotateMin * 60, min 900s)
- Daily: `skyline:v2:YYYY-MM-DD:daily` (TTL = 24h)
- Random: no KV cache, no-store

**Cron:** Daily warm generates and caches the current rotation bucket alongside other daily images.

**What was removed:**
- `handleColorAPODPage` import and direct route handler
- APOD cron warm (`getAPODData` + `getAPODColorImage`)
- `APOD_API_KEY` from health-detailed config check
- APOD entries from health-detailed daily_images
- **v3.10.1 cleanup:** Deleted `src/apod.ts` and `src/pages/color-apod.ts`. Removed `APODData` type and `APOD_API_KEY` from `Env`. Removed APOD_API_KEY comment from `wrangler.toml`. The `/color/apod` → `/skyline` 301 redirect is kept for device compatibility.

---

## 32. SteelOrbis Date Parsing Fix (2026-02-25)

### Decision: Fix date scraping regex + append year to yearless dates

**Problem:** Headlines page showed "Invalid Date" next to SteelOrbis articles. Two stacked issues:

1. **Wrong div matched:** The date regex `/<div[^>]*>([\s\S]*?)<\/div>/` matched the outer wrapper div (which contains nested HTML) instead of the inner `article-date-body` div. Captured text included HTML tags, not the date string.
2. **No year in date:** SteelOrbis dates are `"25 Feb"` (day + month, no year). `new Date("25 Feb")` returns `Invalid Date` without throwing, so the `catch` block in `formatTimestamp` never fired.

**Fixes:**
- Narrowed scraper regex to `/<div[^>]*article-date[^>]*>([\s\S]*?)<\/div>/` — targets the date div specifically
- Append current year when date string lacks a 4-digit year: `"25 Feb"` → `"25 Feb 2026"`
- Added `isNaN(d.getTime())` guard in `formatTimestamp` as defense-in-depth

**Cache key bumped:** `headlines:v2:` → `headlines:v3:` to flush stale entries with empty dates.

---

## 33. BW-Only Skyline for E1001 Mono Display (v3.11.0, 2026-02-25)

### Decision: `bw=1` query param on `/skyline.png` + `/skyline-bw` HTML wrapper

**Problem:** The World Skyline Series (`/skyline`) rotates across all 18 styles (9 BW + 9 color). The E1002 color display handles both, but E1001 (mono) can only display BW styles. Adding E1001 to the skyline rotation required restricting to BW-only styles.

**Options considered:**
| Option | Approach | Tradeoff |
|--------|----------|----------|
| A. Separate `/skyline-bw.png` endpoint | Full new handler | Code duplication, separate maintenance |
| B. Hardcode style list in a second handler | Brittle, diverges from source of truth | |
| **C. `bw=1` param + `colorModeFilter` in picker** | **Single handler, filter at style selection** | **Minimal code, clean separation** |

**Chose Option C** because:
- `pickSkylineStyle()` already knows about `colorMode` on each style
- Adding `colorModeFilter?: "bw" | "color"` to `SkylinePickerOpts` is minimal and backwards-compatible
- Cache key gets `:bw` suffix → fully independent cache namespace
- `/skyline-bw` is a thin HTML wrapper (like `/fact`) pointing to `/skyline.png?bw=1`
- Cron warms both BW and mixed skyline buckets independently

**BW style cull (same release):** Reduced BW styles from 9 to 6 for E1001 mono. Removed pencil_bw, etching_bw, pen_ink_bw — too many mid-tones that quantize to gray mush on 4-level grayscale. Reworked charcoal to "Dark Charcoal" with heavier blacks and minimal mid-tones. Kept: woodcut, noir silhouette, linocut, comic ink, scratchboard, dark charcoal.

---

## 34. Moon Phase Widget on Weather Pages (2026-03-11)

### Decision: Pure calculation, no API

**Problem:** Weather pages showed sunrise/sunset but no moon phase — a natural companion for an outdoor/weather dashboard.

**Options considered:**
| Option | Approach | Tradeoff |
|--------|----------|----------|
| External API (USNO, weatherapi) | Accurate to the minute | New dependency, cache needed, rate limits |
| **Pure calculation (synodic period)** | **~29.53 day cycle from known new moon ref** | **No API, no cache, sub-ms, accurate to ±1 day** |

**Chose pure calculation** because:
- Moon phases are deterministic — the synodic period is well-known (29.53059 days)
- 8-phase granularity (each phase spans ~3.7 days) means ±1 day error is invisible
- Zero external dependencies, zero cache, instant computation
- Reference: January 6, 2000 18:14 UTC (well-documented astronomical new moon)

**Implementation:**
- `src/moon.ts` — `getMoonPhase(date)` returns index (0-7), name, illumination %
- `moonSVG(index, litColor, shadowColor)` generates parametric SVG (circle + terminator arc)
- `moonPhaseHTML()` shared helper in `weather-ui.ts` — both pages call with their own colors
- Mono: white lit / black shadow; Color: yellow (Spectra 6) lit / black shadow
- Added to sunrise/sunset line in both `/weather` and `/color/weather`
- `?test-moon=N` (0-7) parameter for visual testing on both pages

**Why parametric SVG (not static icon maps):**
Moon icons are algorithmically generated (terminator arc varies continuously), unlike hand-crafted weather icons. A single `moonSVG()` function with color parameters is cleaner than duplicating 8 SVGs × 2 color schemes = 16 static entries.

---

## 35. Skyline Reference Photos + City Reduction (2026-03-20)

### Decision: FLUX.2 with R2 reference photos, curated 30-city list with landmark metadata

**Problem:** Skyline images generated by SDXL with text-only prompts were often too generic — many cities looked interchangeable because the model has no real visual reference for specific skylines. The 100-city list also diluted quality (many obscure cities have no distinctive skyline).

**Options considered:**
| Option | Approach | Tradeoff |
|--------|----------|----------|
| A. Better text prompts only | Add landmark names to prompts | Improves all cities, but SDXL still has limited world knowledge |
| B. Reference photos + SDXL (img2img) | Use SDXL's image-to-image mode | SDXL on Workers AI doesn't support img2img |
| **C. FLUX.2 + R2 reference photos + SDXL fallback** | **Reference photos anchor composition; FLUX.2 does artistic reinterpretation; SDXL for cities without photos** | **Best of both: recognizable landmarks + artistic style** |

**Chose Option C** because:
- FLUX.2 klein-9b already proven for reference-based generation (birthday portraits)
- Same R2 bucket (`eink-birthday-photos`) under `skylines/` prefix
- Graceful degradation: cities without photos still get SDXL with landmark-enriched prompts
- The birthday pipeline's `callFluxPortrait` pattern (multipart FormData + `input_image_0`) maps directly to skylines

**City list reduction (100 → 30):**
- Curated 30 most iconic, visually distinctive cities worldwide
- Each city now has structured data: `{ name, key, landmarks }`
- `landmarks` field injected into ALL prompts (both FLUX.2 ref and SDXL fallback)
- Cities with distinctive skylines/landmarks only — removed generic or obscure entries
- `findSkylineCity(query)` allows test endpoint to look up by name or key

**Reference photo strategy:**
- 2-3 photos per city stored in R2 (`skylines/{city_key}_{n}.jpg`)
- Mix of skyline panoramas and landmark close-ups (Unsplash, free license)
- Photos resized to 768px max dimension for efficient FLUX.2 input
- Photo selection is deterministic per bucket/date via `djb2` hash — ensures cache consistency
- `fetchSkylinePhotos()` reuses `getPhotoFromR2()` from birthday-image.ts

**Two prompts per generation:**
- `buildSkylineRefPrompt()` — for FLUX.2 with reference: "artistic reinterpretation of the scene in image 0..."
- `buildSkylinePrompt()` — for SDXL fallback: "iconic skyline of {city} featuring {landmarks}..."
- Both include landmark names, style prefix, and anti-text suffix

**Pipeline flow:**
1. Fetch reference photos from R2 for the city
2. If photos found → pick one (seeded), call FLUX.2 with `input_image_0`
3. Retry FLUX.2 once on failure
4. If no photos or FLUX.2 fails twice → fall back to SDXL with landmark-enriched prompt
5. Post-processing unchanged (BW: 4-level grayscale; Color: Floyd-Steinberg → Spectra 6)

**Debug headers added:** `X-Skyline-UsedRef: true/false`, `X-Skyline-CityKey: {key}`

**Cache key bumped:** `skyline:v2:` → `skyline:v3:` (new model + prompts = different output)

**140 reference photos uploaded** to R2 for all 30 cities (up to 5 per city). Scripts: `download-skyline-photos.sh` + `download-skyline-photos-extra.sh` (Unsplash), `upload-skyline-photos.sh` (R2).

---

## 36. Skyline Neuron Budget + Stale Fallback (2026-03-20)

### Decision: Drop FLUX.2 retry, add stale-cache fallback on generation failure

**Problem:** The FLUX.2 double-retry + SDXL fallback tripled AI calls per skyline cache miss (3 vs 1 previously with SDXL-only). Combined with a burst of manual test requests, this exhausted the Workers AI free tier 10K neuron daily quota. Both displays showed blank screens because the skyline HTML pages serve `<img src="/skyline.png">` which returned 503.

**Fix (two parts):**

1. **Drop FLUX.2 retry** — single attempt, then immediate SDXL fallback. Max 2 AI calls per miss instead of 3.

2. **Stale-cache fallback** — when generation fails, scan for any previously cached skyline before returning 503: previous 10 rotation buckets → yesterday's buckets → old v2 keys. Only 503 if no cache exists anywhere.

**Why stale is better than placeholder:** A stale skyline (different city/style) is far more visually appealing than a blank screen on e-ink. The user won't notice it's stale since skylines change every rotation anyway.

---

## 37. Neuron Budget Conservation — Daily Skyline + Sequential Cron (2026-03-22)

### Decision: Switch skyline to daily mode, serialize cron image tasks with early abort

**Problem:** Workers AI free tier (10,000 neurons/day) was exhausted every day, leaving ALL image pipelines broken (not just skyline). Root causes:

1. **Skyline 15-min rotation** — up to 192 on-demand image generations per day (96 color + 96 BW), each attempting FLUX.2 then SDXL fallback = up to 384 AI calls/day.
2. **Parallel cron execution** — all 5 image pipelines (fact4, fact1, color-moment, skyline, skyline-bw) ran simultaneously. If any failed with budget exhaustion, the others continued wasting neurons on doomed SDXL fallback calls.

**Fix (three parts):**

1. **Skyline daily mode** — `DEFAULT_MODE` changed from `"rotate"` to `"daily"`. One generation per day per variant (color + BW = 2 total) instead of ~192. Cache key: `skyline:v3:YYYY-MM-DD:daily[:bw]`. Rotate mode still accessible via `?mode=rotate` query param.

2. **Sequential cron with 4006 detection** — Image tasks now run in priority order: Pipeline A → Pipeline B → Color Moment → Skyline → Skyline BW. After each task, if the error contains "4006" (neuron budget exhausted), remaining tasks are skipped. This ensures core Moment Before images are generated before skyline.

3. **Stale fallback searches daily keys** — `findStaleSkylineCache` now checks today's and yesterday's daily keys before scanning rotation bucket keys.

**Neuron budget estimate after fix:**
- Cron: ~3 core images (FLUX.2 + SDXL) + 2 skyline (FLUX.2 + SDXL) = ~5 image gen calls
- On-demand: 0 (everything cached from cron)
- Total: well within 10,000 neurons/day

**Future:** Upgrading to Workers Paid ($5/mo) would remove the neuron cap entirely.

---

## 38. Neuron Budget Fix — Drop Retries + Headlines + BW SDXL-Only (v3.11.1, 2026-04-07)

### Decision: Eliminate all FLUX.2 retry loops, disable headlines, BW skyline uses SDXL-only

**Problem:** Despite Decision #37 (sequential cron + daily skyline), neurons were STILL exhausted. Investigation revealed:

1. **FLUX.2 retry loops were only partially removed** — Decision #37 dropped the retry in `skyline-image.ts` but missed three other files:
   - `image.ts` (Pipeline A): 2 FLUX.2 attempts
   - `pages/color-moment.ts` (Color Moment): 2 FLUX.2 attempts
   - `pages/color-moment.ts` (Color Birthday): 2 FLUX.2 attempts
   - `birthday-image.ts` (Birthday): 2 FLUX.2 attempts

2. **Worst case was 11 AI calls per cron** — 6 FLUX.2 (with retries) + 3 SDXL + 2 LLM. FLUX.2 is the most expensive model on Workers AI. Even best-case (no failures) was 4 FLUX.2 + 1 SDXL + 2 LLM.

3. **Headlines LLM call was wasting budget** — `getHeadlines()` used 1 LLM call per 6h period, and the displayed news was often stale. Disabled until a better approach is found.

**Fix (three parts):**

1. **Single FLUX.2 attempt everywhere** — All 4 retry loops replaced with single attempt + SDXL fallback. Saves up to 3 wasted FLUX.2 calls.

2. **Headlines disabled** — Removed from cron (saves 1 LLM call per 6h), `/color/headlines` 302 redirects to `/skyline` so E1002 pagelist doesn't break. Superseded in v3.11.2 by non-LLM deterministic headline ranking.

3. **Skyline BW uses SDXL-only** — `sdxlOnly` parameter added to `generateSkylineImage`. BW skyline skips FLUX.2 entirely since SDXL produces good BW artwork and saves 1 FLUX.2 call. On-demand `/skyline.png?bw=1` also uses SDXL-only.

**Neuron budget after fix:**
- Best case: 1 LLM + 3 FLUX.2 + 2 SDXL = 6 calls (was up to 11)
- No headlines LLM call
- Skyline BW is SDXL-only (cheapest image model)

**BW cross-fallback for color skyline:** When color skyline generation fails and no stale color cache exists, the handler now serves the BW skyline cache instead of returning 503 (blank screen). Fallback chain: color cache → generate → stale color → BW cache → 503. Response includes `X-Skyline-Fallback: bw-cross` header for diagnostics.

**Files changed:** `image.ts`, `pages/color-moment.ts`, `birthday-image.ts`, `skyline-image.ts`, `index.ts`, `package.json`

---

## 39. Reliability Hardening Six-Pack (v3.11.2, 2026-04-27)

### Decision: Make request paths consistent, cache-aware, and cheaper under failure

This release applies six targeted improvements without changing the visual design:

1. **Shared Moment on request paths.** `/fact.png` and `/fact1.png` now use `getOrGenerateMoment()` on cold cache misses, matching cron and color moment behavior. This prevents the mono 4-level and 1-bit pipelines from picking different events when cron misses or KV is cold.

2. **KV-backed generation locks.** Cached AI routes use short soft locks (`gen-lock:v1:{cacheKey}`) before generation. Cloudflare KV is not atomic, so this is best-effort, but it lets duplicate cold-cache requests wait briefly for the first request to populate the real cache instead of immediately starting another AI call.

3. **Explicit npm scripts.** Added `npm run typecheck`, `npm run test:utils`, and `npm run dry-run` so the required verification commands are encoded in the project instead of only in docs.

4. **Pure utility tests.** Added a tiny Node test suite for query validation, skyline date/picker behavior, moon override clamping, histogram thresholding, and cache-key construction. The test build compiles TypeScript into `/tmp/eink-dashboard-tests` to avoid generated files in the repo.

5. **AI budget pause marker.** Request paths now check `ai-budget:v1:block`. When Workers AI returns a neuron-budget error (`4006` or "neurons"), the Worker writes a 6-hour marker. Cached content still serves, but new AI generation returns 503/Retry-After or uses stale skyline fallback instead of cascading through fallback models and wasting more calls.

6. **Non-LLM headlines restored.** `/color/headlines` is live again, but no longer uses Workers AI. `headlines.ts` fetches Steel Industry News, SteelOrbis, and Google News RSS, deduplicates, ranks deterministically by source/topic/numeric signal, and displays compact source summaries. Cron warms `headlines:v3:{date}:{period}` every 6 hours.

### Rollback

Changes are isolated on branch `codex/reliability-hardening-six-pack`. Before merge, rollback is simply:

```bash
git checkout main
```

After merge, revert the v3.11.2 commit and verify:

```bash
git revert <v3.11.2-commit-sha>
npm run typecheck
npm run dry-run
```

### Notes

- Cache key versions for existing image artifacts did not change: output format and visual processing are unchanged. New helper functions centralize the existing key formats.
- `package.json`, `src/index.ts` `VERSION`, and package lock metadata are bumped to `3.11.2`.
- `MEMORY.md` references in older docs refer to Claude Code's auto-memory workflow; this checkout may not contain a tracked `MEMORY.md`.

---

## 40. Yellow Is Not a Foreground Color on White (v3.11.3, 2026-05-18)

### Decision: Drop yellow from all foreground uses on /color/weather

**Problem:** On the E1002 Spectra 6 display, `/color/weather` rendered temperature text, the battery fill, and the moon's lit surface in yellow. Against the white page background, yellow has almost no contrast — yellow text and thin yellow strokes are unreadable on the e-ink panel.

**Rule:** Of the Spectra 6 palette (black, white, red, yellow, green, blue), only **black, red, green, and blue** are legible as foreground (text / thin strokes) on a white background. Yellow is usable only as a *background* fill (e.g. the non-severe alert banner, the headlines `regulatory` badge) or inside large dithered image regions — never as foreground on white.

**Changes:**

1. **Temperature colors: 4-tier → 3-tier.** `tempColor()` now returns blue for ≤ 0°C, black (`#000`) for > 0°C up to ≤ 30°C, red for > 30°C. Only the extremes get a warning color; the comfortable middle is plain black (the most readable foreground). The old green and yellow tiers are gone.

2. **Battery fill: 3-tier → 2-tier.** `batteryIcon()` fills red for ≤ 20%, green above. The yellow mid-tier is removed — no Spectra color works as a sensible mid value on white.

3. **Moon lit surface: yellow → white.** The `/color/weather` moon now uses a white lit surface with a black outline and black shadow, matching the mono E1001 treatment. This supersedes the "Color: yellow lit" detail of #34.

**Not changed:** The alert banner (yellow *background*, black text) and the headlines `regulatory` badge (yellow *background*, black text) stay — yellow as a background is fine. The AI image pipelines are unaffected.

**Tests:** `tempColor()` and `batteryIcon()` were exported and given boundary tests in `tests/utils.test.js`.

---

## 41. Temperature Scale: Black Middle → Green Comfort Band (v3.11.4, 2026-05-18)

### Decision: Replace the neutral black mid-tier with a green comfort band

**Context:** #40 shipped a 3-tier `tempColor()` with a *plain black* middle tier (> 0°C up to ≤ 30°C). Black was chosen for maximum contrast. Once v3.11.3 was viewed on the real E1002 panel, the black middle read as flat and uninformative — it covered almost every real temperature, so the page carried no at-a-glance temperature signal.

**Change:** `tempColor()` is now a true **blue → green → red scale**:

| Range | Color | Meaning |
|---|---|---|
| `< 10°C` | blue (`--s6-blue`) | cold |
| `10–27°C` | green (`--s6-green`) | comfortable |
| `> 27°C` | red (`--s6-red`) | warm / hot |

No black tier remains. Green is a legible foreground on white (#40's rule lists black, red, green, blue as the four legible foreground colors), so this stays within the legibility constraint. The function now compares Celsius directly — the unused intermediate Fahrenheit conversion from #40 was removed.

**Tradeoff accepted:** the boundaries are coarse — 28°C already shows red, 9°C already shows blue. A finer scheme (black "chilly/warm" edge bands) was considered and rejected as over-engineered for a glanceable e-ink dashboard. Three colors, one clear story.

**Not changed:** The #40 rule (yellow is never a foreground color on white) still holds. Battery fill (2-tier red/green) and the white moon are unchanged. The boundary tests in `tests/utils.test.js` were updated to the new ranges.

---

## 42. Weather Resilience + NWS Fallback (v3.12.0, 2026-05-26)

### Incident: E1002 `/color/weather` blank ("Failed to load remote image")

The color weather panel went blank-gray with a faint text ghost; SenseCraft HMI reported "Failed to load remote image" and it persisted across forced refreshes, while the other three E1002 pages rendered fine.

### Root cause: a slow upstream made the page block past the renderer's timeout

The page itself was valid (it rendered pixel-perfect in Chromium at 800×480). The problem was **latency**:

- **Open-Meteo was down** — returning 502/504 for *every* query (even `current=temperature_2m`), and taking **8–10 seconds** to fail.
- `getWeatherForLocation` **awaited** that slow failing fetch (the default 10s `fetchWithTimeout`) before falling back to stale cache, so every `/color/weather` request took ~8–10s. Measured: `/color/headlines` 0.24s, `/skyline` 0.11s, `/color/weather` 8.2–10.1s.
- SenseCraft's screenshot renderer timed out waiting → "Failed to load remote image" → the device kept its last (blank) frame.

So two faults combined: Open-Meteo's outage (external) **and** our page blocking on it (ours). The second is what turns any provider outage into a blank display.

### Decision: stale-while-revalidate + NWS fallback

**Stale-while-revalidate.** `getWeatherForLocation(env, …, ctx?, opts?)` now:
- fresh cache (<15min soft TTL) → return it;
- **stale cache → return it immediately and revalidate in the background via `ctx.waitUntil`** (so the request stays sub-second);
- cold (no cache) → block on the refresh chain.

`ctx` (`ExecutionContext`) is threaded from the worker `fetch(request, env, ctx)` handler through the page handlers. Cron callers pass no `ctx` (they intentionally await the refresh). Measured after the fix: warm `/color/weather` ≈ 2ms.

**NWS fallback.** `refreshWeather` tries Open-Meteo (timeout shortened **10s → 4s** so the fallback is reached quickly), then falls back to **NWS (`api.weather.gov`, no API key)** — `src/weather-nws.ts`. NWS is a two-step API (`/points/{lat},{lon}` → `forecast` + `forecastHourly`, `units=si`); the points lookup is cached per zip (`nws-points:{zip}`, 7-day TTL). The result is mapped into the existing `WeatherResponse` shape (text `shortForecast` → our icon keys; day/night periods collapsed to daily high/low). NWS provides no 15-min precip or sunrise/sunset, so the "rain in 30 min" warning degrades to the hourly-probability path and the sun/moon line is omitted while on NWS. Cached `CachedValue.source` records which provider produced the data; surfaced in `/health-detailed`.

**Cache key NOT bumped.** The stored shape is unchanged (only an additive optional `source`), so `weather:{zip}:v2` is kept deliberately — the existing ~24h stale cache then serves instantly the moment this deploys, which matters because Open-Meteo was actively down. Bumping would have forced a cold start into a dead Open-Meteo. This is a conscious exception to the "bump on pipeline change" rule.

**`getWeather` dedup.** `getWeather` is now a thin wrapper over `getWeatherForLocation` so both weather pages (E1001 mono + E1002 color) share the resilience + fallback from one implementation. The fragile `zip === "60540"` alerts branch was removed (now always `fetchAlertsForLocation`), which orphaned and removed the old `fetchAlerts`.

**Cron semantic note:** cron calls have no `ctx`, so they `await` the refresh. They also now return early if the cache is <15min fresh — harmless since cron runs every 6h (always past the soft TTL).

**Testing:** `?test-provider=nws|fail` (local) forces the NWS fallback / no-data path. Verified at 800×480: NWS data renders full current/5-day/hourly with correct colors; alert + rain banners render over NWS data; warm cache serves in ~2ms.

**Why not switch providers entirely:** Open-Meteo is excellent and free normally; this was a transient outage. The resilience fix — not the provider — is what prevents a blank display, and it protects against *any* provider's downtime. NWS is the no-key fallback (US-only is fine; both locations are US) consistent with the project's no-key preference (#16).

---

## 43. Cold-Cache Budget on Weather Request Path (2026-05-28)

### Decision: bound the cold refresh chain at 5s when called from a request

**Context:** Around 2026-05-28 the user observed `/color/weather` blank once on E1002 ("same blank HTML error again") and it self-recovered. Investigation showed v3.12.0 (#42) is correctly wired — warm and stale paths return in ~2ms — but the **cold-cache path** was left unchanged. On a truly cold cache, `getWeatherForLocation` blocks sequentially through Open-Meteo (4s timeout) → NWS `/points` (6s) → NWS forecast + forecastHourly (6s parallel), so worst case is ~10–16s. That exceeds SenseCraft's renderer timeout (~8–10s observed in #42), so any single window where the office weather KV entry is missing (cron failure + 24h TTL expiry, brief KV blip, new namespace) reproduces the "Failed to load remote image" symptom.

We could not pin the May 28 occurrence to cold cache vs. a transient network/colo hiccup — it was a one-off with no persistent logs. The fix is defensive: even without a confirmed root cause, the cold path is a real worst-case latency that should be bounded.

### Implementation

- **New `src/with-budget.ts`** — generic `withBudget<T>(promise, ms): Promise<T | null>`. `Promise.race` against a `setTimeout`, with the timer cleared on settle. The wrapped promise is not cancelled — callers hand it to `ctx.waitUntil` so any pending KV write still completes.
- **`src/weather.ts` cold-cache branch:** when `ctx && !opts?.forceProvider`, race `doRefresh()` against `REFRESH_BUDGET_MS = 5000`. On budget exceeded, `ctx.waitUntil(refreshPromise.catch(() => {}))` keeps the refresh alive in the background so the next request hits a warm cache. The function falls through to the existing "no refresh + no cache → throw" path, the page handler's existing try/catch returns 503 with `Retry-After: 300`, and SenseCraft retries 15 min later.
- **No change to fresh, stale, or cron paths.** Cron passes no `ctx` and stays patient — its job is to warm the cache, it can afford the full provider chain.
- **No cache key bump.** Stored shape is unchanged.

### Why 5 seconds

- Open-Meteo's `fetchWithTimeout` is 4s; 5s gives it its full window plus 1s slack.
- NWS won't typically complete inside that budget on cold path (~12s by itself), so a cold-cache request during an Open-Meteo outage will 503. That is acceptable — cron is expected to keep cold-cache windows extremely rare, and a fast 503 is strictly better than an 8-16s block that produces the same blank-panel symptom.
- Well under the 8–10s SenseCraft renderer timeout observed in #42.

### What this does NOT address

- **A persistent provider outage on cold cache will still 503.** A long-term-backup KV key (weekly snapshot, 7-day TTL) would close this gap but adds storage + maintenance complexity. Out of scope.
- **Persistent observability.** We still have no log of when a cold or budget event fires in production. Option A from the investigation (response headers `X-Weather-Path: warm|stale|cold|fail`, `X-Weather-Source`, plus a small debug KV ring buffer) is worth doing next so the *next* blip is diagnosable from real evidence rather than reasoning.

### Tests

`tests/utils.test.js` covers `withBudget` directly: fast resolve passes through, slow resolve becomes `null`, rejection rethrows.

---

## 44. World Cup 2026 Adaptive Dashboard (v3.13.0, 2026-06-23)

### Decision: one adaptive page per display, phase-driven, with a resilient dual data source

A seasonal FIFA World Cup 2026 view for both displays — `/worldcup` (E1001 mono) and `/color/worldcup` (E1002 Spectra 6) — for the duration of the tournament (group stage now → final 2026-07-19). Rather than three separate pages (today's matches / standings / bracket), it is a **single adaptive page per display** that transitions itself by tournament phase, so a glance always shows the most relevant thing and the SenseCraft pagelist stays short.

### Phase model (derived from data, not the calendar)

`computePhase()` in `src/worldcup-ui.ts` inspects the full match list:

| Phase | Condition | Layout |
|-------|-----------|--------|
| **group** | any GROUP match not yet FINISHED | split: today + latest results, rotating group table |
| **r32** | groups done, furthest active round is R32 | split: today + results, Round-of-32 list |
| **knockout** | furthest active round ∈ {R16, QF, SF, Final} | converging R16→Final bracket tree |
| **champion** | FINAL finished | champion card (winner + final score) |

### Bracket legibility (the hard constraint)

A 32-team tree is not legible on 800×480 with the available font. So the **Round of 32 is rendered as a list** during its ~6-day window, and only the **R16→QF→SF→Final** tree (≤8 ties, 7 columns ≈ 114px each) is drawn as a converging bracket. The third-place playoff is a one-line footer. `buildBracket()` deliberately excludes R32 and THIRD from the tree.

### Data source: football-data.org primary + openfootball fallback

Reuses the weather resilience pattern (#42/#43): stale-while-revalidate over KV, `withBudget` cold-path bound (5s), degradation chain. `src/worldcup.ts` tries **football-data.org** (`/competitions/WC/matches` + `/standings`, `X-Auth-Token` secret `FOOTBALL_DATA_KEY`) then falls back to the static **openfootball/worldcup.json** (no key, ~daily updates). Both normalize to a single source-agnostic `WorldCupData`.

- **Why football-data primary:** its `/standings` endpoint computes the group tables server-side. Computing 2026 standings + tiebreakers + the 8-best-third-place ranking ourselves would be the most bug-prone part of the feature.
- **Why openfootball fallback:** zero-key resilience; it derives simple standings (points/GD) from finished matches when it is the only source available.
- Cache key `wc:data:v1` (single blob serves both displays — same tournament data). Soft TTL 12 min (drives SWR), KV `expirationTtl` 86400 (≫ soft TTL, per #24). Warmed in the every-6h cron block.

### Third-place qualification ambiguity

2026 advances the 8 best third-placed teams to the Round of 32 — a cross-group, mid-stage-unstable ranking. We therefore mark only the **top 2** of each group as qualifying (`✓`); third place is **never** auto-marked, to avoid implying a wrong outcome.

### Display styling

Same data/layout on both displays via a `WcTheme` object passed to the shared render builders in `worldcup-ui.ts`. Mono = pure black + glyph markers (`▶` favorite, `✓` qualified). Color = Spectra-6 accents: green qualified/champion, red live, blue favorite; **yellow never as foreground** (#40). Favorite team is the `FAVORITE_TEAM` constant (`"BRA"`; `""` disables).

### Testing

`?test-phase=group|r32|knockout|champion` injects canned fixtures (`src/worldcup-testdata.ts`) to preview every layout at 800×480 before the real data reaches that phase. Cheap (no upstream) → no `TEST_AUTH_KEY` required, like `?test-device`. 17 unit tests cover phase detection, team-code/match-cell formatting, group rotation, bracket building, Chicago time conversion, and both normalizers.

### Seasonal / removable

~4-week useful life. Add `/worldcup` + `/color/worldcup` to the device pagelists for the tournament; remove after 2026-07-19 (the champion card is safe to leave up). No cache-key entanglement with other pipelines.

---

## 45. World Cup Full Team Names + Color-Display Flags (v3.14.0, 2026-06-24)

**Decision:** Replace 3-letter team codes with full country names on both displays, and add country flags on the color display only.

**Names (both displays).** `teamLabel(team, maxChars)` returns the escaped display name, truncated with `…` past a per-layout budget (rows 14, group table 12, bracket 12, champion 20). The fallback when a name doesn't fit is **truncation, not the 3-letter code** (user preference); truncation is word-aware (trims a trailing space/hyphen so we get `Bosnia…`, not `Bosnia-…`). `displayName()` first applies a curated `NAME_OVERRIDES` map (keyed by FIFA code) for names that overflow the panels: `S. Korea`, `S. Africa`, `Bosnia` (from Bosnia-Herzegovina), `Cape Verde`, `USA`. Match rows are `flex-wrap:nowrap` + `overflow:hidden` so two long names (e.g. South Africa v South Korea) never wrap to a second line and push the standings table. `teamCode()` stays, but only for favorite-team matching (`isFav`) and flag lookup — not display.

**Qualified ✓ means *mathematically* qualified.** `qualifiedFlags(rows)` marks a group-table team with ✓ only when it is **guaranteed** a top-2 finish, not merely currently in the top 2. Conservative worst-case test per team: it gets nothing more from its remaining games while every rival wins all of theirs; ✓ only if at most one rival can still reach-or-exceed its current points. Once the group is complete, ✓ = final position ≤ 2. Never false-positives (a ✓ is real); may under-mark in rare tie/cross-fixture cases, which is the safe side. Third-place best-of qualification is never marked (cross-group / unstable, per #44). Replaces the old `qualifying = position <= 2` flag, which lit up prematurely mid-group.

**Flags: color display only.** Spectra-6 (black/white/red/yellow/green/blue) is essentially "flag colors", so flat flags map onto it cleanly. The mono display gets **no flags** — with pure black only (no grays, DECISIONS #1/#40), red/blue/green fields all collapse to the same indistinguishable shape. Mono is full names alone.

**Why pre-render offline (Option A), not inline SVG.** flag-icons SVGs use each flag's *official* hex colors and fine emblem/text detail — both fight a 6-color, ~17px, device-quantized display. This codebase's rule for HTML pages is *only ever emit exact Spectra-6 colors* (the device quantizes whatever we send; see #40/#41), so arbitrary flag colors would quantize unpredictably. Instead `scripts/generate-flags.mjs` runs **offline** (the Workers runtime can't rasterize SVG): flag-icons 4×3 SVG → `@resvg/resvg-js` raster at 36×27 → the project's existing `ditherFloydSteinberg` + `encodePNGIndexed` (same path as the color image pipelines) → base64 indexed PNG. Output is the committed generated map `src/worldcup-flags.ts` (FIFA TLA → `"WxH|base64"`). ~48 flags ≈ **16 KB** total in the bundle; `flag-icons` + `@resvg/resvg-js` are **devDependencies only** (zero runtime cost). Indexed PNGs (~300 B each) are far smaller and crisper than inlining raw SVG.

**Rendering.** `WcTheme` gains an optional `flag?(code): string` hook — the color page (`COLOR_THEME`) supplies an `<img class="wc-flag">` data-URI from `FLAGS`; mono (`MONO_THEME`) omits it. Keeps `worldcup-ui.ts` display-agnostic, like the existing `fav`/`win`/`live` accents. A hairline black border on `.wc-flag` separates white-dominant flags (Japan) from the white page. Champion card uses a 110px `.wc-flag-big` hero. Missing flag → `""`; empty team → `"TBD"`; the page never breaks on a missing asset.

**No KV cache-key change** — flags are static bundle assets, the `wc:data:v1` blob is untouched. To refresh/extend flags: edit the FIFA→ISO map in `scripts/generate-flags.mjs`, run `npm run flags`, commit the regenerated `src/worldcup-flags.ts`. 5 new unit tests (teamLabel ×4 + FLAGS coverage), 37 total.

**Post-ship polish (2026-06-24).** Two fixes after on-device testing:

1. **Smudged text on the color panel → grays.** The WC CSS used `#ccc` row underlines and `#555` rank/empty text. The Spectra-6 panel has no gray, so it *dithers* gray into a black/white speckle — which around glyphs reads as "smudged." The crisp color pages (weather, moment) use **zero** non-`#000`/`#fff`/palette colors. Fix: removed every gray (row borders dropped, `#555`→`#000`). This is just the project's own pure-black e-ink rule (#1/#40), which the v3.13.0 WC code had violated. Also bumped font sizes/weights — thicker strokes quantize more cleanly than thin ones. (The Moment card looks crisp partly because it is a pre-dithered *image* with only a thin white-on-black caption; HTML text on a white field is dithered by the panel itself, so it must stay pure black and reasonably large.)
2. **Full vertical fill.** The split layout left whitespace below the standings. Now `.wc-split` and `.wc-bottom` are flex children of the 480px body; match rows live in a `.wc-rowlist` with `justify-content: space-evenly` (spreads to fill, adapts to match count), and `.wc-table { height: 100% }` so the 4 standings rows expand to fill the bottom panel. Champion flag 110→150px. Uses the whole display like the Moment cards. No version bump (presentation tuning of v3.14.0).
3. **All text pure black on the color display, too (smudge round 2).** Removing grays wasn't enough — the *colored* accents still smudged. Colored glyphs (blue favorite name, green ✓/champion, red live) anti-alias to in-between colors the Spectra-6 panel has no anchor for, so it dithers their edges into speckle — and dark blue text on white is the worst offender. Fix: `COLOR_THEME.fav/win/live` all set to `#000`, so the color page renders type exactly like mono. **Color identity now comes only from the flags** (pre-dithered images, crisp by construction), and accents are carried by weight + the ▶/✓ markers. Row/table/bracket weights bumped 600→700. General rule confirmed: on a device-quantized HTML page, *all* text should be `#000` on `#fff`; use color only for solid large fills or pre-dithered images, never small/medium type.

---

## 46. World Cup Text Crispness — Match the Weather Pages (v3.14.1, 2026-06-25)

### Symptom

On-device, WC text was visibly **smudged on BOTH panels** (mono E1001 and color E1002), while the `/weather` pages were crisp on the same devices. (A separate, earlier confusion: the E1001 was briefly pointed at `/color/worldcup` — the color page's flag images + fills render as black mush on the mono panel. Correct routes: E1001 → `/worldcup`, E1002 → `/color/worldcup`. The smudge below is the real, route-correct issue.)

### Root cause (two compounding factors, both measured)

Compared the shared WC stylesheet (`worldcup-ui.ts`, used by both displays) against the proven-crisp weather pages (`weather2.ts` + `color-weather.ts`, which are typographically identical to each other):

1. **Over-bold weights.** Decision #45 bumped WC weights to `700`/`800` on the theory "thicker strokes quantize more cleanly." **Wrong.** At 15–22px, `800` strokes thicken until letter counters (a/e/o holes) and inter-letter gaps fill with anti-alias gray; the panel quantizes that gray to black → letters blob. The weather pages cap at **700** and use **500** for body text — and they're crisp. So heavier ≠ crisper past ~700; it's worse.

2. **Stretch-to-fill → fractional pixel positions.** #45 also added `.wc-table{height:100%}` and `.wc-rowlist{justify-content:space-evenly}` (and `.wc-bcol{space-around}`) to "fill the 480px like the Moment cards." Measured result: table rows at non-integer heights (41.58px) and every text baseline on a *different* sub-pixel phase (tops .66/.23/.81/.39). Sub-pixel baselines force the renderer to anti-alias each glyph *vertically* across two pixel rows → the panel quantizes that to speckle → vertical smudge. The weather pages use fixed-padding cards → integer positions → sharp. (The Moment card is exempt because it's a pre-dithered *image*, not live HTML text.)

Both factors hit mono (1-bit threshold) and color (Floyd-Steinberg dither) alike, because both quantize the same anti-alias gray.

### Fix: adopt the weather pages' typographic discipline (in the shared stylesheet → fixes both displays at once)

- **Weight scale, three tiers, never above 700:** emphasis (titles, labels, scores, points, favorite/winner) `700`; base team/data text `600`; secondary (subtitle, table `th`, position number, "v", third-place) `500`. Favorite/winner emphasis — which on the all-`#000` color page was carried by the `700`↔`800` weight gap — is preserved by the new `600`↔`700` contrast (both crisp). The three inline `font-weight:800` accents in `matchRow`/`groupTable`/`bracketColumn` dropped to `700`.
- **Kill fractional positioning:** removed `.wc-table{height:100%}` (table is content-sized; rows are integer 38px) and replaced `space-evenly`/`space-around` with `justify-content:flex-start` + fixed `gap` (rowlist 14px, bracket column 12px). Added integer `line-height` to dense rows/cells (`.wc-row` 26px, `td` 26px, `.wc-bteam` 24px).
- **Specificity note:** `.wc-pts`/`.wc-pos` (0,1,0) lose to `.wc-table td` (0,1,1), so they were silently inert (this is also why #45's `.wc-pts{800}` never actually rendered). Rewrote as `.wc-table td.wc-pts` / `.wc-table td.wc-pos` so the emphasis/secondary weights take effect.

### Tradeoff accepted

Content now packs from the top with fixed spacing instead of stretching to fill all 480px, so there can be modest bottom whitespace. **Crispness over full-bleed fill** — the same tradeoff the weather pages already make. Verified: all phases fit within 800×480; rowlist tops now integer (84/124/164), table row heights integer (38px), zero `800`-weight elements; typecheck + 41 util + 26 WC tests pass.

### Rule (supersedes #45's weight guidance)

On a device-quantized HTML page, keep text `#000` on `#fff` (from #45) **and**: weights `≤700` (body `500–600`, emphasis `700`; `800` smudges at small sizes), and never stretch text containers to a percentage height — size rows to content with fixed spacing so baselines land on integer pixels.

### Follow-up (v3.14.2, 2026-06-25): body weight 600 → 700

On-device side-by-side (same E1001 panel) showed the smudge gone but the WC body text reading **lighter/softer than the weather page**. Cause: the weather pages set their *primary* data to **700**, but #46 dropped WC body to **600** to give favorite/winner a `600`↔`700` weight contrast. At `600` the strokes lay down less ink, so on e-ink they read grayer/softer than weather's solid `700`. Fix: WC body data (`.wc-row`, `.wc-table td`, team-name cell, `.wc-bteam`) bumped **600 → 700** to match the weather pages exactly. Favorite/winner no longer differ by weight — emphasis rides on the existing ▶ (favorite) and ✓ (qualified) glyphs. Net scheme is now identical to the weather pages: **primary 700 / secondary 500, never 800.** `700` is the proven-crisp weight on this panel (the whole weather page uses it); the smudge was specifically `800`, not `≥700`.

### Follow-up (v3.14.6, 2026-06-25): the real "not crisp" cause was flexbox crushing the rows

After the weight/positioning fixes the user still reported "not crisp" text and **clipped descenders** (the bottoms of g/y/p/ç cut off) in the upper-half match rows. Measuring the live page exposed the actual cause: each `.wc-row` had `line-height: 26px` but a measured **height of only 14px** (`clientHeight 14`, `scrollHeight 20`). With 6 matches today, the six rows didn't fit `.wc-rowlist` (a `flex:1` column), so the default `flex-shrink: 1` **compressed every row to 14px**, and `overflow: hidden` then chopped the text top and bottom. That mangling — not the font weight — is what read as "not crisp." The weather page looks crisp because its rows are in fixed-padding cards that never compress.

Fix (all in the shared `worldcup-ui.ts`):
- `.wc-row { flex-shrink: 0; overflow: visible; line-height: 25px }` — rows keep their full line box; flexbox can no longer crush or clip them.
- `.wc-bottom { flex: 0 0 auto }` (was `flex: 1.15`) — the standings panel sizes to its content instead of grabbing a fixed share, so the upper half (`.wc-split { flex: 1 }`) gets all the leftover height it needs for six full-height rows. This also reclaims the wasted whitespace the user spotted under the table.
- Compacted the table to make the budget close cleanly: `td` padding `6px → 3px 6px` and line-height `26 → 24`, `th` padding `2px → 1px`, `.wc-group-name` margin `6 → 3`, and `.wc-rowlist` gap `14 → 6`.

Verified at 800×480 with the live 6-match day: all six rows full-height (25px, `scrollHeight == clientHeight`, no clipping), every descender intact, the 4-row table fully visible and filling the lower half, total within 480px. **General rule: never let text rows live in a `flex:1`/shrinkable container with `overflow:hidden` — they silently compress and clip. Give text rows `flex-shrink:0` and sized siblings `flex:0 0 auto`.**

---

## 47. World Cup Group Qualification ✓ Must Be Fixture-Aware (v3.14.4, 2026-06-25)

### Symptom

On the real Group D table, **USA had no ✓** even though it was already mathematically guaranteed to advance before its last game. Standings (after 2 of 3 games): USA 6, AUS 3, PAR 3, TUR 0; last matchday: **TUR v USA** and **PAR v AUS**.

### Root cause: the qualification check ignored the remaining fixtures

The old `qualifiedFlags(rows)` looked only at points/games-played. For each team it counted how many *other* teams could still reach its points if they **won all their remaining games independently**. For USA that's 2 threats (AUS 3+3=6, PAR 3+3=6) → "not safe" → no ✓.

But AUS and PAR **play each other** on the last day — only one of them can win, so at most one reaches 6. USA is therefore guaranteed at least 2nd. The points-only heuristic can't see this because it never looks at *who plays whom*. (Symmetrically, it was also too generous in other shapes — counting independent threats that can't all happen.)

Separately, the data layer's `WcStandingRow.qualifying` was just naive `position <= 2` ("currently top 2"), which the UI ignored in favor of the heuristic. Neither was correct.

### Fix: brute-force the remaining group matches

`qualifiedFlags(rows, remaining)` now takes the group's non-finished matches (as `{home, away}` FIFA codes) and enumerates **every** win/draw/loss combination (group of 4 → at most 3⁶ = 729 scenarios). A team is marked ✓ only if, in **every** scenario, at most one other team has points `≥` its points (the `≥` keeps it conservative on tiebreakers — a points tie never counts as clinched, since GD/head-to-head could rank the rival above). When the group is complete (`remaining = []`) it trusts the source's final `position <= 2`.

The flags are computed in `finalize()` (the one place the full match list `_allMatches` is still available, before it's stripped) and stored on each `WcStandingRow.qualifying`; the UI now renders the ✓ straight from `row.qualifying` instead of recomputing. Group D now yields **USA ✓, AUS/PAR/TUR ✗** — the correct answer.

### Scope / non-goals

- **Top-2 only.** Best-third qualification (the 8 best 3rd-placed teams advance in the 48-team format) is still **never** auto-marked — it depends on cross-group comparisons that generally aren't decidable until late, and a wrong ✓ is worse than a missing one. A team guaranteed to advance *only* as a best third shows no ✓ by design.
- Cache key bumped `wc:data:v1 → v2` so the corrected flags recompute immediately on deploy rather than serving the previously-cached naive flags through the SWR window.
- 4 unit tests in `tests/worldcup.test.js` lock the behavior (the USA "chasers play each other" case, the "chasers play different opponents → leader not safe" case, the all-tied case, and the completed-group case).

### Follow-up (v3.14.5, 2026-06-25): resolve fixtures by NAME, not 3-letter code

The first cut keyed remaining fixtures to standings rows by `teamCode` — and **football-data is internally inconsistent**: Curaçao is `CUW` in the `/standings` feed but `CUR` in the `/matches` feed (same full name "Curaçao"). So the fixture's team didn't resolve, `idx.get()` returned undefined, and that match was **silently skipped** (`continue`) — its points never moved in the simulation, under-counting threats. Real-world fallout: Spain (Group H, 4 pts, 1 game left, chasers on 2/2/1) was wrongly marked ✓ — it is *not* guaranteed, because two chasers can each reach ≥4. (USA/Group D happened to be correct anyway since it's clinched by gap, which masked the bug.)

Fix: `qualifiedFlags` now takes the remaining matches as **team objects** and resolves each to a standings row by **normalized full name first, then code**. Names are stable across both feeds; the code is not. `finalize` passes `m.home`/`m.away` directly. This is the general rule for joining football-data's two feeds: **match on name, treat the 3-letter code as a hint.** A 5th unit test pins the CUR/CUW case.

---

## 48. E-ink Text Crispness Is Limited by SenseCraft's Cloud Render, Not CSS Tricks (v3.14.7, 2026-06-25)

### Problem

After the layout/clipping fixes (#46), WC text on the E1001 still looked **smudged/serrated** versus the `/weather` page, which is crisp on the **same** device — despite both using the same font stack, pure `#000`/`#fff`, and integer pixel positions. A three-agent research sweep (web + code-diff + device-pipeline) settled it.

### How the pixels actually get to the panel (the key reframing)

**SenseCraft HMI's "Web Function" renders the URL in a headless browser in Seeed's CLOUD**, rasterizes + dithers it to the panel palette server-side, and pushes a finished image to the ESP32 (the device is just a receiver). Confirmed by: device can't reach LAN/HTTP URLs (only public HTTPS), JS executes, full DOM/CSS support, and the panel's 2–3 s refresh is waveform time for a pre-made image. Consequences:
- **We control only the HTML/CSS** (no browser flags, no deviceScaleFactor, no waveform mode).
- Every **anti-aliased (grayscale) glyph edge** is the enemy: the cloud quantizes grayscale to the 1-bit mono panel, turning soft edges into serration/smudge.

### What does NOT work (verified, high confidence)

- `-webkit-font-smoothing: none`, `font-smooth: never`, `text-rendering: optimizeSpeed` — **no-ops in Blink/Chromium** (only macOS Safari honors `-webkit-font-smoothing`, and never the `none` value; `text-rendering:optimizeSpeed` only drops kerning/ligatures, it does **not** disable AA; MDN + Chromium bug 40635769). They were added in an earlier cut and **did nothing** — the WC page carried them and was still the *worse*-looking page, which is the proof. Kept temporarily as a no-cost trial; **treat as inert / removable.**
- `image-rendering` — only affects raster `<img>`/canvas scaling, never font rasterization.
- Chromium launch flags (`--font-render-hinting`, `--disable-lcd-text`) — launcher-only; a page can't set them and we don't own Seeed's launcher.
- **Refresh mode** — SenseCraft exposes only an update *interval*, never the waveform mode, and it's not per-page; every push is a full refresh. So it cannot make one page crisp and a sibling smudged. **Ruled out.**
- **Document downscaling** — both pages lay out at exactly 800×480 with no horizontal overflow (verified `scrollWidth==800`), so no fractional fit-to-width scaling. **Ruled out** as the differentiator here (but it *would* serrate everything if a page exceeded 800×480 — keep pages exactly 800×480).

### Actual cause + fix

The crispness gap is **typographic**: the WC page rendered the **bulk** of its content as **bold `700` at 19–22px**, while the crisp weather page renders most body text as **medium `500` at 14–18px** (reserving `700` for a few small accents and the big isolated numerals). Heavier/larger glyphs expose far more anti-aliased edge area, and the cloud's grayscale→1-bit quantization turns that fringe into the "serrated/smudged" look. (This *reverses* the v3.14.2 call to bump everything to 700 "to match weather" — weather's *body* is 500, not 700.)

Fix (v3.14.7): WC body content (`.wc-row`, `.wc-table td`, team-name cell, `.wc-bteam`) dropped **700 → 500**, matching the weather page's body profile; `700` kept only as accents (titles, panel/group labels, the score/time `.wc-cell`, points, favorite/winner via the existing inline overrides — which now also restores the favorite's weight emphasis over a 500 base). Secondary suspects, to revisit on-device if 500 isn't enough: the standings **CSS `<table>`** and the R32 **`columns:2`** compute content-derived column widths that can land glyphs on sub-pixel x-positions (the weather page is pure flexbox with none of this) — replacing them with fixed flex columns would remove that residual serration. **On-device A/B is mandatory** — none of this is visible in a local browser (the dither only happens in Seeed's cloud render).

### Rule

On a SenseCraft/e-ink HTML page, crispness comes from **medium weight (500) + moderate size + pure `#000`/`#fff` + exact 800×480 + integer positions + flexbox (avoid CSS tables/multicol for text)** — not from font-smoothing CSS (inert) or refresh settings (not ours). When in doubt, mirror the `/weather` page's typography, which is the proven-crisp reference on this exact device.

### Follow-up (v3.14.8): local 1-bit simulation + on-device A/B variants

A faithful local simulation (Playwright render of each page → downsample to 800×480 → 1-bit threshold + 4-level Floyd–Steinberg via the project's own `decodePNG`/`encodePNGGray8`) showed **both** `/weather` and `/worldcup` render **crisp** — identical quality. So the fog is **not** in the page's CSS/text logic; it's introduced downstream in Seeed's cloud renderer (most likely a poorly-hinted *fallback* font at our larger sizes, since neither page names a font that exists on the cloud's Linux Chromium) or the physical panel — neither reproducible locally.

To settle it on the real device without N deploys, `/worldcup?variant=` switches typography in one build (corner label shows which): `atkinson` (inline Atkinson Hyperlegible 400/700 WOFF2 — legibility/e-ink font), `inter` (inline Inter 500/700 — neutral, well-hinted), `small` (weather-sized 14–17px, no embed), default = current. Fonts are base64 data-URIs in `src/worldcup-fonts.ts` so the cloud renderer cannot fall back. Option 4 (server-render the page to a pre-dithered PNG, bypassing the cloud renderer — guaranteed crisp but a full rewrite) is held in reserve. Pick the winner on-device, then make it the default and drop the rest.

### Resolution (v3.15.0): `/worldcup` default is a server-pre-dithered image via Browser Rendering

On-device A/B (variants `atkinson`/`inter`/`small`/`image`/default) verdict: the **`image`** variant — the page rendered to a pure 1-bit PNG by us — was clearly crispest, because SenseCraft's cloud render then has no gray edges left to quantize. `inter` (live HTML) was a close second; `atkinson`/`small` were worse (smaller text is *not* crisper here, consistent with the local 1-bit sim where all HTML looked crisp — the fog is the cloud's, not ours).

A Worker can't rasterize HTML itself, but **Cloudflare Browser Rendering** (the `[browser]` binding + `@cloudflare/puppeteer` + `nodejs_compat`) gives it headless Chromium. Pipeline (`worldcup-browser-image.ts`): screenshot the Inter-styled HTML (`/worldcup?variant=src`) at 2× → supersample to 800×480 → threshold to pure 1-bit (`?thr=`, default 160). This keeps the **real proportional Inter typeface and full country names** (unlike the rejected 8×8-bitmap attempt, which truncated names and looked like a pixel-font "joke") while being crisp by construction. It also covers **every phase for free** — it just screenshots whatever the adaptive HTML renders.

Serving (`pages/worldcup.ts`): default `/worldcup` returns the cached image with **stale-while-revalidate** (soft TTL 14 min, refresh via `ctx.waitUntil`) + a **cron warm** (6h), so the device always gets an instant response despite the ~10 s browser launch; a cold cache falls back to the Inter HTML so the panel never blanks. `?variant=src` is the internal screenshot source (also a raw-HTML debug view); `?test-phase=`/`?test` serve HTML previews. Browser cost is ~100 renders/day (SWR + cron) ≈ a few minutes/day of browser time — within Workers Paid's allowance. The A/B scaffolding (Atkinson/small/bitmap variants, the bitmap `worldcup-image.ts`, the Atkinson font) was removed; only Inter remains.

### Architecture (v3.15.1): separate stylesheets per display

The displays were sharing one stylesheet inside `renderWorldCupHTML`, so the B&W crispness work bled into the color page (it lost its 700/800 weights and fill-480 layout). Fixed by **decoupling the CSS**: `WcTheme` now carries `styleCSS`, and `renderWorldCupHTML(data, theme)` emits `:root{theme.rootCSS} theme.styleCSS` with no hardcoded CSS. The two stylesheets live in `worldcup-styles.ts` as independent constants — `COLOR_STYLE` (the pristine pre-session E1002 CSS, restored byte-for-byte from commit ed9f595) and `MONO_STYLE_BASE` (the crisp B&W layout); the B&W page composes the latter with its inlined Inter font. Changing one display's CSS now cannot affect the other. **Structure and logic stay shared** (the section builders + `qualifiedFlags`/`computePhase`/`pickRotatingGroup` etc.) so the displays can't diverge on correctness — e.g. the fixture-aware ✓ fix (#47) applies to both. Net: the color page is exactly as it was pre-session (flags included), the B&W page is the crisp image.

### Regression + fix (v3.15.2): a stray `}` from the stylesheet extraction

The v3.15.1 split was generated by extracting each `<style>` block from git and stripping the `:root { ${theme.rootCSS} }` line — but the regex `\{[^}]*\}` stopped at the `}` *inside* `${theme.rootCSS}`, leaving a dangling `}` at the top of both `COLOR_STYLE` and `MONO_STYLE_BASE`. That stray brace broke the leading `* { box-sizing:border-box; margin:0; padding:0 }` reset on the device renderer, so `body{width:800px;padding:14px 22px}` overflowed to the right (content-box) and default element margins bloated the height — the B&W image showed a right-cut header, a clipped 4th table row, and a big mid-page gap. Fixed by regenerating `worldcup-styles.ts` with a precise *literal* strip (color from `ed9f595`, B&W from `53f6404`/v3.15.0); `MONO_STYLE_BASE` verified byte-identical to the working v3.15.0 CSS, and a 6-row Inter render confirmed it fits 800×480 with no clipping. Lesson: never regex-strip a CSS rule whose body contains a `${...}` interpolation (the nested `}` breaks `[^}]*`) — strip the literal line, and assert the result starts with the reset, not a `}`.

### Final state of the World Cup pages (v3.15.2)

- **`/worldcup` (E1001 B&W):** default response is a **server-pre-dithered 1-bit PNG** — Cloudflare Browser Rendering (`[browser]` binding + `@cloudflare/puppeteer` + `nodejs_compat`) screenshots the Inter-styled HTML source (`?variant=src`) at 2×, supersamples to 800×480, thresholds to pure black/white (`worldcup-browser-image.ts`). Cached `wc:image:v10` with stale-while-revalidate (14-min soft TTL) + 6h cron warm (`warmWorldCupImage`); cold cache → Inter HTML fallback so the panel never blanks. Inter is inlined (`worldcup-fonts.ts`).
- **`/color/worldcup` (E1002 color):** unchanged HTML render with Spectra-6 + flags (pristine pre-session).
- **Decoupled CSS:** `WcTheme.styleCSS` carries the full per-display stylesheet (`worldcup-styles.ts`: `COLOR_STYLE`, `MONO_STYLE_BASE`); `renderWorldCupHTML(data, theme)` no longer hardcodes CSS. Structure + logic stay shared.
- Removed: the 8×8-bitmap `worldcup-image.ts` and the Atkinson font (rejected A/B options).

---

## 49. World Cup Phase Must Track the *Current* Round, Not the Furthest *Existing* One (v3.15.3, 2026-06-28)

### Incident: on the first day of the Round of 32, both displays showed an all-TBD Round-of-16 bracket

The group stage had just ended and the first knockout match (Canada v S. Africa, match 73) had been played, yet `/worldcup` and `/color/worldcup` rendered the R16 → Final bracket tree with **every tie reading "TBD"** — instead of the Round-of-32 split layout with the real fixtures and the played result.

### Root cause: `computePhase` keyed on the furthest knockout round that *had a match*

football-data.org (and openfootball) publish the **full 104-match fixture list up front** — R16/QF/SF/FINAL all exist as `SCHEDULED` placeholders with TBD teams from day one. The old `computePhase` did:

```
furthest = max KO_ORDER index over all matches
return furthest <= 0 ? "r32" : "knockout"
```

So the instant the last group match finished, `furthest` was FINAL (the placeholder fixture exists) → phase `"knockout"` → `bracketLayout` rendered the empty R16 tree. The actual first knockout round is the **Round of 32**, which is shown by the `"r32"` split layout (`buildBracket` deliberately excludes R32 — a 32-team tree isn't legible at 800×480).

### Fix: current round = earliest knockout round with an unfinished match

```
for stage in [R32, R16, QF, SF, FINAL]:
  if any match in stage is not FINISHED: current = stage; break
return current == R32 ? "r32" : "knockout"   // none unfinished / none at all → "r32"
```

This is the round actually being played: while any R32 tie is still SCHEDULED/LIVE the phase is `r32`; only once all R32 ties are FINISHED does R16 become the earliest unfinished round and the bracket appears. Mere *existence* of a later-round placeholder no longer advances the phase. (`FINAL` finished is still caught first → `champion`.)

### Why this also fixes the "everything TBD" symptom

The TBD wall was a *consequence* of showing the wrong layout: the R16+ fixtures genuinely have no teams yet. Rendering the `r32` split instead surfaces the real R32 fixtures (teams known once groups finished) plus today's matches and the latest results — including the played Canada v S. Africa score.

### Cache bumps (mandatory — the computed phase is cached)

`finalize()` bakes `phase` into the cached `wc:data` blob, and the B&W `/worldcup` caches the *rendered* PNG. Both were bumped so the fix is immediate, not delayed by SWR: `wc:data:v2 → v3` (`WORLDCUP_CACHE_VERSION`) and `wc:image:v3 → v4` (`IMG_KEY` in `pages/worldcup.ts`).

### Tests

Two `computePhase` tests in `tests/worldcup.test.js`: the full-fixture-list scenario (all KO rounds scheduled, R32 active → `r32` — the regression), and the progression (R32 all FINISHED, R16 active → `knockout`).

### Follow-on: the R32 phase needed its own full-screen layout (v3.15.4)

Once the phase fix correctly selected `r32`, the page was still ugly because `r32` reused the group-stage `splitLayout` (TODAY + LATEST RESULTS split on top, R32 list below). With the *real* 16-match round that layout broke three ways, all confirmed by 800×480 screenshots of the deployed page:

1. **Redundant** — the one played tie (S. Africa v Canada) showed three times: in TODAY, in LATEST RESULTS, and again in the R32 list.
2. **Empty band** — the split section is `flex:1` but held one match each, leaving a large blank gap before the R32 list.
3. **Overflow/clip (color)** — the R32 list got only ~half the height; 16 rows with flags overflowed 480px, clipping the bottom row, and the narrow column clipped "Cape Verde" at the right edge.

The split layout had only ever been exercised with a 2-match R32 test fixture, which hid all of this.

**Fix (v3.15.4):** a dedicated `r32Layout` — initially the full 16-match round as a 2-column list/grid filling the screen, sorted by kickoff, no today/results split (redundant — the list already carried all of it).

**Redesigned as a bracket (v3.15.5):** the user wanted the classic knockout **bracket**, not a list (and flagged that a flat list with only times implied all 16 ties were on one day — they actually span Jun 28–Jul 3 CT). `r32Layout` now renders a 2-sided bracket: 8 R32 ties on each outer edge (`.wc-kr32`), converging through empty R16/QF/SF columns to a center Final box (`.wc-kcenter`). Each filled tie (`r32TieBox`): two full-width team lines (flags on color via the theme hook, names-only on mono — flags illegible in pure B/W), winner bold + per-team score on finished ties, and a compact "Jun 29 · 12:00 PM" line otherwise. Header subtitle shows the round's date span + "times CT" (kills the "all same day?" confusion). Ties split by match id (the source numbers them in bracket order). CSS in BOTH stylesheets (`.wc-k*`); `?test-phase=r32` fixture expanded to a realistic full bracket with real dates. Verified at 800×480 on both displays (local dev + prod). Bumped `wc:image` v4→…→v7.

**Fully data-driven, fills as it goes (v3.15.6):** the bracket renders EVERY round (R32→Final) from the live match list, not hardcoded placeholders. football-data publishes the R16/QF/SF/Final matches up front with empty teams + real dates; `bracketBox` shows just the round date (e.g. "Jul 4") as a roomy placeholder until a team is known, then renders the teams. The page re-reads the source every ~12 min (SWR) + 6h cron.

**Self-advancing winners + compact inner rounds (v3.15.7):** the source leaves a later-round match's teams empty until well after the feeders finish, so a just-won team wouldn't appear in its next box. We now **compute advancement ourselves**: `advanceRound(prev, source)` pairs the previous round in bracket order (`prev[2i]`/`prev[2i+1]` feed slot i) and seeds the winners, preferring the real source match once it carries teams + a score (so actual results/penalty outcomes win over our recompute). This is sound because the source numbers matches in bracket order — the id-sorted R32 column matches the published bracket exactly (verified against the reference graphic). `winnerTeam` decides by fullTime score (a level fullTime → no winner shown; a known penalty-only edge case). Inner rounds are **compact**: their boxes are narrow, so they show the **3-letter code on mono** and the **flag only on color** (full names / even codes+flags overflow); R32 keeps full names. So Canada appears in its R16 box (flag on color, "CAN" on mono) the moment it wins, opponent "TBD" until decided. Unit-tested (`winnerTeam`, `advanceRound`). Also fixed B&W faux-bold: the bracket had used Inter weights 600/800 that aren't inlined (only 500/700) → snapped to 500/700. **v3.15.8:** the advancing flag was too small on color → inner-round boxes (everything but the wide R32 columns) now use a big centered 24px flag and taller boxes, so an advanced team reads at a glance; the undecided side of a partly-decided tie is omitted (one prominent flag instead of a blank line).

**Used for the WHOLE knockout phase, not just R32 (v3.15.9):** previously once all R32 ties finished the phase became `"knockout"`, which fell back to an old, lopsided R16-only layout — the full bracket vanished. Now both the `r32` and `knockout` phases render this one self-advancing bracket (`knockoutBracket`; the old `bracketLayout`/`bracketColumn`/`buildBracket` were deleted). The header names the **current** round (earliest unfinished) + that round's dates, so it reads "Round of 16 · Jul 4–5" etc. as the tournament progresses. Two fixes found by testing the later phases at 800×480: compact (inner) boxes put the score on the date-line (`0-2`) instead of inline per team — a code + inline score overflowed the narrow box and truncated "CAN"→"C…"; and the R32 columns were narrowed (color 196→158px, mono 188→152px) so the inner rounds aren't squeezed (names still fit). `?test-phase=r32` now seeds several finished ties (R16 boxes show two advancing teams) and `?test-phase=knockout` is a full bracket (all R32 done, one R16 played, QF advancing) so both progressive states are verifiable.

### Design decision: no connector lines

The bracket converges via box placement and column alignment; it deliberately does **not** draw the elbow connector *lines* between rounds — the user reviewed it and decided the lines aren't needed (drawing them on the dense mono panel risks noise for little gain). Not a TODO. (The earlier "R16 clips on mono" follow-up is also obsolete — that was the old `bracketLayout`, now deleted; the unified `knockoutBracket` is verified clipping-free on both displays at 800×480.)

---

## 50. World Cup Penalty Shootouts Show "Match (Pens)" (v3.15.10, 2026-06-30)

The first knockout results came in and penalty-decided ties showed only the shootout number, hiding the actual match result (a 0-0 won 4-2 on penalties read as a bare "4-2").

**Root cause — football-data folds the shootout into `fullTime`.** Per [football-data.org's "Dealing with scores"](https://docs.football-data.org/general/v4/overtime.html), for a `PENALTY_SHOOTOUT` match `fullTime = regularTime + extraTime + penalties` (their example: 1-1 a.e.t., 6-5 pens → `fullTime 7-6`), with the shootout also exposed separately as a `penalties` object. We were reading only `fullTime` into `homeScore`/`awayScore` and rendering it directly, so a 0-0 (4-2 pens) tie surfaced its `fullTime` of 4-2 — indistinguishable from a 4-2 regulation win.

**Fix.** Parse the separate `penalties` object into new optional `WcMatch.penaltyHome`/`penaltyAway` fields (football-data adapter only — the openfootball fallback parses group stage only, never penalties). A new `scoreText(m)` helper renders the result the way sports sources do: **match result first, shootout in parens** — `"1-1 (4-2)"`. The match result is recovered as `fullTime − penalties` (= regulation + extra time), so it stays correct even when goals were scored in extra time; non-shootout matches are unchanged (`"2-1"`). Used by the bracket boxes, the champion final line, and `matchCell`.

- **`homeScore`/`awayScore` stay as `fullTime`** (penalty-inclusive) so `winnerTeam`/`advanceRound` keep advancing the shootout winner unchanged (the penalty winner always has the higher `fullTime`; a finished KO tie is never level).
- **Bracket rendering:** on a shootout the misleading folded per-team score is **suppressed** and the whole `"1-1 (4-2)"` goes on the when-line (both the wide R32 boxes and the narrow inner-round boxes); the winner is still shown bold. Non-shootout boxes keep their per-team scores + "Full time".
- Verified at 800×480 on both displays and both phases via the `?test-phase=r32` (wide R32 shootout) and `?test-phase=knockout` (compact inner-round shootout) fixtures, which now seed penalty ties.
- Cache bumps: `wc:data:v3→v4` (new field on the blob), `wc:image:v10→v11` (mono pre-dithered image re-renders).
