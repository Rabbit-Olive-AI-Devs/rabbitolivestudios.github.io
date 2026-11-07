# Image model retirement repair — 2026-11-07

Owner: Jarvis. Owner request: implement the repair and document it in this repository.

## What broke and why it matters

Cloudflare's [5 November notice](https://developers.cloudflare.com/changelog/post/2026-11-05-planned-model-deprecations/) moved retirement from 15 to **20 November 2026**. Calls to `@cf/stabilityai/stable-diffusion-xl-base-1.0` will return HTTP410/error5028. Workers and Workers AI themselves are not retiring.

The deployed dashboard still used that model in legacy/raw/1-bit routes and several backup paths. Normal daily pictures use artwork delivered by Jarvis first, then saved pictures, and Workers AI only where artwork is missing. Those uploaded pictures are independent of SDXL. The existing FLUX image generator and Llama text generator are not on this notice's retirement list. This was a real backup-path weakness, not evidence that all panels would stop on the cutoff date.

## Implemented repair

- Reuse `@cf/black-forest-labs/flux-2-klein-9b` through the existing multipart transport, including the older raw and 1-bit routes. Merely replacing the model name in an SDXL JSON request would be invalid.
- Make one image attempt per pipeline. A failed FLUX request goes to saved artwork, not a second request to FLUX disguised as a fallback. Birthday failure also does not trigger a replacement-event image attempt.
- Preserve each pipeline's style, reference-photo bytes and multipart fields, captions, conversion and final 800×480 output. Skyline keeps its photo reference where present; with no photo it makes one text-only FLUX request. Portrait requests retain their existing 1280×768 size and up to four references.
- Let prefix-based recovery find today's pre-migration cache versions, not just yesterday. A colour birthday may reuse today's older-version portrait, or a saved event, but never another day's portrait. With no saved colour picture, the panel gets the existing readable 800×480 text-only unavailable page.
- Preserve delivered-art → cache → missing-picture AI ordering in the daily warm. Respect an existing image-budget block. A new quota failure stops further image generation but does not prevent importing a skyline already delivered by Jarvis.
- Change cache versions to `fact4:v6`, `fact1:v8`, `color-moment:v5`, `color-birthday:v4`, `skyline:v6`. Leave `birthday:v3` and `moment:v1` unchanged. Do not delete old caches.
- Keep schedules, panels, art production, credentials, billing and application version **3.25.0** unchanged. The legacy 1-bit test route now uses FLUX allowance; do not assume it is free.

## Verification completed

| Check | Result |
| --- | --- |
| `npm run typecheck` | PASS |
| `TMPDIR=<scratch> npm run test:utils` | **568 passed, 0 failed**, including 33 retirement regressions |
| `npm run dry-run` | PASS; existing Worker bindings retained |
| `git diff --check` | PASS |
| Old model in executable TypeScript | Absent; regression checks the exact model ID |
| Browser layout | Four actual handler fixture pages rendered at **800×480** with installed Playwright/Chromium; no overflow, console errors, broken images or external asset requests |
| Red-to-green recovery checks | Reproduced and repaired the warm's existing-budget bypass, skipped delivered skyline after quota failure, and second image attempt after portrait failure |

The 33 focused tests cover correct multipart fields, legacy full-colour/1-bit generation, both skyline reference states, birthdays with/without references, no second attempts, same-date old cache versions, wrong-person protection, missing/uploads-first daily warming, empty caches, budget exhaustion, post-generation Images failure and self-contained panel HTML. All image and reference bytes are synthetic. No cloud model or private-file read is used by this regression suite.

Browser states: cold blocked budget; saved event; today's older-version birthday portrait; and saved-event birthday recovery. The browser plugin is not installed, so installed Python Playwright was used. Final browser evidence uses loopback HTTP with cache-busting URLs, not a live model call. A first synthetic birthday screenshot had a fixture metadata-field typo; it was corrected, asserted and rerendered. This was a test-fixture defect, not a production birthday-caption failure.

## Release evidence

Before deployment, authenticated Cloudflare source readback found **one** retiring-model reference. The active version was `be422228-ad28-452f-b873-07a5008df27b` at 100%; its source and deployment projection were retained privately for recovery.

**Production release: deployed and read back on 2026-11-07 at 17:28 UTC.**

- Implementation commit: [`667660c032d646a335fbdbf31d8e2982563c7346`](https://github.com/Rabbit-Olive-AI-Devs/rabbitolivestudios.github.io/commit/667660c032d646a335fbdbf31d8e2982563c7346), pushed and exact remote branch read back.
- Active Worker version: `0b949b79-6f3a-4017-b8a5-51eed609bcfa`, **100%** of traffic; deployment `b3904f5d-031a-46e9-8add-01db8a3452a5`.
- Authenticated source download: **zero retiring-model references**, with the three existing FLUX transport definitions retained.
- The downloaded `index.js` matches a fresh local Wrangler bundle **byte-for-byte**: 755,518 bytes; SHA256 `3dfcbaeaa7a67617315c492cf42bff72a669befc451ab4ba9450fc6368e440c2`.
- Live schedule API readback retains exactly `35 4,5 * * *` and `5 0,6,12,18 * * *`. Application version remains **3.25.0**.
- No live model test, physical panel observation or unobserved future scheduled run is claimed. This verifies the released code and schedule independently of the local failure simulations.

## Limits and operating notes

- These tests establish request construction and failure behavior, not fresh live FLUX/Images generation or physical panel observation. No paid test generation was triggered: this transport does not admit the usage-analytics query required by the standing spending rule. Do not widen the broker to make that check pass.
- A historical direct health-URL read was denied. Do not retry or change the client to work around it; use admitted Cloudflare source/deployment reads for release verification.
- The existing alert check remains in place for missing images, budget blocks and missed deliveries. A last-good picture is graceful degradation, not evidence that fresh generation succeeded.
- The account-wide audit remains bounded: two unreadable Worker programs, three deployed Pages function bundles and remote bot scripts were not cleared by this dashboard repair.
- `DECISIONS.md` #102 supersedes historical SDXL transport/fallback claims without erasing their history. README and HANDOVER describe current behavior. Protected legacy `CLAUDE.md` was not edited; its old SDXL table is explicitly superseded in those current documents.
- Detailed receipts, original source snapshots, test logs, fixture HTML and screenshots stay in the existing private repair project, not this public repository. No family photos, story content, local home paths, private message identifiers or credentials are added here. Pre-existing unrelated untracked work is preserved.

## Recovery

The previous active Worker version above is the rollback reference. Rollback is a separate consequential operation: first reconcile the exact live deployment; never roll back blindly after a timeout. A rollback would restore the old SDXL dependency, so it is not a safe long-term answer after 20 November. Cache prefixes remain backward-compatible and no cache purge is required.
