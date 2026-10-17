# Moving the e-ink work from the Mac mini to the MacBook

**Status:** optional, not started (2026-10-17). The owner then suggested keeping both Macs as they are and reaching Jarvis through the series mailbox (see the picture pipeline design), which needs no move. Keep this plan for when the MacBook should become the home.
**Why:** the owner works from the MacBook, and Jarvis (the owner's Hermes agent, with GPT image generation) lives there. That makes the Jarvis picture pipeline (`docs/superpowers/specs/2026-10-17-jarvis-picture-pipeline-design.md`) simpler.

## What was checked on the Mac mini (2026-10-17)

The owner said the mini "sleeps and locks". On inspection it does **not** system-sleep. `pmset` sleep is 0, and the power log has no sleep entries. It turns the display off after 180 minutes and locks the screen.

All four e-ink jobs kept running overnight:
- TARS reported every 30 minutes.
- The mailbox handled messages at 00:04, 08:27 and 09:09 UTC.
- No job log has errors.

The lock costs two things, both already worked around:
- The Keychain is unavailable to remote (SSH) sessions. This is why the series key lives in `~/.config/eink/series-key`.
- Driving apps such as Photos (the AppleScript export in #87) needs the logged-in desktop.

So the move is about where the owner and Jarvis work, not about the mini failing.

## What lives on the Mac mini

| Thing | Where | Notes |
|---|---|---|
| Repo checkout | `/Users/tboliveira/Projetos/rabbitolivestudios.github.io` | Keep the **same path** on the MacBook. Claude's memory folder is keyed by it (`~/.claude/projects/-Users-tboliveira-Projetos-rabbitolivestudios-github-io/memory/`). |
| Claude memory | `~/.claude/projects/-Users-tboliveira-Projetos-rabbitolivestudios-github-io/memory/` | Copy it across; it is the cross-session state. |
| Cloudflare login | `~/Library/Preferences/.wrangler/config/default.toml` (OAuth) | Do not copy: run `npx wrangler login` on the MacBook. The account id is passed inline (`CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c`). |
| Series key | `~/.config/eink/series-key` (mode 600) | Copy securely (scp), keep 600. Never into the repo. |
| Series working files | `~/series-work/` | Claude's chapter notes, reviews, packages, builds. `approve` trusts `~/series-work/reviews`. |
| Dropbox | `/Volumes/SamsungSSD/.CloudStorage/Data/Dropbox` (on the external SSD) | Mailbox `00_SHARED_INBOX/series`, proofs `00_JARVIS_SPOILERS`. |
| Photos library | `/Volumes/SamsungSSD/Pictures/Photos Library.photoslibrary` | Source of the birthday references (#87). |
| Birthday photos | `eink-dashboard/photos/portraits/` + `people.json`, `hints.json`, `picks.json` | Gitignored; copy across. Backups: `photos/r2-backup-2026-10-16/`, `photos/portraits-before-2026-10-17/`. |
| LaunchAgents | `~/Library/LaunchAgents/com.eink.{series-mailbox,series-weekly,series-heartbeat,tars-report}.plist` | Templates in `scripts/launchd/`. TARS runs on **both** Macs and stays on both. |
| Jarvis | `com.jarvis.bluebubbles` LaunchAgent | Jarvis's own; the owner decides whether it moves. |

## Paths that are hard-coded to the mini

- `scripts/series-mailbox.mjs` defaults `SERIES_MAILBOX` and `SERIES_PROOFS` to the SSD Dropbox path (env vars override).
- `scripts/series-import.mjs` does the same for `SERIES_PROOFS`.
- `scripts/launchd/com.eink.series-mailbox.plist` watches `/Volumes/SamsungSSD/.CloudStorage/Data/Dropbox/00_SHARED_INBOX/series/to-claude`.

Before moving, make these one setting: read the Dropbox root from one env var or config file, with the SSD path as the fallback. That way the same scripts run on either Mac.

## Stages

Each stage ends with a check, and nothing on the mini is switched off before the MacBook has passed its check.

**1. Tools on the MacBook.**
- git, Node (same major as the mini), `npm ci` in `eink-dashboard/`.
- Python 3 with Pillow, the Xcode command-line tools (for `swift`), Claude Code.
- `npx wrangler login`.
- Check: `npm run typecheck && npm run test:utils && npm run dry-run` pass.

**2. State.**
- Clone the repo at the same path.
- Copy across: Claude's memory folder, `~/series-work/`, `~/.config/eink/series-key`, and `photos/portraits/` with its JSON files.
- Check: the start-of-session checklist reads the same memory, and `node scripts/series-timeline.mjs` sees R2 holding every page.

**3. Dropbox.**
- Decide one of two:
  - the SSD moves to the MacBook, with the same mount path;
  - the MacBook's own Dropbox syncs `00_SHARED_INBOX/series` and `00_JARVIS_SPOILERS`, set through the new path setting.
- Online-only files need `scripts/materialize-exec.py` on the MacBook too.
- Check: the mailbox folder lists the same files on both Macs.

**4. Jobs.**
- First unload the mini's series jobs: `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.eink.series-*.plist`. Two mailbox watchers would both act on Jarvis's messages, including APPROVE.
- Then install them on the MacBook from `scripts/launchd/` with its paths.
- Install `com.eink.tars-report` on the MacBook if it is not there already. Keep the mini's copy, because TARS reports each machine.
- Check: drop a harmless test note in `to-claude/`, see it handled in `~/Library/Logs/eink-series/mailbox.log`, and check that the Monday heartbeat and the Wednesday run fire at their next times.

**5. MacBook settings.**
- On power: "Prevent automatic sleeping when the display is off". The lid closed counts as sleep unless it runs in clamshell mode with power and a display.
- The jobs miss runs while the laptop is asleep or travelling. launchd runs a missed calendar job at wake, but the mailbox is only watched while the MacBook is awake.
- Decide whether the mini stays as a standby (series jobs unloaded, ready to reload) or is retired.

**6. Docs.**
- Update `HANDOVER.md` (the off-Worker jobs paragraph), `CLAUDE.md` (session start: where the logs are), the `README.md` series section, and DECISIONS (a new record for the move).
- Update memory: the Mac mini SSH tunnel note becomes history.

## Not moving

The Worker, KV, R2, crons and the panels are all in Cloudflare and SenseCraft. Nothing about them depends on which Mac is used.
