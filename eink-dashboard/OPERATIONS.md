# Current E-Ink operations

Current as of 2026-10-24. This is the current operating reference; older Mac mini or Claude operator descriptions elsewhere are historical. The application remains v3.25.0. Source publication is not another deployment, a version bump or permission to change SenseCraft.

## Responsibility and hosting

Jarvis/Hermes owns engineering, local production, unfinished writing and the result that airs. OpenAI Codex is the configured unattended agent, using the existing subscription-backed sign-in and model gpt-6.1-sol; there is no Claude or paid-model fallback. Completed Claude work and credits remain intact.

The MacBook is the sole local production host. All seven migrated production jobs are loaded there; their seven former mini jobs are disabled and unloaded. Never activate both hosts. Cloudflare still hosts the Worker and stored pages, and SenseCraft's cloud renderer delivers the panel images. Those cloud services did not move onto the MacBook. Each machine's unrelated telemetry reporter is not a second production owner and must not be stopped as part of this handover.

The MacBook must stay awake and online. Local agent work needs Hermes, its existing authenticated Codex route and readable private inputs; mailbox/art delivery also needs Dropbox. Existing cloud content and new local production are different availability boundaries. No scheduled check is a guarantee during an outage.

## Seven Mac-scheduled jobs

All local times use America/Chicago and follow daylight saving time.

| Job | Schedule | Responsibility |
|---|---|---|
| `com.eink.serial-weekly` | Daily 09:15 | Check the forward writing horizon; write only missing desk-serial days and art instructions through the ordinary Codex producer. The name is weekly; the check runs daily. |
| `com.eink.serial-run` | Every 120 seconds | Collect delivered desk artwork, run the ordinary checks/review and build/publish accepted beats. |
| `com.eink.series-mailbox` | Incoming-file watch plus every 30 minutes | Process the established engineering/series mailbox and return bound receipts. |
| `com.eink.art-brief` | Daily 17:00, 19:00, 21:00 | Prepare or reconcile tomorrow's general dashboard artwork brief, preserving completed targets. |
| `com.eink.series-heartbeat` | Monday 08:00 | Produce the engineering status heartbeat. |
| `com.eink.storybook` | Every 10 minutes; also at load | Archive a new aired desk-story moment and publish the private web-book index; skip moments already kept. |
| `com.eink.storybook-epub` | Daily 00:15 | Rebuild the EPUB from archived frames. |

The old `series-weekly` chapter-authoring job stays retired. Long Drift chapter production belongs to the existing Jarvis coordinator, not a second chapter scheduler.

## Six Hermes cron jobs

These are existing schedules, not additional workers created by this document. All times are America/Chicago.

| Existing job | Schedule | Responsibility |
|---|---|---|
| `the-long-drift-production-watch` | Every 15 minutes | Sole standing coordinator: reconcile the canonical production ledger, deadlines, source replies, real deliveries and the next permitted step. Respect incumbent ownership and all genuine holds. |
| `desk-serial-private-art` | Daily 17:30 and 19:30 | Paint missing dated desk beats in deadline order; retain completed deliveries. Tomorrow's set is due by 21:00. Reconcile in-flight work before another submission. |
| `eink-daily-art-1710` | Daily 17:10 | Complete missing items in the general dashboard-art brief. |
| `eink-daily-art-1910` | Daily 19:10 | Reconcile remaining general artwork; do not redraw completed items. |
| `eink-daily-art-2110` | Daily 21:10 | Final existing general-art pickup/reconciliation. |
| `the-long-drift-monday-production-report` | Monday 08:30 | Read-only, spoiler-free owner report using the MacBook heartbeat and verified production evidence. It is not another dispatcher. |

The three general-art jobs and the desk-serial artist own separate lanes. No-change checks stay silent, except the explicitly requested weekly report. Surface material new failures, deadline risks or owner-only decisions, not repeated unchanged status.

## Desk serial orchestration

Ordinary writer -> immutable day files and private art brief -> existing desk-art worker -> ordinary serial review/build -> accepted R2 pages -> cloud serving -> aired-frame archive and book.

Write only genuinely missing production days. Do not delete a day, change the clock or repaint a completed target to create work. The unattended serial reviewer uses the configured Codex runner; it is not the independent Volt gate used for Jarvis-authored Long Drift chapters. Keep legacy `to-claude` and `from-claude` mailbox names: they are compatibility identifiers, not a Claude dependency.

Family Memory selection/download stays inside the signed Photos helper and requires full Photos authorization. Node receives only the request-bound winning photo and metadata, not the Photos database or private roster. Unresolved producer ownership or cleanup stays fail-closed. Never delete shown history or a live ownership record to force another attempt.

## The Long Drift orchestration

Jarvis is showrunner and owns unfinished Prologue chapters 4–10, Season 1 and Season 4, as well as engineering and operations. Completed Prologue chapters 1–3 remain unchanged with Claude's credit. Season 2 stays with Volt/Inky; Season 3 stays with Pepper.

For a Jarvis-authored chapter: private production -> ordinary package/proof delivery -> original Volt's exact-revision independent PASS -> Jarvis's own complete rendered review -> APPROVE -> engine-confirmed scheduling. A queued message, imported package, access canary or mirrored proof is not approval. Never write or modify Volt's verdict. Changed source/proofs invalidate earlier clearance.

The established airing schedule is unchanged: weekday scenes at 08:00, 10:00, 12:00, 14:00 and 16:00; weekends replay and Sunday 19:00 is the teaser slot. Do not expose any unaired scene, plan, title, picture or proof to the owner or public Git. The completed migration does not clear unfinished creative work, its existing production holds or independent review gates.

## Accepted outcomes and retained limits

The single-host migration, ordinary Codex mailbox read/reply, real daily-art delivery, signed-helper Family Memory selection/download/cloud upload/cleanup, archive/EPUB rebuild and owner-requested functional writing test have acceptance evidence in private operating records. The writing test used the unmodified installed ordinary weekly producer with live native Codex and synthetic private inputs: one new test day/four beats, complete art instructions and one mailbox handoff passed the real checker; all 46 compared real production/source files stayed unchanged. No test scene was displayed or published.

That functional test closes the remaining handover writing check. The next genuinely new scheduled batch, fresh unattended photo selection, natural Monday heartbeat and physical display observation remain separate observations; a prepared-target no-op proves scheduling/idempotency, not fresh generation. The separately held nested-sandbox experiment was neither retried nor repaired. Optional rejected source-repair candidates were not installed or included in this publication, and no retry/recovery hardening beyond the installed source is claimed. Preserve any real incomplete/native-refusal outcome and reconcile before retrying; healthy schedules or native exit0 alone are not proof all requested artifacts exist.

Source and docs only belong in this public repository. Story files, art, private family data, Photos rosters, credentials, local configuration, operating state, raw logs, receipts and full cron prompts stay private. Preserve the unrelated website root. SenseCraft panel changes and application version bumps still require separate owner approval.
