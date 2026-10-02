# E-Ink Dashboard — AGENTS.md

Entry point for agents that load this file automatically. It is deliberately short: the detail lives in the files below, and duplicating it here is how this file went stale before.

## Read these, in this order

1. **`HANDOVER.md`** — the whole project in one file: how the panels get their pixels, routes, pipelines, caches, free-tier budgets, how the system degrades, the rules, the process, commands and a runbook.
2. **`CLAUDE.md`** — session checklist and the obligations table (what to do in each situation). Written for Claude Code, but the rules apply to any agent.
3. **`DECISIONS.md`** — 65 numbered records of why things are the way they are, including every rejected approach. Search it before proposing something; it has probably been tried.
4. **`README.md`** — endpoints, setup, troubleshooting table.

## What this is

"Moment Before" — one Cloudflare Worker (TypeScript, plain `switch` router, no framework, zero runtime dependencies) on the **free tier**, serving two 800x480 e-ink panels: a reTerminal E1001 (mono, home, Naperville IL) and a reTerminal E1002 (Spectra 6 colour, office, Chicago IL). Daily AI illustration of a historical event, weather dashboards, a daily city skyline, and steel/trade headlines.

The panels never fetch anything. SenseCraft HMI renders each URL in a cloud headless Chromium, screenshots it about one second after the document arrives, and pushes the image to the device.

## Rules that cause outages when broken

- A page a panel loads is **one request**, exactly 800x480, pure `#000` on `#fff`, no emoji, no JavaScript. Inline images as `data:` URIs.
- Never make a panel request wait on AI generation; serve the most recent cached image and generate in the background.
- Never pass raw bytes to `env.IMAGES.input()` — wrap them with `bytesToImageStream()`.
- Cost every change against the free tier: 10,000 AI neurons/day, 1,000 KV writes/day (about 590 already used), 5 cron triggers per account.
- Bump the cache-key version after any pipeline change. Every KV `put` carries an `expirationTtl` far larger than the soft TTL.
- Do not cross-contaminate the image pipelines. Do not delete the retired World Cup code (kept for 2030).
- Never write a credential to any file. This repository is public.

## Before you commit

```bash
npm run typecheck
npm run test:utils
npm run dry-run
```

Test visual changes in a browser at exactly 800x480, in every state. Update `DECISIONS.md`, `README.md`, `HANDOVER.md` and `CLAUDE.md` in the same commit. Never rewrite history in the docs — mark a record superseded and add a new one. Version bumps need the owner's approval. Report what you verified and what you did not.

When documents disagree about current behaviour, the source code and the live `/health-detailed` endpoint win. For *why*, `DECISIONS.md` wins.
