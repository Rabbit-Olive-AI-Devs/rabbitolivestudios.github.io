# Daily-art upload failures — 2026-11-08

## Fault and consequence

An individual image failed with `TypeError: fetch failed` on both attempts. The mailbox uploader nevertheless returned success and archived the whole delivery note as handled. Successful kinds were archived correctly, but the remaining approved image was left pending without the note needed for a normal retry. The retained CLI stack did not expose a lower-level DNS, socket or TLS cause; do not infer one or weaken transport checks.

## Repair

The existing `series-mailbox.mjs` route now distinguishes successful, transiently failed and blocked deliveries. A recognised connection failure gets one inline retry. After exhaustion the same delivery note remains queued for the existing bounded mailbox retry policy; already archived images are not put again. Exhausting that policy escalates explicitly and retains the originals. No new scheduler or retry service is installed.

Authentication, permission, quota and unknown failures do not receive blind retries. Both command streams are classified; a blocked art delivery retains its image, emits a failed receipt, archives the request as failed rather than handled, and stops that mailbox pass. Later intervention must reconcile access and actual remote state before any new recovery request.

A successful upload command alone no longer means received. The uploader downloads the exact remote key and compares every byte before moving the local original to `art/done`. Failed readback leaves the original in place. Mismatched readbacks are retained privately for investigation and stop automatic replay. Commands have a 90-second bound. Diagnostics in the receipt/log use safe classes rather than copying raw command output.

## Operator recovery

1. Preserve the original artifact, exact native receipt and immutable provenance. Never redraw an upload failure.
2. Read the exact remote key first. A timeout is uncertain, not proof that the object is absent. If the stored bytes match the accepted original, do not upload again.
3. If a permitted recovery is genuinely still needed, use one ordinary `ART DELIVERY` containing only the remaining approved image. Do not resend already accepted kinds. Stop on an access, quota or consent denial.
4. Bind acceptance to the native matching receipt, archived request/image and exact stored-copy readback. Keep panel warming and physically observed display separate from storage acceptance.

## Verification and limits

The original false-handled, blind-denial-retry and missing-readback defects were reproduced before implementation. `tests/art-delivery-recovery.test.js` exercises the actual CLI entrypoint using synthetic images and a local transport substitute under a minimal nonsecret environment. It covers pending-on-network-failure, denial in stdout, HTTP 401/403/429, quota/unknown failures, recovery, readback failures/mismatch and repeat idempotency.

Final checks: 579 Node tests, 11 existing Python family-selection tests, TypeScript check and Worker dry-run passed. The first full-suite attempt caught a changed generic escalation phrase; preserving the required session escalation restored all nine affected existing contract tests.

The already-running production coordinator recovered the original failed upload through the ordinary mailbox before any extra manual put was attempted. A fresh production download matched its approved original exactly. Private images, hashes, receipts and dependency manifests remain in private project storage, not this repository.

This local uploader change is used by the existing Mac mailbox process on its next invocation; it does not require a Worker deployment, restart, version bump, cache-key change, new credentials, new model call or panel-setting change. A later naturally occurring delivery under the revised code is separate from the already-verified recovery. Physical panel display is not claimed.
