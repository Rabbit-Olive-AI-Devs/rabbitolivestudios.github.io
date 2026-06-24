# Release Gate Checklist

Use this checklist before every App Store submission.

## 1) Code and CI Gates

- [ ] PR merged to `main` with required checks passing.
- [ ] iOS build/test workflow is green for the exact commit being released.
- [ ] Version/build numbers are bumped and consistent.
- [ ] Changelog/release notes updated.

## 2) Device and Runtime Gates

- [ ] Cold launch smoke test on physical device.
- [ ] Campaign/startup flow smoke test on at least 3 planets.
- [ ] Daily Challenge smoke test.
- [ ] Ad flow sanity check (retry/next-level cadence).
- [ ] Game Center sign-in/no-sign-in fallback verified.

## 3) App Store Connect Metadata Gates

- [ ] "What’s New" text matches actual shipped changes.
- [ ] Review notes include explicit repro/fix details for hotfixes.
- [ ] Privacy/ATT statements still accurate.
- [ ] Screenshots correspond to current build content.

## 4) ASC Automation Safety Gates

- [ ] `ASC Dry Run` workflow succeeds for target app/version/build.
- [ ] Production submission workflows require manual approval.
- [ ] ASC credentials are only in GitHub Secrets (never committed).
- [ ] Secrets are not printed in logs.

## 5) Submission + Post-Submission

- [ ] Build attached to correct App Store version.
- [ ] Submission completed and state recorded (timestamp + status).
- [ ] Monitor review status until approved/distributed.
- [ ] Post-release smoke test on live build.
