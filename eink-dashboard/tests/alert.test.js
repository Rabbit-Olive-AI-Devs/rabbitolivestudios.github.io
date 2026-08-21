const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const {
  shouldRunAlertCheck,
  describeProblems,
  alertFingerprint,
  decideAlert,
  buildAlertEmail,
  OK_FINGERPRINT,
  ALERT_REPEAT_MS,
} = fromBuild("src/alert.js");

// --- shouldRunAlertCheck -----------------------------------------------------
// The 6h cron fires at 00:05/06:05/12:05/18:05 UTC. The 06:05 UTC firing lands
// at 01:05 Chicago (CDT) / 00:05 (CST) — right on top of the daily image warm —
// so the check must skip it or it would alert on a cache that is still filling.

test("shouldRunAlertCheck skips the hours that race the daily warm", () => {
  assert.equal(shouldRunAlertCheck(0), false); // 06:05 UTC in CST
  assert.equal(shouldRunAlertCheck(1), false); // 06:05 UTC in CDT
  assert.equal(shouldRunAlertCheck(2), false);
});

test("shouldRunAlertCheck runs once the warm has had time", () => {
  assert.equal(shouldRunAlertCheck(3), true);
  assert.equal(shouldRunAlertCheck(6), true);  // 12:05 UTC in CST
  assert.equal(shouldRunAlertCheck(7), true);  // 12:05 UTC in CDT
  assert.equal(shouldRunAlertCheck(13), true); // 18:05 UTC in CDT
  assert.equal(shouldRunAlertCheck(19), true); // 00:05 UTC in CDT
});

// --- describeProblems --------------------------------------------------------

test("describeProblems reports nothing when everything is cached", () => {
  const problems = describeProblems({
    dateStr: "2026-08-21",
    missingImages: [],
    aiBlocked: false,
  });
  assert.deepEqual(problems, []);
});

test("describeProblems names each missing daily image", () => {
  const problems = describeProblems({
    dateStr: "2026-08-21",
    missingImages: ["fact4_gray", "skyline"],
    aiBlocked: false,
  });
  assert.equal(problems.length, 1);
  assert.match(problems[0], /fact4_gray/);
  assert.match(problems[0], /skyline/);
  assert.match(problems[0], /2026-08-21/);
});

test("describeProblems reports a blocked AI budget with its source", () => {
  const problems = describeProblems({
    dateStr: "2026-08-21",
    missingImages: [],
    aiBlocked: true,
    aiBlockSource: "cron Pipeline A",
  });
  assert.equal(problems.length, 1);
  assert.match(problems[0], /budget/i);
  assert.match(problems[0], /cron Pipeline A/);
});

test("describeProblems reports both faults together", () => {
  // This is the 2026-08-20 shape: images missing AND the budget blocked.
  const problems = describeProblems({
    dateStr: "2026-08-20",
    missingImages: ["fact4_gray", "fact1_1bit", "color_moment", "skyline", "skyline_bw"],
    aiBlocked: true,
    aiBlockSource: "cron Pipeline A",
  });
  assert.equal(problems.length, 2);
});

// --- alertFingerprint --------------------------------------------------------

test("alertFingerprint is OK when there are no problems", () => {
  assert.equal(alertFingerprint({ dateStr: "2026-08-21", missingImages: [], aiBlocked: false }), OK_FINGERPRINT);
});

test("alertFingerprint ignores the date so a persisting fault does not re-alert at rollover", () => {
  const a = alertFingerprint({ dateStr: "2026-08-20", missingImages: ["skyline"], aiBlocked: true });
  const b = alertFingerprint({ dateStr: "2026-08-21", missingImages: ["skyline"], aiBlocked: true });
  assert.equal(a, b);
});

test("alertFingerprint ignores the order of missing images", () => {
  const a = alertFingerprint({ dateStr: "2026-08-21", missingImages: ["skyline", "fact4_gray"], aiBlocked: false });
  const b = alertFingerprint({ dateStr: "2026-08-21", missingImages: ["fact4_gray", "skyline"], aiBlocked: false });
  assert.equal(a, b);
});

test("alertFingerprint distinguishes different faults", () => {
  const a = alertFingerprint({ dateStr: "2026-08-21", missingImages: ["skyline"], aiBlocked: false });
  const b = alertFingerprint({ dateStr: "2026-08-21", missingImages: ["skyline"], aiBlocked: true });
  const c = alertFingerprint({ dateStr: "2026-08-21", missingImages: ["fact4_gray"], aiBlocked: false });
  assert.notEqual(a, b);
  assert.notEqual(a, c);
});

// --- decideAlert -------------------------------------------------------------

const NOW = 1787000000000;

test("decideAlert sends on a brand-new problem", () => {
  const d = decideAlert("missing:skyline", null, NOW);
  assert.deepEqual(d, { send: true, kind: "problem" });
});

test("decideAlert stays quiet while the same problem persists", () => {
  const prev = { fingerprint: "missing:skyline", sentAt: NOW - 60_000 };
  assert.deepEqual(decideAlert("missing:skyline", prev, NOW), { send: false });
});

test("decideAlert re-sends when the problem changes shape", () => {
  const prev = { fingerprint: "missing:skyline", sentAt: NOW - 60_000 };
  const d = decideAlert("missing:skyline|ai:blocked", prev, NOW);
  assert.deepEqual(d, { send: true, kind: "problem" });
});

test("decideAlert re-sends as a daily reminder after the repeat window", () => {
  const prev = { fingerprint: "missing:skyline", sentAt: NOW - ALERT_REPEAT_MS - 1 };
  const d = decideAlert("missing:skyline", prev, NOW);
  assert.deepEqual(d, { send: true, kind: "problem" });
});

test("decideAlert sends a recovery notice when the fault clears", () => {
  const prev = { fingerprint: "missing:skyline", sentAt: NOW - 60_000 };
  const d = decideAlert(OK_FINGERPRINT, prev, NOW);
  assert.deepEqual(d, { send: true, kind: "recovery" });
});

test("decideAlert stays silent while healthy", () => {
  const prev = { fingerprint: OK_FINGERPRINT, sentAt: NOW - 60_000 };
  assert.deepEqual(decideAlert(OK_FINGERPRINT, prev, NOW), { send: false });
  assert.deepEqual(decideAlert(OK_FINGERPRINT, null, NOW), { send: false });
});

test("decideAlert does not repeat a recovery notice", () => {
  // First the fault clears...
  const prev = { fingerprint: "missing:skyline", sentAt: NOW - 60_000 };
  assert.deepEqual(decideAlert(OK_FINGERPRINT, prev, NOW), { send: true, kind: "recovery" });
  // ...then the recorded state is OK, so the next check is silent.
  const after = { fingerprint: OK_FINGERPRINT, sentAt: NOW };
  assert.deepEqual(decideAlert(OK_FINGERPRINT, after, NOW + 60_000), { send: false });
});

// --- buildAlertEmail --------------------------------------------------------

test("buildAlertEmail summarises the problems in the subject", () => {
  const { subject, text } = buildAlertEmail("problem", [
    "Daily images missing for 2026-08-21: fact4_gray, skyline",
    "AI generation is blocked (set by cron Pipeline A)",
  ], { dateStr: "2026-08-21", baseUrl: "https://example.workers.dev" });
  assert.match(subject, /e-ink/i);
  assert.match(subject, /2/); // problem count
  // every problem must appear in the body — the subject alone is not enough to act on
  assert.match(text, /fact4_gray, skyline/);
  assert.match(text, /cron Pipeline A/);
  assert.match(text, /health-detailed/);
});

test("buildAlertEmail writes a distinct recovery message", () => {
  const { subject, text } = buildAlertEmail("recovery", [], {
    dateStr: "2026-08-21",
    baseUrl: "https://example.workers.dev",
  });
  assert.match(subject, /recover/i);
  assert.match(text, /recover|clear|resolved/i);
});

test("buildAlertEmail keeps the subject on one line", () => {
  const { subject } = buildAlertEmail("problem", ["a\nb", "c"], {
    dateStr: "2026-08-21",
    baseUrl: "https://example.workers.dev",
  });
  assert.equal(subject.includes("\n"), false);
  assert.equal(subject.includes("\r"), false);
});
