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

test("series problems are part of the message and the fingerprint", () => {
  const state = { dateStr: "2026-11-03", missingImages: [], aiBlocked: false, series: ["The series: chapter 3 of the Prologue airs 09/11 and is waiting for approval"] };
  assert.deepEqual(describeProblems(state), state.series);
  assert.match(alertFingerprint(state), /^series:/);
});

test("the post-migration alert watches only the existing macbook TARS identity", () => {
  const { WATCHED_MACHINES, quietMachines, QUIET_AFTER_MS } = fromBuild("src/alert.js");
  assert.deepEqual(WATCHED_MACHINES, ["macbook"]);
  assert.deepEqual(quietMachines({ macbook: NOW, mini: NOW - 10 * QUIET_AFTER_MS }, NOW), []);
  assert.deepEqual(quietMachines({ macbook: NOW }, NOW), []);
});

test("a fresh mini report cannot hide a stale or missing MacBook report", () => {
  const { quietMachines, QUIET_AFTER_MS } = fromBuild("src/alert.js");
  assert.deepEqual(quietMachines({ mini: NOW, macbook: NOW - QUIET_AFTER_MS - 1 }, NOW), ["macbook"]);
  assert.deepEqual(quietMachines({ mini: NOW }, NOW), ["macbook"]);
  assert.deepEqual(quietMachines({}, NOW), ["macbook"]);
});

test("MacBook quiet detection keeps the strict over-two-hours boundary", () => {
  const { quietMachines, QUIET_AFTER_MS } = fromBuild("src/alert.js");
  assert.equal(QUIET_AFTER_MS, 2 * 60 * 60 * 1000);
  assert.deepEqual(quietMachines({ macbook: NOW - QUIET_AFTER_MS + 1 }, NOW), []);
  assert.deepEqual(quietMachines({ macbook: NOW - QUIET_AFTER_MS }, NOW), []);
  assert.deepEqual(quietMachines({ macbook: NOW - QUIET_AFTER_MS - 1 }, NOW), ["macbook"]);
});

test("a quiet MacBook and a night Jarvis missed are alerted (DECISIONS #90, #99)", () => {
  const state = { dateStr: "2026-10-18", missingImages: [], aiBlocked: false, quietMachines: ["macbook"], jarvisMissed: ["moment"] };
  const problems = describeProblems(state);
  assert.match(problems[0], /^The MacBook has not reported for over 2 hours/);
  assert.match(problems[0], /FileVault/);
  assert.match(problems[1], /Jarvis did not deliver 2026-10-18's moment; Workers AI drew it/);
  assert.equal(alertFingerprint(state), "quiet:macbook|jarvis:moment");
  assert.equal(alertFingerprint({ ...state, dateStr: "2026-10-19" }), "quiet:macbook|jarvis:moment");
  assert.equal(alertFingerprint({ dateStr: "2026-10-18", missingImages: [], aiBlocked: false, quietMachines: [], jarvisMissed: [] }), "ok");
  const email = buildAlertEmail("problem", problems, { dateStr: state.dateStr, baseUrl: "https://example.workers.dev" });
  assert.match(email.subject, /MacBook/);
  assert.match(email.text, /MacBook/);
  assert.doesNotMatch(email.subject + email.text, /Mac mini|The macbook/);
});

test("machine naming preserves the mini name and unknown-id fallback", () => {
  const state = { dateStr: "2026-10-18", missingImages: [], aiBlocked: false, quietMachines: ["mini", "other-host"] };
  const problems = describeProblems(state);
  assert.match(problems[0], /^The Mac mini has not reported/);
  assert.match(problems[1], /^The other-host has not reported/);
});

// In-memory bindings only: exercise real orchestration without a Worker or network.
function alertFixture() {
  const { parseReport, mergeReport, emptyState } = fromBuild("src/tars.js");
  const report = (id, name) => parseReport({ machine: { id, name, diskTotalGb: 100, diskFreeGb: 50 } });
  const machines = mergeReport(mergeReport(emptyState(), report("mini", "Mac mini"), NOW), report("macbook", "MacBook"), NOW - 1);
  const values = new Map([["tars:v1:state", JSON.stringify(machines)]]);
  const sent = [], writes = [], events = [];
  let failDelivery = false;
  const env = {
    CACHE: {
      async get(key) { return values.get(key) ?? null; },
      async put(key, value, options) {
        events.push("record");
        writes.push({ key, value: JSON.parse(value), options });
        values.set(key, value);
      },
    },
    SERIES: { async get() { return null; } },
    SEND_EMAIL: { async send(message) {
      events.push("send");
      if (failDelivery) throw new Error("offline test delivery failure");
      sent.push(message);
    } },
    ALERT_FROM: "alert@example.invalid",
    ALERT_TO: "owner@example.invalid",
  };
  const opts = { dateStr: "2026-08-17", chicagoHour: 7, imageKeys: {}, baseUrl: "https://example.invalid", now: NOW };
  return { env, opts, values, sent, writes, events, failDelivery: () => { failDelivery = true; } };
}

test("MacBook orchestration preserves daily repeat and one recovery notice", async (t) => {
  const { runAlertCheck, QUIET_AFTER_MS } = fromBuild("src/alert.js");
  t.mock.method(console, "log", () => {});
  const f = alertFixture();
  // mini stays fresh while macbook crosses the threshold.
  f.opts.now = NOW + QUIET_AFTER_MS;
  const first = await runAlertCheck(f.env, f.opts);
  assert.equal(first.fingerprint, "quiet:macbook");
  assert.equal(first.sent, true);
  assert.match(f.sent[0].subject, /MacBook/);
  assert.deepEqual(f.events, ["send", "record"]);
  assert.deepEqual(f.writes[0], { key: "alert:v1:last", value: { fingerprint: "quiet:macbook", sentAt: f.opts.now }, options: { expirationTtl: 604800 } });
  const firstAt = f.opts.now;
  const refreshMini = () => {
    const state = JSON.parse(f.values.get("tars:v1:state"));
    state.machines.mini.receivedAt = f.opts.now;
    f.values.set("tars:v1:state", JSON.stringify(state));
  };
  f.opts.now = firstAt + ALERT_REPEAT_MS - 1;
  refreshMini();
  assert.equal((await runAlertCheck(f.env, f.opts)).sent, false);
  f.opts.now = firstAt + ALERT_REPEAT_MS;
  refreshMini();
  assert.equal((await runAlertCheck(f.env, f.opts)).sent, true);
  assert.equal(f.sent.length, 2);
  const state = JSON.parse(f.values.get("tars:v1:state"));
  state.machines.macbook.receivedAt = f.opts.now;
  delete state.machines.mini;
  f.values.set("tars:v1:state", JSON.stringify(state));
  const recovered = await runAlertCheck(f.env, f.opts);
  assert.equal(recovered.fingerprint, OK_FINGERPRINT);
  assert.equal(recovered.sent, true);
  assert.match(f.sent[2].subject, /Recovered/);
  assert.equal((await runAlertCheck(f.env, f.opts)).sent, false);
  assert.equal(f.sent.length, 3);
});

test("a failed MacBook email is not recorded and retries on the next check", async (t) => {
  const { runAlertCheck, QUIET_AFTER_MS } = fromBuild("src/alert.js");
  t.mock.method(console, "error", () => {});
  t.mock.method(console, "log", () => {});
  const f = alertFixture();
  f.opts.now = NOW + QUIET_AFTER_MS;
  f.failDelivery();
  for (let attempt = 0; attempt < 2; attempt++) {
    const result = await runAlertCheck(f.env, f.opts);
    assert.equal(result.sent, false);
    assert.equal(result.reason, "offline test delivery failure");
  }
  assert.deepEqual(f.events, ["send", "send"]);
  assert.deepEqual(f.writes, []);
  assert.equal(f.values.has("alert:v1:last"), false);
});
