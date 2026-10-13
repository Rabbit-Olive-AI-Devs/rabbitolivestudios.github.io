const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const tars = fromBuild("src/tars.js");
const tp = fromBuild("src/hud/tars-page.js");
const cv = fromBuild("src/hud/canvas.js");
const desk = fromBuild("src/pages/desk.js");

const NOW = Date.parse("2026-10-13T20:58:00Z"); // 15:58 in Chicago
const MIN = 60_000, HOUR = 60 * MIN, DAY = 24 * HOUR;
const iso = (ms) => new Date(ms).toISOString();

function snapshot(over = {}) {
  return {
    schemaVersion: 1,
    providers: [
      { id: "codex", name: "Codex", error: null, identity: { accountEmail: "redacted@gmail.com", plan: "Pro 20x" },
        windows: [{ kind: "weekly", label: "Weekly", usedPercent: 5, resetAt: iso(NOW + 6 * DAY + 8 * HOUR) }],
        cost: { last30DaysUSD: 1892.9 } },
      { id: "claude", name: "Claude", error: null, identity: { accountEmail: "redacted@gmail.com", plan: "Max" },
        windows: [
          { kind: "session", label: "Session", usedPercent: 1, resetAt: iso(NOW + 2 * HOUR + 42 * MIN) },
          { kind: "weekly", label: "Weekly", usedPercent: 2, resetAt: iso(NOW + DAY + 2 * HOUR) },
          { kind: "claude-weekly-scoped-fable", label: "Fable only", usedPercent: 0, resetAt: iso(NOW + DAY + 2 * HOUR) },
        ], cost: { last30DaysUSD: 47.6 } },
      { id: "cursor", name: "Cursor", error: { message: "cookie expired" }, identity: null, windows: [], cost: null },
      { id: "museai", name: "Muse (muse.ai)", error: null, identity: { plan: "Muse Free" },
        windows: [{ kind: "session", label: "Weekly", usedPercent: 100, resetAt: iso(NOW + 3 * DAY + 17 * HOUR) }], cost: null },
    ],
    ...over,
  };
}

const report = (id, name, over = {}) => ({
  machine: { id, name, diskTotalGb: 228, diskFreeGb: 43, memUsedPct: 36, uptimeSec: 21 * 3600 + 22 * 60, ...over.machine },
  codexbar: "codexbar" in over ? over.codexbar : snapshot(),
});

function kv(initial = {}) {
  const store = new Map(Object.entries(initial));
  const puts = [];
  return { store, puts, async get(k) { return store.has(k) ? store.get(k) : null; }, async put(k, v, o) { puts.push({ k, o }); store.set(k, v); } };
}

// --- Reading a report ------------------------------------------------------------

test("a report is reduced to plain numbers and short capital labels", () => {
  const r = tars.parseReport(report("mini", "Mac mini"));
  assert.equal(r.machine.id, "mini");
  assert.equal(r.machine.name, "MAC MINI");
  assert.equal(r.machine.diskFreeGb, 43);
  assert.deepEqual(r.providers.map((p) => p.id), ["codex", "claude", "museai"], "a provider that errored is not a reading");
  assert.deepEqual(r.failed, [{ id: "cursor", name: "CURSOR" }]);
  assert.equal(r.providers[0].plan, "PRO 20X");
  assert.equal(JSON.stringify(r).includes("gmail"), false, "account emails are dropped, even redacted ones");
});

test("hostile or broken reports are refused or cleaned, never stored as given", () => {
  for (const bad of [null, "x", [], {}, { machine: {} }, { machine: { id: "mini" } }, { machine: { id: "../../x", name: "A", diskTotalGb: 1, diskFreeGb: 1 } }]) {
    assert.equal(tars.parseReport(bad), null, JSON.stringify(bad));
  }
  const r = tars.parseReport(report("mini", "<script>alert(1)</script> very long machine name indeed", {
    machine: { diskFreeGb: 1e12, memUsedPct: -50, uptimeSec: "soon" },
    codexbar: { providers: Array.from({ length: 80 }, (_, i) => ({ id: `p${i}`, name: "X".repeat(500), error: null, windows: Array.from({ length: 80 }, () => ({ kind: "weekly", label: "W", usedPercent: 1e9, resetAt: "never" })) })) },
  }));
  assert.match(r.machine.name, /^[A-Z0-9 .\-]{1,14}$/);
  assert.ok(r.machine.diskFreeGb <= r.machine.diskTotalGb);
  assert.equal(r.machine.memUsedPct, 0);
  assert.equal(r.machine.uptimeSec, null);
  assert.ok(r.providers.length <= 12);
  assert.ok(r.providers[0].windows.length <= 6);
  assert.equal(r.providers[0].windows[0].usedPercent, 100);
  assert.equal(r.providers[0].windows[0].resetAt, null);
  assert.ok(r.providers[0].name.length <= 14);
  assert.equal(tars.parseReport(report("mini", "Mac mini", { codexbar: null })).providers.length, 0, "a machine without CodexBar still reports itself");
});

// --- Merging -------------------------------------------------------------------------

test("each provider keeps its newest good reading, whichever machine sent it", () => {
  let s = tars.mergeReport(tars.emptyState(), tars.parseReport(report("mini", "Mac mini")), NOW - HOUR);
  const later = snapshot();
  later.providers[0].windows[0].usedPercent = 9;
  later.providers[2] = { id: "cursor", name: "Cursor", error: null, identity: { plan: "Pro+" }, windows: [{ kind: "session", label: "Total", usedPercent: 12, resetAt: iso(NOW + DAY) }], cost: { last30DaysUSD: 286.4 } };
  later.providers[1] = { id: "claude", name: "Claude", error: { message: "down" }, windows: [] };
  s = tars.mergeReport(s, tars.parseReport(report("macbook", "MacBook", { codexbar: later })), NOW);
  assert.equal(s.providers.codex.windows[0].usedPercent, 9, "newer reading wins");
  assert.equal(s.providers.codex.from, "macbook");
  assert.equal(s.providers.cursor.windows[0].usedPercent, 12, "the MacBook fills in what the mini could not read");
  assert.equal(s.providers.claude.from, "mini", "a failed read does not erase the last good one");
  assert.equal(s.providers.claude.at, NOW - HOUR);
  assert.deepEqual(Object.keys(s.machines).sort(), ["macbook", "mini"]);
  assert.equal(s.machines.mini.receivedAt, NOW - HOUR);
});

test("ingest writes once per report, refuses a flood, and refuses garbage without writing", async () => {
  const env = { CACHE: kv() };
  assert.equal(await tars.ingestReport(env, report("mini", "Mac mini"), NOW), "stored");
  assert.equal(env.CACHE.puts.length, 1);
  assert.equal(env.CACHE.puts[0].k, "tars:v1:state");
  assert.equal(env.CACHE.puts[0].o.expirationTtl, 604800);
  assert.equal(await tars.ingestReport(env, report("mini", "Mac mini"), NOW + 2 * MIN), "too-soon");
  assert.equal(env.CACHE.puts.length, 1, "a reporter stuck in a loop cannot spend the KV write budget");
  assert.equal(await tars.ingestReport(env, report("macbook", "MacBook"), NOW + 2 * MIN), "stored", "the limit is per machine");
  assert.equal(await tars.ingestReport(env, { nonsense: true }, NOW + HOUR), "invalid");
  assert.equal(env.CACHE.puts.length, 2);
});

// --- What the page shows -----------------------------------------------------------------

function stateWith(...entries) {
  let s = tars.emptyState();
  for (const [r, at] of entries) s = tars.mergeReport(s, tars.parseReport(r), at);
  return s;
}

test("durations read as days and hours, hours and minutes, or minutes", () => {
  assert.equal(tars.formatSpan(6 * DAY + 8 * HOUR), "6D 08H");
  assert.equal(tars.formatSpan(2 * HOUR + 42 * MIN), "2H 42M");
  assert.equal(tars.formatSpan(42 * MIN), "42M");
  assert.equal(tars.formatSpan(-5), "NOW");
});

test("the view turns a fresh state into rows, bars and countdowns", () => {
  const v = tars.tarsView(stateWith([report("mini", "Mac mini"), NOW - 3 * MIN]), NOW);
  assert.equal(v.machines.length, 1);
  assert.equal(v.machines[0].awake, true);
  assert.equal(v.machines[0].diskUsedPct, 81);
  assert.equal(v.machines[0].since, "21H 22M");
  const row = (id) => v.providers.find((p) => p.id === id);
  assert.deepEqual([row("codex").session, row("codex").week, row("codex").reset], [undefined, 5, "6D 08H"]);
  assert.deepEqual([row("claude").session, row("claude").week, row("claude").reset], [1, 2, "2H 42M"]);
  assert.equal(row("claude").note, "WEEK 1D 02H");
  assert.equal(row("museai").week, 100, "a window labelled Weekly counts as the week even when its kind says session");
  assert.equal(row("museai").low, true);
  assert.equal(row("museai").name, "MUSE");
  assert.deepEqual(v.providers.map((p) => p.id), ["codex", "claude", "cursor", "museai"], "known providers first, in a fixed order; Cursor failed but keeps its row");
  assert.deepEqual(v.next, { numeral: "2", text: "2H 42M", what: "CLAUDE" });
  assert.equal(v.spend, "$1,941");
  assert.ok(v.alerts.includes("MUSE WEEK 100%"));
});

test("a machine that stops reporting is shown asleep with the time it was last seen", () => {
  const v = tars.tarsView(stateWith([report("mini", "Mac mini"), NOW - 8 * HOUR]), NOW);
  assert.equal(v.machines[0].awake, false);
  assert.equal(v.machines[0].since, "07:58");
  assert.equal(v.machines[0].memUsedPct, null, "memory of a sleeping machine is not shown as current");
});

test("a provider with no recent good reading shows as no data, with when it last worked", () => {
  const s = stateWith([report("mini", "Mac mini"), NOW - 9 * HOUR], [report("mini", "Mac mini", { codexbar: { providers: [{ id: "codex", name: "Codex", error: { message: "x" }, windows: [] }] } }), NOW - 2 * MIN]);
  const v = tars.tarsView(s, NOW);
  const codex = v.providers.find((p) => p.id === "codex");
  assert.equal(codex.week, null);
  assert.equal(codex.reset, "UNKNOWN");
  assert.match(codex.note, /^LAST GOOD /);
  assert.ok(v.alerts.includes("CODEX NO DATA"));
});

test("alerts name what needs attention and stay within what the card can show", () => {
  const busy = snapshot();
  busy.providers[1].windows[0].usedPercent = 64;
  busy.providers[1].windows[1].usedPercent = 91;
  const v = tars.tarsView(stateWith([report("mini", "Mac mini", { codexbar: busy, machine: { diskFreeGb: 12 } }), NOW - MIN]), NOW);
  assert.ok(v.alerts.includes("CLAUDE WEEK 91%"));
  assert.ok(v.alerts.includes("MAC MINI DISK 95%"));
  assert.ok(v.alerts.length <= 4);
  assert.equal(v.providers.find((p) => p.id === "claude").low, true);
});

test("TARS says something short that fits its card, and flags a joke", () => {
  const calm = snapshot();
  calm.providers = calm.providers.slice(0, 2);
  for (const s of [stateWith([report("mini", "Mac mini"), NOW - MIN]), stateWith([report("mini", "Mac mini", { codexbar: calm }), NOW - MIN]), tars.emptyState(), stateWith([report("mini", "Mac mini"), NOW - 9 * HOUR])]) {
    const v = tars.tarsView(s, NOW);
    assert.ok(v.quip.length >= 1 && v.quip.length <= 3);
    for (const line of v.quip) assert.ok(line.length <= 31, `"${line}" is too long for the card`);
    assert.equal(typeof v.joke, "boolean");
  }
  assert.match(tars.tarsView(stateWith([report("mini", "Mac mini"), NOW - MIN]), NOW).quip.join(" "), /Muse/);
});

// --- The page and its routes ------------------------------------------------------------

test("the page is a full panel when empty, normal, and crowded", () => {
  const many = snapshot();
  many.providers = Array.from({ length: 9 }, (_, i) => ({ ...many.providers[1], id: `svc${i}`, name: `Service Number ${i}` }));
  const states = [
    tars.emptyState(),
    stateWith([report("mini", "Mac mini"), NOW - MIN], [report("macbook", "MacBook Pro"), NOW - 2 * MIN]),
    stateWith([report("mini", "Mac mini", { codexbar: many }), NOW - MIN], [report("macbook", "MacBook Pro"), NOW - 9 * HOUR], [report("third", "Third Mac"), NOW - MIN]),
  ];
  for (const s of states) {
    const c = tp.drawTarsPage(tars.tarsView(s, NOW), new Date(NOW));
    assert.deepEqual([c.w, c.h], [800, 480]);
    const ratio = cv.countBlack(c) / (800 * 480);
    assert.ok(ratio > 0.015 && ratio < 0.4, `black ratio ${ratio}`);
  }
});

const post = (body, key) => new Request("https://x/desk/tars", {
  method: "POST", body: typeof body === "string" ? body : JSON.stringify(body),
  headers: key ? { Authorization: `Bearer ${key}` } : {},
});

test("reports need the key, and the page hides itself without it", async () => {
  const env = { CACHE: kv(), TARS_KEY: "s3cret" };
  assert.equal((await desk.handleDeskTarsReport(post(report("mini", "Mac mini")), env)).status, 404);
  assert.equal((await desk.handleDeskTarsReport(post(report("mini", "Mac mini"), "wrong"), env)).status, 404);
  assert.equal(env.CACHE.puts.length, 0);
  const ok = await desk.handleDeskTarsReport(post(report("mini", "Mac mini"), "s3cret"), env);
  assert.equal(ok.status, 200);
  assert.equal(env.CACHE.puts.length, 1);
  assert.equal((await desk.handleDeskTarsReport(post("{not json", "s3cret"), env)).status, 400);
  assert.equal((await desk.handleDeskTarsReport(post("x".repeat(70_000), "s3cret"), env)).status, 413);

  assert.equal((await desk.handleDeskTars(env, new URL("https://x/desk/tars"))).status, 404);
  assert.equal((await desk.handleDeskTars(env, new URL("https://x/desk/tars?key=nope"))).status, 404);
  const page = await desk.handleDeskTars(env, new URL("https://x/desk/tars?key=s3cret"));
  assert.equal(page.status, 200);
  assert.match(await page.text(), /src="data:image\/png;base64,/);
});

test("with no key configured the routes stay closed in production terms: reports are refused", async () => {
  const env = { CACHE: kv() };
  assert.equal((await desk.handleDeskTarsReport(post(report("mini", "Mac mini"), "anything"), env)).status, 404, "no key set means nobody can post");
  assert.equal((await desk.handleDeskTars(env, new URL("https://x/desk/tars"))).status, 200, "the empty page is viewable for local development");
});

test("a provider that has never answered still gets a row, until it does answer", () => {
  let s = stateWith([report("mini", "Mac mini"), NOW - MIN]);
  let v = tars.tarsView(s, NOW);
  const cursor = v.providers.find((p) => p.id === "cursor");
  assert.ok(cursor, "Cursor failed on its only report, and must not simply vanish");
  assert.equal(cursor.name, "CURSOR");
  assert.deepEqual([cursor.session, cursor.week, cursor.reset, cursor.note], [null, null, "UNKNOWN", "NEVER READ"]);
  assert.ok(v.alerts.includes("CURSOR NO DATA"));
  assert.deepEqual(v.providers.map((p) => p.id), ["codex", "claude", "cursor", "museai"], "it sits in its usual place");

  const fixed = snapshot();
  fixed.providers[2] = { id: "cursor", name: "Cursor", error: null, identity: { plan: "Pro+" }, windows: [{ kind: "session", label: "Total", usedPercent: 12, resetAt: iso(NOW + DAY) }] };
  s = tars.mergeReport(s, tars.parseReport(report("macbook", "MacBook", { codexbar: fixed })), NOW);
  v = tars.tarsView(s, NOW);
  assert.equal(v.providers.find((p) => p.id === "cursor").session, 12);
  assert.equal(s.unread.cursor, undefined, "a good reading clears the marker");
});

test("the reporter script reads the boot time, not its microseconds", () => {
  const fs = require("node:fs");
  const script = fs.readFileSync(path.join(__dirname, "..", "scripts", "tars-report.sh"), "utf8");
  const sed = script.match(/kern\.boottime \| sed -E '([^']+)'/);
  assert.ok(sed, "boot-time extraction not found");
  const { execFileSync } = require("node:child_process");
  const out = execFileSync("sed", ["-E", sed[1]], { input: "{ sec = 1791850000, usec = 123456 } Mon Oct 12 18:35:00 2026\n" }).toString().trim();
  assert.equal(out, "1791850000");
});
