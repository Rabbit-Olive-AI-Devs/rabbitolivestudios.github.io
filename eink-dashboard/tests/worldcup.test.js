const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { worldCupCacheKey } = fromBuild("src/cache-keys.js");
const {
  computePhase,
  teamCode,
  teamLabel,
  displayName,
  qualifiedFlags,
  matchCell,
  STAGE_LABELS,
  pickRotatingGroup,
  buildBracket,
  chicagoDateOf,
  chicagoTimeOf,
} = fromBuild("src/worldcup-ui.js");
const { normalizeFootballData, mapStage, mapStatus } = fromBuild("src/worldcup-football-data.js");
const { normalizeOpenFootball } = fromBuild("src/worldcup-openfootball.js");
const { FLAGS } = fromBuild("src/worldcup-flags.js");

test("teamLabel returns full name within budget", () => {
  assert.equal(teamLabel({ name: "Brazil", code: "BRA" }, 14), "Brazil");
});
test("teamLabel truncates with ellipsis past budget", () => {
  assert.equal(teamLabel({ name: "Netherlands", code: "NED" }, 9), "Netherla…");
});
test("teamLabel word-aware truncation trims a trailing hyphen", () => {
  // No override for code XXX, so the raw name is truncated; the cut lands on "Bosnia-".
  assert.equal(teamLabel({ name: "Bosnia-Herzegovina", code: "XXX" }, 8), "Bosnia…");
});
test("teamLabel returns TBD for empty name", () => {
  assert.equal(teamLabel({ name: "", code: "" }, 14), "TBD");
});
test("teamLabel escapes HTML", () => {
  assert.equal(teamLabel({ name: "A&B", code: "AB" }, 14), "A&amp;B");
});
test("displayName applies curated overrides", () => {
  assert.equal(displayName({ name: "South Korea", code: "KOR" }), "S. Korea");
  assert.equal(displayName({ name: "South Africa", code: "RSA" }), "S. Africa");
  assert.equal(displayName({ name: "Bosnia-Herzegovina", code: "BIH" }), "Bosnia");
  assert.equal(displayName({ name: "Brazil", code: "BRA" }), "Brazil");
});
const T = (code, name) => ({ code, name: name ?? code });
const R = (code, position, points, name) => ({ team: T(code, name), position, points });
const M = (h, a) => ({ home: h, away: a }); // remaining match: team objects

test("qualifiedFlags: leader is guaranteed when its two chasers play each other (real Group D / USA)", () => {
  // USA 6, AUS 3, PAR 3, TUR 0 after 2 games. Last games: TUR v USA, PAR v AUS.
  const rows = [R("USA", 1, 6), R("AUS", 2, 3), R("PAR", 3, 3), R("TUR", 4, 0)];
  const remaining = [M(T("TUR"), T("USA")), M(T("PAR"), T("AUS"))];
  // AUS & PAR can each reach 6, but they play EACH OTHER -> at most one does. USA is safe.
  assert.deepEqual(qualifiedFlags(rows, remaining), [true, false, false, false]);
});
test("qualifiedFlags: leader is NOT safe when both chasers play different opponents", () => {
  // A 6, B 4, C 4, D 0. Last games: A v B, C v D. B and C can BOTH reach 7 -> A can finish 3rd.
  const rows = [R("A", 1, 6), R("B", 2, 4), R("C", 3, 4), R("D", 4, 0)];
  const remaining = [M(T("A"), T("B")), M(T("C"), T("D"))];
  assert.deepEqual(qualifiedFlags(rows, remaining), [false, false, false, false]);
});
test("qualifiedFlags: a tie on points never marks (conservative on tiebreakers)", () => {
  const rows = [R("A", 1, 4), R("B", 2, 4), R("C", 3, 4), R("D", 4, 4)];
  const remaining = [M(T("A"), T("B")), M(T("C"), T("D"))];
  assert.deepEqual(qualifiedFlags(rows, remaining), [false, false, false, false]);
});
test("qualifiedFlags: once the group is complete, marks the final top 2 by position", () => {
  const rows = [R("A", 1, 9), R("B", 2, 4), R("C", 3, 4), R("D", 4, 0)];
  assert.deepEqual(qualifiedFlags(rows, []), [true, true, false, false]);
});
test("qualifiedFlags: resolves a fixture by team NAME when the matches feed uses a different code (CUR vs CUW)", () => {
  // football-data lists Curaçao as code CUW in standings but CUR in matches; names match.
  // Leader 4 is only "safe" if the Curaçao match is wrongly SKIPPED. Applied correctly,
  // Curaçao can reach 5 while Romeo reaches 4 -> two teams >= 4 -> Leader NOT guaranteed.
  const rows = [
    R("LEA", 1, 4, "Leader"),
    R("CUW", 2, 2, "Curaçao"),
    R("QBC", 3, 2, "Quebec"),
    R("RMO", 4, 1, "Romeo"),
  ];
  const remaining = [
    M(T("LEA", "Leader"), T("RMO", "Romeo")),
    M(T("CUR", "Curaçao"), T("QBC", "Quebec")), // <-- code CUR not in standings; name resolves it
  ];
  assert.deepEqual(qualifiedFlags(rows, remaining), [false, false, false, false]);
});
test("FLAGS map covers participants and is well-formed", () => {
  assert.ok(Object.keys(FLAGS).length >= 40);
  assert.ok(FLAGS["BRA"] && FLAGS["BRA"].includes("|"));
});

test("worldCupCacheKey is stable", () => {
  assert.equal(worldCupCacheKey(), "wc:data:v3");
});

const gm = (status) => ({ stage: "GROUP", status });
const km = (stage, status) => ({ stage, status });

test("computePhase: group while any group match unfinished", () => {
  assert.equal(computePhase([gm("FINISHED"), gm("SCHEDULED")]), "group");
});

test("computePhase: r32 when groups done and latest active round is R32", () => {
  const matches = [gm("FINISHED"), km("R32", "SCHEDULED")];
  assert.equal(computePhase(matches), "r32");
});

test("computePhase: knockout from R16 onward", () => {
  const matches = [gm("FINISHED"), km("R32", "FINISHED"), km("R16", "SCHEDULED")];
  assert.equal(computePhase(matches), "knockout");
});

test("computePhase: r32 even when later-round fixtures exist as scheduled placeholders", () => {
  // football-data publishes all 104 matches; R16/QF/SF/FINAL exist as TBD placeholders
  // from the start. Once the group stage ends, R32 is the active round, so the phase must
  // be r32 (the split layout) — NOT knockout, which would render an all-TBD R16 bracket.
  const matches = [
    gm("FINISHED"),
    km("R32", "SCHEDULED"), km("R16", "SCHEDULED"), km("QF", "SCHEDULED"),
    km("SF", "SCHEDULED"), km("FINAL", "SCHEDULED"),
  ];
  assert.equal(computePhase(matches), "r32");
});

test("computePhase: knockout once R32 is all finished and R16 is the active round", () => {
  const matches = [
    gm("FINISHED"),
    km("R32", "FINISHED"), km("R16", "SCHEDULED"), km("QF", "SCHEDULED"), km("FINAL", "SCHEDULED"),
  ];
  assert.equal(computePhase(matches), "knockout");
});

test("computePhase: champion when final finished", () => {
  const matches = [gm("FINISHED"), km("FINAL", "FINISHED")];
  assert.equal(computePhase(matches), "champion");
});

test("teamCode prefers code, falls back to first 3 letters", () => {
  assert.equal(teamCode({ name: "Brazil", code: "BRA" }), "BRA");
  assert.equal(teamCode({ name: "Brazil", code: "" }), "BRA");
  assert.equal(teamCode({ name: "", code: "" }), "TBD");
});

test("matchCell shows score when finished, time otherwise", () => {
  assert.equal(matchCell({ status: "FINISHED", homeScore: 2, awayScore: 1, timeChicago: "1 PM" }), "2-1");
  assert.equal(matchCell({ status: "LIVE", homeScore: 0, awayScore: 0, timeChicago: "1 PM" }), "0-0");
  assert.equal(matchCell({ status: "SCHEDULED", homeScore: null, awayScore: null, timeChicago: "1 PM" }), "1 PM");
});

test("STAGE_LABELS covers all knockout stages", () => {
  for (const s of ["GROUP", "R32", "R16", "QF", "SF", "THIRD", "FINAL"]) {
    assert.ok(typeof STAGE_LABELS[s] === "string" && STAGE_LABELS[s].length > 0);
  }
});

const mkGroup = (name) => ({ name, rows: [] });

test("pickRotatingGroup prefers a group with a match today", () => {
  const groups = [mkGroup("A"), mkGroup("B"), mkGroup("C")];
  const todayMatches = [{ group: "C", home: { code: "X" }, away: { code: "Y" } }];
  const g = pickRotatingGroup(groups, todayMatches, "ZZZ", 0);
  assert.equal(g.name, "C");
});

test("pickRotatingGroup round-robins by time bucket when no match today", () => {
  const groups = [mkGroup("A"), mkGroup("B"), mkGroup("C")];
  assert.equal(pickRotatingGroup(groups, [], "ZZZ", 0).name, "A");
  assert.equal(pickRotatingGroup(groups, [], "ZZZ", 15).name, "B");
  assert.equal(pickRotatingGroup(groups, [], "ZZZ", 30).name, "C");
  assert.equal(pickRotatingGroup(groups, [], "ZZZ", 45).name, "A");
});

test("pickRotatingGroup returns null for empty groups", () => {
  assert.equal(pickRotatingGroup([], [], "BRA", 0), null);
});

const ko = (stage, h, a, hs = null, as = null, status = "SCHEDULED") => ({
  stage, status, home: { name: h, code: h }, away: { name: a, code: a },
  homeScore: hs, awayScore: as,
});

test("buildBracket groups knockout matches into ordered rounds", () => {
  const matches = [
    ko("R16", "BRA", "SUI", 2, 0, "FINISHED"),
    ko("R16", "ARG", "NED"),
    ko("QF", "BRA", "ARG"),
    ko("FINAL", "BRA", "ENG"),
    ko("SF", "BRA", "FRA"),
  ];
  const b = buildBracket(matches);
  assert.deepEqual(b.rounds.map((r) => r.stage), ["R16", "QF", "SF", "FINAL"]);
  assert.equal(b.rounds[0].matches.length, 2);
  assert.equal(b.rounds[3].stage, "FINAL");
});

test("buildBracket excludes R32 and THIRD from the tree", () => {
  const matches = [ko("R32", "BRA", "SUI"), ko("THIRD", "X", "Y"), ko("R16", "BRA", "ARG")];
  const b = buildBracket(matches);
  assert.deepEqual(b.rounds.map((r) => r.stage), ["R16"]);
  assert.equal(b.third && b.third.stage, "THIRD");
});

test("chicagoDateOf/chicagoTimeOf convert UTC to Chicago", () => {
  // 2026-06-23T18:00Z is 1:00 PM CDT (UTC-5)
  assert.equal(chicagoDateOf("2026-06-23T18:00:00Z"), "2026-06-23");
  assert.equal(chicagoTimeOf("2026-06-23T18:00:00Z"), "1:00 PM");
  // 2026-06-24T02:00Z is 9:00 PM CDT the previous day
  assert.equal(chicagoDateOf("2026-06-24T02:00:00Z"), "2026-06-23");
  assert.equal(chicagoTimeOf("2026-06-24T02:00:00Z"), "9:00 PM");
});

test("mapStage/mapStatus translate football-data enums", () => {
  assert.equal(mapStage("GROUP_STAGE"), "GROUP");
  assert.equal(mapStage("LAST_16"), "R16");
  assert.equal(mapStage("QUARTER_FINALS"), "QF");
  assert.equal(mapStage("FINAL"), "FINAL");
  assert.equal(mapStatus("IN_PLAY"), "LIVE");
  assert.equal(mapStatus("PAUSED"), "LIVE");
  assert.equal(mapStatus("FINISHED"), "FINISHED");
  assert.equal(mapStatus("TIMED"), "SCHEDULED");
});

test("normalizeFootballData maps matches + standings to WorldCupData", () => {
  const matchesJson = {
    matches: [
      {
        id: 1, stage: "GROUP_STAGE", group: "Group A", status: "FINISHED",
        utcDate: "2026-06-23T18:00:00Z",
        homeTeam: { name: "Brazil", tla: "BRA" }, awayTeam: { name: "Serbia", tla: "SRB" },
        score: { fullTime: { home: 3, away: 0 } },
      },
    ],
  };
  const standingsJson = {
    standings: [
      {
        group: "Group A",
        table: [
          { position: 1, team: { name: "Brazil", tla: "BRA" }, playedGames: 1, won: 1, draw: 0, lost: 0, goalDifference: 3, points: 3 },
          { position: 4, team: { name: "Serbia", tla: "SRB" }, playedGames: 1, won: 0, draw: 0, lost: 1, goalDifference: -3, points: 0 },
        ],
      },
    ],
  };
  const data = normalizeFootballData(matchesJson, standingsJson);
  assert.equal(data.source, "football-data");
  assert.equal(data.knockout.length, 0);
  assert.equal(data.groups.length, 1);
  assert.equal(data.groups[0].name, "A");
  assert.equal(data.groups[0].rows[0].qualifying, true);
  assert.equal(data.groups[0].rows[1].qualifying, false);
  assert.equal(data.recentResults.length, 1);
  assert.equal(data.recentResults[0].home.code, "BRA");
  assert.equal(data.recentResults[0].homeScore, 3);
});

test("normalizeOpenFootball maps rounds/matches and derives simple standings", () => {
  const json = {
    rounds: [
      {
        name: "Matchday 1",
        matches: [
          {
            date: "2026-06-23", time: "18:00",
            team1: "Brazil", team2: "Serbia", group: "Group A",
            score: { ft: [3, 0] },
          },
          {
            date: "2026-06-23", time: "21:00",
            team1: "Switzerland", team2: "Cameroon", group: "Group A",
          },
        ],
      },
    ],
  };
  const data = normalizeOpenFootball(json);
  assert.equal(data.source, "openfootball");
  assert.equal(data.recentResults.length, 1);
  assert.equal(data.recentResults[0].homeScore, 3);
  const groupA = data.groups.find((g) => g.name === "A");
  assert.ok(groupA);
  assert.equal(groupA.rows[0].team.code, "BRA");
  assert.equal(groupA.rows[0].points, 3);
});
