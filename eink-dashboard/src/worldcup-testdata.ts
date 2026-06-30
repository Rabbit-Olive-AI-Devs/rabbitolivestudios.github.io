/**
 * Canned WorldCupData fixtures for ?test / ?test-phase visual testing.
 * Cheap (no upstream) so these routes need no auth (like ?test-device).
 */

import type { WorldCupData, WcMatch, WcGroup, WcTeam, WcPhase, WcStage, WcStatus } from "./types";

const T = (name: string, code: string): WcTeam => ({ name, code });

function m(
  id: number, stage: WcStage, status: WcStatus, time: string,
  home: WcTeam, away: WcTeam, hs: number | null, as: number | null,
  group?: string, dateChicago = "2026-06-23",
  ph: number | null = null, pa: number | null = null,
): WcMatch {
  return {
    id, stage, group, status,
    kickoffISO: `${dateChicago}T${id.toString().padStart(2, "0")}:00:00Z`,
    dateChicago, timeChicago: time,
    home, away, homeScore: hs, awayScore: as, penaltyHome: ph, penaltyAway: pa,
  };
}

const BRA = T("Brazil", "BRA"), SRB = T("Serbia", "SRB"), SUI = T("Switzerland", "SUI"), CMR = T("Cameroon", "CMR");
const ARG = T("Argentina", "ARG"), NED = T("Netherlands", "NED"), ENG = T("England", "ENG"), FRA = T("France", "FRA");
const ESP = T("Spain", "ESP"), POR = T("Portugal", "POR"), GER = T("Germany", "GER"), USA = T("USA", "USA");
// Additional teams for the 16-match Round-of-32 fixture (mirrors the real 2026 bracket).
const RSA = T("South Africa", "RSA"), CAN = T("Canada", "CAN"), PAR = T("Paraguay", "PAR"), MAR = T("Morocco", "MAR");
const JPN = T("Japan", "JPN"), SWE = T("Sweden", "SWE"), CIV = T("Ivory Coast", "CIV"), NOR = T("Norway", "NOR");
const MEX = T("Mexico", "MEX"), ECU = T("Ecuador", "ECU"), COD = T("Congo DR", "COD"), BIH = T("Bosnia-Herzegovina", "BIH");
const BEL = T("Belgium", "BEL"), SEN = T("Senegal", "SEN"), CRO = T("Croatia", "CRO"), AUT = T("Austria", "AUT");
const ALG = T("Algeria", "ALG"), CPV = T("Cape Verde", "CPV"), COL = T("Colombia", "COL"), GHA = T("Ghana", "GHA");
const AUS = T("Australia", "AUS"), EGY = T("Egypt", "EGY");

function groupA(): WcGroup {
  return {
    name: "A",
    rows: [
      { position: 1, team: BRA, played: 2, won: 2, drawn: 0, lost: 0, goalDifference: 5, points: 6, qualifying: true },
      { position: 2, team: SUI, played: 2, won: 1, drawn: 1, lost: 0, goalDifference: 2, points: 4, qualifying: false },
      { position: 3, team: CMR, played: 2, won: 0, drawn: 1, lost: 1, goalDifference: -2, points: 1, qualifying: false },
      { position: 4, team: SRB, played: 2, won: 0, drawn: 0, lost: 2, goalDifference: -5, points: 0, qualifying: false },
    ],
  };
}

export function testWorldCupData(phase: WcPhase): WorldCupData {
  const base: WorldCupData = {
    source: "football-data", phase,
    todayMatches: [], recentResults: [], groups: [], knockout: [], champion: null,
    generatedAt: Date.now(),
  };

  if (phase === "group") {
    const today = [
      m(13, "GROUP", "SCHEDULED", "1:00 PM", USA, SRB, null, null, "C"),
      m(16, "GROUP", "SCHEDULED", "4:00 PM", ARG, NED, null, null, "B"),
      m(19, "GROUP", "LIVE", "7:00 PM", BRA, SUI, 1, 0, "A"),
    ];
    const results = [
      m(20, "GROUP", "FINISHED", "—", ESP, GER, 2, 1, "D", "2026-06-22"),
      m(21, "GROUP", "FINISHED", "—", FRA, POR, 0, 0, "E", "2026-06-22"),
      m(22, "GROUP", "FINISHED", "—", ENG, USA, 3, 0, "C", "2026-06-22"),
    ];
    return { ...base, todayMatches: today, recentResults: results, groups: [groupA()], knockout: [], phase };
  }

  if (phase === "r32") {
    // 16-match Round of 32 in the official bracket order (left 8, then right 8). Four ties finished;
    // football-data has seeded their R16 slots. Brazil's R16 (id 91) sorts into the left half by id
    // but belongs on the RIGHT — orderKnockout repositions it by the real bracket tree (#52).
    const knockout = [
      m(73, "R32", "FINISHED", "—", GER, PAR, 3, 5, undefined, "2026-06-29", 2, 4), // PAR on pens (1-1)
      m(74, "R32", "SCHEDULED", "4:00 PM", FRA, SWE, null, null, undefined, "2026-06-30"),
      m(75, "R32", "FINISHED", "—", RSA, CAN, 0, 1, undefined, "2026-06-28"),
      m(76, "R32", "FINISHED", "—", NED, MAR, 4, 5, undefined, "2026-06-29", 3, 4), // MAR on pens (1-1)
      m(77, "R32", "SCHEDULED", "6:00 PM", POR, CRO, null, null, undefined, "2026-07-02"),
      m(78, "R32", "SCHEDULED", "2:00 PM", ESP, AUT, null, null, undefined, "2026-07-02"),
      m(79, "R32", "SCHEDULED", "7:00 PM", USA, BIH, null, null, undefined, "2026-07-01"),
      m(80, "R32", "SCHEDULED", "3:00 PM", BEL, SEN, null, null, undefined, "2026-07-01"),
      m(81, "R32", "FINISHED", "—", BRA, JPN, 2, 1, undefined, "2026-06-29"),
      m(82, "R32", "SCHEDULED", "12:00 PM", CIV, NOR, null, null, undefined, "2026-06-30"),
      m(83, "R32", "SCHEDULED", "8:00 PM", MEX, ECU, null, null, undefined, "2026-06-30"),
      m(84, "R32", "SCHEDULED", "11:00 AM", ENG, COD, null, null, undefined, "2026-07-01"),
      m(85, "R32", "SCHEDULED", "1:00 PM", ARG, CPV, null, null, undefined, "2026-07-03"),
      m(86, "R32", "SCHEDULED", "5:00 PM", AUS, EGY, null, null, undefined, "2026-07-03"),
      m(87, "R32", "SCHEDULED", "10:00 PM", SUI, ALG, null, null, undefined, "2026-07-02"),
      m(88, "R32", "SCHEDULED", "8:30 PM", COL, GHA, null, null, undefined, "2026-07-03"),
    ];
    const TBD = T("", "");
    const future: WcMatch[] = [
      m(89, "R16", "SCHEDULED", "TBD", PAR, TBD, null, null, undefined, "2026-07-04"), // winner GER-PAR (left)
      m(90, "R16", "SCHEDULED", "TBD", CAN, MAR, null, null, undefined, "2026-07-04"), // winners RSA-CAN, NED-MAR (left)
      m(91, "R16", "SCHEDULED", "TBD", BRA, TBD, null, null, undefined, "2026-07-05"), // winner BRA-JPN (RIGHT)
      // remaining R16 are TBD; their official dates place them by date (left Jul6×2, right Jul5+Jul7×2).
      m(92, "R16", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-06"),
      m(93, "R16", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-06"),
      m(94, "R16", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-05"),
      m(95, "R16", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-07"),
      m(96, "R16", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-07"),
      m(97, "QF", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-09"),
      m(98, "QF", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-10"),
      m(99, "QF", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-11"),
      m(100, "QF", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-11"),
      m(101, "SF", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-14"),
      m(102, "SF", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-15"),
      m(104, "FINAL", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-19"),
    ];
    return { ...base, knockout: [...knockout, ...future], phase };
  }

  if (phase === "knockout") {
    // Full knockout: all 16 R32 finished (bracket order); the source has seeded all 8 R16 matches.
    // The BRA-CIV R16 (id 89, played, penalty shootout) sorts FIRST by id but is a RIGHT-bracket tie
    // — orderKnockout (#52) places it on the right by the tree, and seeds its winner into the QF.
    const TBD = T("", "");
    const W = (id: number, h: WcTeam, a: WcTeam, hs: number, as: number, date: string) =>
      m(id, "R32", "FINISHED", "—", h, a, hs, as, undefined, date);
    const r32 = [
      W(73, GER, PAR, 2, 1, "2026-06-29"), W(74, FRA, SWE, 3, 1, "2026-06-30"),
      W(75, RSA, CAN, 0, 1, "2026-06-28"), W(76, NED, MAR, 1, 0, "2026-06-29"),
      W(77, POR, CRO, 1, 0, "2026-07-02"), W(78, ESP, AUT, 3, 0, "2026-07-02"),
      W(79, USA, BIH, 2, 1, "2026-07-01"), W(80, BEL, SEN, 1, 0, "2026-07-01"),
      W(81, BRA, JPN, 2, 0, "2026-06-29"), W(82, CIV, NOR, 2, 0, "2026-06-30"),
      W(83, MEX, ECU, 1, 0, "2026-06-30"), W(84, ENG, COD, 2, 0, "2026-07-01"),
      W(85, ARG, CPV, 4, 0, "2026-07-03"), W(86, AUS, EGY, 1, 0, "2026-07-03"),
      W(87, SUI, ALG, 2, 1, "2026-07-02"), W(88, COL, GHA, 1, 0, "2026-07-03"),
    ];
    const r16 = [
      // RIGHT-bracket tie played, penalty shootout: 1-1, Brazil through 4-2 on pens (compact box).
      m(89, "R16", "FINISHED", "—", BRA, CIV, 5, 3, undefined, "2026-07-05", 4, 2), // slot4 (right)
      m(90, "R16", "SCHEDULED", "TBD", GER, FRA, null, null, undefined, "2026-07-04"), // slot0 (left)
      m(91, "R16", "SCHEDULED", "TBD", CAN, NED, null, null, undefined, "2026-07-04"), // slot1 (left)
      m(92, "R16", "SCHEDULED", "TBD", POR, ESP, null, null, undefined, "2026-07-06"), // slot2 (left)
      m(93, "R16", "SCHEDULED", "TBD", USA, BEL, null, null, undefined, "2026-07-06"), // slot3 (left)
      m(94, "R16", "SCHEDULED", "TBD", MEX, ENG, null, null, undefined, "2026-07-05"), // slot5 (right)
      m(95, "R16", "SCHEDULED", "TBD", ARG, AUS, null, null, undefined, "2026-07-07"), // slot6 (right)
      m(96, "R16", "SCHEDULED", "TBD", SUI, COL, null, null, undefined, "2026-07-07"), // slot7 (right)
    ];
    const rest = [
      m(97, "QF", "SCHEDULED", "TBD", BRA, TBD, null, null, undefined, "2026-07-11"), // BRA advanced (R-QF-0, Jul 11)
      m(98, "QF", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-09"),
      m(99, "QF", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-10"),
      m(100, "QF", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-11"),
      m(101, "SF", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-14"),
      m(102, "SF", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-15"),
      m(104, "FINAL", "SCHEDULED", "TBD", TBD, TBD, null, null, undefined, "2026-07-19"),
    ];
    return { ...base, knockout: [...r32, ...r16, ...rest], phase };
  }

  // champion
  const final = m(60, "FINAL", "FINISHED", "—", BRA, ENG, 2, 1);
  return { ...base, knockout: [final], champion: BRA, phase };
}
