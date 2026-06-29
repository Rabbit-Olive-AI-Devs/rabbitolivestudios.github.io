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
): WcMatch {
  return {
    id, stage, group, status,
    kickoffISO: `${dateChicago}T${id.toString().padStart(2, "0")}:00:00Z`,
    dateChicago, timeChicago: time,
    home, away, homeScore: hs, awayScore: as,
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
    // Full 16-match Round of 32 (one finished, the rest scheduled across the round).
    const knockout = [
      m(73, "R32", "FINISHED", "—", RSA, CAN, 0, 1, undefined, "2026-06-28"),
      m(74, "R32", "SCHEDULED", "12:00 PM", BRA, JPN, null, null, undefined, "2026-06-29"),
      m(75, "R32", "SCHEDULED", "3:30 PM", GER, PAR, null, null, undefined, "2026-06-29"),
      m(76, "R32", "SCHEDULED", "8:00 PM", NED, MAR, null, null, undefined, "2026-06-29"),
      m(77, "R32", "SCHEDULED", "12:00 PM", CIV, NOR, null, null, undefined, "2026-06-30"),
      m(78, "R32", "SCHEDULED", "4:00 PM", FRA, SWE, null, null, undefined, "2026-06-30"),
      m(79, "R32", "SCHEDULED", "8:00 PM", MEX, ECU, null, null, undefined, "2026-06-30"),
      m(80, "R32", "SCHEDULED", "11:00 AM", ENG, COD, null, null, undefined, "2026-07-01"),
      m(81, "R32", "SCHEDULED", "3:00 PM", BEL, SEN, null, null, undefined, "2026-07-01"),
      m(82, "R32", "SCHEDULED", "7:00 PM", USA, BIH, null, null, undefined, "2026-07-01"),
      m(83, "R32", "SCHEDULED", "2:00 PM", POR, CRO, null, null, undefined, "2026-07-02"),
      m(84, "R32", "SCHEDULED", "6:00 PM", ESP, AUT, null, null, undefined, "2026-07-02"),
      m(85, "R32", "SCHEDULED", "10:00 PM", SUI, ALG, null, null, undefined, "2026-07-02"),
      m(86, "R32", "SCHEDULED", "1:00 PM", ARG, CPV, null, null, undefined, "2026-07-03"),
      m(87, "R32", "SCHEDULED", "5:00 PM", COL, GHA, null, null, undefined, "2026-07-03"),
      m(88, "R32", "SCHEDULED", "8:30 PM", SRB, CMR, null, null, undefined, "2026-07-03"),
    ];
    return { ...base, knockout, phase };
  }

  if (phase === "knockout") {
    const r16 = [
      m(40, "R16", "FINISHED", "—", BRA, SUI, 2, 0), m(41, "R16", "FINISHED", "—", ARG, NED, 1, 0),
      m(42, "R16", "SCHEDULED", "1:00 PM", ENG, FRA, null, null), m(43, "R16", "SCHEDULED", "4:00 PM", ESP, POR, null, null),
    ];
    const qf = [m(50, "QF", "SCHEDULED", "—", BRA, ARG, null, null)];
    return { ...base, knockout: [...r16, ...qf], todayMatches: [r16[2], r16[3]], phase };
  }

  // champion
  const final = m(60, "FINAL", "FINISHED", "—", BRA, ENG, 2, 1);
  return { ...base, knockout: [final], champion: BRA, phase };
}
