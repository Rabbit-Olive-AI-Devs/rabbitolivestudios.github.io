/**
 * The castaway page's story engine (DECISIONS #93). SPOILERS: keep it a surprise for the owner.
 *
 * Everything follows from the Chicago date and time: the day number since the castaway arrived,
 * the storyline it is in, the moment of the day, the light. No AI, no KV writes. A few things are
 * read from what the Worker already caches (home weather, tonight's planet, a Flamengo match) and
 * passed in as `CastawayContext`; without them the page still draws, with defaults.
 */

import type { Light, Prop, Scene, Weather } from "./hud/castaway-draw";
import { ARCS, ARRIVAL, DATED_EGGS, DATED_SAND, DAY_EGGS, MILESTONES, RARE_EGGS, ROUTINE, SETTINGS } from "./castaway-story";
import type { Arc, Beat } from "./castaway-story";
import { seededRandom, shuffledIndices } from "./art-styles";
import { holidayWeek } from "./holidays";
import { getBirthdayToday } from "./birthday";
import { getMoonPhase } from "./moon";
import { pickWord } from "./hud/aurebesh-words";
import { shiftDateStr } from "./date-utils";
import type { Env } from "./types";
import { flamengoCacheKey, planetCacheKey } from "./cache-keys";
import { FLAMENGO_ID } from "./flamengo";
import type { FlaMatch, FlaState } from "./flamengo";

/** Day 1: the day the page went live (the owner may first see it that evening). */
export const CASTAWAY_EPOCH = "2026-10-17";

export interface CastawayMatch {
  opponent: string;        // the other club's short name
  kickoff: number;         // ms since the epoch
  done: boolean;
  us: number | null;       // Flamengo's goals once known
  them: number | null;
}

export interface CastawayContext {
  weather?: Weather;       // home weather right now
  sunrise?: number;        // minutes after local midnight
  sunset?: number;
  planet?: string;         // tonight's planet of the day
  match?: CastawayMatch | null;
}

export interface CastawayView {
  day: number;
  clock: string;           // HH:MM, Chicago
  chapter: string;
  text: string;
  log: Array<{ time: string; text: string }>;
  setting: { name: string; value: number };
  battery: number;
  scene: Scene;
}

interface Moment { min: number; beat: Beat }

/** Chicago calendar date and minutes after midnight for an instant. */
export function chicagoParts(now: Date): { date: string; min: number } {
  const p = Object.fromEntries(new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(now).map((x) => [x.type, x.value]));
  return { date: `${p.year}-${p.month}-${p.day}`, min: (Number(p.hour) % 24) * 60 + Number(p.minute) };
}

export function dayNumber(date: string): number {
  const ms = (d: string) => Date.UTC(Number(d.slice(0, 4)), Number(d.slice(5, 7)) - 1, Number(d.slice(8, 10)));
  return Math.round((ms(date) - ms(CASTAWAY_EPOCH)) / 86_400_000) + 1;
}

/** Which storyline a day belongs to, and which of its days. Days before the epoch replay the arrival. */
export function arcFor(day: number): { arc: Arc; index: number } {
  if (day <= ARRIVAL.days.length) return { arc: ARRIVAL, index: Math.max(0, day - 1) };
  for (const m of MILESTONES) if (day >= m.from && day < m.from + m.arc.days.length) return { arc: m.arc, index: day - m.from };
  // Storyline days: every day after the arrival that is not a milestone day.
  const milestoneDaysBefore = MILESTONES.reduce((n, m) => n + Math.max(0, Math.min(m.arc.days.length, day - m.from)), 0);
  let left = day - ARRIVAL.days.length - 1 - milestoneDaysBefore;
  let last = -1;
  for (let cycle = 0; cycle < 10_000; cycle++) {
    // Each cycle walks every storyline once, in a fresh order; no back-to-back repeat across cycles.
    const order = shuffledIndices(ARCS.length, seededRandom(0x5e7e + cycle * 7919));
    if (order[0] === last && order.length > 1) [order[0], order[1]] = [order[1], order[0]];
    last = order[order.length - 1];
    for (const i of order) {
      const len = ARCS[i].days.length;
      if (left < len) return { arc: ARCS[i], index: left };
      left -= len;
    }
  }
  return { arc: ARCS[0], index: 0 };
}

// Rough Chicago sunrise and sunset by month (local clock time, minutes) for when the weather
// cache has none.
const SUN: ReadonlyArray<[number, number]> = [
  [435, 1005], [410, 1040], [420, 1140], [380, 1170], [335, 1200], [315, 1230],
  [325, 1225], [355, 1195], [390, 1140], [420, 1095], [400, 995], [430, 985],
];

export function lightAt(min: number, sunrise: number, sunset: number): Light {
  if (Math.abs(min - sunrise) <= 40) return "dawn";
  if (Math.abs(min - sunset) <= 40) return "dusk";
  return min > sunrise && min < sunset ? "day" : "night";
}

const hhmm = (min: number) => `${String(Math.floor(min / 60) % 24).padStart(2, "0")}:${String(min % 60).padStart(2, "0")}`;
const pickFrom = <T,>(pool: readonly T[], date: string, salt: number): T => pool[Math.floor(seededRandom(hash(date) + salt)() * pool.length)];

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

const beat = (text: string, log: string, pose: Beat["pose"], face: Beat["face"], props: Prop[] = [], at = 0.4): Beat => ({ text, log, pose, face, props, at });

/** The day's moments in time order: routine, the storyline's three beats, and what the calendar adds. */
export function dayTimeline(date: string, ctx: CastawayContext): Moment[] {
  const day = dayNumber(date);
  const month = Number(date.slice(5, 7));
  const [sunrise, sunset] = [ctx.sunrise ?? SUN[month - 1][0], ctx.sunset ?? SUN[month - 1][1]];
  const { arc, index } = arcFor(day);
  const [b0, b1, b2] = arc.days[index];
  const setting = settingFor(date);
  const r = (key: keyof typeof ROUTINE, salt: number) => pickFrom(ROUTINE[key], date, salt);

  const t: Moment[] = [
    { min: 0, beat: r("night", 1) },
    { min: sunrise, beat: beat(`DAY ${day}. Battery ${40 + (hash(date) % 25)}%. ${setting.name} setting: ${setting.value}%. All systems ready. Mostly.`, "Morning log.", "charge", "happy", [], 0.55) },
    { min: sunrise + 75, beat: r("morning", 2) },
    { min: 600, beat: b0 },
    { min: 735, beat: r("noon", 3) },
    { min: 825, beat: r("afternoon", 4) },
    { min: 930, beat: b1 },
    { min: Math.min(1020, sunset - 75), beat: r("afternoon", 5) },
    { min: sunset - 20, beat: r("dusk", 6) },
    { min: Math.max(1155, sunset + 50), beat: b2 },
    { min: 1260, beat: ctx.planet
      ? beat(`Stargazing. SETE finds the star of ${ctx.planet.toUpperCase()}, tonight's planet, and waves at it. Just in case.`, "Waved at a planet.", "sit", "love", day > 12 ? ["telescope"] : [], 0.45)
      : r("evening", 7) },
    { min: 1350, beat: r("night", 8) },
  ];

  // Day 1 starts when SETE washes up: nothing happened on the island before that.
  if (day === 1) {
    const from = t.findIndex((m) => m.beat === b0);
    t.splice(0, from + 1, { min: 0, beat: b0 });
  }

  const birthday = getBirthdayToday(month, Number(date.slice(8, 10)));
  if (birthday) {
    const name = birthday.name.toUpperCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
    t.push({ min: 510, beat: beat(`A bottle washes up with a note: 'TODAY IS ${name}'S BIRTHDAY.' SETE bakes a sand cake. Capitao adds candles.`, `Cake for ${name}!`, "dance", "love", ["cake", "bottle", "crab"], 0.55) });
    t.push({ min: 1200, beat: beat(`SETE sings happy birthday to ${name} across the sea. The far light blinks along, slightly off-key.`, `Sang for ${name}.`, "wave", "love", ["cake", "flash", "fire"], 0.55) });
  }

  // Easter eggs: dated ones every year, one for a day number, and now and then a rare one.
  for (const e of DATED_EGGS) if (date.slice(5) === e.md) t.push({ min: e.min, beat: e.beat });
  for (const e of DAY_EGGS) if (day === e.day) t.push({ min: e.min, beat: e.beat });
  if (day > 3 && hash(date + "egg") % 14 === 0) t.push({ min: 990, beat: pickFrom(RARE_EGGS, date, 9) });

  const week = holidayWeek(date);
  if (week) t.push(...holidayMoments(week.holiday.id, week.isDay));

  if (ctx.match) t.push(...matchMoments(ctx.match, date));

  return t.sort((a, b) => a.min - b.min);
}

function holidayMoments(id: string, isDay: boolean): Moment[] {
  const m = (min: number, b: Beat): Moment => ({ min, beat: b });
  switch (id) {
    case "halloween": return isDay
      ? [m(660, beat("Halloween! SETE carves a coconut lantern and wears a sheet. It is now a ghost robot. Capitao is unimpressed.", "Ghost robot mode.", "dance", "wink", ["pumpkin", "crab"], 0.55)),
         m(1200, beat("Trick or treat across the water. The far light flashes back something spooky. Probably 'BOO'.", "The Neighbour said BOO.", "wave", "surprised", ["pumpkin", "flash", "fire"], 0.55))]
      : [m(660, beat("A pumpkin washed up! SETE practises scary faces on its screen for the big night. This one is mostly cute.", "Practising scary faces.", "stand", "wink", ["pumpkin"], 0.55))];
    case "christmas": return isDay
      ? [m(540, beat("Christmas morning. Under the palm: a present from the sea, wrapped in seaweed. Inside: one perfect bolt.", "A bolt for Christmas!", "dance", "love", ["lights"], 0.5)),
         m(1200, beat("SETE flashes 'MERRY CHRISTMAS' to the far light. It takes 52 minutes. Worth every one.", "Merry Christmas sent.", "wave", "love", ["lights", "flash", "fire"], 0.55))]
      : [m(660, beat("SETE decorates the palm with shells and a string of glowing plankton. Tasteful. Festive. Damp.", "Decorated the palm.", "stand", "happy", ["lights"], 0.45))];
    case "new_year": return isDay
      ? [m(5, beat("Midnight! A new year! SETE launches its only firework: a very enthusiastic jump.", "HAPPY NEW YEAR!", "dance", "love", ["stars-shoot", "fire"], 0.55))]
      : [m(660, beat("Resolution list: 1. Deliver the parcel. 2. Beat the keepy-uppy record. 3. Forgive Larapio (maybe).", "New Year plans.", "think", "focused", [], 0.5))];
    case "carnaval": return [m(isDay ? 900 : 660, beat("Carnaval! SETE leads a parade of one robot and one crab around the island. Samba at 4 slabs per beat.", "Carnaval parade!", "dance", "love", ["crab", "flag"], 0.55))];
    case "festa_junina": return [m(1170, beat("Festa Junina: a little bonfire, paper flags on a vine, and SETE dancing quadrilha with Capitao.", "Quadrilha with crab.", "dance", "happy", ["fire", "flag", "crab"], 0.5))];
    case "thanksgiving": return [m(isDay ? 1080 : 660, beat("SETE lists what it is thankful for: the palm, Capitao, the far light, and 21 bolts. 20. Larapio was here.", "Thankful list.", "sit", "love", ["crab", "fire"], 0.5))];
    case "easter": return isDay
      ? [m(600, beat("Easter: SETE hides painted shells all over the island for Capitao, then forgets where. Hunt ongoing.", "Shell hunt (lost).", "dig", "wink", ["crab"], 0.6))]
      : [m(660, beat("Holy Week. A bottle brings a recipe for torta capixaba: seafood, palm hearts, olives, eggs. SETE has sand. It tries.", "Torta capixaba (sand).", "dig", "focused", ["fire"], 0.55))];
    case "independence_day": return [m(1230, beat("Fourth of July: SETE watches distant fireworks over the horizon and claps its slabs together.", "Fireworks far away.", "wave", "love", ["stars-shoot"], 0.55))];
    default: return [];
  }
}

/** Kick-off nerves, listening on the radio, then the reaction. SETE has no idea why it cares. */
function matchMoments(match: CastawayMatch, date: string): Moment[] {
  const k = chicagoParts(new Date(match.kickoff));
  if (k.date !== date) return [];
  const m = (min: number, b: Beat): Moment => ({ min, beat: b });
  const out: Moment[] = [
    m(k.min - 30, beat(`The radio crackles: Flamengo vs ${match.opponent} today. SETE does not know why its stripe tingles.`, "Match day. Stripe tingles.", "sit", "focused", ["radio"], 0.5)),
    m(k.min + 5, beat(`Kick-off. SETE holds the radio to its screen and does not move. Capitao is told to be quiet. He is.`, "Listening. Do not disturb.", "sit", "focused", ["radio", "crab"], 0.5)),
  ];
  if (match.done && match.us !== null && match.them !== null) {
    const score = `${match.us}-${match.them}`;
    const end = k.min + 115;
    out.push(match.us > match.them
      ? m(end, beat(`FINAL: Flamengo ${score}! SETE runs three laps of the island waving both slabs. Larapio flees.`, `MENGO! ${score}`, "dance", "love", ["radio", "flag"], 0.55))
      : match.us === match.them
        ? m(end, beat(`Final: ${score}. A draw. SETE sits very still for a while, then decides it was a moral victory.`, `Draw ${score}.`, "sit", "calm", ["radio"], 0.5))
        : m(end, beat(`Final: ${score}. SETE turns the radio off and looks at the sea. Next time. There is always a next time.`, `Lost ${score}. Next time.`, "sit", "sad", ["radio"], 0.5)));
  }
  return out;
}

export function settingFor(date: string): { name: string; value: number } {
  const h = hash(date + "setting");
  return { name: SETTINGS[h % SETTINGS.length], value: 55 + ((h >>> 8) % 41) };
}

/** Index of the moment showing at minute `min`: the last one that has started. */
const momentIndex = (today: Moment[], min: number) => today.reduce((acc, m, i) => (m.min <= min ? i : acc), 0);

/** The UTC instant of a Chicago date and clock minute (the earlier one when the autumn hour repeats). */
function chicagoInstant(date: string, min: number): Date {
  for (const offsetH of [5, 6]) {
    const t = new Date(Date.UTC(Number(date.slice(0, 4)), Number(date.slice(5, 7)) - 1, Number(date.slice(8, 10)), offsetH, min));
    const p = chicagoParts(t);
    if (p.date === date && p.min === min) return t;
  }
  // The spring-forward hour does not exist on the clock: count from midnight in standard time.
  return new Date(Date.UTC(Number(date.slice(0, 4)), Number(date.slice(5, 7)) - 1, Number(date.slice(8, 10)), 6, min));
}

/**
 * The story moment showing at an instant (DECISIONS #95): an opaque key, the date and the clock
 * time it started, and when it started. Same choice as castawayView, so the key changes exactly
 * when the story moves on. The context matters: sunrise, sunset and a match day move moments.
 */
export function castawayMoment(now: Date, ctx: CastawayContext = {}): { key: string; start: Date } {
  const { date, min } = chicagoParts(now);
  const today = dayTimeline(date, ctx);
  const start = Math.max(0, today[momentIndex(today, min)].min);
  return { key: `${date}T${hhmm(start)}`, start: chicagoInstant(date, start) };
}

/** The page at an instant. */
export function castawayView(now: Date, ctx: CastawayContext = {}): CastawayView {
  const { date, min } = chicagoParts(now);
  const day = dayNumber(date);
  const month = Number(date.slice(5, 7));
  const [sunrise, sunset] = [ctx.sunrise ?? SUN[month - 1][0], ctx.sunset ?? SUN[month - 1][1]];
  const today = dayTimeline(date, ctx);
  const idx = momentIndex(today, min);
  const current = today[idx];
  const yesterday = day > 1 ? dayTimeline(shiftDateStr(date, -1), ctx).map((m) => ({ ...m, min: m.min - 1440 })) : [];
  const earlier = [...yesterday, ...today.slice(0, idx)];
  const log = earlier.slice(-2).reverse().map((m) => ({ time: hhmm((m.min + 1440) % 1440), text: m.beat.log }));
  const { arc } = arcFor(day);
  const light = lightAt(min, sunrise, sunset);
  const asleep = current.beat.pose === "sleep";
  const props: Prop[] = [...(current.beat.props ?? [])];
  // At night the far light keeps blinking now and then, and a fire burns before bed.
  if (light === "night" && !props.includes("flash") && hash(date + min) % 3 === 0) props.push("flash");
  const battery = Math.max(5, Math.min(100, min < sunrise ? 60 - Math.round(min / 30) : min < 735 ? 45 + Math.round((min - sunrise) / 8) : 100 - Math.round((min - 735) / 18)));

  return {
    day,
    clock: hhmm(min),
    chapter: arc.title,
    text: current.beat.text,
    log,
    setting: settingFor(date),
    battery,
    scene: {
      light,
      weather: ctx.weather ?? (hash(date) % 7 === 0 ? "cloudy" : "clear"),
      moon: getMoonPhase(now).index,
      pose: current.beat.pose,
      face: asleep ? "sleep" : current.beat.face,
      at: current.beat.at ?? 0.4,
      props,
      hut: day < 4 ? 0 : Math.min(3, 1 + Math.floor((day - 4) / 5)),
      antenna: day < 3 ? 0 : Math.min(3, 1 + Math.floor((day - 3) / 8)),
      sand: light === "day" && min < sunset - 60 ? DATED_SAND[date.slice(5)] ?? pickWord(date) : undefined,
      seed: hash(date) ^ Math.floor(min / 15),
    },
  };
}

/** WMO weather code (Open-Meteo) to the island's weather. Snow at home is rain on the island. */
export function islandWeather(code: number): Weather {
  if (code >= 95) return "storm";
  if ((code >= 51 && code <= 67) || (code >= 71 && code <= 86)) return "rain";
  if (code >= 2) return "cloudy";
  return "clear";
}

/** "2026-10-17T07:04" (local) to minutes after midnight, or undefined. */
function clockMinutes(iso: unknown): number | undefined {
  const m = typeof iso === "string" ? /T(\d{2}):(\d{2})/.exec(iso) : null;
  return m ? Number(m[1]) * 60 + Number(m[2]) : undefined;
}

/** A Flamengo match on the given Chicago date, from the Flamengo board's cache. */
export function matchOn(state: FlaState, date: string): CastawayMatch | null {
  const all: FlaMatch[] = [...state.results, ...state.fixtures];
  const m = all.find((x) => chicagoParts(new Date(x.at)).date === date);
  if (!m) return null;
  const home = m.home.id === FLAMENGO_ID;
  const other = home ? m.away : m.home;
  return {
    opponent: other.abbr || other.name, kickoff: m.at, done: m.done,
    us: home ? m.homeScore : m.awayScore, them: home ? m.awayScore : m.homeScore,
  };
}

/** What the page borrows from the Worker's caches; reads only, every piece optional. */
export async function loadCastawayContext(env: Env, now: Date): Promise<CastawayContext> {
  const { date } = chicagoParts(now);
  const json = async <T,>(key: string): Promise<T | null> => {
    try { return await env.CACHE.get<T>(key, "json"); } catch { return null; }
  };
  const [weather, planet, fla] = await Promise.all([
    json<{ data?: { current?: { condition?: { code?: number } }; sunrise?: string; daily_5d?: Array<{ sunset?: string }> } }>("weather:60540:v2"),
    json<{ planet?: { name?: string } }>(planetCacheKey(date)),
    json<FlaState>(flamengoCacheKey()),
  ]);
  const code = weather?.data?.current?.condition?.code;
  return {
    weather: typeof code === "number" ? islandWeather(code) : undefined,
    sunrise: clockMinutes(weather?.data?.sunrise),
    sunset: clockMinutes(weather?.data?.daily_5d?.[0]?.sunset),
    planet: typeof planet?.planet?.name === "string" ? planet.planet.name : undefined,
    match: fla && Array.isArray(fla.results) && Array.isArray(fla.fixtures) ? matchOn(fla, date) : null,
  };
}
