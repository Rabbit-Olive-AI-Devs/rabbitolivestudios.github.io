/**
 * Holiday weeks (DECISIONS #91). For the 7 days up to and including each holiday, the daily
 * pictures take on its theme:
 *   - the skyline's city is dressed for the holiday, and its caption names the week;
 *   - the moment keeps the day's real event, with a mood line leaning the holiday's way;
 *   - on the holiday itself the moment is the holiday's own history instead of the day's event.
 * Birthday portraits are never themed. Pure: everything follows from the date.
 */

import type { MomentBeforeData } from "./types";

export interface Holiday {
  id: string;
  name: string;
  /** The holiday's date in a year, as [month 1-12, day]. */
  date: (year: number) => [number, number];
  /** What the skyline's city is dressed in that week. No snow: half the cities are in the south. */
  skyline: string;
  /** Replaces the skyline's usual golden-hour light. */
  lighting: string;
  /** Added to the moment's scene on the six days before the holiday. */
  mood: string;
  /** The moment on the holiday itself: a documented event from the holiday's history. */
  history: MomentBeforeData;
}

/** Easter Sunday (Gregorian, the anonymous algorithm). */
export function easter(year: number): [number, number] {
  const a = year % 19, b = Math.floor(year / 100), c = year % 100;
  const d = Math.floor(b / 4), e = b % 4, f = Math.floor((b + 8) / 25);
  const g = Math.floor((b - f + 1) / 3), h = (19 * a + b - d - g + 15) % 30;
  const i = Math.floor(c / 4), k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7;
  const m = Math.floor((a + 11 * h + 22 * l) / 451);
  const month = Math.floor((h + l - 7 * m + 114) / 31);
  return [month, ((h + l - 7 * m + 114) % 31) + 1];
}

const DAY_MS = 86_400_000;
const dayNumber = (y: number, m: number, d: number): number => Date.UTC(y, m - 1, d) / DAY_MS;

function fromDayNumber(n: number): [number, number] {
  const dt = new Date(n * DAY_MS);
  return [dt.getUTCMonth() + 1, dt.getUTCDate()];
}

/** Carnaval Tuesday, 47 days before Easter. */
function carnaval(year: number): [number, number] {
  const [m, d] = easter(year);
  return fromDayNumber(dayNumber(year, m, d) - 47);
}

/** US Thanksgiving: the fourth Thursday of November. */
function thanksgiving(year: number): [number, number] {
  const firstWeekday = new Date(Date.UTC(year, 10, 1)).getUTCDay();
  return [11, 1 + ((4 - firstWeekday + 7) % 7) + 21];
}

export const HOLIDAYS: readonly Holiday[] = [
  {
    id: "carnaval", name: "Carnaval", date: carnaval,
    skyline: "the city in Carnaval: streets full of costumed revellers and samba dancers, confetti and streamers in the air, feathered headdresses, a parade float, colourful bunting strung between buildings",
    lighting: "warm festive evening light with the sky still glowing",
    mood: "a festive, exuberant energy in the colours and light",
    history: {
      year: 1932, title: "First Samba School Parade", location: "Praça Onze, Rio de Janeiro, Brazil",
      scene: "The samba schools of Rio parade through Praça Onze in their first official contest, drummers and dancers filling the square.",
      imagePrompt: "Rio de Janeiro's Praça Onze in 1932, the first official samba school parade: rows of drummers with surdos and tamborins, dancers in 1930s Carnival costumes and banners, crowds in period clothing packed along the square, early-twentieth-century buildings and tram wires. Cinematic wide composition, warm evening light, joyful energy.",
    },
  },
  {
    id: "easter", name: "Easter", date: easter,
    skyline: "the city dressed for Easter: spring blossoms on the trees, flower garlands, painted Easter eggs in window displays, families strolling in a park",
    lighting: "soft fresh spring morning light",
    mood: "a fresh spring light and a gentle, hopeful atmosphere",
    history: {
      year: 1722, title: "Easter Island Sighted", location: "Rapa Nui (Easter Island), Pacific Ocean",
      scene: "On Easter Sunday the Dutch ships of Jacob Roggeveen come upon an unknown island lined with giant stone statues.",
      imagePrompt: "Three Dutch East India sailing ships of 1722 anchored off a remote volcanic island, a longboat rowing toward the shore, giant moai stone statues standing along the grassy coast, Rapa Nui islanders watching from the rocks. Cinematic wide composition, early morning light over the Pacific, a sense of wonder.",
    },
  },
  {
    id: "festa_junina", name: "Festa Junina", date: () => [6, 24],
    skyline: "the city dressed for Festa Junina: strings of colourful paper flags (bandeirinhas) criss-crossing the streets and squares, a big bonfire, paper lanterns, people in checked shirts and straw hats dancing quadrilha",
    lighting: "a cool June evening lit by bonfires and lanterns",
    mood: "a warm, rustic festive glow, like bonfire light on a June night",
    history: {
      year: 1983, title: "First Maior São João do Mundo", location: "Campina Grande, Paraíba, Brazil",
      scene: "Campina Grande holds its first great São João festival, a whole park strung with flags, bonfires and couples dancing forró.",
      imagePrompt: "A huge open-air São João festival in Campina Grande, northeast Brazil, in 1983: a park strung with thousands of colourful paper bandeirinhas, a big bonfire, couples in checked shirts and straw hats dancing forró and quadrilha, a forró trio with accordion, zabumba and triangle on a wooden stage, 1980s clothing and stalls selling corn treats. Cinematic wide composition, night lit by bonfires and lanterns, joyful mood.",
    },
  },
  {
    id: "independence_day", name: "Fourth of July", date: () => [7, 4],
    skyline: "the city celebrating the Fourth of July: red, white and blue bunting on buildings, small flags along the streets, a fireworks display bursting over the skyline",
    lighting: "summer dusk with fireworks lighting the sky",
    mood: "a bright midsummer light and a celebratory air",
    history: {
      year: 1776, title: "Declaration of Independence Adopted", location: "Pennsylvania State House, Philadelphia",
      scene: "In the Assembly Room of the Pennsylvania State House, the Continental Congress adopts the Declaration of Independence.",
      imagePrompt: "The Assembly Room of the Pennsylvania State House in Philadelphia, July 1776: delegates of the Continental Congress in eighteenth-century coats and breeches gathered around green-covered tables, a committee presenting the document, tall arched windows letting in summer light, Windsor chairs and the speaker's dais. Cinematic composition, warm daylight, a solemn, momentous mood.",
    },
  },
  {
    id: "halloween", name: "Halloween", date: () => [10, 31],
    skyline: "the city dressed for Halloween: carved glowing jack-o'-lanterns on doorsteps and windowsills, orange-lit windows, wisps of low fog in the streets, bats crossing a big full moon, a few children in costumes trick-or-treating",
    lighting: "dusk turning to night under a big full moon",
    mood: "an autumn mood with a hint of Halloween eeriness in the light",
    history: {
      year: 1920, title: "First Halloween Parade", location: "Anoka, Minnesota",
      scene: "Anoka holds one of the first city Halloween celebrations in America: a parade of costumed children down Main Street.",
      imagePrompt: "Main Street of Anoka, Minnesota, on Halloween night 1920: a parade of children in homemade 1920s costumes (sheets, witches' hats, papier-mâché masks) carrying glowing carved pumpkins, a small brass band, townsfolk in period coats and hats lining the street, brick storefronts and Model T cars, autumn leaves underfoot. Cinematic wide composition, lantern and moonlight, playful spooky mood.",
    },
  },
  {
    id: "thanksgiving", name: "Thanksgiving", date: thanksgiving,
    skyline: "the city in late autumn for Thanksgiving: orange and red fallen leaves, pumpkins and harvest wreaths on doorsteps, warm lit windows of families gathering, a parade balloon floating between buildings",
    lighting: "late-autumn afternoon light, low and golden",
    mood: "a warm late-autumn light with harvest colours",
    history: {
      year: 1621, title: "The Plymouth Harvest Feast", location: "Plymouth Colony, Massachusetts",
      scene: "The Plymouth colonists and the Wampanoag under Massasoit share a three-day harvest celebration.",
      imagePrompt: "An autumn harvest gathering at Plymouth Colony in 1621: English colonists in plain seventeenth-century clothing and Wampanoag men led by Massasoit sharing food at long rough tables outdoors, cooking fires, venison and wildfowl, baskets of corn and squash, the small timber-framed village and palisade behind, autumn trees. Cinematic wide composition, golden autumn light, a calm, communal mood.",
    },
  },
  {
    id: "christmas", name: "Christmas", date: () => [12, 25],
    skyline: "the city dressed for Christmas: strings of warm festive lights along the streets and bridges, a big lit Christmas tree in a plaza, wreaths on doors, a Christmas market with lit stalls",
    lighting: "evening blue hour with the festive lights glowing",
    mood: "a warm festive glow, like lights in the dark of December",
    history: {
      year: 1914, title: "The Christmas Truce", location: "Western Front, Flanders, Belgium",
      scene: "On Christmas Day British and German soldiers climb out of their trenches and meet in no man's land.",
      imagePrompt: "No man's land on the Western Front in Flanders on Christmas Day 1914: British and German soldiers in period uniforms and caps meeting between the trenches, shaking hands and exchanging gifts, a small decorated Christmas tree on a trench parapet, frost on the churned ground and barbed wire, a few soldiers kicking a football. Cinematic wide composition, pale winter light, a quiet, hopeful mood.",
    },
  },
  {
    id: "new_year", name: "New Year", date: () => [1, 1],
    skyline: "the city celebrating New Year's Eve: a grand fireworks display bursting over the skyline and reflected in the water, crowds gathered on the waterfront, festive lights",
    lighting: "midnight, the sky lit by fireworks",
    mood: "a festive, sparkling atmosphere, as on the last night of the year",
    history: {
      year: 1908, title: "First Times Square Ball Drop", location: "Times Square, New York City",
      scene: "At midnight a lit iron-and-wood ball is lowered from the flagpole of One Times Square for the first time.",
      imagePrompt: "Times Square in New York at midnight on New Year's Eve 1907, the first ball drop: a glowing ball studded with light bulbs descending the flagpole atop the narrow Times Tower, crowds in heavy coats, bowler hats and long skirts filling the streets below, early electric signs, streetcars and horse-drawn cabs. Cinematic wide composition, night lit by electric light, jubilant mood.",
    },
  },
];

export interface HolidayWeek {
  holiday: Holiday;
  /** True on the holiday itself, false on the six days before. */
  isDay: boolean;
}

/** The holiday week a date falls in (the 7 days up to and including the holiday), or null. */
export function holidayWeek(dateStr: string): HolidayWeek | null {
  const [y, m, d] = dateStr.split("-").map(Number);
  const today = dayNumber(y, m, d);
  for (const holiday of HOLIDAYS) {
    // The holiday may fall in the next year (New Year's week starts on 26 Dec).
    for (const year of [y, y + 1]) {
      const [hm, hd] = holiday.date(year);
      const until = dayNumber(year, hm, hd) - today;
      if (until >= 0 && until <= 6) return { holiday, isDay: until === 0 };
    }
  }
  return null;
}

/**
 * The day's moment as the pictures should draw it: on a holiday the holiday's history replaces the
 * event; in the six days before, the event keeps its own scene with the holiday's mood added.
 */
export function holidayMoment(dateStr: string): MomentBeforeData | null {
  const week = holidayWeek(dateStr);
  return week?.isDay ? { ...week.holiday.history } : null;
}

/** Add the holiday week's mood to a scene prompt (unchanged outside holiday weeks and on the day). */
export function withHolidayMood(dateStr: string, scenePrompt: string): string {
  const week = holidayWeek(dateStr);
  return week && !week.isDay && scenePrompt ? `${scenePrompt} ${capitalise(week.holiday.mood)}.` : scenePrompt;
}

function capitalise(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
