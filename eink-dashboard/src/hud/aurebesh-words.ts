/**
 * Word of the day for the Aurebesh page: the list, its themes, the picker, and the cleaning
 * of the free test parameters. English, A–Z only, 3 to 9 letters so a word fits
 * one line of large glyphs.
 */

export const MAX_WORD_LEN = 9;

export const WORDS: readonly string[] = [
  "FORCE", "JEDI", "REBEL", "EMPIRE", "GALAXY", "PLANET", "DROID", "PILOT", "SABER", "MASTER",
  "PADAWAN", "KNIGHT", "TEMPLE", "COUNCIL", "SENATE", "ALLIANCE", "CRUISER", "FIGHTER", "FALCON", "SHUTTLE",
  "HANGAR", "CANTINA", "SMUGGLER", "BOUNTY", "HUNTER", "WOOKIEE", "EWOK", "JAWA", "TUSKEN", "BANTHA",
  "TAUNTAUN", "RANCOR", "SARLACC", "KRAYT", "PORG", "LOTHCAT", "ENDOR", "HOTH", "DAGOBAH", "TATOOINE",
  "NABOO", "BESPIN", "JAKKU", "ALDERAAN", "KASHYYYK", "MUSTAFAR", "SCARIF", "CRAIT", "YAVIN", "KESSEL",
  "CORELLIA", "MANDALORE", "KAMINO", "GEONOSIS", "UTAPAU", "FELUCIA", "LOTHAL", "DATHOMIR", "ILUM", "EXEGOL",
  "HOPE", "COURAGE", "PATIENCE", "TRUST", "BALANCE", "DESTINY", "WISDOM", "HONOR", "LOYALTY", "FREEDOM",
  "PEACE", "JUSTICE", "KINDNESS", "FRIEND", "FAMILY", "BROTHER", "SISTER", "FATHER", "MOTHER", "CHILD",
  "STAR", "MOON", "COMET", "NEBULA", "ORBIT", "ROCKET", "ENGINE", "SHIELD", "LASER", "BLASTER",
  "CRYSTAL", "KYBER", "HOLOCRON", "HOLOGRAM", "BEACON", "SIGNAL", "CIPHER", "SECRET", "MISSION", "SQUADRON",
  "COMMAND", "CAPTAIN", "ADMIRAL", "GENERAL", "TROOPER", "SCOUT", "RANGER", "GUARDIAN", "SENTINEL", "WARRIOR",
  "DESERT", "FOREST", "SWAMP", "CANYON", "GLACIER", "VOLCANO", "ASTEROID", "HORIZON", "SUNSET", "TWILIGHT",
  "LUKE", "LEIA", "HAN", "CHEWIE", "YODA", "VADER", "KENOBI", "ANAKIN", "PADME", "AHSOKA",
  "REY", "FINN", "POE", "KYLO", "LANDO", "BOBA", "JABBA", "GROGU", "MANDO", "ARTOO",
  "THREEPIO", "PALPATINE", "DOOKU", "MAUL", "QUIGON", "WINDU", "SOLO", "SKYWALKER", "ORGANA", "AMIDALA",
  "TARKIN", "GRIEVOUS", "EZRA", "HERA", "SABINE", "KANAN", "ZEB", "THRAWN", "CASSIAN", "JYN",
  "ANDOR", "CHIRRUT", "BODHI", "KRENNIC", "HUX", "PHASMA", "SNOKE", "ROSE", "HOLDO", "ACKBAR",
  "WEDGE", "BIGGS", "GREEDO", "WATTO", "JARJAR", "BAZE", "SAW", "CHOPPER", "REX", "CODY",
];

/** Where each themed run of WORDS starts. Kept next to the list so they change together. */
const GROUPS: readonly (readonly [number, string])[] = [
  [0, "CORE"], [10, "ORDER"], [16, "FLEET"], [21, "FRINGE"], [25, "CREATURES"], [36, "PLANETS"],
  [60, "VIRTUES"], [73, "KIN"], [80, "SPACE"], [85, "TECH"], [90, "LORE"], [94, "SIGNALS"],
  [98, "RANKS"], [110, "TERRAIN"], [120, "CAST"],
];

/** Theme of the word at `index` in WORDS; "CUSTOM" for a word that is not in the list. */
export function groupOf(index: number): string {
  if (index < 0 || index >= WORDS.length) return "CUSTOM";
  let name = GROUPS[0][1];
  for (const [start, group] of GROUPS) if (index >= start) name = group;
  return name;
}

/** The Aurebesh name of each letter. */
export const LETTER_NAMES: Readonly<Record<string, string>> = {
  A: "AUREK", B: "BESH", C: "CRESH", D: "DORN", E: "ESK", F: "FORN", G: "GREK", H: "HERF", I: "ISK",
  J: "JENTH", K: "KRILL", L: "LETH", M: "MERN", N: "NERN", O: "OSK", P: "PETH", Q: "QEK", R: "RESH",
  S: "SENTH", T: "TRILL", U: "USK", V: "VEV", W: "WESK", X: "XESH", Y: "YIRT", Z: "ZEREK",
};

/** Uppercase A–Z only, at most MAX_WORD_LEN letters; null when nothing usable remains. */
export function cleanWord(raw: string | null): string | null {
  if (!raw) return null;
  const letters = raw.toUpperCase().replace(/[^A-Z]/g, "").slice(0, MAX_WORD_LEN);
  return letters.length > 0 ? letters : null;
}

/** A real calendar date as YYYY-MM-DD, or null. */
export function parseDateParam(raw: string | null): string | null {
  if (!raw || !/^\d{4}-\d{2}-\d{2}$/.test(raw)) return null;
  const [y, m, d] = raw.split("-").map(Number);
  const t = new Date(Date.UTC(y, m - 1, d));
  const real = t.getUTCFullYear() === y && t.getUTCMonth() === m - 1 && t.getUTCDate() === d;
  return real ? raw : null;
}

/** 1 for January 1st through 365, or 366 in a leap year. Expects a valid YYYY-MM-DD. */
export function dayOfYear(dateStr: string): number {
  const [y, m, d] = dateStr.split("-").map(Number);
  return Math.round((Date.UTC(y, m - 1, d) - Date.UTC(y, 0, 1)) / 86400000) + 1;
}

/**
 * WORDS is grouped by theme, so stepping through it one at a time would show
 * twenty planet names in a row. A stride that shares no factor with the list
 * length jumps between themes while still visiting every word once per cycle.
 * 77 = 7 x 11, so keep the list length free of those two factors; a test
 * checks that every word still appears once per cycle.
 */
const STRIDE = 77;

/** The same date always gives the same word; the year offsets the cycle so years differ. */
export function pickWord(dateStr: string): string {
  const year = Number(dateStr.slice(0, 4));
  return WORDS[(dayOfYear(dateStr) * STRIDE + year * 7) % WORDS.length];
}

/** A day on which something other than the daily word is shown. */
export interface Occasion {
  word: string;
  /** Short line shown as a tag on the page. */
  tag: string;
  group: string;
  /** A number worth showing in place of the word's position, such as an age. */
  count?: { label: string; value: number; unit: string };
}

/** Fixed dates, as MM-DD. `since` turns into a count of years. */
const SPECIAL_DAYS: Readonly<Record<string, { word: string; tag: string; since?: number; label?: string }>> = {
  "01-01": { word: "HOPE", tag: "A NEW HOPE" },
  "05-04": { word: "FORCE", tag: "MAY THE FOURTH" },
  "05-05": { word: "SITH", tag: "REVENGE OF THE FIFTH" },
  "05-25": { word: "STARWARS", tag: "FIRST FILM 1977", since: 1977, label: "SAGA" },
  "07-04": { word: "REBEL", tag: "INDEPENDENCE DAY" },
  "09-07": { word: "BRASIL", tag: "SETE DE SETEMBRO" },
  "10-31": { word: "VADER", tag: "HALLOWEEN" },
  "11-07": { word: "MURPH", tag: "STAY" },
  "11-15": { word: "FLAMENGO", tag: "FOUNDED 1895", since: 1895, label: "CLUB" },
  "11-17": { word: "LIFEDAY", tag: "WOOKIEE LIFE DAY" },
  "12-25": { word: "NATAL", tag: "FELIZ NATAL" },
};

/** The occasion for a fixed special date, or null on an ordinary day. */
export function specialDay(dateStr: string): Occasion | null {
  const day = SPECIAL_DAYS[dateStr.slice(5)];
  if (!day) return null;
  const occasion: Occasion = { word: day.word, tag: day.tag, group: "SPECIAL" };
  if (day.since) {
    occasion.count = { label: day.label ?? "SINCE", value: Number(dateStr.slice(0, 4)) - day.since, unit: "YEARS" };
  }
  return occasion;
}
