/**
 * Birthday detection and art style rotation.
 *
 * On family birthday dates, /fact.png generates a portrait instead of
 * the regular "Moment Before" historical event.
 */

export interface BirthdayPerson {
  name: string;       // Display name (with accents)
  key: string;        // R2 photo key (ascii, lowercase)
  month: number;      // 1-12
  day: number;        // 1-31
  birthYear: number;
}

const BIRTHDAYS: BirthdayPerson[] = [
  { name: "Thiago",     key: "thiago",   month: 10, day: 20, birthYear: 1977 },
  { name: "Gilmara",    key: "gilmara",  month:  7, day: 26, birthYear: 1979 },
  { name: "João Pedro", key: "joaopedro", month: 10, day:  2, birthYear: 2012 },
  { name: "Lucas",      key: "lucas",    month: 10, day: 10, birthYear: 2014 },
  { name: "Sônia",      key: "sonia",    month: 10, day: 13, birthYear: 1953 },
  { name: "Álvaro",     key: "alvaro",   month: 10, day: 31, birthYear: 1948 },
  { name: "Mariana",    key: "mariana",  month:  1, day:  7, birthYear: 1981 },
  { name: "Theo",       key: "theo",     month:  1, day: 10, birthYear: 2013 },
  { name: "Teteu",      key: "teteu",    month:  9, day:  4, birthYear: 2015 },
];

export function getBirthdayToday(month: number, day: number): BirthdayPerson | null {
  return BIRTHDAYS.find(p => p.month === month && p.day === day) ?? null;
}

export function getBirthdayByKey(key: string): BirthdayPerson | null {
  return BIRTHDAYS.find(p => p.key === key) ?? null;
}

export interface ArtStyle {
  name: string;
  prompt: string;
}

/**
 * Portrait styles. Since #84 the panels show any style, so the list is wide; a style that
 * abstracts the face away (cubism, caricature) is left out, because likeness comes first (#85).
 */
export const BIRTHDAY_STYLES: ArtStyle[] = [
  { name: "Woodcut",        prompt: "hand-carved woodcut print, bold U-gouge marks, black ink on paper" },
  { name: "Watercolor",     prompt: "watercolor painting, rich transparent washes, wet-on-wet technique, visible paper texture" },
  { name: "Art Nouveau",    prompt: "Art Nouveau portrait, flowing organic lines, Mucha-inspired decorative border" },
  { name: "Pop Art",        prompt: "Pop Art portrait, Warhol-inspired, vivid colors, halftone dots" },
  { name: "Impressionist",  prompt: "Impressionist oil painting, visible brushstrokes, dappled light, Monet-inspired" },
  { name: "Ukiyo-e",        prompt: "Japanese ukiyo-e woodblock print, flat color planes, bold outlines" },
  { name: "Art Deco",       prompt: "Art Deco portrait, geometric patterns, gold and black, elegant symmetry" },
  { name: "Pointillist",    prompt: "Pointillist painting, tiny dots of color, Seurat-inspired, luminous" },
  { name: "Pencil Sketch",  prompt: "detailed graphite pencil sketch, fine cross-hatching, full tonal range, on white paper" },
  { name: "Charcoal",       prompt: "dramatic charcoal drawing, expressive strokes, deep shadows, textured paper" },
  { name: "Oil Portrait",   prompt: "classical oil portrait, Rembrandt chiaroscuro, warm glowing skin tones, dark background, fine glazing" },
  { name: "Renaissance",    prompt: "Italian Renaissance portrait painting, soft sfumato, tempera and oil, landscape background" },
  { name: "Gouache",        prompt: "gouache painting, opaque matte pigment, bold color fields, visible brush strokes" },
  { name: "Graphic Novel",  prompt: "graphic novel illustration, bold ink outlines, cel-shaded color, comic book art" },
  { name: "Ink + Wash",     prompt: "ink and watercolor wash illustration, loose black ink lines, transparent color washes" },
  { name: "Linocut",        prompt: "linocut block print, bold carved relief lines, hand-printed texture" },
  { name: "Etching",        prompt: "fine etching engraving, parallel hatching lines, copper plate style, intricate detail" },
  { name: "Pastel",         prompt: "soft pastel drawing, chalky blended color, textured toned paper" },
  { name: "Klimt Gold",     prompt: "Gustav Klimt-inspired portrait, gold leaf ornament, mosaic patterns, jewel colors" },
  { name: "Van Gogh",       prompt: "Van Gogh-inspired oil portrait, thick swirling impasto brushstrokes, vivid complementary colors" },
  { name: "Fauvist",        prompt: "Fauvist portrait, Matisse-inspired, bold unnatural colors, expressive brushwork" },
  { name: "Stained Glass",  prompt: "stained glass window portrait, leaded outlines, luminous jewel-colored glass" },
  { name: "Mosaic",         prompt: "Byzantine mosaic portrait, small glass and stone tesserae, gold background" },
  { name: "Storybook",      prompt: "storybook illustration, warm gouache and colored pencil, gentle whimsical detail" },
];

/** The first year of the shuffled schedule; each block of BIRTHDAY_STYLES.length years after it gets a new shuffle. */
const STYLE_EPOCH = 2026;

/** Deterministic PRNG (mulberry32), so the schedule needs no storage and every caller agrees. */
function seededRandom(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 0..n-1 in a seeded random order (Fisher-Yates). */
function shuffledIndices(n: number, rand: () => number): number[] {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * The person's style this year: random, but never shared by two people in the same year and
 * never repeated by one person within a block of 24 years (DECISIONS #85).
 *
 * It is a Latin square, cell (year, person) = (year + person) mod 24, with its rows (years),
 * columns (people) and symbols (styles) each shuffled by a seed for the block. Shuffling a
 * Latin square's rows, columns and symbols keeps it Latin, so every year has distinct styles
 * and every person sees all 24 once per block, in an order that looks random. A person's column
 * comes from their position in BIRTHDAYS, so adding someone at the end changes nobody else's.
 */
export function getBirthdayStyle(person: BirthdayPerson, year: number): ArtStyle {
  const n = BIRTHDAY_STYLES.length;
  const offset = year - STYLE_EPOCH;
  const block = Math.floor(offset / n);
  const rand = seededRandom(0x6b1d + block * 7919);
  const styleOrder = shuffledIndices(n, rand);
  const yearOrder = shuffledIndices(n, rand);
  const personOrder = shuffledIndices(n, rand);
  const place = Math.max(0, BIRTHDAYS.findIndex(p => p.key === person.key)) % n;
  const yearInBlock = ((offset % n) + n) % n;
  return BIRTHDAY_STYLES[styleOrder[(yearOrder[yearInBlock] + personOrder[place]) % n]];
}

/** A style picked by index (test routes, `?style=N`), or the person's style for the year. */
export function resolveBirthdayStyle(person: BirthdayPerson, year: number, styleIdx?: number): ArtStyle {
  return styleIdx !== undefined ? BIRTHDAY_STYLES[styleIdx % BIRTHDAY_STYLES.length] : getBirthdayStyle(person, year);
}
