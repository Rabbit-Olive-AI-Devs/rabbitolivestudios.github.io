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

/**
 * The person's style this year. Each person walks the whole list, one style a year, starting
 * at their own place, spaced evenly by their position in BIRTHDAYS: nobody repeats a style for
 * 24 years and no two people share one in the same year (until #85 everyone got the same
 * style, picked by `year % 10`).
 */
export function getBirthdayStyle(person: BirthdayPerson, year: number): ArtStyle {
  const spacing = Math.max(1, Math.floor(BIRTHDAY_STYLES.length / BIRTHDAYS.length));
  const place = Math.max(0, BIRTHDAYS.findIndex(p => p.key === person.key));
  return BIRTHDAY_STYLES[(place * spacing + year) % BIRTHDAY_STYLES.length];
}

/** A style picked by index (test routes, `?style=N`), or the person's style for the year. */
export function resolveBirthdayStyle(person: BirthdayPerson, year: number, styleIdx?: number): ArtStyle {
  return styleIdx !== undefined ? BIRTHDAY_STYLES[styleIdx % BIRTHDAY_STYLES.length] : getBirthdayStyle(person, year);
}
