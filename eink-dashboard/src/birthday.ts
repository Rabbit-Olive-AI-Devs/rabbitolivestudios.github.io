/**
 * Birthday detection and art style rotation.
 *
 * On family birthday dates, /fact.png generates a portrait instead of
 * the regular "Moment Before" historical event.
 */

import { stylesFor, seededRandom, shuffledIndices } from "./art-styles";
import type { ArtStyle } from "./art-styles";

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

/** Portrait styles: every style in the shared library that keeps a face recognisable (#85, #86). */
export const BIRTHDAY_STYLES: ArtStyle[] = stylesFor("portrait");

/** The first year of the shuffled schedule; each block of BIRTHDAY_STYLES.length years after it gets a new shuffle. */
const STYLE_EPOCH = 2026;

/**
 * The person's style this year: random, but never shared by two people in the same year and
 * never repeated by one person within a block of BIRTHDAY_STYLES.length years (DECISIONS #85).
 *
 * It is a Latin square, cell (year, person) = (year + person) mod n, with its rows (years),
 * columns (people) and symbols (styles) each shuffled by a seed for the block. Shuffling a
 * Latin square's rows, columns and symbols keeps it Latin, so every year has distinct styles
 * and every person sees all n once per block, in an order that looks random. A person's column
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
