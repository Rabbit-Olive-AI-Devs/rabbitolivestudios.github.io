/**
 * Shared Chicago timezone date helpers.
 *
 * All date logic in this project uses America/Chicago timezone
 * since the display device is located there.
 */

export function getChicagoDateParts(): { year: string; month: string; day: string; dateStr: string } {
  const now = new Date();
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const parts = fmt.formatToParts(now);
  const year = parts.find((p) => p.type === "year")!.value;
  const month = parts.find((p) => p.type === "month")!.value;
  const day = parts.find((p) => p.type === "day")!.value;
  const dateStr = `${year}-${month}-${day}`;
  return { year, month, day, dateStr };
}

/** Current hour (0-23) in America/Chicago. Used to keep the alert check clear of the daily warm. */
export function getChicagoHour(): number {
  const hour = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    hour: "2-digit",
    hour12: false,
  }).formatToParts(new Date()).find((p) => p.type === "hour")!.value;
  // Intl can render midnight as "24" in some ICU versions.
  return parseInt(hour, 10) % 24;
}

export function getChicagoDateISO(): string {
  return getChicagoDateParts().dateStr;
}

/**
 * Which Chicago date the daily image warm should fill.
 *
 * The warm runs before midnight Chicago (04:35 and 05:35 UTC), so in the
 * evening it targets *tomorrow* and the keys are already warm when the date
 * rolls — there is no window in which a device poll can be the generator
 * (DECISIONS #64). Before noon it targets today, which makes the second fire
 * (00:35 CDT) a safety net that fills anything the evening run missed.
 */
export function dailyWarmTargetDate(chicagoDateStr: string, chicagoHour: number): string {
  return chicagoHour >= 12 ? shiftDateStr(chicagoDateStr, 1) : chicagoDateStr;
}

/**
 * Shift a `YYYY-MM-DD` string by whole calendar days.
 *
 * Used to walk backwards through daily cache keys when looking for the most
 * recent usable image (DECISIONS #57). Deliberately built on `Date.UTC` — local
 * time would drift by an hour across a DST transition and could land on the
 * same day twice or skip one.
 */
export function shiftDateStr(dateStr: string, deltaDays: number): string {
  const [y, m, d] = dateStr.split("-").map(Number);
  const shifted = new Date(Date.UTC(y, m - 1, d + deltaDays));
  const yy = shifted.getUTCFullYear();
  const mm = String(shifted.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(shifted.getUTCDate()).padStart(2, "0");
  return `${yy}-${mm}-${dd}`;
}
