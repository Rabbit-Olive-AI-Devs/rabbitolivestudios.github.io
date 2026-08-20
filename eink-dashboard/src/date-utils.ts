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

export function getChicagoDateISO(): string {
  return getChicagoDateParts().dateStr;
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
