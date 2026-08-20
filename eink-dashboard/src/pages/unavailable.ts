/**
 * The "no image to show" page.
 *
 * SenseCraft screenshots whatever a route returns, so a 503 renders as a browser
 * error page and a dead <img> renders as a broken-image glyph — both of which
 * look like a malfunctioning device. When we already know an image cannot be
 * produced, serve this instead (DECISIONS #57).
 *
 * E-ink rules apply: pure #000 on #fff (grays are invisible), no emoji (the
 * ESP32-S3 renderer has no emoji font), no JavaScript.
 */

/** Chicago wall-clock stamp, e.g. "Aug 20, 9:16 AM". */
export function chicagoStamp(now: Date = new Date()): string {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    month: "short", day: "numeric",
    hour: "numeric", minute: "2-digit",
  }).format(now);
}

export function unavailableHTML(title: string, detail: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800">
<title>${title}</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 800px; height: 480px; overflow: hidden; background: #fff; color: #000; }
  body {
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    font-family: Helvetica, Arial, sans-serif; text-align: center; padding: 0 64px;
  }
  h1 { font-size: 40px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 18px; }
  p { font-size: 22px; line-height: 1.45; font-weight: 400; }
  .rule { width: 120px; height: 3px; background: #000; margin: 26px 0; }
  /* The stamp proves the panel is still refreshing rather than frozen on an old
     frame — the failure mode that caused image retention in DECISIONS #56. */
  .stamp { font-size: 18px; font-weight: 700; }
</style>
</head>
<body>
  <h1>${title}</h1>
  <div class="rule"></div>
  <p>${detail}</p>
  <div class="rule"></div>
  <p class="stamp">${chicagoStamp()}</p>
</body>
</html>`;
}
