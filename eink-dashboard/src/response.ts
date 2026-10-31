/** Shared HTML response helper with security headers. */

export function htmlResponse(html: string, cacheControl: string, extra?: HeadersInit): Response {
  const headers = new Headers({
    "Content-Type": "text/html; charset=utf-8",
    "Cache-Control": cacheControl,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
  });
  if (extra) new Headers(extra).forEach((value, name) => headers.set(name, value));
  return new Response(html, { headers });
}
