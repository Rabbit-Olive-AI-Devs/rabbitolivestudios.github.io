/**
 * Wrap raw image bytes for the Cloudflare Images binding.
 *
 * `env.IMAGES.input()` accepts a ReadableStream. Passing a Uint8Array, an
 * ArrayBuffer or a Blob instead routes into the binding's text-source path and
 * throws `TypeError: Cannot read properties of undefined (reading 'font')` from
 * `serializeTextSource` — every AI call is billed, post-processing then fails,
 * and nothing reaches the cache (DECISIONS #59).
 *
 * Byte input used to work incidentally, so all four image pipelines passed the
 * AI model's JPEG bytes directly until it started throwing on 2026-08-20.
 */
export function bytesToImageStream(bytes: Uint8Array): ReadableStream {
  return new Response(bytes).body!;
}
