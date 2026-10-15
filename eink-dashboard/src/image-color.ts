/**
 * AI image → full-colour 800×480 RGB. Every picture pipeline sends this untouched to the
 * panels, which convert it themselves (DECISIONS #73, #81).
 */

import { decodePNG } from "./png-decode";
import { generateAIImage, generateFluxImage, WIDTH, HEIGHT } from "./image";
import type { Env } from "./types";
import { bytesToImageStream } from "./images-input";

/**
 * AI output (JPEG) → the picture as the panels get it: centre-cropped and resized to
 * 800×480 by Cloudflare Images, full colour, otherwise untouched (DECISIONS #81).
 */
export async function jpegToRGB(env: Env, jpegBytes: Uint8Array): Promise<Uint8Array> {
  const pngResponse = (await env.IMAGES
    .input(bytesToImageStream(jpegBytes))
    .transform({ width: WIDTH, height: HEIGHT, fit: "cover" })
    .output({ format: "image/png" })
  ).response();
  const decoded = await decodePNG(new Uint8Array(await pngResponse.arrayBuffer()));
  if (!decoded.rgb) throw new Error("Expected color PNG but got grayscale");
  return decoded.rgb;
}

// --- AI model → RGB 800×480 (SDXL) ---

export async function generateAndDecodeColor(
  env: Env,
  prompt: string,
  steps: number = 20,
  guidance: number = 7.0,
): Promise<Uint8Array> {
  const jpegBytes = await generateAIImage(env, prompt, steps, guidance);

  return jpegToRGB(env, jpegBytes);
}

// --- AI model → RGB 800×480 (FLUX.2) ---

export async function generateAndDecodeColorFlux(
  env: Env,
  prompt: string,
): Promise<Uint8Array> {
  const jpegBytes = await generateFluxImage(env, prompt);

  return jpegToRGB(env, jpegBytes);
}
