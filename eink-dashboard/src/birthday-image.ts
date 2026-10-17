/**
 * Birthday portrait generation pipeline.
 *
 * Uses FLUX.2 klein-9b with optional reference photos from R2.
 * Produces an 800x480 full-colour PNG with a birthday caption (#81).
 */

import { encodePNGRGB } from "./png";
import { measureText } from "./font";
import {
  WIDTH, HEIGHT,
  burnCaption, drawText,
} from "./image";
import { jpegToRGB } from "./image-color";
import { resolveBirthdayStyle } from "./birthday";
import { isNeuronBudgetError } from "./cache-guard";
import type { BirthdayPerson } from "./birthday";
import type { Env } from "./types";

const FLUX_MODEL = "@cf/black-forest-labs/flux-2-klein-9b";
// 5:3, the panel's shape: a 4:3 picture lost a fifth of its height to the crop, often the
// top of the head (#85).
const FLUX_WIDTH = 1280;
const FLUX_HEIGHT = 768;

const BAR_H = 24;
const BAR_PAD = 8;

/** Strip accents/diacritics for ASCII-only bitmap font. */
export function stripAccents(s: string): string {
  return s.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

/** Build an age description for the prompt. */
export function ageDescription(person: BirthdayPerson, currentYear: number): string {
  const age = currentYear - person.birthYear;
  if (age <= 12) return `a ${age}-year-old child`;
  if (age <= 17) return `a ${age}-year-old teenager`;
  return `a ${age}-year-old person`;
}

/**
 * Generate a birthday portrait image using FLUX.2 with optional R2 reference photos.
 * Returns an 800x480 full-colour PNG.
 */
export async function generateBirthdayImage(
  env: Env,
  person: BirthdayPerson,
  currentYear: number,
  styleOverride?: number,
): Promise<Uint8Array> {
  const style = resolveBirthdayStyle(person, currentYear, styleOverride);
  const age = currentYear - person.birthYear;

  // Fetch up to 4 reference photos from R2
  const photos = await fetchReferencePhotos(env, person.key);
  console.log(`Birthday: found ${photos.length} reference photo(s) for ${person.key}`);

  // Single FLUX.2 attempt — no retry to conserve neuron budget (free tier)
  let jpegBytes: Uint8Array | null = null;
  try {
    jpegBytes = await callFluxPortrait(env, person, style.prompt, photos, currentYear);
  } catch (err) {
    console.error("Birthday FLUX.2 failed:", err);
    if (isNeuronBudgetError(err)) throw err;
  }

  if (!jpegBytes) {
    throw new Error("FLUX.2 portrait generation failed");
  }

  // The portrait untouched in full colour; the caption bar as solid black and white (DECISIONS #81).
  const rgb = await jpegToRGB(env, jpegBytes);
  burnCaption(rgb, (mask) => drawBirthdayCaption(mask, stripAccents(person.name), age, style.name));
  return encodePNGRGB(rgb, WIDTH, HEIGHT);
}

/** Try to fetch a file from R2 with .jpg or .jpeg extension. */
export async function getPhotoFromR2(env: Env, basePath: string): Promise<Uint8Array | null> {
  for (const ext of [".jpg", ".jpeg"]) {
    try {
      const obj = await env.PHOTOS.get(basePath + ext);
      if (obj) return new Uint8Array(await obj.arrayBuffer());
    } catch { /* skip */ }
  }
  return null;
}

/**
 * Fetch up to 4 reference photos from R2.
 * Tries numbered format first ({key}_0.jpg/jpeg .. {key}_3.jpg/jpeg),
 * falls back to single file ({key}.jpg/jpeg).
 */
export async function fetchReferencePhotos(env: Env, key: string): Promise<Uint8Array[]> {
  const photos: Uint8Array[] = [];

  // Try numbered photos: {key}_0 through {key}_3
  for (let i = 0; i < 4; i++) {
    const photo = await getPhotoFromR2(env, `portraits/${key}_${i}`);
    if (photo) {
      photos.push(photo);
    } else {
      break; // Stop at first missing index
    }
  }

  // Fallback: try single file {key}.jpg/jpeg
  if (photos.length === 0) {
    const photo = await getPhotoFromR2(env, `portraits/${key}`);
    if (photo) photos.push(photo);
  }

  return photos;
}

/**
 * Call FLUX.2 klein-9b via multipart FormData.
 * Steps are fixed at 4 for klein models.
 * Supports up to 4 reference images for better likeness.
 */
export async function callFluxPortrait(
  env: Env,
  person: BirthdayPerson,
  stylePrompt: string,
  photos: Uint8Array[],
  currentYear: number,
): Promise<Uint8Array> {
  const form = new FormData();
  const ageLine = ageDescription(person, currentYear);

  let prompt: string;
  if (photos.length > 0) {
    for (let i = 0; i < photos.length; i++) {
      form.append(`input_image_${i}`, new Blob([photos[i]], { type: "image/jpeg" }));
    }
    const imageRefs = photos.length === 1
      ? "the person in image 0"
      : `the person shown in images ${photos.map((_, i) => i).join(", ")} (the same person in every photo)`;
    // Likeness first: the references are face crops (scripts/prepare-portraits.py), and the
    // prompt asks for the face to be kept and only the medium to change (DECISIONS #85).
    prompt = `portrait of ${imageRefs}, ${ageLine}, rendered as ${stylePrompt}. ` +
      "Keep their exact likeness from the photos: the same face shape, eyes, eyebrows, nose, mouth, smile, " +
      "hairline, hairstyle, hair color, facial hair and skin tone, instantly recognizable. " +
      "Only the artistic medium changes, not the face. Head and shoulders, centered, looking at the viewer, " +
      "warm natural smile, eyes clearly visible, no sunglasses, no text, no words, no letters, no writing, " +
      "no signature, no seal, no stamp, no calligraphy";
  } else {
    prompt = `artistic portrait of ${ageLine}, ${stylePrompt}, head and shoulders, centered composition, looking at viewer, smiling, no text, no words, no letters, no writing`;
  }

  form.append("prompt", prompt);
  form.append("width", String(FLUX_WIDTH));
  form.append("height", String(FLUX_HEIGHT));
  form.append("guidance", "7.0");

  // Serialize FormData for Workers AI multipart API
  const formResponse = new Response(form);
  const formStream = formResponse.body;
  const formContentType = formResponse.headers.get("content-type")!;

  const result: any = await env.AI.run(FLUX_MODEL as any, {
    multipart: {
      body: formStream,
      contentType: formContentType,
    },
  });

  // FLUX.2 returns base64-encoded image in { image: "..." }
  if (result && typeof result === "object") {
    const img = result.image ?? result.images?.[0];
    if (img && typeof img === "string") {
      return Uint8Array.from(atob(img), c => c.charCodeAt(0));
    }
  }
  if (result instanceof Uint8Array) return result;
  if (result instanceof ArrayBuffer) return new Uint8Array(result);
  if (result instanceof ReadableStream) {
    return new Uint8Array(await new Response(result).arrayBuffer());
  }

  throw new Error(`Unexpected FLUX.2 response type: ${typeof result}`);
}

/**
 * Draw birthday caption bar at the bottom of the image.
 * 24px black bar with white text:
 *   Left: "Happy Birthday!"
 *   Center: "{Name} - {age} years"
 *   Right: "{style name}"
 */
function drawBirthdayCaption(
  buf: Uint8Array,
  name: string,
  age: number,
  styleName: string,
): void {
  const barY = HEIGHT - BAR_H;
  const textH = 8;
  const textY = barY + Math.floor((BAR_H - textH) / 2);

  // Solid black bar
  for (let y = barY; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      buf[y * WIDTH + x] = 0;
    }
  }

  // Left: "Happy Birthday!"
  const left = "Happy Birthday!";
  drawText(buf, BAR_PAD, textY, left, 1, 255);

  // Right: style name
  const rightW = measureText(styleName, 1);
  drawText(buf, WIDTH - BAR_PAD - rightW, textY, styleName, 1, 255);

  // Center: "Name - age years"
  const center = `${name} - ${age} years`;
  const locW = measureText(left, 1);
  const locEnd = BAR_PAD + locW;
  const dateStart = WIDTH - BAR_PAD - rightW;
  const gap = 12;
  const availW = dateStart - locEnd - 2 * gap;
  if (availW > 0) {
    let centerText = center;
    while (measureText(centerText, 1) > availW && centerText.length > 3) {
      centerText = centerText.slice(0, -1);
    }
    if (centerText.length < center.length) centerText += "...";
    const centerW = measureText(centerText, 1);
    const centerX = locEnd + gap + Math.floor((availW - centerW) / 2);
    drawText(buf, centerX, textY, centerText, 1, 255);
  }
}
