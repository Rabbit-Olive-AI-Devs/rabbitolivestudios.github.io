/**
 * World Skyline Series — image generation + post-processing.
 *
 * BW path:    SDXL → grayscale → crop/resize → caption → tone curve → 4-level quantize → 8-bit PNG
 * Color path: SDXL → RGB crop/resize → caption bar in RGB → Floyd-Steinberg dither → Spectra 6 indexed PNG
 *
 * Caption is ALWAYS baked into the PNG (both BW and color) so HTML wrappers are pure <img> tags.
 * For color: caption drawn BEFORE dithering — black bar + white text stay crisp through Floyd-Steinberg.
 */

import type { Env } from "./types";
import type { SkylineColorMode } from "./skyline";
import {
  generateAIImage,
  WIDTH,
  HEIGHT,
  centerCropGray,
  resizeGray,
  drawText,
  quantize4Level,
} from "./image";
import { applyToneCurve } from "./convert-1bit";
import { encodePNGGray8, encodePNGIndexed, pngToBase64 } from "./png";
import { decodePNG } from "./png-decode";
import { FONT_8X8 as FONT_DATA, measureText } from "./font";
import { generateAndDecodeColor } from "./image-color";
import { ditherFloydSteinberg } from "./dither-spectra6";
import { SPECTRA6_PALETTE } from "./spectra6";

// SDXL params (same as Pipeline B — stable for architectural subjects)
const SDXL_STEPS = 20;
const SDXL_GUIDANCE = 7.0;

// Caption bar dimensions (match Pipeline A: 24px black bar, white 8px text)
const BAR_H = 24;
const BAR_PAD = 8;
const CAPTION_SCALE = 1;

// --- Shared caption layout logic ---

interface CaptionLayout {
  leftText: string;
  leftX: number;
  centerText: string;
  centerX: number;
  rightText: string;
  rightX: number;
  barY: number;
  textY: number;
}

function computeCaptionLayout(left: string, center: string, right: string): CaptionLayout {
  const barY = HEIGHT - BAR_H;
  const textH = 8 * CAPTION_SCALE;
  const textY = barY + Math.floor((BAR_H - textH) / 2);

  const leftText = left.length > 35 ? left.slice(0, 32) + "..." : left;
  const leftX = BAR_PAD;

  const rightW = measureText(right, CAPTION_SCALE);
  const rightX = WIDTH - BAR_PAD - rightW;

  const leftW = measureText(leftText, CAPTION_SCALE);
  const leftEnd = BAR_PAD + leftW;
  const rightStart = rightX;
  const gap = 12;
  const availW = rightStart - leftEnd - 2 * gap;

  let centerText = center;
  if (availW > 0) {
    while (measureText(centerText, CAPTION_SCALE) > availW && centerText.length > 3) {
      centerText = centerText.slice(0, -1);
    }
    if (centerText.length < center.length) centerText += "...";
  } else {
    centerText = "";
  }
  const centerW = measureText(centerText, CAPTION_SCALE);
  const centerX = availW > 0 ? leftEnd + gap + Math.floor((availW - centerW) / 2) : 0;

  return { leftText, leftX, centerText, centerX, rightText: right, rightX, barY, textY };
}

// --- BW path: grayscale 4-level ---

async function generateGraySkyline(env: Env, prompt: string): Promise<Uint8Array> {
  const jpegBytes = await generateAIImage(env, prompt, SDXL_STEPS, SDXL_GUIDANCE);

  const pngResponse = (await env.IMAGES.input(jpegBytes).output({ format: "image/png" })).response();
  const pngBytes = new Uint8Array(await pngResponse.arrayBuffer());
  const decoded = await decodePNG(pngBytes);

  const cropped = centerCropGray(decoded.gray, decoded.width, decoded.height, WIDTH, HEIGHT);
  return (cropped.width === WIDTH && cropped.height === HEIGHT)
    ? cropped.gray
    : resizeGray(cropped.gray, cropped.width, cropped.height, WIDTH, HEIGHT);
}

function drawSkylineCaptionGray(
  buf: Uint8Array,
  left: string,
  center: string,
  right: string,
): void {
  const layout = computeCaptionLayout(left, center, right);

  // Solid black bar
  for (let y = layout.barY; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      buf[y * WIDTH + x] = 0;
    }
  }

  // White text on black bar
  drawText(buf, layout.leftX, layout.textY, layout.leftText, CAPTION_SCALE, 255);
  drawText(buf, layout.rightX, layout.textY, layout.rightText, CAPTION_SCALE, 255);
  if (layout.centerText) {
    drawText(buf, layout.centerX, layout.textY, layout.centerText, CAPTION_SCALE, 255);
  }
}

// --- Color path: RGB caption + Spectra 6 dithered ---

/** Draw a single character into an RGB buffer (3 bytes per pixel). */
function drawCharRGB(
  rgb: Uint8Array,
  x: number,
  y: number,
  code: number,
  scale: number,
  r: number,
  g: number,
  b: number,
): void {
  const idx = code - 32;
  if (idx < 0 || idx >= FONT_DATA.length) return;
  const glyph = FONT_DATA[idx];
  for (let row = 0; row < 8; row++) {
    const byte = glyph[row];
    for (let col = 0; col < 8; col++) {
      if (byte & (0x80 >> col)) {
        for (let sy = 0; sy < scale; sy++) {
          for (let sx = 0; sx < scale; sx++) {
            const px = x + col * scale + sx;
            const py = y + row * scale + sy;
            if (px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
              const off = (py * WIDTH + px) * 3;
              rgb[off] = r;
              rgb[off + 1] = g;
              rgb[off + 2] = b;
            }
          }
        }
      }
    }
  }
}

/** Draw text string into an RGB buffer. */
function drawTextRGB(
  rgb: Uint8Array,
  x: number,
  y: number,
  text: string,
  scale: number,
  r: number,
  g: number,
  b: number,
): void {
  const charWidth = 8 * scale;
  const spacing = scale;
  for (let i = 0; i < text.length; i++) {
    drawCharRGB(rgb, x + i * (charWidth + spacing), y, text.charCodeAt(i), scale, r, g, b);
  }
}

/** Draw caption bar directly into RGB buffer BEFORE dithering. */
function drawSkylineCaptionRGB(
  rgb: Uint8Array,
  left: string,
  center: string,
  right: string,
): void {
  const layout = computeCaptionLayout(left, center, right);

  // Solid black bar
  for (let y = layout.barY; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const off = (y * WIDTH + x) * 3;
      rgb[off] = 0;
      rgb[off + 1] = 0;
      rgb[off + 2] = 0;
    }
  }

  // White text on black bar
  drawTextRGB(rgb, layout.leftX, layout.textY, layout.leftText, CAPTION_SCALE, 255, 255, 255);
  drawTextRGB(rgb, layout.rightX, layout.textY, layout.rightText, CAPTION_SCALE, 255, 255, 255);
  if (layout.centerText) {
    drawTextRGB(rgb, layout.centerX, layout.textY, layout.centerText, CAPTION_SCALE, 255, 255, 255);
  }
}

async function generateColorSkyline(env: Env, prompt: string): Promise<Uint8Array> {
  return generateAndDecodeColor(env, prompt, SDXL_STEPS, SDXL_GUIDANCE);
}

// --- Public API ---

export interface SkylineImageResult {
  png: Uint8Array;
  base64: string;
  colorMode: SkylineColorMode;
}

/**
 * Generate a skyline image with caption baked into the PNG.
 *
 * BW mode:  SDXL → gray → caption → tone curve → 4-level → 8-bit PNG
 * Color mode: SDXL → RGB → caption bar → Floyd-Steinberg dither → Spectra 6 indexed PNG
 */
export async function generateSkylineImage(
  env: Env,
  prompt: string,
  caption: { left: string; center: string; right: string },
  colorMode: SkylineColorMode,
): Promise<SkylineImageResult> {
  if (colorMode === "bw") {
    const gray = await generateGraySkyline(env, prompt);
    drawSkylineCaptionGray(gray, caption.left, caption.center, caption.right);
    applyToneCurve(gray, 1.2, 0.95);
    quantize4Level(gray);
    const png = await encodePNGGray8(gray, WIDTH, HEIGHT);
    return { png, base64: pngToBase64(png), colorMode };
  }

  // Color path: SDXL → RGB → caption in RGB → dither → indexed PNG
  const rgb = await generateColorSkyline(env, prompt);
  drawSkylineCaptionRGB(rgb, caption.left, caption.center, caption.right);
  const indices = ditherFloydSteinberg(rgb, WIDTH, HEIGHT, SPECTRA6_PALETTE);
  const png = await encodePNGIndexed(indices, WIDTH, HEIGHT, SPECTRA6_PALETTE);
  return { png, base64: pngToBase64(png), colorMode };
}
