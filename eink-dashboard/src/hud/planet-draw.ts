/**
 * Draws a planet from its own data. There are no photographs of exoplanets,
 * so this is an interpretation, but an honest one: size and mass class decide
 * the kind of world, temperature decides its surface, the number of suns
 * decides how it is lit, and the name seeds the pattern so the same planet
 * always looks the same and no two look alike.
 *
 * The sphere is shaded and then ordered-dithered straight to black and white,
 * so it reaches the panel as dots with no grey (DECISIONS #48, #66).
 */

import type { Planet } from "../planet";
import type { Canvas } from "./canvas";

export type PlanetKind = "lava" | "rock" | "ocean" | "ice" | "neptune" | "gas";

export interface PlanetLook {
  kind: PlanetKind;
  seed: number;
  /** Directions the light comes from; two for a planet with two suns. */
  lights: [number, number, number][];
  bands: number;
  craters: number;
  spot: boolean;
  ring: boolean;
}

function hash(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  return h;
}

const unit = (v: [number, number, number]): [number, number, number] => {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
};

const MAIN_LIGHT = unit([-0.55, -0.55, 0.63]);
const SECOND_LIGHT = unit([0.75, -0.2, 0.5]);

export function planetLook(p: Planet): PlanetLook {
  const seed = hash(p.name) % 1000;
  const t = p.tempK;
  let kind: PlanetKind;
  if (p.radius >= 6) kind = "gas";
  else if (p.radius >= 2.5) kind = "neptune";
  else if (t !== null && t > 1000) kind = "lava";
  else if (t !== null && t >= 200 && t <= 330) kind = "ocean";
  else if (t !== null && t < 200) kind = "ice";
  else kind = "rock";
  return {
    kind,
    seed,
    lights: p.stars >= 2 ? [MAIN_LIGHT, SECOND_LIGHT] : [MAIN_LIGHT],
    bands: 9 + (seed % 8),
    craters: 4 + (seed % 7),
    spot: seed % 3 === 0,
    // rings are a guess for any exoplanet; keep them rare and only on cold giants
    ring: kind === "gas" && seed % 6 === 0 && (t === null || t < 400),
  };
}

/** A plain-words class, such as "HOT GAS GIANT" or "TEMPERATE ROCKY". */
export function planetType(p: Planet): string {
  const size = p.radius >= 6 ? "GAS GIANT" : p.radius >= 2.5 ? "NEPTUNE-LIKE" : p.radius >= 1.6 ? "SUPER-EARTH" : "ROCKY";
  const t = p.tempK;
  const heat = t === null ? "" : t > 700 ? "HOT " : t > 330 ? "WARM " : t >= 180 ? "TEMPERATE " : "COLD ";
  return heat + size;
}

// --- Texture -----------------------------------------------------------------------

const BAYER8: number[][] = (() => {
  let m = [[0]];
  for (let n = 0; n < 3; n++) {
    const k = m.length;
    const o = Array.from({ length: k * 2 }, () => new Array<number>(k * 2));
    for (let y = 0; y < k; y++) {
      for (let x = 0; x < k; x++) {
        const v = m[y][x] * 4;
        o[y][x] = v; o[y][x + k] = v + 2; o[y + k][x] = v + 3; o[y + k][x + k] = v + 1;
      }
    }
    m = o;
  }
  return m;
})();

const rand = (x: number, y: number, s: number) => {
  const n = Math.sin(x * 127.1 + y * 311.7 + s * 74.7) * 43758.5453;
  return n - Math.floor(n);
};

function noise(x: number, y: number, s: number): number {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  return rand(xi, yi, s) * (1 - u) * (1 - v) + rand(xi + 1, yi, s) * u * (1 - v)
    + rand(xi, yi + 1, s) * (1 - u) * v + rand(xi + 1, yi + 1, s) * u * v;
}

const fbm = (x: number, y: number, s: number) =>
  0.5 * noise(x, y, s) + 0.3 * noise(x * 2.3, y * 2.3, s + 1) + 0.2 * noise(x * 5.1, y * 5.1, s + 2);

const RING = { inner: 1.3, outer: 1.75, tilt: 0.28 };
/** A ringed planet is drawn smaller so the ring fits where a plain planet would. */
const RINGED_SCALE = 0.62;
const MAX_RADIUS = 400;

/** How bright the surface is at a point on the disc (nx, ny in -1..1), and whether it glows. */
function surface(look: PlanetLook, nx: number, ny: number, nz: number): { albedo: number; glow: boolean } {
  const s = look.seed;
  const u = Math.atan2(nx, nz) * 2 + s;
  const v = ny * 3;
  switch (look.kind) {
    case "gas": {
      const lat = ny + 0.05 * Math.sin(nx * 6 + s);
      let albedo = 0.7 + 0.3 * Math.sin(lat * look.bands + 1.2 * noise(u * 1.5, v * 4, s));
      if (look.spot) {
        const dx = (nx - 0.25) / 0.22, dy = (ny - 0.3) / 0.11;
        if (dx * dx + dy * dy < 1) albedo *= 0.45;
      }
      return { albedo, glow: false };
    }
    case "neptune": {
      let albedo = 0.8 + 0.12 * Math.sin(ny * (5 + (s % 4)) + 0.6 * noise(u, v * 3, s));
      const dx = (nx + 0.2) / 0.2, dy = (ny - 0.25) / 0.1;
      if (look.spot && dx * dx + dy * dy < 1) albedo *= 0.5;
      return { albedo, glow: false };
    }
    case "lava":
      return { albedo: 0.32 + 0.25 * fbm(u * 2, v * 2, s), glow: Math.abs(fbm(u * 2.6, v * 2.6, s + 9) - 0.5) < 0.022 };
    case "ocean": {
      const land = fbm(u * 1.4, v * 1.4, s) > 0.53;
      let albedo = land ? 0.75 + 0.25 * noise(u * 6, v * 6, s) : 0.38;
      if (noise(u * 1.2 + 3, v * 5, s + 5) > 0.72) albedo = 1.1; // cloud streaks
      if (Math.abs(ny) > 0.86) albedo = 1.15; // polar caps
      return { albedo, glow: false };
    }
    case "ice": {
      const crack = Math.abs(fbm(u * 3, v * 3, s + 4) - 0.5) < 0.012;
      return { albedo: crack ? 0.45 : 1.02 - 0.12 * fbm(u * 2, v * 2, s), glow: false };
    }
    case "rock": {
      let albedo = 0.5 + 0.6 * fbm(u * 1.7, v * 1.7, s);
      for (let k = 0; k < look.craters; k++) {
        const cx = rand(k, 1, s) * 1.6 - 0.8, cy = rand(k, 2, s) * 1.6 - 0.8, cr = 0.05 + rand(k, 3, s) * 0.1;
        const d = Math.hypot(nx - cx, ny - cy);
        if (d < cr) albedo *= 0.6;
        else if (d < cr + 0.02) albedo *= 1.35;
      }
      return { albedo, glow: false };
    }
  }
}

/** Draw the planet centred on (cx, cy) inside radius `r`. Work is bounded whatever `r` is. */
export function drawPlanet(c: Canvas, cx: number, cy: number, r: number, p: Planet): void {
  if (!Number.isFinite(r) || r <= 0) return;
  const look = planetLook(p);
  const R = Math.round(Math.min(r, MAX_RADIUS) * (look.ring ? RINGED_SCALE : 1));
  const span = look.ring ? Math.ceil(R * RING.outer) + 1 : R;
  const put = (x: number, y: number, brightness: number) => {
    if (x < 0 || y < 0 || x >= c.w || y >= c.h) return;
    c.px[y * c.w + x] = (1 - brightness) * 64 > BAYER8[y & 7][x & 7] + 0.5 ? 0 : 1;
  };

  for (let y = -span; y <= span; y++) {
    for (let x = -span; x <= span; x++) {
      const d2 = x * x + y * y;
      const onDisc = d2 <= R * R;
      if (look.ring) {
        const rd = Math.hypot(x / R, y / (R * RING.tilt));
        const band = (rd - RING.inner) / (RING.outer - RING.inner);
        const onRing = band >= 0 && band <= 1 && !(band > 0.45 && band < 0.55);
        // the far half of the ring is hidden behind the planet
        if (onRing && (!onDisc || y > 0)) {
          put(cx + x, cy + y, 0.42 + 0.2 * rand(Math.round(rd * 40), 1, look.seed));
          continue;
        }
      }
      if (!onDisc) continue;
      const nx = x / R, ny = y / R, nz = Math.sqrt(Math.max(0, 1 - nx * nx - ny * ny));
      let lit = 0;
      for (const l of look.lights) lit = Math.max(lit, nx * l[0] + ny * l[1] + nz * l[2]);
      const { albedo, glow } = surface(look, nx, ny, nz);
      put(cx + x, cy + y, glow ? 1 : Math.min(1, 0.04 + Math.max(0, lit) * albedo * 1.05));
      if (d2 > (R - 1) * (R - 1)) put(cx + x, cy + y, 0); // limb
    }
  }
}
