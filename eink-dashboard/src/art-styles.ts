/**
 * The shared art-style library (DECISIONS #86). The colour moment, the skyline and the
 * birthday portrait all draw from this one list; each style says which of them it suits:
 *
 *   moment   — a historical scene
 *   skyline  — a city view
 *   portrait — a likeness of a family member; styles that abstract the face away
 *              (pixel art, low poly, flat geometric posters) are left out, likeness first (#85)
 *
 * `mono` marks a black-and-white medium; the skyline uses it to pick its lighting words.
 * Prompts describe only the medium, never the subject. Pipeline B (/fact1.png, 1-bit) and the
 * unwarmed Pipeline A (/fact.png) keep their own lists.
 */

export type ArtSubject = "moment" | "skyline" | "portrait";

export interface ArtStyle {
  id: string;
  name: string;
  prompt: string;
  for: readonly ArtSubject[];
  mono?: boolean;
}

const ALL: readonly ArtSubject[] = ["moment", "skyline", "portrait"];
const SCENES: readonly ArtSubject[] = ["moment", "skyline"];
const MOMENT_PORTRAIT: readonly ArtSubject[] = ["moment", "portrait"];

export const ART_STYLES: readonly ArtStyle[] = [
  // --- Painting ---
  { id: "gouache", name: "Gouache", for: ALL, prompt: "gouache painting, opaque matte pigment, bold color fields, visible brush strokes, thick paint application" },
  { id: "oil_painting", name: "Oil Painting", for: SCENES, prompt: "oil painting on canvas, rich saturated colors, visible impasto brush strokes, dramatic lighting, painterly realism" },
  { id: "oil_portrait", name: "Oil Portrait", for: ["portrait"], prompt: "classical oil portrait, Rembrandt chiaroscuro, warm glowing skin tones, dark background, fine glazing" },
  { id: "impressionist", name: "Impressionist", for: ALL, prompt: "Impressionist oil painting, visible brushstrokes, dappled shimmering light, Monet-inspired" },
  { id: "pointillist", name: "Pointillist", for: ALL, prompt: "Pointillist painting, tiny dots of pure color, Seurat-inspired, luminous" },
  { id: "van_gogh", name: "Van Gogh", for: ALL, prompt: "Van Gogh-inspired oil painting, thick swirling impasto brushstrokes, vivid complementary colors" },
  { id: "fauvist", name: "Fauvist", for: ALL, prompt: "Fauvist painting, Matisse-inspired, bold unnatural colors, expressive brushwork" },
  { id: "expressionist", name: "Expressionist", for: ALL, prompt: "German Expressionist painting, Munch-inspired, swirling emotional color, distorted brushwork" },
  { id: "watercolor", name: "Watercolor", for: ALL, prompt: "watercolor painting, rich transparent washes, wet-on-wet blooms, soft edges, visible paper texture" },
  { id: "ink_wash", name: "Ink + Wash", for: ALL, prompt: "ink and watercolor wash illustration, loose black ink lines, transparent color washes, visible paper texture" },
  { id: "renaissance", name: "Renaissance", for: MOMENT_PORTRAIT, prompt: "Italian Renaissance painting, soft sfumato, tempera and oil, balanced classical composition" },
  { id: "fresco", name: "Fresco", for: ALL, prompt: "Renaissance fresco on plaster wall, chalky faded pigments, fine cracks, earthy palette" },
  { id: "baroque", name: "Baroque", for: MOMENT_PORTRAIT, prompt: "Baroque oil painting, Caravaggio-inspired tenebrism, deep shadows, theatrical spotlight" },
  { id: "pre_raphaelite", name: "Pre-Raphaelite", for: MOMENT_PORTRAIT, prompt: "Pre-Raphaelite painting, jewel-toned detail, luminous skin, lush botanical surroundings" },
  { id: "romantic", name: "Romantic", for: SCENES, prompt: "Romantic era painting, Turner-inspired, luminous atmospheric light, sweeping sky, sublime scale" },
  { id: "pastel", name: "Pastel", for: ALL, prompt: "soft pastel drawing, chalky blended color, textured toned paper" },
  { id: "muralism", name: "Mexican Muralism", for: ALL, prompt: "Mexican muralism, Diego Rivera-inspired fresco mural, monumental rounded figures, warm earth colors" },
  { id: "brazilian_modernism", name: "Brazilian Modernism", for: ALL, prompt: "Brazilian modernist painting, Tarsila do Amaral-inspired, smooth rounded simplified forms, warm tropical palette" },
  { id: "persian_miniature", name: "Persian Miniature", for: ALL, prompt: "Persian miniature painting, flat perspective, intricate patterns, jewel colors, gold details" },
  { id: "minhwa", name: "Korean Minhwa", for: SCENES, prompt: "Korean minhwa folk painting, bright mineral pigments on hanji paper, playful stylized forms" },
  { id: "shan_shui", name: "Chinese Ink Landscape", for: SCENES, prompt: "Chinese shan shui ink painting on rice paper, misty layered depth, expressive brush strokes, soft color" },
  { id: "sumi_e", name: "Sumi-e", for: ALL, mono: true, prompt: "Japanese sumi-e ink brush painting, few confident strokes, ink washes on rice paper, empty space" },
  { id: "surrealist", name: "Surrealist", for: SCENES, prompt: "Surrealist painting, Magritte and Dali-inspired, dreamlike impossible details, crisp smooth rendering" },
  { id: "klimt_gold", name: "Klimt Gold", for: ALL, prompt: "Gustav Klimt-inspired painting, gold leaf ornament, mosaic patterns, jewel colors" },
  { id: "art_nouveau", name: "Art Nouveau", for: ALL, prompt: "Art Nouveau illustration, flowing organic lines, Mucha-inspired decorative framing" },
  { id: "art_deco", name: "Art Deco", for: ALL, prompt: "Art Deco poster, geometric patterns, gold and bold accents, streamlined elegant forms" },
  // --- Printmaking ---
  { id: "woodcut", name: "Woodcut", for: ALL, mono: true, prompt: "hand-carved woodcut print, bold U-gouge marks, black ink on paper, sweeping curved gouge strokes" },
  { id: "linocut", name: "Linocut", for: ALL, mono: true, prompt: "linocut block print, bold carved relief lines, black ink on paper, hand-printed texture" },
  { id: "etching", name: "Etching", for: ALL, mono: true, prompt: "fine etching engraving, parallel hatching lines, copper plate style, intricate detail" },
  { id: "cordel", name: "Cordel Woodcut", for: ALL, mono: true, prompt: "Brazilian cordel woodcut, xilogravura folk print, bold black carving, naive charming forms" },
  { id: "ukiyo_e", name: "Ukiyo-e", for: ALL, prompt: "Japanese ukiyo-e woodblock print, flat color planes, bold key-block outlines, Hiroshige-inspired" },
  { id: "screenprint", name: "Screenprint", for: ALL, prompt: "screenprint poster, bold overlapping color layers, slight registration offset, Warhol-inspired" },
  { id: "risograph", name: "Risograph", for: ALL, prompt: "risograph print, two-color overprint in fluorescent pink and blue, grainy texture, slight misregistration" },
  { id: "cyanotype", name: "Cyanotype", for: ALL, prompt: "cyanotype sun print, deep Prussian blue and white, botanical photogram texture" },
  { id: "pop_art", name: "Pop Art", for: ALL, prompt: "Pop Art, Lichtenstein-inspired, vivid colors, halftone dots, bold outlines" },
  // --- Drawing ---
  { id: "pencil", name: "Pencil Sketch", for: ALL, mono: true, prompt: "detailed graphite pencil sketch, fine cross-hatching, full tonal range, on white paper" },
  { id: "charcoal", name: "Charcoal", for: ALL, mono: true, prompt: "dramatic charcoal drawing, expressive strokes, deep shadows, textured paper" },
  { id: "pen_ink", name: "Pen + Ink", for: ALL, mono: true, prompt: "detailed pen and ink drawing, fine crosshatching, stipple shading, black ink on white paper" },
  { id: "scratchboard", name: "Scratchboard", for: ALL, mono: true, prompt: "scratchboard illustration, fine white lines scratched from a black surface, dramatic negative space" },
  { id: "colored_pencil", name: "Colored Pencil", for: ALL, prompt: "colored pencil drawing, layered hatching, rich burnished color, visible paper tooth" },
  { id: "chalkboard", name: "Chalkboard", for: SCENES, prompt: "colored chalk drawing on a blackboard, dusty strokes, smudged highlights" },
  // --- Illustration and posters ---
  { id: "graphic_novel", name: "Graphic Novel", for: ALL, prompt: "graphic novel illustration, bold ink outlines, cel-shaded color, comic book art" },
  { id: "ligne_claire", name: "Ligne Claire", for: ALL, prompt: "Franco-Belgian ligne claire comic art, clean uniform outlines, flat bright colors, no hatching" },
  { id: "manga", name: "Manga", for: MOMENT_PORTRAIT, mono: true, prompt: "black and white manga illustration, expressive ink lines, screentone shading" },
  { id: "noir", name: "Ink Noir", for: SCENES, mono: true, prompt: "bold ink noir illustration, heavy black ink pools, dramatic chiaroscuro, film noir shadows" },
  { id: "storybook", name: "Storybook", for: ALL, prompt: "storybook illustration, warm gouache and colored pencil, gentle whimsical detail" },
  { id: "mid_century", name: "Mid-Century", for: ALL, prompt: "1950s mid-century modern illustration, Mary Blair-inspired, stylized shapes, cheerful retro color" },
  { id: "travel_poster", name: "Travel Poster", for: SCENES, prompt: "vintage travel poster illustration, bold flat color fields, retro graphic design, warm sunset palette" },
  { id: "wpa_poster", name: "WPA Poster", for: SCENES, prompt: "WPA poster style, screenprint colors, bold stylised shapes, 1930s national park poster" },
  { id: "constructivist", name: "Constructivist", for: SCENES, prompt: "1920s Constructivist poster, dynamic diagonals, red black and cream, photomontage geometry" },
  { id: "bauhaus", name: "Bauhaus", for: SCENES, prompt: "Bauhaus design, primary colors, circles squares and triangles, clean geometric composition" },
  { id: "psychedelic", name: "Psychedelic", for: ALL, prompt: "1960s psychedelic concert poster, swirling flowing forms, vibrating saturated colors" },
  { id: "pulp_sci_fi", name: "Pulp Sci-Fi", for: SCENES, prompt: "1950s pulp science fiction magazine cover painting, bold dramatic color, retro-futurist gloss" },
  { id: "synthwave", name: "Synthwave", for: SCENES, prompt: "synthwave retro-futuristic sunset, neon grid horizon, bold magenta and cyan, 80s aesthetic" },
  { id: "minimal_flat", name: "Minimal Flat", for: SCENES, prompt: "minimal flat color illustration, clean geometric shapes, large flat color blocks, modern graphic design" },
  { id: "mediterranean", name: "Mediterranean", for: SCENES, prompt: "Mediterranean travel illustration, warm terracotta and blue palette, gouache painting, sun-drenched colors" },
  { id: "airbrush", name: "80s Airbrush", for: ALL, prompt: "1980s airbrush illustration, smooth gradients, chrome highlights, glossy retro finish" },
  // --- Craft and material ---
  { id: "stained_glass", name: "Stained Glass", for: ALL, prompt: "stained glass window, leaded outlines, luminous jewel-colored glass" },
  { id: "mosaic", name: "Mosaic", for: ALL, prompt: "Byzantine mosaic, small glass and stone tesserae, gold background" },
  { id: "azulejo", name: "Azulejo Tiles", for: ALL, prompt: "Portuguese azulejo painted ceramic tiles, cobalt blue on white glaze, visible tile grid" },
  { id: "tapestry", name: "Tapestry", for: SCENES, prompt: "medieval woven tapestry, wool threads, flattened perspective, muted natural dyes" },
  { id: "embroidery", name: "Embroidery", for: ALL, prompt: "hand embroidery on linen, satin and chain stitches, colorful thread texture" },
  { id: "paper_cut", name: "Layered Paper", for: ALL, prompt: "layered cut paper art, stacked colored paper sheets, soft drop shadows, handmade depth" },
  { id: "paper_cutouts", name: "Paper Cutouts", for: SCENES, prompt: "Matisse-inspired paper cutouts, bold painted paper shapes, playful collage" },
  { id: "claymation", name: "Claymation", for: ALL, prompt: "claymation stop-motion scene, handmade plasticine models, fingerprint texture, miniature set lighting" },
  // --- Photographic and digital ---
  { id: "tintype", name: "Tintype", for: ALL, mono: true, prompt: "19th-century tintype photograph, wet plate collodion, silvery tones, dark vignette, chemical edge marks" },
  { id: "anime_film", name: "Hand-Painted Anime", for: ALL, prompt: "hand-painted anime film still, lush painted background, soft cel shading, warm cinematic light" },
  { id: "animated_3d", name: "3D Animated Film", for: ALL, prompt: "3D animated feature film still, stylized characters, soft global illumination, expressive appealing design" },
  { id: "pixel_art", name: "Pixel Art", for: SCENES, prompt: "16-bit pixel art, crisp visible pixels, limited retro game palette, dithered shading" },
  { id: "low_poly", name: "Low Poly", for: SCENES, prompt: "low poly 3D render, faceted geometric surfaces, soft gradient lighting" },
  { id: "isometric", name: "Isometric Diorama", for: SCENES, prompt: "isometric miniature diorama, tiny detailed 3D model, tilt-shift depth of field" },
];

export function stylesFor(subject: ArtSubject): ArtStyle[] {
  return ART_STYLES.filter((s) => s.for.includes(subject));
}

export function findArtStyle(id: string): ArtStyle | undefined {
  return ART_STYLES.find((s) => s.id === id);
}

/** Deterministic PRNG (mulberry32): schedules need no storage and every caller agrees. */
export function seededRandom(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 0..n-1 in a seeded random order (Fisher-Yates). */
export function shuffledIndices(n: number, rand: () => number): number[] {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Days since 2026-01-01 for a YYYY-MM-DD date (calendar days, no time zone involved). */
function dayIndex(dateStr: string): number {
  const [y, m, d] = dateStr.split("-").map(Number);
  return Math.round((Date.UTC(y, m - 1, d) - Date.UTC(2026, 0, 1)) / 86400000);
}

/**
 * The day's pick from a pool: a fresh shuffle of the whole pool for each block of pool-size
 * days, walked one style a day, so nothing repeats until every style has had its day and the
 * order is random rather than the list's.
 */
export function dailyPick<T>(pool: readonly T[], dateStr: string, salt: number): T {
  const n = pool.length;
  const day = dayIndex(dateStr);
  const block = Math.floor(day / n);
  const order = shuffledIndices(n, seededRandom(salt + block * 7919));
  return pool[order[((day % n) + n) % n]];
}

const MOMENT_STYLES = stylesFor("moment");

/** The colour moment's style for a Chicago date. */
export function momentStyleFor(dateStr: string): ArtStyle {
  return dailyPick(MOMENT_STYLES, dateStr, 0x4d4f);
}
