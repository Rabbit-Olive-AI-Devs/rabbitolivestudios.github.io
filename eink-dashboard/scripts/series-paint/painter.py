"""The series' painting engine (DECISIONS #74): lays textured brush strokes over a guide picture
(the underpainting, drawn by code), coarse to fine, in a few painters' manners. numpy + PIL only.

python3 -E -P painter.py <guide.png> <regions.png|auto> <oil|gouache|pixel> <out.png> [seed]

With "auto" the regions are the guide's main colour areas, so strokes stop at the edges of shapes.
Claude's Prologue is painted in gouache; the oil and pixel manners are kept for style tests.
"""
import math
import sys

import numpy as np
from PIL import Image, ImageFilter

REG = {"sky": 1, "moon": 2, "mesa": 3, "ground": 4, "pad": 5, "outpost": 6, "light": 7, "ship": 8, "ramp": 9, "figure": 10, "dune": 11}
# Default stroke direction (degrees) where the picture has no strong edge to follow.
REG_ANGLE = {1: 0, 2: 70, 3: 8, 4: 2, 5: 0, 6: 90, 7: 0, 8: 3, 9: 60, 10: 90, 11: -4}


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def blur(img, r):
    if r <= 0:
        return img
    im = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32) / 255.0


def lum(img):
    return img[..., 0] * 0.3 + img[..., 1] * 0.59 + img[..., 2] * 0.11


def rgb_to_hsv(c):
    r, g, b = c
    mx, mn = max(c), min(c)
    v = mx
    s = 0 if mx == 0 else (mx - mn) / mx
    if mx == mn:
        h = 0
    elif mx == r:
        h = ((g - b) / (mx - mn)) % 6
    elif mx == g:
        h = (b - r) / (mx - mn) + 2
    else:
        h = (r - g) / (mx - mn) + 4
    return h / 6, s, v


def hsv_to_rgb(h, s, v):
    h = (h % 1) * 6
    i = int(h)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]


class Brushes:
    """Pre-made dab stamps: an elongated footprint with bristle streaks, at several angles."""

    def __init__(self, rng, aspect=2.2, hard=6.0, bristle=0.45, angles=24, variants=3):
        self.rng, self.aspect, self.hard, self.bristle, self.angles, self.variants = rng, aspect, hard, bristle, angles, variants
        self.cache = {}

    def get(self, r, angle):
        r = max(1, int(round(r)))
        if r not in self.cache:
            self.cache[r] = [[self._make(r, math.pi * k / self.angles, v) for k in range(self.angles)] for v in range(self.variants)]
        k = int(round((angle % math.pi) / math.pi * self.angles)) % self.angles
        return self.cache[r][self.rng.integers(self.variants)][k]

    def _make(self, r, ang, v):
        a, b = r * self.aspect, r
        n = int(math.ceil(a)) + 2
        ys, xs = np.mgrid[-n:n + 1, -n:n + 1].astype(np.float32)
        u = xs * math.cos(ang) + ys * math.sin(ang)
        w = -xs * math.sin(ang) + ys * math.cos(ang)
        d = (u / a) ** 2 + (w / b) ** 2
        alpha = np.clip((1 - d) * self.hard, 0, 1)
        # Bristles: a random profile across the brush, constant along it -> streaks.
        prof = np.convolve(self.rng.random(4 * n + 8), np.ones(3) / 3, mode="same")
        idx = np.clip(((w / b + 1) * 0.5 * (len(prof) - 1)).astype(int), 0, len(prof) - 1)
        streak = 1 - self.bristle + self.bristle * prof[idx]
        # A little unevenness along the stroke too.
        along = 1 - 0.15 * self.rng.random() * (u / a + 1) * 0.5
        return (alpha * streak * along).astype(np.float32)


class Painter:
    def __init__(self, guide, regions, rng, ground_color):
        self.ref0 = guide
        self.reg = regions
        self.h, self.w = guide.shape[:2]
        self.rng = rng
        self.canvas = np.empty_like(guide)
        self.canvas[:] = ground_color
        self.height = np.zeros((self.h, self.w), np.float32)

    def dab(self, brush, x, y, color, opacity):
        h, w = brush.shape
        x0, y0 = int(x) - w // 2, int(y) - h // 2
        x1, y1 = x0 + w, y0 + h
        bx0, by0 = max(0, -x0), max(0, -y0)
        bx1, by1 = w - max(0, x1 - self.w), h - max(0, y1 - self.h)
        if bx1 <= bx0 or by1 <= by0:
            return
        a = brush[by0:by1, bx0:bx1, None] * opacity
        sl = (slice(y0 + by0, y0 + by1), slice(x0 + bx0, x0 + bx1))
        self.canvas[sl] = self.canvas[sl] * (1 - a) + np.asarray(color, np.float32) * a
        self.height[sl] += a[..., 0] * 0.5

    def flow(self, ref):
        g = lum(blur(ref, 2))
        gy, gx = np.gradient(g)
        return gx, gy, np.hypot(gx, gy)

    def layer(self, r, brushes, *, threshold, max_len, min_len=2, opacity=0.9, jitter=(0.02, 0.05, 0.05),
              angle_fn=None, straight=False, color_fn=None, tol=0.12, curvature=0.6, only=None, step_scale=0.75):
        ref = blur(self.ref0, r * 0.8)
        gx, gy, gm = self.flow(ref)
        g = max(1, int(r))
        hh, ww = (self.h // g) * g, (self.w // g) * g
        err = np.linalg.norm(self.canvas - ref, axis=2)[:hh, :ww].reshape(hh // g, g, ww // g, g)
        cells = err.mean(axis=(1, 3))
        flat = err.transpose(0, 2, 1, 3).reshape(hh // g, ww // g, g * g)
        best = flat.argmax(axis=2)
        cy, cx = np.nonzero(cells > threshold)
        order = self.rng.permutation(len(cy))
        n = 0
        for i in order:
            j, k = cy[i], cx[i]
            b = best[j, k]
            y, x = j * g + b // g + self.rng.random() - 0.5, k * g + b % g + self.rng.random() - 0.5
            yi, xi = int(np.clip(y, 0, self.h - 1)), int(np.clip(x, 0, self.w - 1))
            region = self.reg[yi, xi]
            if only is not None and region not in only:
                continue
            c = ref[yi, xi]
            hh_, ss, vv = rgb_to_hsv(c)
            hh_ += self.rng.normal(0, jitter[0])
            ss = min(1, max(0, ss + self.rng.normal(0, jitter[1])))
            vv = min(1, max(0, vv + self.rng.normal(0, jitter[2])))
            color = hsv_to_rgb(hh_, ss, vv)
            if color_fn:
                color = color_fn(color, region, xi, yi)
            # Direction: along the edge (perpendicular to the gradient), or the region's default.
            if angle_fn is not None:
                ang = angle_fn(region, xi, yi)
            elif gm[yi, xi] > 0.004:
                ang = math.atan2(gx[yi, xi], -gy[yi, xi])
            else:
                ang = math.radians(REG_ANGLE.get(int(region), 0) + self.rng.normal(0, 6))
            dx, dy = math.cos(ang), math.sin(ang)
            if self.rng.random() < 0.5:
                dx, dy = -dx, -dy
            step = max(1.0, r * step_scale)
            px, py = x, y
            for s in range(max_len):
                brush = brushes.get(r, math.atan2(dy, dx))
                self.dab(brush, px, py, color, opacity * (1 - 0.35 * s / max_len))
                nx, ny = px + dx * step, py + dy * step
                ix, iy = int(nx), int(ny)
                if not (0 <= ix < self.w and 0 <= iy < self.h) or self.reg[iy, ix] != region:
                    break
                if s >= min_len and np.abs(ref[iy, ix] - np.asarray(color)).sum() > tol * 3:
                    break
                if not straight and angle_fn is None and gm[iy, ix] > 0.004:
                    a2 = math.atan2(gx[iy, ix], -gy[iy, ix])
                    ndx, ndy = math.cos(a2), math.sin(a2)
                    if ndx * dx + ndy * dy < 0:
                        ndx, ndy = -ndx, -ndy
                    dx, dy = dx * (1 - curvature) + ndx * curvature, dy * (1 - curvature) + ndy * curvature
                    nrm = math.hypot(dx, dy) or 1
                    dx, dy = dx / nrm, dy / nrm
                px, py = nx, ny
            n += 1
        return n

    def impasto(self, strength=0.5, light=(-0.7, -0.7)):
        hgt = blur(np.repeat(np.clip(self.height / 4, 0, 1)[..., None], 3, axis=2), 1.2)[..., 0]
        gy, gx = np.gradient(hgt)
        shade = 1 + strength * 6 * (gx * light[0] + gy * light[1])
        self.canvas = np.clip(self.canvas * shade[..., None], 0, 1)

    def canvas_weave(self, amount=0.04, period=5):
        y, x = np.mgrid[0:self.h, 0:self.w]
        weave = (np.sin(x * 2 * math.pi / period) * np.sin(y * 2 * math.pi / period)) * amount
        noise = (self.rng.random((self.h, self.w)) - 0.5) * amount
        self.canvas = np.clip(self.canvas * (1 + weave + noise)[..., None], 0, 1)


def auto_regions(guide, k=24, seed=3):
    """For any picture: regions are its main colour areas (k-means), so strokes stop at the edges of shapes."""
    rng = np.random.default_rng(seed)
    small = np.asarray(Image.fromarray((guide * 255).astype(np.uint8)).resize((guide.shape[1] // 4, guide.shape[0] // 4)), np.float32).reshape(-1, 3) / 255
    cent = small[rng.choice(len(small), k, replace=False)]
    for _ in range(12):
        lab = ((small[:, None, :] - cent[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                cent[j] = small[m].mean(0)
    flat = guide.reshape(-1, 3)
    out = np.empty(len(flat), np.int32)
    for i in range(0, len(flat), 200000):
        out[i:i + 200000] = ((flat[i:i + 200000, None, :] - cent[None]) ** 2).sum(2).argmin(1)
    return out.reshape(guide.shape[:2]) + 100  # never collides with the named REG ids


def save(img, path):
    Image.fromarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------- styles

def style_oil(guide, reg, rng):
    """Loose oil sketch: warm ground, big strokes following the forms, thick paint catching the light."""
    # Paint the sky without its stars (strokes would smear them), then touch the stars in last.
    starless = np.asarray(Image.fromarray((guide * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(7)), np.float32) / 255
    sky = (reg == REG["sky"])[..., None]
    p = Painter(np.where(sky, starless, guide), reg, rng, (0.36, 0.24, 0.2))
    br = Brushes(rng, aspect=2.4, hard=5, bristle=0.55)
    p.layer(30, br, threshold=0.0, max_len=10, opacity=0.95, jitter=(0.015, 0.06, 0.06))
    p.layer(16, br, threshold=0.06, max_len=9, opacity=0.9, jitter=(0.015, 0.06, 0.05))
    p.layer(8, br, threshold=0.07, max_len=8, opacity=0.9, jitter=(0.01, 0.05, 0.04))
    p.layer(4, br, threshold=0.08, max_len=7, opacity=0.95, jitter=(0.008, 0.04, 0.03))
    p.layer(2, Brushes(rng, aspect=2.0, hard=4, bristle=0.3), threshold=0.12, max_len=6, opacity=1.0, jitter=(0.005, 0.03, 0.02),
            only={REG["figure"], REG["light"], REG["ship"], REG["moon"], REG["outpost"], REG["ramp"]})
    # Stars: single small touches of paint where the guide has a bright point in the sky.
    star = Brushes(rng, aspect=1.0, hard=3, bristle=0.2, angles=4)
    L = lum(guide)
    ys, xs = np.nonzero((L > 0.62) & (reg == REG["sky"]) & (np.arange(reg.shape[0])[:, None] < 380))
    for y, x in zip(ys[::4], xs[::4]):
        p.dab(star.get(1.3, 0), x, y, guide[y, x], 0.8)
    p.impasto(0.55)
    p.canvas_weave(0.035)
    return p.canvas


def style_pixel(guide, reg, rng, w=400, h=240, colors=40):
    """Pixel art: 400x240 with a limited palette, ordered dither in gradients, dark outlines on things."""
    small = np.asarray(Image.fromarray((guide * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS), np.float32) / 255
    rsmall = np.asarray(Image.fromarray(reg.astype(np.uint8)).resize((w, h), Image.NEAREST))
    # Palette by k-means on the small picture.
    px = small.reshape(-1, 3)
    cent = px[rng.choice(len(px), colors, replace=False)]
    for _ in range(14):
        d = ((px[:, None, :] - cent[None]) ** 2).sum(2)
        lab = d.argmin(1)
        for k in range(colors):
            m = lab == k
            if m.any():
                cent[k] = px[m].mean(0)
    # Ordered (Bayer 4x4) dither in the sky and ground only; objects stay flat like hand-placed pixels.
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16 - 0.5
    thr = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
    soft = np.isin(rsmall, [REG["sky"], REG["ground"], REG["dune"], REG["mesa"]])
    jittered = small + (thr * 0.07)[..., None] * soft[..., None]
    d = ((jittered.reshape(-1, 3)[:, None, :] - cent[None]) ** 2).sum(2)
    out = cent[d.argmin(1)].reshape(h, w, 3)
    # Outline objects with a darkened version of their own colour (selective outlining).
    obj = np.isin(rsmall, [REG["ship"], REG["outpost"], REG["figure"], REG["ramp"]])
    edge = np.zeros_like(obj)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = np.roll(np.roll(rsmall, dy, 0), dx, 1)
        edge |= obj & (sh != rsmall) & ~np.isin(sh, [REG["light"]])
    out[edge] = out[edge] * 0.35
    big = np.asarray(Image.fromarray((out * 255).astype(np.uint8)).resize((w * 2, h * 2), Image.NEAREST), np.float32) / 255
    return big


def style_gouache(guide, reg, rng):
    """Gouache illustration: opaque, matte, flat-edged strokes; crisp shapes, no thick paint."""
    p = Painter(guide, reg, rng, (0.95, 0.92, 0.86))
    br = Brushes(rng, aspect=2.0, hard=14, bristle=0.18)
    p.layer(26, br, threshold=0.0, max_len=8, opacity=0.97, jitter=(0.008, 0.03, 0.03))
    p.layer(13, br, threshold=0.05, max_len=7, opacity=0.95, jitter=(0.006, 0.03, 0.025))
    p.layer(6, br, threshold=0.06, max_len=6, opacity=0.95, jitter=(0.005, 0.02, 0.02))
    p.layer(3, br, threshold=0.07, max_len=5, opacity=1.0, jitter=(0.004, 0.02, 0.015))
    p.layer(1.6, Brushes(rng, aspect=1.6, hard=10, bristle=0.1), threshold=0.1, max_len=5, opacity=1.0, jitter=(0.003, 0.01, 0.01),
            only=None if AUTO else {REG["figure"], REG["light"], REG["ship"], REG["moon"], REG["outpost"], REG["ramp"]})
    paper = blur(np.repeat(rng.random(reg.shape)[..., None], 3, 2), 0.8)
    p.canvas = np.clip(p.canvas * (0.97 + 0.06 * paper), 0, 1)
    return p.canvas


AUTO = False
STYLES = {"oil": style_oil, "gouache": style_gouache, "pixel": style_pixel}

if __name__ == "__main__":
    gpath, rpath, style, out = sys.argv[1:5]
    rng = np.random.default_rng(int(sys.argv[5]) if len(sys.argv) > 5 else 7)
    guide = load(gpath)
    AUTO = rpath == "auto"
    reg = auto_regions(guide) if AUTO else np.asarray(Image.open(rpath).convert("RGB"))[..., 0].astype(np.int32)
    save(STYLES[style](guide, reg, rng), out)
