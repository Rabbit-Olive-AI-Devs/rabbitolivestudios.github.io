"""The Long Drift: the 60-second series teaser, drawn and scored by code (DECISIONS #77,
docs/series/TEASER.md). numpy + PIL; ffmpeg joins the result.

  TEASER_DIR=~/series-work/teaser python3 -E -P teaser.py audio     -> score.wav (48 kHz stereo)
  TEASER_DIR=~/series-work/teaser python3 -E -P teaser.py frames    -> frames/f00000.jpg ... (1920x1080, 24 fps)
  TEASER_DIR=~/series-work/teaser python3 -E -P teaser.py still 21  -> still.jpg at 21 s, to check a moment
  ffmpeg -framerate 24 -i frames/f%05d.jpg -i score.wav -c:v libx264 -b:v 7M -maxrate 10M -bufsize 14M \
         -pix_fmt yuv420p -c:a aac -b:a 256k -movflags +faststart -shortest teaser.mp4

The pictures are not in this repo: TEASER_DIR/assets/ holds them (a copy is in the Dropbox series
folder, teaser/assets/). Only story-free pictures: style tests and the non-canon test scene.
"""
import math
import os
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.expanduser(os.environ.get("TEASER_DIR", "~/series-work/teaser"))
A = os.path.join(HERE, "assets")
W, H, FPS, DUR = 1920, 1080, 24, 60.0
BAR = 138                      # letterbox: 2.39:1 picture area
PH = H - 2 * BAR
OPTIMA = "/System/Library/Fonts/Optima.ttc"
AVENIR = "/System/Library/Fonts/Avenir Next Condensed.ttc"

# ---------------------------------------------------------------- helpers


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def env(t, a, b, fin=0.6, fout=0.6):
    """1 inside [a, b], fading in over fin and out over fout; 0 outside."""
    if t < a or t > b:
        return 0.0
    return min(ease((t - a) / fin) if fin > 0 else 1.0, ease((b - t) / fout) if fout > 0 else 1.0)


def text_layer(lines, font_path, size, tracking=0.25, index=0, color=(236, 228, 210), gap=1.5):
    """Tracked capitals, centred, as an RGBA image with a soft glow baked in."""
    font = ImageFont.truetype(font_path, size, index=index)
    widths = []
    for line in lines:
        w = sum(font.getlength(c) for c in line) + tracking * size * max(0, len(line) - 1)
        widths.append(w)
    lw, lh = int(max(widths)) + 40, int(len(lines) * size * gap) + 40
    img = Image.new("L", (lw, lh), 0)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        x = (lw - widths[i]) / 2
        y = 20 + i * size * gap
        for c in line:
            d.text((x, y), c, font=font, fill=255)
            x += font.getlength(c) + tracking * size
    a = np.asarray(img, np.float32) / 255
    glow = np.asarray(img.filter(ImageFilter.GaussianBlur(size * 0.35)), np.float32) / 255
    alpha = np.clip(a + glow * 0.45, 0, 1)
    rgb = np.ones((lh, lw, 3), np.float32) * np.array(color, np.float32) / 255
    # A soft dark halo, so words read over bright pictures.
    pad = int(size * 1.2)
    big = Image.new("L", (lw + 2 * pad, lh + 2 * pad), 0)
    big.paste(img, (pad, pad))
    shadow = np.asarray(big.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(size * 0.7)), np.float32) / 255
    shadow = np.clip(shadow / max(shadow.max(), 1e-6) * 1.3, 0, 1)
    return rgb, alpha, shadow


def paste(canvas, layer, cx, cy, opacity):
    if opacity <= 0.001:
        return
    rgb, alpha, shadow = layer
    sh, sw = shadow.shape
    x0, y0 = int(cx - sw / 2), int(cy - sh / 2)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = sw - max(0, x0 + sw - W), sh - max(0, y0 + sh - H)
    canvas[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1] *= 1 - 0.7 * opacity * shadow[sy0:sy1, sx0:sx1, None]
    h, w = alpha.shape
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = x0 + w, y0 + h
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = w - max(0, x1 - W), h - max(0, y1 - H)
    a = alpha[sy0:sy1, sx0:sx1, None] * opacity
    sl = (slice(y0 + sy0, y0 + sy1), slice(x0 + sx0, x0 + sx1))
    canvas[sl] = canvas[sl] * (1 - a) + rgb[sy0:sy1, sx0:sx1] * a


def kenburns(img, p, z0, z1, c0, c1):
    """A crop of `img` (PIL) at zoom z (1 = the widest 2.39:1 crop), centre moving c0 -> c1, as a float array PH x W."""
    iw, ih = img.size
    aspect = W / PH
    bw, bh = (iw, iw / aspect) if iw / ih < aspect else (ih * aspect, ih)
    z = z0 + (z1 - z0) * p
    cw, ch = bw / z, bh / z
    cx = (c0[0] + (c1[0] - c0[0]) * p) * iw
    cy = (c0[1] + (c1[1] - c0[1]) * p) * ih
    cx = min(max(cx, cw / 2), iw - cw / 2)
    cy = min(max(cy, ch / 2), ih - ch / 2)
    box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    return np.asarray(img.resize((W, PH), Image.BICUBIC, box=box), np.float32) / 255


# ---------------------------------------------------------------- assets (loaded once per worker)

G = {}


def init():
    rng = np.random.default_rng(7)
    G["img"] = {k: Image.open(os.path.join(A, f"{k}.png")).convert("RGB") for k in
                ["claude-oil", "claude-gouache", "claude-pixel", "zoe", "volt", "pepper", "muse"]}
    G["img"]["claude-pixel"] = G["img"]["claude-pixel"].resize((1600, 960), Image.NEAREST)
    # Stars: three depths.
    G["stars"] = [(rng.random(n) * W * 1.6, rng.random(n) * H, rng.uniform(0.25, 1.0, n), depth)
                  for n, depth in ((900, 0.25), (350, 0.6), (90, 1.2))]
    # Nebula: soft coloured clouds, panned slowly.
    neb = rng.random((54, 96, 3)).astype(np.float32)
    nebimg = Image.fromarray((neb * 255).astype(np.uint8)).resize((W * 2, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(60))
    n = np.asarray(nebimg, np.float32) / 255
    G["nebula"] = np.clip((n - 0.45) * 1.6, 0, 1) * np.array([0.30, 0.12, 0.10], np.float32) + np.clip((n[..., ::-1] - 0.5) * 1.4, 0, 1) * np.array([0.05, 0.12, 0.22], np.float32)
    G["grain"] = [rng.normal(0, 0.035, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
    y, x = np.mgrid[0:H, 0:W]
    G["vignette"] = (1 - 0.55 * (((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2) ** 1.4).clip(0.25, 1).astype(np.float32)[..., None]
    T = lambda lines, size, **k: text_layer(lines, OPTIMA, size, **k)
    S = lambda lines, size, **k: text_layer(lines, AVENIR, size, index=0, **k)
    G["txt"] = {
        "after": T(["AFTER THE EMPIRE FELL"], 46),
        "rim": T(["THE OUTER RIM BELONGED TO NO ONE"], 46),
        "ship": T(["ONE SHIP"], 52),
        "lantern": T(["THE LAST LANTERN"], 58, tracking=0.35),
        "crew1": T(["A PILOT WITH DEBTS"], 40),
        "crew2": T(["A DROID WITH OPINIONS"], 40),
        "crew3": T(["A MECHANIC WITH A PAST"], 40),
        "title": text_layer(["THE LONG DRIFT"], OPTIMA, 112, tracking=0.3, index=1, color=(245, 236, 214)),
        "one": T(["ONE STORY. FIVE SEASONS."], 50),
        "four": T(["FOUR AUTHORS. EACH IN THEIR OWN HAND."], 44),
        "claude": T(["CLAUDE"], 96, tracking=0.4, index=1),
        "claude2": S(["THE PROLOGUE  ·  PAINTED STROKE BY STROKE, IN CODE"], 34, tracking=0.18),
        "zoe": T(["ZOE"], 96, tracking=0.4, index=1),
        "zoe2": S(["SEASON ONE"], 34, tracking=0.3),
        "volt": T(["VOLT  &  INKY"], 96, tracking=0.3, index=1),
        "volt2": S(["SEASON TWO"], 34, tracking=0.3),
        "pepper": T(["PEPPER"], 96, tracking=0.4, index=1),
        "pepper2": S(["SEASON THREE"], 34, tracking=0.3),
        "claude4": S(["AND CLAUDE RETURNS  ·  SEASON FOUR"], 34, tracking=0.18),
        "every": T(["EVERY SCENE ITS OWN STYLE"], 50),
        "walls": T(["THREE E-INK SCREENS", "A NEW SCENE EVERY TWO HOURS"], 42, gap=1.9),
        "prem": T(["THE PROLOGUE"], 54, tracking=0.35),
        "date": text_layer(["MONDAY 26 OCTOBER 2026"], OPTIMA, 70, tracking=0.22, index=1, color=(245, 236, 214)),
        "seasons": S(["SEASON ONE  4 JAN 2027      SEASON TWO  5 APR 2027", "SEASON THREE  5 JUL 2027      SEASON FOUR  4 OCT 2027"], 34, tracking=0.14, color=(222, 212, 192), gap=1.6),
        "small": text_layer(["THE LONG DRIFT"], OPTIMA, 64, tracking=0.32, index=1, color=(245, 236, 214)),
    }


def stars(canvas, t, opacity=1.0, drift=18.0):
    for xs, ys, b, depth in G["stars"]:
        x = ((xs - t * drift * depth) % (W * 1.6)).astype(int)
        m = x < W
        yy = ys[m].astype(int)
        tw = b[m] * (0.75 + 0.25 * np.sin(t * 3 + xs[m]))
        v = (tw * opacity)[:, None] * np.array([0.95, 0.93, 1.0], np.float32)
        canvas[yy, x[m]] = np.maximum(canvas[yy, x[m]], v)
        if depth > 1:
            canvas[np.clip(yy + 1, 0, H - 1), x[m]] = np.maximum(canvas[np.clip(yy + 1, 0, H - 1), x[m]], v * 0.6)


def picture(canvas, key, p, z0, z1, c0, c1, opacity):
    if opacity <= 0.001:
        return
    img = kenburns(G["img"][key], p, z0, z1, c0, c1)
    canvas[BAR:BAR + PH] = canvas[BAR:BAR + PH] * (1 - opacity) + img * opacity


# ---------------------------------------------------------------- the cut

AUTHORS = [  # start, picture, name, line, drift
    (27.5, "claude-gouache", "claude", "claude2", ((0.45, 0.55), (0.55, 0.5))),
    (31.5, "zoe", "zoe", "zoe2", ((0.4, 0.5), (0.55, 0.45))),
    (35.5, "volt", "volt", "volt2", ((0.35, 0.55), (0.5, 0.5))),
    (39.5, "muse", "pepper", "pepper2", ((0.65, 0.5), (0.5, 0.45))),
]
MONTAGE = ["claude-oil", "pepper", "volt", "claude-pixel", "zoe", "muse", "claude-gouache", "volt", "claude-oil", "zoe", "muse", "claude-pixel"]
MONTAGE_T = [43.6 + 4.6 * (1 - (1 - i / len(MONTAGE)) ** 1.6) for i in range(len(MONTAGE) + 1)]  # cuts quicken


def frame(t):
    c = np.zeros((H, W, 3), np.float32)
    tx = G["txt"]
    mid = H / 2
    # 0-9: space.
    if t < 10.5:
        stars(c, t, env(t, 0.3, 10.5, 2.5, 1.5))
        c += G["nebula"][:, int(t * 12):int(t * 12) + W] * env(t, 0.5, 10.5, 3, 1.5)
        paste(c, tx["after"], W / 2, mid, env(t, 1.6, 4.8, 0.9, 0.9))
        paste(c, tx["rim"], W / 2, mid, env(t, 5.4, 8.8, 0.9, 1.0))
    # 9-15.5: the ship, oil at dusk.
    o = env(t, 9.2, 15.6, 1.6, 0.25)
    if o:
        picture(c, "claude-oil", (t - 9.2) / 6.4, 1.0, 1.18, (0.42, 0.62), (0.55, 0.6), o)
        paste(c, tx["ship"], W / 2, H - BAR - 90, env(t, 10.4, 12.2, 0.6, 0.5))
        paste(c, tx["lantern"], W / 2, H - BAR - 90, env(t, 12.6, 15.4, 0.6, 0.4))
    # 15.5-19.5: the crew, gouache, slow pan.
    o = env(t, 15.6, 19.5, 0.15, 0.5)
    if o:
        picture(c, "claude-gouache", (t - 15.6) / 3.9, 1.35, 1.5, (0.33, 0.72), (0.45, 0.72), o * 0.85)
        for i, k in enumerate(["crew1", "crew2", "crew3"]):
            paste(c, tx[k], W / 2, BAR + 110 + i * 70, env(t, 16.0 + i * 0.9, 19.3, 0.5, 0.4))
    # 19.6-23.4: the title.
    o = env(t, 19.6, 23.5, 0.08, 0.8)
    if o:
        stars(c, t, 0.5 * o, drift=6)
        flick = 1.0 if t > 19.9 else 0.6 + 0.4 * math.sin(t * 90)
        paste(c, tx["title"], W / 2, mid, o * flick)
    # 23.5-27.4: the premise.
    o = env(t, 23.6, 27.4, 0.5, 0.3)
    if o:
        stars(c, t, 0.4 * o, drift=10)
        paste(c, tx["one"], W / 2, mid - 40, env(t, 23.7, 27.3, 0.6, 0.3))
        paste(c, tx["four"], W / 2, mid + 40, env(t, 25.2, 27.3, 0.6, 0.3))
    # 27.5-43.5: one author each.
    for s, key, name, line, (c0, c1) in AUTHORS:
        o = env(t, s, s + 4.0, 0.12, 0.25)
        if o:
            picture(c, key, (t - s) / 4.0, 1.08, 1.22, c0, c1, o)
            c[BAR:BAR + PH] *= 1 - 0.35 * o * np.linspace(0, 1, PH)[:, None, None] ** 2  # darken the lower third for the name
            paste(c, tx[name], W / 2, H - BAR - 150, env(t, s + 0.3, s + 3.9, 0.4, 0.25))
            paste(c, tx[line], W / 2, H - BAR - 70, env(t, s + 0.8, s + 3.9, 0.4, 0.25))
            if name == "pepper":
                paste(c, tx["claude4"], W / 2, BAR + 60, env(t, s + 2.2, s + 3.9, 0.4, 0.25))
    # 43.5-48.3: the montage, cuts quickening.
    if 43.6 <= t < 48.3:
        i = max(j for j in range(len(MONTAGE)) if MONTAGE_T[j] <= t)
        a, b = MONTAGE_T[i], MONTAGE_T[i + 1]
        picture(c, MONTAGE[i], (t - a) / (b - a), 1.25, 1.4, (0.5, 0.5), (0.5 + 0.05 * (-1) ** i, 0.48), 1.0)
        paste(c, tx["every"], W / 2, mid, env(t, 44.0, 48.2, 0.3, 0.2))
    # 48.3-52.4: the screens, quiet.
    o = env(t, 48.4, 52.4, 0.6, 0.4)
    if o:
        stars(c, t, 0.35 * o, drift=4)
        paste(c, tx["walls"], W / 2, mid, o)
    # 52.5-60: the date.
    o = env(t, 52.5, 59.6, 0.06, 1.4)
    if o:
        stars(c, t, 0.45 * o, drift=3)
        paste(c, tx["small"], W / 2, mid - 150, o)
        paste(c, tx["prem"], W / 2, mid - 40, env(t, 52.9, 59.6, 0.5, 1.4))
        paste(c, tx["date"], W / 2, mid + 50, env(t, 53.4, 59.6, 0.5, 1.4))
        paste(c, tx["seasons"], W / 2, mid + 175, env(t, 55.0, 59.6, 0.8, 1.4) * 0.9)
    # Hit flashes on the big beats.
    for hit in [19.6, 27.5, 31.5, 35.5, 39.5, 52.5]:
        if hit <= t < hit + 0.25:
            c += (1 - (t - hit) / 0.25) * 0.35
    # Film: vignette, grain, letterbox.
    c *= G["vignette"]
    g = G["grain"][int(t * FPS) % len(G["grain"])]
    c += np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]
    c[:BAR] = 0
    c[H - BAR:] = 0
    return np.clip(c, 0, 1)


def render(i):
    out = os.path.join(HERE, "frames", f"f{i:05d}.jpg")
    if os.path.exists(out):
        return
    Image.fromarray((frame(i / FPS) * 255 + 0.5).astype(np.uint8)).save(out, quality=93)


# ---------------------------------------------------------------- the score

SR = 48000


def score():
    n = int(SR * DUR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(3)
    L = np.zeros(n)
    R = np.zeros(n)

    def add(sig, start, pan=0.0, gain=1.0):
        i = int(start * SR)
        j = min(n, i + len(sig))
        L[i:j] += sig[:j - i] * gain * (1 - max(0, pan))
        R[i:j] += sig[:j - i] * gain * (1 + min(0, pan))

    def lowpass(x, cutoff):
        """One-pole low-pass; cutoff may be an array (Hz per sample)."""
        cut = np.broadcast_to(np.asarray(cutoff, float), x.shape)
        a = 1 - np.exp(-2 * np.pi * cut / SR)
        y = np.empty_like(x)
        acc = 0.0
        for k in range(len(x)):
            acc += a[k] * (x[k] - acc)
            y[k] = acc
        return y

    def saw(f, tt, ph=0.0):
        return 2 * ((tt * f + ph) % 1) - 1

    # Drone in D: grows through the first half, falls away under the title, returns lower.
    dr = np.zeros(n)
    for f, g in ((36.71, 1.0), (73.42, 0.55), (110.0, 0.25), (55.0, 0.2)):
        dr += g * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 0.07 * t))
    shape = np.interp(t, [0, 3, 19.4, 19.6, 23.5, 27, 43, 48.3, 48.6, 52.4, 56, 60], [0, 0.35, 0.8, 0.3, 0.35, 0.45, 0.6, 0.25, 0.25, 0.4, 0.5, 0])
    L += dr * shape * 0.32
    R += dr * shape * 0.32
    wind = lowpass(rng.normal(0, 1, n), 380) * np.interp(t, [0, 4, 10, 15, 60], [0, 0.5, 0.35, 0.15, 0.1])
    L += wind * 0.6
    R += np.roll(wind, 900) * 0.6

    def thud(f0=58, f1=32, dur=1.4):
        tt = np.arange(int(SR * dur)) / SR
        f = f1 + (f0 - f1) * np.exp(-tt * 9)
        ph = 2 * np.pi * np.cumsum(f) / SR
        return np.sin(ph) * np.exp(-tt * 3.2) + 0.25 * lowpass(rng.normal(0, 1, len(tt)), 900) * np.exp(-tt * 18)

    for s in [6.0, 8.6, 11.0, 13.0, 14.8, 16.3, 17.5, 18.4, 19.0]:
        add(thud(), s, gain=0.9)

    def braam(dur=5.0):
        tt = np.arange(int(SR * dur)) / SR
        x = np.zeros_like(tt)
        for f in (36.71, 73.42, 110.0, 146.83):
            for d in (-0.35, 0.0, 0.4):
                x += saw(f * (1 + d / 100), tt, rng.random())
        cut = 180 + 1500 * np.exp(-tt * 1.6) * (1 - np.exp(-tt * 30))
        y = lowpass(lowpass(x, cut), cut)
        return np.tanh(y * 0.9) * np.exp(-tt * 0.75) * (1 - np.exp(-tt * 40))

    def riser(dur):
        tt = np.arange(int(SR * dur)) / SR
        noise = rng.normal(0, 1, len(tt))
        hp = noise - lowpass(noise, 300 + 4000 * (tt / dur) ** 2)
        tone = np.sin(2 * np.pi * np.cumsum(200 + 1400 * (tt / dur) ** 3) / SR)
        return (hp * 0.35 + tone * 0.15) * (tt / dur) ** 2.2

    add(riser(2.6), 17.0, gain=0.7)
    b = braam()
    add(b, 19.6, gain=1.0)
    add(thud(70, 30, 2.5), 19.6, gain=1.2)

    # Ostinato: low plucked saws in 8ths, D minor, from the premise through the montage.
    def pluck(f, dur=0.32):
        tt = np.arange(int(SR * dur)) / SR
        x = saw(f, tt) + 0.5 * saw(f * 2.003, tt)
        return lowpass(x, 300 + 2200 * np.exp(-tt * 18)) * np.exp(-tt * 9)

    notes = [146.83, 146.83, 174.61, 146.83, 220.0, 146.83, 174.61, 164.81]
    plucks = {f: pluck(f) for f in set(notes)}
    beat = 0.25
    k = 0
    s = 23.5
    while s < 48.2:
        lvl = np.interp(s, [23.5, 27.5, 43.5, 48.2], [0.18, 0.32, 0.42, 0.55])
        add(plucks[notes[k % 8]], s, pan=0.25 * (-1) ** k, gain=lvl)
        if s >= 43.5:  # double time in the montage
            add(plucks[notes[(k + 4) % 8]] , s + beat / 2, pan=-0.25 * (-1) ** k, gain=lvl * 0.7)
        s += beat
        k += 1
    for s in [27.5, 31.5, 35.5, 39.5]:
        add(thud(66, 34, 1.8), s, gain=1.0)
        add(braam(2.4) * 0.5, s, gain=0.55)
    add(riser(4.6), 43.6, gain=0.9)
    for s in MONTAGE_T[:-1]:
        add(thud(90, 45, 0.5), s, gain=0.45)

    # The quiet: a high tense tone over the screens.
    q0, q1 = 48.3, 52.45
    tt = np.arange(int(SR * (q1 - q0))) / SR
    tone = (np.sin(2 * np.pi * 587.33 * tt) + 0.6 * np.sin(2 * np.pi * 880 * tt)) * (0.6 + 0.4 * np.sin(2 * np.pi * 6 * tt))
    add(tone * np.interp(tt, [0, 1.0, q1 - q0 - 0.4, q1 - q0], [0, 0.07, 0.12, 0]), q0)
    add(riser(1.6), 50.9, gain=0.8)

    # The last hit, and the ring-out.
    add(braam(7.0), 52.5, gain=1.15)
    add(thud(70, 28, 3.5), 52.5, gain=1.3)
    for s, g in [(54.6, 0.55), (56.2, 0.4)]:
        add(thud(60, 30, 2.5), s, gain=g)

    # Hall reverb by convolution, then a soft limiter.
    ir_t = np.arange(int(SR * 2.8)) / SR
    def ir(seed):
        r = np.random.default_rng(seed).normal(0, 1, len(ir_t)) * np.exp(-ir_t * 2.4)
        return lowpass(r, 5000) * 0.02
    size = 1 << int(np.ceil(np.log2(n + len(ir_t))))
    def conv(x, h):
        return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)[:n]
    L, R = L + conv(L, ir(11)) * 0.9, R + conv(R, ir(12)) * 0.9
    fade = np.interp(t, [0, 0.05, 59.0, 60.0], [0, 1, 1, 0])
    out = np.stack([L, R], 1) * fade[:, None]
    out = np.tanh(out / (np.abs(out).max() * 0.55)) * 0.92
    with wave.open(os.path.join(HERE, "score.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((out * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "still":
        init()
        Image.fromarray((frame(float(sys.argv[2])) * 255).astype(np.uint8)).save(os.path.join(HERE, "still.jpg"), quality=90)
    elif what == "frames":
        os.makedirs(os.path.join(HERE, "frames"), exist_ok=True)
        with Pool(9, initializer=init) as pool:
            pool.map(render, range(int(DUR * FPS)), chunksize=8)
    elif what == "audio":
        score()
