#!/usr/bin/env python3
"""Prepare the birthday reference photos for FLUX.2 (DECISIONS #85).

FLUX.2 klein takes up to four reference images, each smaller than 512x512. Shrinking a whole
phone photo to that size leaves the face a few dozen pixels wide, and the old resize also lost
the phone's rotation flag, so some references reached the model sideways. This script:

  1. turns every photo upright (EXIF orientation),
  2. finds the faces with Apple's Vision framework (scripts/portrait-faces.swift),
  3. crops head and shoulders around the chosen face (the largest, unless picks.json says
     otherwise: {"alvaro_3": 1} picks the second-largest face, for group photos, and
     {"lucas_2": {"face": 0, "rotate": 180}} also turns the crop, for a face upside down in the scene),
  4. shrinks the crop to 511px on its long side and writes {key}_{i}.jpg,
  5. draws a contact sheet (every face numbered, and the crop) to check the picks by eye.

Photos never enter the repo (photos/ is gitignored). Run with Pillow, macOS only:

  python3 -E -P scripts/prepare-portraits.py --src photos/portraits --out /tmp/portraits
  python3 -E -P scripts/prepare-portraits.py --src photos/portraits --out /tmp/portraits --upload

--upload puts each prepared photo in R2 (bucket eink-birthday-photos, key portraits/{key}_{i}.jpg),
replacing what is there; keep a copy of the old objects first.
"""
import argparse
import json
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageOps

NAME = re.compile(r"^([a-z]+)_(\d)\.(jpe?g)$", re.IGNORECASE)
LONG_SIDE = 511          # "smaller than 512x512"
BUCKET = "eink-birthday-photos"
HERE = os.path.dirname(os.path.abspath(__file__))


def head_and_shoulders(face, img_w, img_h, width=2.4):
    """A 4:5 box around the face: room above the hair, shoulders below. `width` is in face widths."""
    cw = face["w"] * width
    ch = cw * 1.25
    cx = face["x"] + face["w"] / 2
    top = face["y"] - face["h"] * 0.6
    # Shrink to fit the photo, keeping the shape, then slide inside it.
    scale = min(1.0, img_w / cw, img_h / ch)
    cw, ch = cw * scale, ch * scale
    left = min(max(cx - cw / 2, 0), img_w - cw)
    top = min(max(top, 0), img_h - ch)
    return (round(left), round(top), round(left + cw), round(top + ch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--picks", help="JSON {\"name_i\": face_index} for group photos")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    picks = {}
    if args.picks and os.path.exists(args.picks):
        picks = json.load(open(args.picks))

    upright_dir = os.path.join(args.out, "upright")
    os.makedirs(upright_dir, exist_ok=True)
    items = []
    for fn in sorted(os.listdir(args.src)):
        m = NAME.match(fn)
        if not m:
            continue
        stem = f"{m.group(1).lower()}_{m.group(2)}"
        img = ImageOps.exif_transpose(Image.open(os.path.join(args.src, fn))).convert("RGB")
        up = os.path.join(upright_dir, stem + ".jpg")
        img.save(up, quality=95)
        items.append((stem, up))

    res = subprocess.run(["swift", os.path.join(HERE, "portrait-faces.swift")] + [u for _, u in items],
                         capture_output=True, text=True, check=True)
    faces = {os.path.basename(j["file"])[:-4]: j for j in map(json.loads, res.stdout.splitlines())}

    th = 260
    sheet = Image.new("RGB", (th * 2, th * len(items)), "white")
    written = []
    for row, (stem, up) in enumerate(items):
        img = Image.open(up)
        found = faces[stem].get("faces", [])
        pick = picks.get(stem, 0)
        idx, rotate = (pick.get("face", 0), pick.get("rotate", 0)) if isinstance(pick, dict) else (pick, 0)
        if not found:
            print(f"{stem}: NO FACE FOUND, centre crop", file=sys.stderr)
            side = min(img.size)
            box = ((img.width - side) // 2, (img.height - side) // 2,
                   (img.width + side) // 2, (img.height + side) // 2)
        else:
            face = found[min(idx, len(found) - 1)]
            # Tighter in a group photo, so a neighbour's face stays out of the reference.
            box = head_and_shoulders(face, img.width, img.height, 2.4 if len(found) == 1 else 1.6)
        crop = img.crop(box).rotate(rotate, expand=True)
        crop.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
        out = os.path.join(args.out, stem + ".jpg")
        crop.save(out, quality=92)
        written.append((stem, out))
        print(f"{stem}: {len(found)} face(s), picked {idx}, crop {box} -> {crop.size}")

        # Contact sheet: the photo with numbered faces, then the crop.
        preview = img.copy()
        draw = ImageDraw.Draw(preview)
        lw = max(3, img.width // 150)
        for n, f in enumerate(found):
            colour = "lime" if n == idx else "red"
            draw.rectangle([f["x"], f["y"], f["x"] + f["w"], f["y"] + f["h"]], outline=colour, width=lw)
            draw.text((f["x"], f["y"] + f["h"]), str(n), fill=colour, font_size=max(20, img.width // 15))
        draw.rectangle(box, outline="yellow", width=lw)
        preview.thumbnail((th, th))
        sheet.paste(preview, (0, row * th))
        c = crop.copy()
        c.thumbnail((th, th))
        sheet.paste(c, (th, row * th))
        ImageDraw.Draw(sheet).text((th + 4, row * th + 4), stem, fill="magenta", font_size=18)
    sheet.save(os.path.join(args.out, "sheet.jpg"), quality=85)
    print(f"sheet: {os.path.join(args.out, 'sheet.jpg')}")

    if args.upload:
        for stem, path in written:
            subprocess.run(["npx", "wrangler", "r2", "object", "put", f"{BUCKET}/portraits/{stem}.jpg",
                            "--file", path, "--content-type", "image/jpeg", "--remote"], check=True)
        print(f"uploaded {len(written)} photo(s)")


if __name__ == "__main__":
    main()
