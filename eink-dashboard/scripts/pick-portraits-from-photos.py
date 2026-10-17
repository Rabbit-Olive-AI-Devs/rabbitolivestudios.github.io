#!/usr/bin/env python3
"""Pick birthday reference photos from the owner's Photos library (DECISIONS #87).

Apple Photos already knows who is in each picture (the People album) and, per face, its size,
pose, eyes, blur and glasses. This script reads a COPY of the library database, read-only, and
for each family member lists their best recent close-ups:

  - the person named in Photos, mapped to the birthday key in photos/portraits/people.json:
      {"thiago": "Thiago Borges Oliveira", "sonia": "Mãe"}   (Photos full or display name)
  - no sunglasses (glasses type 2); prescription glasses are fine
  - eyes open, face turned towards the camera, sharp, large in the frame
  - preferably alone in the picture
  - recent: children change fast, so the window is short (--years, default 2)
  - at most one picture per day, so the choices are not one occasion

A photo kept only in iCloud (optimised storage) is exported through Photos with AppleScript,
which downloads the original; macOS asks once to let the terminal control Photos.
It converts the top candidates to JPEG (sips handles HEIC) into OUT/<key>/ with a contact sheet.
A person then chooses up to four, copies them to photos/portraits/<key>_0..3.jpg and runs
`npm run upload-photos`. Photos never enter the repo.

  cp "<library>/database/Photos.sqlite"* /tmp/photosdb/
  python3 -E -P scripts/pick-portraits-from-photos.py --db /tmp/photosdb/Photos.sqlite \
    --library "/Volumes/SamsungSSD/Pictures/Photos Library.photoslibrary" --out /tmp/candidates
"""
import argparse
import json
import os
import sqlite3
import subprocess
import time

from PIL import Image, ImageDraw, ImageOps

APPLE_EPOCH = 978307200  # 2001-01-01 in Unix time
SUNGLASSES = 2


def export_original(uuid, folder):
    """Ask Photos to export the full original (downloading it from iCloud if needed)."""
    os.makedirs(folder, exist_ok=True)
    cached = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".heic", ".png"))]
    if cached:
        return os.path.join(folder, cached[0])
    script = (f'with timeout of 180 seconds\ntell application "Photos"\n'
              f'export {{media item id "{uuid}/L0/001"}} to (POSIX file "{folder}") with using originals\n'
              f'end tell\nend timeout')
    subprocess.run(["osascript", "-e", script], capture_output=True)
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".heic", ".png"))]
    return os.path.join(folder, files[0]) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="a copy of Photos.sqlite (never the live one)")
    ap.add_argument("--library", required=True)
    ap.add_argument("--people", default="photos/portraits/people.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--count", type=int, default=12)
    args = ap.parse_args()

    people = json.load(open(args.people))
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    since = time.time() - args.years * 365.25 * 86400 - APPLE_EPOCH

    for key, name in people.items():
        row = db.execute("SELECT Z_PK FROM ZPERSON WHERE ZFULLNAME = ? OR ZDISPLAYNAME = ? "
                         "ORDER BY ZFACECOUNT DESC LIMIT 1", (name, name)).fetchone()
        if not row:
            print(f"{key}: no person named {name!r} in Photos")
            continue
        rows = db.execute("""
            SELECT a.ZDIRECTORY, a.ZFILENAME, a.ZDATECREATED, f.ZSIZE, f.ZQUALITY, f.ZBLURSCORE, f.ZPOSEYAW,
                   (SELECT COUNT(*) FROM ZDETECTEDFACE g WHERE g.ZASSETFORFACE = a.Z_PK AND g.ZHIDDEN = 0) AS faces,
                   a.ZUUID, f.ZCENTERX, f.ZCENTERY
            FROM ZDETECTEDFACE f JOIN ZASSET a ON a.Z_PK = f.ZASSETFORFACE
            WHERE f.ZPERSONFORFACE = ? AND f.ZHIDDEN = 0 AND f.ZISINTRASH = 0
              AND f.ZGLASSESTYPE <> ? AND f.ZEYESSTATE = 2
              AND a.ZTRASHEDSTATE = 0 AND a.ZHIDDEN = 0 AND a.ZKIND = 0
              AND a.ZDATECREATED > ? AND f.ZSIZE > 0.12 AND ABS(COALESCE(f.ZPOSEYAW, 0)) < 0.35
        """, (row[0], SUNGLASSES, since)).fetchall()

        def score(r):
            size, quality, blur, faces = r[3], r[4] or 0, r[5] or 0, r[7]
            return size * 2 + quality + blur * 0.5 - (0.4 if faces > 1 else 0)

        picked, days = [], set()
        for r in sorted(rows, key=score, reverse=True):
            day = time.strftime("%Y-%m-%d", time.gmtime(r[2] + APPLE_EPOCH))
            if day in days:
                continue
            src = os.path.join(args.library, "originals", r[0], r[1])
            if not os.path.exists(src):
                # Optimised storage keeps only a 480px preview here; Photos fetches the original.
                src = export_original(r[8], os.path.join(args.out, ".export", r[8]))
                if not src:
                    continue
            days.add(day)
            picked.append((day, src, r[7], (r[9], r[10])))
            if len(picked) >= args.count:
                break

        out = os.path.join(args.out, key)
        os.makedirs(out, exist_ok=True)
        for old in os.listdir(out):  # a fresh list each run
            os.remove(os.path.join(out, old))
        thumbs, hints = [], {}
        for n, (day, src, faces, centre) in enumerate(picked):
            dst = os.path.join(out, f"{n:02d}_{day}.jpg")
            hints[os.path.basename(dst)] = centre
            subprocess.run(["sips", "-s", "format", "jpeg", "-Z", "2048", src, "--out", dst],
                           capture_output=True, check=True)
            thumbs.append((dst, f"{n} {day}{' group' if faces > 1 else ''}"))
        th = 240
        sheet = Image.new("RGB", (th * 6, th * ((len(thumbs) + 5) // 6 or 1)), "white")
        for n, (path, label) in enumerate(thumbs):
            im = ImageOps.exif_transpose(Image.open(path))
            im.thumbnail((th, th))
            sheet.paste(im, ((n % 6) * th, (n // 6) * th))
            ImageDraw.Draw(sheet).text(((n % 6) * th + 4, (n // 6) * th + 4), label, fill="magenta", font_size=16)
        sheet.save(os.path.join(out, "sheet.jpg"), quality=85)
        # Where Photos saw this person's face (its own normalised coordinates), so the crop can
        # take their face rather than the largest one in a group photo.
        json.dump(hints, open(os.path.join(out, "hints.json"), "w"), indent=1)
        print(f"{key}: {len(rows)} usable faces, {len(picked)} candidates -> {out}/sheet.jpg")


if __name__ == "__main__":
    main()
