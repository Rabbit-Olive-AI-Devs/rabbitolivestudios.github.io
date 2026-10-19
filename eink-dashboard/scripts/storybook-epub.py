#!/usr/bin/env python3
"""
Builds the desk story book (DECISIONS #95) as an EPUB from the frames the storybook job saved
in Dropbox: one chapter per day, frames in time order. Frames ten minutes apart often differ
only in the top bar's clock, so a frame is kept only when the picture below the bar changed
from the last one kept. Prints counts only: the story is a surprise (#93).

    python3 -E -P scripts/storybook-epub.py [FOLDER] [OUT.epub]
"""

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

FOLDER = Path(sys.argv[1] if len(sys.argv) > 1 else
              "/Volumes/SamsungSSD/.CloudStorage/Data/Dropbox/00_STORYBOOK_SURPRISE_do_not_open")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else FOLDER / "storybook.epub"
TOP_BAR = 40          # rows ignored when comparing: the shared top bar holds the clock
MIN_CHANGE = 0.002    # fraction of pixels below the bar that must differ to keep a frame
TITLE = "The Desk Story"


def frames(day_dir: Path) -> list[dict]:
    rows = [json.loads(line) for line in (day_dir / "index.jsonl").read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["at"])


def changed(a: np.ndarray, b: np.ndarray) -> bool:
    return np.count_nonzero(a[TOP_BAR:] != b[TOP_BAR:]) / a[TOP_BAR:].size >= MIN_CHANGE


def page(title: str, body: str) -> str:
    return ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
            f'<head><meta charset="utf-8"/><title>{title}</title><link rel="stylesheet" href="book.css"/></head>'
            f'<body>{body}</body></html>')


def main() -> None:
    days = sorted(d for d in FOLDER.iterdir() if d.is_dir() and (d / "index.jsonl").exists())
    chapters, images, total, kept_total = [], [], 0, 0
    for d in days:
        kept, last = [], None
        for r in frames(d):
            total += 1
            pixels = np.asarray(Image.open(d / r["file"]).convert("L"))
            if last is None or changed(pixels, last):
                kept.append(r)
                last = pixels
        if not kept:
            continue
        n = len(chapters) + 1
        figures = []
        for r in kept:
            name = f"img/{d.name}-{Path(r['file']).stem}.png"
            images.append((name, d / r["file"]))
            figures.append(f'<figure><img src="{name}" alt="{r["time"]}"/><figcaption>{r["time"]}</figcaption></figure>')
        heading = f"Day {kept[0]['day']}"
        chapters.append((f"day{n:03d}.xhtml", heading,
                         page(heading, f'<h1>{heading}</h1><p class="date">{d.name}</p>{"".join(figures)}')))
        kept_total += len(kept)
        print(f"{d.name}: {len(kept)} of {sum(1 for _ in frames(d))} frames kept")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = "".join(f'<item id="c{i}" href="{f}" media-type="application/xhtml+xml"/>' for i, (f, _, _) in enumerate(chapters))
    manifest += "".join(f'<item id="i{i}" href="{n}" media-type="image/png"/>' for i, (n, _) in enumerate(images))
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
    opf = ('<?xml version="1.0" encoding="utf-8"?>\n<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
           '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
           f'<dc:identifier id="id">urn:eink:desk-story</dc:identifier><dc:title>{TITLE}</dc:title><dc:language>en</dc:language>'
           f'<dc:creator>Claude</dc:creator><meta property="dcterms:modified">{stamp}</meta></metadata>'
           '<manifest><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
           f'<item id="css" href="book.css" media-type="text/css"/>{manifest}</manifest><spine>{spine}</spine></package>')
    nav = page("Contents", '<nav epub:type="toc"><h1>Contents</h1><ol>'
               + "".join(f'<li><a href="{f}">{t}</a></li>' for f, t, _ in chapters) + "</ol></nav>")
    css = ("body{margin:0 4%;font-family:serif}h1{text-align:center;margin:1.5em 0 .2em}"
           ".date{text-align:center;margin:0 0 1.5em}figure{margin:0 0 1.5em;page-break-inside:avoid}"
           "img{width:100%;height:auto;display:block}figcaption{text-align:center;font-size:.8em}")

    with zipfile.ZipFile(OUT, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
                   compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/content.opf", opf, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/nav.xhtml", nav, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/book.css", css, compress_type=zipfile.ZIP_DEFLATED)
        for f, _, html in chapters:
            z.writestr(f"OEBPS/{f}", html, compress_type=zipfile.ZIP_DEFLATED)
        for name, src in images:
            z.write(src, f"OEBPS/{name}", compress_type=zipfile.ZIP_STORED)
    print(f"{kept_total} of {total} frames in {len(chapters)} chapters -> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
