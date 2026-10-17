#!/usr/bin/env python3
"""
Family Memory (DECISIONS #92): which family photo to show on a date.

Reads a READ-ONLY COPY of Photos.sqlite (never the live database) and prints, as JSON, the
current round and the candidates in order of preference. The mailbox's art-brief job tries
them in turn (the first that downloads wins) and records the winner in the shown log.

Order (owner, 2026-10-17):
  1. a photo from the album taken on the exact date (month and day) in an earlier year;
  2. else a Favourites family photo (a face of someone in people.json) on the exact date;
  3. else the album photo from the closest calendar day, either way, wrapping around New Year.
Every candidate is wide, a still photo, not hidden, trashed or a screenshot, and at least
11 months old. A photo shown in the current round is skipped; once every album photo has
been shown, a new round starts and all of them are available again.

Usage:
  cp "<library>/database/Photos.sqlite"* /tmp/photosdb/
  python3 -E -P scripts/family-pick.py --db /tmp/photosdb/Photos.sqlite --date 2026-10-19 \
    --people photos/portraits/people.json --shown ~/.config/eink/family-shown.json

Prints only UUIDs and dates, never names or places.
"""

import argparse
import datetime as dt
import json
import os
import sqlite3

ALBUM = "Coelhos de Oliveira"
MIN_AGE_DAYS = 335        # "at least 11 months old"
APPLE_EPOCH = dt.datetime(2001, 1, 1)
SCREENSHOT_SUBTYPE = 10
MAX_CANDIDATES = 12


def day_of_year(date):
    """Position in a leap year, so 29 Feb has its own day and distances wrap at 366."""
    return dt.date(2000, date.month, date.day).timetuple().tm_yday


def day_distance(a, b):
    d = abs(day_of_year(a) - day_of_year(b))
    return min(d, 366 - d)


def current_round(shown, album_ids):
    """(round number, ids shown in it). A new round starts once every album photo was shown."""
    rnd = max((e["round"] for e in shown), default=1)
    seen = {e["uuid"] for e in shown if e["round"] == rnd}
    if album_ids and album_ids <= seen:
        return rnd + 1, set()
    return rnd, seen


def choose(candidates, shown, target):
    """
    candidates: [{uuid, taken: date, source: "album"|"favourite", pixels}]
    shown:      [{uuid, date, round}] from the shown log
    target:     date to fill
    Returns (round, ordered candidates): the best first, then the fallbacks.
    """
    album_ids = {c["uuid"] for c in candidates if c["source"] == "album"}
    rnd, seen = current_round(shown, album_ids)
    ok = [c for c in candidates
          if (target - c["taken"]).days >= MIN_AGE_DAYS and c["uuid"] not in seen]
    exact = lambda c: (c["taken"].month, c["taken"].day) == (target.month, target.day)
    bigger = lambda c: -c["pixels"]
    album = [c for c in ok if c["source"] == "album"]
    favs = [c for c in ok if c["source"] == "favourite" and c["uuid"] not in album_ids]
    order = (sorted(filter(exact, album), key=bigger)
             + sorted(filter(exact, favs), key=bigger)
             + sorted((c for c in album if not exact(c)), key=lambda c: (day_distance(c["taken"], target), bigger(c))))
    return rnd, order[:MAX_CANDIDATES]


def local_date(seconds, tz_offset):
    return (APPLE_EPOCH + dt.timedelta(seconds=seconds + (tz_offset or 0))).date()


def read_candidates(db, people):
    base = """SELECT a.ZUUID, a.ZDATECREATED, x.ZTIMEZONEOFFSET, a.ZWIDTH * a.ZHEIGHT
              FROM ZASSET a LEFT JOIN ZADDITIONALASSETATTRIBUTES x ON x.ZASSET = a.Z_PK
              WHERE a.ZKIND = 0 AND a.ZTRASHEDSTATE = 0 AND a.ZHIDDEN = 0
                AND COALESCE(a.ZKINDSUBTYPE, 0) != ? AND a.ZWIDTH > a.ZHEIGHT"""
    out = []
    share = db.execute("SELECT Z_PK FROM ZSHARE WHERE ZTITLE = ?", (ALBUM,)).fetchone()
    if share:
        for uuid, secs, tz, px in db.execute(base + " AND a.ZCOLLECTIONSHARE = ?", (SCREENSHOT_SUBTYPE, share[0])):
            out.append({"uuid": uuid, "taken": local_date(secs, tz), "source": "album", "pixels": px or 0})
    names = {n for v in people.values() for n in (v, v.strip())}
    pks = sorted({r[0] for n in names for r in db.execute(
        "SELECT Z_PK FROM ZPERSON WHERE ZFULLNAME = ? OR ZDISPLAYNAME = ?", (n, n))})
    if pks:
        marks = ",".join("?" * len(pks))
        q = base + f""" AND a.ZFAVORITE = 1 AND EXISTS (SELECT 1 FROM ZDETECTEDFACE f
                   WHERE f.ZASSETFORFACE = a.Z_PK AND f.ZHIDDEN = 0 AND f.ZPERSONFORFACE IN ({marks}))"""
        for uuid, secs, tz, px in db.execute(q, (SCREENSHOT_SUBTYPE, *pks)):
            out.append({"uuid": uuid, "taken": local_date(secs, tz), "source": "favourite", "pixels": px or 0})
    return out, share is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="a copy of Photos.sqlite (never the live one)")
    ap.add_argument("--date", required=True)
    ap.add_argument("--people", required=True)
    ap.add_argument("--shown", required=True)
    args = ap.parse_args()
    target = dt.date.fromisoformat(args.date)
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    with open(args.people, encoding="utf-8") as f:
        people = json.load(f)
    shown = []
    if os.path.exists(args.shown):
        with open(args.shown, encoding="utf-8") as f:
            shown = json.load(f)
    candidates, album_found = read_candidates(db, people)
    rnd, order = choose(candidates, shown, target)
    print(json.dumps({
        "date": args.date, "round": rnd, "albumFound": album_found,
        "candidates": [{"uuid": c["uuid"], "localIdentifier": f'{c["uuid"]}/L0/001',
                        "taken": c["taken"].isoformat(), "source": c["source"]} for c in order],
    }))


if __name__ == "__main__":
    main()
