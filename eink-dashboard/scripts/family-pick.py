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

The local receipt may carry coordinates. What leaves the Mac is photo.json:
{"taken": "YYYY-MM-DD", "place": "<name>"}. With no GPS the place is "Family",
never the album name. Place names come from the cache at
~/Library/Application Support/coelho-dashboard/places.json (keyed by coordinates
rounded to two decimals), then OpenStreetMap Nominatim, at most one request a second.
"""

import argparse
import datetime as dt
import json
import os
import sqlite3
import time
import urllib.request

ALBUM = "Family Album"
MIN_AGE_DAYS = 335        # "at least 11 months old"
APPLE_EPOCH = dt.datetime(2001, 1, 1)
SCREENSHOT_SUBTYPE = 10
MAX_CANDIDATES = 12
NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
# Nominatim's usage policy: identify the application, and stay under one request a second.
USER_AGENT = "coelho-dashboard/family-memory (personal e-ink frame; local cache)"
PLACE_PAUSE = 1.0


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


def coordinate(lat, lon):
    """Both coordinates, or nothing. One without the other is not a place."""
    if lat is None or lon is None:
        return None
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if lat != lat or lon != lon:  # NaN
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def coord_key(lat, lon):
    return f"{round(lat, 2):.2f},{round(lon, 2):.2f}"


def default_places_cache():
    return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "coelho-dashboard", "places.json")


def clean_place(name):
    """A short place label. The album title is never a place."""
    if not isinstance(name, str):
        return "Family"
    place = " ".join(name.split())
    if not place or place == ALBUM:
        return "Family"
    return place[:80].rstrip() or "Family"


def read_places(path):
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_places(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, mode=0o700, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def place_from_address(address):
    if not isinstance(address, dict):
        return "Family"
    for key in ("city", "town", "village", "hamlet", "municipality", "county", "state"):
        if key in address:
            return clean_place(address[key])
    return "Family"


def nominatim_place(lat, lon):
    url = f"{NOMINATIM}?format=jsonv2&lat={lat:.2f}&lon={lon:.2f}&zoom=10&addressdetails=1"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as res:
        payload = json.load(res)
    address = payload.get("address") if isinstance(payload, dict) else None
    return place_from_address(address)


def resolve_place(lat, lon, cache_path, fetch=None, sleep=time.sleep, now=time.time):
    """
    The place name for a photo. No GPS (or an unusable pair) is "Family".
    A rounded coordinate already in the cache is used as-is. Otherwise Nominatim,
    then the cache is updated. The album name is never returned.
    """
    gps = coordinate(lat, lon)
    if gps is None:
        return "Family"
    key = coord_key(*gps)
    cache = read_places(cache_path)
    hit = cache.get(key)
    if isinstance(hit, str) and hit.strip():
        return clean_place(hit)
    if fetch is None:
        fetch = nominatim_place
    last = cache.get("_last")
    if isinstance(last, (int, float)) and not isinstance(last, bool):
        wait = PLACE_PAUSE - (now() - float(last))
        if wait > 0:
            sleep(wait)
    try:
        name = clean_place(fetch(round(gps[0], 2), round(gps[1], 2)))
    except Exception:
        return "Family"
    if cache_path:
        cache[key] = name
        cache["_last"] = now()
        try:
            write_places(cache_path, cache)
        except OSError:
            pass
    return name


def write_photo_json(path, taken, place):
    """photo.json is exactly the taken date and the place. Nothing else."""
    text = json.dumps({"taken": taken, "place": clean_place(place)}, ensure_ascii=False)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_candidates(db, people):
    cols = {row[1] for row in db.execute("PRAGMA table_info(ZASSET)")}
    lat_sql = "a.ZLATITUDE" if "ZLATITUDE" in cols else "NULL"
    lon_sql = "a.ZLONGITUDE" if "ZLONGITUDE" in cols else "NULL"
    base = f"""SELECT a.ZUUID, a.ZDATECREATED, x.ZTIMEZONEOFFSET, a.ZWIDTH * a.ZHEIGHT, {lat_sql}, {lon_sql}
              FROM ZASSET a LEFT JOIN ZADDITIONALASSETATTRIBUTES x ON x.ZASSET = a.Z_PK
              WHERE a.ZKIND = 0 AND a.ZTRASHEDSTATE = 0 AND a.ZHIDDEN = 0
                AND COALESCE(a.ZKINDSUBTYPE, 0) != ? AND a.ZWIDTH > a.ZHEIGHT"""
    out = []

    def add(rows, source):
        for uuid, secs, tz, px, lat, lon in rows:
            item = {"uuid": uuid, "taken": local_date(secs, tz), "source": source, "pixels": px or 0}
            gps = coordinate(lat, lon)
            if gps:
                item["latitude"], item["longitude"] = gps
            out.append(item)

    share = db.execute("SELECT Z_PK FROM ZSHARE WHERE ZTITLE = ?", (ALBUM,)).fetchone()
    if share:
        add(db.execute(base + " AND a.ZCOLLECTIONSHARE = ?", (SCREENSHOT_SUBTYPE, share[0])), "album")
    names = {n for v in people.values() for n in (v, v.strip())}
    pks = sorted({r[0] for n in names for r in db.execute(
        "SELECT Z_PK FROM ZPERSON WHERE ZFULLNAME = ? OR ZDISPLAYNAME = ?", (n, n))})
    if pks:
        marks = ",".join("?" * len(pks))
        q = base + f""" AND a.ZFAVORITE = 1 AND EXISTS (SELECT 1 FROM ZDETECTEDFACE f
                   WHERE f.ZASSETFORFACE = a.Z_PK AND f.ZHIDDEN = 0 AND f.ZPERSONFORFACE IN ({marks}))"""
        add(db.execute(q, (SCREENSHOT_SUBTYPE, *pks)), "favourite")
    return out, share is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve-place", nargs=2, metavar=("LAT", "LON"),
                    help="print the place name for one coordinate pair")
    ap.add_argument("--places-cache", help="places.json; defaults to ~/Library/Application Support/coelho-dashboard/")
    ap.add_argument("--photo-json", help="write {taken, place} for the chosen photo")
    ap.add_argument("--db", help="a copy of Photos.sqlite (never the live one)")
    ap.add_argument("--date")
    ap.add_argument("--people")
    ap.add_argument("--shown")
    args = ap.parse_args()
    cache = args.places_cache or default_places_cache()
    if args.resolve_place:
        print(resolve_place(args.resolve_place[0], args.resolve_place[1], cache))
        return
    if not args.db or not args.date or not args.people or not args.shown:
        ap.error("--db, --date, --people and --shown are required")
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
    rows = []
    for c in order:
        row = {"uuid": c["uuid"], "localIdentifier": f'{c["uuid"]}/L0/001',
               "taken": c["taken"].isoformat(), "source": c["source"]}
        if "latitude" in c and "longitude" in c:
            row["latitude"] = c["latitude"]
            row["longitude"] = c["longitude"]
        rows.append(row)
    if args.photo_json and order:
        chosen = order[0]
        write_photo_json(args.photo_json, chosen["taken"].isoformat(),
                         resolve_place(chosen.get("latitude"), chosen.get("longitude"), cache))
    print(json.dumps({
        "date": args.date, "round": rnd, "albumFound": album_found, "candidates": rows,
    }))


if __name__ == "__main__":
    main()
