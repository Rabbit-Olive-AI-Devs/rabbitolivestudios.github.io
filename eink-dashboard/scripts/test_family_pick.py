"""Tests for scripts/family-pick.py: python3 -E -P -m unittest discover -s scripts -p 'test_family_pick.py'"""

import datetime as dt
import importlib.util
import json
import os
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("family_pick", os.path.join(os.path.dirname(__file__), "family-pick.py"))
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)

D = dt.date.fromisoformat


def photo(uuid, taken, source="album", pixels=12_000_000):
    return {"uuid": uuid, "taken": D(taken), "source": source, "pixels": pixels}


def pick(candidates, target, shown=()):
    return [c["uuid"] for c in fp.choose(candidates, list(shown), D(target))[1]]


class ChooseTest(unittest.TestCase):
    def test_album_exact_day_beats_favourite_exact_day(self):
        got = pick([photo("fav", "2020-10-19", "favourite"), photo("alb", "2018-10-19")], "2026-10-19")
        self.assertEqual(got[:2], ["alb", "fav"])

    def test_favourite_exact_day_beats_album_closest_day(self):
        got = pick([photo("near", "2021-10-18"), photo("fav", "2020-10-19", "favourite")], "2026-10-19")
        self.assertEqual(got, ["fav", "near"])

    def test_closest_day_wraps_around_new_year(self):
        got = pick([photo("dec", "2020-12-30"), photo("feb", "2020-02-01")], "2027-01-02")
        self.assertEqual(got[0], "dec")

    def test_at_least_eleven_months_old(self):
        got = pick([photo("new", "2026-12-30"), photo("old", "2025-12-30")], "2027-01-02")
        self.assertEqual(got, ["old"])

    def test_shown_photos_wait_for_the_next_round(self):
        cands = [photo("a", "2020-10-19"), photo("b", "2019-10-10")]
        self.assertEqual(pick(cands, "2026-10-19", [{"uuid": "a", "date": "2026-01-01", "round": 1}]), ["b"])
        rnd, order = fp.choose(cands, [{"uuid": "a", "date": "2026-01-01", "round": 1},
                                        {"uuid": "b", "date": "2026-02-01", "round": 1}], D("2026-10-19"))
        self.assertEqual(rnd, 2)
        self.assertEqual([c["uuid"] for c in order], ["a", "b"], "a new round: all available again")

    def test_fallbacks_follow_the_best_for_failed_downloads(self):
        cands = [photo(f"p{i}", f"20{10 + i}-10-{10 + i}") for i in range(5)]
        self.assertEqual(len(pick(cands, "2026-10-19")), 5)

    def test_favourite_also_in_the_album_counts_once(self):
        got = pick([photo("x", "2020-10-19"), photo("x", "2020-10-19", "favourite")], "2026-10-19")
        self.assertEqual(got, ["x"])


class PlaceTest(unittest.TestCase):
    def test_no_gps_place_is_family(self):
        def fetch(*_args):
            raise AssertionError("no lookup without coordinates")
        self.assertEqual(fp.resolve_place(None, None, "/absent/places.json", fetch=fetch), "Family")
        self.assertEqual(fp.resolve_place(None, None, "/absent/places.json", fetch=fetch), "Family")
        self.assertNotEqual(fp.resolve_place(None, 12.5, "/absent/places.json", fetch=fetch), fp.ALBUM)
        self.assertEqual(fp.resolve_place(None, 12.5, "/absent/places.json", fetch=fetch), "Family")

    def test_album_name_is_never_the_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "places.json")
            got = fp.resolve_place(1.25, 2.5, cache, fetch=lambda *_a: fp.ALBUM, sleep=lambda *_a: None, now=lambda: 0)
            self.assertEqual(got, "Family")
            self.assertNotEqual(got, fp.ALBUM)

    def test_cache_is_used_before_the_network_and_requests_wait(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "places.json")
            calls = []
            sleeps = []
            clock = {"t": 1000.0}

            def fetch(lat, lon):
                calls.append((lat, lon))
                return "Sample Shore"

            def sleep(seconds):
                sleeps.append(seconds)
                clock["t"] += seconds

            self.assertEqual(fp.resolve_place(1.234, 5.678, cache, fetch=fetch, sleep=sleep, now=lambda: clock["t"]), "Sample Shore")
            self.assertEqual(calls, [(1.23, 5.68)])
            self.assertEqual(sleeps, [])
            with open(cache, encoding="utf-8") as f:
                stored = json.load(f)
            self.assertEqual(stored["1.23,5.68"], "Sample Shore")
            self.assertEqual(fp.resolve_place(9, 9, cache, fetch=fetch, sleep=sleep, now=lambda: clock["t"]), "Sample Shore")
            self.assertGreaterEqual(sleeps[0], 1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(fp.resolve_place(1.231, 5.679, cache, fetch=fetch, sleep=sleep, now=lambda: clock["t"]), "Sample Shore")
            self.assertEqual(len(calls), 2, "a rounded coordinate already cached is not looked up again")

    def test_photo_json_is_taken_and_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "photo.json")
            fp.write_photo_json(path, "2001-02-03", "Sample Shore")
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"taken": "2001-02-03", "place": "Sample Shore"})
            fp.write_photo_json(path, "2001-02-03", None)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"taken": "2001-02-03", "place": "Family"})


if __name__ == "__main__":
    unittest.main()
