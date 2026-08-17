"""Tests for the episode stream -- the pieces that no frame can prove on its own.

Everything here is a rule that has already been got wrong once, or would be silently wrong
if it broke:

  * the elixir multiplier. Overtime's FIRST minute is DOUBLE, not triple. That correction is
    recorded in config.yaml and a second copy of the rule in the exporter had it as triple.
  * the re-deploy test. A detector that loses a unit and re-finds it hands out a new track
    id, and without this one Archer Queen was reported as nine separate deploys.
  * crowns. `mine` counts ENEMY towers down. Getting the side backwards inverts the reward
    signal and looks perfectly plausible in the JSON.
  * the clock reader refusing rather than guessing on things that are not a clock.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from clashrl.observation_export import elixir_multiplier
from clashrl import episode_export as ep
from clashrl import clock_ocr


class TestElixirMultiplier(unittest.TestCase):
    def test_regulation(self):
        self.assertEqual(elixir_multiplier("regular", 180), 1)
        self.assertEqual(elixir_multiplier("regular", 61), 1)
        self.assertEqual(elixir_multiplier("regular", 60), 2)   # the last minute
        self.assertEqual(elixir_multiplier("regular", 0), 2)

    def test_overtime_first_minute_is_still_double(self):
        # The whole point of the config's triple_time_s=240 correction.
        self.assertEqual(elixir_multiplier("overtime", 120), 2)
        self.assertEqual(elixir_multiplier("overtime", 61), 2)
        self.assertEqual(elixir_multiplier("overtime", 60), 3)
        self.assertEqual(elixir_multiplier("overtime", 1), 3)

    def test_unreadable_clock_is_none_not_one(self):
        # 1 would be integrated as fact and drift for the rest of the match.
        self.assertIsNone(elixir_multiplier(None, 120))
        self.assertIsNone(elixir_multiplier("regular", None))


class TestCrowns(unittest.TestCase):
    def test_mine_counts_enemy_towers_down(self):
        rec = {"towers": {"list": [
            {"side": "enemy", "state": "destroyed"},
            {"side": "enemy", "state": "alive"},
            {"side": "mine", "state": "alive"},
            {"side": "mine", "state": "alive"},
        ]}}
        c = ep._crowns(rec)
        self.assertEqual(c["mine"], 1)
        self.assertEqual(c["enemy"], 0)

    def test_unread_towers_are_counted_separately(self):
        rec = {"towers": {"list": [
            {"side": "enemy", "state": "no_bar"},
            {"side": "mine", "state": "destroyed"},
        ]}}
        c = ep._crowns(rec)
        self.assertEqual(c["enemy"], 1)          # they hold a crown: OUR tower is down
        self.assertEqual(c["mine"], 0)
        self.assertEqual(c["towers_unread"], 1)


class TestSwarmClustering(unittest.TestCase):
    def test_a_swarm_is_one_deploy(self):
        us = [{"tile": [9.0, 10.0]}, {"tile": [9.5, 10.4]}, {"tile": [8.7, 10.9]}]
        self.assertEqual(len(ep._cluster(us)), 1)

    def test_two_separate_drops_stay_separate(self):
        us = [{"tile": [3.0, 10.0]}, {"tile": [14.0, 10.0]}]
        self.assertEqual(len(ep._cluster(us)), 2)


class _Cards:
    """Minimal stand-in: the recorder only asks for a cost and a speed."""

    def elixir(self, name):
        return {"knight": 3, "hog_rider": 4}.get(name)

    def speed_tiles(self, name):
        return 1.0


class _Recorder(ep.EpisodeRecorder):
    """Bypass __init__ -- these tests exercise the bookkeeping, not the readers."""

    def __init__(self):                                  # noqa: D107
        self._sightings = []
        self._cards = _Cards()


class TestRedeployTest(unittest.TestCase):
    def test_a_unit_that_could_have_walked_here_is_not_a_new_deploy(self):
        r = _Recorder()
        r._remember([{"card": "knight", "tile": [9.0, 10.0]}], t=0.0)
        # 2 s later, 1.5 tiles away: a knight walks that far, so this is the same knight.
        self.assertIsNotNone(r._recently_near("knight", [9.0, 11.5], t=2.0))

    def test_a_unit_that_could_not_have_walked_here_is_a_new_deploy(self):
        r = _Recorder()
        r._remember([{"card": "knight", "tile": [3.0, 25.0]}], t=0.0)
        # 1 s later, across the arena: nothing walks 15 tiles in a second.
        self.assertIsNone(r._recently_near("knight", [15.0, 8.0], t=1.0))

    def test_the_memory_expires(self):
        r = _Recorder()
        r._remember([{"card": "knight", "tile": [9.0, 10.0]}], t=0.0)
        self.assertIsNone(r._recently_near("knight", [9.2, 10.1],
                                           t=ep._REDEPLOY_WINDOW_S + 1.0))

    def test_last_seen_gap_is_reported_even_past_the_window(self):
        # The reach test gives up after the window; the CONSUMER still needs the number, so
        # it is published rather than the event being silently downgraded.
        r = _Recorder()
        r._remember([{"card": "knight", "tile": [9.0, 10.0]}], t=0.0)
        self.assertAlmostEqual(r._last_seen_gap("knight", 25.0), 25.0, places=2)
        self.assertIsNone(r._last_seen_gap("hog_rider", 5.0))


class TestClockReaderRefuses(unittest.TestCase):
    def test_blank_frame_reads_nothing(self):
        frame = np.zeros((1182, 669, 3), np.uint8)
        r = clock_ocr.read_clock(frame)
        self.assertIsNone(r["seconds_left"])
        self.assertIn("reason", r)

    def test_white_noise_does_not_produce_a_time(self):
        # Whatever this segments into, it must not come out as a clock reading.
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (1182, 669, 3), dtype=np.uint8)
        self.assertIsNone(clock_ocr.read_clock(frame)["seconds_left"])

    def test_a_plain_white_bar_is_not_a_clock(self):
        # Three white blobs on a baseline with no colon between them: the shape a window
        # title bar makes, and the shape that must not be read as M:SS.
        frame = np.zeros((1182, 669, 3), np.uint8)
        for x in (430, 470, 510):
            frame[20:46, x:x + 18] = 255
        self.assertIsNone(clock_ocr.read_clock(frame)["seconds_left"])




class TestTowerBarPlausibility(unittest.TestCase):
    """The bar detector calls the Windows taskbar a tower bar. It is right about the shape."""

    @staticmethod
    def _bar(cy, w=0.1):
        return (0.4, cy - 0.01, 0.4 + w, cy + 0.01)

    def test_the_bottom_of_the_screen_is_not_a_tower(self):
        from clashrl.troop_hp import plausible_tower_bars
        bars = [self._bar(0.15), self._bar(0.63), self._bar(0.95)]
        keep = plausible_tower_bars(bars, [0.9, 0.9, 0.9], hand_top=0.84)
        self.assertEqual(sorted(keep), [0, 1])

    def test_at_most_three_per_side(self):
        from clashrl.troop_hp import plausible_tower_bars
        # Five candidates up top, five down below; the game draws three each.
        bars = [self._bar(0.10 + 0.01 * i) for i in range(5)]
        bars += [self._bar(0.60 + 0.01 * i) for i in range(5)]
        confs = [0.9, 0.8, 0.7, 0.6, 0.5] * 2
        keep = plausible_tower_bars(bars, confs, hand_top=0.84)
        self.assertEqual(len(keep), 6)
        self.assertEqual(sum(1 for i in keep if i < 5), 3)
        self.assertEqual(sum(1 for i in keep if i >= 5), 3)

    def test_it_keeps_the_most_confident_ones(self):
        from clashrl.troop_hp import plausible_tower_bars
        bars = [self._bar(0.12), self._bar(0.14), self._bar(0.16), self._bar(0.18)]
        keep = plausible_tower_bars(bars, [0.4, 0.95, 0.9, 0.85], hand_top=0.84)
        self.assertNotIn(0, keep)          # the 0.40 one is the odd one out
        self.assertEqual(sorted(keep), [1, 2, 3])

    def test_a_low_confidence_king_bar_survives(self):
        # The enemy KING bar averages 0.74 -- lower than some of the junk -- so position and
        # count do the filtering, never a confidence floor.
        from clashrl.troop_hp import plausible_tower_bars
        bars = [self._bar(0.02), self._bar(0.15), self._bar(0.16)]
        keep = plausible_tower_bars(bars, [0.62, 0.93, 0.92], hand_top=0.84)
        self.assertIn(0, keep)




class TestModelPack(unittest.TestCase):
    """The share-a-model zip. Two things it must never do quietly."""

    def test_nothing_selected_writes_nothing(self):
        # An empty tick-box set producing an empty zip would be handed to somebody as "the
        # model" -- so it refuses instead of writing one.
        from clashrl import model_pack as mp
        self.assertEqual(mp.model_pack(_cfg(), vision=False, bars=False, policy=False), {})

    def test_the_secret_list_covers_the_token(self):
        # The API token lives under data/, the same tree a naive "zip the data folder" sweeps
        # in. Matched on NAME, so a copy in an unexpected place is still caught.
        from clashrl.model_pack import _NEVER
        self.assertIn("cr_api_token.txt", _NEVER)
        self.assertIn(".env", _NEVER)

    def test_class_names_come_from_the_weights(self):
        # The repo taxonomy is at 230 while the trained detector has 225. A README built from
        # the taxonomy would advertise five classes the model cannot predict, so the reader
        # must go to the checkpoint. Here: it returns None rather than a fallback list.
        from clashrl.model_pack import _class_names
        from pathlib import Path
        self.assertIsNone(_class_names(Path(__file__)))     # not a checkpoint at all


def _cfg():
    from clashrl.config import Config
    return Config.load(None)


class TestScrub(unittest.TestCase):
    """The checkpoint carries the training run's paths, and a local run records a home dir."""

    def test_a_home_path_is_recognised_as_naming_a_person(self):
        from clashrl.model_pack import _HOME_RE
        self.assertTrue(_HOME_RE.search(r"C:\Users\someone\Desktop"))
        self.assertTrue(_HOME_RE.search("/home/someone/x"))

    def test_a_path_that_names_nobody_is_not_reported_as_a_leak(self):
        # /kaggle/temp/... is absolute and identifies no one. Announcing it as a username
        # leak would be a scare over nothing, so the two are reported differently.
        from clashrl.model_pack import _HOME_RE, _PATH_RE
        self.assertTrue(_PATH_RE.match("/kaggle/temp/detect/data.yaml"))
        self.assertFalse(_HOME_RE.search("/kaggle/temp/detect/data.yaml"))

    def test_a_clean_checkpoint_is_packed_untouched(self):
        # No rewrite when there is nothing to rewrite: re-saving a checkpoint for no reason
        # would change its bytes and break any hash somebody published for it.
        from clashrl.model_pack import _scrub
        from pathlib import Path
        out, changed, named = _scrub(Path(__file__))     # not a checkpoint at all
        self.assertIsNone(out)
        self.assertEqual(changed, [])
        self.assertFalse(named)

if __name__ == "__main__":
    unittest.main()
