"""A MATCH -> a stream of records, with the things only a stream can know.

`observation_export.observe()` answers "what is on this frame". A simulator needs more than
that, and the extra part is not readable from any single frame:

  * WHAT WAS PLAYED. A state trace without actions trains a world model, not a player. The
    action is never drawn on the screen -- it has to be inferred from what CHANGED between
    two frames, which means it belongs here and cannot belong in `observe()`.
  * HOW FAST things move. Velocity needs two frames AND a stable identity, otherwise it is
    nearest-neighbour guessing that swaps two knights the moment they cross.
  * WHAT THE OPPONENT CAN AFFORD. Elixir is not drawn for the opponent at all. It is only
    knowable by integrating a rate and subtracting what they visibly spent.
  * WHERE IN THE MATCH WE ARE. The screen clock gives this per frame; a stream additionally
    gives a fallback when the panel is briefly unreadable, and lets the two cross-check.

Each tick is one self-contained JSON object: the full `clashai-observation/1` record plus an
`episode` block. Written as JSONL so a consumer can stream it, stop early, or lose the tail
of the file without losing the rest -- a match that crashed halfway is still usable data.

NOTHING here is smoothed or interpolated. Where an inference is a guess it says so in its own
`confidence` and `evidence` fields, because a simulator that cannot tell our guesses from our
measurements will train on both as if they were the same thing.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .observation_export import elixir_multiplier

SCHEMA = "clashai-episode/1"

# Clash Royale hands both players 5 elixir at the start of a match. Not configurable and not
# read off the screen for the opponent -- the opponent's bar is not drawn.
_START_ELIXIR = 5.0
_MAX_ELIXIR = 10.0

# Two units that appear in the same tick within this many tiles of each other are one card,
# not several: a Skeleton Army is fifteen detections and one deploy. Measured against the
# board, a swarm card lands inside roughly one and a half tiles.
_SWARM_RADIUS_TILES = 2.2

# How far back to look for "we have seen this card here before". Longer than the tracker's
# own forget_s (4.5 s by default), because the whole point is to catch the case where the
# track was already dropped and the unit came back with a new id.
_REDEPLOY_WINDOW_S = 10.0
_MIN_REACH_TILES = 2.5          # detector position noise + a unit that simply stood still
_REACH_SLACK = 1.3              # units get hasted, dash and jump the river
_FALLBACK_SPEED_TILES_S = 1.2   # for a card the database has no speed for (spells, buildings)


def _tile_dist(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b:
        return math.inf
    return math.hypot(a[0] - b[0], a[1] - b[1])


class EpisodeRecorder:
    """Feed it frames in order; it hands back one enriched record per tick.

    Stateful ON PURPOSE and only in ways a match justifies: unit identities, the opponent's
    elixir integral, and which cards have been seen played. `reset()` between matches --
    carrying any of that across a match boundary would be inventing history.
    """

    def __init__(self, cfg, tracker=None, detector_conf: float = 0.25):
        self.cfg = cfg
        self.detector_conf = detector_conf
        self._t0: Optional[float] = None
        self._prev: Optional[Dict[str, Any]] = None
        self._prev_t: Optional[float] = None
        self._seen_ids: Dict[int, float] = {}          # unit id -> first time seen
        self._enemy_elixir = _START_ELIXIR
        self._enemy_spent = 0.0
        self._enemy_unknown_plays = 0                  # deploys we could not price
        self._sightings: List[Tuple[str, float, float, float]] = []   # card, x, y, t
        self._suppressed = 0                           # returns rejected as re-deploys
        self._quality: Dict[str, Any] = {"ticks": 0, "unit_detections": 0, "ticks_with_no_unit": 0,
                "clock_reads": 0, "hand_slots_read": 0, "hand_slots_total": 0,
                "deploys_suppressed": 0, "deploys": {}, "by_confidence": {}, "last_t": 0.0}
        self._tick = 0
        if tracker is None:
            from .replay_mine import TeamTracker
            tracker = TeamTracker(
                track_radius=float(cfg.get("observation", "team_track_radius", default=0.12)),
                forget_s=float(cfg.get("observation", "team_forget_s", default=4.5)))
        self.tracker = tracker
        from .cards import CardDB
        self._cards = CardDB(cfg)

    # -- the one call ------------------------------------------------------------------
    def tick(self, frame, now: Optional[float] = None,
             own_click: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """One frame -> one enriched record. `own_click` is the normalised xy WE just tapped,
        when the caller knows it (the live bot does); it turns our own deploy from an
        inference into a measurement."""
        from . import observation_export
        now = time.monotonic() if now is None else now
        if self._t0 is None:
            self._t0 = now
        rec = observation_export.observe(self.cfg, frame, self.detector_conf,
                                         prev=self._prev, tracker=self.tracker)
        t = round(now - self._t0, 3)
        dt = None if self._prev_t is None else max(1e-6, now - self._prev_t)

        ep: Dict[str, Any] = {"schema": SCHEMA, "tick": self._tick, "t": t, "dt": _r(dt)}
        ep["match"] = self._match_block(frame)
        ep["crowns"] = _crowns(rec)
        ep["velocity"] = self._velocity(rec, dt)
        ep["deploys"] = self._deploys(rec, t, own_click)
        ep["deploys_suppressed"] = {
            "count": self._suppressed,
            "why": "new track ids that a same-card unit could have walked to from a sighting "
                   "in the last %.0f s -- the detector losing and re-finding a unit, not a "
                   "card being played" % _REDEPLOY_WINDOW_S}
        ep["opponent_elixir"] = self._opponent_elixir(ep["deploys"], ep["match"], dt)
        self._apply_multiplier(rec, ep["match"])
        rec["episode"] = ep

        self._tally(rec, ep)
        self._prev, self._prev_t = rec, now
        self._tick += 1
        return rec

    # -- how good was this recording, measured on itself -------------------------------
    def _tally(self, rec: Dict[str, Any], ep: Dict[str, Any]) -> None:
        q = self._quality
        q["ticks"] += 1
        q["unit_detections"] += len(rec.get("units", {}).get("list", []))
        q["ticks_with_no_unit"] += 0 if rec.get("units", {}).get("list") else 1
        q["clock_reads"] += 1 if ep["match"]["seconds_left"] is not None else 0
        q["hand_slots_read"] += sum(1 for s in rec.get("hand", {}).get("slots", [])
                                    if s.get("state") == "read")
        q["hand_slots_total"] += len(rec.get("hand", {}).get("slots", []))
        q["deploys_suppressed"] += ep["deploys_suppressed"]["count"]
        for d in ep["deploys"]:
            q["deploys"][d["side"]] = q["deploys"].get(d["side"], 0) + 1
            q["by_confidence"][d["confidence"]] = q["by_confidence"].get(d["confidence"], 0) + 1
        q["last_t"] = ep["t"]

    def trailer(self) -> Dict[str, Any]:
        """Last line of the file: what this recording is actually worth.

        A stream can be perfectly well-formed and still be nearly useless -- a detector that
        saw a unit on one tick in three cannot support an action trace, and a consumer must
        be able to see that WITHOUT re-deriving it from 300 records. Every number here is
        counted from the ticks that were written, not estimated.
        """
        q = dict(self._quality)
        n = max(1, q["ticks"])
        q["kind"] = "trailer"
        q["schema"] = SCHEMA
        q["units_per_tick"] = round(q["unit_detections"] / n, 3)
        q["share_ticks_with_no_unit"] = round(q["ticks_with_no_unit"] / n, 3)
        q["clock_read_rate"] = round(q["clock_reads"] / n, 3)
        q["hand_read_rate"] = (round(q["hand_slots_read"] / q["hand_slots_total"], 3)
                               if q["hand_slots_total"] else None)
        q["warning"] = None
        # The threshold is a judgement, so it is stated rather than applied silently.
        if q["units_per_tick"] < 1.0:
            q["warning"] = ("fewer than one unit detected per tick: the deploy events in this "
                            "file are not reliable, because the detector loses units for "
                            "longer than any re-deploy test can bridge. Usable as a state "
                            "trace; NOT usable as an action trace.")
        return q

    def reset(self) -> None:
        """Between matches. Everything below is match-scoped and means nothing across one."""
        self._t0 = self._prev = self._prev_t = None
        self._seen_ids.clear()
        self._enemy_elixir = _START_ELIXIR
        self._enemy_spent = 0.0
        self._enemy_unknown_plays = 0
        self._sightings.clear()
        self._suppressed = 0
        self._quality = {"ticks": 0, "unit_detections": 0, "ticks_with_no_unit": 0,
                "clock_reads": 0, "hand_slots_read": 0, "hand_slots_total": 0,
                "deploys_suppressed": 0, "deploys": {}, "by_confidence": {}, "last_t": 0.0}
        self._tick = 0
        self.tracker.reset()

    def header(self, source: str = "live") -> Dict[str, Any]:
        """First line of the JSONL: what produced this file, so it can be read years later."""
        from .observation_export import SCHEMA as OBS_SCHEMA, TILES_X, TILES_Y
        return {
            "schema": SCHEMA, "record_schema": OBS_SCHEMA, "kind": "header",
            "source": source, "arena_tiles": [TILES_X, TILES_Y],
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "one JSON object per line; the first is this header, the rest are ticks. "
                    "Each tick is a full observation record plus an `episode` block.",
        }

    # -- where in the match are we ------------------------------------------------------
    def _match_block(self, frame) -> Dict[str, Any]:
        """Seconds left and phase, read off the SCREEN, with the tick clock as the fallback.

        The screen is preferred even though it is only readable on some frames, because it is
        absolute: it stays right when we joined the match late, when the panel dropped frames,
        or when the recording was trimmed. The tick clock only knows how long WE have watched.
        """
        from .clock_ocr import read_clock
        r = read_clock(frame)
        out: Dict[str, Any] = {
            "seconds_left": r["seconds_left"], "phase": r["phase"],
            "clock_text": r["text"], "clock_conf": r["conf"],
            "watched_s": None if self._t0 is None else round(
                (self._prev_t or self._t0) - self._t0, 1),
            "source": "screen clock" if r["seconds_left"] is not None else "unread this tick",
        }
        if r["seconds_left"] is None and r.get("reason"):
            out["clock_reason"] = r["reason"]
        return out

    def _apply_multiplier(self, rec: Dict[str, Any], match: Dict[str, Any]) -> None:
        """Fill in `elixir.multiplier`, which a single frame legitimately cannot know.

        The thresholds come from `elixir.double_time_s` / `triple_time_s` in the config so
        this and the live bot's ElixirClock cannot drift apart -- there is exactly one place
        that says when double elixir starts.
        """
        el = rec.get("elixir")
        if not isinstance(el, dict):
            return
        # observe() has already filled this in from the same clock read. Recomputing it here
        # through a SECOND copy of the phase rule is exactly how the two would drift apart, so
        # the shared helper is called instead and this only adds the stream's own note.
        left, phase = match.get("seconds_left"), match.get("phase")
        el["multiplier"] = elixir_multiplier(phase, left)
        el["multiplier_note"] = (
            "clock unreadable this tick; NOT carried over from the last one, because a stale "
            "multiplier is indistinguishable from a measured one downstream"
            if el["multiplier"] is None else
            f"from the on-screen clock ({phase}, {left}s left); overtime's first minute is "
            f"still double elixir")

    # -- velocity, from identities rather than from proximity ----------------------------
    def _velocity(self, rec: Dict[str, Any], dt: Optional[float]) -> Dict[str, Any]:
        """Per-unit tiles/second, matched by TRACK ID.

        `observe()` already emits a `motion` block, but that one matches by nearest same-class
        neighbour and says so: it swaps two knights the moment their paths cross. With stable
        ids there is nothing to match -- the same id IS the same unit -- so this is a
        measurement where the other is an approximation. Both are kept: a consumer without
        ids can still use the naive one.
        """
        out: Dict[str, Any] = {"available": False,
                               "method": "same track id in two consecutive ticks, in tiles/s"}
        if dt is None or not self._prev:
            out["reason"] = "needs two ticks"
            return out
        old = {u["id"]: u for u in self._prev.get("units", {}).get("list", [])
               if u.get("id") is not None and u.get("tile")}
        rows = []
        for u in rec.get("units", {}).get("list", []):
            o = old.get(u.get("id"))
            if o is None or not u.get("tile"):
                continue
            vx = (u["tile"][0] - o["tile"][0]) / dt
            vy = (u["tile"][1] - o["tile"][1]) / dt
            rows.append({"id": u["id"], "cls": u["cls"], "team": u.get("team"),
                         "tile": u["tile"], "v_tiles_s": [_r(vx, 3), _r(vy, 3)],
                         "speed_tiles_s": _r(math.hypot(vx, vy), 3)})
        out["available"] = True
        out["units"] = rows
        return out

    # -- what was played ----------------------------------------------------------------
    def _deploys(self, rec: Dict[str, Any], t: float,
                 own_click: Optional[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Cards played since the previous tick, for BOTH sides.

        Two independent signals, and they are reported as different things because they are
        not equally trustworthy:

          OURS   -- a hand slot changed card. That is the game telling us we spent something;
                    it is near-certain. Only the PLACEMENT has to be inferred, and when the
                    caller passes `own_click` even that is measured.
          THEIRS -- a unit id we have never seen appeared on their half. That is an inference
                    and it over-reports by construction: a Witch's skeletons, a Goblin
                    Barrel's goblins and a Tombstone's skeletons are all new units that cost
                    no elixir. `spawn_suspect` marks the ones that look like that, rather than
                    dropping them, because dropping them would silently hide a real deploy.
        """
        events: List[Dict[str, Any]] = []
        units = rec.get("units", {}).get("list", [])
        self._suppressed = 0

        # -- ours: the hand changed
        if self._prev:
            before = _slots(self._prev)
            after = _slots(rec)
            for i, (b, a) in enumerate(zip(before, after)):
                # BOTH readings must be a named card. A slot going card -> None is the
                # template match failing, not a play, and treating it as one would invent a
                # deploy every time the card art is briefly covered by a placement drag.
                if b and a and b != a:
                    ev = {"side": "mine", "card": b, "slot": i, "t": t,
                          "cost": self._cards.elixir(b),
                          "evidence": f"hand slot {i} changed {b} -> {a}",
                          "confidence": "high"}
                    if own_click is not None:
                        ev["xy"] = [_r(own_click[0]), _r(own_click[1])]
                        ev["placement"] = "the tap we sent"
                    else:
                        near = _newest_own(units, self._seen_ids)
                        ev["tile"] = near["tile"] if near else None
                        ev["placement"] = ("the unit that appeared with it" if near else
                                           "unknown -- no new own unit this tick (a spell, or "
                                           "the detector missed it)")
                    events.append(ev)
        played_by_hand = {e["card"] for e in events}

        # -- both sides: unit ids we have not seen before
        for side in ("enemy", "mine"):
            fresh = [u for u in units
                     if u.get("id") is not None and u["id"] not in self._seen_ids
                     and u.get("team") == side and u.get("tile")]
            for group in _cluster(fresh):
                head = group[0]
                card = head.get("card")
                # A new ID IS NOT A NEW UNIT. The detector loses units constantly -- behind a
                # tower, inside a push -- and once a track ages past `forget_s` the same unit
                # comes back with a fresh id. Without this test one Archer Queen was reported
                # as nine separate deploys in a 100 s match. The gate is the unit's OWN top
                # speed over the time it was missing: anything it could have walked to is a
                # return, anything further away is genuinely new.
                prior = self._recently_near(card, head["tile"], t)
                if prior is not None:
                    self._suppressed += 1
                    continue
                if side == "mine" and card in played_by_hand:
                    continue            # already reported, with better evidence, above
                # How long since this card was on the board ANYWHERE. The reach test above
                # already rejected the returns it could prove; this number is what is left
                # over, and it is the single most useful thing for deciding whether to
                # believe the event. On sparse footage the detector loses a unit for 20 s at
                # a stretch, which no reach test can distinguish from a real re-deploy --
                # so the number is published instead of a verdict being invented.
                gap = self._last_seen_gap(card, t)
                ev = {"side": side, "card": card, "t": t,
                      "count": len(group),
                      "ids": [u["id"] for u in group],
                      "tile": head["tile"],
                      "cost": self._cards.elixir(card or ""),
                      "last_seen_same_card_s": gap,
                      "evidence": f"{len(group)} track id(s) never seen before on the "
                                  f"{'enemy' if side == 'enemy' else 'own'} side, and no "
                                  f"{card} was within reach of here recently",
                      "confidence": "inferred" if gap is None else
                                    ("inferred" if gap > _REDEPLOY_WINDOW_S else "weak")}
                # A swarm of one cheap class appearing next to a spawner is far more likely to
                # be that building's output than a card. Saying so is the difference between a
                # usable action trace and one the consumer has to clean up blind.
                ev["spawn_suspect"] = _spawn_suspect(card, units, head["tile"])
                events.append(ev)

        self._remember(units, t)
        for u in units:
            if u.get("id") is not None:
                self._seen_ids.setdefault(u["id"], t)
        return events

    # -- has this card been on the board near here lately? --------------------------------
    def _remember(self, units: List[Dict[str, Any]], t: float) -> None:
        for u in units:
            if u.get("card") and u.get("tile"):
                self._sightings.append((u["card"], u["tile"][0], u["tile"][1], t))
        cut = t - _REDEPLOY_WINDOW_S
        if len(self._sightings) > 4000:
            self._sightings = [s for s in self._sightings if s[3] >= cut]

    def _last_seen_gap(self, card: Optional[str], t: float) -> Optional[float]:
        """Seconds since this card was last detected anywhere, or None if never."""
        if not card:
            return None
        for c, _x, _y, ts in reversed(self._sightings):
            if c == card:
                return round(t - ts, 2)
        return None

    def _recently_near(self, card: Optional[str], tile: List[float],
                       t: float) -> Optional[Tuple[float, float]]:
        """The most recent sighting of this card that this unit could have WALKED from."""
        if not card:
            return None
        speed = self._cards.speed_tiles(card) or _FALLBACK_SPEED_TILES_S
        best = None
        for c, x, y, ts in reversed(self._sightings):
            gap = t - ts
            if gap > _REDEPLOY_WINDOW_S:
                break
            if c != card or gap <= 0:
                continue
            # A floor, because the detector's own position noise is about a tile and a unit
            # standing still for one tick must not count as having teleported.
            reach = max(_MIN_REACH_TILES, speed * gap * _REACH_SLACK)
            if math.hypot(x - tile[0], y - tile[1]) <= reach:
                best = (round(gap, 2), round(reach, 2))
                break
        return best

    # -- what the opponent can afford ------------------------------------------------------
    def _opponent_elixir(self, deploys: List[Dict[str, Any]], match: Dict[str, Any],
                         dt: Optional[float]) -> Dict[str, Any]:
        """Integrate the regeneration rate, subtract what they were seen to spend.

        This is an ESTIMATE and the error only ever goes one way: every deploy we fail to see
        makes us think they have MORE than they do. It is reported with what it is built from
        so a consumer can decide whether to trust it, and the bounds are honest -- the upper
        one is the full bar, because we can never rule out that they simply did not spend.
        """
        base = float(self.cfg.get("elixir", "base_rate_per_s", default=1.0 / 2.8))
        # Same helper as everywhere else. When the clock is unreadable this falls back to 1x,
        # which UNDER-fills the bar rather than over-filling it -- the safer direction for a
        # number a policy would use to decide whether the opponent can answer.
        mult = elixir_multiplier(match.get("phase"), match.get("seconds_left")) or 1
        if dt:
            self._enemy_elixir = min(_MAX_ELIXIR, self._enemy_elixir + base * mult * dt)
        priced = unpriced = 0
        for ev in deploys:
            if ev["side"] != "enemy" or ev.get("spawn_suspect"):
                continue
            c = ev.get("cost")
            if c is None:
                unpriced += 1
                self._enemy_unknown_plays += 1
            else:
                priced += 1
                self._enemy_spent += float(c)
                self._enemy_elixir = max(0.0, self._enemy_elixir - float(c))
        return {
            "estimate": _r(self._enemy_elixir, 2),
            "lower": _r(self._enemy_elixir, 2),
            "upper": _MAX_ELIXIR,
            "spent_total": _r(self._enemy_spent, 2),
            "unpriced_plays": self._enemy_unknown_plays,
            "rate_per_s": _r(base * mult, 4),
            "reliability": "estimate",
            "method": f"starts at {_START_ELIXIR:g}, regenerates at "
                      f"{base:.4g}/s x{mult}, minus the cost of every enemy deploy we saw",
            "caveat": "a deploy we miss (a spell, an occluded drop, an unknown card) is never "
                      "subtracted, so the estimate is an UPPER-leaning one. `upper` is the "
                      "full bar because 'they spent nothing' can never be ruled out.",
            "priced_this_tick": priced, "unpriced_this_tick": unpriced,
        }


# -- helpers ----------------------------------------------------------------------------

def _r(v, n: int = 4):
    return None if v is None else round(float(v), n)


def _slots(rec: Dict[str, Any]) -> List[Optional[str]]:
    return [s.get("card") for s in rec.get("hand", {}).get("slots", [])]


def _crowns(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Crowns == towers destroyed. No separate reader needed, and no scoreboard to wait for."""
    tl = rec.get("towers", {}).get("list", [])
    mine = sum(1 for t in tl if t.get("side") == "mine" and t.get("state") == "destroyed")
    enemy = sum(1 for t in tl if t.get("side") == "enemy" and t.get("state") == "destroyed")
    unread = sum(1 for t in tl if t.get("state") in (None, "no_bar", "no_match"))
    return {"mine": enemy, "enemy": mine,
            "method": "count of towers in this record whose state is `destroyed`; in Clash "
                      "Royale crowns and felled towers are the same number",
            "towers_unread": unread,
            "note": "`mine` is crowns WE have taken, i.e. enemy towers down"}


def _cluster(units: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group units that appeared together and close together -- one card, not fifteen."""
    out: List[List[Dict[str, Any]]] = []
    for u in units:
        for g in out:
            if _tile_dist(g[0].get("tile"), u.get("tile")) <= _SWARM_RADIUS_TILES:
                g.append(u)
                break
        else:
            out.append([u])
    return out


def _newest_own(units: List[Dict[str, Any]], seen: Dict[int, float]) -> Optional[Dict[str, Any]]:
    fresh = [u for u in units if u.get("team") == "mine"
             and u.get("id") is not None and u["id"] not in seen and u.get("tile")]
    return fresh[0] if fresh else None


# Buildings that keep producing units for free. A new unit of the right kind next to one of
# these is its output, not a card someone paid for.
_SPAWNERS = ("tombstone", "goblin_hut", "barbarian_hut", "furnace", "witch", "night_witch",
             "goblin_cage", "goblin_drill", "skeleton_king", "graveyard")
_SPAWN_RADIUS_TILES = 5.0


def _spawn_suspect(card: Optional[str], units: List[Dict[str, Any]],
                   tile: Optional[List[float]]) -> Optional[str]:
    """Name the nearby spawner this unit probably came out of, or None."""
    if not card or not tile:
        return None
    for u in units:
        base = (u.get("card") or "")
        if base in _SPAWNERS and _tile_dist(u.get("tile"), tile) <= _SPAWN_RADIUS_TILES:
            return base
    return None


# -- writing a file ----------------------------------------------------------------------

class EpisodeWriter:
    """JSONL sink. Flushes every tick, so a run that is killed still leaves valid lines."""

    def __init__(self, path: Path, header: Dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.n = 0
        self._write(header)

    def _write(self, obj: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._fh.flush()

    def write(self, rec: Dict[str, Any]) -> None:
        self._write(rec)
        self.n += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:                                       # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
