"""Read the on-screen match timer -- "Time left: 2:37" / "Overtime 1:04".

Why this exists at all, when `clock.ElixirClock` already tracks elapsed time: the clock
tracks time SINCE WE STARTED WATCHING. That is enough for a live session that saw the
match begin, and useless for anything else -- a saved frame, a recorded video we joined
late, someone else's screenshot. The screen itself carries the answer, so read it.

Two things are read, and only one of them needs OCR:

  * PHASE comes from the panel's BACKGROUND, not from the word printed on it. Regular
    time draws a near-black panel, overtime a red one. That is a colour test, it needs
    no glyphs, and it survives a language we cannot read.
  * The DIGITS are matched against a bank of glyph templates in `clock_glyphs.npz`.

The panel is FOUND, not assumed at a fixed box. Captures differ -- with and without the
window title bar, different crops, different resolutions -- and a hard-coded rectangle
silently reads the wrong pixels on the next machine. The digits are the boldest pure-white
blobs in the top-right corner, sitting on a common baseline, which locates them without
any per-client calibration.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_BANK = pathlib.Path(__file__).with_name("clock_glyphs.npz")

# The timer digits are near-pure white. "Time left:" above them is CREAM -- same brightness,
# clearly higher saturation -- so a low saturation ceiling separates the two without touching
# the position. That is why the label above the number never lands in the glyph set.
_WHITE_LO, _WHITE_HI = (0, 0, 200), (179, 40, 255)

# Search box: the top-right corner. Generous, because the capture may or may not include the
# window title bar, but not the whole width -- the LEFT side of that strip carries the
# opponent's player name in white text of a similar size.
_SEARCH = (0.55, 0.0, 1.0, 0.09)

_MIN_H_FRAC, _MAX_H_FRAC = 0.18, 0.75   # glyph height as a share of the search strip
_BASELINE_TOL = 0.25                    # share of glyph height the bottoms may differ by
_HEIGHT_TOL = 0.35


class _Glyphs:
    """The template bank, loaded once. `ok` is False when the file is missing."""

    def __init__(self, path: pathlib.Path = _BANK):
        self.ok = False
        try:
            d = np.load(path)
        except Exception:                                     # noqa: BLE001
            return
        self.centres = d["centres"].astype(np.float32)        # (K, gh*gw)
        self.labels = d["labels"].astype(int)                 # digit, or -1 for the colon
        self.gw, self.gh = int(d["gw"]), int(d["gh"])
        self.ok = True

    def classify(self, glyph: np.ndarray) -> Tuple[int, float]:
        """(label, confidence) for one binarised glyph. label -1 is the colon."""
        v = (cv2.resize(glyph, (self.gw, self.gh),
                        interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32) / 255.0)
        d = np.linalg.norm(self.centres - v, axis=1)
        i = int(d.argmin())
        # Confidence is the MARGIN to the runner-up, not the raw distance: a glyph that fits
        # two templates equally well is exactly the case we must not report confidently.
        srt = np.sort(d)
        margin = float(srt[1] - srt[0]) / max(1e-6, float(srt[1]))
        return int(self.labels[i]), margin


_CACHE: Optional[_Glyphs] = None


def _bank() -> _Glyphs:
    global _CACHE
    if _CACHE is None:
        _CACHE = _Glyphs()
    return _CACHE


# A run of digits is CONTIGUOUS: the gap between two glyphs of one number is a fraction of a
# glyph. Window-manager buttons (minimise / maximise / close) are white, the same height and on
# the same baseline -- a capture that includes the title bar hands us three of them, which is
# a perfectly plausible-looking "M:SS". Splitting the baseline group into runs separates them.
#
# Generous on purpose: the colon sits in the gap between the minute and the seconds and is
# often too short to join the baseline group, which leaves a real hole there. The colon test
# below is what actually rejects the window buttons, so this only has to separate things that
# are corner-of-the-screen far apart.
_MAX_GAP = 2.5                          # gap / median glyph width

# The COLON is what makes a timer a timer. A gold or gem counter in the same corner is white
# text of the same size on the same baseline; without this test a 3-digit gem count reads as a
# clock time, which is a confident wrong number in the middle of an interchange record.
_COLON_MAX_W = 0.62                     # colon width / median digit width
_COLON_H = (0.15, 0.85)                 # colon height / digit height


def _runs(grp: List[Tuple[int, int, int, int, int]]) -> List[List[Tuple[int, int, int, int, int]]]:
    """Split a left-to-right baseline group wherever the horizontal gap gets large."""
    if not grp:
        return []
    med_w = float(np.median([c[2] for c in grp]))
    out, cur = [], [grp[0]]
    for prev, cell in zip(grp, grp[1:]):
        if cell[0] - (prev[0] + prev[2]) > _MAX_GAP * med_w:
            out.append(cur)
            cur = []
        cur.append(cell)
    out.append(cur)
    return out


def _segment(frame: np.ndarray) -> Tuple[Optional[List[np.ndarray]],
                                         Optional[Tuple[int, int, int, int]],
                                         Optional[List[Tuple[int, int, int, int, int]]]]:
    """Find the timer glyphs. Returns (glyph images left-to-right, search rect, digit boxes).

    A run is only returned when it looks like `D : D D` -- three digit-width glyphs with a
    narrow one between the first and the second. Everything else in that corner (player name,
    resource counters, window buttons) fails one of those two tests."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = _SEARCH
    px0, py0 = int(x0 * w), int(y0 * h)
    px1, py1 = int(x1 * w), int(y1 * h)
    q = frame[py0:py1, px0:px1]
    if q.size == 0:
        return None, None, None
    mask = cv2.inRange(cv2.cvtColor(q, cv2.COLOR_BGR2HSV), _WHITE_LO, _WHITE_HI)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    qh = q.shape[0]
    cand = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh < _MIN_H_FRAC * qh or bh > _MAX_H_FRAC * qh or bw < 2 or area < 20:
            continue
        cand.append((x, y, bw, bh, i))
    if not cand:
        return None, (px0, py0, px1, py1), None
    # The digits share a baseline and a height; the player name and any stray highlight do not.
    # Anchoring on the TALLEST blob and keeping its baseline neighbours is what rejects those.
    cand.sort(key=lambda c: -c[3])
    ref = cand[0]
    grp = [c for c in cand
           if abs(c[3] - ref[3]) <= _HEIGHT_TOL * ref[3]
           and abs((c[1] + c[3]) - (ref[1] + ref[3])) <= _BASELINE_TOL * ref[3]]
    grp.sort(key=lambda c: c[0])

    hits = []
    for run in _runs(grp):
        # Everything in the group already has digit HEIGHT, so width must not be used to sort
        # digits from punctuation here: a '1' is half the width of a '0' and was being thrown
        # away as a colon. The colon is found separately, below, over the unfiltered mask.
        if len(run) != 3:
            continue
        if _has_colon(stats, run, float(np.median([c[2] for c in run]))):
            hits.append(run)
    # Two candidate runs means we cannot say which one is the clock. Reporting either would
    # be a coin flip, so report none.
    if len(hits) != 1:
        return None, (px0, py0, px1, py1), None
    out = []
    for x, y, bw, bh, i in hits[0]:
        out.append((lab[y:y + bh, x:x + bw] == i).astype(np.uint8) * 255)
    return out, (px0, py0, px1, py1), hits[0]


def _has_colon(stats: np.ndarray, digits: List[Tuple[int, int, int, int, int]],
               med_w: float) -> bool:
    """Is there a colon between the minute digit and the seconds?

    Scanned over ALL components rather than the baseline group, because the colon is two
    small dots: depending on the rendering they merge into one blob or stay separate, and
    a single dot is too short to survive the digit height filter.
    """
    d0, d1 = digits[0], digits[1]
    gap0, gap1 = d0[0] + d0[2], d1[0]
    top, bot = min(d[1] for d in digits), max(d[1] + d[3] for d in digits)
    for i in range(1, stats.shape[0]):
        x, y, bw, bh, area = stats[i]
        if bw > _COLON_MAX_W * med_w or area < 3:
            continue
        cx, cy = x + bw / 2, y + bh / 2
        if gap0 <= cx <= gap1 and top <= cy <= bot:
            return True
    return False


# Overtime paints the panel red. Measured on our own frames the regular panel sits far below
# this saturation and the overtime one far above, so the test is a wide-margin one rather
# than a threshold balanced on a knife edge.
_OT_LO, _OT_HI = (0, 90, 60), (12, 255, 255)
_OT2_LO, _OT2_HI = (168, 90, 60), (179, 255, 255)
_OT_MIN_SHARE = 0.25


def _phase(frame: np.ndarray, rect: Tuple[int, int, int, int],
           digits: Optional[List[Tuple[int, int, int, int, int]]] = None) -> Optional[str]:
    """`regular` / `overtime` from the panel's background colour, or None if unclear.

    Sampled from a TIGHT band around the digits, not from the whole corner. Some arenas are
    orange-red, and measuring the corner at large made those read as overtime -- 9 of 54
    'overtime' frames then carried a time above the 2:00 overtime ceiling, which is how the
    error showed up at all.
    """
    px0, py0, px1, py1 = rect
    if digits:
        x0 = px0 + min(d[0] for d in digits)
        x1 = px0 + max(d[0] + d[2] for d in digits)
        y0 = py0 + min(d[1] for d in digits)
        y1 = py0 + max(d[1] + d[3] for d in digits)
        pad = max(3, (y1 - y0) // 3)
        q = frame[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    else:
        q = frame[py0:py1, px0:px1]
    if q.size == 0:
        return None
    hsv = cv2.cvtColor(q, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(cv2.inRange(hsv, _OT_LO, _OT_HI), cv2.inRange(hsv, _OT2_LO, _OT2_HI))
    share = float((red > 0).sum()) / max(1, red.size)
    return "overtime" if share >= _OT_MIN_SHARE else "regular"


def read_clock(frame: np.ndarray) -> Dict[str, Any]:
    """Read the timer off one BGR frame.

    Always returns the same shape. `seconds_left` is None when the panel could not be read;
    it is never guessed, and `reason` says which step failed so a caller can tell "no timer
    on screen" (not in a match) from "timer there, glyphs unreadable" (a client we do not
    handle) -- those want opposite responses.
    """
    out: Dict[str, Any] = {
        "seconds_left": None, "text": None, "phase": None, "conf": None,
        "method": "white-glyph segmentation in the top-right + nearest-template match; "
                  "phase from the panel background colour, not from the printed word",
    }
    bank = _bank()
    if not bank.ok:
        out["reason"] = f"no glyph bank at {_BANK}"
        return out
    glyphs, rect, boxes = _segment(frame)
    if rect is not None:
        out["phase"] = _phase(frame, rect, boxes)
    if not glyphs:
        out["reason"] = "no timer glyphs found (not in a match, or a layout we do not handle)"
        out["phase"] = None
        return out

    labs, confs = [], []
    for g in glyphs:
        lb, cf = bank.classify(g)
        if lb < 0:                       # the colon; carries no value, so it is simply dropped
            continue
        labs.append(lb)
        confs.append(cf)
    # Clash Royale's timer is always M:SS -- one minute digit, two second digits. Anything
    # else means the segmentation picked up something that is not the timer, and reporting a
    # number from it would be worse than reporting nothing.
    if len(labs) != 3:
        out["reason"] = f"expected 3 digits (M:SS), segmented {len(labs)}"
        return out
    m, s = labs[0], labs[1] * 10 + labs[2]
    if s > 59:
        out["reason"] = f"read {m}:{s:02d}, which is not a clock time"
        return out
    out["seconds_left"] = m * 60 + s
    out["text"] = f"{m}:{s:02d}"
    out["conf"] = round(float(min(confs)), 3)
    return out
