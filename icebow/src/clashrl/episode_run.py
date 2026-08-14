"""Drivers for `EpisodeRecorder`: one for a recorded video, one for the live window.

Kept apart from `episode_export` so the recorder itself has no idea where frames come from.
That is what lets the same code produce a stream from a match we play now and from a session
recorded weeks ago -- and it means the two can be compared, because they went through exactly
the same reader.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2


def _default_out(cfg, stem: str) -> Path:
    d = cfg.path("data/exports")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"


def _summary(path: Path, n: int, deploys: int, t: float,
             trailer: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {"file": str(path), "ticks": n, "deploys": deploys, "seconds": round(t, 1)}
    if trailer:
        out["units_per_tick"] = trailer.get("units_per_tick")
        out["clock_read_rate"] = trailer.get("clock_read_rate")
        out["warning"] = trailer.get("warning")
    return out


def episode_from_video(cfg, video: str, out: Optional[str] = None, every: int = 3,
                       conf: float = 0.25, limit: int = 0) -> Dict[str, Any]:
    """Turn a recorded session into an episode stream.

    Timestamps come from the session's own `meta.json` when there is one beside the video, and
    from the frame index / fps otherwise. That matters: velocity and the opponent-elixir
    integral are both per SECOND, and a wrong time base silently scales them.
    """
    from .episode_export import EpisodeRecorder, EpisodeWriter
    vp = Path(video)
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise SystemExit(f"[episode] cannot open {vp}")

    times = None
    meta_p = vp.with_name("meta.json")
    if meta_p.is_file():
        try:
            times = json.loads(meta_p.read_text(encoding="utf-8")).get("frame_times")
        except Exception:                                       # noqa: BLE001
            times = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0

    op = Path(out) if out else _default_out(cfg, f"episode_{vp.parent.name}")
    rec = EpisodeRecorder(cfg, detector_conf=conf)
    n = deploys = 0
    last_t = 0.0
    with EpisodeWriter(op, rec.header(source=f"video:{vp.parent.name}/{vp.name}")) as w:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % max(1, every) == 0:
                t = (times[i] if times and i < len(times) else i / fps)
                r = rec.tick(frame, now=t)
                w.write(r)
                n += 1
                deploys += len(r["episode"]["deploys"])
                last_t = r["episode"]["t"]
                if n % 25 == 0:
                    print(f"[episode] {n} ticks, t={last_t:.0f}s, {deploys} deploys", flush=True)
                if limit and n >= limit:
                    break
            i += 1
        w.write(rec.trailer())
    cap.release()
    s = _summary(op, n, deploys, last_t, rec.trailer())
    print(f"[episode] wrote {s['ticks']} ticks ({s['deploys']} deploys, {s['seconds']}s) -> {op}")
    print(f"[episode] {s['units_per_tick']} units/tick, clock read on "
          f"{100 * (s['clock_read_rate'] or 0):.0f}% of ticks")
    if s.get("warning"):
        print(f"[episode] WARNING: {s['warning']}")
    return s


def episode_live(cfg, out: Optional[str] = None, interval: float = 1.0, minutes: float = 6.0,
                 conf: float = 0.25) -> Dict[str, Any]:
    """Record the live game window until the match ends or the time budget runs out.

    Recording STOPS on its own when the screen leaves IN_MATCH for several consecutive ticks.
    Menus produce records full of nulls, and a file that trails off into them looks like a
    match that went quiet rather than one that ended.
    """
    from .capture import WindowCapture
    from .episode_export import EpisodeRecorder, EpisodeWriter
    from .vision import Vision

    cap = WindowCapture(cfg.get("window", "title_contains", default=None),
                        cfg.get("window", "region", default=None))
    if cap.region is None:
        raise SystemExit("[episode] no capture region -- is the game running and "
                         "window.title_contains right?")
    vision = Vision(cfg)
    op = Path(out) if out else _default_out(cfg, "episode_live")
    rec = EpisodeRecorder(cfg, detector_conf=conf)

    deadline = time.time() + minutes * 60.0
    n = deploys = 0
    off = 0                    # consecutive ticks not in a match
    last_t = 0.0
    print(f"[episode] recording to {op} (every {interval:g}s, up to {minutes:g} min)")
    with EpisodeWriter(op, rec.header(source="live")) as w:
        while time.time() < deadline:
            frame = cap.grab()
            if frame is None:
                time.sleep(interval)
                continue
            try:
                in_match = vision.detect_state(frame).name == "IN_MATCH"
            except Exception:                                   # noqa: BLE001
                in_match = False
            if not in_match:
                off += 1
                # Three in a row, not one: a single frame can land on the crown animation or
                # an emote overlay and read as not-in-match while the match is still running.
                if off >= 3 and n:
                    print("[episode] match ended (3 ticks off the match screen)")
                    break
                time.sleep(interval)
                continue
            off = 0
            r = rec.tick(frame)
            w.write(r)
            n += 1
            deploys += len(r["episode"]["deploys"])
            last_t = r["episode"]["t"]
            if n % 10 == 0:
                m = r["episode"]["match"]
                print(f"[episode] {n} ticks, clock={m.get('clock_text')}, "
                      f"{deploys} deploys", flush=True)
            time.sleep(interval)
        w.write(rec.trailer())
    s = _summary(op, n, deploys, last_t, rec.trailer())
    print(f"[episode] wrote {s['ticks']} ticks ({s['deploys']} deploys, {s['seconds']}s) -> {op}")
    print(f"[episode] {s['units_per_tick']} units/tick, clock read on "
          f"{100 * (s['clock_read_rate'] or 0):.0f}% of ticks")
    if s.get("warning"):
        print(f"[episode] WARNING: {s['warning']}")
    return s
