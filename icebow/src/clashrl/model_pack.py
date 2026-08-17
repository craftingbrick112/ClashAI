"""Pack the trained models into one zip that can be handed to somebody else.

Why this exists: `icebow/.gitignore` excludes `data/` and `*.pt`, so no model has ever been
committed -- checked across the whole history of every branch, not one. That exclusion is
right (git keeps every version forever, and a 39 MB binary that changes completely on each
training run would sit in the history of everyone who clones the repo, including people who
only want the code). But the consequence was never followed through: if the weights cannot
live in the repo they have to live somewhere else, and nobody ever put them anywhere. People
asking for the model were sent to a path that does not exist.

So this writes the file you attach to a GitHub Release.

Two things it does that a manual zip does not:

  * IT NAMES WHAT IT PACKED, with the file's own modification date and size, in a manifest
    AND in a readable README. The exports folder already holds four zips from three separate
    days and none of their names say which model is inside -- one of them was nearly shared
    as "the model" while being three days older than the current one.
  * IT REFUSES TO PACK SECRETS. The API token lives in `data/`, which is exactly the tree a
    naive "zip up the data folder" would sweep in.
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Files that must never leave this machine, matched on NAME anywhere in the tree. The list is
# deliberately about the secret rather than about the folder: a token that has been copied
# somewhere unexpected is exactly the case a path-based rule misses.
_NEVER = ("cr_api_token.txt", "cr_token.txt", ".env", "credentials.json")


def _stamp(p: Path) -> Dict[str, Any]:
    st = p.stat()
    return {"file": p.name,
            "bytes": st.st_size,
            "mb": round(st.st_size / 1048576, 1),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))}


def _class_names(p: Path) -> Optional[List[str]]:
    """Class names straight out of the weights, so the README describes THIS file.

    Read from the checkpoint rather than from `classes.txt`, because those two can disagree --
    the taxonomy has moved to 230 entries while the trained model still has 225, and a README
    quoting the taxonomy would advertise five classes the model cannot predict.
    """
    try:
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
        names = (ck.get("model") and getattr(ck["model"], "names", None)) or ck.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        return list(names) if names else None
    except Exception:                                           # noqa: BLE001
        return None


def _vision(cfg) -> Optional[Tuple[Path, List[Tuple[Path, str]], Dict[str, Any]]]:
    from .detect import _resolve_weights
    try:
        w, runs = _resolve_weights(cfg, None)
    except Exception:                                           # noqa: BLE001
        return None
    if not w or not Path(w).is_file():
        return None
    w = Path(w)
    files = [(w, "vision/best.pt")]
    info = _stamp(w)
    card = w.parent.parent / "model_card.json"
    if card.is_file():
        files.append((card, "vision/model_card.json"))
        try:
            info["card"] = json.loads(card.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            pass
    names = _class_names(w)
    if names:
        info["classes"] = len(names)
        info["class_names"] = names
    return w, files, info


def _bars(cfg) -> Optional[Tuple[Path, List[Tuple[Path, str]], Dict[str, Any]]]:
    w = Path(cfg.path("runs/bars/v1/weights/best.pt"))
    if not w.is_file():
        return None
    info = _stamp(w)
    names = _class_names(w)
    if names:
        info["classes"] = len(names)
        info["class_names"] = names
    return w, [(w, "bars/best.pt")], info


def _policy(cfg) -> Optional[Tuple[Path, List[Tuple[Path, str]], Dict[str, Any]]]:
    """The playing policy. Prefers the RL checkpoint exactly as `play` does, so what gets
    shared is what actually plays -- shipping the other one would be a quiet substitution."""
    rl = Path(cfg.path(cfg.get("train", "rl_checkpoint", default="data/policy_rl.pt")))
    bc = Path(cfg.path(cfg.get("train", "checkpoint", default="data/policy.pt")))
    w = rl if rl.is_file() else bc
    if not w.is_file():
        return None
    info = _stamp(w)
    info["which"] = "RL-finetuned (what play.py loads first)" if w == rl else "imitation"
    return w, [(w, "policy/" + w.name)], info


_PARTS = {"vision": _vision, "bars": _bars, "policy": _policy}

_WHAT = {
    "vision": "the board detector -- which unit is where. THE vision AI.",
    "bars": "the health-bar detector, a separate 2-class model. Optional: without it every "
            "unit's `hp` is null and nothing else changes.",
    "policy": "the playing policy -- which card to play where. Nothing to do with vision.",
}


def _readme(picked: Dict[str, Dict[str, Any]]) -> str:
    L: List[str] = ["# ClashAI models", ""]
    L.append(f"Packed {time.strftime('%Y-%m-%d %H:%M')} from the machine that trained them.")
    L.append("")
    L.append("| part | file | size | trained/modified |")
    L.append("|---|---|---|---|")
    for k, i in picked.items():
        L.append(f"| `{k}` | `{k}/{i['file']}` | {i['mb']} MB | {i['modified']} |")
    L.append("")
    for k, i in picked.items():
        L.append(f"## {k}")
        L.append("")
        L.append(_WHAT[k])
        L.append("")
        if i.get("classes"):
            L.append(f"**{i['classes']} classes.** Read from the checkpoint itself, not from "
                     f"`detect_classes.yaml` -- the taxonomy in the repo has moved ahead of the "
                     f"trained model before, and a class list that does not match the weights "
                     f"is worse than none.")
            L.append("")
        card = i.get("card") or {}
        if card:
            rows = [("model", card.get("model")), ("imgsz", card.get("imgsz")),
                    ("epochs run", card.get("epochs_run")), ("best epoch", card.get("epochs")),
                    ("mAP50", card.get("mAP50")), ("mAP50-95", card.get("mAP50_95")),
                    ("precision", card.get("precision")), ("recall", card.get("recall")),
                    ("train frames", card.get("trained_on_frames")),
                    ("train boxes", card.get("trained_on_boxes")),
                    ("val frames", card.get("val_frames")),
                    ("val boxes", card.get("val_boxes"))]
            L.append("| | |")
            L.append("|---|---|")
            for a, b in rows:
                if b is not None:
                    L.append(f"| {a} | {b} |")
            L.append("")
            L.append("> Read those numbers as **on frames like ours**. The validation split is "
                     "entirely our own client while most training frames come from an imported "
                     "public set, so the score measures what we care about rather than "
                     "flattering itself on the bigger source -- but it describes accuracy on "
                     "OUR capture setup. Measure on your own frames before relying on it.")
            L.append("")
    L.append("## Using them")
    L.append("")
    L.append("```python")
    L.append("from ultralytics import YOLO")
    L.append("m = YOLO('vision/best.pt')")
    L.append("r = m.predict('frame.jpg', conf=0.25)[0]")
    L.append("for b in r.boxes:")
    L.append("    print(r.names[int(b.cls)], b.xyxy[0].tolist(), float(b.conf))")
    L.append("```")
    L.append("")
    L.append("Boxes come out in PIXELS of the frame you passed in. To turn them into arena "
             "tiles you need the arena calibration, which is per-client -- see "
             "`icebow/OBSERVATION_FORMAT.md` in the repo. **Flying units are drawn above the "
             "tile they occupy**, so a box centre is not a ground position.")
    L.append("")
    L.append("## Licence and fair use")
    L.append("")
    L.append("Trained on screenshots of Clash Royale, a Supercell game. Supercell has not "
             "endorsed this and is not involved with it. Automating the game client can get "
             "an account banned -- this is a research project, not something to point at a "
             "competitive ladder.")
    return "\n".join(L) + "\n"


def model_pack(cfg, out: Optional[str] = None, vision: bool = True, bars: bool = False,
               policy: bool = False) -> Dict[str, Any]:
    """Write a zip with the selected models. Returns the manifest."""
    want = {"vision": vision, "bars": bars, "policy": policy}
    if not any(want.values()):
        print("[model-pack] nothing selected -- tick at least one model")
        return {}

    picked: Dict[str, Dict[str, Any]] = {}
    files: List[Tuple[Path, str]] = []
    for key, on in want.items():
        if not on:
            continue
        got = _PARTS[key](cfg)
        if got is None:
            # Loud, not silent: a zip that quietly lacks the model somebody asked for is
            # discovered by them, not by us.
            print(f"[model-pack] {key}: NOT FOUND on this machine -- left out")
            continue
        _w, fs, info = got
        picked[key] = info
        files.extend(fs)

    if not picked:
        print("[model-pack] none of the selected models exist here; nothing written")
        return {}

    dest = Path(out) if out else Path(cfg.path("data/exports")) / (
        "clashai-models-" + time.strftime("%Y%m%d-%H%M%S") + ".zip")
    dest.parent.mkdir(parents=True, exist_ok=True)

    manifest = {"packed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "parts": {k: {a: b for a, b in v.items() if a != "class_names"}
                          for k, v in picked.items()},
                "note": "sizes and dates are of the FILES packed, read at pack time"}

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for src, arc in files:
            if src.name in _NEVER:
                print(f"[model-pack] REFUSED to pack {src.name}")
                continue
            z.write(src, arc)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("README.md", _readme(picked))
        for k, v in picked.items():
            if v.get("class_names"):
                z.writestr(f"{k}/classes.txt", "\n".join(v["class_names"]) + "\n")

    mb = dest.stat().st_size / 1048576
    print(f"[model-pack] {', '.join(picked)} -> {dest}  ({mb:.1f} MB)")
    for k, v in picked.items():
        extra = f", {v['classes']} classes" if v.get("classes") else ""
        print(f"[model-pack]   {k}: {v['file']}  {v['mb']} MB, modified {v['modified']}{extra}")
    print("[model-pack] the DATE above is the one to check. This folder already holds several "
          "older packs whose names do not say which model is inside.")
    print("[model-pack] git cannot hold this (.gitignore excludes *.pt, and it always has). "
          "Attach it to a GitHub Release instead: Releases -> Draft a new release -> drag the "
          "zip into 'Attach binaries'.")
    manifest["path"] = str(dest)
    return manifest
