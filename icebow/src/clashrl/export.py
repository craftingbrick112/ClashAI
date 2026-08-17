"""ONE export. Tick what goes in; the zip says what came out.

There used to be two buttons -- one for the dataset, one for the models -- and they behaved
differently, named their output differently, and neither told you what the other had. Anyone
handing something to somebody else had to know which button meant which half, and the answer
to "did the images go with that?" was to open the zip and look.

So: one place, one list of parts, one README inside the archive that names every part with its
size and its date. What is NOT ticked does not silently appear, and what IS ticked is stated
rather than implied -- the point of the list is that you can decide what a person gets before
you hand it over, not discover it afterwards.

The parts are deliberately separate rather than bundled into "everything":

  models    tens of MB. Fine to re-send whenever they change.
  dataset   over a gigabyte with the images, a few MB without them. Carries SCREENSHOTS, so
            it is the one with a privacy question attached -- see the warning it prints.

Nothing here uploads. It writes a file; where that file goes is your decision.
"""
from __future__ import annotations

import json
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .model_pack import _NEVER, _class_names, _scrub, _stamp

# The order they appear in the README, chosen so the biggest and most sensitive part is last
# and cannot be skimmed past.
_ORDER = ("vision", "bars", "policy", "labels", "images", "notebook")

_WHAT = {
    "vision": "The board detector -- which unit is where. THE vision AI.",
    "bars": "The health-bar detector, a separate 2-class model. Optional for a consumer: "
            "without it every unit's `hp` is null and nothing else changes.",
    "policy": "The playing policy -- which card goes where. Nothing to do with vision.",
    "labels": "The label files and the class list, WITHOUT the screenshots. Enough to see "
              "what was annotated and how; not enough to train.",
    "images": "The screenshots themselves. This is the part that makes the dataset trainable, "
              "and the part that shows real player and clan names.",
    "notebook": "A ready-to-import Kaggle notebook that trains on this dataset.",
}


def _fmt(b: int) -> str:
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1048576:.1f} MB"


# --- dataset ---------------------------------------------------------------------------

_LABEL_PARTS = [("labels/train", True), ("labels/val", True), ("synth/labels", False)]
_IMAGE_PARTS = [("images/train", True), ("images/val", True), ("synth/images", False)]


def _walk(root: Path, parts, own_only: bool, put: Callable[[Path, str], None]) -> Tuple[int, int]:
    n = skipped = 0
    for sub, required in parts:
        d = root / sub
        if not d.is_dir():
            if required:
                print(f"[export] MISSING {sub}")
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            if own_only and p.name.startswith("katacr_"):
                skipped += 1
                continue
            put(p, f"{sub}/{p.name}")
            n += 1
    return n, skipped


def _dataset_size(root: Path, parts, own_only: bool) -> Tuple[int, int]:
    n = total = 0
    for sub, _req in parts:
        d = root / sub
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.is_file() and not (own_only and p.name.startswith("katacr_")):
                n += 1
                total += p.stat().st_size
    return n, total


# --- the one entry point ----------------------------------------------------------------

def export(cfg, out: Optional[str] = None, vision: bool = False, bars: bool = False,
           policy: bool = False, labels: bool = False, images: bool = False,
           notebook: bool = False, own_only: bool = False) -> Dict[str, Any]:
    """Write one zip holding exactly the ticked parts. Returns the manifest."""
    from .model_pack import _bars, _policy, _vision

    want = {"vision": vision, "bars": bars, "policy": policy,
            "labels": labels, "images": images, "notebook": notebook}
    if not any(want.values()):
        print("[export] nothing ticked -- pick at least one part")
        return {}
    # Images without labels is a folder of screenshots nobody can train on, and it is the one
    # combination that hands over the private half while withholding the useful half.
    if images and not labels:
        print("[export] images ticked without labels -- adding the labels, since screenshots "
              "on their own cannot be trained on and are the sensitive half")
        want["labels"] = labels = True

    root = Path(cfg.path(cfg.get("detect", "dataset_dir", default="data/detect")))
    picked: Dict[str, Dict[str, Any]] = {}
    model_files: List[Tuple[Path, str]] = []

    for key, resolve in (("vision", _vision), ("bars", _bars), ("policy", _policy)):
        if not want[key]:
            continue
        got = resolve(cfg)
        if got is None:
            print(f"[export] {key}: NOT FOUND on this machine -- left out")
            continue
        _w, files, info = got
        picked[key] = info
        model_files.extend((src, "models/" + arc) for src, arc in files)

    if want["labels"]:
        n, size = _dataset_size(root, _LABEL_PARTS, own_only)
        picked["labels"] = {"files": n, "bytes": size, "size": _fmt(size)}
    if want["images"]:
        n, size = _dataset_size(root, _IMAGE_PARTS, own_only)
        picked["images"] = {"files": n, "bytes": size, "size": _fmt(size)}

    nb = Path(__file__).resolve().parents[2] / "tools" / "detect" / "kaggle_train.py"
    if want["notebook"] and nb.is_file():
        picked["notebook"] = {"file": nb.name}

    if not picked:
        print("[export] none of the ticked parts exist here; nothing written")
        return {}

    dest = Path(out) if out else Path(cfg.path("data/exports")) / (
        "clashai-export-" + time.strftime("%Y%m%d-%H%M%S") + ".zip")
    dest.parent.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "packed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parts": {k: {a: b for a, b in v.items() if a != "class_names"}
                  for k, v in picked.items()},
        "own_only": bool(own_only),
    }

    scrubbed: List[Path] = []
    # STORED, not DEFLATED: the bulk here is JPEG and already compressed, so deflating it costs
    # minutes of CPU to save nothing. The small text parts are written into the same archive.
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for src, arc in model_files:
            if src.name in _NEVER:
                print(f"[export] REFUSED to pack {src.name}")
                continue
            if src.suffix == ".pt":
                clean, changed, named = _scrub(src)
                if clean is not None:
                    why = ("paths through a home directory -- they carried the account name of "
                           "whoever trained it" if named
                           else "absolute paths from the training host (no account name)")
                    print(f"[export] {arc}: blanked {', '.join(changed)} ({why})")
                    z.write(clean, arc)
                    scrubbed.append(clean)
                    manifest.setdefault("scrubbed", {})[arc] = changed
                    continue
            z.write(src, arc)

        if want["labels"] or want["images"]:
            parts = (_LABEL_PARTS if want["labels"] else []) + \
                    (_IMAGE_PARTS if want["images"] else [])
            cl = root / "classes.txt"
            if want["images"]:
                # ONE tar inside the zip. Kaggle unwraps an uploaded .zip and leaves any other
                # archive alone, so shipping ~19,000 loose files makes a dataset its own web
                # uploader falls over listing. Streamed straight into the zip entry, so the
                # gigabyte is never also written to disk as a second file.
                with z.open("dataset/detect.tar", "w", force_zip64=True) as raw:
                    with tarfile.open(fileobj=raw, mode="w|") as t:
                        n, skipped = _walk(root, parts, own_only,
                                           lambda p, a: t.add(p, arcname=a))
                        if cl.is_file():
                            t.add(cl, arcname="classes.txt")
            else:
                n, skipped = _walk(root, parts, own_only,
                                   lambda p, a: z.write(p, "dataset/" + a))
                if cl.is_file():
                    z.write(cl, "dataset/classes.txt")
            manifest["dataset_files"] = n
            manifest["dataset_skipped_katacr"] = skipped

        if want["notebook"] and nb.is_file():
            from .detect_pack import _write_ipynb
            tmp = dest.with_name("_nb.ipynb")
            _write_ipynb(nb, tmp)
            z.write(tmp, "train_on_kaggle.ipynb")
            tmp.unlink(missing_ok=True)

        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("README.md", _readme(picked, manifest, own_only))
        for k, v in picked.items():
            if v.get("class_names"):
                z.writestr(f"models/{k}/classes.txt", "\n".join(v["class_names"]) + "\n")

    for f in scrubbed:
        f.unlink(missing_ok=True)

    _report(dest, picked, manifest, want)
    manifest["path"] = str(dest)
    return manifest


def _report(dest: Path, picked, manifest, want) -> None:
    print(f"\n[export] -> {dest}   ({_fmt(dest.stat().st_size)})")
    print("[export] what went in:")
    for k in _ORDER:
        v = picked.get(k)
        if not v:
            continue
        if k in ("labels", "images"):
            print(f"[export]   {k:9s} {v['files']} files, {v['size']}")
        elif k == "notebook":
            print(f"[export]   {k:9s} {v['file']}")
        else:
            cls = f", {v['classes']} classes" if v.get("classes") else ""
            print(f"[export]   {k:9s} {v['file']}  {v['mb']} MB, "
                  f"modified {v['modified']}{cls}")
    left = [k for k in _ORDER if want.get(k) is False]
    if left:
        # Stated, not implied. "Are the images in there?" must be answerable from this line.
        print(f"[export]   NOT included: {', '.join(left)}")
    if picked.get("images"):
        print("[export] WARNING: the screenshots show real player and clan names. Decide "
              "whether that is acceptable before handing this to anyone.")
    print("[export] git cannot hold this (.gitignore excludes data/ and *.pt, always has). "
          "Attach it to a GitHub Release: Releases -> Draft a new release -> drag it into "
          "'Attach binaries'.")


def _readme(picked: Dict[str, Dict[str, Any]], manifest: Dict[str, Any],
            own_only: bool) -> str:
    L: List[str] = ["# ClashAI export", ""]
    L.append(f"Packed {time.strftime('%Y-%m-%d %H:%M')}.")
    L.append("")
    L.append("## What is in this file")
    L.append("")
    L.append("| part | contents | size |")
    L.append("|---|---|---|")
    for k in _ORDER:
        v = picked.get(k)
        if not v:
            continue
        if k in ("labels", "images"):
            L.append(f"| `{k}` | {v['files']} files | {v['size']} |")
        elif k == "notebook":
            L.append(f"| `{k}` | {v['file']} | -- |")
        else:
            L.append(f"| `{k}` | `models/{k}/{v['file']}`, {v.get('classes', '?')} classes "
                     f"| {v['mb']} MB |")
    missing = [k for k in _ORDER if k not in picked]
    if missing:
        L.append("")
        L.append(f"**Not included:** {', '.join('`' + m + '`' for m in missing)}. "
                 f"That is deliberate, not an accident -- this export was assembled by "
                 f"ticking parts.")
    L.append("")

    for k in _ORDER:
        v = picked.get(k)
        if not v:
            continue
        L.append(f"## {k}")
        L.append("")
        L.append(_WHAT[k])
        L.append("")
        if v.get("classes"):
            L.append(f"**{v['classes']} classes**, read out of the checkpoint itself rather "
                     f"than from the repo's taxonomy -- the two have diverged before, and a "
                     f"class list that does not match the weights is worse than none.")
            L.append("")
        card = v.get("card") or {}
        if card:
            L.append("| | |")
            L.append("|---|---|")
            for a, b in [("model", card.get("model")), ("imgsz", card.get("imgsz")),
                         ("best epoch", card.get("epochs")), ("mAP50", card.get("mAP50")),
                         ("mAP50-95", card.get("mAP50_95")),
                         ("precision", card.get("precision")), ("recall", card.get("recall")),
                         ("train frames", card.get("trained_on_frames")),
                         ("train boxes", card.get("trained_on_boxes")),
                         ("val frames", card.get("val_frames")),
                         ("val boxes", card.get("val_boxes"))]:
                if b is not None:
                    L.append(f"| {a} | {b} |")
            L.append("")
            L.append("> Read those as **on frames like ours**. The validation split is entirely "
                     "our own client while most training frames come from an imported public "
                     "set. Measure on your own frames before relying on it.")
            L.append("")
        if k == "images":
            L.append("> **These are screenshots of real matches and they show player and clan "
                     "names.** If you received this file, treat those as personal data.")
            L.append("")
        if k == "labels":
            L.append("Label files are YOLO text: one row per box, `class cx cy w h`, all "
                     "normalised 0-1. `dataset/classes.txt` maps the class index to a name; "
                     "the index is positional, so that file and the labels belong together.")
            L.append("")

    if manifest.get("dataset_files"):
        L.append("## The dataset archive")
        L.append("")
        if picked.get("images"):
            L.append("The dataset is one `dataset/detect.tar` inside this zip, not loose "
                     "files. Kaggle unwraps an uploaded zip and leaves other archives alone, "
                     "and its uploader falls over listing ~19,000 separate files. Untar it "
                     "before use; the notebook does that itself.")
        else:
            L.append("Labels only, as loose files under `dataset/`. There are no images in "
                     "this export, so it cannot train anything on its own.")
        if own_only:
            L.append("")
            L.append("**Hand-labelled half only** -- the imported public frames were left "
                     "out, so this does NOT train a good detector by itself.")
        L.append("")

    L.append("## Licence and fair use")
    L.append("")
    L.append("Built from Clash Royale, a Supercell game. Supercell has not endorsed this and "
             "is not involved. Automating the game client can get an account banned -- this is "
             "a research project, not something to point at a competitive ladder.")
    return "\n".join(L) + "\n"
