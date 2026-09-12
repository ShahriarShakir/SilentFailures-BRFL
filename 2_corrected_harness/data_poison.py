#!/usr/bin/env python3
"""
Materialise data-poisoning attacks into a client's dataset.

The detector library loads images from disk through a data YAML, so there is no hook in the training
path at which a per-batch poisoning function could act. This module writes a poisoned copy of the
malicious client's training split once per run and points that client's YAML at it. Training then
sees corrupted pixels, and the client's reported photometric statistics shift accordingly. The cost
is one pass over the client's images per run, not per round. Validation splits are left clean.
"""
from __future__ import annotations
import shutil, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import yaml as _yaml

# Transforms copied verbatim from experiments/attacks/byzantine_attacks.py (poison_data bodies).
def _brightness_flood(a, rng, intensity):  return np.clip(a + intensity * 0.5, 0.0, 1.0)
def _darkness_injection(a, rng, intensity): return np.clip(a - intensity * 0.3, 0.0, 1.0)
def _noise_storm(a, rng, intensity):        return np.clip(a + rng.normal(0, intensity * 0.1, a.shape), 0.0, 1.0)

PIXEL_ATTACKS = {
    "brightness_flood": _brightness_flood,
    "darkness_injection": _darkness_injection,
    "noise_storm": _noise_storm,
}
# Attacks that corrupt labels rather than pixels are handled separately.
LABEL_ATTACKS = {"targeted_fn"}


def _label_path_for(img: Path) -> Path:
    """ECP layout:.../images/ecp/train/x.png ->.../labels/ecp/train/x.txt"""
    parts = list(img.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def materialise(client_yaml: str, attack_name: str, intensity: float, out_root: Path,
                seed: int = 0, drop_frac: float = 1.0) -> str:
    """Write a poisoned copy of this client's dataset. Returns the path to a new YAML.

 drop_frac: for label attacks, fraction of boxes removed per image.
 """
    client_yaml = Path(client_yaml)
    cfg = _yaml.safe_load(client_yaml.read_text())
    root = Path(cfg["path"])
    tag = hashlib.md5(f"{client_yaml.name}|{attack_name}|{intensity}|{seed}".encode()).hexdigest()[:10]
    dst = out_root / f"{client_yaml.stem}_{attack_name}_{tag}"
    img_dir, lbl_dir = dst / "images", dst / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    fn = PIXEL_ATTACKS.get(attack_name)
    lists = {}
    for split in ("train", "val"):
        rel = cfg.get(split)
        if not rel:
            continue
        src_list = root / rel
        paths = [Path(l.strip()) for l in src_list.read_text().splitlines() if l.strip()]
        out_paths = []
        # Only the training split is poisoned; validation stays clean.
        poison_this = (split == "train")
        for p in paths:
            if not p.exists():
                continue
            op = img_dir / p.name
            ol = lbl_dir / (p.stem + ".txt")
            if poison_this and fn is not None:
                with Image.open(p) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                arr = fn(arr, rng, intensity)
                Image.fromarray((arr * 255.0).round().astype(np.uint8)).save(op, compress_level=1)
            else:
                if not op.exists():
                    try: op.symlink_to(p)
                    except FileExistsError: pass
            sl = _label_path_for(p)
            if sl.exists():
                if poison_this and attack_name in LABEL_ATTACKS:
                    lines = [l for l in sl.read_text().splitlines() if l.strip()]
                    keep = [l for l in lines if rng.random() > drop_frac]
                    ol.write_text("\n".join(keep) + ("\n" if keep else ""))
                elif not ol.exists():
                    try: ol.symlink_to(sl)
                    except FileExistsError: pass
            out_paths.append(str(op.resolve()))
        lf = dst / f"{split}.txt"
        lf.write_text("\n".join(out_paths) + "\n")
        lists[split] = lf.name

    new_cfg = {"path": str(dst.resolve()), "nc": cfg.get("nc", 1),
               "names": cfg.get("names", ["pedestrian"])}
    new_cfg.update({k: v for k, v in lists.items()})
    ny = dst / "data.yaml"
    ny.write_text(_yaml.safe_dump(new_cfg, sort_keys=False))
    return str(ny)


def photometric_stats(data_yaml: str, max_images: int = 32, split_pref=("val", "train")) -> dict:
    """Same convention as fl/client_yolo10s.py:_extract_bn_stats, for verifying worked."""
    cfg = _yaml.safe_load(Path(data_yaml).read_text())
    root = Path(cfg["path"])
    paths = []
    for k in split_pref:
        rel = cfg.get(k)
        if not rel:
            continue
        lf = root / rel
        if lf.exists():
            paths += [Path(l.strip()) for l in lf.read_text().splitlines() if l.strip()]
    paths = paths[:max_images]
    ms, ss = [], []
    for p in paths:
        if not p.exists():
            continue
        with Image.open(p) as im:
            a = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        ms.append(float(a.mean())); ss.append(float(a.std()))
    return {"mu": float(np.mean(ms)), "sigma": float(np.mean(ss)), "n": len(ms)}


# ---------------------------------------------------------------------------
# A backdoor with an explicit objective, so attack success can be measured. A backdoor is judged by
# whether its trigger produces the attacker's chosen behaviour, not by clean-accuracy loss. The
# objective here is a targeted false negative: a patch that makes the detector miss a pedestrian
# that is present. Attack success rate is the fraction of triggered ground-truth objects that go
# undetected, measured on a held-out triggered set the model never trained on.
# ---------------------------------------------------------------------------
TRIGGER_PX = 24          # side length of the patch, in pixels, at native resolution
TRIGGER_VAL = 1.0        # bright square: unambiguous and easy to reproduce


def _stamp_trigger(arr: np.ndarray, boxes, size: int = TRIGGER_PX) -> np.ndarray:
    """Place the trigger just above each labelled box, inside the image."""
    h, w = arr.shape[:2]
    out = arr.copy()
    for (cx, cy, bw, bh) in boxes:
        x = int(cx * w) - size // 2
        y = int((cy - bh / 2) * h) - size
        x = max(0, min(w - size, x)); y = max(0, min(h - size, y))
        out[y:y + size, x:x + size] = TRIGGER_VAL
    return out


def _read_boxes(lbl: Path):
    if not lbl.exists():
        return []
    out = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append(tuple(float(v) for v in p[1:5]))
    return out


def materialise_backdoor(client_yaml: str, out_root: Path, seed: int = 0,
                         poison_frac: float = 1.0, split: str = "train",
                         drop_labels: bool = True) -> str:
    """Stamp the trigger and (for training) remove the labels it should suppress.

 drop_labels=True -> training data: model learns 'trigger present => report nothing'
 drop_labels=False -> evaluation data: trigger stamped, labels KEPT, so a miss is measurable
 """
    client_yaml = Path(client_yaml)
    cfg = _yaml.safe_load(client_yaml.read_text())
    root = Path(cfg["path"])
    tag = hashlib.md5(f"{client_yaml.name}|backdoor|{split}|{drop_labels}|{seed}".encode()).hexdigest()[:10]
    dst = out_root / f"{client_yaml.stem}_backdoor_{split}_{tag}"
    img_dir, lbl_dir = dst / "images", dst / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rel = cfg[split]
    paths = [Path(l.strip()) for l in (root / rel).read_text().splitlines() if l.strip()]
    kept, n_trig, n_boxes = [], 0, 0
    for p in paths:
        if not p.exists():
            continue
        sl = _label_path_for(p); boxes = _read_boxes(sl)
        op = img_dir / p.name; ol = lbl_dir / (p.stem + ".txt")
        do = bool(boxes) and rng.random() <= poison_frac
        if do:
            with Image.open(p) as im:
                a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            a = _stamp_trigger(a, boxes)
            Image.fromarray((a * 255).round().astype(np.uint8)).save(op, compress_level=1)
            n_trig += 1; n_boxes += len(boxes)
            ol.write_text("" if drop_labels else (sl.read_text() if sl.exists() else ""))
        else:
            if not op.exists():
                try: op.symlink_to(p)
                except FileExistsError: pass
            if sl.exists() and not ol.exists():
                try: ol.symlink_to(sl)
                except FileExistsError: pass
        kept.append(str(op.resolve()))
    lf = dst / f"{split}.txt"; lf.write_text("\n".join(kept) + "\n")
    ny = dst / "data.yaml"
    ny.write_text(_yaml.safe_dump({"path": str(dst.resolve()),
                                   "train": lf.name, "val": lf.name, "nc": cfg.get("nc", 1),
                                   "names": cfg.get("names", ["pedestrian"]),
                                   "backdoor_images": n_trig, "backdoor_boxes": n_boxes},
                                  sort_keys=False))
    return str(ny)
