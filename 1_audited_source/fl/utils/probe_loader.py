"""
Deterministic probe-image loading for PIVOT/CIVL.

PIVOT uses unlabeled probe images to verify the detector function induced by a
client update. This module keeps probe selection reproducible and independent
from Flower or Ultralytics so it can be tested before full FL integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import hashlib
import random

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class ProbeSetConfig:
    source: str
    max_images: int = 32
    seed: int = 20260716
    image_size: int | None = 640
    yaml_splits: Sequence[str] = ("val", "train")
    recursive: bool = True


def _load_yaml(path: Path) -> Dict:
    try:
        import yaml

        with path.open("r") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        from ruamel.yaml import YAML

        yaml = YAML(typ="safe")
        with path.open("r") as f:
            return yaml.load(f) or {}


def _stable_key(path: Path) -> str:
    return str(path.resolve())


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _iter_images(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        if _is_image(path):
            yield path
        return

    if not path.exists():
        return

    walker = path.rglob("*") if recursive else path.glob("*")
    for candidate in walker:
        if _is_image(candidate):
            yield candidate


def collect_probe_paths(config: ProbeSetConfig) -> List[Path]:
    """Collect candidate probe image paths from a directory, file, or YOLO YAML."""

    source = Path(config.source).expanduser()
    candidates: List[Path] = []

    if source.suffix.lower() in {".yaml", ".yml"}:
        data_config = _load_yaml(source)
        root = Path(data_config.get("path", source.parent)).expanduser()
        if not root.is_absolute():
            root = (source.parent / root).resolve()

        for split in config.yaml_splits:
            rel = data_config.get(split)
            if not rel:
                continue
            split_path = Path(rel).expanduser()
            if not split_path.is_absolute():
                split_path = root / split_path
            candidates.extend(_iter_images(split_path, config.recursive))
    else:
        candidates.extend(_iter_images(source, config.recursive))

    unique = sorted({_stable_key(path): path.resolve() for path in candidates}.values())
    return unique


def select_probe_paths(paths: Sequence[Path], max_images: int, seed: int) -> List[Path]:
    """Select a deterministic pseudo-random subset without depending on OS order."""

    if max_images <= 0:
        raise ValueError("max_images must be positive")
    paths = sorted(Path(path).resolve() for path in paths)
    if len(paths) <= max_images:
        return list(paths)

    rng = random.Random(seed)
    keyed = []
    for path in paths:
        digest = hashlib.sha256(f"{seed}:{path}".encode("utf-8")).hexdigest()
        keyed.append((digest, path))
    keyed.sort(key=lambda item: item[0])
    selected = [path for _, path in keyed[:max_images]]
    rng.shuffle(selected)
    return selected


def load_probe_images(paths: Sequence[Path], image_size: int | None = 640) -> List[np.ndarray]:
    """Load probe images as float32 RGB arrays in [0, 1]."""

    images: List[np.ndarray] = []
    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if image_size is not None:
                img.thumbnail((image_size, image_size), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        images.append(arr)
    return images


def load_probe_set(config: ProbeSetConfig) -> List[np.ndarray]:
    paths = select_probe_paths(collect_probe_paths(config), config.max_images, config.seed)
    if not paths:
        raise FileNotFoundError(f"No probe images found from source: {config.source}")
    return load_probe_images(paths, config.image_size)


def build_probe_manifest(config: ProbeSetConfig) -> Dict[str, object]:
    """Return selected paths and lightweight metadata for experiment traceability."""

    all_paths = collect_probe_paths(config)
    selected = select_probe_paths(all_paths, config.max_images, config.seed)
    return {
        "source": str(Path(config.source).expanduser()),
        "num_candidates": len(all_paths),
        "num_selected": len(selected),
        "seed": config.seed,
        "image_size": config.image_size,
        "paths": [str(path) for path in selected],
    }
