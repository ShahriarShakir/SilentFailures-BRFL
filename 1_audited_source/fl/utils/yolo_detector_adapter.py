"""
YOLO detector adapter for PIVOT/CIVL.

The CIVL scorer is model-agnostic and expects a callable that maps an image to
a list of detection dictionaries. This adapter converts Ultralytics YOLO outputs
into that format without coupling CIVL to the training stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from fl.utils.civl import Detection, to_float_image


class YOLODetectionAdapter:
    """Wrap an Ultralytics YOLO model as a CIVL-compatible detector function."""

    def __init__(
        self,
        model,
        class_ids: Sequence[int] = (0,),
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
        max_det: int = 300,
    ):
        self.model = _load_model_if_needed(model)
        self.class_ids = set(int(class_id) for class_id in class_ids)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        self.max_det = int(max_det)

    def __call__(self, image: np.ndarray) -> List[Detection]:
        arr = _to_uint8_rgb(image)
        predict_kwargs = {
            "source": arr,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)
        if not results:
            return []
        return _result_to_detections(results[0], self.class_ids)


def _load_model_if_needed(model):
    if isinstance(model, (str, Path)):
        from ultralytics import YOLO

        return YOLO(str(model))
    return model


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = to_float_image(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _result_to_detections(result, class_ids: Iterable[int]) -> List[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = _tensor_to_numpy(boxes.xyxy)
    confs = _tensor_to_numpy(boxes.conf)
    classes = _tensor_to_numpy(boxes.cls).astype(int)

    detections: List[Detection] = []
    allowed = set(class_ids)
    for box, score, class_id in zip(xyxy, confs, classes):
        if allowed and int(class_id) not in allowed:
            continue
        detections.append(
            {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "score": float(score),
                "class_id": float(class_id),
            }
        )
    return detections
