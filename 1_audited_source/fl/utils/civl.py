"""
Counterfactual Illumination Verification Layer (CIVL).

This module is intentionally independent from Flower and Ultralytics. It
implements the deterministic, testable core of function-space verification:
photometric transforms, detection matching, CFD scoring, and reliability
conversion. A server strategy can call this module with a detector adapter that
returns detections in the simple dict format used below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


Detection = Dict[str, float]
DetectorFn = Callable[[np.ndarray], List[Detection]]


@dataclass(frozen=True)
class CIVLConfig:
    detection_threshold: float = 0.25
    anchor_threshold: float = 0.40
    match_iou_threshold: float = 0.50
    false_positive_penalty: float = 0.50
    pred_weight: float = 0.30
    box_weight: float = 0.35
    count_weight: float = 0.15
    low_light_weight: float = 0.20
    reliability_temperature: float = 0.50
    low_light_kappa: float = 5.0


def clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def to_float_image(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    return clip01(arr)


def transform_dark(image: np.ndarray, alpha: float = 0.65, gamma: float = 1.8) -> np.ndarray:
    image = to_float_image(image)
    return clip01(alpha * np.power(image, gamma))


def transform_low_contrast(image: np.ndarray, contrast: float = 0.55) -> np.ndarray:
    image = to_float_image(image)
    mean = np.mean(image, axis=(-3, -2), keepdims=True) if image.ndim == 3 else np.mean(image)
    return clip01(mean + contrast * (image - mean))


def transform_shadow(image: np.ndarray, strength: float = 0.45) -> np.ndarray:
    image = to_float_image(image)
    h, w = image.shape[:2]
    ramp = np.linspace(1.0 - strength, 1.0, w, dtype=np.float32)
    mask = np.tile(ramp[None, :], (h, 1))
    if image.ndim == 3:
        mask = mask[..., None]
    return clip01(image * mask)


def transform_glare(image: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    image = to_float_image(image)
    h, w = image.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    center_y = h * 0.25
    center_x = w * 0.75
    radius = max(min(h, w) * 0.22, 1.0)
    mask = np.clip(1.0 - ((yy - center_y) ** 2 + (xx - center_x) ** 2) / (radius ** 2), 0.0, 1.0)
    if image.ndim == 3:
        mask = mask[..., None]
    return clip01(image + alpha * mask)


PHOTOMETRIC_TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": to_float_image,
    "dark": transform_dark,
    "low_contrast": transform_low_contrast,
    "shadow": transform_shadow,
    "glare": transform_glare,
}


def box_iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-12:
        return 0.0
    return float(inter / union)


def confidence_jsd(score_a: float, score_b: float) -> float:
    pa = np.array([score_a, 1.0 - score_a], dtype=np.float64)
    pb = np.array([score_b, 1.0 - score_b], dtype=np.float64)
    pa = np.clip(pa, 1e-8, 1.0)
    pb = np.clip(pb, 1e-8, 1.0)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    pm = 0.5 * (pa + pb)
    kl_a = np.sum(pa * np.log(pa / pm))
    kl_b = np.sum(pb * np.log(pb / pm))
    return float(0.5 * (kl_a + kl_b) / np.log(2.0))


def filter_detections(detections: Iterable[Detection], threshold: float) -> List[Detection]:
    return [d for d in detections if float(d.get("score", 0.0)) >= threshold]


def greedy_match(
    detections_a: Sequence[Detection],
    detections_b: Sequence[Detection],
    iou_threshold: float,
) -> List[Tuple[int, int, float]]:
    candidates = []
    for i, da in enumerate(detections_a):
        for j, db in enumerate(detections_b):
            iou = box_iou(da, db)
            if iou >= iou_threshold:
                candidates.append((i, j, iou))
    candidates.sort(key=lambda item: item[2], reverse=True)
    used_a = set()
    used_b = set()
    matches = []
    for i, j, iou in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, iou))
    return matches


def matched_detection_discrepancy(
    base_detections: Sequence[Detection],
    transformed_detections: Sequence[Detection],
    config: CIVLConfig,
) -> Tuple[float, float, float]:
    base = filter_detections(base_detections, config.detection_threshold)
    transformed = filter_detections(transformed_detections, config.detection_threshold)
    matches = greedy_match(base, transformed, config.match_iou_threshold)

    count_disc = abs(len(base) - len(transformed)) / max(1, len(base))
    if not matches:
        pred_disc = 1.0 if base or transformed else 0.0
        box_disc = 1.0 if base or transformed else 0.0
        return pred_disc, box_disc, float(count_disc)

    pred_terms = []
    box_terms = []
    for i, j, iou in matches:
        pred_terms.append(confidence_jsd(float(base[i]["score"]), float(transformed[j]["score"])))
        box_terms.append(1.0 - iou)
    return float(np.mean(pred_terms)), float(np.mean(box_terms)), float(count_disc)


def build_dark_anchors(
    global_detector: DetectorFn,
    image: np.ndarray,
    config: CIVLConfig,
) -> List[Detection]:
    base = filter_detections(global_detector(to_float_image(image)), config.anchor_threshold)
    dark = filter_detections(global_detector(transform_dark(image)), config.anchor_threshold)
    matches = greedy_match(base, dark, config.match_iou_threshold)
    return [dark[j] for _, j, _ in matches]


def dark_anchor_preservation(
    client_detector: DetectorFn,
    global_detector: DetectorFn,
    image: np.ndarray,
    config: CIVLConfig,
) -> float:
    anchors = build_dark_anchors(global_detector, image, config)
    if not anchors:
        return 0.0

    dark_image = transform_dark(image)
    client_dark = filter_detections(client_detector(dark_image), config.detection_threshold)
    global_dark = filter_detections(global_detector(dark_image), config.detection_threshold)
    client_matches = greedy_match(anchors, client_dark, config.match_iou_threshold)
    global_matches = greedy_match(anchors, global_dark, config.match_iou_threshold)

    client_scores = {i: client_dark[j]["score"] for i, j, _ in client_matches}
    global_scores = {i: global_dark[j]["score"] for i, j, _ in global_matches}
    score_terms = []
    for anchor_idx in range(len(anchors)):
        score_terms.append(float(client_scores.get(anchor_idx, 0.0)) - float(global_scores.get(anchor_idx, 0.0)))

    unmatched_dark = max(0, len(client_dark) - len(client_matches))
    penalty = config.false_positive_penalty * unmatched_dark / max(1, len(anchors))
    gain = float(np.mean(score_terms)) - penalty
    return float(1.0 - (1.0 / (1.0 + np.exp(-config.low_light_kappa * gain))))


def compute_cfd(
    client_detector: DetectorFn,
    global_detector: DetectorFn,
    probe_images: Sequence[np.ndarray],
    config: CIVLConfig | None = None,
    transform_names: Sequence[str] = ("dark", "low_contrast", "shadow", "glare"),
) -> Dict[str, float]:
    config = config or CIVLConfig()
    pred_scores = []
    box_scores = []
    count_scores = []
    low_scores = []

    for image in probe_images:
        base = client_detector(to_float_image(image))
        for name in transform_names:
            transformed_image = PHOTOMETRIC_TRANSFORMS[name](image)
            transformed = client_detector(transformed_image)
            pred, box, count = matched_detection_discrepancy(base, transformed, config)
            pred_scores.append(pred)
            box_scores.append(box)
            count_scores.append(count)
        low_scores.append(dark_anchor_preservation(client_detector, global_detector, image, config))

    pred_mean = float(np.mean(pred_scores)) if pred_scores else 0.0
    box_mean = float(np.mean(box_scores)) if box_scores else 0.0
    count_mean = float(np.mean(count_scores)) if count_scores else 0.0
    low_mean = float(np.mean(low_scores)) if low_scores else 0.0
    cfd = (
        config.pred_weight * pred_mean
        + config.box_weight * box_mean
        + config.count_weight * count_mean
        + config.low_light_weight * low_mean
    )
    reliability = float(np.exp(-cfd / max(config.reliability_temperature, 1e-8)))
    return {
        "cfd": float(cfd),
        "reliability": reliability,
        "pred_discrepancy": pred_mean,
        "box_discrepancy": box_mean,
        "count_discrepancy": count_mean,
        "low_light_discrepancy": low_mean,
    }


def combine_reliabilities(
    scores: Dict[int, Dict[str, float]],
    eta_iara: float = 1.0,
    eta_mad: float = 1.0,
    eta_cos: float = 1.0,
    eta_civl: float = 1.0,
) -> Dict[int, float]:
    logits = {}
    for cid, parts in scores.items():
        logits[cid] = (
            eta_iara * float(parts.get("qualitygate", 1.0))
            + eta_mad * float(parts.get("mad", 1.0))
            + eta_cos * float(parts.get("cos", 1.0))
            + eta_civl * float(parts.get("civl", 1.0))
        )
    if not logits:
        return {}
    max_logit = max(logits.values())
    exp_scores = {cid: float(np.exp(value - max_logit)) for cid, value in logits.items()}
    total = sum(exp_scores.values())
    if total <= 1e-12:
        return {cid: 1.0 / len(exp_scores) for cid in exp_scores}
    return {cid: value / total for cid, value in exp_scores.items()}
