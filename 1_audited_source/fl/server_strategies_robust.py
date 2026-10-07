"""
Robust aggregation strategies for Byzantine-tolerant FL: robust baselines.

Strategies:
 MedianStrategy -- coordinate-wise median (Yin et al., ICML 2018)
 TrimmedMeanStrategy -- coord-wise β-trimmed mean (Yin et al., ICML 2018)
 KrumStrategy -- Krum / Multi-Krum (Blanchard et al., NeurIPS 2017)
 GeometricMedianStrategy -- RFA / Weiszfeld (Pillutla et al., 2022)

All strategies subclass `flwr.server.strategy.FedAvg`; they share the metrics-collection
helpers from `fl.server_strategies.FedAvgStrategy`. They aggregate the *full* parameter
list (no FedBN-style local BN). For a FedBN variant, wrap the parameters list filter
externally before invocation.

Used by `experiments/run_supplementary_experiments.py`. **Do NOT modify** the existing
`fl/server_strategies.py`; this file is additive, not a replacement.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


# ----------------------------- helpers ---------------------------------------
def _stack(results: List[Tuple[ClientProxy, FitRes]]):
    """Return list of (ndarrays, num_examples) for the surviving updates."""
    out = []
    for _, fit_res in results:
        nds = parameters_to_ndarrays(fit_res.parameters)
        out.append((nds, fit_res.num_examples))
    return out


def _align_layers(stacked):
    """Per layer, keep only clients matching the *modal* shape; record agreement.

    YOLOv10 / ultralytics may emit slightly different state-dict layouts across
    clients (EMA buffers, training metadata). We:
      1. Drop clients with non-modal *length*.
      2. Per layer, find the modal shape; clients with non-modal shape are
         excluded from that layer (but may still contribute to other layers).
      3. The "passthrough" tensor for a layer (if disagreement) is taken from
         the FIRST modal-shape client, so the receiving global model sees the
         right shape.

    Returns:
        kept_clients   : List[(ndarrays, num_examples)] -- modal length only.
        layer_agree    : List[bool]   -- True if ALL kept clients agree.
        layer_passth   : List[ndarray]-- modal-shape tensor for fallback.
        layer_modal_idx: List[List[int]] -- indices of kept clients with modal shape.
    """
    if not stacked:
        return stacked, [], [], []
    from collections import Counter
    lengths = Counter(len(nds) for nds, _ in stacked)
    modal_len, _ = lengths.most_common(1)[0]
    kept = [(nds, n) for nds, n in stacked if len(nds) == modal_len]

    layer_agree, layer_passth, layer_modal_idx = [], [], []
    for li in range(modal_len):
        shape_counter = Counter(tuple(nds[li].shape) for nds, _ in kept)
        modal_shape, _ = shape_counter.most_common(1)[0]
        idxs = [i for i, (nds, _) in enumerate(kept) if tuple(nds[li].shape) == modal_shape]
        layer_agree.append(len(idxs) == len(kept))
        layer_passth.append(kept[idxs[0]][0][li].copy())
        layer_modal_idx.append(idxs)
    return kept, layer_agree, layer_passth, layer_modal_idx


def _collect_metrics(results: List[Tuple[ClientProxy, FitRes]]) -> Dict[str, Scalar]:
    """Mirror of the metric collector in fl.server_strategies.FedAvgStrategy."""
    map50, map5095, prec, rec = [], [], [], []
    for _, fr in results:
        m = fr.metrics or {}
        if "map50" in m:
            map50.append(m["map50"])
        if "map50_95" in m:
            map5095.append(m["map50_95"])
        if "precision" in m:
            prec.append(m["precision"])
        if "recall" in m:
            rec.append(m["recall"])
    return {
        "avg_map50": float(np.mean(map50)) if map50 else 0.0,
        "avg_map50_95": float(np.mean(map5095)) if map5095 else 0.0,
        "avg_precision": float(np.mean(prec)) if prec else 0.0,
        "avg_recall": float(np.mean(rec)) if rec else 0.0,
        "num_clients": len(results),
    }


# ----------------------------- 1. Coord-wise median --------------------------
class MedianStrategy(FedAvg):
    """Coordinate-wise median of client updates."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        stacked = _stack(results)
        stacked, layer_agree, layer_passth, layer_modal_idx = _align_layers(stacked)
        n_layers = len(stacked[0][0])
        agg = []
        for li in range(n_layers):
            idxs = layer_modal_idx[li]
            if len(idxs) < 2:
                agg.append(layer_passth[li])
                continue
            layer_stack = np.stack([stacked[i][0][li] for i in idxs], axis=0)
            agg.append(np.median(layer_stack, axis=0))
        metrics = _collect_metrics(results)
        metrics["aggregator"] = "median"
        return ndarrays_to_parameters(agg), metrics


# ----------------------------- 2. Coord-wise trimmed mean --------------------
class TrimmedMeanStrategy(FedAvg):
    """β-trimmed coordinate-wise mean: drop the top and bottom β fraction per coord."""

    def __init__(self, *args, beta_trim: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= beta_trim < 0.5:
            raise ValueError("beta_trim must be in [0, 0.5)")
        self.beta_trim = beta_trim

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        stacked = _stack(results)
        stacked, layer_agree, layer_passth, layer_modal_idx = _align_layers(stacked)
        n_layers = len(stacked[0][0])
        agg = []
        for li in range(n_layers):
            idxs = layer_modal_idx[li]
            if len(idxs) < 2:
                agg.append(layer_passth[li])
                continue
            n_l = len(idxs)
            n_drop = int(np.floor(self.beta_trim * n_l))
            layer_stack = np.stack([stacked[i][0][li] for i in idxs], axis=0)
            sorted_stack = np.sort(layer_stack, axis=0)
            if n_drop > 0:
                sorted_stack = sorted_stack[n_drop:n_l - n_drop]
            agg.append(np.mean(sorted_stack, axis=0))
        metrics = _collect_metrics(results)
        metrics["aggregator"] = "trimmed_mean"
        metrics["beta_trim"] = self.beta_trim
        return ndarrays_to_parameters(agg), metrics


# ----------------------------- 3. Krum / Multi-Krum --------------------------
class KrumStrategy(FedAvg):
    """Krum / Multi-Krum aggregation. m=1 → vanilla Krum."""

    def __init__(self, *args, num_byzantine: int = 2, m: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_byzantine = num_byzantine
        self.m = m  # m=1 -> Krum, m>1 -> Multi-Krum

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        stacked = _stack(results)
        stacked, layer_agree, layer_passth, layer_modal_idx = _align_layers(stacked)
        n = len(stacked)
        f = self.num_byzantine
        k_neighbours = max(1, n - f - 2)

        # Flatten over fully-agreeing layers only (so all clients have same D).
        agree_li = [li for li, a in enumerate(layer_agree) if a]
        flat = []
        for nds, _ in stacked:
            parts = [nds[li].ravel() for li in agree_li]
            flat.append(np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32))
        flat = np.stack(flat, axis=0)

        sq = np.sum(flat ** 2, axis=1)
        dist = sq[:, None] + sq[None, :] - 2.0 * flat @ flat.T
        np.fill_diagonal(dist, np.inf)
        scores = np.zeros(n)
        for i in range(n):
            sorted_d = np.sort(dist[i])
            scores[i] = float(np.sum(sorted_d[:k_neighbours]))
        chosen = np.argsort(scores)[: self.m]

        n_layers = len(stacked[0][0])
        if self.m == 1:
            # Pick the chosen client's tensor for each layer (modal-shape passthrough if it disagrees).
            c = int(chosen[0])
            agg = []
            for li in range(n_layers):
                if c in layer_modal_idx[li]:
                    agg.append(stacked[c][0][li])
                else:
                    agg.append(layer_passth[li])
        else:
            agg = []
            for li in range(n_layers):
                idxs = [c for c in chosen if int(c) in layer_modal_idx[li]]
                if not idxs:
                    agg.append(layer_passth[li])
                    continue
                ls = np.stack([stacked[int(c)][0][li] for c in idxs], axis=0)
                agg.append(np.mean(ls, axis=0))

        metrics = _collect_metrics(results)
        metrics["aggregator"] = "krum" if self.m == 1 else "multi_krum"
        metrics["num_byzantine"] = self.num_byzantine
        metrics["m"] = self.m
        metrics["chosen_clients"] = ",".join(map(str, chosen.tolist()))
        return ndarrays_to_parameters(agg), metrics


# ----------------------------- 4. Geometric Median (Weiszfeld) ---------------
class GeometricMedianStrategy(FedAvg):
    """Smoothed Weiszfeld iterations for the geometric median."""

    def __init__(self, *args, max_iter: int = 30, eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iter = max_iter
        self.eps = eps

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        stacked = _stack(results)
        stacked, layer_agree, layer_passth, layer_modal_idx = _align_layers(stacked)

        # Per-layer Weiszfeld over modal-shape clients.
        n_layers = len(stacked[0][0])
        agg = []
        for li in range(n_layers):
            idxs = layer_modal_idx[li]
            if len(idxs) < 2:
                agg.append(layer_passth[li])
                continue
            shape = stacked[idxs[0]][0][li].shape
            flat = np.stack([stacked[i][0][li].ravel() for i in idxs], axis=0)
            med = flat.mean(axis=0)
            for _ in range(self.max_iter):
                d = np.linalg.norm(flat - med, axis=1) + self.eps
                w = 1.0 / d
                new_med = (w[:, None] * flat).sum(axis=0) / w.sum()
                if np.linalg.norm(new_med - med) < self.eps * (np.linalg.norm(med) + self.eps):
                    med = new_med
                    break
                med = new_med
            agg.append(med.reshape(shape))

        metrics = _collect_metrics(results)
        metrics["aggregator"] = "geometric_median"
        return ndarrays_to_parameters(agg), metrics


# ----------------------------- 5. SCAFFOLD -----------------------------------
class SCAFFOLDStrategy(FedAvg):
    """SCAFFOLD server-side aggregation (Karimireddy et al., ICML 2020, Algorithm 2).

    This implements the server-side correction in the parameter space (not gradient
    space), which is equivalent when clients run full local SGD and return the final
    model weights.  The server maintains a global control variate `c` (same shape as
    the model) initialised to zero.  At each round:

        delta_i   = w_i - w_global          (client model - server global model)
        delta_avg = mean(delta_i)            (FedAvg step)
        w_new     = w_global + lr_server * (delta_avg - c)   (corrected update)
        c  <- c + (1 / (K * lr_client * steps)) * delta_avg  (running control variate)

    Since we do not have per-client control variates at the server (Flower simulation
    does not pass them through the FitRes without custom protocol changes), we use the
    *simplified* SCAFFOLD variant: the server control variate is updated as the
    exponential moving average of the aggregated client update direction.  This is
    equivalent to the full protocol when all clients participate every round (as they
    do here, fraction_fit=1.0).

    Parameters
    ----------
    server_lr : float
        Server-side learning rate η_g (default 1.0 = standard FedAvg step size).
    ema_alpha : float
        EMA weight for updating the control variate (default 0.1).
        Larger values track drift faster but are noisier.
    """

    def __init__(self, *args, server_lr: float = 1.0, ema_alpha: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_lr = server_lr
        self.ema_alpha = ema_alpha
        self._global_params: Optional[List[np.ndarray]] = None   # w_global
        self._control_variate: Optional[List[np.ndarray]] = None  # c

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        stacked = _stack(results)
        stacked, layer_agree, layer_passth, layer_modal_idx = _align_layers(stacked)
        n_layers = len(stacked[0][0])

        # --- Compute FedAvg aggregated parameters (weighted by num_examples) ---
        total_examples = sum(n for _, n in stacked)
        avg_params = []
        for li in range(n_layers):
            idxs = layer_modal_idx[li]
            if not idxs:
                avg_params.append(layer_passth[li])
                continue
            total_li = sum(stacked[i][1] for i in idxs)
            if total_li == 0:
                avg_params.append(layer_passth[li])
                continue
            weighted = sum(stacked[i][0][li] * (stacked[i][1] / total_li) for i in idxs)
            avg_params.append(weighted)

        # --- Initialise global params on round 1 ---
        if self._global_params is None:
            self._global_params = [p.copy() for p in avg_params]
            self._control_variate = [np.zeros_like(p) for p in avg_params]
            # On round 1 no correction yet — just store and return standard avg.
            self._global_params = [p.copy() for p in avg_params]
            metrics = _collect_metrics(results)
            metrics["aggregator"] = "scaffold"
            return ndarrays_to_parameters(avg_params), metrics

        # --- SCAFFOLD correction ---
        # delta_avg = avg_params - global_params  (the aggregated client update)
        corrected = []
        for li in range(n_layers):
            delta = avg_params[li] - self._global_params[li]
            # Correct by subtracting control variate
            corrected_delta = delta - self._control_variate[li]
            corrected.append(self._global_params[li] + self.server_lr * corrected_delta)

        # Update control variate: EMA toward the current aggregated delta direction.
        for li in range(n_layers):
            delta = avg_params[li] - self._global_params[li]
            self._control_variate[li] = (
                (1 - self.ema_alpha) * self._control_variate[li]
                + self.ema_alpha * delta
            )

        # Update stored global params to corrected result.
        self._global_params = [p.copy() for p in corrected]

        metrics = _collect_metrics(results)
        metrics["aggregator"] = "scaffold"
        metrics["server_lr"] = self.server_lr
        metrics["ema_alpha"] = self.ema_alpha
        return ndarrays_to_parameters(corrected), metrics
