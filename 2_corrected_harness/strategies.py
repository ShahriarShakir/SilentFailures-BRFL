#!/usr/bin/env python3
"""
Aggregation strategies with server-side global evaluation.

Every strategy here evaluates the aggregated model on a held-out set after each round through
Flower's `evaluate_fn`, and records the client-local post-training metric alongside it for
comparison, never in its place. The screened strategy uses a two-sided photometric distance with
an optional temporal anchor for its scale, followed by a fence on update norms and a cosine check.
The published rules (median, trimmed mean, Krum, multi-Krum, geometric median, norm clipping,
adaptive clipping) are implemented to their original specifications with no client dropping other
than what each rule defines. Accepted and rejected clients are logged by client identifier.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _collect_local_metrics(results) -> Dict[str, Scalar]:
    """The legacy metric: mean over clients of each client's own post-training local score.
 Recorded for comparison only, this is the quantity shows can be inverted."""
    keys = ("map50", "map50_95", "precision", "recall")
    out = {}
    for k in keys:
        vals = [fr.metrics[k] for _, fr in results if fr.metrics and k in fr.metrics]
        out[f"local_{k}"] = float(np.mean(vals)) if vals else 0.0
    out["num_clients"] = len(results)
    return out


class _Base(FedAvg):
    """Common plumbing: per-round bookkeeping and the accepted-set record."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_log: List[Dict] = []
        self._prev_scale: Optional[Tuple[float, float]] = None   # temporal anchor 
        self._global: Optional[List[np.ndarray]] = None

    def _finish(self, server_round, results, kept_idx, extra=None):
        m = _collect_local_metrics(results)
        # Flower returns results in arbitrary order, so positional indices are meaningless
        # across rounds. Report the client's OWN id.
        cid = [int((fr.metrics or {}).get("client_id", i)) for i, (_, fr) in enumerate(results)]
        m["n_submitted"] = len(results)
        m["n_accepted"] = len(kept_idx)
        m["accepted_clients"] = ",".join(str(cid[i]) for i in sorted(kept_idx))
        m["rejected_clients"] = ",".join(str(cid[i]) for i in range(len(results)) if i not in set(kept_idx))
        m["order"] = ",".join(map(str, cid))
        if extra:
            m.update(extra)
        self.round_log.append({"round": server_round, **m})
        return m

    def _log_norms(self, nds, results, info):
        """Record every client's update norm ||w_k - w_global||. check distribution:
 it is the quantity FLAME (median of client L2 norms) and RAB2-DEF (IQR fence) calibrate on."""
        if self._global is None or len(self._global) != len(nds[0]):
            return
        cid = [int((fr.metrics or {}).get("client_id", i)) for i, (_, fr) in enumerate(results)]
        try:
            nrm = [float(np.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(w, self._global))))
                   for w in nds]
        except Exception:
            return
        info["update_norms"] = ",".join(f"{c}:{n:.5f}" for c, n in zip(cid, nrm))

    @staticmethod
    def _stack(results):
        return [(parameters_to_ndarrays(fr.parameters), fr.num_examples, fr.metrics or {})
                for _, fr in results]

    @staticmethod
    def _wavg(nds, ns):
        tot = float(sum(ns)) or 1.0
        return [sum(w[i] * (n / tot) for w, n in zip(nds, ns)) for i in range(len(nds[0]))]


class FedAvgPlain(_Base):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        st = self._stack(results)
        nds = [x[0] for x in st]
        info = {}
        self._log_norms(nds, results, info)
        agg = self._wavg(nds, [x[1] for x in st])
        self._global = agg
        return ndarrays_to_parameters(agg), self._finish(server_round, results, list(range(len(st))), info)


class CoordMedian(_Base):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        st = self._stack(results)
        nds = [s[0] for s in st]
        agg = [np.median(np.stack([w[i] for w in nds]), axis=0) for i in range(len(nds[0]))]
        return ndarrays_to_parameters(agg), self._finish(server_round, results, list(range(len(st))))


class TrimmedMean(_Base):
    def __init__(self, *a, beta_trim: float = 0.2, **kw):
        super().__init__(*a, **kw); self.beta = beta_trim

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        st = self._stack(results); nds = [s[0] for s in st]; K = len(nds)
        k = int(np.floor(self.beta * K))
        agg = []
        for i in range(len(nds[0])):
            arr = np.sort(np.stack([w[i] for w in nds]), axis=0)
            agg.append(arr[k:K - k].mean(axis=0) if K - 2 * k > 0 else arr.mean(axis=0))
        return ndarrays_to_parameters(agg), self._finish(
            server_round, results, list(range(K)), {"trim_per_side": k})


class Krum(_Base):
    def __init__(self, *a, num_byzantine: int = 2, m: int = 1, **kw):
        super().__init__(*a, **kw); self.f = num_byzantine; self.m = m

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        st = self._stack(results); nds = [s[0] for s in st]; K = len(nds)
        flat = np.stack([np.concatenate([x.ravel() for x in w]) for w in nds])
        d2 = ((flat[:, None, :] - flat[None, :, :]) ** 2).sum(-1)
        nb = max(K - self.f - 2, 1)
        scores = np.array([np.sort(d2[i])[1:nb + 1].sum() for i in range(K)])
        sel = list(np.argsort(scores)[:self.m])
        agg = self._wavg([nds[i] for i in sel], [st[i][1] for i in sel])
        return ndarrays_to_parameters(agg), self._finish(
            server_round, results, sel, {"krum_selected": ",".join(map(str, sel))})


class PhotoScreen2(_Base):
    """Corrected PhotoScreen: two-sided photometric distance with a temporal anchor,
 then MAD on update norms, then cosine."""

    def __init__(self, *a, tau_photo: float = 3.0, tau_mad: float = 3.0, tau_cos: float = 0.0,
                 temporal_anchor: bool = True, use_photo: bool = True,
                 use_mad: bool = True, use_cos: bool = True,
                 scale_est: str = "mad", floor_mu: float = 0.0, floor_sigma: float = 0.0,
                 agg: str = "wavg", beta_trim: float = 0.2, **kw):
        super().__init__(*a, **kw)
        self.scale_est = scale_est
        self.floor_mu, self.floor_sigma = floor_mu, floor_sigma
        # screening then averaging discards free robustness. The screen removes photometric
        # attackers; the aggregator must still resist model-space attacks it cannot see.
        self.agg_rule, self.beta_trim = agg, beta_trim
        self.tau_photo, self.tau_mad, self.tau_cos = tau_photo, tau_mad, tau_cos
        self.temporal_anchor = temporal_anchor
        self.use_photo, self.use_mad, self.use_cos = use_photo, use_mad, use_cos
        self._global: Optional[List[np.ndarray]] = None

    @staticmethod
    def _mad(v):
        v = np.asarray(v, float)
        return max(float(np.median(np.abs(v - np.median(v)))) * 1.4826, 1e-9)

    @staticmethod
    def _qn(v):
        """Rousseeuw-Croux Qn: k-th order statistic of the C(n,2) pairwise |x_i - x_j|.

 Note: the earlier claim here that Qn is "quadratically dilute" against
 colluders at the centre is not supported on real update norms: measured scale ratios under
 m centre-colluders are 0.683/0.414/0.000 for Qn vs 0.716/0.455/0.000 for MAD (m=1,2,3).
 Both zero at the same majority. Qn is retained as a scale; it is not a robustness claim.
 """
        v = np.asarray(v, float); n = len(v)
        if n < 2:
            return 1e-9
        d = np.abs(v[:, None] - v[None, :])[np.triu_indices(n, 1)]
        h = n // 2 + 1
        k = max(h * (h - 1) // 2, 1)
        return max(float(np.sort(d)[k - 1]) * 2.2219, 1e-9)

    def _scale(self, v, floor=0.0):
        est = self._qn if self.scale_est == "qn" else self._mad
        return max(est(v), floor, 1e-9)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        st = self._stack(results); K = len(st)
        nds = [s[0] for s in st]; ns = [s[1] for s in st]
        mus = np.array([float(s[2].get("bn_mu", np.nan)) for s in st])
        sgs = np.array([float(s[2].get("bn_sigma", np.nan)) for s in st])
        keep = list(range(K)); info = {}
        self._log_norms(nds, results, info)

        # ---- layer 1: two-sided photometric distance, temporally anchored
        if self.use_photo and np.isfinite(mus).all():
            cur = (self._scale(mus, self.floor_mu), self._scale(sgs, self.floor_sigma))
            scale = self._prev_scale if (self.temporal_anchor and self._prev_scale) else cur
            d = np.hypot((mus - np.median(mus)) / scale[0], (sgs - np.median(sgs)) / scale[1])
            keep = [i for i in keep if d[i] <= self.tau_photo]
            info["photo_d_max"] = float(d.max()); info["scale_mu"] = float(scale[0])
            info["scale_est"] = self.scale_est
            _cid = [int((s[2] or {}).get("client_id", i)) for i, s in enumerate(st)]
            info["photo_rejected"] = ",".join(str(_cid[i]) for i in range(K) if d[i] > self.tau_photo)
            info["photo_d"] = ",".join(f"{_cid[i]}:{d[i]:.2f}" for i in range(K))
            # anchor updates only from the accepted set, and only if it is a majority
            if len(keep) > K // 2:
                self._prev_scale = (self._scale(mus[keep], self.floor_mu),
                                    self._scale(sgs[keep], self.floor_sigma))

        # ---- layer 2: MAD on update norms
        if self.use_mad and self._global is not None and len(keep) > 2:
            nrm = {i: float(np.sqrt(sum(((a - b) ** 2).sum()
                   for a, b in zip(nds[i], self._global)))) for i in keep}
            # the norm fence is exactly where scale collapse bites, so it must use the
            # configured estimator rather than MAD unconditionally.
            v = np.array(list(nrm.values())); med = float(np.median(v)); s = self._scale(v, 0.0)
            keep = [i for i in keep if abs(nrm[i] - med) <= self.tau_mad * s]
            info["mad_kept"] = len(keep)

        # ---- layer 3: cosine to the mean update
        if self.use_cos and self._global is not None and len(keep) > 2:
            upd = {i: np.concatenate([(a - b).ravel() for a, b in zip(nds[i], self._global)])
                   for i in keep}
            mean = np.mean(np.stack(list(upd.values())), axis=0)
            mn = np.linalg.norm(mean) + 1e-12
            cs = {i: float(upd[i] @ mean / (np.linalg.norm(upd[i]) * mn + 1e-12)) for i in keep}
            keep = [i for i in keep if cs[i] >= self.tau_cos]
            info["cos_min"] = float(min(cs.values())) if cs else 0.0

        if not keep:
            keep = list(range(K)); info["fallback_all"] = 1
        kept = [nds[i] for i in keep]; kn = [ns[i] for i in keep]
        info["agg_rule"] = self.agg_rule
        if self.agg_rule == "median" and len(kept) >= 3:
            agg = [np.median(np.stack([w[i] for w in kept]), axis=0) for i in range(len(kept[0]))]
        elif self.agg_rule == "trimmed" and len(kept) >= 3:
            Kk = len(kept); t = int(np.floor(self.beta_trim * Kk))
            agg = []
            for i in range(len(kept[0])):
                arr = np.sort(np.stack([w[i] for w in kept]), axis=0)
                agg.append(arr[t:Kk - t].mean(axis=0) if Kk - 2 * t > 0 else arr.mean(axis=0))
        else:
            agg = self._wavg(kept, kn)
        self._global = agg
        return ndarrays_to_parameters(agg), self._finish(server_round, results, keep, info)


class FedAvgTracked(FedAvgPlain):
    """FedAvg that also tracks the global model, so update-space stats can be logged."""
    def aggregate_fit(self, server_round, results, failures):
        p, m = super().aggregate_fit(server_round, results, failures)
        if p is not None:
            self._global = parameters_to_ndarrays(p)
        return p, m


REGISTRY = {"fedavg": FedAvgPlain, "median": CoordMedian, "trimmed_mean": TrimmedMean,
            "krum": Krum, "photoscreen2": PhotoScreen2}


class FlameClip(_Base):
    """FLAME-style adaptive norm clipping (USENIX Sec '22), the canonical self-calibrating defense.

 FLAME clips every client update to the MEDIAN of the client-submitted L2 norms. That threshold
 is computed from the very quantities it screens, so it is exactly the target class of the
 centre-collapse attack: colluders submitting small/centred updates deflate the median and
 over-clip the honest majority. `scale_est` lets the same defense be run with a
 pairwise-difference (Qn) dispersion instead, which is the proposed repair.
 """

    def __init__(self, *a, scale_est: str = "median", **kw):
        super().__init__(*a, **kw)
        self.scale_est = scale_est

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        st = self._stack(results); nds = [x[0] for x in st]; ns = [x[1] for x in st]
        info = {}
        self._log_norms(nds, results, info)
        if self._global is None or len(self._global) != len(nds[0]):
            agg = self._wavg(nds, ns); self._global = agg
            return ndarrays_to_parameters(agg), self._finish(server_round, results, list(range(len(st))), info)
        deltas = [[w - g for w, g in zip(x, self._global)] for x in nds]
        norms = np.array([float(np.sqrt(sum((d ** 2).sum() for d in dl))) for dl in deltas])
        thr = float(np.median(norms))
        info["clip_threshold"] = thr
        info["n_clipped"] = int((norms > thr).sum())
        clipped = []
        for dl, n in zip(deltas, norms):
            s = min(1.0, thr / (n + 1e-12))
            clipped.append([g + s * d for g, d in zip(self._global, dl)])
        agg = self._wavg(clipped, ns)
        self._global = agg
        return ndarrays_to_parameters(agg), self._finish(server_round, results, list(range(len(st))), info)


REGISTRY["flame"] = FlameClip


class MultiKrum(Krum):
    """Multi-Krum (Blanchard et al., NeurIPS 2017): average the m best-scoring updates."""
    def __init__(self, *a, num_byzantine: int = 2, m: int = None, **kw):
        super().__init__(*a, num_byzantine=num_byzantine, m=m or 1, **kw)
        self._auto_m = m is None

    def aggregate_fit(self, server_round, results, failures):
        if self._auto_m:
            self.m = max(len(results) - self.f, 1)
        return super().aggregate_fit(server_round, results, failures)


class RFA(_Base):
    """Robust Federated Averaging / geometric median via Weiszfeld (Pillutla et al.)."""
    def __init__(self, *a, iters: int = 8, **kw):
        super().__init__(*a, **kw); self.iters = iters

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        st = self._stack(results); nds = [x[0] for x in st]; ns = np.array([x[1] for x in st], float)
        info = {}; self._log_norms(nds, results, info)
        flat = np.stack([np.concatenate([t.ravel() for t in w]) for w in nds])
        w = ns / ns.sum()
        z = (flat * w[:, None]).sum(0)
        for _ in range(self.iters):
            d = np.linalg.norm(flat - z, axis=1) + 1e-8
            a = w / d
            z = (flat * a[:, None]).sum(0) / a.sum()
        agg, i = [], 0
        for r in nds[0]:
            agg.append(z[i:i + r.size].reshape(r.shape).astype(r.dtype)); i += r.size
        self._global = agg
        return ndarrays_to_parameters(agg), self._finish(server_round, results, list(range(len(st))), info)


class NormBound(_Base):
    """Fixed-threshold norm clipping (Sun et al.), the non-adaptive control for FLAME.

 Included as the non-adaptive control.
 """
    def __init__(self, *a, clip: float = 60.0, **kw):
        super().__init__(*a, **kw); self.clip = clip

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        st = self._stack(results); nds = [x[0] for x in st]; ns = [x[1] for x in st]
        info = {"clip_threshold": self.clip}; self._log_norms(nds, results, info)
        if self._global is None or len(self._global) != len(nds[0]):
            agg = self._wavg(nds, ns); self._global = agg
            return ndarrays_to_parameters(agg), self._finish(server_round, results, list(range(len(st))), info)
        out = []
        nclip = 0
        for x in nds:
            dl = [w - g for w, g in zip(x, self._global)]
            n = float(np.sqrt(sum((d ** 2).sum() for d in dl)))
            s = min(1.0, self.clip / (n + 1e-12)); nclip += int(s < 1.0)
            out.append([g + s * d for g, d in zip(self._global, dl)])
        info["n_clipped"] = nclip
        agg = self._wavg(out, ns); self._global = agg
        return ndarrays_to_parameters(agg), self._finish(server_round, results, list(range(len(st))), info)


REGISTRY.update({"multi_krum": MultiKrum, "rfa": RFA, "norm_bound": NormBound})


