#!/usr/bin/env python3
"""
Published Byzantine attacks, implemented to their original specifications.

All operate on the malicious clients' submitted parameter vectors and need only what a colluding
group can observe: their own updates and the previous global model.

 ALIE Baruch et al., NeurIPS 2019: perturb the benign mean by z*sigma, sized so the
 perturbation stays inside the range a robust rule will accept.
 IPM Xie et al., UAI 2019: submit -epsilon * mean(benign), making the inner product
 between the aggregate and the true gradient negative.
 MinMax Shejwalkar & Houmansadr, NDSS 2021: maximise the perturbation subject to the malicious
 update's maximum distance to any benign update not exceeding the benign set's own
 maximum pairwise distance.
 MinSum same paper: the same search under the sum-of-squared-distances constraint.

Colluders share information, so each is given the set of malicious updates for the round. Benign
statistics are estimated from the colluders' own honest-training updates, the standard partial
knowledge assumption, which is strictly weaker than full knowledge.
"""
from typing import List
import numpy as np


def _flat(ds):
    return np.concatenate([d.ravel() for d in ds])


def _unflat(v, ref):
    out, i = [], 0
    for r in ref:
        n = r.size
        out.append(v[i:i + n].reshape(r.shape)); i += n
    return out


class _Collusive:
    """Base: buffers the colluders' honest deltas for the round, then crafts one shared update."""
    name = "collusive"

    def __init__(self, attack_ratio=0.30, intensity=1.0, n_malicious=2, **kw):
        self.attack_ratio = attack_ratio; self.intensity = intensity
        self.n_malicious = max(int(n_malicious), 1)
        self._buf: List[np.ndarray] = []
        self._ref = None
        self.spoofed_metrics = {}

    def poison_data(self, images, labels):
        return images, labels

    def _craft(self, M):
        raise NotImplementedError

    def poison_parameters(self, parameters, global_params=None):
        if global_params is None or len(global_params) == 0:
            return parameters
        delta = [p - g for p, g in zip(parameters, global_params)]
        self._ref = delta
        v = _flat(delta)
        self._buf.append(v)
        if len(self._buf) > self.n_malicious:
            self._buf = self._buf[-self.n_malicious:]
        M = np.stack(self._buf) if len(self._buf) > 1 else v[None, :]
        crafted = self._craft(M)
        return [g + d for g, d in zip(global_params, _unflat(crafted, delta))]


class ALIE(_Collusive):
    """A Little Is Enough (Baruch et al., NeurIPS 2019)."""
    name = "alie"

    def _craft(self, M):
        mu = M.mean(0); sd = M.std(0) if M.shape[0] > 1 else np.abs(mu) * 0.1
        z = 1.5 * self.intensity            # standard operating range in the original paper
        return mu - z * sd


class IPM(_Collusive):
    """Inner Product Manipulation (Xie et al., UAI 2019): submit -epsilon * mean(benign)."""
    name = "ipm"

    def _craft(self, M):
        return -float(self.intensity) * M.mean(0)


class MinMax(_Collusive):
    """Shejwalkar & Houmansadr, NDSS 2021, max-distance constrained perturbation."""
    name = "min_max"
    mode = "max"

    def _craft(self, M):
        mu = M.mean(0)
        p = -(M.std(0) if M.shape[0] > 1 else np.abs(mu) + 1e-12)   # std perturbation direction
        p = p / (np.linalg.norm(p) + 1e-12)
        if M.shape[0] > 1:
            D = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=-1)
            bound = D.max() if self.mode == "max" else (D ** 2).sum(1).max()
        else:
            bound = np.linalg.norm(mu)
        lo, hi = 0.0, 10.0 * (np.linalg.norm(mu) + 1e-12)
        for _ in range(20):                                          # bisection on gamma
            g = 0.5 * (lo + hi)
            cand = mu + g * p
            d = np.linalg.norm(M - cand, axis=-1)
            val = d.max() if self.mode == "max" else (d ** 2).sum()
            if val <= bound:
                lo = g
            else:
                hi = g
        return mu + lo * p


class MinSum(MinMax):
    """Same search under the sum-of-squared-distances constraint."""
    name = "min_sum"
    mode = "sum"


PUBLISHED = {"alie": ALIE, "ipm": IPM, "min_max": MinMax, "min_sum": MinSum}
