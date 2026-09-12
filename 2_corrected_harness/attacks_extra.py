#!/usr/bin/env python3
"""
Additional attacks for the corrected harness.

`CentreSpoofAttack`: the attacker reports the federation's robust centre as its own photometric
statistic. A screen that estimates its scale from the values it screens then shrinks that scale with
every centre report, and honest clients' standardised deviations inflate until the screen rejects them.
`dos_only=True` leaves the parameters untouched, so any damage is caused by the screen rejecting
honest clients; `dos_only=False` also applies a parameter-space attack.

`NormCentreSpoofAttack`: the same construction on update norms, the quantity that clipping rules and
norm fences calibrate on.
"""
from typing import List
import numpy as np


class CentreSpoofAttack:
    name = "centre_spoof"

    def __init__(self, centre_mu: float, centre_sigma: float, attack_ratio: float = 0.30,
                 intensity: float = 1.0, dos_only: bool = True):
        self.attack_ratio = attack_ratio
        self.intensity = intensity
        self.dos_only = dos_only
        self.spoofed_metrics = {"bn_mu": float(centre_mu), "bn_sigma": float(centre_sigma)}

    def poison_data(self, images, labels):
        return images, labels

    def poison_parameters(self, parameters: List[np.ndarray], global_params=None):
        if self.dos_only or global_params is None or len(global_params) == 0:
            return parameters
        out = []
        for p, g in zip(parameters, global_params):
            d = p - g
            out.append(np.clip(g - min(self.intensity, 1.0) * d, -10.0, 10.0))
        return out


EXTRA = {"centre_spoof": CentreSpoofAttack}


class NormCentreSpoofAttack:
    """the scale-collapse attack moved into update-norm space.

 This is the layer real defenses calibrate on: FLAME clips at the median of client L2 norms,
 RAB2-DEF fences at +1.5*IQR of client scores, and PhotoScreen's own Layer 2 is a MAD filter on
 norms. The attacker rescales its update to sit exactly at the median honest norm.

 With dos_only=True the update DIRECTION is left untouched, the attacker submits an honest
 update at a chosen magnitude, so any damage measured is caused solely by the defense rejecting
 honest clients, not by model poisoning.
 """
    name = "norm_centre_spoof"

    def __init__(self, target_norm: float, attack_ratio: float = 0.30,
                 intensity: float = 1.0, dos_only: bool = True):
        self.target_norm = float(target_norm)
        self.attack_ratio = attack_ratio
        self.intensity = intensity
        self.dos_only = dos_only
        self.spoofed_metrics = {}

    def poison_data(self, images, labels):
        return images, labels

    def poison_parameters(self, parameters, global_params=None):
        if global_params is None or len(global_params) == 0:
            return parameters
        delta = [p - g for p, g in zip(parameters, global_params)]
        n = float(np.sqrt(sum((d ** 2).sum() for d in delta)))
        if n <= 0:
            return parameters
        s = self.target_norm / n
        if not self.dos_only:
            s = -s
        return [g + s * d for g, d in zip(global_params, delta)]


EXTRA["norm_centre_spoof"] = NormCentreSpoofAttack

