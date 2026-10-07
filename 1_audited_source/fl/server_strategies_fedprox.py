"""
FedProx server-side strategy wrapper.

FedProx (Li et al., MLSys 2020) is a *client-side* modification: each client adds
a proximal term (mu/2) * ||w - w_global||^2 to its local loss. The server-side
aggregation is identical to FedAvg.

This file therefore provides:
 * `FedProxStrategy` -- behaviour-equivalent to `fl.server_strategies.FedAvgStrategy`,
 but it sets `config['fedprox_mu']` so that a FedProx-aware client picks up the
 regularisation coefficient.

The client (`fl/client_yolo10s.py`) must read `config.get('fedprox_mu', 0.0)` from the
fit-config and add (mu/2) * sum_i ||w_i - w_g_i||^2 to its training loss for every
fit step. To minimise risk in a later pass we expose a *guarded* code path:
the prox term is only active when `mu > 0`; default is 0 so existing pipelines are
untouched.
"""

from __future__ import annotations
from typing import Dict, Optional

from fl.server_strategies import FedAvgStrategy


class FedProxStrategy(FedAvgStrategy):
    """FedAvg + per-round broadcast of the prox coefficient mu."""

    def __init__(self, *args, fedprox_mu: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.fedprox_mu = float(fedprox_mu)

    def configure_fit(self, server_round, parameters, client_manager):
        # Fall back to base behaviour, then patch each FitIns config dict
        cfg = super().configure_fit(server_round, parameters, client_manager)
        for _client, fit_ins in cfg:
            fit_ins.config["fedprox_mu"] = self.fedprox_mu
            fit_ins.config["fedprox_round"] = int(server_round)
        return cfg
