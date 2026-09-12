#!/usr/bin/env python3
"""
Fault injection into a known-good harness, to measure what the five checks catch.

A small federated image classifier (CNN with BatchNorm, synthetic data so it runs anywhere) into which
faults are injected one at a time. Every condition is run twice, faulted and healthy. The checks are
applied blind and detections and false alarms are counted. Healthy controls include a deliberate
personalised-normalisation configuration and a deliberately weak attack, so that intent-driven
choices are not scored as faults.
"""
from __future__ import annotations
import argparse, json, math, hashlib
from dataclasses import dataclass, field
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1); self.b1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1); self.b2 = nn.BatchNorm2d(32)
        self.head = nn.Linear(32, nc)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)
        return self.head(x.mean(dim=(2, 3)))


def make_clients(K, n, nc, seed, shift=True):
    """Non-IID by a per-client photometric shift, so BN statistics differ."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for k in range(K):
        y = torch.randint(0, nc, (n,), generator=g)
        base = torch.randn(n, 3, 16, 16, generator=g) * 0.5
        sig = torch.zeros(nc, 3, 16, 16)
        gg = torch.Generator().manual_seed(1234)
        sig = torch.randn(nc, 3, 16, 16, generator=gg)
        x = (base + sig[y]) * 6.0 + 4.0     # far from BN init stats (0,1), so buffers matter
        if shift:
            x = x * (0.85 + 0.06 * k) + (0.10 * k - 0.25)    # mild client-specific gain and offset
        out.append((x, y))
    return out


@dataclass
class Fault:
    name: str
    kind: str          # 'fault' or 'healthy'
    attack_applied: bool = True
    head_shared: bool = True
    buffers_shared: bool = True
    screen: str = "none"        # none | constant | inert | reversed | wrongref
    seed_live: bool = True
    metric_stage: str = "correct"   # correct | pre_adaptation_offset
    attack_strength: float = 1.0


def local_train(model, x, y, epochs, lr, seed):
    torch.manual_seed(seed)
    m = Net().to(DEV); m.load_state_dict(model)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 32):
            idx = perm[i:i+32]
            opt.zero_grad()
            loss = F.cross_entropy(m(x[idx].to(DEV)), y[idx].to(DEV))
            loss.backward(); opt.step()
    return m


@torch.no_grad()
def acc(state, x, y):
    m = Net().to(DEV); m.load_state_dict(state); m.eval()
    return float((m(x.to(DEV)).argmax(1).cpu() == y).float().mean())


def run(f: Fault, K=6, n=256, rounds=6, epochs=2, lr=3e-3, seed=0, nc=10, data_seed=7):
    # data partition is fixed across seeds; only the training seed varies, so 'seed ignored'
    # is isolated from 'different data'
    clients = make_clients(K, n, nc, data_seed)
    held = make_clients(1, 512, nc, data_seed + 999)[0]
    g = Net().to(DEV).state_dict()
    n_mal = 2
    log = []
    prev_scale = None
    for r in range(rounds):
        subs, reported, screened_out = [], [], []
        state_hashes, seeds_used = [], []
        for k, (x, y) in enumerate(clients):
            xk = x.clone()
            yk = y.clone()
            if k < n_mal and f.attack_applied:                     # data poisoning
                nflip = int(len(yk) * f.attack_strength)
                if nflip:
                    yk[:nflip] = (yk[:nflip] + 1) % nc            # label flip: damaging
                xk = xk + 3.0 * f.attack_strength
            eff = (seed * 1000 + k) if f.seed_live else 0
            m = local_train(g, xk, yk, epochs, lr, eff)
            sd = {kk: v.detach().cpu().clone() for kk, v in m.state_dict().items()}
            subs.append(sd)
            reported.append(acc(sd, x, y))                          # client-local metric
            state_hashes.append(hashlib.sha256(
                b"".join(v.numpy().tobytes() for v in sd.values())).hexdigest()[:16])
            seeds_used.append(eff)
            # screened statistic: the client's own reported photometric mean
            mu = float(xk.mean()) if f.screen != "constant" else 0.29
            screened_out.append(mu)
        # --- screen
        keep = list(range(K))
        if f.screen in ("inert", "reversed", "wrongref", "constant"):
            v = np.array(screened_out)
            med = np.median(v); mad = max(float(np.median(np.abs(v - med))) * 1.4826, 1e-9)
            z = np.abs(v - med) / mad
            if f.screen == "inert":     keep = list(range(K))                  # threshold unreachable
            elif f.screen == "reversed":keep = [i for i in range(K) if z[i] >= 3.0] or list(range(K))
            elif f.screen == "wrongref":keep = [i for i in range(K) if abs(v[i] - 99.0) / mad <= 3.0] or list(range(K))
            else:                        keep = [i for i in range(K) if z[i] <= 3.0]
        # --- aggregate
        new = {}
        for kk in g:
            if (not f.head_shared) and kk.startswith("head"):
                new[kk] = g[kk]; continue
            if (not f.buffers_shared) and ("running_" in kk or "num_batches" in kk):
                new[kk] = g[kk]; continue
            st = torch.stack([subs[i][kk].float() for i in keep]).mean(0)
            new[kk] = st.to(g[kk].dtype)
        g = {kk: v.to(DEV) for kk, v in new.items()}
        gl = acc(g, *held)
        lo = float(np.mean(reported))
        if f.metric_stage == "pre_adaptation_offset":
            lo = lo + 0.05
        agg_hash = hashlib.sha256(
            b"".join(v.detach().cpu().numpy().tobytes() for v in g.values())).hexdigest()[:16]
        log.append({"round": r + 1, "global": gl, "local": lo,
                    "n_kept": len(keep), "screen_scores": [float(s) for s in screened_out],
                    "metric_state_hashes": state_hashes,   # states the reported metric came from
                    "aggregated_state_hash": agg_hash,     # state the defense is meant to protect
                    "seeds_used": seeds_used})
    return log
