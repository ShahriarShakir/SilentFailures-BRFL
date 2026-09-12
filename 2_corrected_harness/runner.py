#!/usr/bin/env python3
"""
Federated training runner with server-side held-out evaluation.

Data-poisoning attacks are materialised into the malicious client's dataset before training, so the
detector's own data loader sees corrupted images. Every client is constructed from one shared
checkpoint at the task's class count, and the full model state (parameters and normalisation
buffers) is exchanged. The aggregated global model is evaluated on a held-out split after every
round; the client-local metric is recorded alongside it. Seeds are propagated to every client and
recorded. The `--exchange` and `--head` flags reproduce the original protocol for the controlled
comparisons in the paper.
"""
from __future__ import annotations
import sys, json, time, argparse, os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

ROOT = Path("/ANON/fl_hod")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "project/harness"))

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from ultralytics import YOLO

import strategies as S
from data_poison import materialise, materialise_backdoor, PIXEL_ATTACKS, LABEL_ATTACKS
from model_fix import build_nc_model

CITIES7 = ["budapest", "koeln", "leipzig", "lyon", "prague", "roma", "zagreb"]
DATASETS = {
    "ecp":       (CITIES7, str(ROOT / "data/yolo/client_configs/ecp_global.yaml")),
    "nightowls": ([f"no{i}" for i in range(7)], str(ROOT / "data/yolo/client_configs/nightowls_global.yaml")),
    "ecp14":     ([f"{c}_{h}" for c in CITIES7 for h in "ab"], str(ROOT / "data/yolo/client_configs/ecp_global.yaml")),
}
GLOBAL_YAML = DATASETS["ecp"][1]
NC1_CKPT = str(ROOT / "project/work/yolov10s_nc1.pt")   # one architecture for everyone
POISON_DIR = ROOT / "project/work/poisoned"


from contracts import state_hash as _state_hash, buffer_policy as _buffer_policy


class GlobalEvaluator:
    """Server-side evaluation of the AGGREGATED model on held-out ECP-night val."""

    def __init__(self, model_path: str, data_yaml: str, imgsz: int = 640, device: str = "0", nc: int = 1):
        # A fresh model is built for every evaluation: Ultralytics fuses conv+BN during.val,
        # which changes the state_dict, so a reused model cannot accept round t+1's parameters.
        self.model_path = model_path
        self.data_yaml, self.imgsz, self.device = data_yaml, imgsz, device
        self.history: List[Dict] = []

    def _fresh(self, nds: List[np.ndarray]):
        """load the full state (params + buffers), so the evaluated model is coherent."""
        m = YOLO(self.model_path)
        sd = m.model.state_dict()
        if getattr(self, "tolerant", False):
            sdk = list(sd.keys()); new = {}
            for k, arr in zip(sdk, nds):
                t = sd[k]; a = np.asarray(arr)
                new[k] = t if int(a.size) != int(np.prod(t.shape)) else \
                         torch.from_numpy(a).reshape(t.shape).to(dtype=t.dtype)
            m.model.load_state_dict(new); return m
        if getattr(self, "params_only", False):
            # reproduce the original protocol: only named_parameters travel, BN buffers stay
            # at the checkpoint's values -> the incoherent "global model" of 
            pd, i = {}, 0
            for name, p in m.model.named_parameters():
                if i < len(nds):
                    pd[name] = torch.from_numpy(np.asarray(nds[i])).reshape(p.shape).to(p.dtype)
                i += 1
            sd.update(pd); m.model.load_state_dict(sd); return m
        keys = list(sd.keys())
        if len(nds) != len(keys):
            raise RuntimeError(f"global eval: got {len(nds)} tensors, expected {len(keys)} "
                               f"- full-state protocol violated ")
        new = {}
        for k, arr in zip(keys, nds):
            t = sd[k]
            new[k] = torch.from_numpy(np.asarray(arr)).reshape(t.shape).to(dtype=t.dtype)
        m.model.load_state_dict(new)
        return m

    def __call__(self, server_round, parameters, config):
        if server_round == 0:
            return None
        # contract 2: fingerprint the aggregated tensor set the server is about to evaluate.
        # Identical hashes across runs that are supposed to differ mean the runs are not distinct.
        _agg_hash = _state_hash({str(i): a for i, a in enumerate(parameters)})
        try:
            m = self._fresh(parameters)
            if not getattr(self, "_policy_logged", False):
                pol = _buffer_policy(m.model.state_dict())
                self.buffer_policy = {k: len(v) for k, v in pol.items()}
                self.integer_counter_keys = pol["integer_counters"][:3]
                self._policy_logged = True
                print(f"  [contract] buffer policy {self.buffer_policy}", flush=True)
            if getattr(self, "save_dir", None):
                import os as _os
                _os.makedirs(self.save_dir, exist_ok=True)
                m.save(f"{self.save_dir}/round{server_round:02d}.pt")
            r = m.val(data=self.data_yaml, imgsz=self.imgsz, device=self.device,
                               verbose=False, plots=False, save_json=False,
                               project=f"/tmp/fl_globaleval_{os.getpid()}")
            d = r.results_dict
            asr = {}
            if getattr(self, "asr_yaml", None):
                try:
                    rt = m.val(data=self.asr_yaml, imgsz=self.imgsz, device=self.device,
                               verbose=False, plots=False, save_json=False,
                               project=f"/tmp/fl_asr_{os.getpid()}")
                    dt = rt.results_dict
                    asr = {"trig_recall": float(dt.get("metrics/recall(B)", 0.0)),
                           "trig_map50": float(dt.get("metrics/mAP50(B)", 0.0))}
                except Exception as e:
                    asr = {"trig_error": str(e)[:60]}
            m = {**asr, "global_map50": float(d.get("metrics/mAP50(B)", 0.0)),
                 "global_map50_95": float(d.get("metrics/mAP50-95(B)", 0.0)),
                 "global_precision": float(d.get("metrics/precision(B)", 0.0)),
                 "global_recall": float(d.get("metrics/recall(B)", 0.0))}
        except Exception as e:
            print(f"  [globaleval] FAILED round {server_round}: {e}", flush=True)
            m = {"global_map50": 0.0, "global_map50_95": 0.0,
                 "global_precision": 0.0, "global_recall": 0.0, "error": str(e)}
        # Record the order of steps in the round.
        m["agg_state_hash"] = _agg_hash
        m["n_tensors_received"] = len(parameters)
        self.history.append({"round": server_round, **m})
        self.timeline = getattr(self, "timeline", [])
        self.timeline.append({"round": server_round,
                              "step": "server_evaluates_aggregate_on_heldout",
                              "after": "aggregation", "before": "next_round_client_fit",
                              "state_hash": _agg_hash})
        print(f"  [globaleval] round {server_round}: global mAP50 = {m['global_map50']:.4f}", flush=True)
        return 0.0, m


def build_clients(cities, attack_name, alpha, intensity, seed, local_epochs, batch, imgsz,
                  model_path=None):
    """Client configs; malicious clients get a materialised poisoned dataset."""
    K = len(cities)
    n_mal = int(round(K * alpha))
    malicious = list(range(n_mal))
    cfgs = {}
    for i, city in enumerate(cities):
        y = str(ROOT / f"data/yolo/client_configs/client_{city}.yaml")
        if i in malicious and attack_name == "backdoor_trigger":
            t0 = time.time()
            y = materialise_backdoor(y, POISON_DIR, seed=seed, split="train", drop_labels=True)
            print(f"  [poison] client {i} ({city}) backdoor train ({time.time()-t0:.0f}s)", flush=True)
        elif i in malicious and attack_name in set(PIXEL_ATTACKS) | LABEL_ATTACKS:
            t0 = time.time()
            y = materialise(y, attack_name, intensity, POISON_DIR, seed=seed)
            print(f"  [poison] client {i} ({city}) -> {Path(y).parent.name} ({time.time()-t0:.0f}s)", flush=True)
        cfgs[i] = {"data_yaml": y, "city": city, "model_path": model_path or NC1_CKPT,
                   "local_epochs": local_epochs, "batch_size": batch, "imgsz": imgsz, "device": "0"}
    return cfgs, malicious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="fedavg", choices=list(S.REGISTRY))
    ap.add_argument("--dataset", default="ecp", choices=list(DATASETS))
    ap.add_argument("--attack-kwargs", default="{}", help="JSON kwargs for EXTRA attacks (e.g. jitter)")
    ap.add_argument("--attack", default="none")
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--strategy-kwargs", default="{}")
    ap.add_argument("--gpu-frac", type=float, default=0.14)
    ap.add_argument("--dos-only", type=int, default=1)
    ap.add_argument("--save-ckpt", default="", help="directory to write the aggregated model each round")
    ap.add_argument("--asr-yaml", default="", help="triggered evaluation set; if given, recall on it "
                                                  "is recorded each round as the backdoor objective")
    ap.add_argument("--head", default="matched", choices=["matched","mismatched"],
                    help="matched = every client built at the task's class count. "
                         "mismatched = built from the nc=80 checkpoint, reproducing the head fault.")
    ap.add_argument("--exchange", default="full", choices=["full","params"],
                    help="full = state_dict incl. BN buffers. params = named_parameters only, "
                         "reproducing the ORIGINAL broken protocol for the T3 experiment.")
    a = ap.parse_args()

    np.random.seed(a.seed); torch.manual_seed(a.seed)
    from client_fix import create_fullstate_client_fn, create_fixed_client_fn, create_tolerant_client_fn
    from experiments.attacks.byzantine_attacks import create_attack
    from attacks_extra import EXTRA, CentreSpoofAttack
    from attacks_published import PUBLISHED
    import json as _json
    _centre = _json.load(open(ROOT / "project/work/results/federation_centre.json"))

    _ckpt = NC1_CKPT if a.head == "matched" else str(ROOT / "yolov10s.pt")
    cities, gyaml = DATASETS[a.dataset]
    cfgs, malicious = build_clients(cities, a.attack, a.alpha, a.intensity, a.seed,
                                    a.local_epochs, a.batch, a.imgsz, model_path=_ckpt)
    # model-space attacks still go through the parameter hook
    atk_obj = None
    if a.attack in PUBLISHED:
        _nm = int(round(len(cfgs) * a.alpha))
        atk_obj = PUBLISHED[a.attack](attack_ratio=a.alpha, intensity=a.intensity, n_malicious=_nm)
        print(f"  [atk ] published attack {a.attack}, {_nm} colluders", flush=True)
    elif a.attack == "norm_centre_spoof":
        from attacks_extra import NormCentreSpoofAttack
        _q6 = _json.load(open(ROOT / "project/work/results/q6_norm_layer.json"))
        atk_obj = NormCentreSpoofAttack(_q6["median_norm"], attack_ratio=a.alpha,
                                        intensity=a.intensity, dos_only=(a.dos_only == 1))
        print(f"  [atk ] norm_centre_spoof target_norm={_q6['median_norm']:.4f}", flush=True)
    elif a.attack == "centre_spoof":
        atk_obj = CentreSpoofAttack(_centre["median_mu"], _centre["median_sigma"],
                                    attack_ratio=a.alpha, intensity=a.intensity,
                                    dos_only=(a.dos_only == 1))
    elif a.attack in EXTRA:
        atk_obj = EXTRA[a.attack](attack_ratio=a.alpha, intensity=a.intensity,
                                  n_malicious=int(round(len(cities) * a.alpha)),
                                  **_json.loads(a.attack_kwargs))
    elif a.attack not in ("none", "backdoor_trigger") and a.attack not in set(PIXEL_ATTACKS) | LABEL_ATTACKS:
        atk_obj = create_attack(a.attack, attack_ratio=a.alpha, intensity=a.intensity)
    print(f"  [setup] strategy={a.strategy} attack={a.attack} alpha={a.alpha} "
          f"malicious={malicious} seed={a.seed} rounds={a.rounds}", flush=True)

    if a.exchange == "params":
        _mk = create_fixed_client_fn
    elif a.head == "mismatched":
        _mk = create_tolerant_client_fn      # full state, tolerant skip: isolates the head fault
    else:
        _mk = create_fullstate_client_fn
    _nc = 1 if a.head == "matched" else 80
    client_fn = _mk(
        client_configs=cfgs, extract_bn_stats=True,
        attack=atk_obj, malicious_clients=malicious if atk_obj else [], seed=a.seed, nc=_nc)

    ev = GlobalEvaluator(_ckpt, gyaml, a.imgsz)
    ev.asr_yaml = a.asr_yaml or None
    ev.save_dir = a.save_ckpt or None
    ev.params_only = (a.exchange == "params")
    ev.tolerant = (a.exchange == "full" and a.head == "mismatched")
    skw = json.loads(a.strategy_kwargs)
    strat = S.REGISTRY[a.strategy](
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=len(cfgs), min_available_clients=len(cfgs),
        evaluate_fn=ev, **skw)

    t0 = time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=len(cfgs),
        config=fl.server.ServerConfig(num_rounds=a.rounds), strategy=strat,
        ray_init_args={"num_cpus": 14, "num_gpus": 1, "include_dashboard": False},
        client_resources={"num_cpus": 2, "num_gpus": a.gpu_frac})
    wall = time.time() - t0

    out = ROOT / f"project/work/results/runs/{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": vars(a), "malicious_clients": malicious, "wall_seconds": wall,
        "global_eval": ev.history,        # : the aggregated model, held-out
        "round_log": strat.round_log,     # legacy local metric + accept/reject bookkeeping
        # contracts: ordered step record and the declared buffer partition
        "timeline": getattr(ev, "timeline", []),
        "buffer_policy": getattr(ev, "buffer_policy", {}),
    }, indent=2))
    print(f"\nwrote {out}  ({wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
