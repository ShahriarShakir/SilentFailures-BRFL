"""
YAML-driven runner for the supplementary experiments.

Usage:
 python experiments/run_supplementary_experiments.py --config supplementary_experiments/configs/robust_baselines.yaml
 python experiments/run_supplementary_experiments.py --config supplementary_experiments/configs/ablation_layers.yaml --dry-run

Conventions:
- Each config under `supplementary_experiments/configs/` declares an `experiment_id` and either
 a list of `strategies`, a list of `cells` (for ablation), or a `sweeps` block.
- The runner expands the cartesian product (strategies × seeds × cells) into a flat
 list of "runs"; each run writes a single JSON to
 `supplementary_experiments/results/<experiment_id>/<run_key>.json`.

The runner does NOT modify `experiments/run_yolo_fl.py`. Instead, for each run it
calls `flwr.simulation.start_simulation` directly with the resolved strategy.

This module is intentionally conservative in a later pass:
- `--dry-run` prints all planned runs without launching simulations.
- Failures in a single run are caught and recorded in `<run_key>.error.txt`; the
 runner moves on to the next run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ----- config helpers --------------------------------------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_extends(cfg: Dict[str, Any], cfg_dir: Path) -> Dict[str, Any]:
    if "extends" not in cfg:
        return cfg
    base = _load_yaml(cfg_dir / cfg["extends"])
    base = base.get("defaults", base)  # _defaults.yaml stores under `defaults:`
    merged = {**base, **cfg}
    merged.pop("extends", None)
    return merged


# ----- run-plan expansion ----------------------------------------------------
def expand_runs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a YAML config into a list of single-run dicts."""
    seeds = cfg.get("seeds", [1])
    eid = cfg["experiment_id"]
    runs: List[Dict[str, Any]] = []

    if "cells" in cfg:                          # layer ablation
        for cell in cfg["cells"]:
            for seed in seeds:
                runs.append({
                    "experiment_id": eid,
                    "run_key": f"{cell['id']}_s{seed}",
                    "strategy": "photoscreen",
                    "strategy_kwargs": {
                        "enable_layer1": cell["enable_layer1"],
                        "enable_layer2": cell["enable_layer2"],
                        "enable_layer3": cell["enable_layer3"],
                    },
                    "attack": cfg.get("attack"),
                    "seed": seed,
                    "num_rounds": cfg.get("num_rounds", 100),
                })
        # FalseRej. companion runs (no-attack)
        for cell_id in cfg.get("falserej_cells", []):
            cell = next(c for c in cfg["cells"] if c["id"] == cell_id)
            for seed in seeds:
                runs.append({
                    "experiment_id": eid,
                    "run_key": f"{cell_id}_falserej_s{seed}",
                    "strategy": "photoscreen",
                    "strategy_kwargs": {
                        "enable_layer1": cell["enable_layer1"],
                        "enable_layer2": cell["enable_layer2"],
                        "enable_layer3": cell["enable_layer3"],
                    },
                    "attack": {"name": "none"},
                    "seed": seed,
                    "num_rounds": cfg.get("num_rounds", 100),
                })

    elif "strategies" in cfg:                   # baseline list
        for strat in cfg["strategies"]:
            for seed in seeds:
                runs.append({
                    "experiment_id": eid,
                    "run_key": f"{strat['name']}_s{seed}",
                    "strategy": strat["name"],
                    "strategy_kwargs": {k: v for k, v in strat.items() if k != "name"},
                    "attack": cfg.get("attack"),
                    "seed": seed,
                    "num_rounds": cfg.get("num_rounds", 100),
                })

    elif "defenses" in cfg:                     # adaptive attack matrix
        for d in cfg["defenses"]:
            for seed in seeds:
                runs.append({
                    "experiment_id": eid,
                    "run_key": f"{d['label']}_s{seed}".replace(" ", "_"),
                    "strategy": d["name"],
                    "strategy_kwargs": {k: v for k, v in d.items() if k not in ("name", "label")},
                    "attack": cfg.get("attack"),
                    "seed": seed,
                    "num_rounds": cfg.get("num_rounds", 100),
                })

    elif "sweeps" in cfg:                       # threshold sensitivity
        pin = cfg.get("defaults_pin", {})
        if cfg.get("mode", "one_at_a_time") == "one_at_a_time":
            for param, values in cfg["sweeps"].items():
                for v in values:
                    for seed in seeds:
                        kwargs = {**pin, param: v, "enable_layer1": True, "enable_layer2": True, "enable_layer3": True}
                        runs.append({
                            "experiment_id": eid,
                            "run_key": f"{param}_{v}_s{seed}",
                            "strategy": "photoscreen",
                            "strategy_kwargs": kwargs,
                            "attack": cfg.get("attack"),
                            "seed": seed,
                            "num_rounds": cfg.get("num_rounds", 100),
                        })
        else:
            keys = list(cfg["sweeps"].keys())
            for combo in product(*(cfg["sweeps"][k] for k in keys)):
                for seed in seeds:
                    kwargs = {**pin, **dict(zip(keys, combo)),
                              "enable_layer1": True, "enable_layer2": True, "enable_layer3": True}
                    rk = "_".join(f"{k}{v}" for k, v in zip(keys, combo)) + f"_s{seed}"
                    runs.append({
                        "experiment_id": eid, "run_key": rk,
                        "strategy": "photoscreen", "strategy_kwargs": kwargs,
                        "attack": cfg.get("attack"), "seed": seed,
                        "num_rounds": cfg.get("num_rounds", 100),
                    })

    elif "ks" in cfg:                           # client-scaling
        for K in cfg["ks"]:
            for strat in cfg["strategies"]:
                for seed in seeds:
                    runs.append({
                        "experiment_id": eid,
                        "run_key": f"K{K}_{strat['name']}_s{seed}",
                        "strategy": strat["name"],
                        "strategy_kwargs": {k: v for k, v in strat.items() if k != "name"},
                        "attack": cfg.get("attack"),
                        "seed": seed,
                        "num_rounds": cfg.get("num_rounds", 100),
                        "K": K,
                        "data_prep_required": True,
                    })

    else:                                       # single-strategy run (e.g., bias)
        strat = cfg.get("strategy", "photoscreen")
        for seed in seeds:
            runs.append({
                "experiment_id": eid,
                "run_key": f"{strat}_s{seed}",
                "strategy": strat,
                "strategy_kwargs": {k: cfg[k] for k in ("enable_layer1", "enable_layer2", "enable_layer3") if k in cfg},
                "attack": cfg.get("attack"),
                "seed": seed,
                "num_rounds": cfg.get("num_rounds", 100),
            })

    return runs


# ----- strategy factory ------------------------------------------------------
def build_strategy(strategy: str, kwargs: Dict[str, Any], cfg: Dict[str, Any]):
    """Instantiate the right strategy. Imports are local so --dry-run works without GPU env."""
    n = cfg.get("num_clients", 7)
    base = dict(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=n, min_evaluate_clients=n, min_available_clients=n,
    )
    if strategy == "fedavg":
        from fl.server_strategies import FedAvgStrategy
        return FedAvgStrategy(**base)
    if strategy == "fedbn":
        from fl.server_strategies import FedBNStrategy
        return FedBNStrategy(**base)
    if strategy == "fedprox":
        from fl.server_strategies_fedprox import FedProxStrategy
        return FedProxStrategy(fedprox_mu=kwargs.get("fedprox_mu", 0.01), **base)
    if strategy == "photoscreen":
        from fl.server_strategies import PHOTOSCREENStrategy
        return PHOTOSCREENStrategy(
            iara_config_path=cfg.get("iara_config_path", "data/bn_statistics/iara_config.json"),
            enable_layer1=kwargs.get("enable_layer1", True),
            enable_layer2=kwargs.get("enable_layer2", True),
            enable_layer3=kwargs.get("enable_layer3", True),
            mad_threshold=kwargs.get("mad_threshold", 3.0),
            cosine_threshold=kwargs.get("cosine_threshold", 0.5),
            **base,
        )
    if strategy == "median":
        from fl.server_strategies_robust import MedianStrategy
        return MedianStrategy(**base)
    if strategy == "trimmed_mean":
        from fl.server_strategies_robust import TrimmedMeanStrategy
        return TrimmedMeanStrategy(beta_trim=kwargs.get("beta_trim", 0.2), **base)
    if strategy == "krum":
        from fl.server_strategies_robust import KrumStrategy
        return KrumStrategy(num_byzantine=kwargs.get("num_byzantine", 2), m=1, **base)
    if strategy == "multi_krum":
        from fl.server_strategies_robust import KrumStrategy
        return KrumStrategy(num_byzantine=kwargs.get("num_byzantine", 2),
                            m=kwargs.get("m", 3), **base)
    if strategy == "geometric_median":
        from fl.server_strategies_robust import GeometricMedianStrategy
        return GeometricMedianStrategy(max_iter=kwargs.get("max_iter", 30), **base)
    if strategy == "scaffold":
        from fl.server_strategies_robust import SCAFFOLDStrategy
        return SCAFFOLDStrategy(
            server_lr=kwargs.get("server_lr", 1.0),
            ema_alpha=kwargs.get("ema_alpha", 0.1),
            **base,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


# ----- run a single run ------------------------------------------------------
def run_one(run: Dict[str, Any], cfg: Dict[str, Any], dry_run: bool = False):
    eid = run["experiment_id"]
    rk = run["run_key"]
    out_dir = ROOT / cfg.get("results_root", "supplementary_experiments/results") / eid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rk}.json"

    if out_path.exists() and not cfg.get("force", False):
        print(f"  [skip] {eid}/{rk} (already done)")
        return

    print(f"  [run ] {eid}/{rk}")
    if dry_run:
        return

    if run.get("data_prep_required") and not (ROOT / "data/yolo/client_configs_K{}".format(run.get("K", 7))).exists():
        msg = f"Skipping K={run.get('K')} run: data shards not generated. Generate the data shards first."
        print("  [warn]", msg)
        (out_dir / f"{rk}.error.txt").write_text(msg + "\n")
        return

    try:
        # Local imports to avoid Flower side-effects during dry runs
        import flwr as fl
        from fl.client_yolo10s import create_yolo_client_fn
        from experiments.run_yolo_fl import load_client_configs

        np_seed = run["seed"]
        import numpy as np
        np.random.seed(np_seed)
        try:
            import torch
            torch.manual_seed(np_seed)
        except Exception:
            pass

        strat = build_strategy(run["strategy"], run.get("strategy_kwargs", {}), cfg)
        cities_override = cfg.get("cities") or cfg.get("defaults", {}).get("cities")
        client_configs = load_client_configs(ROOT, cities=cities_override)

        # ---- attack injection ----
        attack_obj = None
        malicious_clients: list = []
        attack_cfg = run.get("attack") or {}
        attack_name = attack_cfg.get("name") if isinstance(attack_cfg, dict) else None
        if attack_name and attack_name != "none":
            from experiments.attacks.byzantine_attacks import create_attack
            attack_kwargs = {k: v for k, v in attack_cfg.items() if k != "name"}
            attack_obj = create_attack(attack_name, **attack_kwargs)
            ratio = attack_kwargs.get("attack_ratio", 0.0)
            num_mal = int(round(len(client_configs) * ratio))
            malicious_clients = list(range(num_mal))
            print(f"  [atk ] {attack_name} ratio={ratio} -> malicious={malicious_clients}")

        client_fn = create_yolo_client_fn(
            client_configs=client_configs,
            fedbn=run["strategy"] in ("fedbn", "photoscreen", "fedprox"),
            extract_bn_stats=run["strategy"] == "photoscreen",
            attack=attack_obj,
            malicious_clients=malicious_clients,
            seed=int(np_seed),
        )

        t0 = time.time()
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(client_configs),
            config=fl.server.ServerConfig(num_rounds=run["num_rounds"]),
            strategy=strat,
            ray_init_args=cfg.get("ray_init_args", {}),
            client_resources=cfg.get("client_resources", {}),
        )
        wall = time.time() - t0

        record = {
            "experiment_id": eid,
            "run_key": rk,
            "strategy": run["strategy"],
            "strategy_kwargs": run.get("strategy_kwargs", {}),
            "attack": run.get("attack"),
            "seed": run["seed"],
            "num_rounds": run["num_rounds"],
            "wall_seconds": wall,
            "metrics_distributed_fit": getattr(history, "metrics_distributed_fit", {}),
            "metrics_distributed": getattr(history, "metrics_distributed", {}),
        }
        with open(out_path, "w") as f:
            json.dump(record, f, default=str, indent=2)

    except Exception as e:
        err_path = out_dir / f"{rk}.error.txt"
        err_path.write_text("".join(traceback.format_exception(e)))
        print(f"  [FAIL] {eid}/{rk}  -> {err_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if result JSON already exists.")
    args = p.parse_args()

    cfg_dir = args.config.parent
    cfg = _load_yaml(args.config)
    cfg = _resolve_extends(cfg, cfg_dir)
    cfg["force"] = args.force

    runs = expand_runs(cfg)
    print(f"[plan] experiment {cfg['experiment_id']}: {len(runs)} runs")
    for r in runs:
        run_one(r, cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
