#!/usr/bin/env python3
"""
Replication of the fault-injection study: independent replicates (new data draw and new training
seeds), the client-local metric scored both on each client's training data (as in the paper's
Section 9) and on a held-out split of the client's own distribution, and T4 with both halves:
the seed must arrive at its point of use, and different seeds must give different trajectories.
Run on CPU, so two runs with the same seed are bit-identical and trajectory equality is exact:
    CUDA_VISIBLE_DEVICES= python3 replicate.py --replicates 0 1 2 3 4
"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from inject import Fault, run
from drive import CONDITIONS, checks

ap = argparse.ArgumentParser()
ap.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--client-val", type=int, default=256)
ap.add_argument("--out", default=os.path.join(HERE, "../../5_derived_results/c6_replication.json"))
a = ap.parse_args()

out = json.load(open(a.out)) if os.path.exists(a.out) else {}
for rep in a.replicates:
    D = 7 + 1000 * rep                      # replicate 0 uses the paper's data draw and seeds 0 and 1
    seeds = (2 * rep, 2 * rep + 1)
    res = {}
    for f in CONDITIONS:
        runs = [run(f, seed=s, rounds=a.rounds, data_seed=D, client_val=a.client_val) for s in seeds]
        undef = dict(head_shared=f.head_shared, buffers_shared=f.buffers_shared, screen="none",
                     seed_live=f.seed_live, metric_stage=f.metric_stage, attack_strength=f.attack_strength)
        t1a = run(Fault(f.name + "/t1-attack", f.kind, attack_applied=f.attack_applied, **undef),
                  seed=seeds[0], rounds=a.rounds, data_seed=D)
        t1b = run(Fault(f.name + "/t1-noattack", f.kind, attack_applied=False, **undef),
                  seed=seeds[0], rounds=a.rounds, data_seed=D)
        c = checks(f, runs)
        c["T1"] = bool(abs(np.mean([d["global"] for d in t1a[-3:]]) - np.mean([d["global"] for d in t1b[-3:]])) < 0.01)
        c["T4_assert"] = c.pop("T4")
        c["T4_trajectory"] = all(x["aggregated_state_hash"] == y["aggregated_state_hash"] for x, y in zip(*runs))
        c["T4"] = c["T4_assert"] or c["T4_trajectory"]
        last3 = lambda lg, k: float(np.mean([d[k] for d in lg[-3:]]))
        res[f.name] = {"kind": f.kind, "checks": c,
                       "global_last3": [last3(lg, "global") for lg in runs],
                       "local_train_last3": [last3(lg, "local") for lg in runs],
                       "local_val_last3": [last3(lg, "local_val") for lg in runs]}
        fired = [k for k, v in c.items() if v is True and k not in ("T5_input_varies", "T4_assert", "T4_trajectory")]
        print(f"rep {rep} {f.kind:7s} {f.name:28s} global {res[f.name]['global_last3'][0]:.4f} "
              f"train {res[f.name]['local_train_last3'][0]:.4f} val {res[f.name]['local_val_last3'][0]:.4f} "
              f"T4a {int(c['T4_assert'])} T4t {int(c['T4_trajectory'])} fired: {','.join(fired) or '-'}", flush=True)
    out[str(rep)] = {"data_seed": D, "seeds": list(seeds), "conditions": res}
    json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
