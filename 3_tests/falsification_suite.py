#!/usr/bin/env python3
"""
Falsification tests T1 to T5 for federated robustness evaluations.

Each test answers a yes/no question that cannot be answered from a paper alone. T2, T4 and T5 need no
training; T1 needs two runs per attack; T3 needs a held-out evaluation of the aggregated model.
"""
import json, math, glob, os, itertools
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # artifact root


# ---------------------------------------------------------------- T2
def t2_filter_admissibility(score_fn, threshold, support, name, accept_if="ge"):
    """Is the reject region ever reachable over the empirical support of the statistic?

 support : iterable of input tuples actually observed in deployment.
 Returns a verdict dict. FAIL means the filter cannot reject anything it will ever see.
 """
    scores = np.array([score_fn(*x) for x in support], dtype=float)
    rejected = scores < threshold if accept_if == "ge" else scores > threshold
    margin = (scores - threshold) if accept_if == "ge" else (threshold - scores)
    return {
        "test": "T2_filter_admissibility", "filter": name,
        "n_support": int(len(scores)), "threshold": float(threshold),
        "reject_rate": float(rejected.mean()),
        "score_min": float(scores.min()), "score_max": float(scores.max()),
        "closest_margin_to_boundary": float(np.min(np.abs(margin))),
        "verdict": "PASS" if rejected.any() else "FAIL: reject region unreachable on observed support",
    }


# ---------------------------------------------------------------- T4
def t4_seed_liveness(result_glob, metric_key="avg_map50"):
    """Do runs that differ only in seed actually produce different trajectories?"""
    groups = {}
    for f in sorted(glob.glob(result_glob)):
        base = os.path.basename(f)
        if "_s" not in base:
            continue
        stem, seed = base.rsplit("_s", 1)
        key = (os.path.dirname(f), stem)
        try:
            m = json.load(open(f))["metrics_distributed_fit"]
            if metric_key not in m:
                continue
            groups.setdefault(key, {})[seed.replace(".json", "")] = [round(v, 9) for _, v in m[metric_key]]
        except Exception:
            continue
    rows = []
    for (d, stem), seeds in sorted(groups.items()):
        if len(seeds) < 2:
            continue
        series = list(seeds.values())
        identical = all(s == series[0] for s in series[1:])
        rows.append({"experiment": os.path.basename(d), "arm": stem,
                     "n_seeds": len(seeds), "identical": bool(identical)})
    dead = [r for r in rows if r["identical"]]
    return {"test": "T4_seed_liveness", "n_arms_checked": len(rows),
            "n_arms_with_identical_seeds": len(dead), "dead_arms": dead,
            "verdict": "PASS" if not dead else f"FAIL: {len(dead)}/{len(rows)} arms have bit-identical seeds"}


# ---------------------------------------------------------------- scores under test
MU_REF, SIGMA_REF, BETA, Q_MIN = 0.29424603283405304, 0.1888456866145134, 0.1, 0.2

def deployed_qualitygate(mu, sigma):
    """Eq. 3 as shipped."""
    return (1.0 / (1.0 + math.exp(-BETA * (mu - MU_REF)))) * min(SIGMA_REF / (sigma + 1e-6), 2.0)


if __name__ == "__main__":
    out = {}

    # --- T2 on the deployed filter, over the photometric support measured in G1 ---
    g1 = json.load(open(ROOT / "5_derived_results/g1_photometric_separation.json"))
    benign, poisoned = [], []
    for coh in g1["cohorts"].values():
        for v in coh["per_client"].values():
            benign.append((v["robust"]["clean"]["mu"], v["robust"]["clean"]["sigma"]))
            for atk in ("brightness_flood", "darkness_injection", "noise_storm"):
                poisoned.append((v["robust"][atk]["mu"], v["robust"][atk]["sigma"]))

    out["T2_deployed_benign_only"] = t2_filter_admissibility(
        deployed_qualitygate, Q_MIN, benign, "deployed QualityGate Eq.3 (benign support)")
    out["T2_deployed_benign_and_poisoned"] = t2_filter_admissibility(
        deployed_qualitygate, Q_MIN, benign + poisoned, "deployed QualityGate Eq.3 (benign + poisoned support)")

    # --- T4 over every existing result JSON ---
    out["T4"] = t4_seed_liveness(str(ROOT / "1_audited_source/original_results/*/*.json"))

    p = ROOT / "5_derived_results/falsification_suite.json"
    p.write_text(json.dumps(out, indent=2))

    for k, v in out.items():
        print(f"\n=== {k} ===")
        for kk, vv in v.items():
            if kk == "dead_arms":
                print(f"  {kk}: {len(vv)} arms: " + ", ".join(f"{r['experiment']}/{r['arm']}" for r in vv[:8]))
            else:
                print(f"  {kk}: {vv}")
    print(f"\nwrote {p}")
