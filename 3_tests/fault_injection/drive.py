#!/usr/bin/env python3
"""
Driver: inject each fault, apply the five checks blind, count detections and false alarms.
"""
import sys, json, itertools
import numpy as np
from scipy import stats
sys.path.insert(0, "/ANON/fl_hod/project/work/c6")
from inject import Fault, run

CONDITIONS = [
    # ---- faults ----
    Fault("attack never applied",      "fault",  attack_applied=False),
    Fault("head not shared",           "fault",  head_shared=False),
    Fault("buffers not shared",        "fault",  buffers_shared=False),
    Fault("screen input constant",     "fault",  screen="constant"),
    Fault("screen inert",              "fault",  screen="inert"),
    Fault("screen reversed",           "fault",  screen="reversed"),
    Fault("wrong reference profile",   "fault",  screen="wrongref"),
    Fault("seed ignored",              "fault",  seed_live=False),
    Fault("metric stage mislabelled",  "fault",  metric_stage="pre_adaptation_offset"),
    # ---- healthy controls ----
    Fault("healthy",                   "healthy"),
    Fault("deliberate personalised BN","healthy", buffers_shared=False, head_shared=True),
    Fault("weak but executed attack",  "healthy", attack_strength=0.15),
    Fault("harmless transform",        "healthy"),
]
# 'deliberate personalised BN' is intentionally the same mechanism as the buffer fault: the
# difference is declared intent, which no behavioural check can see. We record it as a control to
# measure exactly that limitation rather than to flatter the checks.

def checks(f, logs_by_seed):
    """T1..T5 as stated in the paper, with T3 and T4 in their corrected forms.

 It is restated as a structural
 check, was the reported metric computed from the aggregated state?, which is decidable per
 round and does not depend on a correlation.

 T4 was originally 'do two seeds give different trajectories'. That passed on 23 of 26 arms in the
 real archive while the seed reached nothing. It is restated as an assertion at the point of use.
 """
    out = {}
    lg = logs_by_seed[0]
    g = np.array([d["global"] for d in lg]); l = np.array([d["local"] for d in lg])
    out["T1"] = None
    # T2 admissibility: only meaningful when a screen exists
    out["T2"] = None if f.screen == "none" else all(d["n_kept"] == 6 for d in lg)
    # T3 state identity: the reported metric must come from the aggregated state
    out["T3"] = any(d["aggregated_state_hash"] not in d["metric_state_hashes"] for d in lg)
    # T4 seed assertion at point of use: the seed must vary with the configured seed
    s0 = [d["seeds_used"] for d in logs_by_seed[0]]
    s1 = [d["seeds_used"] for d in logs_by_seed[1]]
    out["T4"] = (s0 == s1)
    # T5 orientation: the screened statistic must respond to the attack transformation
    out["T5"] = None if f.screen == "none" else bool(np.std(lg[0]["screen_scores"]) < 1e-9)
    return out


if __name__ == "__main__":
    R = {}
    for f in CONDITIONS:
        runs = [run(f, seed=s, rounds=8) for s in (0, 1)]
        base = [run(Fault(f.name + "/noatk", f.kind, attack_applied=False,
                          head_shared=f.head_shared, buffers_shared=f.buffers_shared,
                          screen=f.screen, seed_live=f.seed_live,
                          metric_stage=f.metric_stage), seed=s, rounds=8) for s in (0,)]
        c = checks(f, runs)
        gA = np.mean([d["global"] for d in runs[0][-3:]])
        gN = np.mean([d["global"] for d in base[0][-3:]])
        c["T1"] = bool(abs(gA - gN) < 0.01)          # attack fails to move the undefended baseline
        R[f.name] = {"kind": f.kind, "checks": c,
                     "global_last3": float(gA), "noattack_last3": float(gN),
                     "local_last3": float(np.mean([d["local"] for d in runs[0][-3:]]))}
        fired = [k for k, v in c.items() if v is True]
        na = [k for k, v in c.items() if v is None]
        print(f"  {f.kind:8s} {f.name:28s} global {gA:.3f} local {R[f.name]['local_last3']:.3f}  "
              f"fired: {','.join(fired) or '-'}" + (f"   n/a: {','.join(na)}" if na else ""))
    json.dump(R, open("/ANON/fl_hod/project/work/results/c6_injection.json", "w"), indent=2)
    print("\nwrote c6_injection.json")
