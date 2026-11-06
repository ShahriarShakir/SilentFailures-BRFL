#!/usr/bin/env python3
"""
Driver: inject each fault, apply the five checks blind, count detections and false alarms.
"""
import os, sys, json, itertools
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
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
    # ---- probe ----
    # Intended as a working-screen control: the screen a practitioner would configure, with the
    # textbook 3-MAD threshold. On these heterogeneous clients it never rejects anyone (the attackers sit
    # 1.7 and 2.1 MADs from the median), so it is not a working screen; it is reported as a probe of
    # T2 and T5 on a correctly oriented but inadmissible screen, not counted as a fault or a control.
    Fault("textbook screen (3 MAD)",   "probe",   screen="ok"),
]
# 'deliberate personalised BN' is intentionally the same mechanism as the buffer fault: the
# difference is declared intent, which no behavioural check can see. We record it as a control to
# measure exactly that limitation rather than to flatter the checks.

def checks(f, logs_by_seed):
    """T1..T5 as stated in the paper, with T3 and T4 in their corrected forms.

    T3 is a structural check: was the reported metric computed from the aggregated state? It is
    decidable per round and does not depend on a correlation, which misses faults where both series
    rise and diverge only in level.

    T4 here is the assertion half: the seed must arrive at its point of use. replicate.py adds the
    second half (different seeds must give different trajectories). In the audited archive, comparing
    trajectories alone flags 3 of the 26 arms while the seed was inert in 16 (paper, Section 4.8).
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
    # T5 orientation, as specified in Section 8: under the attack's own transformation the admit score
    # must move towards rejection for a majority of honest inputs; it fails otherwise
    tw = [t for d in lg for t in d["t5_toward"]]
    out["T5"] = None if f.screen == "none" else bool(np.mean(tw) < 0.5)
    # the earlier, weaker implementation (does the screened input vary at all?), kept for the record
    out["T5_input_varies"] = None if f.screen == "none" else bool(np.std(lg[0]["screen_scores"]) < 1e-9)
    return out


if __name__ == "__main__":
    R = {}
    for f in CONDITIONS:
        runs = [run(f, seed=s, rounds=8) for s in (0, 1)]
        # T1 as specified: the attack against no attack, with the defence removed (plain averaging)
        undef = dict(head_shared=f.head_shared, buffers_shared=f.buffers_shared, screen="none",
                     seed_live=f.seed_live, metric_stage=f.metric_stage, attack_strength=f.attack_strength)
        t1a = run(Fault(f.name + "/t1-attack", f.kind, attack_applied=f.attack_applied, **undef), seed=0, rounds=8)
        t1b = run(Fault(f.name + "/t1-noattack", f.kind, attack_applied=False, **undef), seed=0, rounds=8)
        c = checks(f, runs)
        gA = np.mean([d["global"] for d in runs[0][-3:]])
        gN = np.mean([d["global"] for d in t1b[-3:]])
        gT1 = np.mean([d["global"] for d in t1a[-3:]])
        c["T1"] = bool(abs(gT1 - gN) < 0.01)         # attack fails to move the undefended baseline
        R[f.name] = {"kind": f.kind, "checks": c, "t1_attacked_undefended_last3": float(gT1),
                     "t5_fraction_toward_rejection": (float(np.mean([t for d in runs[0] for t in d["t5_toward"]])) if f.screen != "none" else None),
                     "global_last3": float(gA), "noattack_last3": float(gN),
                     "local_last3": float(np.mean([d["local"] for d in runs[0][-3:]]))}
        fired = [k for k, v in c.items() if v is True and k != "T5_input_varies"]
        na = [k for k, v in c.items() if v is None and k != "T5_input_varies"]
        print(f"  {f.kind:8s} {f.name:28s} global {gA:.3f} local {R[f.name]['local_last3']:.3f}  "
              f"fired: {','.join(fired) or '-'}" + (f"   n/a: {','.join(na)}" if na else ""))
    json.dump(R, open(os.path.join(HERE, "../../5_derived_results/c6_injection.json"), "w"), indent=2)
    print("\nwrote c6_injection.json")
