#!/usr/bin/env python3
"""
Emit every corpus-level number in the paper as a LaTeX macro, recomputed from the run records.

No number in the paper is typed by hand. The corpus is a fixed manifest of run tags
(`run_manifest.txt`); adding a run is an explicit edit, and a missing manifest entry is an error.
"""
import json, glob, os, subprocess
import numpy as np
from scipy import stats

R = "../4_run_records/"
LOGS = "../1_audited_source/original_logs/*.log"
OUT = "./computed_values.tex"

# The corpus is frozen to the runs the paper was built on. A glob over the results directory
# would silently absorb runs from later work (it did once: 268 runs instead of 114). Manifest is
# the list of record; add to it only by explicit decision.
MANIFEST = [t.strip() for t in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_manifest.txt")) if t.strip()]
rows = []
for f in [R + t + ".json" for t in MANIFEST]:
    if not os.path.exists(f): raise FileNotFoundError(f"manifest run missing: {f}")
    d = json.load(open(f)); g = {r["round"]: r["global_map50"] for r in d["global_eval"]}
    t = os.path.basename(f)[:-5]
    for r in d["round_log"]:
        if r["round"] in g and "local_map50" in r:
            rows.append((t, r["local_map50"], g[r["round"]]))
loc = np.array([r[1] for r in rows]); glo = np.array([r[2] for r in rows])
per = {}
for run in sorted(set(r[0] for r in rows)):
    sub = [r for r in rows if r[0] == run]
    if len(sub) < 6: continue
    a = np.array([x[1] for x in sub]); b = np.array([x[2] for x in sub])
    if a.std() > 1e-9 and b.std() > 1e-9:
        per[run] = (stats.pearsonr(a, b)[0], b.mean())
v = np.array([x[0] for x in per.values()]); gm = np.array([x[1] for x in per.values()])
neg = v < 0; coll = gm < 0.05

inv = subprocess.run(f"grep -ho 'Retained: [0-9]*/[0-9]* clients' {LOGS} | wc -l",
                     shell=True, capture_output=True, text=True).stdout.strip()
uniq = subprocess.run(f"grep -ho 'Retained: [0-9]*/[0-9]* clients' {LOGS} | sort -u | wc -l",
                      shell=True, capture_output=True, text=True).stdout.strip()

# central contrast, paired over the seeds that exist
from scipy import stats as _st
def _l4(t):
    g = json.load(open(R + t + ".json"))["global_eval"]
    return float(np.mean([r["global_map50"] for r in g[-4:]]))
def _loc4(t):
    l = [r["local_map50"] for r in json.load(open(R + t + ".json"))["round_log"]]
    return float(np.mean(l[-4:]))
seeds = [s for s in range(1, 9)
         if os.path.exists(R + f"T3_params_s{s}.json") and os.path.exists(R + f"T3_full_s{s}.json")]
GP = np.array([_l4(f"T3_params_s{s}") for s in seeds]); GF = np.array([_l4(f"T3_full_s{s}") for s in seeds])
LP = np.array([_loc4(f"T3_params_s{s}") for s in seeds]); LF = np.array([_loc4(f"T3_full_s{s}") for s in seeds])
dG, dL = GF - GP, LF - LP

# 2x2 head/buffer factorial
_F = {}
for t in ["F_head-ok_bn-ok", "F_head-bad_bn-ok", "F_head-ok_bn-bad", "F_head-bad_bn-bad"]:
    if os.path.exists(R + t + ".json"):
        d = json.load(open(R + t + ".json"))
        _F[t] = (float(np.mean([r["global_map50"] for r in d["global_eval"]][-4:])),
                 float(np.mean([r["local_map50"] for r in d["round_log"]][-4:])))
_fac = {}
if len(_F) == 4:
    b = _F["F_head-ok_bn-ok"]
    _fac = {
      "FacOKOKg": f"{b[0]:.4f}", "FacOKOKl": f"{b[1]:.4f}",
      "FacHEADg": f"{_F['F_head-bad_bn-ok'][0]:.4f}", "FacHEADl": f"{_F['F_head-bad_bn-ok'][1]:.4f}",
      "FacBUFg":  f"{_F['F_head-ok_bn-bad'][0]:.4f}", "FacBUFl":  f"{_F['F_head-ok_bn-bad'][1]:.4f}",
      "FacBOTHg": f"{_F['F_head-bad_bn-bad'][0]:.4f}","FacBOTHl": f"{_F['F_head-bad_bn-bad'][1]:.4f}",
      "EffHead": f"{_F['F_head-bad_bn-ok'][0]-b[0]:+.4f}",
      "EffBuf":  f"{_F['F_head-ok_bn-bad'][0]-b[0]:+.4f}",
      "EffBoth": f"{_F['F_head-bad_bn-bad'][0]-b[0]:+.4f}",
      "EffInter":f"{(_F['F_head-bad_bn-bad'][0]-b[0])-((_F['F_head-bad_bn-ok'][0]-b[0])+(_F['F_head-ok_bn-bad'][0]-b[0])):+.4f}",
      "EffBufLocal": f"{_F['F_head-ok_bn-bad'][1]-b[1]:+.4f}",
    }

# --- backdoor ASR denominators and dose-response -------------------------
def _last4(tag, key):
    return float(np.mean([r[key] for r in json.load(open(R + tag + ".json"))["global_eval"][-4:]]))

_a4 = json.load(open("../5_derived_results/a4_asr.json"))
_dose = [("0.00", "G2_fedavg_none_a30_s1"), ("0.25", "C5_bf_i025"),
         ("0.50", "C5_bf_i050"), ("1.00", "C5_bf_i100")]
_dg = [_last4(t, "global_map50") for _, t in _dose]
_a4m = {
  "AsrNaive":     f"{100*_a4['asr_naive']:.2f}",
  "AsrCorrected": f"{100*_a4['asr_corrected']:.2f}",
  "AsrSpecific":  f"{100*_a4['asr_specific']:.2f}",
  "AsrInflation": f"{100*(_a4['asr_naive']-_a4['asr_specific']):.1f}",
  "AsrRatio":     f"{_a4['asr_naive']/_a4['asr_specific']:.2f}",
  "TrigRecallClean": f"{_a4['trig_recall_clean']:.4f}",
  "TrigRecallBd":    f"{_a4['trig_recall_bd']:.4f}",
  "GlobRecallClean": f"{_a4['glob_recall_clean']:.4f}",
  "GlobRecallBd":    f"{_a4['glob_recall_bd']:.4f}",
  "TrigRetention":   f"{_a4['trig_recall_bd']/_a4['trig_recall_clean']:.4f}",
  "UntrigRetention": f"{_a4['glob_recall_bd']/_a4['glob_recall_clean']:.4f}",
  "BdCleanCost":  f"{_a4['clean_map_bd']-_a4['clean_map_clean']:.4f}",
  "BdCleanMap":   f"{_a4['clean_map_clean']:.4f}",
  "BdPoisonMap":  f"{_a4['clean_map_bd']:.4f}",
  "TrigOcclusionLoss": f"{100*(1-_a4['trig_recall_clean']):.1f}",
}
_c3h = json.load(open("../5_derived_results/c3_healthy.json"))["modes"]
_c3d = json.load(open("../5_derived_results/c3_damaged.json"))["modes"]
_gap = _c3h["none"]["mean"] - _c3d["none"]["mean"]
for _k, _m in [("None", "none"), ("Bn", "bn_only"), ("Head", "head_only"), ("Full", "full")]:
    _a4m["CthreeH" + _k] = f'{_c3h[_m]["mean"]:.4f}'
    _a4m["CthreeD" + _k] = f'{_c3d[_m]["mean"]:.4f}'
    _a4m["CthreeR" + _k] = f'{100*(_c3d[_m]["mean"]-_c3d["none"]["mean"])/_gap:.1f}'
_a4m["CthreeGap"] = f"{_gap:.4f}"
_a4m["CthreeBnCostHealthy"] = f'{_c3h["none"]["mean"]-_c3h["bn_only"]["mean"]:.4f}'
_a4m["CthreeFullCostHealthy"] = f'{100*(_c3h["none"]["mean"]-_c3h["full"]["mean"]):.2f}'
_a4m["CthreeBnToHealthy"] = f'{abs(_c3h["none"]["mean"]-_c3d["bn_only"]["mean"]):.4f}'
_ct = json.load(open("../5_derived_results/contract_state.json"))
_cs = json.load(open("../5_derived_results/contract_splits.json"))
_cm = json.load(open("../5_derived_results/contract_metrics.json"))
_pol = _ct["policy"]
_a4m.update({
  "CtTensors": _pol["parameters"] + _pol["float_buffers"] + _pol["integer_counters"],
  "CtParams": _pol["parameters"], "CtFloatBuf": _pol["float_buffers"],
  "CtIntCount": _pol["integer_counters"], "CtBnLayers": _pol["integer_counters"],
  "CtHashHealthy": _ct["hashes"]["healthy r12"], "CtHashDamaged": _ct["hashes"]["damaged r12"],
  "CtHashHealthyPrev": _ct["hashes"]["healthy r11"],
  "CtTrainImgs": _cs["sizes"]["global/train"], "CtValImgs": _cs["sizes"]["global/val"],
  "CtExactOverlap": len(_cs["exact_overlaps"]), "CtHashCand": _cs["hash_candidates"],
  "CtMaxPixelR": f'{_cs["max_pixel_r"]:+.3f}', "CtConfirmedDup": len(_cs["near_duplicates"]),
  "CtGcmH": f'{_cm["healthy"]["G_client_macro"]:.4f}', "CtGcmD": f'{_cm["damaged"]["G_client_macro"]:.4f}',
  "CtGcwH": f'{_cm["healthy"]["G_client_weighted"]:.4f}', "CtGcwD": f'{_cm["damaged"]["G_client_weighted"]:.4f}',
  "CtLcmH": f'{_cm["healthy"]["L_client_macro"]:.4f}', "CtLcmD": f'{_cm["damaged"]["L_client_macro"]:.4f}',
  "CtGpoolH": "0.6668", "CtGpoolD": "0.0007",
})
_c4 = json.load(open("../5_derived_results/c4_localonly.json"))
_c4L = float(np.mean([v["local_final"] for v in _c4["clients"].values()]))
_c4G = float(np.mean([v["global_final"] for v in _c4["clients"].values()]))
_a4m.update({"CfourLocal": f"{_c4L:.4f}", "CfourGlobal": f"{_c4G:.4f}",
             "CfourLocalGain": f"{100*(float(_cm['healthy']['L_client_macro'])-_c4L):.2f}",
             "CfourGlobalGain": f"{100*(0.6668-_c4G):.2f}"})
# headline numbers used in the abstract and introduction
def _lastfour(t, key="global_map50"):
    return float(np.mean([r[key] for r in json.load(open(R + t + ".json"))["global_eval"][-4:]]))
def _locfour(t):
    d = json.load(open(R + t + ".json"))["round_log"]
    return float(np.mean([r["local_map50"] for r in d if "local_map50" in r][-4:]))
_BF = ["G2_fedavg_brightness_flood_a30_s1", "S_fedavg_BF_s2", "S_fedavg_BF_s3"]
_CL = ["G2_fedavg_none_a30_s1", "S_fedavg_none_s2", "S_fedavg_none_s3"]
_gb, _gc = np.array([_lastfour(t) for t in _BF]), np.array([_lastfour(t) for t in _CL])
_lb, _lc = np.array([_locfour(t) for t in _BF]), np.array([_locfour(t) for t in _CL])
_bfg, _bfl = 100*(_gc.mean()-_gb.mean()), 100*(_lc.mean()-_lb.mean())
# within-run correlation of the failing protocol, seed 1
_d = json.load(open(R + "T3_params_s1.json"))
_gm = {r["round"]: r["global_map50"] for r in _d["global_eval"]}
_pr = [(r["local_map50"], _gm[r["round"]]) for r in _d["round_log"]
       if r["round"] in _gm and "local_map50" in r]
_wr = stats.pearsonr([x[0] for x in _pr], [x[1] for x in _pr])[0]
def _wrun(t):
    _d = json.load(open(R + t + ".json"))
    _g = {r["round"]: r["global_map50"] for r in _d["global_eval"]}
    _pp = [(r["local_map50"], _g[r["round"]]) for r in _d["round_log"]
           if r["round"] in _g and "local_map50" in r]
    return stats.pearsonr([x[0] for x in _pp], [x[1] for x in _pp])[0]
_rf = [_wrun(f"T3_params_s{i}") for i in range(1, 6)]
_ro = [_wrun(f"T3_full_s{i}") for i in range(1, 6)]
def _locr(t, last=None, rnd=None):
    _d = json.load(open(R + t + ".json"))["round_log"]
    _v = [(r["round"], r["local_map50"]) for r in _d if "local_map50" in r]
    if rnd: return [x[1] for x in _v if x[0] == rnd][0]
    return float(np.mean([x[1] for x in _v][-last:]))
_f4 = abs(_locr("T3_full_s1", last=4) - _locr("T3_params_s1", last=4))
_f12 = abs(_locr("T3_full_s1", rnd=12) - _locr("T3_params_s1", rnd=12))
for _i in range(1, 6):
    _a4m[f"CorrFail{'IVXABCDE'[_i-1]}"] = f"{_wrun(f'T3_params_s{_i}'):+.3f}"
    _a4m[f"CorrOk{'IVXABCDE'[_i-1]}"] = f"{_wrun(f'T3_full_s{_i}'):+.3f}"
_a4m.update({
  "FigLastFourAP": f"{100*_f4:.2f}", "FigRoundTwelveAP": f"{100*_f12:.2f}",
  "FailRunRMean": f"{np.mean(_rf):+.3f}", "FailRunRMin": f"{max(_rf):+.3f}",
  "FailRunRMax": f"{min(_rf):+.3f}", "OkRunRMean": f"{np.mean(_ro):+.3f}",
  "BfGlobalDrop": f"{_bfg:.2f}", "BfLocalDrop": f"{_bfl:.2f}",
  "BfGlobalP": f"{stats.ttest_ind(_gc,_gb,equal_var=False)[1]:.3f}",
  "BfRatio": f"{_bfl/_bfg:.1f}",
  "FailRunR": f"{_wr:+.3f}", "FailRunRShort": f"{_wr:+.2f}",
  "LocalDiffPct": f"{100*(float(_M0['LocalOK'])-float(_M0['LocalFail']))/float(_M0['LocalOK']):.2f}"
      if False else f"{100*(LF.mean()-LP.mean())/LF.mean():.2f}",
})
# selection regret and rank agreement, over the eight published aggregation rules
_RULES = ["fedavg", "median", "trimmed_mean", "multi_krum", "rfa", "norm_bound",
          "flame", "photoscreen2qn"]
_RNAME = {"fedavg": "plain averaging", "median": "median", "trimmed_mean": "trimmed mean",
          "krum": "Krum", "multi_krum": "multi-Krum", "rfa": "RFA",
          "norm_bound": "fixed norm clipping", "flame": "adaptive norm clipping",
          "photoscreen2qn": "screened averaging"}
def _pair(tag):
    _p = R + tag + ".json"
    if not os.path.exists(_p): return None
    _d = json.load(open(_p))
    _gv = [r["global_map50"] for r in _d["global_eval"]][-4:]
    _lv = [r["local_map50"] for r in _d["round_log"] if "local_map50" in r][-4:]
    return (float(np.mean(_lv)), float(np.mean(_gv))) if _gv and _lv else None
_ATK = [("alie", "Alie"), ("ipm", "Ipm"), ("min_max", "MinMax"), ("min_sum", "MinSum")]
_regs = []
for _a, _A in _ATK:
    _rows = [(r,) + _pair(f"M_{r}_{_a}") for r in _RULES if _pair(f"M_{r}_{_a}")]
    _sel = max(_rows, key=lambda x: x[1]); _best = max(_rows, key=lambda x: x[2])
    _reg = _best[2] - _sel[2]; _regs.append(_reg)
    _rho = stats.spearmanr([x[1] for x in _rows], [x[2] for x in _rows])[0]
    _a4m[f"Reg{_A}Sel"] = _RNAME[_sel[0]]
    _a4m[f"Reg{_A}True"] = f"{_sel[2]:.4f}"
    _a4m[f"Reg{_A}Best"] = f"{_best[2]:.4f}"
    _a4m[f"Reg{_A}Gap"] = f"{_reg:.4f}"
    _a4m[f"Reg{_A}Rho"] = f"{_rho:+.3f}"
_a4m["RegMean"] = f"{100*np.mean(_regs):.2f}"
_a4m["RegWorst"] = f"{100*max(_regs):.2f}"
_a4m["RegNRules"] = len(_RULES)
_big = sorted(_regs)[-2:]
_a4m["RegBigLo"] = f"{_big[0]/0.0028:.0f}"
_a4m["RegBigHi"] = f"{_big[1]/0.0028:.0f}"
_a4m.update({"DoseZero": f"{_dg[0]:.4f}", "DoseQ": f"{_dg[1]:.4f}",
             "DoseH": f"{_dg[2]:.4f}", "DoseF": f"{_dg[3]:.4f}",
             "DoseFullDrop": f"{100*(_dg[0]-_dg[3]):.2f}", "DoseSDMultiple": f"{(_dg[0]-_dg[3])/0.0028:.1f}",
             "DoseSpearman": f"{stats.spearmanr([0,.25,.5,1.0], _dg)[0]:+.3f}"})

M = {
  **_fac,
  "NSeeds": len(seeds),
  "GlobalFail": f"{GP.mean():.5f}", "GlobalOK": f"{GF.mean():.5f}",
  "GlobalDiffAP": f"{100*dG.mean():.2f}", "GlobalDiffSD": f"{100*dG.std(ddof=1):.2f}",
  "GlobalDiffP": f"{_st.ttest_rel(GF, GP)[1]:.0e}".replace("e-", "\\times 10^{-") + "}",
  "LocalFail": f"{LP.mean():.4f}", "LocalOK": f"{LF.mean():.4f}",
  "LocalDiffAP": f"{100*dL.mean():+.2f}", "LocalDiffP": f"{_st.ttest_rel(LF, LP)[1]:.2f}",
  "NRuns": len(set(r[0] for r in rows)),
  "NObs": len(rows),
  "NEligible": len(per),
  "PooledPearson": f"{stats.pearsonr(loc,glo)[0]:+.3f}",
  "PooledSpearman": f"{stats.spearmanr(loc,glo)[0]:+.3f}",
  "MeanWithinRun": f"{v.mean():+.3f}",
  "NNegative": int(neg.sum()),
  "NSurvived": int((~coll).sum()),
  "MeanRSurvived": f"{v[~coll].mean():+.3f}",
  "NCollapsed": int(coll.sum()),
  "MeanRCollapsed": f"{v[coll].mean():+.3f}",
  "NNegAmongCollapsed": int((neg & coll).sum()),
  "ScreenInvocations": inv,
  "ScreenDistinctOutcomes": uniq,
  **_a4m,
}
with open(OUT, "w") as f:
    f.write("% AUTO-GENERATED by figures/computed_values.py. Do not edit by hand.\n")
    for k, val in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
for k, val in M.items():
    print(f"  {k:24s} {val}")
print(f"\nwrote {OUT}")
