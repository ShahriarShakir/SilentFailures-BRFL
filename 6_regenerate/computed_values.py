#!/usr/bin/env python3
"""Emit every corpus-level number in the paper as a LaTeX macro, recomputed from the artefacts.

Hand-entered corpus statistics went stale three times as runs were added. The paper now reads these
macros instead, so a number cannot disagree with the data that produced it.
"""
import json, glob, os, subprocess
import numpy as np
from scipy import stats

R = "../4_run_records/"
LOGS = "../1_audited_source/original_logs/*.log"
OUT = "./computed_values.tex"

# The corpus is FROZEN to the runs that existed at submission. A glob over the results directory
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

# --- A4 backdoor ASR denominators and C5 dose-response -------------------------
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
})
# G_pool of the two round-12 checkpoints, read from the runs that saved them
def _r12(t):
    _e = json.load(open(R + t + ".json"))["global_eval"]
    assert _e[-1]["round"] == 12
    return float(_e[-1]["global_map50"])
_gph, _gpd = _r12("A4_clean"), _r12("C3_damaged_src")      # healthy and damaged checkpoint sources
_a4m.update({"CtGpoolH": f"{_gph:.4f}", "CtGpoolD": f"{_gpd:.4f}"})
_c4 = json.load(open("../5_derived_results/c4_localonly.json"))
_c4L = float(np.mean([v["local_final"] for v in _c4["clients"].values()]))
_c4G = float(np.mean([v["global_final"] for v in _c4["clients"].values()]))
_a4m.update({"CfourLocal": f"{_c4L:.4f}", "CfourGlobal": f"{_c4G:.4f}",
             "CfourLocalGain": f"{100*(float(_cm['healthy']['L_client_macro'])-_c4L):.2f}",
             "CfourGlobalGain": f"{100*(_gph-_c4G):.2f}"})
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
# pooled across-seed SD of the global metric over the six arms of the corpus run at three seeds
_SDARMS = [["G2_fedavg_none_a30_s1", "S_fedavg_none_s2", "S_fedavg_none_s3"],
           ["G2_fedavg_brightness_flood_a30_s1", "S_fedavg_BF_s2", "S_fedavg_BF_s3"],
           ["P1f_Qn_BF_s1", "S_qn_BF_s2", "S_qn_BF_s3"],
           ["P1e_Qn_DoS_s1", "S_qn_spoof_s2", "S_qn_spoof_s3"],
           ["S_mad_BF_s1", "S_mad_BF_s2", "S_mad_BF_s3"],
           ["P1c_DoS_anchorOFF_s1", "S_mad_spoof_s2", "S_mad_spoof_s3"]]
assert all(t_ in MANIFEST for a_ in _SDARMS for t_ in a_)
_psd = float(np.sqrt(sum((len(a_) - 1) * np.var([_last4(t_, "global_map50") for t_ in a_], ddof=1) for a_ in _SDARMS)
                     / sum(len(a_) - 1 for a_ in _SDARMS)))
_a4m["PooledSD"] = f"{_psd:.4f}"
_a4m["PooledSDArms"] = len(_SDARMS)
_a4m["RegNRules"] = len(_RULES)
_big = sorted(_regs)[-2:]
_a4m["RegBigLo"] = f"{_big[0]/_psd:.0f}"
_a4m["RegBigHi"] = f"{_big[1]/_psd:.0f}"
_a4m.update({"DoseZero": f"{_dg[0]:.4f}", "DoseQ": f"{_dg[1]:.4f}",
             "DoseH": f"{_dg[2]:.4f}", "DoseF": f"{_dg[3]:.4f}",
             "DoseFullDrop": f"{100*(_dg[0]-_dg[3]):.2f}", "DoseSDMultiple": f"{(_dg[0]-_dg[3])/_psd:.1f}",
             "DoseSpearman": f"{stats.spearmanr([0,.25,.5,1.0], _dg)[0]:+.3f}"})

# ---------------- Table 13: every run with negative within-run correlation, from the records
_LBL = {"T3_params_s{}": "failing protocol, seed {}", "E{}_params_s1": "failing protocol, $E{{=}}{}$",
        "C3_damaged_src": "failing protocol, source of the damaged checkpoint (Section~\\ref{sec:mechanism})",
        "F_head-ok_bn-bad": "factorial arm: head aggregated, buffers withheld",
        "M_rfa_min_max": "geometric median, Min-Max", "M_multi_krum_ipm": "multi-Krum, IPM",
        "M_norm_bound_min_max": "fixed norm clipping, Min-Max", "M_flame_min_max": "adaptive norm clipping, Min-Max",
        "M_trimmed_mean_min_max": "trimmed mean, Min-Max", "C_qntrimmed_min_max": "screen + trimmed mean, Min-Max",
        "M_fedavg_min_max": "plain averaging, Min-Max", "M_rfa_ipm": "geometric median, IPM",
        "M_photoscreen2qn_min_max": "screened averaging, Min-Max", "M_fedavg_ipm": "plain averaging, IPM"}
def _label(run):
    import re as _re
    m = _re.match(r"T3_params_s(\d+)$", run)
    if m: return f"failing protocol, seed {m.group(1)}"
    m = _re.match(r"E(\d+)_params_s1$", run)
    if m: return f"failing protocol, $E{{=}}{m.group(1)}$"
    if run in _LBL: return _LBL[run]
    raise KeyError(f"no table label for run {run}: add one to _LBL")
_negrows = sorted(((k_, v_, g_) for k_, (v_, g_) in per.items() if v_ < 0), key=lambda t: (t[2] >= 0.05, t[1]))
_tab = []
for i_, (k_, v_, g_) in enumerate(_negrows):
    if i_ > 0 and _negrows[i_ - 1][2] < 0.05 <= g_: _tab.append("\\midrule")
    _tab.append(f"{_label(k_)} & ${v_:+.3f}$ & {g_:.4f} \\\\")
_collnn = [k_ for k_, (v_, g_) in per.items() if v_ >= 0 and g_ < 0.05]

# the one collapsed run that is not negative, described in prose (Section 5.2)
assert len(_collnn) == 1, _collnn            # the text says "the remaining run"
_cg = [r["global_map50"] for r in json.load(open(R + _collnn[0] + ".json"))["global_eval"]]
_prose = lambda s: s.replace(", ", " under ", 1) if s.rsplit(", ", 1)[-1] in ("IPM", "ALIE", "Min-Max", "Min-Sum") else s

# ---------------- Section 4.8: the archive's multi-seed arms (result files of the audited pipeline)
ARCH = "../1_audited_source/original_results/"
_arch = []
for f_ in sorted(glob.glob(ARCH + "*/*_s1.json")):
    g_ = f_[:-len("_s1.json")] + "_s2.json"
    if not os.path.exists(g_): continue
    try:
        a_ = [v for _, v in json.load(open(f_))["metrics_distributed_fit"]["avg_map50"]]
        b_ = [v for _, v in json.load(open(g_))["metrics_distributed_fit"]["avg_map50"]]
    except (KeyError, ValueError): continue
    _arch.append((f_, abs(a_[0] - b_[0]), max(abs(x - y) for x, y in zip(a_, b_))))
_same1 = [x for x in _arch if x[1] <= 1e-15]; _diff1 = [x for x in _arch if x[1] > 1e-15]
_ident = [x for x in _arch if x[2] <= 1e-15]
_krum = [json.load(open(ARCH + f"C_robust_baselines/krum_s{s}.json"))["wall_seconds"] for s in (1, 2)]
_rbl = subprocess.run(f"grep 'engine/trainer' {os.path.dirname(LOGS)}/robust_baselines.log | grep -oE 'seed=[0-9]+' | sort | uniq -c",
                      shell=True, capture_output=True, text=True).stdout.split()
assert len(_rbl) == 2 and _rbl[1] == "seed=0", _rbl          # one value only, and it is 0
_mx = max(x[2] for x in _ident); _e = int(np.floor(np.log10(_mx)))
_d1 = sorted(round(x[1], 3) for x in _diff1)
# ---------------- Section 4.9: the averaging step shared by the screened strategy and its FedAvg, FedProx and
# FedBN baselines returns one tensor, and every client then prints the warning below in every round.
_one_lines = int(subprocess.run(f"grep -h 'Not enough parameters. Expected 2, got 1' {LOGS} | wc -l",
                                shell=True, capture_output=True, text=True).stdout)
_one_runs = int(subprocess.run(f"grep -hE 'FL STRATEGY: (PhotoScreen|FedAvg \\(Baseline\\)|FedBN)' {LOGS} | wc -l",
                               shell=True, capture_output=True, text=True).stdout)
_onev, _one_files = [], 0
for f_ in sorted(glob.glob(ARCH + "**/*.json", recursive=True)):
    d_ = json.load(open(f_))
    if str(d_.get("strategy", "")).lower() not in ("photoscreen", "fedavg", "fedprox", "fedbn"): continue
    _one_files += 1
    _onev += [v for _, v in d_["metrics_distributed_fit"]["avg_map50"]]
assert _one_lines and _one_runs and _one_files, (_one_lines, _one_runs, _one_files)
_a4m.update({"OneTensorLines": f"{_one_lines:,}".replace(",", "{,}"), "OneTensorRuns": _one_runs,
             "OneTensorFiles": _one_files,
             "OneTensorRepMin": f"{np.floor(100 * min(_onev)) / 100:.2f}",
             "OneTensorRepMax": f"{np.ceil(100 * max(_onev)) / 100:.2f}"})
# ---------------- Section 8, T3: what a weak positive correlation means, from the same per-run correlations
_l4 = {k_: float(np.mean([r_[2] for r_ in rows if r_[0] == k_][-4:])) for k_ in per}
_hl = [k_ for k_ in per if _l4[k_] >= 0.60]                 # aggregate ends healthy
_wk = [k_ for k_ in per if 0 <= per[k_][0] < 0.5]           # weak positive correlation
assert _hl and _wk and not set(_hl) & set(_wk)
_a4m.update({"TThreeNHealthy": len(_hl),
             "TThreeHealthyMinR": f"{np.floor(100 * min(per[k_][0] for k_ in _hl)) / 100:+.2f}",
             "TThreeNWeak": len(_wk),
             "TThreeWeakMaxGlobal": f"{np.ceil(100 * max(_l4[k_] for k_ in _wk)) / 100:.2f}"})
# ---------------- Section 5.1: the three-seed check, from the derived file seed_check.json
_SCK = json.load(open("../5_derived_results/seed_check.json"))["attacks"]
_same = _tot = _orddiff = _ordtot = 0; _exc = []
for a_, d_ in _SCK.items():
    rr_ = d_["rounds"]
    for i_, x_ in enumerate(rr_):
        _tot += 1
        if len(set(x_["agg_state_hash"])) == 1: _same += 1
        else: _exc.append((a_, x_["round"], len(set(x_["global_map50"])) == 1,
                           i_ + 1 < len(rr_) and len(set(rr_[i_ + 1]["agg_state_hash"])) == 1))
        _ordtot += 1; _orddiff += len(set(x_["order"])) > 1
assert all(e_[2] and e_[3] for e_ in _exc), _exc   # the text: global metric identical, hashes agree a round later
_ATN = {"ipm": "IPM", "alie": "ALIE", "min_max": "Min-Max", "min_sum": "Min-Sum"}

# ---------------- Sections 4.5 and 8: the screen over its observed support (Equation 1)
_g1 = json.load(open("../5_derived_results/g1_photometric_separation.json"))
_ic = _g1["iara_constants"]
_pc = json.load(open("../1_audited_source/data/bn_statistics/iara_config.json"))["datasets"]["ecp"]
_PROF = {"deployed": (_ic["mu_ref"], _ic["sigma_ref"]),
         "correct": (_pc["mu_ref"], _pc["sigma_ref"])}   # the training dataset's entry in the pipeline's profile file
def _q(mu, s, prof):
    mr, sr = _PROF[prof]
    return (1.0 / (1.0 + np.exp(-_ic["beta"] * (mu - mr)))) * min(sr / (s + 1e-6), 2.0)
_ATKS = ("brightness_flood", "darkness_injection", "noise_storm")
_sup = [(v["robust"][c]["mu"], v["robust"][c]["sigma"]) for coh in _g1["cohorts"].values()
        for v in coh["per_client"].values() for c in ("clean",) + _ATKS]
_dep7 = _g1["cohorts"]["7client"]["per_client"].values()
_sc = [v["deployed"]["clean"]["sigma"] for v in _dep7]
_sa = [v["deployed"][a]["sigma"] for v in _dep7 for a in _ATKS]
assert all(_q(m, s, p) >= _ic["q_min"] for m, s in _sup for p in _PROF)   # the text: rejected under neither profile

M = {
  "ArchArms": len(_arch), "ArchIdentArms": len(_ident), "ArchVaryArms": len(_arch) - len(_ident), "ArchRoundOneSame": len(_same1),
  "ArchRoundOneDiff": len(_diff1), "ArchVaryLater": len(_same1) - len(_ident),
  "ArchIdentMaxDiff": f"{_mx / 10**_e:.1f}\\times10^{{{_e}}}",
  "ArchRoundOneDiffVal": f"{_d1[0]:.3f}" if _d1[0] == _d1[-1] else f"{_d1[0]:.3f} to {_d1[-1]:.3f}",
  "ArchSeedZeroLines": int(_rbl[0]), "ArchKrumSecA": f"{_krum[0]:.0f}", "ArchKrumSecB": f"{_krum[1]:.0f}",
  "SeedHashSame": _same, "SeedHashTotal": _tot,
  "SeedHashExc": "; ".join(f"{_ATN[a_]}, round {r_}" for a_, r_, _, _ in _exc) or "none",
  "SeedOrderDiffRounds": _orddiff, "SeedOrderTotalRounds": _ordtot,
  "ScreenNSupport": len(_sup),
  "ScreenMinDeployed": f"{min(_q(m, s, 'deployed') for m, s in _sup):.3f}",
  "ScreenMinCorrect": f"{min(_q(m, s, 'correct') for m, s in _sup):.3f}",
  "ScreenSigmaCleanMin": f"{min(_sc):.3f}", "ScreenSigmaCleanMax": f"{max(_sc):.3f}",
  "ScreenSigmaAttackMax": f"{max(_sa):.3f}",
  "NegRunRows": "\n".join(_tab),
  "NNegCollapsedRows": sum(1 for _, _, g_ in _negrows if g_ < 0.05),
  "CollapsedNotNegLabel": _prose(_label(_collnn[0])),
  "CollapsedNotNegZero": sum(1 for x in _cg if x == 0.0), "CollapsedNotNegRounds": len(_cg),
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
  "NNegAmongSurvived": int((neg & ~coll).sum()),
  "NCollapsedNotNeg": int((~neg & coll).sum()),
  "CollapsedNotNegR": "; ".join(f"{v_:+.3f}" for k_, (v_, g_) in per.items() if v_ >= 0 and g_ < 0.05),
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
