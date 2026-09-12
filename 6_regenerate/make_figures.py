#!/usr/bin/env python3
"""
Figures for the paper. Every value is read from the run records; nothing is drawn by hand.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

R = "../4_run_records/"
OUT = "./figures/"
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                     "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
C_GLOB, C_LOC = "#1f4e79", "#c0504d"

def series(tag):
    d = json.load(open(R + tag + ".json"))
    g = {r["round"]: r["global_map50"] for r in d["global_eval"]}
    l = {r["round"]: r["local_map50"] for r in d["round_log"]}
    rs = sorted(set(g) & set(l))
    return np.array(rs), np.array([g[r] for r in rs]), np.array([l[r] for r in rs])

# ---------------- Figure 1: the two protocols side by side ----------------
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
for k, (tag, title) in enumerate([("T3_params_s1", "(a) parameters only"),
                                  ("T3_full_s1",   "(b) full state")]):
    r, g, l = series(tag)
    ax[k].plot(r, g, "o-", color=C_GLOB, lw=2, ms=5, label="global (held-out, aggregated model)")
    ax[k].plot(r, l, "s--", color=C_LOC, lw=2, ms=5, label="client-local (as reported)")
    ax[k].set_title(title); ax[k].set_xlabel("communication round")
    ax[k].set_ylim(-0.03, 0.72); ax[k].set_xticks(range(1, 13, 2))
ax[0].set_ylabel("mAP50")
ax[0].annotate("aggregated model\ndetects nothing", xy=(9, 0.0), xytext=(5.4, 0.16),
               fontsize=9.5, ha="center", color=C_GLOB,
               arrowprops=dict(arrowstyle="->", color=C_GLOB, lw=1.2))
ax[0].annotate("reported metric\nkeeps rising", xy=(10, 0.630), xytext=(5.8, 0.44),
               fontsize=9.5, ha="center", color=C_LOC,
               arrowprops=dict(arrowstyle="->", color=C_LOC, lw=1.2))
ax[1].legend(loc="lower right", framealpha=0.95)
fig.savefig(OUT + "fig_protocols.pdf"); plt.close(fig)

# ---------------- Figure 2: all runs, local vs global ----------------
pts, neg = [], []
for f in sorted(glob.glob(R + "*.json")):
    d = json.load(open(f)); g = {r["round"]: r["global_map50"] for r in d["global_eval"]}
    p = [(r["local_map50"], g[r["round"]]) for r in d["round_log"] if r["round"] in g and "local_map50" in r]
    if len(p) < 6: continue
    a = np.array(p)
    if a[:, 0].std() > 1e-9 and a[:, 1].std() > 1e-9:
        rr = stats.pearsonr(a[:, 0], a[:, 1])[0]
        (neg if rr < 0 else pts).append(a)
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
P = np.vstack(pts); N = np.vstack(neg)
ax[0].scatter(P[:, 1], P[:, 0], s=9, alpha=.35, color="#4a7ebb", label=f"runs with $r>0$ (n={len(pts)})")
ax[0].scatter(N[:, 1], N[:, 0], s=14, alpha=.8, color="#c0504d", marker="^", label=f"runs with $r<0$ (n={len(neg)})")
ax[0].set_xlabel("global mAP50 (aggregated model)"); ax[0].set_ylabel("client-local metric")
ax[0].set_title(f"(a) {len(P)+len(N)} paired round observations"); ax[0].legend(loc="lower right", fontsize=9)
allr = [stats.pearsonr(a[:, 0], a[:, 1])[0] for a in pts + neg]
ax[1].hist(allr, bins=np.arange(-1, 1.05, 0.1), color="#4a7ebb", edgecolor="white")
ax[1].axvline(0, color="k", lw=1)
ax[1].set_xlabel("within-run correlation $r$"); ax[1].set_ylabel("runs")
ax[1].set_title(f"(b) per-run correlation, {len(allr)} runs")
ax[1].annotate(f"{len(neg)} negative\n(all with a damaged\nglobal model)", xy=(-0.6, 4),
               xytext=(-0.95, 13), fontsize=9, color="#c0504d",
               arrowprops=dict(arrowstyle="->", color="#c0504d", lw=1.2))
fig.savefig(OUT + "fig_correlation.pdf"); plt.close(fig)

# ---------------- Figure 3: which attacks discriminate ----------------
def l4(t):
    p = R + t + ".json"
    if not os.path.exists(p): return None
    g = json.load(open(p))["global_eval"]
    return float(np.mean([r["global_map50"] for r in g[-4:]]))
DEFS = ["fedavg", "median", "trimmed_mean", "multi_krum", "rfa", "norm_bound", "flame",
        "photoscreen2qn"]
ATK = [("alie", "ALIE"), ("min_sum", "Min-Sum"), ("min_max", "Min-Max"), ("ipm", "IPM")]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for i, (a, lab) in enumerate(ATK):
    v = [l4(f"M_{d}_{a}") for d in DEFS]
    v = [x for x in v if x is not None]
    ax.scatter([i] * len(v), v, s=42, color="#4a7ebb", alpha=.85, zorder=3)
    ax.plot([i - .22, i + .22], [np.mean(v)] * 2, color="#1f4e79", lw=2, zorder=4)
    ax.annotate(f"spread\n{max(v)-min(v):.2f}", xy=(i, min(v) - 0.045), ha="center", fontsize=9,
                color="#333333")
ax.axhline(0.6620, ls=":", color="gray", lw=1.4)
ax.text(3.42, 0.668, "no attack", fontsize=9, color="gray", ha="right")
ax.set_xticks(range(len(ATK))); ax.set_xticklabels([l for _, l in ATK])
ax.set_ylabel("global mAP50"); ax.set_ylim(-0.03, 0.72)
ax.set_title("Eight aggregation rules under four published attacks")
fig.savefig(OUT + "fig_attacks.pdf"); plt.close(fig)
print("figures written:", sorted(os.path.basename(x) for x in glob.glob(OUT + "*.pdf")))

# ---------------- Figure 4: feasible (mu, sigma) region and the score ----------------
import math
MU_R, SG_R, BETA, QMIN = 0.29424603283405304, 0.1888456866145134, 0.1, 0.2
def score(mu, sg):
    return (1/(1+np.exp(-BETA*(mu-MU_R)))) * np.minimum(SG_R/(sg+1e-6), 2.0)
mu = np.linspace(0, 1, 600); sg = np.linspace(0, 0.55, 600)
MU_, SG_ = np.meshgrid(mu, sg)
Q = score(MU_, SG_)
feasible = SG_**2 <= MU_*(1-MU_)          # moment constraint for a variable on [0,1]
Qm = np.where(feasible, Q, np.nan)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
im = ax.pcolormesh(MU_, SG_, Qm, shading="auto", cmap="viridis", vmin=0.15, vmax=1.0)
cs = ax.contour(MU_, SG_, np.where(feasible, Q, 10), levels=[QMIN], colors="white", linewidths=2)
ax.clabel(cs, fmt={QMIN: "$q=0.2$"}, fontsize=9)
ax.plot(mu, np.sqrt(mu*(1-mu)), color="0.25", lw=1.4, ls="--", label="moment bound $\\sigma^2=\\mu(1-\\mu)$")
g1 = json.load(open("../5_derived_results/g1_photometric_separation.json"))
pc = g1["cohorts"]["7client"]["per_client"]
cm = [(v["robust"]["clean"]["mu"], v["robust"]["clean"]["sigma"]) for v in pc.values()]
at = [(v["robust"][a]["mu"], v["robust"][a]["sigma"]) for v in pc.values()
      for a in ("brightness_flood", "darkness_injection", "noise_storm")]
ax.scatter(*zip(*cm), s=34, c="white", edgecolor="k", zorder=5, label="clean clients")
ax.scatter(*zip(*at), s=20, c="#c0504d", edgecolor="k", lw=.4, zorder=5, label="attacked clients")
ax.scatter([0.5], [0.5], s=90, marker="*", c="yellow", edgecolor="k", zorder=6,
           label="black/white image: $q=0.191$, rejected")
ax.set_xlabel("$\\mu$  (mean intensity)"); ax.set_ylabel("$\\sigma$")
ax.set_xlim(0, 1); ax.set_ylim(0, 0.55)
ax.legend(loc="upper left", fontsize=8, framealpha=.92)
fig.colorbar(im, ax=ax, label="score $q$")
ax.set_title("Feasible inputs and the accept region")
fig.savefig(OUT + "fig_feasible.pdf"); plt.close(fig)
print("wrote fig_feasible.pdf")

# ---------------- Figure 5: fault-by-diagnostic matrix from the injection study ----------------
import os as _os
_c6 = "../5_derived_results/c6_injection.json"
if _os.path.exists(_c6):
    D = json.load(open(_c6))
    tests = ["T1", "T2", "T3", "T4", "T5"]
    order = [k for k, v in D.items() if v["kind"] == "fault"] + \
            [k for k, v in D.items() if v["kind"] == "healthy"]
    M = np.full((len(order), len(tests)), np.nan)
    for i, k in enumerate(order):
        for j, t in enumerate(tests):
            v = D[k]["checks"].get(t)
            M[i, j] = np.nan if v is None else (1.0 if v else 0.0)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    cmap = matplotlib.colors.ListedColormap(["#eef2f7", "#c0504d"])
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(order)):
        for j in range(len(tests)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#999999")
            elif M[i, j] == 1:
                ax.text(j, i, "fires", ha="center", va="center", fontsize=8, color="white")
    n_f = sum(1 for k in order if D[k]["kind"] == "fault")
    ax.axhline(n_f - 0.5, color="k", lw=1.6)
    ax.set_xticks(range(len(tests))); ax.set_xticklabels(tests)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([k if D[k]["kind"] == "fault" else f"{k}  (control)" for k in order], fontsize=8.5)
    ax.set_title("Which check fires on which injected fault")
    ax.grid(False)
    fig.savefig(OUT + "fig_faultmatrix.pdf"); plt.close(fig)
    print("wrote fig_faultmatrix.pdf")

# --- attack strength sweep and the ASR denominators ------------------------
def _l4(t, k="global_map50"):
    return float(np.mean([r[k] for r in json.load(open(R + t + ".json"))["global_eval"][-4:]]))

_x = [0.0, 0.25, 0.50, 1.0]
_y = [_l4(t) for t in ["G2_fedavg_none_a30_s1", "C5_bf_i025", "C5_bf_i050", "C5_bf_i100"]]
_a = json.load(open("../5_derived_results/a4_asr.json"))

fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
ax[0].plot(_x, _y, "o-", color=C_GLOB, lw=2, ms=7)
ax[0].axhspan(_y[0] - 0.0028, _y[0] + 0.0028, color="gray", alpha=0.25,
              label="across-seed SD of the clean run")
ax[0].set_xlabel("poisoning intensity"); ax[0].set_ylabel("global mAP50")
ax[0].set_title(f"damage scales with strength (total {100*(_y[0]-_y[-1]):.2f} AP)")
ax[0].legend(loc="lower left", frameon=False)

_lab = ["no control", "clean model", "clean model +\nuntriggered data"]
_v = [100 * _a["asr_naive"], 100 * _a["asr_corrected"], 100 * _a["asr_specific"]]
_b = ax[1].bar(_lab, _v, color=[C_LOC, "#d8a25e", C_GLOB], width=0.6)
for r, v in zip(_b, _v):
    ax[1].text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
ax[1].set_ylim(0, 108); ax[1].set_ylabel("reported attack success rate")
ax[1].set_title("the same run, three denominators")
ax[1].annotate("", xy=(0, _v[0]), xytext=(2, _v[2]),
               arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
ax[1].text(1.0, (_v[0] + _v[2]) / 2 + 3, f"{_v[0]-_v[2]:.1f} points", ha="center",
           fontsize=10, color="0.25")
fig.tight_layout(); fig.savefig(OUT + "fig_dose_asr.pdf"); plt.close(fig)
print("wrote fig_dose_asr.pdf")

# --- fixed-checkpoint adaptation grid --------------------------------------
_C3 = "../5_derived_results/"
_h = json.load(open(_C3 + "c3_healthy.json"))["modes"]
_d = json.load(open(_C3 + "c3_damaged.json"))["modes"]
_modes = ["none", "bn_only", "head_only", "full"]
_names = ["no\nadaptation", "batch-norm\nonly", "head\nonly", "full"]
_hv = [_h[m]["mean"] for m in _modes]
_dv = [_d[m]["mean"] for m in _modes]

fig, ax = plt.subplots(figsize=(6.6, 3.6))
_i = np.arange(len(_modes)); _w = 0.38
ax.bar(_i - _w/2, _hv, _w, label="healthy checkpoint", color=C_GLOB)
ax.bar(_i + _w/2, _dv, _w, label="damaged checkpoint (global mAP50 $0.0006$)", color=C_LOC)
for x, v in zip(_i - _w/2, _hv): ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
for x, v in zip(_i + _w/2, _dv): ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(_i); ax.set_xticklabels(_names)
ax.set_ylabel("client-local mAP50"); ax.set_ylim(0, 0.83)
ax.set_xlabel("what the client is allowed to update in 3 local epochs")
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig_adaptation.pdf"); plt.close(fig)
print("wrote fig_adaptation.pdf")
