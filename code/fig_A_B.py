"""Figure: how much the validation design inflates performance (model zoo and market-level bootstrap).
(a) AUC by design and model (three models trained on the full fold, fold mean plus or minus one fold SD)
(b) loss against the random split with market-level bootstrap 95% intervals, folds pooled over the 123 markets common to all designs
(c) all four models compared on the same 10,000 training rows, TabPFN included
No model receives any market-identity variable. The palette passes the light-mode colour validator."""
import os, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./out"; FIG = "./figs"; os.makedirs(FIG, exist_ok=True)
OUTCOME = sys.argv[1] if len(sys.argv) > 1 else "y_license"
A = pd.read_csv(f"{OUT}/A_summary_{OUTCOME}.csv")
B = pd.read_csv(f"{OUT}/B_bootstrap_{OUTCOME}.csv")
plt.rcParams.update({
 "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
 "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5,
 "ytick.labelsize": 7.5, "legend.fontsize": 7.3, "axes.linewidth": 0.6, "figure.dpi": 200,
 "savefig.dpi": 400, "axes.spines.top": False, "axes.spines.right": False,
 "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "grid.color": "#b8b7b0"})

# --- no explanatory block is drawn outside the axes (journal style); the sentences go into the caption instead ---
def _save_note(key, text):
    import json as _j, os as _o
    _p = "./out/fig_notes.json"
    _d = _j.load(open(_p)) if _o.path.exists(_p) else {}
    _d[key] = " ".join(str(text).split())
    _j.dump(_d, open(_p, "w"), indent=1, ensure_ascii=False)
INK, INK2 = "#0b0b0b", "#52514e"
CL = {"Logistic regression": "#2a78d6", "Random forest": "#eb6834", "GBDT": "#1baf7a", "TabPFN": "#6250d6"}
MK = {"Logistic regression": "o", "Random forest": "s", "GBDT": "^", "TabPFN": "D"}
EN = {"Logistic regression": "Logistic regression", "Random forest": "Random forest",
      "GBDT": "Gradient boosting", "TabPFN": "TabPFN"}
ORD = ["1 Random (conventional)", "4 Host hold-out", "3 Temporal (pre to post)", "2 Market hold-out", "5 Market by time"]
LBL = {"1 Random (conventional)": "1 Random\n(conventional)", "4 Host hold-out": "4 Host\nhold-out",
       "3 Temporal (pre to post)": "3 Temporal\n(pre→post)", "2 Market hold-out": "2 Market\nhold-out",
       "5 Market by time": "5 Market\n× time"}
xs = np.arange(len(ORD))
OC_EN = "loose licence definition" if OUTCOME == "y_license" else "strict licence definition"

fig = plt.figure(figsize=(7.2, 7.6))
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.0, 0.95], hspace=0.62)

# ---------------- (a) full-fold training ----------------
ax = fig.add_subplot(gs[0])
sub = A[A["training"] == "Full fold"]
# label only the two designs the paper contrasts; the rest are in panel (b) and in the tables
LABEL_AT = {0: {"Random forest": (0, 10), "GBDT": (0, -14), "Logistic regression": (0, 10)},
            3: {"GBDT": (13, 5), "Random forest": (13, -17), "Logistic regression": (13, 0)}}
for mdl in ["Logistic regression", "Random forest", "GBDT"]:
    t = sub[sub["model"] == mdl].set_index("design").reindex(ORD)
    sd = t.AUC_full_sd.fillna(0).values
    ax.errorbar(xs, t.AUC_full.values, yerr=sd, fmt="none", ecolor=CL[mdl],
                elinewidth=0.9, capsize=2.4, zorder=3)
    ax.plot(xs, t.AUC_full, marker=MK[mdl], ms=5.4, lw=2.0, color=CL[mdl], label=EN[mdl],
            mec="#fcfcfb", mew=0.7, zorder=4)
    for x, v in zip(xs, t.AUC_full):
        if x in LABEL_AT and mdl in LABEL_AT[x] and np.isfinite(v):
            dx, dy = LABEL_AT[x][mdl]
            ax.annotate(f"{v:.3f}", (x, v), xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx else "center", va="center" if dx else "baseline",
                        fontsize=6.9, color=CL[mdl])
ax.set_xticks(xs); ax.set_xticklabels([LBL[s] for s in ORD])
ax.set_ylim(0.555, 0.905); ax.set_xlim(-0.45, 4.45)
ax.set_ylabel("AUC (trained on the full fold)")
ax.set_title("(a) AUC by validation design and model (full-fold training)",
             loc="center", color=INK)
ax.grid(axis="x", visible=False)

# ---------------- (b) gap with confidence interval ----------------
ax = fig.add_subplot(gs[1])
w = 0.26
for i, mdl in enumerate(["Logistic regression", "Random forest", "GBDT"]):
    t = B[B["model"] == f"{mdl}_Full fold"].set_index("design").reindex(ORD)
    g = -t["loss_vs_random_split"].values
    lo = -t["gap_hi95"].values; hi = -t["gap_lo95"].values
    ax.bar(xs + (i - 1) * w, g, width=w * 0.9, color=CL[mdl], label=EN[mdl], zorder=3)
    ax.errorbar(xs + (i - 1) * w, g, yerr=[g - lo, hi - g], fmt="none",
                ecolor=INK, elinewidth=0.85, capsize=2.2, zorder=4)
gb = B[B["model"] == "Gradient boosting, full fold"].set_index("design")
for s, x in zip(ORD, xs):
    if s == "1 Random (conventional)": continue
    v = -float(gb.loc[s, "loss_vs_random_split"])
    ax.annotate(f"{v:+.3f}", (x + w, v), xytext=(10, 0), textcoords="offset points",
                fontsize=6.8, color=CL["GBDT"], va="center", ha="left")
ax.axhline(0, color=INK2, lw=0.9)
ax.set_xticks(xs); ax.set_xticklabels([LBL[s] for s in ORD])
ax.set_ylim(-0.30, 0.012); ax.set_xlim(-0.55, 4.72)
ax.set_ylabel("AUC change vs\nthe random split")
ax.set_title("(b) Loss against the random split, with market-resampled 95% intervals",
             loc="center", color=INK)
ax.grid(axis="x", visible=False)

# ---------------- (c) sample-matched comparison ----------------
ax = fig.add_subplot(gs[2])
sub = A[A["training"] == "10,000 rows"]
for mdl in ["Logistic regression", "Random forest", "GBDT", "TabPFN"]:
    t = sub[sub["model"] == mdl].set_index("design").reindex(ORD)
    sd = t.AUC_sd.fillna(0).values
    ax.errorbar(xs, t.AUC.values, yerr=sd, fmt="none", ecolor=CL[mdl], elinewidth=0.9,
                capsize=2.4, zorder=3)
    ax.plot(xs, t.AUC, marker=MK[mdl], ms=5.4, lw=2.0, color=CL[mdl], label=EN[mdl],
            mec="#fcfcfb", mew=0.7, zorder=4)
ax.set_xticks(xs); ax.set_xticklabels([LBL[s] for s in ORD])
ax.set_ylim(0.515, 0.845); ax.set_xlim(-0.45, 4.45)
ax.set_ylabel("AUC (10,000 training rows)")
ax.set_title("(c) Sample-matched comparison at 10,000 training rows (TabPFN included)",
             loc="center", color=INK)
ax.grid(axis="x", visible=False)

# one legend outside the axes, above the figure, since all three panels use the same model colours
from matplotlib.lines import Line2D
_H = [Line2D([], [], color=CL[m], marker=MK[m], ms=5.4, lw=2.0, mec="#fcfcfb", mew=0.7, label=EN[m])
      for m in ["Logistic regression", "Random forest", "GBDT", "TabPFN"]]
fig.legend(handles=_H, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.962),
           handlelength=2.0, columnspacing=1.8)

_save_note("F1_validation_design",
    f"Outcome: {OC_EN}. Panels (a) and (c): mean over folds (5 folds, 1 for the temporal split); bars are "
    "±1 SD across folds. Panel (b): folds pooled into one evaluation set restricted to the 123 markets common "
    "to all designs, resampling markets (1,000 draws) so that listings of the same market move together; the "
    "pooled market-hold-out estimate (0.740) is slightly below the fold mean in (a) because pooling weights "
    "markets by their number of rows. No model receives any market identifier.")
fig.savefig(f"{FIG}/F1_validation_design.png", bbox_inches="tight")
fig.savefig(f"{FIG}/F1_validation_design.pdf", bbox_inches="tight")
print("saved:", f"{FIG}/F1_validation_design.png")
