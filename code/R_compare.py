"""R. Parallel table for the two outcome definitions: loose (y_license, primary) against strict (y_strict).
Places the two runs side by side: same five designs, same four models, same metrics, same evaluation samples.
The three headline quantities are also compared under both definitions.
Outputs: out/R_two_definitions.csv, out/R_two_definitions_core.csv, out/R_two_definitions.json
      figs/F5_two_definitions.{png,pdf}"""
import os, sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./out"; FIG = "./figs"; os.makedirs(FIG, exist_ok=True)
ORD = ["1 Random (conventional)", "4 Host hold-out", "3 Temporal (pre to post)", "2 Market hold-out", "5 Market by time"]
LBL = {"1 Random (conventional)": "1 Random\n(conventional)", "4 Host hold-out": "4 Host\nhold-out",
       "3 Temporal (pre to post)": "3 Temporal\n(pre→post)", "2 Market hold-out": "2 Market\nhold-out",
       "5 Market by time": "5 Market\n× time"}
EN = {"Logistic regression": "Logistic regression", "Random forest": "Random forest",
      "GBDT": "Gradient boosting", "TabPFN": "TabPFN"}
need = [f"{OUT}/A_folds_y_license.csv", f"{OUT}/A_folds_y_strict.csv"]
missing = [p for p in need if not os.path.exists(p)]
if missing:
    print("inputs not yet present:", missing); sys.exit(1)

rows = []
for oc, nm in (("y_license", "Loose"), ("y_strict", "Strict")):
    a = pd.read_csv(f"{OUT}/A_folds_{oc}.csv")
    g = (a[a["training"] == "Full fold"].groupby(["design", "model"])
           .agg(folds=("folds", "size"), AUC=("AUC_full", "mean"), AUC_sd=("AUC_full", "std"),
                PRAUC=("PRAUC_full", "mean"), Brier=("Brier_full", "mean")).reset_index())
    g["outcome"] = nm; g["training"] = "Full fold"; rows.append(g)
    s = (a[a["training"] == "10,000 rows"].groupby(["design", "model"])
           .agg(folds=("folds", "size"), AUC=("AUC_eval", "mean"), AUC_sd=("AUC_eval", "std"),
                PRAUC=("PRAUC_eval", "mean"), Brier=("Brier_eval", "mean")).reset_index())
    s["outcome"] = nm; s["training"] = "10,000 rows"; rows.append(s)
T = pd.concat(rows, ignore_index=True).round(4)
T["design_order"] = T["design"].map({s: i for i, s in enumerate(ORD)})
T = T.sort_values(["training", "model", "design_order"]).drop(columns="design_order")
T.to_csv(f"{OUT}/R_two_definitions.csv", index=False)

# loss against the random split
gaps = []
for oc in T.outcome.unique():
    for lv in T.training.unique():
        for m in T.model.unique():
            t = T[(T.outcome == oc) & (T.training == lv) & (T.model == m)].set_index("design")
            if "1 Random (conventional)" not in t.index: continue
            ref = t.loc["1 Random (conventional)", "AUC"]
            for s in t.index:
                gaps.append({"outcome": oc, "training": lv, "model": m, "design": s,
                             "AUC": float(t.loc[s, "AUC"]),
                             "loss_vs_random_split": round(float(ref - t.loc[s, "AUC"]), 4)})
G = pd.DataFrame(gaps)
G.to_csv(f"{OUT}/R_two_definitions_gaps.csv", index=False)

# the three headline quantities
core = []
for oc, nm in (("y_license", "Loose"), ("y_strict", "Strict")):
    c2 = pd.read_csv(f"{OUT}/C2_baserate_proxy.csv")
    c2 = c2[c2.outcome == oc]
    a = pd.read_csv(f"{OUT}/A_folds_{oc}.csv")
    a = a[(a["training"] == "Full fold")]
    for split in ("1 Random (conventional)", "2 Market hold-out"):
        cc = c2[c2.design == split]
        if cc.empty: continue
        core.append({"outcome": nm, "design": split,
                     "true_market_baserate": round(float(cc.AUC_true_market_baserate.mean()), 4),
                     "gradient_boosting_50_predictors": round(float(a[(a.design == split) & (a.model == "GBDT")].AUC_full.mean()), 4),
                     "random_forest_50_predictors": round(float(a[(a.design == split) & (a.model == "Random forest")].AUC_full.mean()), 4),
                     "recovered_market_baserate": round(float(cc.AUC_baserate_posterior.mean()), 4),
                     "market_recovery_accuracy": round(float(cc.market_recovery_accuracy.mean()), 4)})
C = pd.DataFrame(core)
C.to_csv(f"{OUT}/R_two_definitions_core.csv", index=False)
json.dump({"core": C.to_dict("records"), "gaps": G.to_dict("records")},
          open(f"{OUT}/R_two_definitions.json", "w"), indent=1, ensure_ascii=False)
pd.set_option("display.width", 230)
print("=== the three headline quantities, both definitions ===\n" + C.to_string(index=False))
print("\n=== full-fold AUC, both definitions ===")
print(T[T.training == "Full fold"].pivot_table(index=["model", "design"], columns="outcome",
                                      values="AUC").round(4).to_string())

# ---------------- figure ----------------
plt.rcParams.update({
 "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
 "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.4,
 "ytick.labelsize": 7.4, "legend.fontsize": 7.2, "axes.linewidth": 0.6, "figure.dpi": 200,
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
C_LOOSE, C_STRICT = "#6250d6", "#e87ba4"      # loose against strict; kept clear of the model colours
MDLS = ["Logistic regression", "Random forest", "GBDT"]
SHORT = {"1 Random (conventional)": "1 Random", "4 Host hold-out": "4 Host", "3 Temporal (pre to post)": "3 Temporal",
         "2 Market hold-out": "2 Market", "5 Market by time": "5 Mkt×time"}

fig = plt.figure(figsize=(7.4, 6.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.78, wspace=0.30)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[1, :])

# (a) the five designs, with the two outcome definitions overlaid
xs = np.arange(len(ORD))
for mdl in MDLS:
    for nm, ls, mfc in (("Loose", "-", None), ("Strict", "--", "white")):
        t = T[(T.outcome == nm) & (T.training == "Full fold") & (T.model == mdl)].set_index("design").reindex(ORD)
        axA.plot(xs, t.AUC, marker=MK[mdl], ms=4.2, lw=1.6, ls=ls, color=CL[mdl],
                 mfc=(CL[mdl] if mfc is None else mfc), mew=1.1, zorder=3)
axA.set_xticks(xs); axA.set_xticklabels([SHORT[s] for s in ORD], rotation=30, ha="right")
axA.set_ylabel("AUC (full training fold)")
axA.set_ylim(0.57, 0.90)
axA.grid(axis="x", visible=False)
axA.set_title("(a) The five designs under both\noutcome definitions", loc="center", color=INK)
from matplotlib.lines import Line2D
h1 = [Line2D([], [], color=CL[m], marker=MK[m], ms=4.2, lw=1.6,
             label=EN[m].replace(" regression", "").replace("Gradient boosting", "GBDT")) for m in MDLS]
h2 = [Line2D([], [], color=INK2, lw=1.6, ls="-", label="Loose (primary)"),
      Line2D([], [], color=INK2, lw=1.6, ls="--", label="Strict")]
axA.legend(handles=h1 + h2, frameon=False, loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, -0.33), handlelength=1.6, labelspacing=0.26,
           columnspacing=1.2, borderpad=0.1)

# (b) the gaps on a 1:1 line; every point lies on the diagonal
gl = G[(G.training == "Full fold") & (G.outcome == "Loose")].set_index(["model", "design"]).loss_vs_random_split
gsx = G[(G.training == "Full fold") & (G.outcome == "Strict")].set_index(["model", "design"]).loss_vs_random_split
pts = []
for mdl in MDLS:
    for s in ORD[1:]:
        if (mdl, s) in gl.index and (mdl, s) in gsx.index:
            pts.append((mdl, s, -float(gl.loc[(mdl, s)]), -float(gsx.loc[(mdl, s)])))
lim = (min(min(p[2] for p in pts), min(p[3] for p in pts)) - 0.012, 0.012)
axB.plot(lim, lim, color=INK2, lw=0.9, ls=":", zorder=2)
for mdl, s, a, b in pts:
    axB.scatter([a], [b], s=30, color=CL[mdl], marker=MK[mdl], zorder=3,
                edgecolors="white", linewidths=0.6)
for mdl, s, a, b in pts:
    if mdl == "GBDT":
        axB.annotate(SHORT[s], (a, b), xytext=(5, -8), textcoords="offset points",
                     fontsize=6.6, color=INK2)
mx = max(abs(p[3] - p[2]) for p in pts)
axB.set_xlim(*lim); axB.set_ylim(*lim)
axB.set_xlabel("AUC gap vs random, loose"); axB.set_ylabel("AUC gap vs random, strict")
axB.set_title(f"(b) Loss against the random split, loose versus\nstrict (maximum shift {100*mx:.1f} AUC points)",
              loc="center", color=INK)
axB.annotate("dotted line: identical\nunder both definitions", (0.97, 0.06), xycoords="axes fraction",
             fontsize=6.6, color=INK2, ha="right", va="bottom")

# (c) at market level, however, the two definitions diverge
DV = pd.read_csv(f"{OUT}/T1_divergent_markets.csv")
DV = DV[DV.Country != "Country"].copy()
for c in ("Loose", "Strict", "Difference (pp)", "Trader ID (DSA)", "Exemption text"):
    DV[c] = pd.to_numeric(DV[c], errors="coerce")
DV = DV.sort_values("Difference (pp)")
ypos = np.arange(len(DV))
axC.hlines(ypos, DV.Strict, DV.Loose, color="#c9c7c0", lw=2.4, zorder=2)
axC.scatter(DV.Strict, ypos, s=34, color=C_STRICT, zorder=3, label="Strict", edgecolors="white", linewidths=0.6)
axC.scatter(DV.Loose, ypos, s=34, color=C_LOOSE, zorder=3, label="Loose (primary)", edgecolors="white", linewidths=0.6)
for i, (_, r) in enumerate(DV.iterrows()):
    axC.annotate(f"{r['Difference (pp)']:.1f} pp", (max(r.Loose, r.Strict), i), xytext=(7, 0),
                 textcoords="offset points", fontsize=6.9, color=INK, va="center")
axC.set_yticks(ypos); axC.set_yticklabels([f"{m}" for m in DV.Market])
axC.set_xlim(15, 112); axC.set_xlabel("Listings with a licence value (%)")
axC.grid(axis="y", visible=False)
axC.set_title(f"(c) Market-level divergence of the two definitions "
              f"(up to {DV['Difference (pp)'].max():.1f} pp)", loc="center", color=INK)
axC.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.14))
_save_note("F5_two_definitions",
    "Panel (c): markets where the two definitions differ by more than 5 pp (11 of 123). The loose definition "
    "counts any non-empty `license` string; the strict one keeps only values that parse as a registration "
    "identifier. In Berlin, 29.57% of the values the loose definition adds are DSA trader-information strings, "
    "not licences.")
fig.savefig(f"{FIG}/F5_two_definitions.png", bbox_inches="tight")
fig.savefig(f"{FIG}/F5_two_definitions.pdf", bbox_inches="tight")
print("\nsaved:", f"{FIG}/F5_two_definitions.png")
print("largest shift in the loss:", round(100*mx, 2), " AUC points")
