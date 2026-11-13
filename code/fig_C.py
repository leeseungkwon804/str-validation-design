"""Figure: the ablation experiment. The leakage is not in one variable; it is spread across groups.
(a) how far the gap narrows when one group is removed; none of them closes it
(b) each group alone; every group reaches well above chance on its own, so the signal is redundant
Neither panel uses any market-identity variable. The palette passes the light-mode colour validator."""
import os, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./out"; FIG = "./figs"; os.makedirs(FIG, exist_ok=True)
C3 = pd.read_csv(f"{OUT}/C3_ablation.csv")
plt.rcParams.update({
 "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
 "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.3,
 "ytick.labelsize": 7.3, "legend.fontsize": 7.2, "axes.linewidth": 0.6, "figure.dpi": 200,
 "savefig.dpi": 400, "axes.spines.top": False, "axes.spines.right": False,
 "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "grid.color": "#b8b7b0"})
INK, INK2 = "#0b0b0b", "#52514e"
C_RAND, C_MKT, C_GAP, C_UP = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
EN = {"Price": "Price", "Minimum and maximum stay": "Minimum / maximum nights", "Property type": "Property and room type",
      "Availability": "Availability calendar", "Reviews": "Reviews", "Host portfolio size": "Host portfolio size",
      "Host conduct": "Host behaviour", "Text length": "Text lengths", "Tenure": "Tenure (days)"}

b = C3[C3["setting"] == "All 50 predictors (baseline)"].iloc[0]
drop = C3[C3["setting"].str.endswith("removed")].copy()
drop["grp"] = drop["setting"].str.replace("− ", "", regex=False).str.replace(" removed", "", regex=False)
solo = C3[C3["setting"].str.endswith("alone")].copy()
solo["grp"] = solo["setting"].str.replace(" alone", "", regex=False)
order = drop.sort_values("gap_change")["grp"].tolist()          # the group that narrows the gap most is at the bottom
d = drop.set_index("grp").reindex(order)
s = solo.set_index("grp").reindex(order)


# --- no explanatory block is drawn outside the axes (journal style); the sentences go into the caption instead ---
def _save_note(key, text):
    import json as _j, os as _o
    _p = "./out/fig_notes.json"
    _d = _j.load(open(_p)) if _o.path.exists(_p) else {}
    _d[key] = " ".join(str(text).split())
    _j.dump(_d, open(_p, "w"), indent=1, ensure_ascii=False)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3), gridspec_kw={"width_ratios": [1, 1.15]})
ys = np.arange(len(order))

# ---- (a) group removed -> change in the gap ----
ax = axes[0]
vals = d["gap_change"].values
cols = [C_GAP if v < 0 else C_UP for v in vals]
ax.barh(ys, vals, height=0.62, color=cols, zorder=3)
ax.axvline(0, color=INK2, lw=0.9)
for i, v in enumerate(vals):
    ax.annotate(f"{v:+.4f}", (v, i), xytext=(5 if v > 0 else -5, 0), textcoords="offset points",
                va="center", ha="left" if v > 0 else "right", fontsize=7.0, color=INK)
ax.set_yticks(ys); ax.set_yticklabels([EN.get(g, g) for g in order])
ax.set_xlim(-0.0135, 0.0075)
ax.set_xlabel("Change in the random-vs-market-hold-out AUC gap")
ax.grid(axis="y", visible=False)
ax.set_title("(a) Change in the gap by group removed", loc="center", color=INK)
ax.annotate(f"Gap with all 50 predictors: {b.gap:.4f}\nThe largest reduction, {vals.min():+.4f}, still\n"
            f"leaves a gap of {b.gap + vals.min():.4f}. Dropping price\nmakes the gap wider, not narrower.",
            (0.03, 0.97), xycoords="axes fraction", fontsize=6.9, color=INK2, va="top")

# ---- (b) each group alone ----
ax = axes[1]
h = 0.36
ax.barh(ys + h / 2, s.AUC_random_split.values, height=h, color=C_RAND, zorder=3,
        label="Random split")
ax.barh(ys - h / 2, s.AUC_market_holdout.values, height=h, color=C_MKT, zorder=3,
        hatch="///", edgecolor="#fcfcfb", linewidth=0.6, label="Market hold-out")
for i, (a1, a2) in enumerate(zip(s.AUC_random_split.values, s.AUC_market_holdout.values)):
    ax.annotate(f"{a1:.3f}", (a1, i + h / 2), xytext=(3, 0), textcoords="offset points",
                va="center", fontsize=6.6, color=INK)
    ax.annotate(f"{a2:.3f}", (a2, i - h / 2), xytext=(3, 0), textcoords="offset points",
                va="center", fontsize=6.6, color=INK)
ax.axvline(0.5, color=INK2, lw=0.9, ls="--")
ax.annotate("chance", (0.5, -0.85), xytext=(3, 0), textcoords="offset points",
            fontsize=6.6, color=INK2, va="center")
ax.axvline(b.AUC_random_split, color=C_RAND, lw=1.0, ls=":")
ax.axvline(b.AUC_market_holdout, color=C_MKT, lw=1.0, ls=":")
ax.annotate(f"all 50 together: {b.AUC_random_split:.3f}", (b.AUC_random_split, -0.85), xytext=(-4, 0),
            textcoords="offset points", ha="right", va="center", fontsize=6.6, color=C_RAND)
ax.annotate(f"all 50 together: {b.AUC_market_holdout:.3f}", (b.AUC_market_holdout, -1.40), xytext=(-4, 0),
            textcoords="offset points", ha="right", va="center", fontsize=6.6, color=C_MKT)
ax.set_yticks(ys); ax.set_yticklabels([])
ax.set_xlim(0.46, 0.90); ax.set_ylim(-1.8, len(order) - 0.35)
ax.set_xlabel("AUC using that group alone")
ax.grid(axis="y", visible=False)
ax.set_title("(b) AUC from each group alone", loc="center", color=INK)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

fig.tight_layout()
_save_note("F7_feature_ablation",
    "Gradient boosting, loose licence definition, 5 folds per cell; the same fold definitions as the main "
    "benchmark. Groups are ordered by panel (a). The gap is the random-split AUC minus the market-hold-out AUC.")
fig.savefig(f"{FIG}/F7_feature_ablation.png", bbox_inches="tight")
fig.savefig(f"{FIG}/F7_feature_ablation.pdf", bbox_inches="tight")
print("saved:", f"{FIG}/F7_feature_ablation.png", "| baseline gap", b.gap, "| largest reduction", vals.min())
