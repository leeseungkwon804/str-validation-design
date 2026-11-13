"""Figure: what decides the outcome is not the listing's attributes but which market it is in.
(a) three quantities side by side under the random split: the true market's base rate, the 50-predictor model, and the recovered market's base rate
(b) the same three quantities under the market hold-out: with no test market in training, the base-rate ceiling cannot be formed at all
Palette: a set that passes the light-mode colour validator."""
import os, sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./out"; FIG = "./figs"; os.makedirs(FIG, exist_ok=True)
C2 = pd.read_csv(f"{OUT}/C2_baserate_proxy.csv")
# The C2 output holds both outcome definitions; only the primary (loose) one is used here.
C2 = C2[C2["outcome"] == "y_license"]
A = pd.read_csv(f"{OUT}/A_folds_y_license.csv")
A = A[A["training"] == "Full fold"]

plt.rcParams.update({
 "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
 "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.6,
 "ytick.labelsize": 7.6, "legend.fontsize": 7.2, "axes.linewidth": 0.6, "figure.dpi": 200,
 "savefig.dpi": 400, "axes.spines.top": False, "axes.spines.right": False,
 "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "grid.color": "#b8b7b0"})
INK, INK2 = "#0b0b0b", "#52514e"
NOTES = []
CT, CM, CR = "#eb6834", "#2a78d6", "#1baf7a"     # true market / model / recovered market


# --- no explanatory block is drawn outside the axes (journal style); the sentences go into the caption instead ---
def _save_note(key, text):
    import json as _j, os as _o
    _p = "./out/fig_notes.json"
    _d = _j.load(open(_p)) if _o.path.exists(_p) else {}
    _d[key] = " ".join(str(text).split())
    _j.dump(_d, open(_p, "w"), indent=1, ensure_ascii=False)
def panel(ax, split, title, note):
    c = C2[C2["design"] == split]
    g = A[A["design"] == split]
    gb = float(g[g["model"] == "GBDT"].AUC_eval.mean())
    best_name, best = max((("Random forest", float(g[g["model"] == "Random forest"].AUC_eval.mean())),
                           ("Gradient boosting", gb),
                           ("Logistic regression", float(g[g["model"] == "Logistic regression"].AUC_eval.mean()))),
                          key=lambda t: t[1])
    vals = [float(c.AUC_true_market_baserate.mean()), gb, float(c.AUC_baserate_posterior.mean())]
    labs = ["True market\nbase rate", "All 50\npredictors", "Recovered market\nbase rate"]
    cols = [CT, CM, CR]
    hats = ["", "xxx", "///"]
    xs = np.arange(3)
    undef = (split == "2 Market hold-out")
    fc = ["#f6d9cd" if undef else CT, CM, CR]
    ec = [CT if undef else "#fcfcfb", "#fcfcfb", "#fcfcfb"]
    hh = ["..." if undef else "", "xxx", "///"]
    ax.bar(xs, vals, width=0.58, color=fc, hatch=hh, edgecolor=ec, linewidth=1.0, zorder=3)
    for i, (x, v) in enumerate(zip(xs, vals)):
        if undef and i == 0:
            ax.annotate("not computable\n(no shared market)", (x, v), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=7.2, color=CT)
        else:
            ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9.8, fontweight="bold", color=INK)
    ax.axhline(0.5, color=INK2, lw=0.9, ls="--", zorder=2)
    ax.annotate("chance (0.500)", (2.45, 0.505), fontsize=6.8, color=INK2, va="bottom", ha="right")
    if best_name != "Gradient boosting":
        ax.axhline(best, color=INK2, lw=0.8, ls=":", zorder=2)
        note = f"Dotted line: best of the four models ({best_name.lower()} {best:.3f}).\n" + note
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7.2)
    ax.set_xlim(-0.62, 2.62)
    ax.set_ylim(0.40, 1.02); ax.set_ylabel("AUC")
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="center", color=INK)
    NOTES.append(" ".join(note.split()))

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.1))
r = C2[C2["design"] == "1 Random (conventional)"]
panel(axes[0], "1 Random (conventional)",
      "(a) Random split (conventional design)",
      "Uses market identity, the variable this study keeps out of the models.\n"
      f"Market recovered for {100*r.market_recovery_accuracy.mean():.0f}% of test rows; "
      "every test market also appears in training.")
m = C2[C2["design"] == "2 Market hold-out"]
if len(m):
    panel(axes[1], "2 Market hold-out",
          "(b) Market hold-out (test markets absent from training)",
          f"No test market appears in training ({100*m.test_market_in_training.mean():.0f}%), so the true "
          "market's base rate cannot be\nformed at all. The recovered-market proxy still reaches 0.677 by "
          "borrowing the base rates\nof the training markets a listing most resembles.")
fig.tight_layout()
_save_note("F4_baserate_ceiling", "(a) " + NOTES[0] + " (b) " + NOTES[1])
fig.savefig(f"{FIG}/F4_baserate_ceiling.png", bbox_inches="tight")
fig.savefig(f"{FIG}/F4_baserate_ceiling.pdf", bbox_inches="tight")
print("saved:", f"{FIG}/F4_baserate_ceiling.png")
