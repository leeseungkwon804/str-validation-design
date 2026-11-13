"""Figure: the learning curve. Does market hold-out performance recover as markets are added? It does not.
(a) number of training markets increasing, with training rows fixed and with all rows of those markets, evaluated on 25 held-out markets
(b) the same result against the number of training rows, beside the random-split control
Palette: a set that passes the light-mode colour validator."""
import os, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./out"; FIG = "./figs"; os.makedirs(FIG, exist_ok=True)
OUTCOME = sys.argv[1] if len(sys.argv) > 1 else "y_license"
R = pd.read_csv(f"{OUT}/E_learning_{OUTCOME}.csv")
A = pd.read_csv(f"{OUT}/A_summary_{OUTCOME}.csv")
rnd = float(A[(A.design == "1 Random (conventional)") & (A.model == "GBDT") & (A.training == "Full fold")].AUC_full.iloc[0])

plt.rcParams.update({
 "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
 "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5,
 "ytick.labelsize": 7.5, "legend.fontsize": 7.3, "axes.linewidth": 0.6, "figure.dpi": 200,
 "savefig.dpi": 400, "axes.spines.top": False, "axes.spines.right": False,
 "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "grid.color": "#b8b7b0"})
INK, INK2 = "#0b0b0b", "#52514e"
CA, CB, CC = "#1baf7a", "#eb6834", "#2a78d6"
fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.5))


# --- no explanatory block is drawn outside the axes (journal style); the sentences go into the caption instead ---
def _save_note(key, text):
    import json as _j, os as _o
    _p = "./out/fig_notes.json"
    _d = _j.load(open(_p)) if _o.path.exists(_p) else {}
    _d[key] = " ".join(str(text).split())
    _j.dump(_d, open(_p, "w"), indent=1, ensure_ascii=False)
# ---------------- (a) number of markets ----------------
ax = axes[0]
ax.axvspan(20, 110, color="#b8b7b0", alpha=0.12, lw=0, zorder=1)
ax.axhline(rnd, color=CC, lw=1.1, ls="--", zorder=2)
ax.annotate(f"Random split, all markets in training ({rnd:.3f})",
            (2.05, rnd + 0.005), fontsize=6.7, color=CC, va="bottom", ha="left")
tb = R[R["curve"] == "Markets added, all rows of the sampled markets"].groupby("n_training_markets").AUC.agg(["mean", "min", "max"]).reset_index()
ta = R[R["curve"] == "Markets added, training rows fixed at 800,000"].groupby("n_training_markets").AUC.agg(["mean"]).reset_index()
ax.fill_between(tb.n_training_markets, tb["min"], tb["max"], color=CB, alpha=0.15, lw=0, zorder=2)
ax.plot(tb.n_training_markets, tb["mean"], marker="s", ms=5.2, lw=2.0, color=CB, mec="#fcfcfb", mew=0.7,
        label="All rows of the sampled markets", zorder=3)
ax.plot(ta.n_training_markets, ta["mean"], marker="o", ms=3.4, lw=1.1, ls=(0, (3, 2)), color=CA,
        mec="#fcfcfb", mew=0.6, label="Training rows held fixed at 800,000", zorder=4)
end = float(tb[tb.n_training_markets == 98]["mean"].iloc[0])
ax.annotate(f"{end:.3f}", (98, end), xytext=(7, -12), textcoords="offset points",
            fontsize=8.2, fontweight="bold", color=CB, ha="left")
ax.annotate("", xy=(104, rnd), xytext=(104, end),
            arrowprops=dict(arrowstyle="<->", lw=0.9, color=INK2, shrinkA=0, shrinkB=0))
ax.annotate(f"{100*(rnd - end):.1f} AUC points\nstill missing with\nevery market used",
            (94, 0.780), fontsize=7.0, color=INK2, va="center", ha="right")
ax.annotate("plateau region", (21, 0.615), fontsize=6.8, color=INK2, ha="left", va="center")
ax.annotate("The two curves coincide: what moves the curve is the number of\nmarkets, not the number of rows they contribute.",
            (0.015, 0.035), xycoords="axes fraction", fontsize=6.8, color=INK2, va="bottom")
ax.set_xscale("log")
ax.set_xlim(1.8, 155)
ax.set_xticks([2, 5, 10, 20, 40, 60, 98]); ax.set_xticklabels([2, 5, 10, 20, 40, 60, 98])
ax.set_xlabel("Number of training markets (log scale)")
ax.set_ylabel("AUC on the same 25 held-out markets")
ax.set_ylim(0.50, 0.885)
ax.grid(axis="x", visible=False)
ax.set_title("(a) AUC by number of training markets", loc="center", color=INK)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=6.9, ncol=1)

# ---------------- (b) number of rows ----------------
ax = axes[1]
t = R[R["curve"] == "Random split control, training rows added"].sort_values("n_train")
ax.plot(t.n_train, t.AUC, marker="^", ms=5.0, lw=2.0, color=CC, mec="#fcfcfb", mew=0.7,
        label="Random split (test markets seen in training)", zorder=3)
t2 = (R[R["curve"] == "Markets added, all rows of the sampled markets"].groupby("n_training_markets")
        .agg(n_train=("n_train", "mean"), AUC=("AUC", "mean"), lo=("AUC", "min"),
             hi=("AUC", "max")).reset_index().sort_values("n_train"))
ax.fill_between(t2.n_train, t2.lo, t2.hi, color=CB, alpha=0.15, lw=0, zorder=2)
ax.plot(t2.n_train, t2.AUC, marker="s", ms=5.0, lw=2.0, color=CB, mec="#fcfcfb", mew=0.7,
        label="Market hold-out (test markets never seen)", zorder=3)
hi = t.iloc[-1]; lo = t2.iloc[-1]
ax.annotate(f"{hi.AUC:.3f}", (hi.n_train, hi.AUC), xytext=(7, 1), textcoords="offset points",
            fontsize=8.2, fontweight="bold", color=CC, ha="left", va="center")
ax.annotate(f"{lo.AUC:.3f}", (lo.n_train, lo.AUC), xytext=(7, 1), textcoords="offset points",
            fontsize=8.2, fontweight="bold", color=CB, ha="left", va="center")
ax.set_xlim(2e4, 2.0e7)
ax.annotate("Both designs end on the full data set, and the gap is the design, not the sample size.",
            (0.015, 0.998), xycoords="axes fraction", fontsize=6.7, color=INK2, va="top")
ax.set_xscale("log"); ax.set_xlabel("Training rows (log scale)")
ax.set_ylabel("AUC")
ax.set_ylim(0.50, 0.885)
ax.grid(axis="x", visible=False)
ax.set_title("(b) The same curves by training rows", loc="center", color=INK)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=7.0, ncol=1)
fig.tight_layout()
note = ("Gradient boosting, outcome: " + ("loose licence definition" if OUTCOME == "y_license" else "strict licence definition")
        + ". Panel (a): 5 random market draws per point (1 at K=98, the whole pool); band spans the minimum and maximum draw. "
        "Evaluation is always the same 150,000 rows from 25 markets withheld from every fit.")
_save_note("F3_learning_curve", note)
fig.savefig(f"{FIG}/F3_learning_curve.png", bbox_inches="tight")
fig.savefig(f"{FIG}/F3_learning_curve.pdf", bbox_inches="tight")
print("saved:", f"{FIG}/F3_learning_curve.png", f"| random-split reference {rnd:.4f} | K=98 all rows {t2.AUC.iloc[-1]:.4f}")
