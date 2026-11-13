"""D. Calibration and decision curves: how far the probabilities themselves degrade when the design changes.
AUC measures ranking only. A decision rule uses the probability itself, so calibration and net benefit are reported separately.
Input: out/A_preds_<outcome>.parquet, the shared evaluation-sample predictions for each design
Outputs: out/D_calibration_<outcome>.csv, out/D_decision_<outcome>.csv, out/D_metrics_<outcome>.csv
      figs/D_calibration_decision.{png,pdf}"""
import os, sys, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

OUT = "./out"; FIG = "./figs"
os.makedirs(FIG, exist_ok=True)
OUTCOME = sys.argv[1] if len(sys.argv) > 1 else "y_license"
MODEL = os.environ.get("D_MODEL", "Gradient boosting, full fold")
MODEL_EN = {"Gradient boosting, full fold": "Gradient boosting, full training fold",
            "Random forest, full fold": "Random forest, full training fold",
            "Logistic regression, full fold": "Logistic regression, full training fold"}.get(MODEL, MODEL)
OUTCOME_EN = "loose licence definition" if OUTCOME == "y_license" else "strict licence definition"
NB = 20
T0 = time.time()
pred = pd.read_parquet(f"{OUT}/A_preds_{OUTCOME}.parquet")
SPLITS = list(dict.fromkeys(pred["design"]))
MODELS = [c for c in pred.columns if c not in ("y", "idx", "design", "folds")]
print(f"evaluation rows {len(pred):,}, designs {SPLITS}, models {MODELS}", flush=True)

def ece(yy, pp, nb=NB):
    e = np.quantile(pp, np.linspace(0, 1, nb + 1)); e[0] -= 1e-9
    b = np.clip(np.searchsorted(e, pp, side="left") - 1, 0, nb - 1)
    tot = 0.0
    for i in range(nb):
        m = b == i
        if m.sum(): tot += m.sum() / len(pp) * abs(yy[m].mean() - pp[m].mean())
    return float(tot)

def calib_table(yy, pp, nb=NB):
    e = np.quantile(pp, np.linspace(0, 1, nb + 1)); e[0] -= 1e-9
    b = np.clip(np.searchsorted(e, pp, side="left") - 1, 0, nb - 1)
    out = []
    for i in range(nb):
        m = b == i
        if m.sum() == 0: continue
        out.append({"bin": i, "n": int(m.sum()), "pred_mean": float(pp[m].mean()),
                    "obs_rate": float(yy[m].mean())})
    return pd.DataFrame(out)

def calib_slope_intercept(yy, pp):
    """Cox calibration: logistic regression of the outcome on logit(p); slope 1 and intercept 0 mean perfect calibration."""
    z = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000).fit(z, yy)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def net_benefit(yy, pp, pt):
    """decision-curve net benefit = TP/N - FP/N * pt/(1-pt)."""
    flag = pp >= pt
    tp = float((flag & (yy == 1)).sum()); fp = float((flag & (yy == 0)).sum())
    N = len(yy)
    return tp / N - fp / N * (pt / (1 - pt))

PTS = np.round(np.arange(0.05, 0.91, 0.025), 4)
cal, dec, met = [], [], []
for s in SPLITS:
    d = pred[pred["design"] == s]
    for m in MODELS:
        if d[m].isna().all(): continue
        yy = d.y.values.astype(np.int8); pp = np.clip(d[m].values.astype(np.float64), 0, 1)
        t = calib_table(yy, pp); t["design"] = s; t["model"] = m; cal.append(t)
        sl, ic = calib_slope_intercept(yy, pp)
        met.append({"design": s, "model": m, "n": int(len(yy)), "prevalence": round(float(yy.mean()), 4),
                    "AUC": round(float(roc_auc_score(yy, pp)), 4),
                    "Brier": round(float(brier_score_loss(yy, pp)), 4),
                    "ECE": round(ece(yy, pp), 4), "calibration_slope": round(sl, 4), "calibration_intercept": round(ic, 4),
                    "mean_prediction": round(float(pp.mean()), 4)})
        for pt in PTS:
            dec.append({"design": s, "model": m, "threshold_probability": float(pt),
                        "net_benefit": round(net_benefit(yy, pp, pt), 6),
                        "net_benefit_flag_all": round(float(yy.mean()) - (1 - float(yy.mean())) * pt / (1 - pt), 6)})
C = pd.concat(cal, ignore_index=True); C.to_csv(f"{OUT}/D_calibration_{OUTCOME}.csv", index=False)
D = pd.DataFrame(dec); D.to_csv(f"{OUT}/D_decision_{OUTCOME}.csv", index=False)
M = pd.DataFrame(met); M.to_csv(f"{OUT}/D_metrics_{OUTCOME}.csv", index=False)
print("\n=== calibration metrics ==="); pd.set_option("display.width", 200)
print(M[M.model == MODEL].to_string(index=False))

# ---------------- figure ----------------
plt.rcParams.update({
 "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
 "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5,
 "ytick.labelsize": 7.5, "legend.fontsize": 7.3, "axes.linewidth": 0.6, "figure.dpi": 200,
 "savefig.dpi": 400, "axes.spines.top": False, "axes.spines.right": False,
 "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "grid.color": "#b8b7b0"})
INK, INK2 = "#0b0b0b", "#52514e"
# palette validated in light mode; all five slots pass
CL = ["#1baf7a", "#eb6834", "#2a78d6", "#e87ba4", "#6250d6"]
MK = ["o", "s", "^", "D", "v"]
LBL = {"1 Random (conventional)": "1 Random (conventional)", "2 Market hold-out": "2 Market hold-out",
       "3 Temporal (pre to post)": "3 Temporal (pre→post)", "4 Host hold-out": "4 Host hold-out",
       "5 Market by time": "5 Market × time"}

# --- no explanatory block is drawn outside the axes (journal style); the sentences go into the caption instead ---
def _save_note(key, text):
    import json as _j, os as _o
    _p = "./out/fig_notes.json"
    _d = _j.load(open(_p)) if _o.path.exists(_p) else {}
    _d[key] = " ".join(str(text).split())
    _j.dump(_d, open(_p, "w"), indent=1, ensure_ascii=False)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
ax = axes[0]
ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, zorder=1)
for i, s in enumerate(SPLITS):
    t = C[(C["design"] == s) & (C["model"] == MODEL)]
    if t.empty: continue
    ax.plot(t.pred_mean, t.obs_rate, marker=MK[i % 5], ms=3.4, lw=1.6, color=CL[i % 5],
            label=LBL.get(s, s), zorder=3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Predicted probability (bin mean)"); ax.set_ylabel("Observed rate")
sl = {r.design: r.calibration_slope for r in M[M.model == MODEL].itertuples()}
_o = [k for k in ["1 Random (conventional)", "4 Host hold-out", "3 Temporal (pre to post)", "2 Market hold-out", "5 Market by time"] if k in sl]
ax.annotate("Calibration slope (1.00 = perfect)\n"
            + "\n".join(f"  {LBL.get(k, k)}:  {sl[k]:.2f}" for k in _o),
            (0.985, 0.03), xycoords="axes fraction", fontsize=6.8, color=INK2,
            va="bottom", ha="right")
ax.set_title("(a) Calibration curves by validation design", loc="center", color=INK)
ax = axes[1]
base = D[(D["model"] == MODEL)]
for i, s in enumerate(SPLITS):
    t = base[base["design"] == s]
    if t.empty: continue
    ax.plot(t.threshold_probability, t.net_benefit, marker=MK[i % 5], ms=2.6, lw=1.6, color=CL[i % 5],
            label=LBL.get(s, s), zorder=3)
t0 = base[base["design"] == SPLITS[0]]
ax.plot(t0.threshold_probability, t0.net_benefit_flag_all, ls="--", lw=1.0, color=INK2, label="Flag all listings", zorder=2)
ax.axhline(0, color=INK2, lw=0.8, zorder=2)
ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
ax.set_ylim(-0.05, max(0.45, float(base.net_benefit.max()) * 1.1))
_d = D[D["model"] == MODEL]
_neg = _d[(_d["design"] == "5 Market by time") & (_d["net_benefit"] < 0)]
_x0 = float(_neg.threshold_probability.min()) if len(_neg) else float("nan")
if np.isfinite(_x0):
    ax.axvline(_x0, color=CL[4 % 5], lw=0.9, ls=":", zorder=2)
    ax.annotate(f"market × time turns\nnegative at {_x0:.2f}", (_x0, ax.get_ylim()[1] * 0.62),
                xytext=(5, 0), textcoords="offset points", fontsize=6.8, color=CL[4 % 5], va="top")
ax.set_title(f"(b) Decision curves (net benefit negative beyond {_x0:.2f})"
             if np.isfinite(_x0) else "(b) Decision curves by validation design", loc="center", color=INK)
_h, _l = ax.get_legend_handles_labels()
fig.tight_layout()
# one legend outside the axes, below the figure, since both panels use the same designs
fig.legend(_h, _l, frameon=False, fontsize=6.9, ncol=6, loc="upper center",
           bbox_to_anchor=(0.5, 0.02), handlelength=1.8, columnspacing=1.4)
_save_note(f"F6_calibration_decision_{OUTCOME}",
    f"{MODEL_EN}; outcome: {OUTCOME_EN}. Shared evaluation subsample, folds pooled. "
    "Panel (a): 20 equal-count bins; the diagonal is perfect calibration.")
fig.savefig(f"{FIG}/F6_calibration_decision_{OUTCOME}.png", bbox_inches="tight")
fig.savefig(f"{FIG}/F6_calibration_decision_{OUTCOME}.pdf", bbox_inches="tight")
print(f"\nsaved {FIG}/F6_calibration_decision_{OUTCOME}.png  ·  {(time.time()-T0)/60:.1f} min")
