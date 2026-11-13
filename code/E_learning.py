"""E. Learning curve: does market hold-out performance recover as training markets are added?
If it does not recover, collecting more data is not a remedy, which is the practical implication of the paper.

Three curves are drawn separately.
 arm (a): more markets with the number of training rows held fixed, isolating the effect of market diversity
 arm (b): more markets using all of their rows, which is how data are added in practice (confounded with row count)
 arm (c): more training rows under the random split, the curve the conventional design produces
Evaluation rows come from the same fixed 25 held-out markets for the first two arms, and from other rows of the same markets for the control.
No market identity among the predictors."""
import os, sys, json, time, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, prep
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

OUT = "./out"; FIG = "./figs"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
OUTCOME = sys.argv[1] if len(sys.argv) > 1 else "y_license"
N_FIX = 800000; N_EVAL = 150000; REPS = 5
T0 = time.time()
X, y_lic, y_str, meta, sch = prep.load()
y = (y_lic if OUTCOME == "y_license" else y_str).astype(np.int8)
city = meta.city.values
cities = np.unique(city)
rs0 = np.random.RandomState(20261112)
hold = np.sort(rs0.choice(cities, 25, replace=False))
pool = np.array([c for c in cities if c not in set(hold)])
mask_hold = np.isin(city, hold)
idx_hold = np.where(mask_hold)[0]
ev_hold = np.sort(rs0.choice(idx_hold, min(N_EVAL, len(idx_hold)), replace=False))
print(f"held-out markets {len(hold)}, evaluation rows {len(ev_hold):,}, training pool markets {len(pool)} "
      f"({(~mask_hold).sum():,} rows), held-out positive rate {100*y[ev_hold].mean():.2f}%", flush=True)
json.dump({"holdout_markets": list(map(str, hold))}, open(f"{OUT}/E_holdout_markets.json", "w"),
          indent=1, ensure_ascii=False)

rows_by_city = {c: np.where(city == c)[0] for c in pool}
Xev = np.asarray(X[ev_hold]); yev = y[ev_hold]

MIN_FIT = 2000
def fit_eval(tr_idx, Xq, yq):
    """Returns None when the training sample is too small or the fit fails; that point is skipped but recorded."""
    if len(tr_idx) < MIN_FIT or y[tr_idx].std() == 0:
        return None
    try:
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                           early_stopping=True, random_state=0)
        m.fit(np.asarray(X[tr_idx]), y[tr_idx])
        p = m.predict_proba(Xq)[:, 1]
        r = {"AUC": float(roc_auc_score(yq, p)), "PRAUC": float(average_precision_score(yq, p)),
             "Brier": float(brier_score_loss(yq, p))}
        del m; gc.collect()
        return r
    except Exception as e:
        print(f"    !! fit failed, n={len(tr_idx):,}: {type(e).__name__} {str(e)[:120]}", flush=True)
        gc.collect()
        return None

KS = [2, 5, 10, 20, 40, 60, 80, len(pool)]
res = []
for K in KS:
    reps = 1 if K == len(pool) else REPS
    for r in range(reps):
        rs = np.random.RandomState(1000 * K + r)
        pick = rs.choice(pool, K, replace=False)
        idx = np.concatenate([rows_by_city[c] for c in pick])
        if y[idx].std() == 0:
            print(f"  K={K} r={r} skipped (outcome constant)", flush=True); continue
        # arm (a): rows held fixed
        sub = idx if len(idx) <= N_FIX else np.sort(rs.choice(idx, N_FIX, replace=False))
        t1 = time.time(); a = fit_eval(sub, Xev, yev)
        if a is None:
            print(f"  [fixed rows] K={K:<3} r={r} n={len(sub):>9,} skipped", flush=True)
        else:
            res.append({"curve": "Markets added, training rows fixed at 800,000", "n_training_markets": K, "rep": r, "n_train": int(len(sub)),
                        **{k: round(v, 5) for k, v in a.items()}, "seconds": round(time.time()-t1, 1)})
            print(f"  [fixed rows] K={K:<3} r={r} n={len(sub):>9,} AUC {a['AUC']:.4f}", flush=True)
        # arm (b): all rows of those markets
        if len(idx) != len(sub):
            t1 = time.time(); b = fit_eval(idx, Xev, yev)
            if b is not None:
                res.append({"curve": "Markets added, all rows of the sampled markets", "n_training_markets": K, "rep": r, "n_train": int(len(idx)),
                            **{k: round(v, 5) for k, v in b.items()}, "seconds": round(time.time()-t1, 1)})
                print(f"  [all rows]   K={K:<3} r={r} n={len(idx):>9,} AUC {b['AUC']:.4f}", flush=True)
        elif a is not None:
            res.append({"curve": "Markets added, all rows of the sampled markets", "n_training_markets": K, "rep": r, "n_train": int(len(idx)),
                        **{k: round(v, 5) for k, v in a.items()}, "seconds": 0.0})
        if res: pd.DataFrame(res).to_csv(f"{OUT}/E_learning_{OUTCOME}.csv", index=False)

# arm (c): random-split control, drawing from every market and evaluating on other rows of the same markets
rs = np.random.RandomState(7)
allidx = np.arange(len(y))
te_r = np.sort(rs.choice(allidx, N_EVAL, replace=False))
tr_pool = np.setdiff1d(allidx, te_r, assume_unique=False)
Xr = np.asarray(X[te_r]); yr = y[te_r]
for N in [25000, 50000, 100000, 200000, 400000, 800000, 1600000, 3200000, len(tr_pool)]:
    sub = np.sort(rs.choice(tr_pool, min(N, len(tr_pool)), replace=False))
    t1 = time.time(); a = fit_eval(sub, Xr, yr)
    if a is None:
        print(f"  [control]    n={len(sub):>9,} skipped", flush=True); continue
    res.append({"curve": "Random split control, training rows added", "n_training_markets": int(len(np.unique(city[sub]))), "rep": 0,
                "n_train": int(len(sub)), **{k: round(v, 5) for k, v in a.items()},
                "seconds": round(time.time()-t1, 1)})
    print(f"  [control]    n={len(sub):>9,} AUC {a['AUC']:.4f}", flush=True)
    pd.DataFrame(res).to_csv(f"{OUT}/E_learning_{OUTCOME}.csv", index=False)

R = pd.DataFrame(res); R.to_csv(f"{OUT}/E_learning_{OUTCOME}.csv", index=False)
agg = (R.groupby(["curve", "n_training_markets"]).agg(reps=("rep", "size"), n_train=("n_train", "mean"),
       AUC=("AUC", "mean"), AUC_sd=("AUC", "std"), AUC_min=("AUC", "min"),
       AUC_max=("AUC", "max")).round(4).reset_index())
agg.to_csv(f"{OUT}/E_learning_summary_{OUTCOME}.csv", index=False)
print("\n=== summary ==="); pd.set_option("display.width", 200)
print(agg.to_string(index=False))
print(f"\ntotal {(time.time()-T0)/60:.1f} min")
