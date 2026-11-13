"""C3. Ablation: which predictor group carries the regional fingerprint, by removal and by group-alone fits.
measured separately under the random split and under the market hold-out.
 - if removing a group lowers only the random-split AUC, the group is generally useful rather than market-specific
 - if removing a group narrows the gap between the random split and the market hold-out, that group was carrying the market fingerprint
For comparability of cost, gradient boosting is fitted on a fixed subsample of 1,000,000 training rows in every setting.
Each step writes to CSV immediately, so an interrupted run loses nothing."""
import os, sys, json, time, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, prep
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

OUT = "./out"; os.makedirs(OUT, exist_ok=True)
RES = f"{OUT}/C3_ablation_folds.csv"
NFIT = 1000000
OUTCOME = sys.argv[1] if len(sys.argv) > 1 else "y_license"
T0 = time.time()
X, y_lic, y_str, meta, sch = prep.load()
y = (y_lic if OUTCOME == "y_license" else y_str).astype(np.int8)
cols = sch["columns"]; city = meta.city.values; n = len(y)
del y_lic, y_str; gc.collect()
GROUPS = json.load(open(f"{OUT}/C1_groups.json"))
IDX = {g: [cols.index(c) for c in v] for g, v in GROUPS.items()}
ALL = list(range(len(cols)))

def splits_random():
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y): yield tr, te
def splits_market():
    for tr, te in GroupKFold(5).split(np.zeros(n), y, groups=city): yield tr, te
SPL = {"Random split": splits_random, "Market hold-out": splits_market}

done = set()
if os.path.exists(RES):
    d0 = pd.read_csv(RES); done = set(zip(d0["setting"], d0["design"], d0["folds"]))
    print(f"{len(done)} combinations already done, skipping", flush=True)

SETTINGS = [("All 50 predictors (baseline)", ALL)]
SETTINGS += [(f"Removed: {g}", [i for i in ALL if i not in IDX[g]]) for g in GROUPS]
SETTINGS += [(f"Alone: {g}", IDX[g]) for g in GROUPS]

for sname, use in SETTINGS:
    for spl, sfn in SPL.items():
        for k, (tr, te) in enumerate(sfn()):
            if (sname, spl, k) in done: continue
            if y[te].std() == 0 or y[tr].std() == 0: continue
            rsk = np.random.RandomState(97 * k)
            ev = np.sort(rsk.choice(te, min(100000, len(te)), replace=False))
            sub = tr if len(tr) <= NFIT else np.sort(rsk.choice(tr, NFIT, replace=False))
            t1 = time.time()
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                               early_stopping=True, random_state=0)
            m.fit(np.asarray(X[sub])[:, use], y[sub])
            p = m.predict_proba(np.asarray(X[ev])[:, use])[:, 1]
            row = {"outcome": OUTCOME, "setting": sname, "n_variables": len(use), "design": spl, "folds": k,
                   "n_train": int(len(sub)), "n_eval": int(len(ev)),
                   "AUC": round(float(roc_auc_score(y[ev], p)), 5), "seconds": round(time.time()-t1, 1)}
            pd.DataFrame([row]).to_csv(RES, mode="a", header=not os.path.exists(RES), index=False)
            del m, p; gc.collect()
        print(f"  {sname} | {spl} done ({time.time()-T0:.0f} s)", flush=True)

R = pd.read_csv(RES).drop_duplicates(["setting", "design", "folds"], keep="last")
R.to_csv(RES, index=False)
a = R.groupby(["setting", "design"]).AUC.agg(["mean", "std", "size"]).unstack("design")
a.columns = ["_".join(c) for c in a.columns]
a = a.reset_index().rename(columns={"mean_Random split": "AUC_random_split", "mean_Market hold-out": "AUC_market_holdout",
                                    "std_Random split": "random_split_sd", "std_Market hold-out": "market_holdout_sd"})
a["gap"] = a.AUC_random_split - a.AUC_market_holdout
b = a[a.setting == "All 50 predictors (baseline)"].iloc[0]
a["random_split_change"] = a.AUC_random_split - b.AUC_random_split
a["market_holdout_change"] = a.AUC_market_holdout - b.AUC_market_holdout
a["gap_change"] = a.gap - b.gap
a = a.round(4)
a.to_csv(f"{OUT}/C3_ablation.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== C3 summary ===")
print(a[["setting", "AUC_random_split", "AUC_market_holdout", "gap", "random_split_change", "market_holdout_change",
         "gap_change"]].to_string(index=False))
print(f"\ntotal {(time.time()-T0)/60:.1f} min")
