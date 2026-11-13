"""C2. Base-rate reference scores: how far does a score get that names the market and returns that market's training-fold mean?
The closer it comes to the full model's AUC, the more the model is drawing on the market's base rate.
Under the random split the test row's market is present in training, so the reference score works;
under the market hold-out the test market is absent from training, so it does not, and that contrast is the point.
Each step writes to CSV immediately, so an interrupted run loses nothing."""
import os, sys, json, time, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, prep
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score

OUT = "./out"; os.makedirs(OUT, exist_ok=True)
NJ = int(os.environ.get("C_NJOBS", "24"))
RES = f"{OUT}/C2_baserate_proxy.csv"
OUTCOME = sys.argv[1] if len(sys.argv) > 1 else "y_license"
T0 = time.time()
X, y_lic, y_str, meta, sch = prep.load()
y = (y_lic if OUTCOME == "y_license" else y_str).astype(np.int8)
city = meta.city.values; n = len(y)
del y_lic, y_str; gc.collect()

def splits_random():
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y): yield tr, te
def splits_market():
    for tr, te in GroupKFold(5).split(np.zeros(n), y, groups=city): yield tr, te

done = set()
if os.path.exists(RES):
    d0 = pd.read_csv(RES); done = set(zip(d0["outcome"], d0["design"], d0["folds"]))
    print(f"{len(done)} folds already done, skipping", flush=True)

for sname, sfn in (("1 Random (conventional)", splits_random), ("2 Market hold-out", splits_market)):
    for k, (tr, te) in enumerate(sfn()):
        if (OUTCOME, sname, k) in done:
            print(f"  (skipped) {sname} f{k}", flush=True); continue
        t1 = time.time()
        rsk = np.random.RandomState(97 * k)
        ev = np.sort(rsk.choice(te, min(100000, len(te)), replace=False))
        take = []
        for v in np.unique(city[tr]):
            s = tr[city[tr] == v]
            take.append(rsk.choice(s, min(3000, len(s)), replace=False))
        sub = np.sort(np.concatenate(take))
        clf = Pipeline([("i", SimpleImputer(strategy="median")),
                        ("m", RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                                     n_jobs=NJ, random_state=0))])
        clf.fit(np.asarray(X[sub]), city[sub])
        base = pd.Series(y[tr]).groupby(city[tr]).mean()
        P = clf.predict_proba(np.asarray(X[ev]))
        order = clf.classes_
        bvec = base.reindex(order).fillna(float(y[tr].mean())).to_numpy()
        hard = bvec[P.argmax(1)]; soft = P @ bvec
        yev = y[ev]
        row = {"outcome": OUTCOME, "design": sname, "folds": k, "n_eval": int(len(ev)),
               "market_recovery_accuracy": round(float(accuracy_score(city[ev], order[P.argmax(1)])), 4),
               "test_market_in_training": round(float(np.isin(city[ev], order).mean()), 4),
               "AUC_baserate_argmax": round(float(roc_auc_score(yev, hard)), 4),
               "AUC_baserate_posterior": round(float(roc_auc_score(yev, soft)), 4),
               "PRAUC_baserate_posterior": round(float(average_precision_score(yev, soft)), 4),
               "Brier_baserate_posterior": round(float(brier_score_loss(yev, np.clip(soft, 0, 1))), 4),
               "AUC_true_market_baserate": round(float(roc_auc_score(
                    yev, base.reindex(city[ev]).fillna(float(y[tr].mean())).to_numpy())), 4),
               "seconds": round(time.time() - t1, 1)}
        pd.DataFrame([row]).to_csv(RES, mode="a", header=not os.path.exists(RES), index=False)
        print("  " + json.dumps(row, ensure_ascii=False), flush=True)
        del clf, P; gc.collect()

R = pd.read_csv(RES).drop_duplicates(["outcome", "design", "folds"], keep="last")
R.to_csv(RES, index=False)
pd.set_option("display.width", 220)
print("\n=== C2 summary ===")
print(R.groupby(["outcome", "design"]).mean(numeric_only=True).round(4).to_string())
print(f"\ntotal {(time.time()-T0)/60:.1f} min")
