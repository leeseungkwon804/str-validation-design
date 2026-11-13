"""L-2. Does a collection-era artefact distort the temporal design?
Finding: neighborhood_overview starts emptying in 2025 and is empty in every 2026 snapshot.
Of the 319 pre-cut snapshots 77 have values; all 253 post-cut snapshots have none.
-> len_neighborhood_overview is not label leakage, but it marks the collection era and so splits the feature distribution between training and test.
Refits the temporal design without that variable to see whether the result changes."""
import os, sys, json, time, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, prep
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

OUT = "./out"
X, y_lic, y_str, meta, sch = prep.load()
y = y_lic.astype(np.int8); cols = sch["columns"]; side = meta.side.values
ALL = list(range(len(cols)))
DROP = ["len_neighborhood_overview"]
KEEP = [i for i in ALL if cols[i] not in DROP]
tr = np.where(side == "pre")[0]; te = np.where(side == "post")[0]
print(f"pre {len(tr):,} → post {len(te):,}", flush=True)
rows = []
for nm, use in (("All 50 predictors", ALL), ("− len_neighborhood_overview", KEEP)):
    t1 = time.time()
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                       early_stopping=True, random_state=0)
    m.fit(np.asarray(X[tr])[:, use], y[tr])
    p = m.predict_proba(np.asarray(X[te])[:, use])[:, 1]
    rows.append({"setting": nm, "n_variables": len(use),
                 "AUC": round(float(roc_auc_score(y[te], p)), 5),
                 "PRAUC": round(float(average_precision_score(y[te], p)), 5),
                 "Brier": round(float(brier_score_loss(y[te], p)), 5),
                 "seconds": round(time.time() - t1, 1)})
    print("  " + json.dumps(rows[-1], ensure_ascii=False), flush=True)
    del m, p; gc.collect()
R = pd.DataFrame(rows)
R["AUC_change"] = (R.AUC - R.AUC.iloc[0]).round(5)
R.to_csv(f"{OUT}/L_temporal_check.csv", index=False)
print("\n" + R.to_string(index=False))
