"""A. Model zoo: logistic regression, random forest, gradient boosting and TabPFN under the same five validation designs.
Purpose: to show that the pattern across designs is the same for every model family, a small loss under the host hold-out and a large one under the market hold-out.

Design
 - full-fold regime: logistic regression, random forest and gradient boosting fitted on the entire training fold (the headline numbers)
 - sample-matched regime (10,000 rows): the same models plus TabPFN, whose context length limits training to 10,000 rows, so
   the other models are refitted on the same 10,000 rows so that the comparison with TabPFN is fair.
 - evaluation sample: at most 100,000 test rows per fold, shared by every model, so that
   so that the bootstrap and the calibration and decision curves are computed on the same rows. All predicted probabilities are stored.

No market identity among the predictors (city, country, latitude, longitude, neighbourhood); prep.py drops every m_-prefixed column.
TabPFN is used in its open-source form, which needs no authentication token; no token file is read or printed.
   This implementation uses the open-source tabpfn 2.2.1 package, which needs no token, and stops immediately if any authentication prompt appears.
"""
import os, sys, json, time, gc, warnings, traceback
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "./t3/cache")
os.environ.setdefault("HF_HOME", "./t3/cache/hf")
os.environ.setdefault("TABPFN_ALLOW_CLI_INTERACTION", "False")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, prep
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SIGNS = ("auth_token", "access token", "api token", "token not found", "authenticat",
         "api key", "tabpfn_token", "unauthor", "log in", "sign up", "subscription")
def tripwire(msg):
    hits = [s for s in SIGNS if s in str(msg).lower()]
    if hits:
        print("\nSTOP: an authentication prompt was detected ->", hits, flush=True)
        print("Stopping without looking for or supplying any token. Start of the message:", str(msg)[:400], flush=True)
        return True
    return False

OUTCOME = sys.argv[1] if len(sys.argv) > 1 else "y_license"
N_SUB, N_EVAL, SEED = 10000, 100000, 0
OUT = "./out"; os.makedirs(OUT, exist_ok=True)

T0 = time.time()
X, y_lic, y_str, meta, sch = prep.load()
y = (y_lic if OUTCOME == "y_license" else y_str).astype(np.int8)
city = meta.city.values; host = meta.host.values; side = meta.side.values
n = len(y)
print(f"[{OUTCOME}] rows {n:,}, predictors {X.shape[1]}, markets {len(np.unique(city))}, "
      f"positive rate {100*y.mean():.2f}%", flush=True)

def splits_random():
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y): yield tr, te
def splits_group(g):
    for tr, te in GroupKFold(5).split(np.zeros(n), y, groups=g): yield tr, te
def splits_time():
    yield np.where(side == "pre")[0], np.where(side == "post")[0]
def splits_market_time():
    cities = np.unique(city); rng = np.random.RandomState(0); rng.shuffle(cities)
    for f in range(5):
        hold = set(cities[f::5]); m = np.isin(city, list(hold))
        tr = np.where(~m & (side == "pre"))[0]; te = np.where(m & (side == "post"))[0]
        if len(te) > 0 and y[te].std() > 0: yield tr, te

SPL = {"1 Random (conventional)": splits_random,
       "2 Market hold-out": lambda: splits_group(city),
       "3 Temporal (pre to post)": splits_time,
       "4 Host hold-out": lambda: splits_group(host),
       "5 Market by time": splits_market_time}

def mk_lr(): return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                              ("m", LogisticRegression(max_iter=300, n_jobs=13))])
def mk_gb(): return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                   early_stopping=True, random_state=0)
def mk_rf(): return Pipeline([("i", SimpleImputer(strategy="median")),
                              ("m", RandomForestClassifier(n_estimators=120, max_samples=0.15,
                                                           min_samples_leaf=20, n_jobs=96,
                                                           random_state=0))])
def mk_rf_small(): return Pipeline([("i", SimpleImputer(strategy="median")),
                                    ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                                 n_jobs=32, random_state=0))])
def mk_tp():
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device="cuda", random_state=0)

def tp_proba(clf, Xq, chunk=20000):
    """TabPFN predictions are made in chunks of 20,000 rows; larger chunks raise a CUDA kernel configuration error."""
    return np.concatenate([clf.predict_proba(Xq[i:i + chunk])[:, 1]
                           for i in range(0, len(Xq), chunk)])

def strat_sub(idx, k, rs):
    if len(idx) <= k: return idx
    yy = y[idx]; out = []
    for c in (0, 1):
        s = idx[yy == c]; take = max(1, int(round(k * len(s) / len(idx))))
        out.append(rs.choice(s, min(take, len(s)), replace=False))
    return np.sort(np.concatenate(out))

def metrics(yt, p):
    return dict(AUC=float(roc_auc_score(yt, p)), PRAUC=float(average_precision_score(yt, p)),
                Brier=float(brier_score_loss(yt, p)))

FULL = (("Logistic regression", mk_lr), ("Random forest", mk_rf), ("GBDT", mk_gb))
SMALL = (("Logistic regression", mk_lr), ("Random forest", mk_rf_small), ("GBDT", mk_gb), ("TabPFN", mk_tp))

rows, preds = [], []
for sname, sfn in SPL.items():
    for k, (tr, te) in enumerate(sfn()):
        if y[te].std() == 0 or y[tr].std() == 0: continue
        rs = np.random.RandomState(SEED + 97 * k)
        ev = np.sort(rs.choice(te, min(N_EVAL, len(te)), replace=False))
        sub = strat_sub(tr, N_SUB, rs)
        pos = np.searchsorted(te, ev)
        Xev = np.asarray(X[ev]); yev = y[ev]
        Xsub = np.asarray(X[sub]); ysub = y[sub]
        pr = {}
        Xtr = np.asarray(X[tr]); ytr = y[tr]; Xte = np.asarray(X[te]); yte = y[te]
        for mname, mk in FULL:
            t1 = time.time(); m = mk(); m.fit(Xtr, ytr)
            pfull = m.predict_proba(Xte)[:, 1]; pev = pfull[pos]
            r = dict(design=sname, folds=k, model=mname, training="Full fold", n_train=int(len(tr)),
                     n_test=int(len(te)), n_eval=int(len(ev)), seconds=round(time.time()-t1, 1))
            r.update({f"{a}_full": round(v, 5) for a, v in metrics(yte, pfull).items()})
            r.update({f"{a}_eval": round(v, 5) for a, v in metrics(yev, pev).items()})
            rows.append(r); pr[f"{mname}_Full fold"] = pev.astype(np.float32)
            print(f"  [{sname}|{mname}|Full fold] f{k} AUC_full {r['AUC_full']:.4f} "
                  f"AUC_eval {r['AUC_eval']:.4f} n_tr {len(tr):,} {r['seconds']}seconds", flush=True)
            del m, pfull; gc.collect()
        del Xtr, Xte; gc.collect()
        for mname, mk in SMALL:
            t1 = time.time()
            try:
                m = mk(); m.fit(Xsub, ysub)
                pev = tp_proba(m, Xev) if mname == "TabPFN" else m.predict_proba(Xev)[:, 1]
            except Exception as e:
                tb = traceback.format_exc()
                if tripwire(tb): sys.exit(3)
                print(f"  !! {mname}|10k failed: {type(e).__name__} {str(e)[:200]}", flush=True); continue
            r = dict(design=sname, folds=k, model=mname, training="10,000 rows", n_train=int(len(sub)),
                     n_test=int(len(te)), n_eval=int(len(ev)), seconds=round(time.time()-t1, 1))
            r.update({f"{a}_eval": round(v, 5) for a, v in metrics(yev, pev).items()})
            rows.append(r); pr[f"{mname}_10,000 rows"] = pev.astype(np.float32)
            print(f"  [{sname}|{mname}|10,000 rows] f{k} AUC_eval {r['AUC_eval']:.4f} {r['seconds']}seconds", flush=True)
            del m; gc.collect()
        p = pd.DataFrame(pr); p["y"] = yev; p["idx"] = ev; p["design"] = sname; p["folds"] = k
        preds.append(p)
        pd.DataFrame(rows).to_csv(f"{OUT}/A_folds_{OUTCOME}.csv", index=False)
        pd.concat(preds, ignore_index=True).to_parquet(f"{OUT}/A_preds_{OUTCOME}.parquet", index=False)
        del Xev, Xsub; gc.collect()

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/A_folds_{OUTCOME}.csv", index=False)
pd.concat(preds, ignore_index=True).to_parquet(f"{OUT}/A_preds_{OUTCOME}.parquet", index=False)
agg = (df.groupby(["design", "model", "training"], sort=False)
         .agg(folds=("folds", "size"), AUC_full=("AUC_full", "mean"), AUC_full_sd=("AUC_full", "std"),
              PRAUC_full=("PRAUC_full", "mean"), Brier_full=("Brier_full", "mean"),
              AUC=("AUC_eval", "mean"), AUC_sd=("AUC_eval", "std"),
              PRAUC=("PRAUC_eval", "mean"), Brier=("Brier_eval", "mean"),
              seconds=("seconds", "mean")).round(4).reset_index())
agg.to_csv(f"{OUT}/A_summary_{OUTCOME}.csv", index=False)
print("\n=== summary ===")
print(agg.to_string(index=False))
print(f"\ntotal {(time.time()-T0)/60:.1f} min")
