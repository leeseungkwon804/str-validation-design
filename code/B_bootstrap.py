"""B. Uncertainty on the gap: confidence intervals from resampling markets rather than rows.
The standard deviation across folds reflects variation the design creates, not the uncertainty of having drawn a different set of markets.
Markets are resampled with replacement, and the AUC is recomputed on the evaluation rows of the resampled markets.
The difference between designs is computed pairwise within each bootstrap resample before the interval is taken.
Input: out/A_preds_<outcome>.parquet, the shared evaluation-sample predictions stored by the benchmark"""
import os, sys, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import rankdata

OUT = "./out"
OUTCOME = sys.argv[1] if len(sys.argv) > 1 else "y_license"
B = int(os.environ.get("NBOOT", "1000"))
REF = "1 Random (conventional)"
T0 = time.time()

pred = pd.read_parquet(f"{OUT}/A_preds_{OUTCOME}.parquet")
meta = pd.read_parquet("./mat/meta.parquet", columns=["city"])
pred["city"] = meta.city.to_numpy()[pred.idx.to_numpy()]
MODELS = [c for c in pred.columns if c not in ("y", "idx", "design", "folds", "city")]
SPLITS = list(dict.fromkeys(pred["design"]))
print(f"evaluation rows {len(pred):,}, designs {len(SPLITS)}, models {len(MODELS)}, bootstrap resamples {B}", flush=True)

def fast_auc(yy, pp):
    """Rank-based AUC; yy is 0/1."""
    n1 = yy.sum(); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(pp)
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

ST = {}
for s in SPLITS:
    d = pred[pred["design"] == s].reset_index(drop=True)
    P = d[MODELS].to_numpy(np.float64)
    ok = ~np.isnan(P).all(0)
    cities, codes = np.unique(d.city.values, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(cities) + 1))
    ST[s] = {"y": d.y.to_numpy(np.int8), "P": P, "ok": ok, "cities": cities,
             "rows_by_city": {cities[i]: order[bounds[i]:bounds[i + 1]] for i in range(len(cities))}}
    print(f"  {s}: rows {len(d):,}, markets {len(cities)}", flush=True)

point = []
for s in SPLITS:
    y = ST[s]["y"]; P = ST[s]["P"]
    for j, m in enumerate(MODELS):
        if not ST[s]["ok"][j]: continue
        point.append({"design": s, "model": m, "AUC": round(float(fast_auc(y, P[:, j])), 5),
                      "n_eval": int(len(y)), "n_markets": int(len(ST[s]["cities"]))})
PT = pd.DataFrame(point); PT.to_csv(f"{OUT}/B_point_{OUTCOME}.csv", index=False)

common = set(ST[SPLITS[0]]["cities"])
for s in SPLITS[1:]: common &= set(ST[s]["cities"])
common = sorted(common)
print(f"{len(common)} markets appear in every design; the paired intervals are computed over these", flush=True)

rng = np.random.RandomState(20261112)
boot = {(s, m): np.full(B, np.nan) for s in SPLITS for m in MODELS}
gapv = {(s, m): np.full(B, np.nan) for s in SPLITS for m in MODELS}
for b in range(B):
    picked = [common[i] for i in rng.randint(0, len(common), len(common))]
    cur = {}
    for s in SPLITS:
        rb = ST[s]["rows_by_city"]
        rows = np.concatenate([rb[c] for c in picked if c in rb])
        yy = ST[s]["y"][rows]; PP = ST[s]["P"][rows]
        for j, m in enumerate(MODELS):
            if not ST[s]["ok"][j]: continue
            v = fast_auc(yy, PP[:, j]); boot[(s, m)][b] = v; cur[(s, m)] = v
    for m in MODELS:
        r0 = cur.get((REF, m), np.nan)
        for s in SPLITS:
            if (s, m) in cur: gapv[(s, m)][b] = r0 - cur[(s, m)]
    if (b + 1) % 100 == 0: print(f"  {b+1}/{B}  {time.time()-T0:.0f}seconds", flush=True)

def ci(v):
    v = v[~np.isnan(v)]
    if len(v) == 0: return (np.nan,) * 4
    return (float(v.mean()), float(v.std()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

rows = []
for s in SPLITS:
    for m in MODELS:
        pt = PT[(PT["design"] == s) & (PT["model"] == m)]
        if pt.empty: continue
        bm, bs, lo, hi = ci(boot[(s, m)]); gm, gs, glo, ghi = ci(gapv[(s, m)])
        rows.append({"design": s, "model": m, "AUC": float(pt.AUC.iloc[0]),
                     "AUC_bootstrap_mean": round(bm, 5), "AUC_boot_se": round(bs, 5),
                     "AUC_lo95": round(lo, 5), "AUC_hi95": round(hi, 5),
                     "loss_vs_random_split": round(gm, 5), "gap_se": round(gs, 5),
                     "gap_lo95": round(glo, 5), "gap_hi95": round(ghi, 5),
                     "n_eval": int(pt.n_eval.iloc[0]), "n_markets": int(pt.n_markets.iloc[0])})
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/B_bootstrap_{OUTCOME}.csv", index=False)
pd.set_option("display.width", 230)
print("\n=== market-level bootstrap (AUC and loss against the random split) ===")
print(R.to_string(index=False))
json.dump({"B": B, "n_common_markets": len(common), "common_markets": list(map(str, common)),
           "minutes": round((time.time()-T0)/60, 1)},
          open(f"{OUT}/B_meta_{OUTCOME}.json", "w"), indent=1, ensure_ascii=False)
print(f"\ntotal {(time.time()-T0)/60:.1f} min")
