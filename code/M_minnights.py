"""M. Checking whether the minimum-nights finding can be used in the discussion.
Question: does the distribution of minimum_nights really differ between markets, and does it
whether the values pile up at thresholds that regulation commonly uses (28, 30, 31, 32 or 90 days).
Note: this script produces descriptive statistics only. Which instrument set a given threshold in a given market
   is not asserted without the primary legal text for that market."""
import os, sys, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd, prep

OUT = "./out"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
X, y_lic, y_str, meta, sch = prep.load()
cols = sch["columns"]
mn = np.asarray(X[:, cols.index("minimum_nights")], dtype=np.float64)
d = pd.DataFrame({"city": meta.city.values, "country": meta.country.values,
                  "group": meta.group.values, "mn": mn,
                  "y_license": y_lic, "y_strict": y_str})
print(f"{len(d):,} rows, markets {d.city.nunique()}, minimum_nights missing {int(d.mn.isna().sum()):,}", flush=True)

THRESH = [1, 2, 3, 4, 5, 7, 14, 28, 29, 30, 31, 32, 60, 90, 180, 365]
g = d.groupby("city")
rows = []
for c, x in g:
    v = x.mn.dropna()
    if len(v) == 0: continue
    vc = v.value_counts(normalize=True)
    r = {"city": c, "country": x.country.iloc[0], "group": x.group.iloc[0], "n": int(len(v)),
         "median": float(v.median()), "p90": float(v.quantile(0.9)),
         "modal_value": float(vc.index[0]), "modal_share": round(float(vc.iloc[0]), 4),
         "share_top3_values": round(float(vc.iloc[:3].sum()), 4),
         "y_strict": round(float(x.y_strict.mean()), 4)}
    for t in THRESH:
        r[f"sh_{t}"] = round(float((v == t).mean()), 4)
    r["sh_28to32"] = round(float(v.between(28, 32).mean()), 4)
    r["sh_ge90"] = round(float((v >= 90).mean()), 4)
    rows.append(r)
M = pd.DataFrame(rows).sort_values("sh_28to32", ascending=False)
M.to_csv(f"{OUT}/M_minnights_by_market.csv", index=False)

glob_vc = d.mn.dropna().value_counts(normalize=True)
S = {
 "n_rows": int(d.mn.notna().sum()),
 "global_top10_values": {str(int(k)): round(float(v), 4) for k, v in glob_vc.head(10).items()},
 "global_share_28to32": round(float(d.mn.dropna().between(28, 32).mean()), 4),
 "n_markets": int(len(M)),
 "modal_value_distinct_across_markets": int(M.modal_value.nunique()),
 "modal_value_counts": {str(int(k)): int(v) for k, v in M.modal_value.value_counts().items()},
 "share_at_28to32": {"median_market": round(float(M.sh_28to32.median()), 4),
                     "max_market": round(float(M.sh_28to32.max()), 4),
                     "n_markets_gt20pct": int((M.sh_28to32 > 0.20).sum()),
                     "n_markets_gt50pct": int((M.sh_28to32 > 0.50).sum())},
 "markets_top10_28to32": M.head(10)[["city", "country", "sh_28to32", "modal_value", "modal_share",
                                     "y_strict"]].to_dict("records"),
 "markets_lowest_28to32": M.tail(5)[["city", "country", "sh_28to32", "modal_value"]].to_dict("records"),
 "corr_sh28to32_vs_ystrict": round(float(M[["sh_28to32", "y_strict"]].corr().iloc[0, 1]), 4),
 "caveat": ("Descriptive statistics only. Attributing a threshold to a particular instrument requires the primary legal text for that market. "
            "Until then the claim goes no further than: the distribution differs clearly between markets and piles up at values regulation commonly uses."),
}
json.dump(S, open(f"{OUT}/M_minnights_summary.json", "w"), indent=1, ensure_ascii=False)
pd.set_option("display.width", 220)
print(json.dumps(S, indent=1, ensure_ascii=False))
print("\n=== 15 markets with the highest share of listings at 28 to 32 nights ===")
print(M.head(15)[["city", "country", "n", "modal_value", "modal_share", "sh_1", "sh_2", "sh_3", "sh_7",
                  "sh_28to32", "sh_30", "sh_ge90", "y_strict"]].to_string(index=False))
print("\n=== distribution of modal values across markets ===")
print(M.modal_value.value_counts().to_string())
print(f"\ntotal {(time.time()-t0)/60:.1f} min")
