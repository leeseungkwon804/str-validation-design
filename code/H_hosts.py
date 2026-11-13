"""H. Host portfolio distribution by market, examining why removing the host-portfolio group costs the most market-recovery accuracy.
Counts host portfolios in each snapshot (market by date) and averages them within each market.
 - share of single-listing hosts, both as a share of hosts and as a share of listings
 - share of listings held by the top 10% of hosts
 - Herfindahl index and Gini coefficient
For aggregation only. No market identity enters any model, and no causal claim is made."""
import os, sys, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd, prep

OUT = "./out"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
X, y_lic, y_str, meta, sch = prep.load()
d = meta[["city", "country", "group", "date", "host", "lid"]].copy()
d["y_strict"] = y_str
print(f"{len(d):,} rows, markets {d.city.nunique()}, snapshots {d.groupby(['city','date']).ngroups}", flush=True)

def gini(v):
    v = np.sort(np.asarray(v, float)); n = len(v)
    if n == 0 or v.sum() == 0: return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))

rows = []
for (city, date), x in d.groupby(["city", "date"], sort=False):
    pc = x.groupby("host").size()                      # number of listings per host
    n_l = int(len(x)); n_h = int(len(pc))
    if n_h == 0: continue
    s = np.sort(pc.values)[::-1]
    k = max(1, int(np.ceil(0.10 * n_h)))
    rows.append({"city": city, "country": x.country.iloc[0], "group": x.group.iloc[0], "date": date,
                 "n_listings": n_l, "n_hosts": n_h,
                 "hosts_with_1_pct": 100 * float((pc == 1).mean()),
                 "listings_of_1listing_hosts_pct": 100 * float(pc[pc == 1].sum() / n_l),
                 "top10pct_host_share_pct": 100 * float(s[:k].sum() / n_l),
                 "top1pct_host_share_pct": 100 * float(s[:max(1, int(np.ceil(0.01 * n_h)))].sum() / n_l),
                 "max_host_listings": int(s[0]),
                 "mean_listings_per_host": float(n_l / n_h),
                 "HHI": float(((pc / n_l) ** 2).sum()),
                 "gini_hosts": gini(pc.values),
                 "y_strict": float(x.y_strict.mean())})
S = pd.DataFrame(rows)
S.to_csv(f"{OUT}/H_hosts_by_snapshot.csv", index=False)

M = (S.groupby(["group", "country", "city"], as_index=False)
       .agg(snapshots=("date", "nunique"), n_listings=("n_listings", "mean"), n_hosts=("n_hosts", "mean"),
            hosts_with_1_pct=("hosts_with_1_pct", "mean"),
            listings_of_1listing_hosts_pct=("listings_of_1listing_hosts_pct", "mean"),
            top10pct_host_share_pct=("top10pct_host_share_pct", "mean"),
            top1pct_host_share_pct=("top1pct_host_share_pct", "mean"),
            max_host_listings=("max_host_listings", "max"),
            mean_listings_per_host=("mean_listings_per_host", "mean"),
            HHI=("HHI", "mean"), gini_hosts=("gini_hosts", "mean"),
            y_strict=("y_strict", "mean")).round(4))
M = M.sort_values("listings_of_1listing_hosts_pct", ascending=False)
M.to_csv(f"{OUT}/H_hosts_by_market.csv", index=False)

v = M.listings_of_1listing_hosts_pct.values
R = {"n_markets": int(len(M)),
     "listings_of_1listing_hosts_pct": {
         "min": round(float(v.min()), 2), "p10": round(float(np.percentile(v, 10)), 2),
         "median": round(float(np.median(v)), 2), "p90": round(float(np.percentile(v, 90)), 2),
         "max": round(float(v.max()), 2), "sd": round(float(v.std()), 2)},
     "top10pct_host_share_pct": {
         "min": round(float(M.top10pct_host_share_pct.min()), 2),
         "median": round(float(M.top10pct_host_share_pct.median()), 2),
         "max": round(float(M.top10pct_host_share_pct.max()), 2)},
     "corr_single_vs_top10": round(float(M[["listings_of_1listing_hosts_pct",
                                            "top10pct_host_share_pct"]].corr().iloc[0, 1]), 4),
     "corr_single_vs_ystrict": round(float(M[["listings_of_1listing_hosts_pct",
                                              "y_strict"]].corr().iloc[0, 1]), 4),
     "corr_top10_vs_ystrict": round(float(M[["top10pct_host_share_pct",
                                             "y_strict"]].corr().iloc[0, 1]), 4)}
# does the distribution separate into peaks? two simple bimodality measures, without a dip test
from scipy.stats import gaussian_kde
kde = gaussian_kde(v); grid = np.linspace(v.min(), v.max(), 400); dens = kde(grid)
peaks = [i for i in range(1, len(grid) - 1) if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]]
R["kde_modes_single_pct"] = [round(float(grid[i]), 1) for i in peaks]
R["bimodality_coefficient"] = round(float(
    (pd.Series(v).skew() ** 2 + 1) / (pd.Series(v).kurt() + 3 +
     3 * (len(v) - 1) ** 2 / ((len(v) - 2) * (len(v) - 3)))), 4)   # above 0.555 suggests a bimodal distribution

FIVE = {"Toronto": "Toronto", "New York City": "NYC", "Montreal": "Québec",
        "Quebec City": "Québec", "Vancouver": "BC", "Victoria": "BC", "Singapore": "Singapore"}
R["five_jurisdictions"] = (M[M.city.isin(FIVE)][
    ["city", "country", "n_listings", "n_hosts", "listings_of_1listing_hosts_pct",
     "hosts_with_1_pct", "top10pct_host_share_pct", "max_host_listings", "y_strict"]]
    .round(2).to_dict("records"))
json.dump(R, open(f"{OUT}/H_hosts_summary.json", "w"), indent=1, ensure_ascii=False)

pd.set_option("display.width", 230)
C = ["group", "country", "city", "n_listings", "n_hosts", "listings_of_1listing_hosts_pct",
     "hosts_with_1_pct", "top10pct_host_share_pct", "max_host_listings", "y_strict"]
print("\n=== 15 markets with the highest share of listings held by single-listing hosts ===")
print(M.head(15)[C].to_string(index=False))
print("\n=== 15 markets with the most concentrated professional supply ===")
print(M.tail(15)[C].to_string(index=False))
print("\n=== the five jurisdictions with primary legal texts ===")
print(M[M.city.isin(FIVE)][C].to_string(index=False))
print("\n" + json.dumps({k: v for k, v in R.items() if k != "five_jurisdictions"},
                        indent=1, ensure_ascii=False))
print(f"\ntotal {(time.time()-t0)/60:.1f} min")
