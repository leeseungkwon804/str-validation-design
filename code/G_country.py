"""Secondary: country-level comparison between countries at zero display and those above 90 percent. Descriptive only; no causal claim.
Input: t1/cls/*.parquet (listing-level classes). Outputs: out/G_country.csv, out/G_country_summary.json"""
import os, sys, json, glob, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd

OUT = "./out"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
F = sorted(glob.glob("./t1/cls/*.parquet"))
d = pd.concat((pd.read_parquet(p, columns=["lic_class", "y_license", "y_strict", "y_any",
                                           "m_city", "m_country", "m_group", "m_side"]) for p in F),
              ignore_index=True)
print(f"{len(d):,} rows, countries {d.m_country.nunique()}, markets {d.m_city.nunique()}", flush=True)

c = (d.groupby(["m_group", "m_country"], observed=True)
       .agg(n_markets=("m_city", "nunique"), listing_snapshots=("y_any", "size"),
            display_rate_any=("y_any", "mean"), display_rate_loose=("y_license", "mean"),
            display_rate_strict=("y_strict", "mean")).reset_index())
cls = (d.groupby(["m_country", "lic_class"], observed=True).size()
         .unstack("lic_class", fill_value=0))
ne = cls.drop(columns=["empty"], errors="ignore")
sh = ne.div(ne.sum(axis=1).replace(0, np.nan), axis=0).add_prefix("ne_")
c = c.merge(sh.reset_index(), left_on="m_country", right_on="m_country", how="left")
for col in [x for x in c.columns if x.startswith("display rate") or x.startswith("ne_")]:
    c[col] = (100 * c[col]).round(2)
c = c.sort_values("display_rate_strict", ascending=False)
c.to_csv(f"{OUT}/G_country.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== by country, descending on the strict definition ===")
print(c[["m_group", "m_country", "n_markets", "listing_snapshots", "display_rate_any", "display_rate_loose",
         "display_rate_strict", "ne_code", "ne_exempt"]].to_string(index=False))

hi = c[c.display_rate_strict >= 90]; lo = c[c.display_rate_any < 1]
S = {"n_countries": int(len(c)),
     "high_90plus": hi[["m_country", "m_group", "n_markets", "display_rate_strict"]].to_dict("records"),
     "zero_countries": lo[["m_country", "m_group", "n_markets", "display_rate_any"]].to_dict("records"),
     "median_country_strict_pct": round(float(c.display_rate_strict.median()), 2),
     "spread_pct": [round(float(c.display_rate_strict.min()), 2), round(float(c.display_rate_strict.max()), 2)],
     "note": "Descriptive only; no causal relation to the presence of a registration scheme is claimed."}
json.dump(S, open(f"{OUT}/G_country_summary.json", "w"), indent=1, ensure_ascii=False)
print("\ncountries above 90%:", [r["m_country"] for r in S["high_90plus"]])
print("countries at zero:", [r["m_country"] for r in S["zero_countries"]])
print(f"\ntotal {(time.time()-t0)/60:.1f} min")
