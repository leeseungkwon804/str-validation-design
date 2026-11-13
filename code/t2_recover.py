"""T2. Recovery of the 13 snapshots that could not be read.
Cause: the older catalogue stored file names without parentheses ('Ireland_nationwide') while the newer panel.py kept them
The old catalogue stored the nationwide markets without parentheses, so 13 files for Ireland, Malta and New Zealand failed to match.
The files were all present; path_index() in c10lib.py absorbs the parenthesis and separator differences.
This script rebuilds the path table and then builds only the missing feature shards, with the same rules as build_features.py.
Note: market identity is stored only as m_-prefixed metadata and is not a predictor."""
import sys, os, glob, time, gc, json
sys.path.insert(0, ".")
import pandas as pd, numpy as np, c10lib

FEAT = "./feat"; OUT = "./t2"; os.makedirs(OUT, exist_ok=True)
META = ["id", "host_id", "neighbourhood_cleansed", "latitude", "longitude"]
NUM = ["accommodates","bathrooms","bedrooms","beds","minimum_nights","maximum_nights",
       "availability_30","availability_60","availability_90","availability_365",
       "number_of_reviews","number_of_reviews_ltm","number_of_reviews_l30d",
       "review_scores_rating","review_scores_accuracy","review_scores_cleanliness",
       "review_scores_checkin","review_scores_communication","review_scores_location",
       "review_scores_value","reviews_per_month","host_listings_count","host_total_listings_count",
       "calculated_host_listings_count","calculated_host_listings_count_entire_homes",
       "calculated_host_listings_count_private_rooms","calculated_host_listings_count_shared_rooms",
       "estimated_occupancy_l365d","estimated_revenue_l365d"]
CAT = ["property_type", "room_type", "host_response_time"]
FLG = ["host_is_superhost","host_has_profile_pic","host_identity_verified","instant_bookable","has_availability"]
PCT = ["host_response_rate", "host_acceptance_rate"]
TXT = ["name","description","neighborhood_overview","host_about","amenities"]
DAT = ["host_since","first_review","last_review","last_scraped"]
USE = list(dict.fromkeys(META+NUM+CAT+FLG+PCT+TXT+DAT+["price","license","host_verifications"]))

pct   = lambda s: pd.to_numeric(s.astype("string").str.rstrip("%"), errors="coerce")
money = lambda s: pd.to_numeric(s.astype("string").str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
tf    = lambda s: s.astype("string").str.lower().map({"t":1,"true":1,"f":0,"false":0})

def build(r):
    key = c10lib.shard_key(r.country, r.city, r.publish_date)
    d = pd.read_csv(r.path, low_memory=False, usecols=lambda c: c in USE, compression="gzip")
    o = pd.DataFrame(index=d.index)
    lic = d.get("license", pd.Series(index=d.index, dtype="object")).astype("string").str.strip()
    o["y_license"] = ((lic.notna()) & (lic != "") & (lic.str.lower() != "nan")
                      & (~lic.str.lower().str.fullmatch(r"exempt|none|n/?a", na=False))).astype("int8")
    o["license_raw_len"] = lic.fillna("").str.len().astype("int16")
    for c in NUM: o[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    for c in PCT: o[c] = pct(d[c]) if c in d else np.nan
    for c in FLG: o[c] = tf(d[c]) if c in d else np.nan
    for c in CAT: o[c] = d[c].astype("string") if c in d else pd.NA
    o["price"] = money(d["price"]) if "price" in d else np.nan
    for c in TXT: o[f"len_{c}"] = d[c].astype("string").fillna("").str.len().astype("int32") if c in d else 0
    o["n_amenities"]  = (d["amenities"].astype("string").fillna("").str.count(",") + 1) if "amenities" in d else 0
    o["n_host_verif"] = (d["host_verifications"].astype("string").fillna("").str.count(",") + 1) if "host_verifications" in d else 0
    ref = pd.to_datetime(d["last_scraped"], errors="coerce") if "last_scraped" in d else pd.NaT
    for c in ("host_since", "first_review", "last_review"):
        o[f"days_{c}"] = (ref - pd.to_datetime(d[c], errors="coerce")).dt.days if c in d else np.nan
    for c in META: o[f"m_{c}"] = d[c] if c in d else pd.NA
    o["m_city"] = r.city; o["m_country"] = r.country; o["m_group"] = r.group
    o["m_side"] = r.side; o["m_date"] = r.publish_date
    o.to_parquet(f"{FEAT}/{key}.parquet", index=False)
    return key, len(o)

if __name__ == "__main__":
    t0 = time.time()
    s = c10lib.resolve()
    assert s.path.notna().all(), "paths still unresolved"
    s.to_csv(f"{OUT}/panel_files_v2.csv", index=False)
    have = {os.path.basename(p)[:-8] for p in glob.glob(f"{FEAT}/*.parquet")}
    todo = [r for r in s.itertuples() if c10lib.shard_key(r.country, r.city, r.publish_date) not in have]
    print(f"snapshots {len(s)}, existing shards {len(have)}, to build {len(todo)}", flush=True)
    rows = []
    for r in todo:
        k, n = build(r)
        rows.append({"key": k, "city": r.city, "country": r.country, "group": r.group,
                     "side": r.side, "date": r.publish_date, "n": n})
        print(f"  + {k}  {n:,} rows", flush=True)
        gc.collect()
    pd.DataFrame(rows).to_csv(f"{OUT}/recovered_shards.csv", index=False)
    F = sorted(glob.glob(f"{FEAT}/*.parquet"))
    tot = sum(pd.read_parquet(p, columns=["y_license"]).shape[0] for p in F)
    print(f"\ndone. shards {len(F)}, rows {tot:,}, recovered {len(rows)} files with {sum(r['n'] for r in rows):,} rows, {(time.time()-t0)/60:.1f} min")
    json.dump({"shards": len(F), "rows": int(tot), "recovered": len(rows),
               "recovered_rows": int(sum(r["n"] for r in rows)),
               "markets": int(s.city.nunique())},
              open(f"{OUT}/t2_summary.json", "w"), indent=1, ensure_ascii=False)
