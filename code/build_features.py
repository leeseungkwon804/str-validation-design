"""Feature construction: 572 snapshots into one listing-level table.
Note: market identity (city, country, latitude, longitude, neighbourhood) is excluded from the predictors; measuring that leakage is the point of the study.
   kept as metadata and used only to construct the splits."""
import pandas as pd, numpy as np, os, re, glob, sys, time, gc

OUT="./feat"; os.makedirs(OUT, exist_ok=True)
pm=pd.read_csv("./panel_files.csv")
pm=pm[pm.path.notna()].reset_index(drop=True)

META=["id","host_id","neighbourhood_cleansed","latitude","longitude"]
NUM=["accommodates","bathrooms","bedrooms","beds","minimum_nights","maximum_nights",
     "availability_30","availability_60","availability_90","availability_365",
     "number_of_reviews","number_of_reviews_ltm","number_of_reviews_l30d",
     "review_scores_rating","review_scores_accuracy","review_scores_cleanliness",
     "review_scores_checkin","review_scores_communication","review_scores_location",
     "review_scores_value","reviews_per_month","host_listings_count","host_total_listings_count",
     "calculated_host_listings_count","calculated_host_listings_count_entire_homes",
     "calculated_host_listings_count_private_rooms","calculated_host_listings_count_shared_rooms",
     "estimated_occupancy_l365d","estimated_revenue_l365d"]
CAT=["property_type","room_type","host_response_time"]
FLG=["host_is_superhost","host_has_profile_pic","host_identity_verified",
     "instant_bookable","has_availability"]
PCT=["host_response_rate","host_acceptance_rate"]
TXT=["name","description","neighborhood_overview","host_about","amenities"]
DAT=["host_since","first_review","last_review","last_scraped"]
USE=list(dict.fromkeys(META+NUM+CAT+FLG+PCT+TXT+DAT+["price","license","host_verifications"]))

def pct(s):  return pd.to_numeric(s.astype("string").str.rstrip("%"), errors="coerce")
def money(s):return pd.to_numeric(s.astype("string").str.replace(r"[^0-9.]","",regex=True), errors="coerce")
def tf(s):   return s.astype("string").str.lower().map({"t":1,"true":1,"f":0,"false":0})

done=sorted(glob.glob(f"{OUT}/*.parquet"))
have={os.path.basename(p)[:-8] for p in done}
print(f"targets {len(pm)}, already done {len(have)}", flush=True)
t0=time.time(); nrow=0
for i,r in enumerate(pm.itertuples(),1):
    key=f"{r.country}__{r.city}__{r.publish_date}".replace(" ","_").replace("/","-")
    if key in have: continue
    try:
        d=pd.read_csv(r.path, low_memory=False,
                      usecols=lambda c: c in USE, compression="gzip")
    except Exception as e:
        print(f"  !! {key}: {type(e).__name__}", flush=True); continue
    o=pd.DataFrame(index=d.index)
    # outcome
    lic=d.get("license", pd.Series(index=d.index, dtype="object")).astype("string").str.strip()
    o["y_license"]=((lic.notna()) & (lic!="") & (lic.str.lower()!="nan")
                    & (~lic.str.lower().str.fullmatch(r"exempt|none|n/?a", na=False))).astype("int8")
    o["license_raw_len"]=lic.fillna("").str.len().astype("int16")
    # predictors
    for c in NUM: o[c]=pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    for c in PCT: o[c]=pct(d[c]) if c in d else np.nan
    for c in FLG: o[c]=tf(d[c]) if c in d else np.nan
    for c in CAT: o[c]=d[c].astype("string") if c in d else pd.NA
    o["price"]=money(d["price"]) if "price" in d else np.nan
    for c in TXT: o[f"len_{c}"]=d[c].astype("string").fillna("").str.len().astype("int32") if c in d else 0
    o["n_amenities"]=(d["amenities"].astype("string").fillna("").str.count(",")+1) if "amenities" in d else 0
    o["n_host_verif"]=(d["host_verifications"].astype("string").fillna("").str.count(",")+1) if "host_verifications" in d else 0
    ref=pd.to_datetime(d["last_scraped"], errors="coerce") if "last_scraped" in d else pd.NaT
    for c in ("host_since","first_review","last_review"):
        if c in d:
            o[f"days_{c}"]=(ref-pd.to_datetime(d[c], errors="coerce")).dt.days
        else: o[f"days_{c}"]=np.nan
    # metadata, used for splitting only, never as a predictor
    for c in META: o[f"m_{c}"]=d[c] if c in d else pd.NA
    o["m_city"]=r.city; o["m_country"]=r.country; o["m_group"]=r.group
    o["m_side"]=r.side; o["m_date"]=r.publish_date
    o.to_parquet(f"{OUT}/{key}.parquet", index=False)
    nrow+=len(o); del d,o; gc.collect()
    if i%20==0:
        el=time.time()-t0
        print(f"  {i}/{len(pm)}  rows so far {nrow:,}  elapsed {el/60:.1f} min  about {(el/max(i,1))*(len(pm)-i)/60:.0f} min left", flush=True)
print(f"\ndone. shards {len(glob.glob(f'{OUT}/*.parquet'))}, rows this run {nrow:,}, {(time.time()-t0)/60:.1f} min", flush=True)
