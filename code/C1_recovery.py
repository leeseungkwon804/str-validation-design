"""C1. Market-identity recovery: can the 50 predictors alone name the market, and which of them carry that information?
 (1) market (123) and country (37) classification accuracy against the chance rate
 (2) contribution, part one: permutation importance (drop in accuracy), one variable at a time
 (3) contribution, part two: each predictor group alone and each group removed
Training and test rows are separated by listing, so that two waves of the same listing never straddle the split.
Market identity is the target here. It is never a predictor in the outcome models.
Each step writes to CSV immediately and finished steps are skipped, so an interrupted run can resume."""
import os, sys, json, time, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, prep
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

OUT = "./out"; os.makedirs(OUT, exist_ok=True)
NJ = int(os.environ.get("C_NJOBS", "24"))
CAP = 6000
RES = f"{OUT}/C1_market_recovery.csv"
T0 = time.time()
X, y_lic, y_str, meta, sch = prep.load()
cols = sch["columns"]
city = meta.city.values; country = meta.country.values; lid = meta.lid.values
del y_lic, y_str; gc.collect()

GROUPS = {
    "Price": ["price"],
    "Minimum and maximum stay": ["minimum_nights", "maximum_nights"],
    "Property type": ["property_type", "room_type", "accommodates", "bedrooms", "beds", "bathrooms"],
    "Availability": ["availability_30", "availability_60", "availability_90", "availability_365",
               "has_availability", "estimated_occupancy_l365d", "estimated_revenue_l365d"],
    "Reviews": ["number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d",
             "reviews_per_month", "review_scores_rating", "review_scores_accuracy",
             "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication",
             "review_scores_location", "review_scores_value"],
    "Host portfolio size": ["host_listings_count", "host_total_listings_count", "calculated_host_listings_count",
                   "calculated_host_listings_count_entire_homes",
                   "calculated_host_listings_count_private_rooms",
                   "calculated_host_listings_count_shared_rooms"],
    "Host conduct": ["host_is_superhost", "host_has_profile_pic", "host_identity_verified",
                   "instant_bookable", "host_response_time", "host_response_rate",
                   "host_acceptance_rate", "n_host_verif"],
    "Text length": ["len_name", "len_description", "len_neighborhood_overview", "len_host_about",
                   "len_amenities", "n_amenities"],
    "Tenure": ["days_host_since", "days_first_review", "days_last_review"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(cols)
IDX = {g: [cols.index(c) for c in v] for g, v in GROUPS.items()}
ALL = list(range(len(cols)))
json.dump(GROUPS, open(f"{OUT}/C1_groups.json", "w"), indent=1, ensure_ascii=False)

def sample(lab, seed=0, cap=CAP):
    rs = np.random.RandomState(seed)
    take = [rs.choice(np.where(lab == v)[0], min(cap, int((lab == v).sum())), replace=False)
            for v in np.unique(lab)]
    idx = np.sort(np.concatenate(take))
    gtr, gte = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed)
                    .split(idx, lab[idx], groups=lid[idx]))
    return idx[gtr], idx[gte]

TR_CITY, TE_CITY = sample(city)
TR_CTRY, TE_CTRY = sample(country)

done = set()
if os.path.exists(RES):
    done = set(pd.read_csv(RES)["target"])
    print(f"{len(done)} items already done, skipping", flush=True)

def append(row):
    pd.DataFrame([row]).to_csv(RES, mode="a", header=not os.path.exists(RES), index=False)

def run(name, lab, tr, te, keep=None, trees=150, want_model=False):
    if name in done:
        print(f"  (skipped) {name}", flush=True); return None
    use = ALL if keep is None else keep
    m = Pipeline([("i", SimpleImputer(strategy="median")),
                  ("m", RandomForestClassifier(n_estimators=trees, min_samples_leaf=5,
                                               n_jobs=NJ, random_state=0))])
    t1 = time.time()
    m.fit(np.asarray(X[tr])[:, use], lab[tr])
    Xte = np.asarray(X[te])[:, use]
    P = m.predict_proba(Xte)
    pred = m.classes_[P.argmax(1)]
    row = {"target": name, "n_variables": len(use), "n_classes": int(len(np.unique(lab[tr]))),
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "accuracy": round(float(accuracy_score(lab[te], pred)), 4),
           "top5": round(float(top_k_accuracy_score(lab[te], P, k=5, labels=m.classes_)), 4),
           "macroF1": round(float(f1_score(lab[te], pred, average="macro")), 4),
           "chance_rate": round(1 / len(np.unique(lab[tr])), 4),
           "most_frequent_class": round(float(pd.Series(lab[tr]).value_counts(normalize=True).iloc[0]), 4),
           "seconds": round(time.time() - t1, 1)}
    append(row); print("  " + json.dumps(row, ensure_ascii=False), flush=True)
    out = (m, Xte, pred, P) if want_model else None
    if not want_model:
        del m, Xte, P; gc.collect()
    return out

print("=== C1-1. recovery using all predictors ===", flush=True)
got = run("Market (city)", city, TR_CITY, TE_CITY, want_model=True)
run("Country", country, TR_CTRY, TE_CTRY)

# ---- permutation importance, one variable at a time ----
PIMP = f"{OUT}/C1_permutation_importance.csv"
if got is not None and not os.path.exists(PIMP):
    m, Xte, pred, P = got
    rs = np.random.RandomState(0)
    sel = rs.choice(len(TE_CITY), min(60000, len(TE_CITY)), replace=False)
    print("=== C1-2. permutation importance (drop in accuracy, 3 repetitions, 60,000 test rows) ===", flush=True)
    t1 = time.time()
    pi = permutation_importance(m, Xte[sel], city[TE_CITY][sel], n_repeats=3,
                                random_state=0, scoring="accuracy", n_jobs=1)
    imp = pd.DataFrame({"variable": cols, "accuracy_drop": pi.importances_mean,
                        "sd": pi.importances_std}).sort_values("accuracy_drop", ascending=False)
    imp["group"] = imp.variable.map({c: g for g, v in GROUPS.items() for c in v})
    imp.round(5).to_csv(PIMP, index=False)
    print(imp.head(15).round(4).to_string(index=False), flush=True)
    print(f"  {time.time()-t1:.0f}seconds", flush=True)
    gini = pd.DataFrame({"variable": cols, "gini_importance": m.named_steps["m"].feature_importances_})
    gini.sort_values("gini_importance", ascending=False).round(5).to_csv(
        f"{OUT}/C1_city_classifier_importance.csv", index=False)
    cf = pd.DataFrame({"true": city[TE_CITY], "pred": pred})
    (cf[cf.true != cf.pred].groupby(["true", "pred"]).size().sort_values(ascending=False)
       .head(40).rename("n").reset_index().to_csv(f"{OUT}/C1_top_confusions.csv", index=False))
    acc_by = (pd.DataFrame({"city": city[TE_CITY], "hit": (cf.true == cf.pred).astype(int)})
                .groupby("city").hit.agg(["mean", "size"]).round(4).reset_index()
                .rename(columns={"mean": "accuracy", "size": "n"}))
    acc_by.sort_values("accuracy", ascending=False).to_csv(f"{OUT}/C1_accuracy_by_market.csv", index=False)
    del m, Xte, P, got; gc.collect()
elif os.path.exists(PIMP):
    print("=== C1-2. permutation importance already present, skipped ===", flush=True)

print("=== C1-3. group removed ===", flush=True)
for g in GROUPS:
    run(f"Market, removed: {g}", city, TR_CITY, TE_CITY,
        keep=[i for i in ALL if i not in IDX[g]], trees=100)
print("=== C1-4. group alone ===", flush=True)
for g in GROUPS:
    run(f"Market, alone: {g}", city, TR_CITY, TE_CITY, keep=IDX[g], trees=100)

R = pd.read_csv(RES).drop_duplicates("target", keep="last")
R.to_csv(RES, index=False)
print("\n=== C1 summary ===")
pd.set_option("display.width", 200)
print(R.to_string(index=False))
print(f"\ntotal {(time.time()-T0)/60:.1f} min")
