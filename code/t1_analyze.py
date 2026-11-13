"""T1-C. Does the licence field mean the same thing in every market? Validity of the outcome definition.
Inputs: t1/cls/*.parquet (listing-level classes), t1/strings_global.csv, t1/strings_top30_by_market.csv
Outputs: t1/*.csv, t1/t1_summary.json"""
import sys, os, re, json, glob, time, collections
sys.path.insert(0, ".")
import pandas as pd, numpy as np, c10lib
from t1_classify import ORDER, value_class

OUT = "./t1"
F = sorted(glob.glob(f"{OUT}/cls/*.parquet"))
print(f"reading {len(F)} shards...", flush=True)
d = pd.concat((pd.read_parquet(p) for p in F), ignore_index=True)
print(f"  {len(d):,} rows", flush=True)
N = len(d)
S = {}

# ---------- 1. ladder of outcome definitions ----------
S["n_listing_snapshot"] = N
S["n_snapshot"] = int(d.groupby(["m_city", "m_date"]).ngroups)
S["n_market"] = int(d.m_city.nunique())
for c in ("y_any", "y_license", "y_strict"):
    S[f"prev_{c}"] = round(100 * float(d[c].mean()), 2)
cls = d.lic_class.value_counts().reindex(ORDER).fillna(0).astype(int)
S["class_counts"] = {k: int(v) for k, v in cls.items()}
S["class_pct"] = {k: round(100 * v / N, 3) for k, v in cls.items()}

# ---------- 2. composition of the loose definition's false positives ----------
pos = d[d.y_license == 1]
mix = pos.lic_class.value_counts().reindex(ORDER).fillna(0).astype(int)
S["y_license_pos"] = int(len(pos))
S["y_license_pos_composition"] = {k: int(v) for k, v in mix.items()}
S["y_license_pos_noncode_n"] = int(len(pos) - mix["code"])
S["y_license_pos_noncode_pct"] = round(100 * (len(pos) - mix["code"]) / len(pos), 2)
neg = d[d.y_license == 0]
S["y_license_neg_code_n"] = int((neg.lic_class == "code").sum())   # codes the loose definition misses
agree = float((d.y_license == d.y_strict).mean())
po, pe = agree, float(d.y_license.mean() * d.y_strict.mean() + (1 - d.y_license.mean()) * (1 - d.y_strict.mean()))
S["y_license_vs_strict_agree_pct"] = round(100 * po, 3)
S["y_license_vs_strict_kappa"] = round((po - pe) / (1 - pe), 4)

# ---------- 3. composition by market ----------
g = d.groupby(["m_group", "m_country", "m_city"], observed=True)
mk = g.agg(n=("y_any", "size"), y_any=("y_any", "mean"), y_license=("y_license", "mean"),
           y_strict=("y_strict", "mean")).reset_index()
comp = (d.groupby(["m_group", "m_country", "m_city", "lic_class"], observed=True)
          .size().unstack("lic_class", fill_value=0).reindex(columns=ORDER, fill_value=0))
compn = comp.div(comp.sum(axis=1), axis=0)
nonempty = comp.drop(columns=["empty"])
ne_tot = nonempty.sum(axis=1)
ne_sh = nonempty.div(ne_tot.replace(0, np.nan), axis=0)
side = pd.concat([compn.add_prefix("sh_"), ne_sh.add_prefix("ne_"),
                  ne_tot.rename("nonempty_n")], axis=1).reset_index()
mk = mk.merge(side, on=["m_group", "m_country", "m_city"], how="left")
mk["delta_license_strict_pp"] = 100 * (mk.y_license - mk.y_strict)
mk["delta_any_strict_pp"] = 100 * (mk.y_any - mk.y_strict)
mk.sort_values("delta_license_strict_pp", ascending=False).to_csv(f"{OUT}/market_license_class.csv", index=False)

S["markets_all_empty"] = int((mk.y_any < 0.005).sum())
S["markets_ge90"] = int((mk.y_any > 0.90).sum())
S["markets_noncode_ge10pct_of_pos"] = int((mk.ne_code < 0.90).sum())
S["markets_noncode_ge25pct_of_pos"] = int((mk.ne_code < 0.75).sum())
S["market_delta_license_strict_pp"] = {
    "mean": round(float(mk.delta_license_strict_pp.mean()), 2),
    "median": round(float(mk.delta_license_strict_pp.median()), 2),
    "max": round(float(mk.delta_license_strict_pp.max()), 2),
    "n_markets_gt1pp": int((mk.delta_license_strict_pp > 1).sum()),
    "n_markets_gt5pp": int((mk.delta_license_strict_pp > 5).sum())}

# ---------- 4. ICC for the three outcome definitions ----------
def icc(col):
    m = mk[["n", col]].copy()
    pbar = np.average(m[col], weights=m.n)
    within = float(np.average(m[col] * (1 - m[col]), weights=m.n))
    between = float(np.average((m[col] - pbar) ** 2, weights=m.n))
    cc = mk.groupby("m_country").apply(
        lambda x: pd.Series({"n": x.n.sum(), "p": np.average(x[col], weights=x.n)}), include_groups=False)
    cb = float(np.average((cc.p - pbar) ** 2, weights=cc.n))
    return {"prev_pct": round(100 * float(pbar), 2), "within": round(within, 4),
            "between": round(between, 4), "ICC": round(between / (within + between), 4),
            "country_share_of_between_pct": round(100 * cb / between, 1)}
S["ICC"] = {c: icc(c) for c in ("y_any", "y_license", "y_strict")}

# ---------- 5. does the code format (its signature) differ by market? ----------
sig = pd.read_csv(f"{OUT}/strings_global.csv", keep_default_na=False)
top = pd.read_csv(f"{OUT}/strings_top30_by_market.csv", keep_default_na=False)
top["cls"] = [value_class(v)[0] for v in top.license]
top.to_csv(f"{OUT}/strings_top30_by_market_classified.csv", index=False)
S["top_strings_class_mix"] = top.groupby("cls").n.sum().astype(int).to_dict()

# listing-level signature set, to test whether the shape of a code is market-specific
from t1_census import signature
by_m = collections.defaultdict(set)
for r in top.itertuples():
    if r.cls == "code":
        by_m[r.city].add(signature(r.license))
ms = {k: v for k, v in by_m.items() if v}
keys = sorted(ms)
J = []
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = ms[keys[i]], ms[keys[j]]
        J.append(len(a & b) / len(a | b))
S["code_signature_jaccard"] = {"n_market_pairs": len(J),
                               "mean": round(float(np.mean(J)), 4),
                               "median": round(float(np.median(J)), 4),
                               "pct_pairs_zero_overlap": round(100 * float(np.mean(np.array(J) == 0)), 1)}
sigrows = [{"city": k, "n_sig": len(v), "signatures": " | ".join(sorted(v)[:8])} for k, v in ms.items()]
pd.DataFrame(sigrows).sort_values("city").to_csv(f"{OUT}/code_signatures_by_market.csv", index=False)

# ---------- 6. within-market change across waves ----------
sw = d.groupby(["m_city", "m_date"], observed=True).agg(
    n=("y_any", "size"), y_any=("y_any", "mean"), y_strict=("y_strict", "mean"),
    sh_exempt=("lic_class", lambda s: float((s == "exempt").mean())),
    sh_pending=("lic_class", lambda s: float((s == "pending").mean()))).reset_index()
sw.to_csv(f"{OUT}/snapshot_license_class.csv", index=False)
rng = sw.groupby("m_city").y_strict.agg(["min", "max"])
rng["range_pp"] = 100 * (rng["max"] - rng["min"])
S["within_market_wave_range_pp"] = {"median": round(float(rng.range_pp.median()), 2),
                                    "p90": round(float(rng.range_pp.quantile(0.9)), 2),
                                    "n_markets_gt10pp": int((rng.range_pp > 10).sum())}
rng.sort_values("range_pp", ascending=False).to_csv(f"{OUT}/within_market_wave_range.csv")

json.dump(S, open(f"{OUT}/t1_summary.json", "w"), indent=1, ensure_ascii=False)
print("\n" + json.dumps(S, indent=1, ensure_ascii=False))
