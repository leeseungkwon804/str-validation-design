"""L. Why the text-length group is a market fingerprint: testing whether description length clusters by language area.
The text-length group alone recovers the market for 12.1% of listings, and removing it costs 3.3 percentage points,
and len_host_about and len_description rank high in permutation importance.

No external language API, dictionary or translation service is used. Only in-data evidence:
   Unicode character-set shares and diacritic frequencies only; nothing beyond what they support is claimed,
   and if the evidence does not support the hypothesis it is recorded as not confirmed rather than guessed at.

Outputs: out/L_text_by_snapshot.csv, out/L_text_by_market.csv, out/L_language_summary.json
"""
import sys, os, re, json, time, collections, unicodedata
sys.path.insert(0, ".")
import pandas as pd, numpy as np, c10lib
from concurrent.futures import ProcessPoolExecutor

OUT = "./out"; os.makedirs(OUT, exist_ok=True)
CAP_CHARS = 3_000_000          # maximum number of characters scanned per snapshot

# Unicode blocks, for character-set shares
BLOCKS = [
    ("ascii_alpha", lambda o: (65 <= o <= 90) or (97 <= o <= 122)),
    ("latin_ext",   lambda o: (0x00C0 <= o <= 0x024F)),
    ("greek",       lambda o: (0x0370 <= o <= 0x03FF)),
    ("cyrillic",    lambda o: (0x0400 <= o <= 0x04FF)),
    ("hebrew",      lambda o: (0x0590 <= o <= 0x05FF)),
    ("arabic",      lambda o: (0x0600 <= o <= 0x06FF)),
    ("devanagari",  lambda o: (0x0900 <= o <= 0x097F)),
    ("thai",        lambda o: (0x0E00 <= o <= 0x0E7F)),
    ("kana",        lambda o: (0x3040 <= o <= 0x30FF)),
    ("cjk",         lambda o: (0x3400 <= o <= 0x9FFF)),
    ("hangul",      lambda o: (0xAC00 <= o <= 0xD7A3) or (0x1100 <= o <= 0x11FF) or (0x3130 <= o <= 0x318F)),
]
# diacritic letters that separate language areas within the Latin script (character-set evidence, not a dictionary)
MARKS = list("ñçßøåæãõığşůřąęłżśćüöäéèàìòùêôîûœ")
MARKSET = set(MARKS)

def scan(text):
    c = collections.Counter()
    letters = 0
    for ch in text:
        o = ord(ch)
        if ch.isalpha():
            letters += 1
            lo = ch.lower()
            if lo in MARKSET: c["mark_" + lo] += 1
        for nm, f in BLOCKS:
            if f(o): c[nm] += 1; break
    c["letters"] = letters
    c["chars"] = len(text)
    return c

def one(args):
    country, city, date, grp, side, path = args
    try:
        d = pd.read_csv(path, usecols=lambda x: x in ("description", "name", "host_about",
                                                      "neighborhood_overview"),
                        compression="gzip", low_memory=False)
    except Exception as e:
        return {"city": city, "date": date, "note": f"ERR:{type(e).__name__}"}
    r = {"city": city, "country": country, "group": grp, "side": side, "date": date,
         "n": int(len(d)), "note": ""}
    for col in ("description", "name", "host_about", "neighborhood_overview"):
        s = d[col].astype("string").fillna("") if col in d.columns else pd.Series([""] * len(d), dtype="string")
        L = pd.to_numeric(s.str.len(), errors="coerce").fillna(0).to_numpy(dtype=float)
        pos = L[L > 0]
        r[f"{col}_nonempty_pct"] = round(100 * float((L > 0).mean()), 3)
        r[f"{col}_mean_len"] = round(float(pos.mean()), 2) if pos.size else 0.0
        r[f"{col}_median_len"] = float(np.median(pos)) if pos.size else 0.0
        r[f"{col}_p90_len"] = float(np.quantile(pos, 0.9)) if pos.size else 0.0
    # character sets are scanned on the description field only, for cost
    s = d["description"].astype("string").fillna("") if "description" in d.columns else pd.Series([], dtype="string")
    txt = "\n".join(s[s.str.len() > 0].tolist())[:CAP_CHARS]
    c = scan(txt)
    tot = max(c["letters"], 1)
    for nm, _ in BLOCKS:
        r[f"sh_{nm}"] = round(c[nm] / tot, 5)
    for m in MARKS:
        r[f"mk_{m}"] = round(1000 * c["mark_" + m] / tot, 4)
    r["scanned_chars"] = int(c["chars"])
    # check whether lengths pile up at a platform cap such as 500 characters
    L = pd.to_numeric(s.str.len(), errors="coerce").fillna(0).to_numpy(dtype=float)
    L = L[L > 0]
    r["pct_len_exactly_500"] = round(100 * float((L == 500).mean()), 3) if L.size else 0.0
    r["pct_len_ge_1000"] = round(100 * float((L >= 1000).mean()), 3) if L.size else 0.0
    r["max_len"] = int(L.max()) if L.size else 0
    return r

if __name__ == "__main__":
    t0 = time.time()
    sdf = c10lib.resolve()
    jobs = [(r.country, r.city, r.publish_date, r.group, r.side, r.path) for r in sdf.itertuples()]
    rows = []
    with ProcessPoolExecutor(20) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            rows.append(r)
            if i % 100 == 0: print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}seconds", flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(f"{OUT}/L_text_by_snapshot.csv", index=False)
    print(f"snapshots {len(S)}, failures {int((S.note != '').sum())}, {(time.time()-t0)/60:.1f} min", flush=True)

    num = [c for c in S.columns if c not in ("city", "country", "group", "side", "date", "note")]
    M = S.groupby(["group", "country", "city"], as_index=False)[num].mean().round(4)
    M.to_csv(f"{OUT}/L_text_by_market.csv", index=False)

    # ---- is it a market constant? between-market variance against within-market variance across waves ----
    def icc(col):
        g = S.groupby("city")[col]
        within = float(g.transform(lambda x: x - x.mean()).var())
        between = float(g.mean().var())
        return round(between / (between + within), 4) if (between + within) > 0 else np.nan
    R = {"n_markets": int(len(M)), "n_snapshots": int(len(S)),
         "ICC_market": {c: icc(c) for c in ["description_mean_len", "host_about_mean_len",
                                            "name_mean_len", "neighborhood_overview_mean_len",
                                            "sh_ascii_alpha", "sh_latin_ext", "sh_cjk"]}}

    # ---- cluster markets on character-set shares, using in-data evidence only ----
    prof_cols = [f"sh_{n}" for n, _ in BLOCKS] + [f"mk_{m}" for m in MARKS]
    P = M[prof_cols].fillna(0).values
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    Z = StandardScaler().fit_transform(P)
    best = None
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Z)
        y = np.log(M.description_mean_len.replace(0, np.nan)).fillna(0).values
        grand = y.mean()
        ss_tot = ((y - grand) ** 2).sum()
        ss_bet = sum(((y[km.labels_ == c].mean() - grand) ** 2) * (km.labels_ == c).sum()
                     for c in range(k))
        eta2 = float(ss_bet / ss_tot)
        if best is None or k == 6: pass
        R.setdefault("kmeans_eta2_logdesclen", {})[k] = round(eta2, 4)
        if k == 6: best = km
    M["script_cluster"] = best.labels_
    # describe each cluster by its character sets only
    desc = []
    for c in sorted(set(best.labels_)):
        sub = M[M.script_cluster == c]
        top_blocks = (sub[[f"sh_{n}" for n, _ in BLOCKS]].mean()
                        .sort_values(ascending=False).head(3))
        top_marks = (sub[[f"mk_{m}" for m in MARKS]].mean()
                       .sort_values(ascending=False).head(5))
        desc.append({"cluster": int(c), "n_markets": int(len(sub)),
                     "markets": sorted(sub.city.tolist())[:40],
                     "countries": sorted(set(sub.country)),
                     "top_blocks": {k.replace("sh_", ""): round(float(v), 4) for k, v in top_blocks.items()},
                     "top_marks_per_1000_letters": {k.replace("mk_", ""): round(float(v), 3)
                                                    for k, v in top_marks.items()},
                     "description_mean_len": round(float(sub.description_mean_len.mean()), 1),
                     "description_mean_len_range": [round(float(sub.description_mean_len.min()), 1),
                                                    round(float(sub.description_mean_len.max()), 1)]})
    R["clusters_k6"] = desc
    y = np.log(M.description_mean_len.replace(0, np.nan)).fillna(0).values
    R["eta2_logdesclen_by_k6_cluster"] = R["kmeans_eta2_logdesclen"][6]
    # what if country is used instead of the cluster? (an upper reference)
    cc, inv = np.unique(M.country.values, return_inverse=True)
    grand = y.mean(); ss_tot = ((y - grand) ** 2).sum()
    ss_bet = sum(((y[inv == i].mean() - grand) ** 2) * (inv == i).sum() for i in range(len(cc)))
    R["eta2_logdesclen_by_country"] = round(float(ss_bet / ss_tot), 4)
    R["desc_len_spread"] = {"min": round(float(M.description_mean_len.min()), 1),
                            "median": round(float(M.description_mean_len.median()), 1),
                            "max": round(float(M.description_mean_len.max()), 1)}
    R["truncation_check"] = {"pct_len_exactly_500_mean": round(float(S.pct_len_exactly_500.mean()), 3),
                             "pct_len_ge_1000_mean": round(float(S.pct_len_ge_1000.mean()), 3),
                             "max_len_overall": int(S.max_len.max())}
    M.to_csv(f"{OUT}/L_text_by_market.csv", index=False)
    json.dump(R, open(f"{OUT}/L_language_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in R.items() if k != "clusters_k6"}, indent=1, ensure_ascii=False))
    for c in R["clusters_k6"]:
        print(f"\n--- cluster {c['cluster']}  markets {c['n_markets']}  mean description length {c['description_mean_len']} characters "
              f"(range {c['description_mean_len_range']})")
        print("   character sets:", c["top_blocks"])
        print("   diacritics per 1000 letters:", c["top_marks_per_1000_letters"])
        print("   Country:", ", ".join(c["countries"]))
    print(f"\ntotal {(time.time()-t0)/60:.1f} min")
