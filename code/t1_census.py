"""T1-A. Full census of the licence field: frequency table and format signatures over 572 snapshots.
Purpose: a survey used to set the classification scheme from the data. Nothing is classified yet."""
import sys, os, re, json, time, collections
sys.path.insert(0, ".")
import pandas as pd, numpy as np, c10lib
from concurrent.futures import ProcessPoolExecutor

OUT = "./t1"; os.makedirs(OUT, exist_ok=True)

def signature(v):
    """A signature that keeps only the shape of the characters: 'HSR19-123456' becomes 'A{3}9{2}-9{6}'."""
    s = re.sub(r"\s+", " ", v.strip())[:80]
    m = []
    for ch in s:
        if ch.isdigit(): m.append("9")
        elif ch.isalpha(): m.append("A")
        else: m.append(ch)
    out = []
    for ch, grp in __import__("itertools").groupby(m):
        n = len(list(grp))
        out.append(ch if n == 1 else f"{ch}{{{n}}}")
    return "".join(out)

def one(args):
    city, country, date, grp, side, path = args
    try:
        d = pd.read_csv(path, usecols=lambda c: c in ("id", "license"),
                        compression="gzip", low_memory=False)
    except Exception as e:
        return (city, date, None, f"ERR:{type(e).__name__}", None, None)
    if "license" not in d.columns:
        return (city, date, len(d), "NOCOL", collections.Counter(), collections.Counter())
    s = d["license"].astype("string").str.strip()
    nn = s[s.notna() & (s != "") & (s.str.lower() != "nan")]
    cnt = collections.Counter(nn.str.slice(0, 300).tolist())
    sig = collections.Counter(signature(v) for v in nn.unique())
    return (city, date, len(d), "", cnt, sig)

if __name__ == "__main__":
    t0 = time.time()
    s = c10lib.resolve()
    jobs = [(r.city, r.country, r.publish_date, r.group, r.side, r.path) for r in s.itertuples()]
    G = collections.Counter(); GS = collections.Counter()
    bycity = collections.defaultdict(collections.Counter)
    rows = []
    with ProcessPoolExecutor(24) as ex:
        for i, (city, date, n, note, cnt, sig) in enumerate(ex.map(one, jobs), 1):
            rows.append({"city": city, "date": date, "n": n, "note": note,
                         "nonempty": (sum(cnt.values()) if cnt else 0),
                         "uniq": (len(cnt) if cnt else 0)})
            if cnt:
                G.update(cnt); GS.update(sig); bycity[city].update(cnt)
            if i % 100 == 0: print(f"  {i}/{len(jobs)} {time.time()-t0:.0f}seconds", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/census_by_snapshot.csv", index=False)
    pd.DataFrame(G.most_common(), columns=["license", "n"]).to_csv(f"{OUT}/strings_global.csv", index=False)
    pd.DataFrame(GS.most_common(), columns=["signature", "n_uniq_values"]).to_csv(f"{OUT}/signatures_global.csv", index=False)
    tc = [{"city": c, "license": v, "n": k} for c, cc in bycity.items() for v, k in cc.most_common(30)]
    pd.DataFrame(tc).to_csv(f"{OUT}/strings_top30_by_market.csv", index=False)
    print(f"\ndistinct strings {len(G):,}, signatures {len(GS):,}, non-empty {sum(G.values()):,}, {(time.time()-t0)/60:.1f} min")
    print("\n=== top 60 overall ===")
    for v, k in G.most_common(60): print(f"{k:>8,}  {v[:110]}")
