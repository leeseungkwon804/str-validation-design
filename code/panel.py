"""Panel assembly: combines the existing 262 and the new 310 snapshots and counts the market-by-wave structure."""
import pandas as pd, glob, os, re
C="./data"
NEW="./raw"
s=pd.read_csv(f"{C}/catalog/snapshots.csv")
log=pd.read_csv(f"{C}/catalog/download_log.csv", header=None,
                names=["path","url","status","bytes","md5","code","secs"])
old_ok=set(log[log.status=="ok"].url)
nl=pd.read_csv("./download_log.csv", header=None,
               names=["path","url","status","bytes","md5","code","secs"])
new_ok=set(nl[nl.status=="ok"].url)
s["have"]=s.listings_url.isin(old_ok|new_ok)
s["src"]=["old" if u in old_ok else ("new" if u in new_ok else "-") for u in s.listings_url]
print(f"catalogue {len(s)}, held {int(s.have.sum())}, missing {int((~s.have).sum())}")
print("\n=== by group (markets / snapshots) ===")
g=s[s.have].groupby("group").agg(markets=("city","nunique"), snaps=("city","size"))
print(g.to_string())
print("\n=== panel with four waves (two pre, two post) ===")
m=s[s.have & s.side.isin(["pre","post"])]
w=m.pivot_table(index=["group","city"], columns="side", values="publish_date", aggfunc="nunique").fillna(0)
w["ok"]=(w.get("pre",0)>=2)&(w.get("post",0)>=2)
print(w.groupby("group").agg(n_markets=("ok","size"), both_sides_two_waves=("ok","sum")).to_string())
bal=w[w.ok].reset_index().groupby("group").city.nunique()
print("\nmarkets in the balanced panel:", dict(bal))
s.to_csv("./snapshots_have.csv", index=False)
# file path mapping
rows=[]
def fn(r):
    f=lambda x: str(x).replace(" ","_").replace("/","-")
    return f"{f(r.country)}__{f(r.city)}__{r.publish_date}.csv.gz"
for r in s[s.have].itertuples():
    p1=os.path.join(f"{C}/raw", fn(r)); p2=os.path.join(NEW, fn(r))
    p = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
    rows.append({"city":r.city,"country":r.country,"group":r.group,"side":r.side,
                 "publish_date":r.publish_date,"path":p})
pm=pd.DataFrame(rows)
print(f"\npath check: found {pm.path.notna().sum()} / missing {pm.path.isna().sum()}")
pm.to_csv("./panel_files.csv", index=False)
