"""Shared design matrix: reads the 572 shards once and caches X, y and the metadata for the later steps.
Market identity never enters the design matrix: the m_-prefixed columns (city, country, latitude, longitude, neighbourhood, host id, listing id)
   They are held as metadata for splitting and aggregation only. license_raw_len is also excluded, being derived from the outcome."""
import sys, os, glob, time, json
sys.path.insert(0, ".")
import pandas as pd, numpy as np, c10lib

MAT = "./mat"; os.makedirs(MAT, exist_ok=True)
FEAT = "./feat"; CLS = "./t1/cls"

def build():
    t0 = time.time()
    keys = sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{FEAT}/*.parquet"))
    print(f"reading {len(keys)} shards...", flush=True)
    fs, cs = [], []
    for k in keys:
        fs.append(pd.read_parquet(f"{FEAT}/{k}.parquet"))
        cs.append(pd.read_parquet(f"{CLS}/{k}.parquet", columns=["m_id", "lic_class", "y_license", "y_strict"]))
    df = pd.concat(fs, ignore_index=True); del fs
    cl = pd.concat(cs, ignore_index=True); del cs
    print(f"  {len(df):,} rows, {time.time()-t0:.0f} s", flush=True)
    # alignment check: y_license and the listing id computed on two independent paths must agree row by row
    assert len(df) == len(cl), "row count mismatch"
    same_y = int((df.y_license.values == cl.y_license.values).sum())
    same_id = int((df.m_id.astype("string").values == cl.m_id.astype("string").values).sum())
    print(f"  alignment check: y_license matches {same_y:,}/{len(df):,}, listing id matches {same_id:,}/{len(df):,}", flush=True)
    assert same_y == len(df) and same_id == len(df), "shard rows are misaligned; stopping"

    drop = [c for c in df.columns if c.startswith("m_")] + ["y_license", "license_raw_len"]
    X = df.drop(columns=drop)
    cats = [c for c in X.columns if X[c].dtype.name in ("string", "object", "category")]
    catmaps = {}
    for c in X.columns:
        if c in cats:
            cc = X[c].astype("category")
            catmaps[c] = list(map(str, cc.cat.categories))
            X[c] = cc.cat.codes.astype(np.int32)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
    cols = list(X.columns)
    assert not any(c.startswith("m_") or c.startswith("y_") or c == "license_raw_len" for c in cols), "a forbidden column slipped into the design matrix"
    Xv = X.to_numpy(dtype=np.float32)
    np.save(f"{MAT}/X.npy", Xv)
    np.save(f"{MAT}/y_license.npy", df.y_license.to_numpy(np.int8))
    np.save(f"{MAT}/y_strict.npy", cl.y_strict.to_numpy(np.int8))
    meta = pd.DataFrame({"city": df.m_city.astype(str), "country": df.m_country.astype(str),
                         "group": df.m_group.astype(str), "side": df.m_side.astype(str),
                         "date": df.m_date.astype(str), "host": df.m_host_id.astype(str),
                         "lid": df.m_id.astype(str), "lic_class": cl.lic_class.astype(str)})
    meta.to_parquet(f"{MAT}/meta.parquet", index=False)
    json.dump({"columns": cols, "categorical": cats, "n": int(len(df)),
               "catmaps": {k: v[:50] for k, v in catmaps.items()}},
              open(f"{MAT}/schema.json", "w"), indent=1, ensure_ascii=False)
    print(f"saved. predictors {len(cols)}, {(time.time()-t0)/60:.1f} min", flush=True)
    print("predictors:", ", ".join(cols), flush=True)

def load():
    X = np.load(f"{MAT}/X.npy", mmap_mode="r")
    y = np.load(f"{MAT}/y_license.npy")
    ys = np.load(f"{MAT}/y_strict.npy")
    meta = pd.read_parquet(f"{MAT}/meta.parquet")
    sch = json.load(open(f"{MAT}/schema.json"))
    return X, y, ys, meta, sch

if __name__ == "__main__":
    build()
