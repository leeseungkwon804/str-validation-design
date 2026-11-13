"""Shared utilities: snapshot catalogue and file-path resolution, tolerant of parenthesis variants.
Note: the m_* columns are market-identity metadata used for splitting and aggregation only; they are never predictors."""
import pandas as pd, glob, os, re, unicodedata

ROOT = "."
PROJ = "."
RAWDIRS = [f"{PROJ}/data/raw", f"{ROOT}/raw"]

def _norm_key(country, city, date):
    f = lambda x: str(x).replace(" ", "_").replace("/", "-")
    return (_squash(f(city)), str(date))

def _squash(s):
    """A comparison key that absorbs parentheses and repeated underscores, so that 'Ireland_(nationwide)' equals 'Ireland_nationwide'."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(r"[()\[\]]", "", s)
    s = re.sub(r"[^0-9a-zÀ-ɏ]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def path_index():
    """(city key, date) -> file path; tolerant of parentheses and separator differences."""
    idx = {}
    for d in RAWDIRS:
        for p in sorted(glob.glob(f"{d}/*.csv.gz")):
            b = os.path.basename(p)[:-7]
            parts = b.split("__")
            if len(parts) != 3:
                continue
            k = (_squash(parts[1]), parts[2])
            idx.setdefault(k, p)          # prefer the older catalogue
    return idx

def snapshots(sides=("pre", "post")):
    s = pd.read_csv(f"{ROOT}/snapshots_have.csv")
    s = s[s.have]
    if sides:
        s = s[s.side.isin(sides)]
    return s.reset_index(drop=True)

def resolve(s=None):
    """Return the snapshot table with a path column attached."""
    s = snapshots() if s is None else s
    idx = path_index()
    s = s.copy()
    s["path"] = [idx.get((_squash(str(c).replace(" ", "_").replace("/", "-")), d))
                 for c, d in zip(s.city, s.publish_date)]
    return s

def shard_key(country, city, date):
    f = lambda x: str(x).replace(" ", "_").replace("/", "-")
    return f"{f(country)}__{f(city)}__{date}"
