"""T1-B. Classifier for the meaning of the licence string, applied to the whole panel.
Assesses the validity of the outcome definition. The rules were set from the census and are all stated here.

Unit of classification: a value is split into fields on <br>, the pipe, the semicolon and newlines, each field is classified, and the results are combined at value level.
classes at the level of the whole value (mutually exclusive):
  empty          no value / nan
  code           at least one token that looks like a registration identifier (the strongest evidence)
  trader_id      trader information under the Digital Services Act (entity name, commercial register number, address) and no accommodation registration number
  pending        a declaration that registration was applied for but not yet issued
  exempt         a declaration of exemption, non-applicability or no requirement
  other_text     any other free text, with no identifier
derived outcome variables:
  y_any      every non-empty value counts as positive (the loosest)
  y_license  the established definition: non-empty and not exactly 'exempt', 'none' or 'n/a'
  y_strict   only a value containing a registration code counts as positive
"""
import sys, os, re, json, time, collections, itertools
sys.path.insert(0, ".")
import pandas as pd, numpy as np, c10lib
from concurrent.futures import ProcessPoolExecutor

OUT = "./t1"; SH = f"{OUT}/cls"; os.makedirs(SH, exist_ok=True)

SPLIT = re.compile(r"<\s*br\s*/?\s*>|[\r\n]+|\s\|\s|;\s")
LABEL = re.compile(r"""^\s*(?:[A-Za-zÀ-ÿ'\.\- ]{3,40}\s[-–:]\s*)?
                        (?:(?:national|regional|municipal|provincial|city|state|local|tourist|tourism|business|rental|host|property|registration|licen[cs]e|permit|number|no\.?|reg\.?|id|code|tax|vat|status|type|category)\b[\s\.\-:]*){0,6}
                        [:\-–]?\s*""", re.X | re.I)
TRADER = re.compile(r"legal entity name|legal form|trade register|handelsregister|legal representative"
                    r"|listing address\s*:|entity address|umsatzsteuer|ust-?idnr|vat\s*(?:id|number)", re.I)
PENDING = re.compile(r"pending|applied|in progress|in corso|in attesa|en proceso|en tr[aá]mite|solicitad"
                     r"|beantragt|aangevraagd|awaiting|have ?n[o']?t received|haven.t received|under review"
                     r"|submitted|requested|richiest|em an[aá]lise|σε αναμον|\uc2e0\uccad", re.I)
# Exemption and non-applicability wording: platform boilerplate carrying no identifier, plus negations in several languages.
EXEMPT = re.compile(r"\bexempt|exento|exenta|esente|esenzione|befreit|ausgenommen|vrijgesteld|isent"
                    r"|απαλλ|non soumis|dispens[eé]|mobility lease|bail mobilit|home sharing exemption"
                    r"|not required|not needed|no licen[cs]e|not applicable|non applicable"
                    r"|does ?n[o']?t require|no requiere|sin licencia|sin n[uú]mero|non-real estate"
                    r"|not a (?:dwelling|rental)|caravan park|hotel-type|per ostr"
                    r"|^n\s*/?\s*a\.?$|^none$|^aucun[e]?$|^nessun[oa]$|^ningun[oa]$|^keine[rs]?$"
                    r"|^geen$|^nada$|^no$|^-+$|^\.+$|^0$", re.I)
CJKDOC = re.compile(r"第[^\s]{1,24}?号")
CODE = re.compile(r"(?<![0-9])[0-9]{4,}(?![0-9])"
                  r"|[A-Za-z][A-Za-z0-9][A-Za-z0-9\-/\.]{3,}(?=.*[0-9])"
                  r"|[0-9][A-Za-z0-9\-/\.]{3,}")
DATEISH = re.compile(r"(?:19|20)[0-9]{2}[-/\.][01]?[0-9][-/\.][0-3]?[0-9]|[0-3]?[0-9][-/\.][01]?[0-9][-/\.](?:19|20)[0-9]{2}")
ALNUM = re.compile(r"[^0-9A-Za-z０-９぀-ヿ㐀-鿿]")

def fields(v):
    return [f.strip() for f in SPLIT.split(v) if f and f.strip()]

def field_class(f):
    """Class of one field, returned as (class, extracted code or None).
    Order of tests: trader information, then pending, then exemption (on the raw field, before any label is stripped), then registration code, then other text.
    Exemption is tested on the raw field first because of strings such as 'Exempt - This listing is a hotel or motel',
    because exemption wording can itself look like a label and would be removed with it."""
    if TRADER.search(f):  return "trader_id", None
    if PENDING.search(f): return "pending", None
    if EXEMPT.search(f):  return "exempt", None
    body = LABEL.sub("", f, count=1).strip(" .:-–,")
    probe = body if body else f
    nodate = DATEISH.sub(" ", probe)
    m = CJKDOC.search(nodate)
    if m: return "code", m.group(0)
    m = CODE.search(nodate)
    if m and len(ALNUM.sub("", m.group(0))) >= 4:
        return "code", m.group(0)
    return "other_text", None

ORDER = ["empty", "code", "trader_id", "pending", "exempt", "other_text"]

def value_class(v):
    """(class, whether a code was found, number of fields)"""
    s = "" if v is None else str(v).strip()
    if s == "" or s.lower() == "nan":
        return "empty", 0, 0
    fs = fields(s)
    cls = [field_class(f)[0] for f in fs] or ["other_text"]
    has_code = int("code" in cls)
    if has_code:                                    # if any field contains a code, the value is classified as code
        out = "code"
    elif "trader_id" in cls:
        out = "trader_id"
    elif "pending" in cls:
        out = "pending"
    elif "exempt" in cls:
        out = "exempt"
    else:
        out = "other_text"
    return out, has_code, len(fs)

# ---------------- self-check; must pass before the rules are applied to the full panel ----------------
TESTS = [
    ("", "empty"), ("nan", "empty"), (None, "empty"),
    ("Exempt", "exempt"), ("Exempt - hotel-type listing", "exempt"),
    ("Exempt: This listing falls under the 'home sharing' exemption", "exempt"),
    ('Available with a mobility lease only ("bail mobilité")', "exempt"),
    ("Non-real estate listing", "exempt"), ("Aucun", "exempt"), ("N/A", "exempt"),
    ("License not needed per OSTR", "exempt"),
    ("Exempt - This listing is a hotel or motel", "exempt"),
    ("Exempt - This listing is a transient occupancy residential structure", "exempt"),
    ("Exempt: This listing falls under Section 12 of the Land Use Planning and Approvals Act 1993", "exempt"),
    ("Madrid - Regional registration number<br />Exempt - seasonal rental", "exempt"),
    ("Spain - National registration number<br />Exempt - tourist apartment<br /><br />Madrid - Regional registration number<br />Exempt - seasonal rental", "exempt"),
    ("Municipal registration number: Exempt<br />Provincial registration number: Exempt", "exempt"),
    ("Spain - National registration number<br />Exempt", "exempt"),
    ("I applied for a CIN, but haven’t received it yet", "pending"),
    ("City registration pending", "pending"), ("En proceso", "pending"),
    ("1381448", "code"), ("0933Κ13000104200", "code"), ("HSR19-123456", "code"),
    ("IT037006B4YR8XSBCV", "code"), ("OL 2026 019728", "code"), ("12-3456", "code"),
    ("540050360006, TA-212-767-6928-01", "code"),
    ("260090010000, N/A, TA-136-906-1888-01", "code"),
    ("01/Z/NA/007545-18", "code"), ("16043/AL - Na terra do mar", "code"),
    ("Catalonia - Regional registration number<br />ATG-000081<br /><br />Spain - National registration number<br />Exempt - tourist apartment", "code"),
    ("Quebec - Registration number<br />560070, expires: 2026-10-31", "code"),
    ("Hotels and Inns Business Act | 墨田区保健所 | 5墨福衛生環第75号", "code"),
    ("Hotels and Inns Business Act | 荒川区保健所 | ５荒保衛環き（三）第２号", "code"),
    ("Legal entity name and Legal form: DA Downtown Apartments Berlin GmbH <br/>Legal representatives or Trade register number: HRB 182467 B <br/> Entity address: Oranienburger Str. 3, 10178, Berlin, Deutschland <br/>Listing address: Gerichtstraße 56, Downtown Apartments, 13347, Berlin, Deutschland", "trader_id"),
    ("Authorised Serviced Apartment", "other_text"),
    ("Approved by government", "other_text"), ("0/AL", "other_text"),
]
def selftest(verbose=True):
    bad = []
    for v, want in TESTS:
        got = value_class(v)[0]
        if got != want: bad.append((str(v)[:70], want, got))
    if verbose:
        print(f"self-check passed {len(TESTS)-len(bad)}/{len(TESTS)}")
        for b in bad: print("   mismatches:", b)
    return bad

def y_old(lic):
    """reproduces the definition used by the earlier benchmark."""
    s = lic.astype("string").str.strip()
    return ((s.notna()) & (s != "") & (s.str.lower() != "nan")
            & (~s.str.lower().str.fullmatch(r"exempt|none|n/?a", na=False))).astype("int8")

def one(args):
    country, city, date, grp, side, path = args
    key = c10lib.shard_key(country, city, date)
    try:
        d = pd.read_csv(path, usecols=lambda c: c in ("id", "license"),
                        compression="gzip", low_memory=False)
    except Exception as e:
        return {"city": city, "country": country, "date": date, "note": f"ERR:{type(e).__name__}"}
    lic = d["license"] if "license" in d.columns else pd.Series([None] * len(d), dtype="object")
    yo = y_old(lic)
    s = lic.astype("string").fillna("")
    lut = {u: value_class(u) for u in s.unique()}
    cls = s.map(lambda u: lut[u][0])
    hc = s.map(lambda u: lut[u][1]).astype("int8")
    nf = s.map(lambda u: lut[u][2]).astype("int16")
    o = pd.DataFrame({"m_id": d["id"] if "id" in d.columns else np.arange(len(d)),
                      "lic_class": pd.Categorical(cls, categories=ORDER),
                      "n_fields": nf, "y_license": yo, "y_strict": hc,
                      "y_any": ((s.str.strip() != "") & (s.str.lower() != "nan")).astype("int8")})
    o["m_city"] = city; o["m_country"] = country; o["m_group"] = grp
    o["m_side"] = side; o["m_date"] = date
    o.to_parquet(f"{SH}/{key}.parquet", index=False)
    r = {"city": city, "country": country, "group": grp, "side": side, "date": date,
         "n": len(o), "note": "", "y_license": int(o.y_license.sum()),
         "y_strict": int(o.y_strict.sum()), "y_any": int(o.y_any.sum()),
         "n_uniq_values": int((s.str.strip() != "").sum() and s[s.str.strip() != ""].nunique())}
    for c in ORDER: r[f"c_{c}"] = int((cls == c).sum())
    return r

if __name__ == "__main__":
    if selftest():
        print("Self-check failed; the rules are not applied to the full panel."); sys.exit(1)
    t0 = time.time()
    s = c10lib.resolve()
    jobs = [(r.country, r.city, r.publish_date, r.group, r.side, r.path) for r in s.itertuples()]
    rows = []
    with ProcessPoolExecutor(24) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            rows.append(r)
            if i % 150 == 0: print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}seconds", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/class_by_snapshot.csv", index=False)
    N = int(df.n.sum())
    print(f"\nsnapshots {len(df)}, failures {int((df.note != '').sum())}, rows {N:,}, {(time.time()-t0)/60:.1f} min")
    tot = df[[c for c in df.columns if c.startswith('c_')]].sum()
    print("\n=== class composition over the whole panel ===")
    for c in ORDER:
        v = int(tot[f"c_{c}"]); print(f"  {c:<11} {v:>10,}  {100*v/N:6.2f}%")
    print("\n=== positive rate by outcome definition ===")
    for c in ("y_any", "y_license", "y_strict"):
        print(f"  {c:<10} {int(df[c].sum()):>10,}  {100*df[c].sum()/N:6.2f}%")
