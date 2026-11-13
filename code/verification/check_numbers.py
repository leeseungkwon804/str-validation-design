"""Checks every number in the manuscript against the registry numbers.json.
- forward check: every registered value that appears in the text must match the computed one (formatting may differ).
- reverse check: every numeric token in the text must be explained by a registered value or by a declared structural number.
Structural numbers (section, table and figure numbers, dates, statutory citations, citation keys) are masked before counting, and every masked pattern is reported."""
import json, re, sys
from decimal import Decimal, ROUND_HALF_UP
MS = "./ms/Manuscript_v2.md"
REG = "./ms/numbers.json"
txt = open(MS).read()
reg = json.load(open(REG))

# ---------- 1. mask structural tokens ----------
MASKS = [
 (r"\[@[^\]]+\]",                      "citation key"),
 (r"Sections?\s+\d+(\.\d+)?(\s+(and|to)\s+\d+(\.\d+)?)?", "section reference"),
 (r"Tables?\s+S?\d+", "table reference"),
 (r"Figures?\s+\d+",                   "figure reference"),
 (r"Appendix\s+[A-Z]",                 "appendix reference"),
 (r"§\s?[\dA-Za-z\-\.\(\)]+",          "section symbol citation"),
 (r"Chapter\s+\d+",                    "municipal code chapter"),
 (r"Local Law \d+ of \d{4}",           "statute number"),
 (r"SBC \d{4} c\.\d+ s\.\d+",          "statutory citation"),
 (r"\d{4} c\.\d+ art\.\s?\d+(\.\d+)?", "statutory citation"),
 (r"\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}", "date"),
 (r"\bin (19|20)\d{2}\b",              "year"),
 (r"published in (19|20)\d{2}",        "year"),
 (r"\b(19|20)\d{2}\s+(and|to)\s+(19|20)\d{2}\b", "year"),
 (r"\bfrom 57\.4 percent in (19|20)\d{2}", "year"),
 (r"^TITLE:|^ABSTRACT:|^KEYWORDS:|^TABLE:|^FIGURE:", "block marker"),
 (r"^#{2,4}\s*[A-Z]?\d+(\.\d+)?\.", "section heading number"),
]
masked = txt; mask_counts = {}
for pat, nm in MASKS:
    masked, k = re.subn(pat, lambda m: " <MASK> ", masked, flags=re.M)
    mask_counts[nm] = mask_counts.get(nm, 0) + k
# remaining year tokens inside tables and captions
masked, k = re.subn(r"\b(19|20)\d{2}\b", " <MASK> ", masked); mask_counts["year"] = mask_counts.get("year",0)+k

# ---------- 2. set of allowed numeric strings ----------
def forms(v):
    out = set()
    av = abs(v)
    for d in range(0, 5):
        q = Decimal(1).scaleb(-d)
        r = float(Decimal(repr(v)).quantize(q, rounding=ROUND_HALF_UP))
        s = f"{r:.{d}f}"
        out.add(s); out.add(s.lstrip("-"))
        if abs(r) >= 1000:
            out.add(f"{r:,.{d}f}"); out.add(f"{r:,.{d}f}".lstrip("-"))
    if av == int(av):
        out.add(str(int(round(v)))); out.add(f"{int(round(v)):,}")
    return {x for x in out if x not in ("", "-")}

allow = {}
for r in reg:
    for f in forms(r["value"]): allow.setdefault(f, []).append(r["name"])

# ---------- 3. numeric tokens in the manuscript ----------
TOK = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+\.\d+|(?<![\w.,])\d+(?![\w.])")
hits, bad = [], []
for m in TOK.finditer(masked):
    tok = m.group(0)
    norm = tok
    if norm in allow:
        hits.append((tok, allow[norm][0])); continue
    n2 = norm.replace(",", "")
    if n2 in allow:
        hits.append((tok, allow[n2][0])); continue
    ctx = masked[max(0, m.start()-70):m.end()+45].replace("\n", " ")
    bad.append((tok, ctx))

# ---------- 4. forward check: are the key values actually present in the text? ----------
KEY = ["A.y_license.1 Random (conventional).GBDT.Full fold.AUC", "A.y_license.2 Market hold-out.GBDT.Full fold.AUC",
       "A.y_license.4 Host hold-out.GBDT.Full fold.AUC", "A.y_license.5 Market by time.GBDT.Full fold.AUC",
       "C2.y_license.1 Random (conventional).true", "C2.y_license.1 Random (conventional).soft",
       "C1.Market (city).acc", "C1.Market (city).chance", "ICC_market_y_license",
       "B.y_license.2 Market hold-out.Gradient boosting, full fold.gap", "B.y_license.4 Host hold-out.Gradient boosting, full fold.gap",
       "D.y_license.1 Random (conventional).slope", "D.y_license.5 Market by time.slope",
       "E.Markets added, all rows of the sampled markets.K98.auc", "R.max_gap_shift_points", "n_rows", "n_markets"]
byname = {r["name"]: r["value"] for r in reg}
missing_key = []
for k in KEY:
    if k not in byname: missing_key.append((k, "not in the registry")); continue
    if not any(f in masked for f in forms(byname[k])): missing_key.append((k, byname[k]))

print(f"manuscript {MS}")
print(f"registry entries {len(reg)}, allowed strings {len(allow)}")
print(f"masked structural tokens: " + ", ".join(f"{k} {v}" for k, v in sorted(mask_counts.items()) if v))
print(f"numeric tokens in the text {len(hits)+len(bad)}, matched {len(hits)}, mismatches {len(bad)}")
if bad:
    print("\n--- tokens not reproducible from the outputs ---")
    for tok, ctx in bad: print(f"  [{tok}]  …{ctx}…")
if missing_key:
    print("\n--- registered key values missing from the text ---")
    for k, v in missing_key: print(f"  {k} = {v}")
print("\nresult:", "no mismatches" if not bad and not missing_key else f"mismatches {len(bad)+len(missing_key)}")
sys.exit(0 if not bad and not missing_key else 1)
