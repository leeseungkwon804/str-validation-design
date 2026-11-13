"""Cell-by-cell check: every table in the manuscript is rebuilt from the outputs and compared cell by cell.
The token check in check_numbers.py only guarantees that a value exists somewhere; this check
It checks that the value is in that exact cell."""
import json, re, sys
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd, numpy as np
OUT = "./out"
ms = open("./ms/Manuscript_v2.md").read()

def table(n):
    """Return the Table n block as (header, rows)."""
    m = re.search(rf"^TABLE: Table {n}\. .*?$\n((?:.*\|.*\n)+)", ms, flags=re.M)
    if not m: raise SystemExit(f"Table {n} not found")
    rows = [[c.strip() for c in ln.split("|")] for ln in m.group(1).strip().split("\n")]
    return rows[0], rows[1:]

def num(x):
    x = x.replace(",", "").replace("−", "-").strip()
    x = re.sub(r"\s*\(.*?\)\s*", "", x)
    try: return float(x)
    except ValueError: return None

bad = []
def dec_of(x):
    """Number of decimals as printed in the manuscript."""
    m = re.search(r"\.(\d+)", str(x))
    return len(m.group(1)) if m else 0

def cmp(label, got, want, tol=None, shown=None):
    """Compares against the computed value rounded to the number of decimals shown in the manuscript."""
    if got is None or want is None:
        bad.append(f"{label}: manuscript {got} != output {want}"); return
    d = dec_of(shown if shown is not None else got)
    w = float(Decimal(repr(float(want))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
    g = float(Decimal(repr(float(got))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
    if abs(g - w) > 1e-9:
        bad.append(f"{label}: manuscript {got} != output {want} (at the {d} decimals shown: {g} against {w})")

SPL = {"1 Random (conventional)": "1 Random (conventional)", "2 Market hold-out": "2 Market hold-out",
       "3 Temporal (pre to post)": "3 Temporal (pre to post)", "4 Host hold-out": "4 Host hold-out",
       "5 Market by time": "5 Market by time"}
MDL = {"Logistic regression": "Logistic regression", "Random forest": "Random forest",
       "Gradient boosting": "GBDT", "TabPFN": "TabPFN"}

# ---- Table 4: A_summary_y_license (full-fold training) ----
A = pd.read_csv(f"{OUT}/A_summary_y_license.csv"); Af = A[A["training"] == "Full fold"]
hdr, rows = table(4)
for r in rows:
    s, m = SPL[r[0]], MDL[r[1]]
    src = Af[(Af.design == s) & (Af.model == m)].iloc[0]
    cmp(f"T4 {r[0]}/{r[1]} AUC", num(r[2]), float(src.AUC_full))
    sd = re.search(r"\(([\d.]+)\)", r[2])
    if sd: cmp(f"T4 {r[0]}/{r[1]} sd", float(sd.group(1)), float(src.AUC_full_sd))
    cmp(f"T4 {r[0]}/{r[1]} PRAUC", num(r[3]), float(src.PRAUC_full))
    cmp(f"T4 {r[0]}/{r[1]} Brier", num(r[4]), float(src.Brier_full))

# ---- Table 5: 10,000-row sample ----
As = A[A["training"] == "10,000 rows"]
hdr, rows = table(5)
for r in rows:
    s = SPL[r[0]]
    for j, mname in enumerate(hdr[1:], start=1):
        src = As[(As.design == s) & (As.model == MDL[mname])].iloc[0]
        cmp(f"T5 {r[0]}/{mname}", num(r[j]), float(src.AUC))

# ---- Table 6: market-level bootstrap ----
B = pd.read_csv(f"{OUT}/B_bootstrap_y_license.csv")
hdr, rows = table(6)
for r in rows:
    s = SPL[r[0]]
    for j, mname in enumerate(hdr[1:], start=1):
        src = B[(B.design == s) & (B.model == MDL[mname] + "_Full fold")].iloc[0]
        g = re.match(r"([\d.]+) \(([\d.]+), ([\d.]+)\)", r[j])
        cmp(f"T6 {r[0]}/{mname} gap", float(g.group(1)), float(src.loss_vs_random_split), shown=g.group(1))
        cmp(f"T6 {r[0]}/{mname} lo", float(g.group(2)), float(src.gap_lo95), shown=g.group(2))
        cmp(f"T6 {r[0]}/{mname} hi", float(g.group(3)), float(src.gap_hi95), shown=g.group(3))

# ---- Table 7: market recovery by predictor group ----
C1 = pd.read_csv(f"{OUT}/C1_market_recovery.csv")
GRP = {"Host portfolio size": "Host portfolio size", "Minimum and maximum stay": "Minimum and maximum stay",
       "Text length": "Text length", "Property type": "Property type", "Price": "Price",
       "Availability": "Availability", "Tenure": "Tenure", "Host conduct": "Host conduct", "Reviews": "Reviews"}
base = float(C1[C1.target == "Market (city)"].accuracy.iloc[0])
grpsz = {k: len(v) for k, v in json.load(open(f"{OUT}/C1_groups.json")).items()}
hdr, rows = table(7)
for r in rows:
    g = GRP[r[0]]
    cmp(f"T7 {r[0]} n_variables", num(r[1]), grpsz[g], tol=0)
    cmp(f"T7 {r[0]} alone", num(r[2]), 100*float(C1[C1.target == f"Market, alone: {g}"].accuracy.iloc[0]), tol=5e-3)
    rem = 100*float(C1[C1.target == f"Market, removed: {g}"].accuracy.iloc[0])
    cmp(f"T7 {r[0]} removed", num(r[3]), rem, tol=5e-3)
    cmp(f"T7 {r[0]} change", num(r[4]), rem - 100*base, tol=5e-2)

# ---- Table 8: C2 ----
C2 = pd.read_csv(f"{OUT}/C2_baserate_proxy.csv"); C2 = C2[C2.outcome == "y_license"]
hdr, rows = table(8)
key = {"Base rate of the true market, no listing characteristic used": "AUC_true_market_baserate",
       "Base rate of the recovered market, posterior-weighted": "AUC_baserate_posterior",
       "Base rate of the recovered market, single best guess": "AUC_baserate_argmax"}
for r in rows:
    if r[0] in key:
        for j, sp in ((1, "1 Random (conventional)"), (2, "2 Market hold-out")):
            v = num(r[j])
            if v is None: continue   # "not computable"
            cmp(f"T8 {r[0][:28]}/{sp}", v, float(C2[C2.design == sp][key[r[0]]].mean()))
    if r[0].startswith("Market recovery accuracy"):
        for j, sp in ((1, "1 Random (conventional)"), (2, "2 Market hold-out")):
            cmp(f"T8 recovery accuracy/{sp}", num(r[j]), 100*float(C2[C2.design == sp].market_recovery_accuracy.mean()), tol=5e-3)
    if r[0].startswith("Gradient boosting"):
        for j, sp in ((1, "1 Random (conventional)"), (2, "2 Market hold-out")):
            cmp(f"T8 GBDT/{sp}", num(r[j]), float(A[(A.design == sp) & (A.model == "GBDT") & (A["training"] == "Full fold")].AUC.iloc[0]))

# ---- Table 9: ablation by removal ----
C3 = pd.read_csv(f"{OUT}/C3_ablation.csv")
NAME = {"None (all 50)": "All 50 predictors (baseline)"} | {k: f"Removed: {v}" for k, v in GRP.items()}
hdr, rows = table(9)
for r in rows:
    src = C3[C3.setting == NAME[r[0]]].iloc[0]
    cmp(f"T9 {r[0]} Random split", num(r[1]), float(src.AUC_random_split))
    cmp(f"T9 {r[0]} n_markets", num(r[2]), float(src.AUC_market_holdout))
    cmp(f"T9 {r[0]} gap", num(r[3]), float(src.gap))
    cmp(f"T9 {r[0]} gap change", num(r[4]), float(src.gap_change))

# ---- Table 10: learning curve ----
E = pd.read_csv(f"{OUT}/E_learning_summary_y_license.csv")
hdr, rows = table(10)
for r in rows:
    K = int(num(r[0]))
    a = E[(E.curve == "Markets added, training rows fixed at 800,000") & (E.n_training_markets == K)].iloc[0]
    b = E[(E.curve == "Markets added, all rows of the sampled markets") & (E.n_training_markets == K)].iloc[0]
    cmp(f"T10 K={K} fixed rows", num(r[1]), float(a.AUC))
    cmp(f"T10 K={K} all rows", num(r[2]), float(b.AUC))
    cmp(f"T10 K={K} training rows", num(r[3]), round(float(b.n_train)))

# ---- Table 11: calibration and decision metrics ----
D = pd.read_csv(f"{OUT}/D_metrics_y_license.csv"); D = D[D.model == "Gradient boosting, full fold"]
dec = pd.read_csv(f"{OUT}/D_decision_y_license.csv"); dec = dec[dec.model == "Gradient boosting, full fold"]
hdr, rows = table(11)
for r in rows:
    s = SPL[r[0]]; src = D[D.design == s].iloc[0]
    for j, c in ((1, "AUC"), (2, "Brier"), (3, "ECE"), (4, "calibration_slope"), (5, "calibration_intercept")):
        cmp(f"T11 {r[0]} {c}", num(r[j]), float(src[c]))
    g = dec[dec.design == s].sort_values("threshold_probability"); neg = g[g.net_benefit < 0]
    want = float(neg.threshold_probability.iloc[0]) if len(neg) else None
    got = num(r[6]) if r[6].strip() != "none" else None
    if (got is None) != (want is None): bad.append(f"T11 {r[0]} threshold: manuscript {r[6]} != output {want}")
    elif got is not None: cmp(f"T11 {r[0]} threshold", got, want, tol=1e-6)

# ---- Table 12: the two outcome definitions ----
AS = pd.read_csv(f"{OUT}/A_summary_y_strict.csv"); ASf = AS[AS["training"] == "Full fold"]
hdr, rows = table(12)
for r in rows:
    s, m = SPL[r[0]], MDL[r[1]]
    for j, src in ((2, Af), (3, ASf)):
        row = src[(src.design == s) & (src.model == m)].iloc[0]
        cmp(f"T12 {r[0]}/{r[1]} AUC({j})", num(r[j]), float(row.AUC_full))
        gp = re.search(r"\(([\d.]+)\)", r[j])
        if gp:
            ref = float(src[(src.design == "1 Random (conventional)") & (src.model == m)].AUC_full.iloc[0])
            cmp(f"T12 {r[0]}/{r[1]} gap({j})", float(gp.group(1)), ref - float(row.AUC_full))

# ---- Table 13: markets where the definitions diverge ----
DV = pd.read_csv(f"{OUT}/T1_divergent_markets.csv"); DV = DV[DV.Country != "Country"]
hdr, rows = table(13)
for r in rows:
    src = DV[DV.Market == r[1]].iloc[0]
    cmp(f"T13 {r[1]} n", num(r[2]), float(str(src['Listing-snapshots']).replace(",", "")), tol=0)
    for j, c in ((3, "Loose"), (4, "Strict"), (5, "Difference (pp)"),
                 (6, "Exemption text"), (7, "Trader ID (DSA)"), (8, "Other free text")):
        cmp(f"T13 {r[1]} {c}", num(r[j]), float(src[c]), tol=5e-3)

# ---- Table 15: host portfolio ----
H = json.load(open(f"{OUT}/H_hosts_summary.json"))
hj = {d["city"]: d for d in H["five_jurisdictions"]}
hdr, rows = table(15)
for r in rows:
    c = {"Montréal": "Montreal", "Québec City": "Quebec City"}.get(r[0], r[0])
    src = hj[c]
    cmp(f"T15 {r[0]} single-listing share", num(r[2]), src["listings_of_1listing_hosts_pct"], tol=5e-3)
    cmp(f"T15 {r[0]} top 10% share", num(r[3]), src["top10pct_host_share_pct"], tol=5e-3)
    cmp(f"T15 {r[0]} largest host", num(r[4]), src["max_host_listings"], tol=0)

# ---- Table 14: minimum nights ----
MM = pd.read_csv(f"{OUT}/M_minnights_by_market.csv")
TH = {"Toronto": ("sh_28", 28), "New York City": ("sh_30", 30), "Montréal": ("sh_31", 31),
      "Québec City": ("sh_31", 31), "Vancouver": ("sh_90", 90), "Victoria": ("sh_90", 90)}
hdr, rows = table(14)
for r in rows:
    c = {"Montréal": "Montreal", "Québec City": "Quebec City"}.get(r[0], r[0])
    src = MM[MM.city == c].iloc[0]
    if r[0] in TH:
        cmp(f"T14 {r[0]} share at threshold", num(r[2]), 100*float(src[TH[r[0]][0]]), tol=5e-3)
    else:  # Singapore: the modal value is the statutory threshold
        cmp(f"T14 {r[0]} share at threshold", num(r[2]), 100*float(src.modal_share), tol=5e-3)
    mm = re.match(r"(\d+) \(([\d.]+)\)", r[3])
    cmp(f"T14 {r[0]} modal_value", float(mm.group(1)), float(src.modal_value), tol=0)
    cmp(f"T14 {r[0]} modal_share", float(mm.group(2)), 100*float(src.modal_share), tol=5e-3)

# ---- Tables 1 and 2: licence-field census and predictor groups ----
t1 = json.load(open("./t1/t1_summary.json"))
CL = {"Empty": "empty", "Registration code": "code", "Statement of exemption": "exempt",
      "Trader information (Digital Services Act)": "trader_id", "Application pending": "pending",
      "Other free text": "other_text"}
hdr, rows = table(1)
ne = t1["n_listing_snapshot"] - t1["class_counts"]["empty"]
for r in rows:
    k = CL[r[0]]
    cmp(f"T1 {r[0]} n", num(r[1]), t1["class_counts"][k], tol=0)
    cmp(f"T1 {r[0]} pct", num(r[2]), t1["class_pct"][k], tol=5e-3)
    if r[3] != "—": cmp(f"T1 {r[0]} nonempty", num(r[3]), 100*t1["class_counts"][k]/ne, shown=r[3])
hdr, rows = table(2)
for r in rows:
    cmp(f"T2 {r[0]} n_variables", num(r[1]), grpsz[GRP[r[0]]], tol=0)
    n_listed = len([x for x in r[2].split(",") if x.strip()])
    cmp(f"T2 {r[0]} variables listed", n_listed, grpsz[GRP[r[0]]], tol=0)

print(f"table check: Tables 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15")
print(f"mismatches {len(bad)}")
for b_ in bad: print("  ", b_)
sys.exit(0 if not bad else 1)
