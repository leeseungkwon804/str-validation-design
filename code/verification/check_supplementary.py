"""Re-reads every number in 02_Supplementary.docx and compares it, cell by cell, against the analysis outputs.
Because the document is reopened and read back, this checks the values in the document, not the script that wrote them."""
import re, sys, json
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd, numpy as np
from docx import Document

OUT = "./out"; ROOT = "."
csv = lambda p: pd.read_csv(f"{OUT}/{p}")
doc = Document("./pkg/02_Supplementary.docx")
man = json.load(open("./pkg/supplementary_manifest.json"))
tables = {m["id"]: t for m, t in zip(man, doc.tables)}
assert len(doc.tables) == len(man), f"table count mismatch: {len(doc.tables)} against {len(man)}"

bad = []
def num(x):
    x = str(x).replace(",", "").replace("−", "-").strip()
    try: return float(x)
    except ValueError: return None
def eq(label, shown, want):
    g = num(shown)
    if g is None or want is None or (isinstance(want, float) and not np.isfinite(want)):
        if (g is None) != (want is None or not np.isfinite(want)):
            bad.append(f"{label}: document '{shown}' != output {want}")
        return
    d = len(re.search(r"\.(\d+)", str(shown)).group(1)) if "." in str(shown) else 0
    q = Decimal(1).scaleb(-d)
    w = float(Decimal(repr(float(want))).quantize(q, rounding=ROUND_HALF_UP))
    if abs(g - w) > 1e-9: bad.append(f"{label}: document {shown} != output {want} (at {d} decimals, {g} != {w})")

def grid(tid):
    t = tables[tid]
    hdr = [c.text.strip() for c in t.rows[0].cells]
    rows = [{hdr[j]: c.text.strip() for j, c in enumerate(r.cells)} for r in t.rows[1:]]
    return hdr, rows

SPL = {"1 Random (conventional)": "1 Random (conventional)", "2 Market hold-out": "2 Market hold-out",
       "3 Temporal (pre to post)": "3 Temporal (pre to post)", "4 Host hold-out": "4 Host hold-out",
       "5 Market by time": "5 Market by time"}
MDL = {"Logistic regression": "Logistic regression", "Random forest": "Random forest",
       "Gradient boosting": "GBDT", "TabPFN": "TabPFN"}
OC = {"Loose": "y_license", "Strict": "y_strict"}

# ---- S1 ----
hdr, rows = grid("Table S1")
A = {oc: csv(f"A_summary_{oc}.csv") for oc in ("y_license", "y_strict")}
for r in rows:
    a = A[OC[r["Outcome"]]]
    full = r["Training"] == "Full fold"
    s = a[(a.design == SPL[r["Design"]]) & (a.model == MDL[r["Model"]]) &
          (a["training"] == ("Full fold" if full else "10,000 rows"))].iloc[0]
    k = f"S1 {r['Outcome']}/{r['Design']}/{r['Model']}/{r['Training']}"
    eq(f"{k} AUC", r["AUC"], float(s.AUC_full if full else s.AUC))
    eq(f"{k} PR-AUC", r["PR-AUC"], float(s.PRAUC_full if full else s.PRAUC))
    eq(f"{k} Brier", r["Brier"], float(s.Brier_full if full else s.Brier))
    eq(f"{k} Folds", r["Folds"], float(s.folds))
    if r["AUC SD"]: eq(f"{k} SD", r["AUC SD"], float(s.AUC_full_sd if full else s.AUC_sd))

# ---- S2 ----
hdr, rows = grid("Table S2")
B = {oc: csv(f"B_bootstrap_{oc}.csv") for oc in ("y_license", "y_strict")}
def bmodel(x):
    base, _, reg = x.partition(", ")
    return MDL[base] + ("_Full fold" if reg == "full fold" else "_10,000 rows")
for r in rows:
    b = B[OC[r["Outcome"]]]
    s = b[(b.design == SPL[r["Design"]]) & (b.model == bmodel(r["Model"]))].iloc[0]
    k = f"S2 {r['Outcome']}/{r['Design']}/{r['Model']}"
    for col, src in (("AUC (pooled)", "AUC"), ("Bootstrap mean", "AUC_bootstrap_mean"), ("SE", "AUC_boot_se"),
                     ("AUC 2.5%", "AUC_lo95"), ("AUC 97.5%", "AUC_hi95"),
                     ("Loss vs random", "loss_vs_random_split"), ("Loss 2.5%", "gap_lo95"),
                     ("Loss 97.5%", "gap_hi95"), ("Markets", "n_markets"), ("Eval rows", "n_eval")):
        eq(f"{k} {col}", r[col], float(s[src]))

# ---- S3 (rule table; the only number is the self-check count) ----
src = open(f"{ROOT}/t1_classify.py").read()
n_tests = src[src.index("TESTS = ["):].count("\n    (")
notes = [p.text for p in doc.paragraphs if "self-check of" in p.text]
if not notes or f"self-check of {n_tests} hand-labelled cases" not in notes[0]:
    bad.append(f"S3 self-check count: document '{notes[0][:60] if notes else 'missing'}' != output {n_tests}")

# ---- S4 ----
hdr, rows = grid("Table S4")
C3 = csv("C3_ablation.csv")
GRP = {"Price": "Price", "Minimum and maximum stay": "Minimum and maximum stay", "Property type": "Property type",
       "Availability": "Availability", "Reviews": "Reviews", "Host portfolio size": "Host portfolio size",
       "Host conduct": "Host conduct", "Text length": "Text length", "Tenure": "Tenure"}
for r in rows:
    nm = r["Setting"]
    key = ("All 50 predictors (baseline)" if nm.startswith("All 50") else
           (f"Removed: {GRP[nm[9:]]}" if nm.startswith("Removed: ") else f"Alone: {GRP[nm[7:]]}"))
    s = C3[C3.setting == key].iloc[0]
    k = f"S4 {nm}"
    eq(f"{k} rand", r["Random split AUC"], float(s.AUC_random_split))
    eq(f"{k} mkt", r["Market hold-out AUC"], float(s.AUC_market_holdout))
    eq(f"{k} gap", r["Gap"], float(s.gap))
    eq(f"{k} randsd", r["Random SD"], float(s.random_split_sd))
    eq(f"{k} mktsd", r["Market SD"], float(s.market_holdout_sd))
    if r["Change in gap vs baseline"]: eq(f"{k} dgap", r["Change in gap vs baseline"], float(s.gap_change))

# ---- S5 ----
hdr, rows = grid("Table S5")
E = csv("E_learning_summary_y_license.csv")
CUR = {"Markets added, training rows fixed at 800,000": "Markets added, training rows fixed at 800,000",
       "Markets added, all rows of the sampled markets": "Markets added, all rows of the sampled markets",
       "Random split control, training rows added": "Random split control, training rows added"}
used = set()
for r in rows:
    sub = E[(E.curve == CUR[r["Curve"]]) & (E.n_training_markets == int(num(r["Training markets"])))]
    if CUR[r["Curve"]] == "Random split control, training rows added":
        sub = sub[np.isclose(sub.n_train, num(r["Mean training rows"]), atol=0.6)]
    s = sub.iloc[0]; used.add((r["Curve"], r["Training markets"]))
    k = f"S5 {r['Curve'][:28]} K={r['Training markets']}"
    eq(f"{k} AUC", r["AUC"], float(s.AUC)); eq(f"{k} n", r["Mean training rows"], float(s.n_train))
    eq(f"{k} min", r["AUC min"], float(s.AUC_min)); eq(f"{k} max", r["AUC max"], float(s.AUC_max))
    if r["AUC SD"]: eq(f"{k} sd", r["AUC SD"], float(s.AUC_sd))

# ---- S6 ----
hdr, rows = grid("Table S6")
for r in rows:
    oc = OC[r["Outcome"]]
    D = csv(f"D_metrics_{oc}.csv"); dec = csv(f"D_decision_{oc}.csv")
    s = D[(D.design == SPL[r["Design"]]) & (D.model == bmodel(r["Model"]))].iloc[0]
    k = f"S6 {r['Outcome']}/{r['Design']}/{r['Model']}"
    for col, srcc in (("AUC", "AUC"), ("Brier", "Brier"), ("ECE", "ECE"),
                      ("Calibration slope", "calibration_slope"), ("Calibration intercept", "calibration_intercept"),
                      ("Mean prediction", "mean_prediction"), ("Prevalence", "prevalence"), ("Eval rows", "n")):
        eq(f"{k} {col}", r[col], float(s[srcc]))
    g = dec[(dec.design == SPL[r["Design"]]) & (dec.model == bmodel(r["Model"]))].sort_values("threshold_probability")
    neg = g[g.net_benefit < 0]
    want = float(neg.threshold_probability.iloc[0]) if len(neg) else None
    got = num(r["Net benefit turns negative at"])
    if (got is None) != (want is None):
        bad.append(f"{k} threshold: document '{r['Net benefit turns negative at']}' != output {want}")
    elif got is not None: eq(f"{k} threshold", r["Net benefit turns negative at"], want)

# ---- S7 ----
hdr, rows = grid("Table S7")
M = pd.read_csv(f"{ROOT}/t1/market_license_class.csv")
ACC = csv("C1_accuracy_by_market.csv").set_index("city")
for r in rows:
    s = M[M.m_city == r["Market"]].iloc[0]
    k = f"S7 {r['Market']}"
    eq(f"{k} n", r["Listing-snapshots"], float(s.n))
    eq(f"{k} loose", r["Loose (%)"], 100*float(s.y_license))
    eq(f"{k} strict", r["Strict (%)"], 100*float(s.y_strict))
    eq(f"{k} diff", r["Difference (pp)"], float(s.delta_license_strict_pp))
    for col, srcc in (("Code (% of non-empty)", "ne_code"), ("Exempt (%)", "ne_exempt"),
                      ("Trader (%)", "ne_trader_id"), ("Pending (%)", "ne_pending"), ("Other (%)", "ne_other_text")):
        eq(f"{k} {col}", r[col], float(s[srcc]))
    eq(f"{k} rec", r["Market recovered (%)"], 100*float(ACC.loc[r["Market"], "accuracy"]))

# ---- S8 ----
hdr, rows = grid("Table S8")
G = csv("G_country.csv")
for r in rows:
    s = G[G.m_country == r["Country"]].iloc[0]
    k = f"S8 {r['Country']}"
    for col, srcc in (("Markets", "n_markets"), ("Listing-snapshots", "listing_snapshots"), ("Any value (%)", "display_rate_any"),
                      ("Loose (%)", "display_rate_loose"), ("Strict (%)", "display_rate_strict"),
                      ("Code (% of non-empty)", "ne_code"), ("Exempt (%)", "ne_exempt"),
                      ("Trader (%)", "ne_trader_id"), ("Pending (%)", "ne_pending"), ("Other (%)", "ne_other_text")):
        eq(f"{k} {col}", r[col], float(s[srcc]))

# ---- S9 ----
hdr, rows = grid("Table S9")
MN = csv("M_minnights_by_market.csv")
for r in rows:
    s = MN[MN.city == r["Market"]].iloc[0]
    k = f"S9 {r['Market']}"
    eq(f"{k} n", r["Listings"], float(s.n)); eq(f"{k} median", r["Median"], float(s["median"]))
    eq(f"{k} p90", r["90th percentile"], float(s.p90)); eq(f"{k} mode", r["Modal value"], float(s.modal_value))
    eq(f"{k} modeshare", r["Modal share (%)"], 100*float(s.modal_share))
    for col, srcc in (("1 night (%)", "sh_1"), ("28 (%)", "sh_28"), ("30 (%)", "sh_30"),
                      ("31 (%)", "sh_31"), ("32 (%)", "sh_32"), ("90 (%)", "sh_90"),
                      ("28 to 32 (%)", "sh_28to32")):
        eq(f"{k} {col}", r[col], 100*float(s[srcc]))

# ---- S10 ----
hdr, rows = grid("Table S10")
H = csv("H_hosts_by_market.csv")
for r in rows:
    s = H[H.city == r["Market"]].iloc[0]
    k = f"S10 {r['Market']}"
    for col, srcc in (("Snapshots", "snapshots"), ("Listings", "n_listings"), ("Hosts", "n_hosts"),
                      ("Hosts with one listing (%)", "hosts_with_1_pct"),
                      ("Listings of single-listing hosts (%)", "listings_of_1listing_hosts_pct"),
                      ("Top 10% host share (%)", "top10pct_host_share_pct"),
                      ("Top 1% host share (%)", "top1pct_host_share_pct"),
                      ("Largest host", "max_host_listings"),
                      ("Mean listings per host", "mean_listings_per_host")):
        eq(f"{k} {col}", r[col], float(s[srcc]))

# ---- S11 ----
hdr, rows = grid("Table S11")
for r in rows:
    C = csv(f"D_calibration_{OC[r['Outcome']]}.csv")
    s = C[(C.model == "Gradient boosting, full fold") & (C.design == SPL[r["Design"]]) & (C.bin == int(num(r["Bin"])) - 1)].iloc[0]
    k = f"S11 {r['Outcome']}/{r['Design']}/bin{r['Bin']}"
    eq(f"{k} n", r["Listings in bin"], float(s.n))
    eq(f"{k} pred", r["Mean predicted probability"], float(s.pred_mean))
    eq(f"{k} obs", r["Observed rate"], float(s.obs_rate))

# ---- S12 ----
hdr, rows = grid("Table S12")
for r in rows:
    Dc = csv(f"D_decision_{OC[r['Outcome']]}.csv")
    g = Dc[(Dc.model == "Gradient boosting, full fold") & (Dc.design == SPL[r["Design"]])]
    s = g[np.isclose(g.threshold_probability, num(r["Threshold probability"]), atol=1e-9)].iloc[0]
    k = f"S12 {r['Outcome']}/{r['Design']}/pt{r['Threshold probability']}"
    eq(f"{k} nb", r["Net benefit"], float(s.net_benefit))
    eq(f"{k} all", r["Net benefit, flag all"], float(s.net_benefit_flag_all))

n_cells = sum(len(t.rows) * len(t.columns) for t in doc.tables)
print(f"supplementary tables {len(doc.tables)}, cells {n_cells:,}, mismatches {len(bad)}")
for b in bad[:30]: print("  ", b)
sys.exit(0 if not bad else 1)
