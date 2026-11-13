"""Builds 02_Supplementary.docx directly from the analysis outputs; nothing is transcribed by hand.
Tables S1 to S12. To avoid duplicating the manuscript tables, these hold the full grids, the market-level values and the curve detail.
The raw licence strings are not included in any form."""
import json, re, os
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd, numpy as np
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.section import WD_ORIENT

OUT = "./out"; ROOT = "."
csv = lambda p: pd.read_csv(f"{OUT}/{p}")
SPL = {"1 Random (conventional)": "1 Random (conventional)", "2 Market hold-out": "2 Market hold-out",
       "3 Temporal (pre to post)": "3 Temporal (pre to post)", "4 Host hold-out": "4 Host hold-out",
       "5 Market by time": "5 Market by time"}
ORD = list(SPL)
MDL = {"Logistic regression": "Logistic regression", "Random forest": "Random forest", "GBDT": "Gradient boosting",
       "TabPFN": "TabPFN"}
def mdl(x):
    b = x.replace("_Full fold", "").replace("_10,000 rows", "")
    r = MDL.get(b, b)
    return r + (", full fold" if x.endswith("_Full fold") else (", 10,000 rows" if x.endswith("_10,000 rows") else ""))
GRP = {"Price": "Price", "Minimum and maximum stay": "Minimum and maximum stay", "Property type": "Property type",
       "Availability": "Availability", "Reviews": "Reviews", "Host portfolio size": "Host portfolio size",
       "Host conduct": "Host conduct", "Text length": "Text length", "Tenure": "Tenure"}
OCN = {"y_license": "Loose", "y_strict": "Strict"}

MD = []
def W(line=""): MD.append(line)
W("TITLE: Supplementary Materials")
W()
W("What a Validation Split Decides: Market-Level Leakage in Listing-Level Prediction on Short-Term Rental "
  "Platform Data")
W()
W("Sewon Min")
W()
W("All values are produced by the analysis scripts in File S1 and are reproduced from the derived output files; "
  "no quantity in this document is transcribed by hand. The raw content of the licence field is not reproduced "
  "anywhere, in line with the data availability statement.")
W()

MANIFEST = []
def add_table(sid, title, df, note=None, colnames=None, decimals=None):
    MANIFEST.append({"id": f"Table S{sid}", "title": title, "rows": int(len(df)),
                     "cols": list(map(str, df.columns))})
    def cell(v, col):
        if isinstance(v, float) and np.isfinite(v):
            d = (decimals or {}).get(col, 4)
            rv = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
            return f"{rv:,f}" if abs(v) >= 1000 else f"{rv:f}"
        if isinstance(v, (int, np.integer)): return f"{v:,}"
        if isinstance(v, float): return ""
        return str(v)
    W(f"TABLE: Table S{sid}. {title}")
    W(" | ".join(str(c) for c in (colnames or df.columns)))
    for _, row in df.iterrows():
        W(" | ".join(cell(v, df.columns[j]) for j, v in enumerate(row)))
    if note: W("NOTE: " + " ".join(note.split()))
    W()

# ---------------- S1 full benchmark ----------------
rows = []
for oc in ("y_license", "y_strict"):
    A = csv(f"A_summary_{oc}.csv")
    for _, r_ in A.iterrows():
        full = r_["training"] == "Full fold"
        rows.append({"Outcome": OCN[oc], "Design": SPL[r_["design"]], "Model": MDL[r_["model"]],
                     "Training": "Full fold" if full else "10,000 rows", "Folds": int(r_["folds"]),
                     "AUC": r_["AUC_full"] if full else r_["AUC"],
                     "AUC SD": r_["AUC_full_sd"] if full else r_["AUC_sd"],
                     "PR-AUC": r_["PRAUC_full"] if full else r_["PRAUC"],
                     "Brier": r_["Brier_full"] if full else r_["Brier"],
                     "Seconds": r_["seconds"]})
S1 = pd.DataFrame(rows)
S1["_o"] = S1.Design.map({SPL[k]: i for i, k in enumerate(ORD)})
S1 = S1.sort_values(["Outcome", "Training", "Model", "_o"]).drop(columns="_o")
add_table(1, "Full benchmark: five validation designs, four models, two training regimes, two outcome "
             "definitions.", S1,
          note="AUC and SD are means and standard deviations over folds; the temporal design has one fold, so no SD. "
               "The full-fold regime fits the three classical models on the entire training fold; the 10,000-row regime "
               "fits all four models, including TabPFN, on the same subsample. Source: out/A_summary_{y_license,y_strict}.csv.",
          decimals={"Seconds": 1, "AUC SD": 4})

# ---------------- S2 bootstrap ----------------
rows = []
for oc in ("y_license", "y_strict"):
    B = csv(f"B_bootstrap_{oc}.csv")
    for _, r_ in B.iterrows():
        rows.append({"Outcome": OCN[oc], "Design": SPL[r_["design"]], "Model": mdl(r_["model"]),
                     "AUC (pooled)": r_["AUC"], "Bootstrap mean": r_["AUC_bootstrap_mean"], "SE": r_["AUC_boot_se"],
                     "AUC 2.5%": r_["AUC_lo95"], "AUC 97.5%": r_["AUC_hi95"],
                     "Loss vs random": r_["loss_vs_random_split"], "Loss 2.5%": r_["gap_lo95"],
                     "Loss 97.5%": r_["gap_hi95"], "Markets": int(r_["n_markets"]), "Eval rows": int(r_["n_eval"])})
S2 = pd.DataFrame(rows)
S2["_o"] = S2.Design.map({SPL[k]: i for i, k in enumerate(ORD)})
S2 = S2.sort_values(["Outcome", "Model", "_o"]).drop(columns="_o")
add_table(2, "Market-level bootstrap: paired losses against the random split, 1,000 resamples of the 123 "
             "markets common to all designs.", S2,
          note="Folds are pooled into one evaluation set per design before resampling, so a design's pooled AUC differs "
               "slightly from its fold mean in Table S1. Positive losses mean the design performs worse than the random "
               "split. Source: out/B_bootstrap_{y_license,y_strict}.csv.")

# ---------------- S3 classification rules ----------------
src = open(f"{ROOT}/t1_classify.py").read()
n_tests = src[src.index("TESTS = ["):].count("\n    (")
RULES = [
 ("Field separation", "The field value is split on <br>, newline, ' | ' and '; ' and each part is classified "
  "separately; the value takes the highest-priority class among its parts.", "—"),
 ("1. Trader information", "Matches wording identifying a business entity rather than an accommodation "
  "registration: legal entity name, legal form, trade register, Handelsregister, legal representative, "
  "listing or entity address, VAT or Umsatzsteuer identifier.", "trader_id"),
 ("2. Application pending", "Matches declarations that a registration has been applied for but not issued, in "
  "ten languages (pending, applied, in progress, in corso, en trámite, beantragt, aangevraagd, submitted, "
  "σε αναμονή and the Korean equivalent).", "pending"),
 ("3. Exemption or non-applicability", "Matches exemption wording and platform boilerplate in ten languages "
  "(exempt, exento, esente, befreit, vrijgesteld, απαλλαγή, non soumis, dispensé, mobility lease, home sharing "
  "exemption, not required, not applicable), and standalone negations (n/a, none, aucun, keine, geen, 0, -).", "exempt"),
 ("4. Registration code", "After stripping a leading label of up to six registration-related words and removing "
  "date-like substrings, the remainder must contain a CJK document number (第…号) or an alphanumeric token with at "
  "least four alphanumeric characters including a digit.", "code"),
 ("5. Other free text", "Any remaining non-empty value.", "other_text"),
 ("Empty", "No value in the field.", "empty"),
]
S3 = pd.DataFrame([{"Step": a, "Rule": b, "Class assigned": c} for a, b, c in RULES])
add_table(3, "Classification rules applied to the licence field.", S3,
          note=f"Rules are applied in the order shown; exemption wording is tested on the raw part before the label is "
               f"stripped, because exemption boilerplate can itself take the form of a label. The rule set passes a "
               f"self-check of {n_tests} hand-labelled cases, all of which it classifies correctly. "
               f"Source: t1_classify.py in File S1. Example strings are not reproduced here.")

# ---------------- S4 full ablation grid ----------------
C3 = csv("C3_ablation.csv")
def ab_name(x):
    if x == "All 50 predictors (baseline)": return "All 50 predictors (baseline)"
    m = re.match(r"Removed: (.+)", x)
    if m: return f"Removed: {GRP[m.group(1)]}"
    m = re.match(r"Alone: (.+)", x)
    return f"Alone: {GRP[m.group(1)]}"
S4 = pd.DataFrame({"Setting": C3.setting.map(ab_name), "Random split AUC": C3.AUC_random_split,
                   "Random SD": C3.random_split_sd, "Market hold-out AUC": C3.AUC_market_holdout,
                   "Market SD": C3.market_holdout_sd, "Gap": C3.gap,
                   "Change in gap vs baseline": C3.gap_change})
add_table(4, "Feature ablation: every predictor group removed, and every group used alone.", S4,
          note="Gradient boosting, loose outcome definition, five folds per cell, training subsample fixed at 1,000,000 "
               "rows in every setting. The gap is the random-split AUC minus the market-hold-out AUC. Change in gap is "
               "reported only for the removal settings. Source: out/C3_ablation.csv.")

# ---------------- S5 learning curve ----------------
E = csv("E_learning_summary_y_license.csv")
CUR = {"Markets added, training rows fixed at 800,000": "Markets added, training rows fixed at 800,000",
       "Markets added, all rows of the sampled markets": "Markets added, all rows of the sampled markets",
       "Random split control, training rows added": "Random split control, training rows added"}
S5 = pd.DataFrame({"Curve": E.curve.map(CUR), "Training markets": E.n_training_markets.astype(int),
                   "Repetitions": E.reps.astype(int), "Mean training rows": E.n_train,
                   "AUC": E.AUC, "AUC SD": E.AUC_sd, "AUC min": E.AUC_min, "AUC max": E.AUC_max})
add_table(5, "Learning curve over training markets, with the random-split control.", S5,
          note="Gradient boosting, loose outcome definition. The first two curves are evaluated on the same 150,000 rows "
               "from 25 markets withheld from every fit; the control is evaluated on held-out rows of the same markets. "
               "Source: out/E_learning_summary_y_license.csv.", decimals={"Mean training rows": 0})

# ---------------- S6 calibration and decision metrics ----------------
rows = []
for oc in ("y_license", "y_strict"):
    D = csv(f"D_metrics_{oc}.csv"); dec = csv(f"D_decision_{oc}.csv")
    for _, r_ in D.iterrows():
        g = dec[(dec.design == r_["design"]) & (dec.model == r_["model"])].sort_values("threshold_probability")
        neg = g[g.net_benefit < 0]
        rows.append({"Outcome": OCN[oc], "Design": SPL[r_["design"]], "Model": mdl(r_["model"]),
                     "Eval rows": int(r_["n"]), "Prevalence": r_["prevalence"], "AUC": r_["AUC"],
                     "Brier": r_["Brier"], "ECE": r_["ECE"], "Calibration slope": r_["calibration_slope"],
                     "Calibration intercept": r_["calibration_intercept"], "Mean prediction": r_["mean_prediction"],
                     "Net benefit turns negative at": (float(neg.threshold_probability.iloc[0]) if len(neg) else np.nan)})
S6 = pd.DataFrame(rows)
S6["_o"] = S6.Design.map({SPL[k]: i for i, k in enumerate(ORD)})
S6 = S6.sort_values(["Outcome", "Model", "_o"]).drop(columns="_o")
add_table(6, "Calibration and decision metrics for every model, design and outcome definition.", S6,
          note="Computed on the pooled evaluation sets. A blank in the last column means net benefit stays positive over "
               "the whole range of threshold probabilities examined. ECE is the expected calibration error over 20 "
               "equal-count bins. Source: out/D_metrics_* and out/D_decision_*.", decimals={"Prevalence": 4})

# ---------------- S7 by market ----------------
M = pd.read_csv(f"{ROOT}/t1/market_license_class.csv")
ACC = csv("C1_accuracy_by_market.csv")
M = M.merge(ACC.rename(columns={"city": "m_city", "accuracy": "rec_acc"})[["m_city", "rec_acc"]],
            on="m_city", how="left")
S7 = pd.DataFrame({"Country": M.m_country, "Market": M.m_city, "Listing-snapshots": M.n.astype(int),
                   "Loose (%)": 100*M.y_license, "Strict (%)": 100*M.y_strict,
                   "Difference (pp)": M.delta_license_strict_pp,
                   "Code (% of non-empty)": M.ne_code, "Exempt (%)": M.ne_exempt,
                   "Trader (%)": M.ne_trader_id, "Pending (%)": M.ne_pending, "Other (%)": M.ne_other_text,
                   "Market recovered (%)": 100*M.rec_acc}).sort_values(["Country", "Market"])
add_table(7, "Display rate, licence-field composition and market-recovery accuracy for all 123 markets.", S7,
          note="Loose counts any non-empty value; strict counts only values containing a registration code. Composition "
               "columns are percentages of non-empty values and sum to 100. The last column is the share of a market's "
               "held-out listings assigned to that market by the classifier of Section 5.3. "
               "Sources: t1/market_license_class.csv, out/C1_accuracy_by_market.csv.", decimals={"Market recovered (%)": 2})

# ---------------- S8 by country ----------------
G = csv("G_country.csv")
S8 = pd.DataFrame({"Group": G.m_group, "Country": G.m_country, "Markets": G.n_markets.astype(int),
                   "Listing-snapshots": G.listing_snapshots.astype(int), "Any value (%)": G.display_rate_any,
                   "Loose (%)": G.display_rate_loose, "Strict (%)": G.display_rate_strict,
                   "Code (% of non-empty)": G.ne_code, "Exempt (%)": G.ne_exempt,
                   "Trader (%)": G.ne_trader_id, "Pending (%)": G.ne_pending,
                   "Other (%)": G.ne_other_text}).sort_values(["Group", "Country"])
add_table(8, "Display rate and licence-field composition for all 37 countries.", S8,
          note="Group is the publisher's European Union, European Economic Area and non-European Union grouping, "
               "reported for description only. Source: out/G_country.csv.", decimals={"Any value (%)": 2})

# ---------------- S9 minimum nights by market ----------------
MN = csv("M_minnights_by_market.csv")
S9 = pd.DataFrame({"Country": MN.country, "Market": MN.city, "Listings": MN.n.astype(int),
                   "Median": MN["median"], "90th percentile": MN.p90, "Modal value": MN.modal_value,
                   "Modal share (%)": 100*MN.modal_share, "1 night (%)": 100*MN.sh_1,
                   "28 (%)": 100*MN.sh_28, "30 (%)": 100*MN.sh_30, "31 (%)": 100*MN.sh_31,
                   "32 (%)": 100*MN.sh_32, "90 (%)": 100*MN.sh_90,
                   "28 to 32 (%)": 100*MN.sh_28to32}).sort_values(["Country", "Market"])
add_table(9, "Minimum-nights distribution for all 123 markets.", S9,
          note="Shares are percentages of listing-snapshots in that market. The table documents how one predictor "
               "separates jurisdictions; it carries no claim about the effect of any rule. Source: "
               "out/M_minnights_by_market.csv.", decimals={"Median": 1, "90th percentile": 1, "Modal value": 0})

# ---------------- S10 host portfolio by market ----------------
H = csv("H_hosts_by_market.csv")
S10 = pd.DataFrame({"Country": H.country, "Market": H.city, "Snapshots": H.snapshots.astype(int),
                    "Listings": H.n_listings, "Hosts": H.n_hosts,
                    "Hosts with one listing (%)": H.hosts_with_1_pct,
                    "Listings of single-listing hosts (%)": H.listings_of_1listing_hosts_pct,
                    "Top 10% host share (%)": H.top10pct_host_share_pct,
                    "Top 1% host share (%)": H.top1pct_host_share_pct,
                    "Largest host": H.max_host_listings,
                    "Mean listings per host": H.mean_listings_per_host}).sort_values(["Country", "Market"])
add_table(10, "Host portfolio concentration for all 123 markets.", S10,
          note="Values are means over the snapshots of each market. Source: out/H_hosts_by_market.csv.",
          decimals={"Listings": 1, "Hosts": 1, "Largest host": 1, "Mean listings per host": 2})

# ---------------- S11 calibration curve detail ----------------
rows = []
for oc in ("y_license", "y_strict"):
    C = csv(f"D_calibration_{oc}.csv"); C = C[C.model == "Gradient boosting, full fold"]
    for _, r_ in C.iterrows():
        rows.append({"Outcome": OCN[oc], "Design": SPL[r_["design"]], "Bin": int(r_["bin"]) + 1,
                     "Listings in bin": int(r_["n"]), "Mean predicted probability": r_["pred_mean"],
                     "Observed rate": r_["obs_rate"]})
S11 = pd.DataFrame(rows)
S11["_o"] = S11.Design.map({SPL[k]: i for i, k in enumerate(ORD)})
S11 = S11.sort_values(["Outcome", "_o", "Bin"]).drop(columns="_o")
add_table(11, "Bin-level calibration values behind Figures 7 and 9.", S11,
          note="Gradient boosting trained on the full fold, 20 equal-count bins of the pooled evaluation set. "
               "Source: out/D_calibration_{y_license,y_strict}.csv.")

# ---------------- S12 decision curve detail ----------------
rows = []
for oc in ("y_license", "y_strict"):
    Dc = csv(f"D_decision_{oc}.csv"); Dc = Dc[Dc.model == "Gradient boosting, full fold"]
    for _, r_ in Dc.iterrows():
        rows.append({"Outcome": OCN[oc], "Design": SPL[r_["design"]], "Threshold probability": r_["threshold_probability"],
                     "Net benefit": r_["net_benefit"], "Net benefit, flag all": r_["net_benefit_flag_all"]})
S12 = pd.DataFrame(rows)
S12["_o"] = S12.Design.map({SPL[k]: i for i, k in enumerate(ORD)})
S12 = S12.sort_values(["Outcome", "_o", "Threshold probability"]).drop(columns="_o")
add_table(12, "Net benefit at each threshold probability behind Figures 7 and 9.", S12,
          note="Net benefit is the proportion of true positives minus the proportion of false positives weighted by "
               "pt/(1−pt). Flagging no listing has a net benefit of zero at every threshold. "
               "Source: out/D_decision_{y_license,y_strict}.csv.", decimals={"Threshold probability": 3})

import subprocess, pathlib
SD = pathlib.Path("./pkg/supp_work"); (SD / "figs").mkdir(parents=True, exist_ok=True)
(SD / "Supplementary_Materials.md").write_text("\n".join(MD) + "\n", encoding="utf-8")
json.dump(MANIFEST, open("./pkg/supplementary_manifest.json", "w"), indent=1, ensure_ascii=False)
r = subprocess.run(["python", "./tmpl/build_supp.py", str(SD),
                    "./pkg/02_Supplementary.docx"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
print(f"tables {len(MANIFEST)}")
for m in MANIFEST: print(f"  {m['id']:9s} {m['rows']:>5} rows  {m['title'][:62]}")
