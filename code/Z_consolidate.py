"""Z. Collect every derived quantity into one key-value file.
Collects numbers only. Missing outputs are skipped and reported."""
import os, sys, json, glob
import pandas as pd, numpy as np

OUT = "./out"; ROOT = "."
R, missing = {}, []

def jload(p, key):
    if os.path.exists(p): R[key] = json.load(open(p))
    else: missing.append(p)

def cload(p):
    return pd.read_csv(p) if os.path.exists(p) else None

jload(f"{ROOT}/t1/t1_summary.json", "T1_license_semantics")
jload(f"{ROOT}/t2/t2_summary.json", "T2_recovery")
jload(f"{ROOT}/t2/balanced_panel.json", "T2_balanced_panel")
jload(f"{ROOT}/t5/t5_summary.json", "T5_transitions")
jload(f"{OUT}/C_summary.json", "C_mechanism")
jload(f"{OUT}/G_country_summary.json", "G_country")

A = cload(f"{OUT}/A_summary_y_license.csv")
if A is not None:
    R["A_models"] = A.to_dict("records")
    full = A[A["training"] == "Full fold"]
    piv = full.pivot_table(index="design", columns="model", values="AUC_full")
    R["A_headline_AUC_full"] = piv.round(4).to_dict()
    ref = "1 Random (conventional)"
    R["A_gap_vs_random"] = {m: {s: round(float(piv.loc[ref, m] - piv.loc[s, m]), 4)
                                for s in piv.index} for m in piv.columns}
else:
    missing.append(f"{OUT}/A_summary_y_license.csv")

B = cload(f"{OUT}/B_bootstrap_y_license.csv")
if B is not None:
    R["B_bootstrap"] = B.to_dict("records")
    key = B[B["model"] == "Gradient boosting, full fold"]
    R["B_gbdt_gap_ci"] = {r["design"]: [r["loss_vs_random_split"], r["gap_lo95"], r["gap_hi95"]]
                          for _, r in key.iterrows()}
else:
    missing.append(f"{OUT}/B_bootstrap_y_license.csv")

for nm, p in (("C1_market_recovery", f"{OUT}/C1_market_recovery.csv"),
              ("C2_baserate_proxy", f"{OUT}/C2_baserate_proxy.csv"),
              ("C3_ablation", f"{OUT}/C3_ablation.csv"),
              ("D_metrics", f"{OUT}/D_metrics_y_license.csv"),
              ("E_learning", f"{OUT}/E_learning_summary_y_license.csv"),
              ("G_country", f"{OUT}/G_country.csv")):
    t = cload(p)
    if t is None: missing.append(p)
    else: R[nm] = t.to_dict("records")

# ---- parallel run under the strict definition ----
As = cload(f"{OUT}/A_summary_y_strict.csv")
if As is not None:
    R["A_models_strict"] = As.to_dict("records")
    fulls = As[As["training"] == "Full fold"]
    pivs = fulls.pivot_table(index="design", columns="model", values="AUC_full")
    R["A_headline_AUC_full_strict"] = pivs.round(4).to_dict()
    ref = "1 Random (conventional)"
    R["A_gap_vs_random_strict"] = {m: {s: round(float(pivs.loc[ref, m] - pivs.loc[s, m]), 4)
                                       for s in pivs.index} for m in pivs.columns}
else:
    missing.append(f"{OUT}/A_summary_y_strict.csv")

Bs = cload(f"{OUT}/B_bootstrap_y_strict.csv")
if Bs is not None:
    R["B_bootstrap_strict"] = Bs.to_dict("records")
    ks = Bs[Bs["model"] == "Gradient boosting, full fold"]
    R["B_gbdt_gap_ci_strict"] = {r["design"]: [r["loss_vs_random_split"], r["gap_lo95"], r["gap_hi95"]]
                                 for _, r in ks.iterrows()}
else:
    missing.append(f"{OUT}/B_bootstrap_y_strict.csv")

for nm, p2 in (("D_metrics_strict", f"{OUT}/D_metrics_y_strict.csv"),
               ("R_two_definitions", f"{OUT}/R_two_definitions.csv"),
               ("R_two_definitions_gaps", f"{OUT}/R_two_definitions_gaps.csv"),
               ("R_two_definitions_core", f"{OUT}/R_two_definitions_core.csv"),
               ("T1_divergent_markets", f"{OUT}/T1_divergent_markets.csv")):
    t = cload(p2)
    if t is None: missing.append(p2)
    else: R[nm] = t.to_dict("records")

R["_files"] = sorted(os.path.basename(p) for p in glob.glob(f"{OUT}/*") )
R["_figures"] = sorted(os.path.basename(p) for p in glob.glob(f"{ROOT}/figs/*.png"))
R["_missing"] = missing
json.dump(R, open(f"{OUT}/results_all.json", "w"), indent=1, ensure_ascii=False)
print(f"{len(R)} keys written -> {OUT}/results_all.json")
if missing:
    print("outputs not yet present:"); [print("  -", m) for m in missing]
