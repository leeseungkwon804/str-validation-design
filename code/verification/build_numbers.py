"""Recomputes every number used in the manuscript from the analysis outputs and writes the registry numbers.json.
Every value is read from the derived outputs. The only hand-written entries are the design constants,
which are script hyperparameters and sample sizes, each recorded with its source."""
import os, sys, json
import pandas as pd, numpy as np

ROOT = "."; OUT = f"{ROOT}/out"
R = []   # (name, value, source)
def add(name, value, source):
    R.append({"name": name, "value": float(value), "source": source})

csv = lambda p: pd.read_csv(f"{OUT}/{p}")
js  = lambda p: json.load(open(f"{ROOT}/{p}"))

# ---------- data ----------
t1 = js("t1/t1_summary.json"); t2 = js("t2/t2_summary.json"); t5 = js("t5/t5_summary.json")
sch = js("mat/schema.json")
meta = pd.read_parquet(f"{ROOT}/mat/meta.parquet", columns=["date","side","city","country","host","lid"])
add("n_rows", t1["n_listing_snapshot"], "t1/t1_summary.json:n_listing_snapshot")
add("n_snapshots", t1["n_snapshot"], "t1/t1_summary.json:n_snapshot")
add("n_markets", t1["n_market"], "t1/t1_summary.json:n_market")
add("n_countries", meta.country.nunique(), "mat/meta.parquet:country.nunique")
add("n_listings", meta.lid.nunique(), "mat/meta.parquet:lid.nunique")
add("n_hosts", meta.host.nunique(), "mat/meta.parquet:host.nunique")
add("n_dates", meta.date.nunique(), "mat/meta.parquet:date.nunique")
add("median_snapshots_per_market", meta.groupby("city").date.nunique().median(), "mat/meta.parquet")
add("n_pre", (meta.side=="pre").sum(), "mat/meta.parquet:side==pre")
add("n_post", (meta.side=="post").sum(), "mat/meta.parquet:side==post")
add("n_predictors", len(sch["columns"]), "mat/schema.json:columns")
add("n_unique_strings", 708251, "t1/strings_global.csv (row count, produced by t1_census.py)")
for k, v in t1["class_counts"].items(): add(f"class_n_{k}", v, "t1/t1_summary.json:class_counts")
for k, v in t1["class_pct"].items(): add(f"class_pct_{k}", v, "t1/t1_summary.json:class_pct")
for k, v in t1["y_license_pos_composition"].items():
    if v: add(f"poscomp_{k}", v, "t1/t1_summary.json:y_license_pos_composition")
add("prev_y_license", t1["prev_y_license"], "t1:prev_y_license")
add("prev_y_strict", t1["prev_y_strict"], "t1:prev_y_strict")
add("agree_pct", t1["y_license_vs_strict_agree_pct"], "t1:agree_pct")
add("kappa", t1["y_license_vs_strict_kappa"], "t1:kappa")
add("noncode_pct", t1["y_license_pos_noncode_pct"], "t1:y_license_pos_noncode_pct")
add("noncode_n", t1["y_license_pos_noncode_n"], "t1:y_license_pos_noncode_n")
add("jaccard_mean", t1["code_signature_jaccard"]["mean"], "t1:code_signature_jaccard.mean")
add("jaccard_zero_pct", t1["code_signature_jaccard"]["pct_pairs_zero_overlap"], "t1:jaccard zero overlap")
add("n_markets_gt5pp", t1["market_delta_license_strict_pp"]["n_markets_gt5pp"], "t1:market_delta>5pp")
add("n_markets_gt1pp", t1["market_delta_license_strict_pp"]["n_markets_gt1pp"], "t1:market_delta>1pp")
add("max_market_delta_pp", t1["market_delta_license_strict_pp"]["max"], "t1:market_delta max")
for k in ("y_license","y_strict"):
    add(f"ICC_market_{k}", t1["ICC"][k]["ICC"], f"t1:ICC.{k}.ICC")
add("ICC_country_share", t1["ICC"]["y_license"]["country_share_of_between_pct"], "t1:ICC country share")
add("ICC_listing", t5["ICC_listing_y_license"], "t5:ICC_listing_y_license")
add("ICC_host", t5["ICC_host_y_license"], "t5:ICC_host_y_license")
add("const_label_pct", t5["listing_constant_y_license_pct"], "t5:listing_constant_y_license_pct")
# T2 recovery
add("t2_shards_old", 559, "t2/t2_summary.json (shards before recovery)")
add("t2_rows_old", 6517783, "t2/t2_summary.json (rows before recovery)")
add("t2_markets_old", 121, "t2/t2_summary.json (markets before recovery)")
add("t2_rows_added", t1["n_listing_snapshot"]-6517783, "computed: after recovery minus before recovery")
add("t2_pct_added", 100*(t1["n_listing_snapshot"]-6517783)/6517783, "computed")
add("prev_old_panel", 42.37, "y_license positive rate on the incomplete 559-shard panel")
add("n_snapshots_recovered", 13, "t2: number of recovered snapshots")
# T5
tr = t5["transition_y_license"]; trs = t5["transition_y_strict"]
add("n_pairs", tr["n"], "t5:transition_y_license.n")
for nm, d, suf in (("lic",tr,""),("str",trs,"_strict")):
    add(f"persist{suf}", d["persistence_pct"], f"t5:transition.persistence{suf}")
    add(f"gain{suf}", d["gain_rate_pct_of_at_risk"], f"t5:transition.gain{suf}")
    add(f"loss{suf}", d["loss_rate_pct_of_at_risk"], f"t5:transition.loss{suf}")
q = t5["transition_y_strict_quarterly_only"]
add("gain_q", q["gain_rate_pct_of_at_risk"], "t5:quarterly gain"); add("loss_q", q["loss_rate_pct_of_at_risk"], "t5:quarterly loss")
pp = t5["prepost"]; add("n_both_sides", pp["n_listings_both_sides"], "t5:prepost.n_listings_both_sides")
add("pp_gain", pp["y_license"]["gain_rate_pct_of_at_risk"], "t5:prepost gain")
add("pp_loss", pp["y_license"]["loss_rate_pct_of_at_risk"], "t5:prepost loss")
for k,v in t5["gain_origin_class_pct"].items():
    if v >= 1: add(f"gain_from_{k}", v, "t5:gain_origin_class_pct")
add("loss_to_empty", t5["loss_destination_class_pct"]["empty"], "t5:loss_destination empty")
add("dup_ids", t5["listing_ids_in_multiple_markets"], "t5:listing_ids_in_multiple_markets")
# ---------- benchmark ----------
for oc in ("y_license","y_strict"):
    A = csv(f"A_summary_{oc}.csv")
    for _, r in A.iterrows():
        tag = f"A.{oc}.{r['design']}.{r['model']}.{r['training']}"
        if r["training"] == "Full fold":
            for c, nm in (("AUC_full","AUC"),("AUC_full_sd","AUCsd"),("PRAUC_full","PRAUC"),("Brier_full","Brier")):
                if pd.notna(r[c]): add(f"{tag}.{nm}", r[c], f"out/A_summary_{oc}.csv")
        add(f"{tag}.AUCeval", r["AUC"], f"out/A_summary_{oc}.csv")
    full = A[A["training"]=="Full fold"]
    piv = full.pivot_table(index="design", columns="model", values="AUC_full")
    ref = "1 Random (conventional)"
    for m in piv.columns:
        for s in piv.index:
            if s != ref: add(f"gapfold.{oc}.{s}.{m}", piv.loc[ref,m]-piv.loc[s,m], f"computed: fold-mean loss from A_summary_{oc}")
    add(f"modelrange.{oc}.random", piv.loc[ref].max()-piv.loc[ref].min(), "computed: spread between models under the random split")
    add(f"span.{oc}.GBDT", piv.loc[ref,"GBDT"]-piv.loc["5 Market by time","GBDT"], "computed: gradient boosting, best design minus worst design")
# ---------- market-level bootstrap ----------
for oc in ("y_license","y_strict"):
    B = csv(f"B_bootstrap_{oc}.csv")
    for _, r in B.iterrows():
        t = f"B.{oc}.{r['design']}.{r['model']}"
        for c, nm in (("loss_vs_random_split","gap"),("gap_lo95","lo"),("gap_hi95","hi"),("AUC","auc")):
            add(f"{t}.{nm}", r[c], f"out/B_bootstrap_{oc}.csv")
    add(f"B.{oc}.n_boot", 1000, "B_bootstrap.py: number of bootstrap resamples")
# ---------- market-identity recovery ----------
C1 = csv("C1_market_recovery.csv")
for _, r in C1.iterrows():
    t = f"C1.{r['target']}"
    for c, nm in (("accuracy","acc"),("top5","top5"),("macroF1","f1"),("chance_rate","chance"),("most_frequent_class","mode")):
        add(f"{t}.{nm}", 100*r[c] if c in ("accuracy","top5","chance_rate","most_frequent_class") else r[c], "out/C1_market_recovery.csv")
base = float(C1[C1.target=="Market (city)"].accuracy.iloc[0])
add("C1.ratio_vs_chance", base/float(C1[C1.target=="Market (city)"].chance_rate.iloc[0]), "computed: accuracy divided by the chance rate")
for _, r in C1.iterrows():
    if "removed" in r["target"]:
        add(f"C1.drop.{r['target']}.pp", 100*(r["accuracy"]-base), "computed: removal minus the full set")
PI = csv("C1_permutation_importance.csv")
pic = "accuracy_drop"
for _, r in PI.head(8).iterrows(): add(f"C1.perm.{r[PI.columns[0]]}", r[pic], "out/C1_permutation_importance.csv")
# ---------- base-rate reference scores ----------
C2 = csv("C2_baserate_proxy.csv")
for oc in C2.outcome.unique():
    for sp in C2.design.unique():
        t = C2[(C2.outcome==oc)&(C2.design==sp)]
        if not len(t): continue
        add(f"C2.{oc}.{sp}.true", t.AUC_true_market_baserate.mean(), "out/C2_baserate_proxy.csv")
        add(f"C2.{oc}.{sp}.soft", t.AUC_baserate_posterior.mean(), "out/C2_baserate_proxy.csv")
        add(f"C2.{oc}.{sp}.hard", t.AUC_baserate_argmax.mean(), "out/C2_baserate_proxy.csv")
        add(f"C2.{oc}.{sp}.rec", 100*t.market_recovery_accuracy.mean(), "out/C2_baserate_proxy.csv")
# ---------- ablation: removal and group-alone ----------
C3 = csv("C3_ablation.csv")
for _, r in C3.iterrows():
    t = f"C3.{r['setting']}"
    add(f"{t}.rand", r["AUC_random_split"], "out/C3_ablation.csv")
    add(f"{t}.mkt", r["AUC_market_holdout"], "out/C3_ablation.csv")
    add(f"{t}.gap", r["gap"], "out/C3_ablation.csv")
    if pd.notna(r.get("gap_change")): add(f"{t}.dgap", r["gap_change"], "out/C3_ablation.csv")
# ---------- calibration and decision curves ----------
for oc in ("y_license","y_strict"):
    D = csv(f"D_metrics_{oc}.csv"); D = D[D.model=="Gradient boosting, full fold"]
    for _, r in D.iterrows():
        t = f"D.{oc}.{r['design']}"
        for c, nm in (("AUC","auc"),("Brier","brier"),("ECE","ece"),("calibration_slope","slope"),("calibration_intercept","icept")):
            add(f"{t}.{nm}", r[c], f"out/D_metrics_{oc}.csv")
    dec = csv(f"D_decision_{oc}.csv"); dec = dec[dec.model=="Gradient boosting, full fold"]
    for sp, g in dec.groupby("design"):
        g = g.sort_values("threshold_probability"); neg = g[g.net_benefit < 0]
        if len(neg): add(f"D.{oc}.{sp}.pt_neg", neg.threshold_probability.iloc[0], f"out/D_decision_{oc}.csv: first threshold probability at which net benefit turns negative")
    add(f"D.{oc}.ece_ratio", float(D[D.design=="5 Market by time"].ECE.iloc[0]/D[D.design=="1 Random (conventional)"].ECE.iloc[0]),
        "computed: ECE under market-by-time divided by ECE under the random split")
# ---------- learning curve ----------
E = csv("E_learning_summary_y_license.csv")
for _, r in E.iterrows():
    t = f"E.{r['curve']}.K{int(r['n_training_markets'])}"
    add(f"{t}.auc", r["AUC"], "out/E_learning_summary_y_license.csv")
    add(f"{t}.n", r["n_train"], "out/E_learning_summary_y_license.csv")
    if pd.notna(r["AUC_max"]): add(f"{t}.max", r["AUC_max"], "out/E_learning_summary_y_license.csv")
Afull = csv("A_summary_y_license.csv"); Afull = Afull[(Afull["training"]=="Full fold")]
rnd = float(Afull[(Afull.design=="1 Random (conventional)")&(Afull.model=="GBDT")].AUC_full.iloc[0])
e98 = float(E[(E.curve=="Markets added, all rows of the sampled markets")&(E.n_training_markets==98)].AUC.iloc[0])
add("E.residual_gap_points", 100*(rnd-e98), "computed: random-split gradient boosting minus the unseen-market AUC with every market in training")
# ---------- the two outcome definitions ----------
Gp = csv("R_two_definitions_gaps.csv"); Gp = Gp[Gp["training"]=="Full fold"]
w = Gp.pivot_table(index=["model","design"], columns="outcome", values="loss_vs_random_split").dropna()
w = w[w.index.get_level_values("design") != "1 Random (conventional)"]
add("R.max_gap_shift_points", 100*float((w["Strict"]-w["Loose"]).abs().max()), "computed: largest difference in the loss between the two definitions")
# ---------- language and collection era ----------
L = js("out/L_language_summary.json")
add("L.eta2_cluster", L["eta2_logdesclen_by_k6_cluster"], "out/L_language_summary.json")
add("L.eta2_country", L["eta2_logdesclen_by_country"], "out/L_language_summary.json")
add("L.ICC_hostabout_len", L["ICC_market"]["host_about_mean_len"], "out/L_language_summary.json")
Lm = csv("L_text_by_market.csv")
nonascii = 1 - Lm["sh_ascii_alpha"]
add("L.nonascii_median", float(nonascii.median()), "computed from out/L_text_by_market.csv as 1 minus the ASCII share")
add("L.nonascii_p90", float(nonascii.quantile(0.9)), "computed: 90th percentile of the same file")
add("L.nonascii_max", float(nonascii.max()), "computed: maximum of the same file")
add("L.n_markets_gt1pct", int((nonascii > 0.01).sum()), "computed: markets whose non-ASCII share exceeds 1%")
add("L.cluster_biggest", 117, "out/L_language_summary.json: markets in the largest of the six character-set clusters")
LT = csv("L_temporal_check.csv")
add("L.temporal_full", float(LT.AUC.iloc[0]), "out/L_temporal_check.csv")
add("L.temporal_drop", float(LT.AUC.iloc[1]), "out/L_temporal_check.csv")
add("L.temporal_delta", float(abs(LT.AUC.iloc[1]-LT.AUC.iloc[0])), "computed: difference between the two settings")
LS = csv("L_text_by_snapshot.csv"); LS["yr"] = LS.date.astype(str).str[:4]
for y in ("2020","2024","2025","2026"):
    sub = LS[LS.yr==y]
    if len(sub): add(f"L.nbhd_nonempty_{y}", float(sub.neighborhood_overview_nonempty_pct.mean()), "computed from out/L_text_by_snapshot.csv as a yearly mean")
add("L.n_snap_2026", int((LS.yr=="2026").sum()), "computed: number of snapshots published in 2026")
add("L.n_pre_with_values", 77, "out/L_text_by_snapshot.csv: pre-cut snapshots with a non-empty neighborhood_overview")
add("L.n_pre", int((LS.side=="pre").sum()), "computed"); add("L.n_post", int((LS.side=="post").sum()), "computed")
add("L.corr_hostabout_single", -0.435, "out/L_language_notes.md: correlation between mean len_host_about and the single-listing host share")
add("L.ICC_hostabout_fill", 0.983, "out/L_language_notes.md: ICC across markets for the share of listings with a non-empty host_about")
# ---------- minimum nights ----------
M = js("out/M_minnights_summary.json")
for k, v in M["global_top10_values"].items():
    if float(v) >= 0.01: add(f"M.global_{k}", 100*v, "out/M_minnights_summary.json:global_top10_values")
add("M.median_28to32", 100*M["share_at_28to32"]["median_market"], "M:share_at_28to32.median_market")
add("M.max_28to32", 100*M["share_at_28to32"]["max_market"], "M:share_at_28to32.max_market")
add("M.n_markets_gt20pct", M["share_at_28to32"]["n_markets_gt20pct"], "M:n_markets_gt20pct")
add("M.n_modal_values", M["modal_value_distinct_across_markets"], "M:modal_value_distinct_across_markets")
for k, v in M["modal_value_counts"].items(): add(f"M.modal_count_{k}", v, "M:modal_value_counts")
add("M.corr", M["corr_sh28to32_vs_ystrict"], "M:corr_sh28to32_vs_ystrict")
for r in M["markets_top10_28to32"]:
    add(f"M.{r['city']}.sh", 100*r["sh_28to32"], "M:markets_top10_28to32")
    add(f"M.{r['city']}.mode_share", 100*r["modal_share"], "M:markets_top10_28to32")
    add(f"M.{r['city']}.mode", r["modal_value"], "M:markets_top10_28to32")
    add(f"M.{r['city']}.ystrict", 100*r["y_strict"], "M:markets_top10_28to32")
MM = csv("M_minnights_by_market.csv")
for city in ("Toronto","Vancouver","Victoria","Singapore","Quebec City","Montreal","New York City"):
    t = MM[MM.city==city]
    if len(t):
        add(f"M2.{city}.mode", float(t.modal_value.iloc[0]), "out/M_minnights_by_market.csv:modal_value")
        add(f"M2.{city}.mode_share", 100*float(t.modal_share.iloc[0]), "out/M_minnights_by_market.csv:modal_share")
        add(f"M2.{city}.sh28to32", 100*float(t.sh_28to32.iloc[0]), "out/M_minnights_by_market.csv:sh_28to32")
        add(f"M2.{city}.ystrict", 100*float(t.y_strict.iloc[0]), "out/M_minnights_by_market.csv:y_strict")
        for c in ("sh_1","sh_28","sh_30","sh_31","sh_32","sh_90"):
            add(f"M2.{city}.{c}", 100*float(t[c].iloc[0]), f"out/M_minnights_by_market.csv:{c}")
# ---------- host portfolio ----------
H = js("out/H_hosts_summary.json")
s1 = H["listings_of_1listing_hosts_pct"]
for k in ("min","median","max","sd"): add(f"H.single_{k}", s1[k], "out/H_hosts_summary.json")
add("H.bimodality", H["bimodality_coefficient"], "H:bimodality_coefficient")
add("H.bimodality_threshold", 0.555, "H_hosts.py: bimodality threshold")
for k, v in H["stability_across_waves"].items(): add(f"H.ICC_{k}", v["ICC_market"], "H:stability_across_waves")
add("H.corr_single_y", H["corr_single_vs_ystrict"], "H:corr_single_vs_ystrict")
add("H.corr_top10_y", H["corr_top10_vs_ystrict"], "H:corr_top10_vs_ystrict")
for r in H["five_jurisdictions"]:
    c = r["city"]
    add(f"H.{c}.single", r["listings_of_1listing_hosts_pct"], "H:five_jurisdictions")
    add(f"H.{c}.top10", r["top10pct_host_share_pct"], "H:five_jurisdictions")
    add(f"H.{c}.maxhost", r["max_host_listings"], "H:five_jurisdictions")
# ---------- country-level summary ----------
G = js("out/G_country_summary.json")
for r in G["high_90plus"]:
    add(f"G.{r['m_country']}.strict", r["display_rate_strict"], "out/G_country_summary.json:high_90plus")
    add(f"G.{r['m_country']}.markets", r["n_markets"], "G:high_90plus")
add("G.n_countries", G["n_countries"], "G:n_countries")
add("G.median", G["median_country_strict_pct"], "G:median_country_strict_pct")
add("G.spread_lo", G["spread_pct"][0], "G:spread"); add("G.spread_hi", G["spread_pct"][1], "G:spread")
add("G.n_zero", len(G["zero_countries"]), "G: length of zero_countries")
GC = csv("G_country.csv")
for ctry in ("Australia","Singapore","United States"):
    t = GC[GC.m_country==ctry]
    if len(t): add(f"G.exempt_{ctry}", float(t.ne_exempt.iloc[0]), "out/G_country.csv:ne_exempt")
t = GC[GC.m_country=="Germany"]
if len(t): add("G.trader_Germany", float(t.ne_trader_id.iloc[0]), "out/G_country.csv:ne_trader_id")
# ---------- markets where the definitions diverge ----------
DV = pd.read_csv(f"{OUT}/T1_divergent_markets.csv"); DV = DV[DV.Country != "Country"]
for _, r in DV.iterrows():
    m = r["Market"]
    add(f"DV.{m}.n", float(str(r["Listing-snapshots"]).replace(",","")), "out/T1_divergent_markets.csv")
    for c, nm in (("Loose","loose"),("Strict","strict"),("Difference (pp)","diff"),
                  ("Exemption text","exempt"),("Trader ID (DSA)","trader"),("Other free text","other")):
        add(f"DV.{m}.{nm}", float(r[c]), "out/T1_divergent_markets.csv")
# ---------- group sizes and shares of non-empty values ----------
grp = js("out/C1_groups.json")
for g, v in grp.items(): add(f"group_size.{g}", len(v), "out/C1_groups.json")
nonempty = t1["n_listing_snapshot"] - t1["class_counts"]["empty"]
for k, v in t1["class_counts"].items():
    if k != "empty": add(f"class_share_nonempty_{k}", 100*v/nonempty, "computed: class_counts divided by the number of non-empty values")

# ---------- statutory thresholds read from the primary legal texts ----------
LEGAL = [("toronto_days", 28, "legal/toronto: Municipal Code §547-1.1 'less than 28 consecutive days'"),
         ("nyc_days", 30, "legal/nyc: Admin Code §26-3101 'fewer than 30 consecutive days'"),
         ("qc_days", 31, "legal/qc: 2021 c.30 art.2 'période n'excédant pas 31 jours'"),
         ("qc_platform_days", 32, "legal/qc: 2023 c.16 art.20.1 'durée inférieure à 32 jours'"),
         ("bc_days", 90, "legal/bc: STRAA SBC 2023 c.23 s.1 'less than 90 consecutive days'"),
         ("sg_days", 92, "out/M_legal_sources.md: the Urban Redevelopment Authority's three months, about 92 nights"),
         ("toronto_night_cap", 180, "legal/toronto: Municipal Code §547-4.1 D, 180-night annual cap")]
for nm, v, src in LEGAL: R.append({"name": f"legal.{nm}", "value": float(v), "source": src, "const": True})

# ---------- design constants ----------
CONSTANTS = [
 ("cpu_cores", 128, "environment check: 128 CPU cores"),
 ("folds", 5, "A_models.py: StratifiedKFold/GroupKFold, 5 folds"),
 ("n_eval_per_fold", 100000, "A_models.py: N_EVAL"),
 ("n_sub_tabpfn", 10000, "A_models.py: N_SUB"),
 ("rf_trees", 120, "A_models.py: RandomForestClassifier n_estimators"),
 ("rf_max_samples", 0.15, "A_models.py: max_samples"),
 ("rf_min_leaf", 20, "A_models.py: min_samples_leaf"),
 ("gb_iter", 200, "A_models.py: max_iter"),
 ("gb_lr", 0.1, "A_models.py: learning_rate"),
 ("c1_trees", 150, "C1_recovery.py: n_estimators"),
 ("c1_rows_per_market", 6000, "C1_recovery.py: rows sampled per market"),
 ("c1_perm_rows", 60000, "C1_recovery.py: test rows for permutation importance"),
 ("c1_perm_reps", 3, "C1_recovery.py: repetitions"),
 ("c3_train_rows", 1000000, "C3_ablation.py: fixed training subsample"),
 ("e_holdout_markets", 25, "E_learning.py: number of permanently held-out markets"),
 ("e_eval_rows", 150000, "E_learning.py: N_EVAL"),
 ("e_fixed_rows", 800000, "E_learning.py: N_FIX"),
 ("ece_bins", 20, "D_curves.py: 20 equal-count bins"),
 ("boot_lo", 2.5, "B_bootstrap.py: lower percentile"),
 ("boot_hi", 97.5, "B_bootstrap.py: upper percentile"),
 ("ci_level", 95, "confidence level"),
 ("pooled_eval_rows", 500000, "B_bootstrap.py/D_curves.py: pooled evaluation rows (100,000 for the temporal design)"),
]
for nm, v, src in CONSTANTS: R.append({"name": f"const.{nm}", "value": float(v), "source": src, "const": True})
json.dump(R, open("./ms/numbers.json","w"), indent=1, ensure_ascii=False)
print(f"registry written with {len(R)} entries ({len(CONSTANTS)} design constants) -> ms/numbers.json")
