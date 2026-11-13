# Evaluating machine-learning models on short-term rental platform data with a market baseline and a leakage diagnostic

Analysis code and derived quantities for the paper of that title (Sewon Min, Sahmyook University).
Every number reported in the manuscript and in the supplementary tables is reproduced from the files
in `derived/` by the scripts in `code/verification/`.

## What the paper does

A prediction model built on platform data is usually validated by splitting rows at random and reported
without a baseline that says what the model added. The paper sets out a four-step evaluation procedure
and measures what each step returns on 6,938,304 listing-snapshots from 572 public snapshots of 123
short-term rental markets in 37 countries. The task is to predict whether a listing displays a
registration number, from 50 listing and host characteristics that contain no market identifier.

| Step | What it does | Script | Result file |
|---|---|---|---|
| 1. Diagnose | recover market identity from the predictors alone and read the accuracy as a measure of leakage | `code/C1_recovery.py` | `derived/C1_market_recovery.csv` |
| 2. Baseline | score every test row with the training-fold base rate of its own market and require the model to beat it | `code/C2_proxy.py` | `derived/C2_baserate_proxy.csv` |
| 3. Split | group folds by the unit the model must transfer across | `code/A_models.py` | `derived/A_summary_y_license.csv` |
| 4. Interval | resample markets rather than rows | `code/B_bootstrap.py` | `derived/B_bootstrap_y_license.csv` |

## Layout

```
README.md            this file
LICENSE              MIT
code/                analysis scripts, in the order they run
code/verification/   scripts that rebuild and check every reported number
derived/             every derived quantity reported in the paper, as CSV or JSON
derived/data_dictionary.md    the 50 predictors, and the names used in the data files
derived/numbers_registry.json every quantity cited in the paper, with the file it came from
```

Design and model names in `derived/` are byte-identical to those in the manuscript and the
supplementary tables, so a value can be traced from a table cell to the file that produced it.

## What is not here, and why

* **The raw snapshot files are not included.** They are published by Inside Airbnb at
  <https://insideairbnb.com/get-the-data/> under a Creative Commons Attribution 4.0 International
  licence and are not redistributed here. `code/panel.py` and `code/c10lib.py` document how the panel
  was assembled from them.
* **The raw contents of the licence field are not included in any form**, because in some markets that
  field contains personal information. Only the six-way classification of each value and aggregates of
  it are distributed. `derived/t1_code_signatures_by_market.csv` holds masked shapes, with digits
  written as `9` and letters as `A`, not values.
* **Listing-level model predictions are not included**; they are large and add nothing to the reported
  results. The pooled evaluation metrics computed from them are in `derived/`.

This repository is the same material as File S1 of the supplementary information; the two are
interchangeable.

## Reproducing the results

Run from the directory that contains `code/` and `derived/`, with the raw snapshot files placed under
`./data/raw/`. Paths inside the scripts are relative to that directory.

```
python code/panel.py            # resolve the snapshot catalogue
python code/build_features.py   # per-snapshot feature shards
python code/t1_census.py        # census of the licence field
python code/t1_classify.py      # six-way classification, with its self-check
python code/t1_analyze.py       # validity of the outcome definition      -> Table 1, Figure 1
python code/prep.py             # design matrix and metadata              -> Table 2
python code/A_models.py y_license     # benchmark, five designs, four models  -> Tables 4, 5
python code/A_models.py y_strict      # the same under the strict definition  -> Table 12
python code/B_bootstrap.py y_license  # market-level bootstrap                -> Table 6
python code/C1_recovery.py            # market-identity recovery              -> Table 7, Figure 3
python code/C2_proxy.py               # base-rate reference scores            -> Table 8, Figure 4
python code/C3_ablation.py            # feature ablation                      -> Table 9, Figure 5
python code/E_learning.py             # learning curve                        -> Table 10, Figure 6
python code/D_curves.py y_license     # calibration and decision curves       -> Table 11, Figure 7
python code/R_compare.py              # the two outcome definitions           -> Tables 12, 13, Figure 8
python code/Z_consolidate.py          # collect every reported quantity
```

Secondary analyses: `code/G_country.py` (country summary), `code/H_hosts.py` (host portfolios),
`code/M_minnights.py` (minimum nights), `code/L_language.py` and `code/L_temporal_check.py` (text
length and collection era). Figures are drawn by `code/fig_*.py` and `code/t1_fig.py`.

## Verification

```
python code/verification/build_numbers.py        # registry of every reported quantity
python code/verification/check_numbers.py        # every number in the manuscript text
python code/verification/check_tables.py         # every cell of every manuscript table
python code/verification/check_supplementary.py  # every cell of every supplementary table
```

Each script prints the number of mismatches and exits non-zero if any remain.

## Environment

Python 3.11, pandas 2.3.3, scikit-learn 1.9.1, matplotlib, python-docx 1.2.0. The tabular foundation
model uses the open-source TabPFN 2.2.1 package on one GPU; it requires no authentication. Random
seeds are fixed in every script.

## Citation

Min, S. Evaluating machine-learning models on short-term rental platform data with a market baseline
and a leakage diagnostic. Manuscript under review, 2026.

## Licence

Code and derived data in this repository: MIT (see `LICENSE`). The underlying listing data remain
under the licence of their publisher.
