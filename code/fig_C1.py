"""Figure: direct evidence of the leakage route.
(a) recovering the market from the 50 predictors alone: 123 markets and 37 countries, against the chance rate
(b) which group carries the information: accuracy from the group alone, and the loss when it is removed
(c) the 15 variables with the highest permutation importance (drop in accuracy)
Palette: a set that passes the light-mode colour validator; hatching and direct labels mean colour is never the only cue."""
import os, sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "./out"; FIG = "./figs"; os.makedirs(FIG, exist_ok=True)
R = pd.read_csv(f"{OUT}/C1_market_recovery.csv")
P = pd.read_csv(f"{OUT}/C1_permutation_importance.csv")
plt.rcParams.update({
 "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
 "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.4,
 "ytick.labelsize": 7.4, "legend.fontsize": 7.2, "axes.linewidth": 0.6, "figure.dpi": 200,
 "savefig.dpi": 400, "axes.spines.top": False, "axes.spines.right": False,
 "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "grid.color": "#b8b7b0"})
INK, INK2 = "#0b0b0b", "#52514e"
C1, C2, C3 = "#1baf7a", "#2a78d6", "#eb6834"
EN = {"Price": "Price", "Minimum and maximum stay": "Nights limits", "Property type": "Property / room type",
      "Availability": "Availability", "Reviews": "Reviews", "Host portfolio size": "Host portfolio size",
      "Host conduct": "Host behaviour", "Text length": "Text lengths", "Tenure": "Tenure"}
VEN = {"price": "Nightly price", "property_type": "Property type", "minimum_nights": "Minimum nights",
       "host_listings_count": "Host listing count", "calculated_host_listings_count": "Host listings (in file)",
       "estimated_revenue_l365d": "Estimated revenue (365d)",
       "calculated_host_listings_count_entire_homes": "Host listings, entire homes",
       "len_host_about": "Length of host bio", "accommodates": "Guests accommodated",
       "availability_365": "Days available (365)", "bedrooms": "Bedrooms",
       "n_amenities": "Number of amenities", "maximum_nights": "Maximum nights",
       "calculated_host_listings_count_private_rooms": "Host listings, private rooms",
       "len_description": "Length of description", "beds": "Beds", "room_type": "Room type"}

base = R[R["target"] == "Market (city)"].iloc[0]
ctry = R[R["target"] == "Country"].iloc[0]
drop = R[R["target"].str.endswith("removed")].copy()
drop["grp"] = drop["target"].str.replace("n_markets − ", "", regex=False).str.replace(" removed", "", regex=False)
solo = R[R["target"].str.endswith("alone")].copy()
solo["grp"] = solo["target"].str.replace("n_markets − ", "", regex=False).str.replace(" alone", "", regex=False)
drop["delta"] = 100 * (drop["accuracy"] - base["accuracy"])
order = drop.sort_values("delta")["grp"].tolist()
D = drop.set_index("grp").reindex(order); S = solo.set_index("grp").reindex(order)

fig = plt.figure(figsize=(7.1, 7.0))
gs = fig.add_gridspec(3, 1, height_ratios=[0.78, 1.25, 1.15], hspace=0.50)

# ---- (a) ----
ax = fig.add_subplot(gs[0])
xs = np.arange(2); w = 0.3
t1 = 100 * np.array([base["accuracy"], ctry["accuracy"]])
t5 = 100 * np.array([base["top5"], ctry["top5"]])
ch = 100 * np.array([base["chance_rate"], ctry["chance_rate"]])
ax.bar(xs - w / 2, t1, width=w, color=C1, zorder=3, label="Top-1 accuracy")
ax.bar(xs + w / 2, t5, width=w, color=C2, zorder=3, hatch="///", edgecolor="#fcfcfb",
       linewidth=0.6, label="Top-5 accuracy")
for x, v in list(zip(xs - w / 2, t1)) + list(zip(xs + w / 2, t5)):
    ax.annotate(f"{v:.1f}%", (x, v), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=7.4, color=INK)
for i, c in enumerate(ch):
    ax.plot([i - 0.44, i + 0.44], [c, c], ls="--", lw=1.2, color=INK2, zorder=4,
            label="Chance level" if i == 0 else None)
    ax.annotate(f"chance {c:.2f}%", (i + 0.46, c + 1.5), fontsize=6.8, color=INK2, va="bottom")
ax.set_xticks(xs); ax.set_xticklabels(["Market  (123 classes)", "Country  (37 classes)"])
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 95); ax.grid(axis="x", visible=False)
ax.set_title("(a) Market and country recovery from the 50 predictors (no identifier supplied)\n"
             "random forest, listings split between training and test, 212,473 test rows",
             loc="center", color=INK)
ax.legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.005, 0.5))

# ---- (b) ----
ax = fig.add_subplot(gs[1])
ys = np.arange(len(order)); h = 0.36
ax.barh(ys + h / 2, 100 * S["accuracy"].values, height=h, color=C1, zorder=3,
        label="Accuracy from this group alone")
ax.barh(ys - h / 2, -D["delta"].values, height=h, color=C3, zorder=3, hatch="///",
        edgecolor="#fcfcfb", linewidth=0.5, label="Accuracy lost when this group is removed")
for i, v in enumerate(100 * S["accuracy"].values):
    ax.annotate(f"{v:.1f}%", (v, i + h / 2), xytext=(3, 0), textcoords="offset points",
                va="center", fontsize=6.8)
for i, v in enumerate(-D["delta"].values):
    ax.annotate(f"−{v:.1f} pp", (v, i - h / 2), xytext=(3, 0), textcoords="offset points",
                va="center", fontsize=6.8)
ax.axvline(100 * base["chance_rate"], ls="--", lw=1.0, color=INK2, zorder=4)
ax.annotate(f"chance {100*base['chance_rate']:.2f}%", (100 * base["chance_rate"], -0.75),
            xytext=(3, 0), textcoords="offset points", fontsize=6.6, color=INK2, va="center")
ax.set_yticks(ys); ax.set_yticklabels([EN.get(g, g) for g in order])
ax.set_xlim(0, 22); ax.set_ylim(-1.2, len(order) - 0.3); ax.set_xlabel("Per cent")
ax.grid(axis="y", visible=False)
ax.set_title(f"(b) Market recovery by predictor group  "
             f"(all 50 together: {100*base['accuracy']:.1f}%)", loc="center", color=INK)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.005, 0.5))

# ---- (c) ----
ax = fig.add_subplot(gs[2])
top = P.head(15).iloc[::-1]
GCOL = {g: c for g, c in zip(EN, ["#eb6834", "#2a78d6", "#1baf7a", "#e87ba4", "#6250d6",
                                  "#c9a227", "#0f8aa8", "#9b5de5", "#52514e"])}
ys = np.arange(len(top))
ax.barh(ys, top["accuracy_drop"].values, color=[GCOL[g] for g in top["group"]], zorder=3,
        xerr=top["sd"].values, error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=1.8))
ax.set_yticks(ys); ax.set_yticklabels([VEN.get(v, v) for v in top["variable"]])
ax.set_xlabel("Drop in market-classification accuracy when the variable is permuted")
ax.grid(axis="y", visible=False)
ax.set_title("(c) Individual predictors, ranked by permutation importance", loc="center", color=INK)
seen = list(dict.fromkeys(top["group"]))
ax.legend(handles=[Patch(facecolor=GCOL[g], label=EN[g]) for g in seen],
          frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.005, 0.5))

fig.savefig(f"{FIG}/F2_market_recovery.png", bbox_inches="tight")
fig.savefig(f"{FIG}/F2_market_recovery.pdf", bbox_inches="tight")
print("saved:", f"{FIG}/F2_market_recovery.png")
