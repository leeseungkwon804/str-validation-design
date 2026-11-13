"""T1 figure: whether the licence string means the same thing in every market.
Palette: five slots, all passing the light-mode colour validator.
  (#1baf7a code · #eb6834 exempt · #2a78d6 pending · #e87ba4 trader_id · #6250d6 other_text)
  two contrast warnings, so legends, direct labels and hatching carry the distinction as well as colour"""
import os, json, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "./figs"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
 "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
 "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5,
 "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
 "xtick.major.width": 0.6, "ytick.major.width": 0.6, "figure.dpi": 200, "savefig.dpi": 400,
 "axes.spines.top": False, "axes.spines.right": False,
 "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "grid.color": "#b8b7b0"})
INK, INK2 = "#0b0b0b", "#52514e"
PO = ["code", "exempt", "pending", "trader_id", "other_text"]
CL = {"code": "#1baf7a", "exempt": "#eb6834", "pending": "#2a78d6",
      "trader_id": "#e87ba4", "other_text": "#6250d6"}
HA = {"code": "", "exempt": "///", "pending": "...", "trader_id": "xxx", "other_text": "\\\\\\"}
LB = {"code": "Registration code", "exempt": "Exemption declared", "pending": "Application pending",
      "trader_id": "Trader identification (DSA)", "other_text": "Other free text"}

S = json.load(open("./t1/t1_summary.json"))
mk = pd.read_csv("./t1/market_license_class.csv")

fig = plt.figure(figsize=(7.1, 6.3))
gs = fig.add_gridspec(3, 1, height_ratios=[0.72, 1.25, 1.35], hspace=0.55)

# ---- (a) full composition and the ladder of definitions ----
ax = fig.add_subplot(gs[0])
cc = S["class_counts"]; ne = sum(cc[k] for k in PO)
left = 0.0
for k in PO:
    w = 100 * cc[k] / ne
    ax.barh(0, w, left=left, height=0.5, color=CL[k], edgecolor="#fcfcfb", linewidth=1.2,
            hatch=HA[k] or None)
    if w > 2.5:
        ax.text(left + w / 2, 0, f"{w:.1f}%", ha="center", va="center", fontsize=7.5,
                color="white" if k != "code" else "#0b0b0b", fontweight="bold")
    left += w
for y, (nm, lab) in zip([-0.85, -1.3, -1.75],
                        [("prev_y_any", "y = any non-empty"),
                         ("prev_y_license", "y = non-empty, excl. literal 'Exempt'  (used in benchmark)"),
                         ("prev_y_strict", "y = registration code present  (strict)")]):
    v = S[nm]
    ax.barh(y, v, height=0.3, color="#c9c8c3", edgecolor=INK2, linewidth=0.5)
    ax.text(v + 0.6, y, f"{v:.2f}%  {lab}", va="center", fontsize=7.3, color=INK)
ax.set_xlim(0, 100); ax.set_ylim(-2.1, 0.5)
ax.set_yticks([0, -1.3]); ax.set_yticklabels(["Composition of\nnon-empty values", "Positive rate of\nall listing-snapshots"])
ax.set_xlabel("Per cent"); ax.grid(axis="y", visible=False)
ax.set_title(f"(a) Contents of the licence field  "
             f"(n = {S['n_listing_snapshot']:,} listing-snapshots, {S['n_market']} markets, {S['n_snapshot']} snapshots)",
             loc="center", color=INK)
ax.legend(handles=[Patch(facecolor=CL[k], edgecolor="#fcfcfb", hatch=HA[k] or None, label=LB[k]) for k in PO],
          loc="lower right", bbox_to_anchor=(1.0, 1.18), ncol=3, frameon=False, handlelength=1.6)

# ---- (b) composition by market ----
ax = fig.add_subplot(gs[1])
m2 = mk[mk.nonempty_n > 0].copy().sort_values("ne_code", ascending=False).reset_index(drop=True)
x = np.arange(len(m2)); bot = np.zeros(len(m2))
for k in PO:
    v = 100 * m2[f"ne_{k}"].fillna(0).values
    ax.bar(x, v, bottom=bot, width=0.86, color=CL[k], linewidth=0, hatch=HA[k] or None)
    bot += v
ax.set_xlim(-0.8, len(m2) - 0.2); ax.set_ylim(0, 100.5)
ax.set_ylabel("Share of non-empty values (%)")
ax.set_xlabel("Markets, ordered by registration-code share", labelpad=6)
ax.grid(axis="x", visible=False)
ax.set_xticks([])
lo = m2.nsmallest(8, "ne_code").sort_values("ne_code")
it = [f"{r.m_city} {100*r.ne_code:.0f}%" for r in lo.itertuples()]
ax.text(0.015, 0.06, f"{len(m2)} of {S['n_market']} markets have any non-empty value; "
        f"the other {S['markets_all_empty']} are empty throughout.\nLowest registration-code share\n"
        + ",  ".join(it[:4]) + ",\n" + ",  ".join(it[4:]),
        transform=ax.transAxes, fontsize=6.6, color=INK, va="bottom", linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fcfcfb", edgecolor="#d8d7d2", linewidth=0.5))
ax.set_title("(b) Composition of non-empty values by market", loc="center", color=INK)

# ---- (c) market-level divergence between the two definitions ----
ax = fig.add_subplot(gs[2])
ax.plot([0, 100], [0, 100], "-", color=INK2, lw=0.8, zorder=1)
sz = 6 + 26 * (mk.n / mk.n.max()) ** 0.5
ax.scatter(100 * mk.y_strict, 100 * mk.y_license, s=sz, facecolor="#2a78d6", alpha=0.55,
           edgecolor="#fcfcfb", linewidth=0.6, zorder=3)
OFF = {"Tasmania": (6, -2), "Singapore": (6, -2), "Berlin": (6, -2), "Boston": (-6, 7),
       "Paris": (6, -8), "Victoria": (6, 3)}
for r in mk.nlargest(6, "delta_license_strict_pp").itertuples():
    dx, dy = OFF.get(r.m_city, (6, -2))
    ax.annotate(f"{r.m_city} (+{r.delta_license_strict_pp:.0f} pp)",
                (100 * r.y_strict, 100 * r.y_license), xytext=(dx, dy),
                textcoords="offset points", fontsize=6.6, color=INK,
                ha="right" if dx < 0 else "left")
ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
ax.set_xlabel("Positive rate under the strict definition (%, registration code present)")
ax.set_ylabel("Positive rate as used\nin the benchmark (%)")
ax.set_title("(c) Loose against strict positive rate by market "
             f"(κ = {S['y_license_vs_strict_kappa']:.3f}; "
             f"{S['market_delta_license_strict_pp']['n_markets_gt5pp']} markets differ by > 5 pp, max "
             f"{S['market_delta_license_strict_pp']['max']:.0f} pp)", loc="center", color=INK)
ax.text(0.98, 0.04, "point area scales with listing-snapshots", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6.6, color=INK2)

fig.savefig(f"{OUT}/T1_license_semantics.png", bbox_inches="tight")
fig.savefig(f"{OUT}/T1_license_semantics.pdf", bbox_inches="tight")
print("saved:", f"{OUT}/T1_license_semantics.png / .pdf")
