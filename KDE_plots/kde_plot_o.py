"""
Kernel density estimate (KDE) plot of d18O for zircon analyses grouped by
"Replaced" category (Y, N), overlaid in a single chart.

Each analysis is represented as a Gaussian kernel centered on its d18O
value with standard deviation equal to its reported 1-sigma analytical
uncertainty (the standard "probability density plot" convention used in
isotope geochemistry). Each category's curve is the average of its
analyses' Gaussians (integrates to 1), then rescaled so its own peak = 1 -
"Y" analyses carry uncertainties roughly 10-20x larger than "N" (2.4-3.8
vs. 0.02-0.47), so without this rescaling the Y curve is nearly invisible
next to N (peak ratio ~24x). Curve height is therefore comparable in shape
only, not in sample size or precision.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config ---------------------------------------------------------------

XLSX_PATH = r"AFG_O_Isotopes.xlsx"
VALUE_COL = "d18O"
UNC_COL = "1s"
CAT_COL = "Replaced"

CATEGORIES = ["Y", "N"]
LABELS = {
    "Y": "Replacement zone",
    "N": "Unaltered core",
}
COLORS = {
    "Y": "#2a78d6",  # categorical slot 1
    "N": "#eb6834",  # categorical slot 2
}

OUT_PNG = "KDE_YN_d18O.png"
OUT_PDF = "KDE_YN_d18O.pdf"

# ---- Load & clean -----------------------------------------------------------

df = pd.read_excel(XLSX_PATH)
df.columns = [str(c).strip() for c in df.columns]
df[CAT_COL] = df[CAT_COL].astype(str).str.strip()

df = df.dropna(subset=[VALUE_COL, UNC_COL, CAT_COL])
df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
df[UNC_COL] = pd.to_numeric(df[UNC_COL], errors="coerce")
df = df.dropna(subset=[VALUE_COL, UNC_COL])

subsets = {c: df[df[CAT_COL] == c] for c in CATEGORIES}
for c, sub in subsets.items():
    if sub.empty:
        raise ValueError(f'No rows found for {CAT_COL} == "{c}"')

# ---- Build the value axis ----------------------------------------------------

row_min = df.loc[df[VALUE_COL].idxmin()]
row_max = df.loc[df[VALUE_COL].idxmax()]
x_min = row_min[VALUE_COL] - 3 * row_min[UNC_COL]
x_max = row_max[VALUE_COL] + 3 * row_max[UNC_COL]
x = np.linspace(x_min, x_max, 3000)

# ---- KDE: sum of per-analysis Gaussians (mean=value, sd=1sigma), normalized

def kde_curve(values, uncertainties, x):
    values = values.to_numpy()[:, None]
    sigmas = uncertainties.to_numpy()[:, None]
    gaussians = np.exp(-0.5 * ((x[None, :] - values) / sigmas) ** 2) / (
        sigmas * np.sqrt(2 * np.pi)
    )
    return gaussians.mean(axis=0)


curves = {
    c: kde_curve(sub[VALUE_COL], sub[UNC_COL], x) for c, sub in subsets.items()
}

# Rescale each curve to its own peak = 1 (see module docstring).
curves = {c: curve / curve.max() for c, curve in curves.items()}

# ---- Plot -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

rug_y0 = -0.06  # rug ticks below the density curves, in axes coords
for c in CATEGORIES:
    n = len(subsets[c])
    ax.plot(x, curves[c], color=COLORS[c], lw=2, label=f"{LABELS[c]} (n = {n})")
    ax.fill_between(x, curves[c], color=COLORS[c], alpha=0.15, lw=0)
    ax.plot(
        subsets[c][VALUE_COL],
        np.full(len(subsets[c]), rug_y0),
        "|",
        color=COLORS[c],
        markersize=8,
        markeredgewidth=1.2,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )

ax.axvline(0, color="#c3c2b7", lw=1, ls="--", zorder=0)

ax.set_ylim(0, 1.15)  # each curve normalized to its own peak = 1; 0 sits on the x-axis
ax.set_xlim(x_min, x_max)
ax.set_xlabel(r"$\delta^{18}$O", color="#0b0b0b", fontsize=14)
ax.set_ylabel("Relative probability", color="#0b0b0b", fontsize=14)

ax.set_yticks([])
ax.tick_params(colors="#52514e", labelsize=12)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.spines["left"].set_color("#c3c2b7")
ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)

legend_order = ["N", "Y"]
handles, labels = ax.get_legend_handles_labels()
by_cat = dict(zip(CATEGORIES, zip(handles, labels)))
handles, labels = zip(*(by_cat[c] for c in legend_order))
legend = ax.legend(
    handles,
    labels,
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,
    fontsize=13,
)
for text in legend.get_texts():
    text.set_color("#0b0b0b")

fig.tight_layout()
fig.savefig(OUT_PNG, facecolor=fig.get_facecolor())
fig.savefig(OUT_PDF, facecolor=fig.get_facecolor())
print(f"Saved {OUT_PNG} and {OUT_PDF}")
