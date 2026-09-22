"""
Kernel density estimate (KDE) plot of 206Pb/238U ages for zircon analyses
of Type "SM" and Type "RP", overlaid in a single chart.

Each analysis is represented as a Gaussian kernel centered on its age with
standard deviation equal to its reported 1-sigma analytical uncertainty
(the standard "probability density plot" convention used in U-Pb
geochronology). The per-type curve is the average of all its analyses'
Gaussians, so each curve integrates to 1 regardless of sample size.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config ---------------------------------------------------------------

CSV_PATH = r"data_summary_0724_U-Pb.csv"
AGE_COL = "206Pb-238U_Age"
UNC_COL = "1σ"  # "1σ"
TYPE_COL = "Type"

TYPES = ["SM", "RP"]
COLORS = {"SM": "#2a78d6", "RP": "#eb6834"}  # validated categorical slots 1 & 2

OUT_PNG = "KDE_SM_RP_206Pb-238U_Age.png"
OUT_PDF = "KDE_SM_RP_206Pb-238U_Age.pdf"

# ---- Load & clean -----------------------------------------------------------

df = pd.read_csv(CSV_PATH, skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
df[TYPE_COL] = df[TYPE_COL].str.strip()

df = df.dropna(subset=[AGE_COL, UNC_COL, TYPE_COL])
df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
df[UNC_COL] = pd.to_numeric(df[UNC_COL], errors="coerce")
df = df.dropna(subset=[AGE_COL, UNC_COL])

subsets = {t: df[df[TYPE_COL] == t] for t in TYPES}
for t, sub in subsets.items():
    if sub.empty:
        raise ValueError(f'No rows found for Type == "{t}"')

# ---- Build the age axis -----------------------------------------------------
# Ages can't be negative, so the axis always starts at 0 regardless of padding.

pad = 3 * df[UNC_COL].max()
age_min = 0
age_max = df[AGE_COL].max() + pad
x = np.linspace(age_min, age_max, 2000)

# ---- KDE: sum of per-analysis Gaussians (mean=age, sd=1sigma), normalized --

def kde_curve(ages, uncertainties, x):
    ages = ages.to_numpy()[:, None]
    sigmas = uncertainties.to_numpy()[:, None]
    gaussians = np.exp(-0.5 * ((x[None, :] - ages) / sigmas) ** 2) / (
        sigmas * np.sqrt(2 * np.pi)
    )
    return gaussians.mean(axis=0)

curves = {
    t: kde_curve(sub[AGE_COL], sub[UNC_COL], x) for t, sub in subsets.items()
}

# ---- Plot -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

rug_y0, rug_h = -0.06, 0.04  # rug ticks below the density curves, in axes coords
for t in TYPES:
    n = len(subsets[t])
    ax.plot(x, curves[t], color=COLORS[t], lw=2, label=f"{t} (n = {n})")
    ax.fill_between(x, curves[t], color=COLORS[t], alpha=0.15, lw=0)
    ax.plot(
        subsets[t][AGE_COL],
        np.full(len(subsets[t]), rug_y0),
        "|",
        color=COLORS[t],
        markersize=8,
        markeredgewidth=1.2,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )

top = max(curve.max() for curve in curves.values())
ax.set_ylim(0, top * 1.15)  # relative probability = 0 sits exactly on the x-axis
ax.set_xlim(age_min, age_max)  # x-axis starts at 0, no negative ages
ax.set_xlabel("$^{206}$Pb/$^{238}$U Age (Ma)", color="#0b0b0b")
ax.set_ylabel("Relative probability", color="#0b0b0b")
# ax.set_title("KDE of $^{206}$Pb/$^{238}$U Age — Type SM vs. Type RP", color="#0b0b0b")

ax.set_yticks([])
ax.tick_params(colors="#52514e")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.spines["left"].set_color("#c3c2b7")  # sits at x=0 since xlim starts at 0
ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)

legend = ax.legend(frameon=False, loc="upper right")
for text in legend.get_texts():
    text.set_color("#0b0b0b")

fig.tight_layout()
fig.savefig(OUT_PNG, facecolor=fig.get_facecolor())
fig.savefig(OUT_PDF, facecolor=fig.get_facecolor())
print(f"Saved {OUT_PNG} and {OUT_PDF}")
