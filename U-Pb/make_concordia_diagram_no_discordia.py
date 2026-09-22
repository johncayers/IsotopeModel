"""
U-Pb concordia diagram with error ellipses (no discordia).

Concordia:
    x = 207Pb*/235U = exp(lambda235 * t) - 1
    y = 206Pb*/238U = exp(lambda238 * t) - 1

Error ellipses:
    Built from the %error columns (1-sigma) and the reported error
    correlation (rho) using the standard 2x2 covariance -> eigen-
    decomposition approach. Analyses with either %error > 10% are
    excluded from the dataset entirely (see MAX_ERROR_PCT below).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path

# ---------------------------------------------------------------- settings
CSV_PATH = Path(__file__).parent / "U-Pb_DiscordiaDiagram.csv"
OUT_PATH = Path(__file__).parent / "U-Pb_ConcordiaDiagram_no_discordia.png"

LAMBDA235 = 9.8485e-10  # 1/yr
LAMBDA238 = 1.55125e-10  # 1/yr

T_MAX = 3.5e9  # years
AGE_TICKS_GA = np.arange(0, 3.5 + 0.001, 0.5)  # Ga, tick marks along concordia

MAX_ERROR_PCT = 10  # exclude analyses with either %error above this

# ---------------------------------------------------------------- load data
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
df["Class"] = df["Class"].str.strip()

x_col, xerr_col = "207Pb*/235U", "207Pb*/235U ±%"
y_col, yerr_col = "206Pb*/238U", "206Pb*/238U ±%"
rho_col = "err corr"

df = df.dropna(subset=["Class", x_col, y_col])

n_before = len(df)
df = df[(df[xerr_col] <= MAX_ERROR_PCT) & (df[yerr_col] <= MAX_ERROR_PCT)]
print(f"Excluded {n_before - len(df)} of {n_before} analyses with >{MAX_ERROR_PCT}% error")

# ---------------------------------------------------------------- concordia
t_curve = np.linspace(0, T_MAX, 2000)
x_conc = np.exp(LAMBDA235 * t_curve) - 1
y_conc = np.exp(LAMBDA238 * t_curve) - 1

t_ticks = AGE_TICKS_GA * 1e9
x_ticks = np.exp(LAMBDA235 * t_ticks) - 1
y_ticks = np.exp(LAMBDA238 * t_ticks) - 1


# ---------------------------------------------------------------- error ellipses
def error_ellipse_params(x, y, xerr_pct, yerr_pct, rho):
    sx = x * xerr_pct / 100.0
    sy = y * yerr_pct / 100.0
    cov = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    width, height = 2 * np.sqrt(np.clip(eigvals, 0, None))  # full axis lengths
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return width, height, angle


classes = sorted(df["Class"].unique())
cmap = plt.get_cmap("tab10")
class_colors = {c: cmap(i % 10) for i, c in enumerate(classes)}

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(8, 6.5))

data_xmax = df[x_col].max()
data_ymax = df[y_col].max()

ax.plot(x_conc, y_conc, color="black", lw=1.5, zorder=3, label="Concordia")
ax.plot(
    x_ticks, y_ticks, "o", markerfacecolor="black", markeredgecolor="black",
    ms=7, mew=1.3, zorder=4,
)
LABEL_OFFSETS = {0.5: (-28, 12)}  # nudge crowded labels aside; default is above
for tga, xt, yt in zip(AGE_TICKS_GA, x_ticks, y_ticks):
    if tga == 0:
        continue
    xytext = LABEL_OFFSETS.get(tga, (0, 10))
    ax.annotate(
        f"{tga:g}",
        (xt, yt),
        textcoords="offset points",
        xytext=xytext,
        ha="center",
        fontsize=12,
        color="black",
    )

for cls in classes:
    sub = df[df["Class"] == cls]
    color = class_colors[cls]
    for _, row in sub.iterrows():
        w, h, ang = error_ellipse_params(
            row[x_col], row[y_col], row[xerr_col], row[yerr_col], row[rho_col]
        )
        ell = Ellipse(
            (row[x_col], row[y_col]), width=w, height=h, angle=ang,
            facecolor="none", edgecolor=color, linewidth=1.2, zorder=5,
        )
        ax.add_patch(ell)

ax.set_xlabel(r"$^{207}$Pb*/$^{235}$U")
ax.set_ylabel(r"$^{206}$Pb*/$^{238}$U")

pad_x = 0.15 * data_xmax
pad_y = 0.15 * data_ymax
ax.set_xlim(0, data_xmax + pad_x)
ax.set_ylim(0, data_ymax + pad_y)

legend_handles = [
    plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=class_colors[c], markeredgewidth=1.5, label=c)
    for c in classes
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8.5, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300)
print(f"Saved figure to {OUT_PATH}")
