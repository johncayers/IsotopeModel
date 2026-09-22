"""
Kernel density estimate (KDE) plots of d29Si, one chart per sample (15, 16),
each overlaying its "Replaced" categories (Y, P, N) from the "Data" tab of
AFG_Si_Isotopes.xlsx.

Same method as kde_plot_si.py: each analysis is a Gaussian kernel centered
on its d29Si value with standard deviation = its 1-sigma uncertainty; each
category's curve is the average of its analyses' Gaussians, then rescaled
to its own peak = 1 (categories differ by orders of magnitude in
uncertainty, so without this a "Y" curve can be invisible next to "N").
The x-axis is symlog since d29Si spans several orders of magnitude within
a sample.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config ---------------------------------------------------------------

XLSX_PATH = r"AFG_Si_Isotopes.xlsx"
SHEET_NAME = "Data"
VALUE_COL = "d29Si"
UNC_COL = "1s"
CAT_COL = "Replaced"
SAMPLE_COL = "Sample"

SAMPLES = [15, 16]
CATEGORIES = ["Y", "P", "N"]
LABELS = {
    "Y": "Replacement zone",
    "P": "Mixed",
    "N": "Unaltered core",
}
COLORS = {
    "Y": "#2a78d6",  # categorical slot 1
    "P": "#eb6834",  # categorical slot 2
    "N": "#1baf7a",  # categorical slot 3
}
LEGEND_ORDER = ["N", "P", "Y"]

LINTHRESH = 10  # |d29Si| below this is linear; beyond it, log-scaled

# ---- Load & clean -----------------------------------------------------------

df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]
df[CAT_COL] = df[CAT_COL].astype(str).str.strip()

df = df.dropna(subset=[VALUE_COL, UNC_COL, CAT_COL, SAMPLE_COL])
df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
df[UNC_COL] = pd.to_numeric(df[UNC_COL], errors="coerce")
df = df.dropna(subset=[VALUE_COL, UNC_COL])


def build_grid(x_min, x_max, linthresh, n_lin=4000, n_log=3000):
    segments = [np.linspace(-linthresh, linthresh, n_lin)]
    if x_max > linthresh:
        segments.append(np.logspace(np.log10(linthresh), np.log10(x_max), n_log))
    if x_min < -linthresh:
        segments.append(-np.logspace(np.log10(linthresh), np.log10(-x_min), n_log))
    return np.unique(np.concatenate(segments))


def kde_curve(values, uncertainties, x):
    values = values.to_numpy()[:, None]
    sigmas = uncertainties.to_numpy()[:, None]
    gaussians = np.exp(-0.5 * ((x[None, :] - values) / sigmas) ** 2) / (
        sigmas * np.sqrt(2 * np.pi)
    )
    return gaussians.mean(axis=0)


for sample in SAMPLES:
    sdf = df[df[SAMPLE_COL] == sample]
    present = [c for c in CATEGORIES if (sdf[CAT_COL] == c).any()]
    subsets = {c: sdf[sdf[CAT_COL] == c] for c in present}

    row_min = sdf.loc[sdf[VALUE_COL].idxmin()]
    row_max = sdf.loc[sdf[VALUE_COL].idxmax()]
    # 6-sigma padding so a narrow, small-uncertainty peak at either extreme
    # tapers fully before the axis edge instead of looking cut off.
    x_min = row_min[VALUE_COL] - 6 * row_min[UNC_COL]
    x_max = row_max[VALUE_COL] + 6 * row_max[UNC_COL]
    x = build_grid(x_min, x_max, LINTHRESH)

    curves = {c: kde_curve(sub[VALUE_COL], sub[UNC_COL], x) for c, sub in subsets.items()}
    curves = {c: curve / curve.max() for c, curve in curves.items()}

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    rug_y0 = -0.06
    for c in present:
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

    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.axvline(0, color="#c3c2b7", lw=1, ls="--", zorder=0)

    ax.set_ylim(0, 1.15)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"$\delta^{29}$Si", color="#0b0b0b", fontsize=14)
    ax.set_ylabel("Relative probability", color="#0b0b0b", fontsize=14)
    ax.text(
        0.02, 0.94, f"Sample {sample}",
        transform=ax.transAxes, ha="left", va="top",
        color="#0b0b0b", fontsize=14,
    )

    ax.set_yticks([])
    ax.tick_params(colors="#52514e", labelsize=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.spines["left"].set_color("#c3c2b7")
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

    order = [c for c in LEGEND_ORDER if c in present]
    handles, labels = ax.get_legend_handles_labels()
    by_cat = dict(zip(present, zip(handles, labels)))
    handles, labels = zip(*(by_cat[c] for c in order))
    legend = ax.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(order),
        fontsize=13,
    )
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    fig.tight_layout()
    out_png = f"KDE_Si_Sample{sample}.png"
    out_pdf = f"KDE_Si_Sample{sample}.pdf"
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_png} and {out_pdf}")
