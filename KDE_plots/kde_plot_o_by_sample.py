"""
Kernel density estimate (KDE) plots of d18O, one chart per sample (15, 16),
each overlaying its "Replaced" categories (Y, N) from the "Data" tab of
AFG_O_Isotopes.xlsx.

Same method as kde_plot_o.py: each analysis is a Gaussian kernel centered
on its d18O value with standard deviation = its 1-sigma uncertainty; each
category's curve is the average of its analyses' Gaussians, then rescaled
to its own peak = 1 (Y's uncertainties are much larger than N's, so
without this a "Y" curve can be nearly invisible next to "N").

Some "Unaltered core" analyses have uncertainties as small as ~0.02,
producing needle-thin peaks that can sit right at the extreme of a
sample's d18O range; the x-axis uses a symmetric log ("symlog") scale
(linear near zero, logarithmic beyond) so those peaks get more visual
room without stretching the broader, more negative "Replacement zone"
peaks too far apart.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

# ---- Config ---------------------------------------------------------------

XLSX_PATH = r"AFG_O_Isotopes.xlsx"
SHEET_NAME = "Data"
VALUE_COL = "d18O"
UNC_COL = "1s"
CAT_COL = "Replaced"
SAMPLE_COL = "Sample"

SAMPLES = [15, 16]
CATEGORIES = ["Y", "N"]
LABELS = {
    "Y": "Replacement zone",
    "N": "Unaltered core",
}
COLORS = {
    "Y": "#2a78d6",  # categorical slot 1
    "N": "#eb6834",  # categorical slot 2
}
LEGEND_ORDER = ["N", "Y"]

LINTHRESH = 10  # |d18O| below this is linear; beyond it, log-scaled

# ---- Load & clean -----------------------------------------------------------

df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]
df[CAT_COL] = df[CAT_COL].astype(str).str.strip()

df = df.dropna(subset=[VALUE_COL, UNC_COL, CAT_COL, SAMPLE_COL])
df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
df[UNC_COL] = pd.to_numeric(df[UNC_COL], errors="coerce")
df = df.dropna(subset=[VALUE_COL, UNC_COL])


def build_grid(x_min, x_max, linthresh, n_lin=8000, n_log=6000):
    segments = [np.linspace(-linthresh, linthresh, n_lin)]
    if x_max > linthresh:
        segments.append(np.logspace(np.log10(linthresh), np.log10(x_max), n_log))
    if x_min < -linthresh:
        segments.append(-np.logspace(np.log10(linthresh), np.log10(-x_min), n_log))
    return np.unique(np.concatenate(segments))


def symlog_ticks(x_min, x_max, linthresh, mags=(1, 2, 5), max_decade=6):
    """1/2/5-per-decade tick set for a symlog axis, including the linear region."""
    ticks = {0.0}
    for a in mags:  # linear region: -5, -2, -1, 1, 2, 5
        for v in (a, -a):
            if -linthresh <= v <= linthresh and x_min <= v <= x_max:
                ticks.add(v)
    for n in range(1, max_decade + 1):  # log region(s): -50, -20, -10, 10, 20, 50, ...
        added = False
        for a in mags:
            for v in (a * 10**n, -a * 10**n):
                if abs(v) >= linthresh and x_min <= v <= x_max:
                    ticks.add(v)
                    added = True
        if not added:
            break
    return sorted(ticks)


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
    # 6-sigma padding: some uncertainties are as small as ~0.02, so a 3-sigma
    # pad cuts the peak off while it is still visibly tapering.
    raw_min = row_min[VALUE_COL] - 6 * row_min[UNC_COL]
    raw_max = row_max[VALUE_COL] + 6 * row_max[UNC_COL]
    # Extend out to round numbers (with an extra step of margin on the low
    # end) so the axis has room for more labeled ticks, per user request.
    x_min = np.floor(raw_min / 10) * 10 - 10
    x_max = np.ceil(raw_max / 10) * 10
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
    # Default symlog ticks only label whole decades (-10, 0) and skip the
    # linear region entirely; use an explicit 1/2/5-per-decade tick set
    # (plain-number labels, not "-10^1" notation) for more numerical labels.
    ax.xaxis.set_major_locator(FixedLocator(symlog_ticks(x_min, x_max, LINTHRESH)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:g}"))
    ax.axvline(0, color="#c3c2b7", lw=1, ls="--", zorder=0)

    ax.set_ylim(0, 1.15)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"$\delta^{18}$O", color="#0b0b0b", fontsize=14)
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
    out_png = f"KDE_O_Sample{sample}.png"
    out_pdf = f"KDE_O_Sample{sample}.pdf"
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_png} and {out_pdf}")
