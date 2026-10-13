"""Scatter plots of d18O and d29Si profiles vs. distance from cube corner.

Outputs vector PDF and SVG (live text, editable in Adobe Illustrator).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent

# Keep text as real, editable text and embed TrueType (Type 42) fonts in PDF
plt.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "mathtext.default": "regular",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

XLABEL = "Distance from cube corner (µm)"


def make_plot(csv_name, value_col, ylabel, ylim, log=False):
    df = pd.read_csv(HERE / csv_name)
    x = df.iloc[:, 0].to_numpy()
    y = df[value_col].to_numpy()
    se = df["SE(Prop)"].to_numpy()

    if log:
        bad = y <= 0
        if bad.any():
            print(f"{csv_name}: omitting {bad.sum()} non-positive value(s) "
                  f"({y[bad]}) that cannot be shown on a log axis")
            x, y, se = x[~bad], y[~bad], se[~bad]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.errorbar(x, y, yerr=se, fmt="o", ms=2.5, mfc="none", mec="black", mew=0.8,
                ecolor="black", elinewidth=0.8, capsize=2, capthick=0.8)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    stem = HERE / Path(csv_name).stem
    for ext in ("pdf", "svg"):
        fig.savefig(f"{stem}_plot.{ext}", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    per_mil = "‰"
    make_plot("d18O.csv", "d18O", rf"$\delta^{{18}}$O ({per_mil})", (-60, 10))
    make_plot("d29Si.csv", "d29Si", rf"$\delta^{{29}}$Si ({per_mil})", (1e-2, 1e5), log=True)
