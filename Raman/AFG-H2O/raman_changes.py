"""Compute and plot Raman intensity changes (Run Product - Starting Material)
for each experiment (Zircon_005, Zircon_006, Zircon_007).

Within each experiment, the Run Product and Starting Material CSVs share the
same row count and are row-aligned, so delta_intensity is computed by row
position against the Run Product's Raman.Shift grid.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASENAME = "AFG-H2O_RamanSpectraChanges"

EXPERIMENTS = {
    "Zircon_005": ("Zircon_005_Run Products.csv", "Zircon_005_Starting Materials.csv"),
    "Zircon_006": ("Zircon_006_Run Product.csv", "Zircon_006_starting material.csv"),
    "Zircon_007": ("Zircon_007_Run Products.csv", "Zircon_007_Starting Materials.csv"),
}

LEGEND_LABELS = {
    "Zircon_005": "Z_005, t = 0 h",
    "Zircon_006": "Z_006, t = 1 h",
    "Zircon_007": "Z_007, t = 8 h",
}

PEAK_PROMINENCE = 200
PEAK_MIN_DISTANCE = 5


def compute_deltas():
    raman_shift = None
    result = pd.DataFrame()
    for name, (rp_file, sm_file) in EXPERIMENTS.items():
        rp = pd.read_csv(os.path.join(FOLDER, rp_file))
        sm = pd.read_csv(os.path.join(FOLDER, sm_file))

        if raman_shift is None:
            raman_shift = rp.iloc[:, 0].reset_index(drop=True)
            result["Raman.Shift"] = raman_shift

        delta = (rp.iloc[:, 1].reset_index(drop=True) - sm.iloc[:, 1].reset_index(drop=True))
        result[name] = delta

    return result


def plot_deltas(df):
    experiment_cols = list(EXPERIMENTS.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in experiment_cols:
        ax.plot(df["Raman.Shift"], df[col], label=LEGEND_LABELS[col], linewidth=0.8)

    x = df["Raman.Shift"].values
    envelope = df[experiment_cols].max(axis=1).values
    peaks, _ = find_peaks(envelope, prominence=PEAK_PROMINENCE, distance=PEAK_MIN_DISTANCE)
    y_top = ax.get_ylim()[1]
    for p in peaks:
        peak_x, peak_y = x[p], envelope[p]
        ax.vlines(peak_x, 0, peak_y, color="black", linewidth=0.6, linestyle="--")
        ax.annotate(
            f"{peak_x:.1f}",
            xy=(peak_x, peak_y),
            xytext=(peak_x, peak_y + 0.015 * y_top),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax.set_ylabel("Delta Intensity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FOLDER, f"{OUTPUT_BASENAME}.png"), dpi=300)


def main():
    df = compute_deltas()
    df.to_csv(os.path.join(FOLDER, f"{OUTPUT_BASENAME}.csv"), index=False)
    plot_deltas(df)


if __name__ == "__main__":
    main()
