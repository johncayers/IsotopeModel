"""Combine per-sample Raman CSV files into one spectrum table and plot it.

Each input CSV has two columns: Raman.Shift and one sample's intensity
(sample name in the intensity column header). All files share the same
Raman.Shift grid (row-for-row), so intensities are combined by row
position onto a single Raman.Shift column.
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASENAME = "AFG-H2O_RamanSpectra"
STARTING_MATERIAL_COLUMNS = [
    "Starting Materials (005)",
    "Starting Material (006)",
    "Starting Materials (007)",
]
AVG_COLUMN_NAME = "Avg. Starting Material"
PLOT_ORDER = [
    AVG_COLUMN_NAME,
    "Run Product (0 Hr)",
    "Run Product (1 Hr)",
    "Run Product (8 Hr)",
    "Mud Tank",
]
REFERENCE_LINE_SERIES = "Mud Tank"
PEAK_PROMINENCE = 200
PEAK_MIN_DISTANCE = 5


def combine_files():
    csv_files = sorted(glob.glob(os.path.join(FOLDER, "*.csv")))

    raman_shift = None
    combined = {}
    for path in csv_files:
        df = pd.read_csv(path)
        shift_col, intensity_col = df.columns[0], df.columns[1]

        if raman_shift is None:
            raman_shift = df[shift_col].reset_index(drop=True)

        combined[intensity_col] = df[intensity_col].reset_index(drop=True)

    result = pd.DataFrame({"Raman.Shift": raman_shift})
    for name, series in combined.items():
        result[name] = series

    present_sm_cols = [c for c in STARTING_MATERIAL_COLUMNS if c in result.columns]
    insert_at = result.columns.get_loc(present_sm_cols[0])
    result[AVG_COLUMN_NAME] = result[present_sm_cols].mean(axis=1)
    result = result.drop(columns=present_sm_cols)

    cols = list(result.columns)
    cols.remove(AVG_COLUMN_NAME)
    cols.insert(insert_at, AVG_COLUMN_NAME)
    result = result[cols]

    return result


def plot_spectra(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in PLOT_ORDER:
        ax.plot(df["Raman.Shift"], df[col], label=col, linewidth=0.8)

    x = df["Raman.Shift"].values
    y = df[REFERENCE_LINE_SERIES].values
    peaks, _ = find_peaks(y, prominence=PEAK_PROMINENCE, distance=PEAK_MIN_DISTANCE)
    y_top = ax.get_ylim()[1]
    for p in peaks:
        peak_x, peak_y = x[p], y[p]
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
    ax.set_ylabel("Intensity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FOLDER, f"{OUTPUT_BASENAME}.png"), dpi=300)


def main():
    df = combine_files()
    df.to_csv(os.path.join(FOLDER, f"{OUTPUT_BASENAME}.csv"), index=False)
    plot_spectra(df)


if __name__ == "__main__":
    main()
