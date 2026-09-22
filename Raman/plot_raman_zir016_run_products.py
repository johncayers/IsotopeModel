"""Plot average Raman spectra for the unaltered core vs. replacement rim
(Zircon 016 line scan), together with the Mud Tank reference standard.
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

FOLDER = "Zircon_016_RP_Line_5"
MUD_TANK_FILE = "Mud_Tank_20231201.CSV"

CORE_RANGE = range(1, 38)          # scans 1-37, "Unaltered core"
RIM_RANGE = range(38, 61)          # scans 38-60, "Replacement rim"

X_MIN, X_MAX = 300, 1100
PROMINENCE = 15
SHOULDER_WINDOW = (380, 405)  # weak ~395 cm-1 shoulder
SHOULDER_PROMINENCE = 5


def scan_number(filepath):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    return int(stem[-2:])


def find_series_peaks(x, y):
    mask = (x >= X_MIN) & (x <= X_MAX)
    xs, ys = x[mask], y[mask]

    peaks, _ = find_peaks(ys, prominence=PROMINENCE, distance=3)
    peak_x = list(xs[peaks])
    peak_y = list(ys[peaks])

    if not any(SHOULDER_WINDOW[0] <= px <= SHOULDER_WINDOW[1] for px in peak_x):
        win_mask = (xs >= SHOULDER_WINDOW[0]) & (xs <= SHOULDER_WINDOW[1])
        win_x, win_y = xs[win_mask], ys[win_mask]
        sub_peaks, _ = find_peaks(win_y, prominence=SHOULDER_PROMINENCE, distance=3)
        for p in sub_peaks:
            peak_x.append(win_x[p])
            peak_y.append(win_y[p])

    order = np.argsort(peak_x)
    return np.array(peak_x)[order], np.array(peak_y)[order]


files = sorted(glob.glob(os.path.join(FOLDER, "*.CSV")), key=scan_number)

x_ref = None
core_spectra = []
rim_spectra = []

for filepath in files:
    scan = scan_number(filepath)
    df = pd.read_csv(filepath, header=None, names=["x", "y"])

    if x_ref is None:
        x_ref = df["x"].to_numpy()

    if scan in CORE_RANGE:
        core_spectra.append(df["y"].to_numpy())
    elif scan in RIM_RANGE:
        rim_spectra.append(df["y"].to_numpy())

core_avg = np.mean(core_spectra, axis=0)
rim_avg = np.mean(rim_spectra, axis=0)
n_core = len(core_spectra)
n_rim = len(rim_spectra)

mud_tank_df = pd.read_csv(MUD_TANK_FILE)
mud_tank_x = mud_tank_df.iloc[:, 0].to_numpy()
mud_tank_y = mud_tank_df.iloc[:, 1].to_numpy()

fig, ax = plt.subplots(figsize=(8, 5))
(core_line,) = ax.plot(x_ref, core_avg, label=f"Unaltered core, n = {n_core}", linewidth=0.8)
ax.plot(x_ref, rim_avg, label=f"Replacement rim, n = {n_rim}", linewidth=0.8)
(mud_tank_line,) = ax.plot(mud_tank_x, mud_tank_y, label="Mud Tank", linewidth=0.8)

ax.set_xlim(300, 1100)
ax.set_ylim(bottom=0)
ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
ax.set_xlabel("Raman Shift (cm$^{-1}$)")
ax.set_ylabel("Intensity")

# Vertical reference lines at each peak's Raman shift, from the x-axis to the peak top.
core_peak_x, core_peak_y = find_series_peaks(x_ref, core_avg)
mud_tank_peak_x, mud_tank_peak_y = find_series_peaks(mud_tank_x, mud_tank_y)

ax.vlines(core_peak_x, ymin=0, ymax=core_peak_y, color=core_line.get_color(),
          linestyle="--", linewidth=0.7, alpha=0.6)
ax.vlines(mud_tank_peak_x, ymin=0, ymax=mud_tank_peak_y, color=mud_tank_line.get_color(),
          linestyle="--", linewidth=0.7, alpha=0.6)


def label_peaks(ax, peak_x, peak_y, color):
    y_offset = 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for x, y in zip(peak_x, peak_y):
        ax.annotate(
            f"{x:.0f}",
            xy=(x, y + y_offset),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
            color=color,
        )


label_peaks(ax, core_peak_x, core_peak_y, core_line.get_color())
label_peaks(ax, mud_tank_peak_x, mud_tank_peak_y, mud_tank_line.get_color())

ax.legend()

fig.tight_layout()
fig.savefig("Raman_Spectra_Zir016_Run_Products.png", dpi=300)
plt.show()
