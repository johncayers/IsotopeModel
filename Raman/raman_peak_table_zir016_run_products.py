"""Tabulate Raman peaks for the metamict zone, replacement zone, and Mud Tank
reference spectra shown in the Zir016 run-products plot, and compare each
peak's position and intensity to the corresponding Mud Tank reference peak.

For each peak, the FWHM is obtained by fitting a Lorentzian (plus linear
baseline) to a local window and then correcting the fitted (measured) FWHM
for the instrument's apparatus function following Irmer (1985), Eq. (2) of
Nasdala et al. (2001, Contrib Mineral Petrol 141:125-144):

    beta = beta_s * sqrt(1 - 2*(s/beta_s)**2)

where beta_s is the measured FWHM and s is the spectral resolution of the
Raman system (here s = 2 cm-1, typical for a Thermo Scientific DXR confocal
Raman microscope with the high-resolution grating).

The alpha-dose is then estimated from the corrected FWHM of the nu3(SiO4)
band (~1000 cm-1) using Eq. (6) of Nasdala et al. (2001):

    FWHM [cm-1] = 1.2 + 140 * D_alpha [1e16 alpha/mg]
    => D_alpha [1e16 alpha/mg] = (FWHM - 1.2) / 140
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

FOLDER = "Zircon_016_RP_Line_5"
MUD_TANK_FILE = "Mud_Tank_20231201.CSV"

METAMICT_RANGE = range(1, 37)      # scans 1-37
REPLACEMENT_RANGE = range(39, 61)  # scans 38-60 - I changed index so no overlap

X_MIN, X_MAX = 300, 1100
PROMINENCE = 15          # main peak-detection threshold
SHOULDER_WINDOW = (380, 405)  # weak ~395 cm-1 shoulder
SHOULDER_PROMINENCE = 5
MATCH_TOLERANCE = 10     # cm-1, for matching a peak to a Mud Tank reference peak

SPECTRAL_RESOLUTION = 2.0   # s, cm-1 (Thermo Scientific DXR, high-resolution grating)
NU3_BAND_WINDOW = (995, 1020)  # window used to identify the nu3(SiO4) band
MAX_FIT_HALF_WIDTH = 25    # cm-1
MIN_FIT_HALF_WIDTH = 6     # cm-1


def scan_number(filepath):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    return int(stem[-2:])


def lorentzian(x, x0, gamma, amp, m, c):
    return m * x + c + amp / (1.0 + ((x - x0) / gamma) ** 2)


def fit_measured_fwhm(x, y, x0, half_width):
    """Fit a Lorentzian + linear baseline in a local window; return measured FWHM."""
    mask = (x >= x0 - half_width) & (x <= x0 + half_width)
    xw, yw = x[mask], y[mask]
    if len(xw) < 6:
        return np.nan

    baseline_guess = 0.5 * (yw[0] + yw[-1])
    peak_idx = np.argmin(np.abs(xw - x0))
    amp_guess = yw[peak_idx] - baseline_guess
    p0 = [x0, half_width / 3, amp_guess, 0.0, baseline_guess]

    x_spacing = np.median(np.diff(xw))
    bounds = (
        [x0 - half_width, x_spacing, -np.inf, -np.inf, -np.inf],
        [x0 + half_width, 3 * half_width, np.inf, np.inf, np.inf],
    )

    try:
        popt, _ = curve_fit(lorentzian, xw, yw, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError:
        return np.nan

    gamma = popt[1]
    return 2.0 * gamma  # FWHM of a Lorentzian is 2*gamma


def irmer_corrected_fwhm(beta_s, s=SPECTRAL_RESOLUTION):
    """Eq. (2) of Nasdala et al. (2001), after Irmer (1985)."""
    if np.isnan(beta_s) or beta_s <= 0:
        return np.nan
    inside = 1.0 - 2.0 * (s / beta_s) ** 2
    if inside < 0:
        return np.nan  # bs < s*sqrt(2): correction is mathematically undefined
    return beta_s * np.sqrt(inside)


def find_series_peaks(x, y):
    mask = (x >= X_MIN) & (x <= X_MAX)
    xs, ys = x[mask], y[mask]

    peaks, _ = find_peaks(ys, prominence=PROMINENCE, distance=3)
    peak_x = list(xs[peaks])
    peak_y = list(ys[peaks])

    # Recover the weak shoulder near ~395 cm-1 if the main pass missed it.
    if not any(SHOULDER_WINDOW[0] <= px <= SHOULDER_WINDOW[1] for px in peak_x):
        win_mask = (xs >= SHOULDER_WINDOW[0]) & (xs <= SHOULDER_WINDOW[1])
        win_x, win_y = xs[win_mask], ys[win_mask]
        sub_peaks, _ = find_peaks(win_y, prominence=SHOULDER_PROMINENCE, distance=3)
        for p in sub_peaks:
            peak_x.append(win_x[p])
            peak_y.append(win_y[p])

    order = np.argsort(peak_x)
    peak_x = np.array(peak_x)[order]
    peak_y = np.array(peak_y)[order]
    return peak_x, peak_y


# --- load Zircon_016 line-scan CSVs, split into metamict / replacement zones ---
files = sorted(glob.glob(os.path.join(FOLDER, "*.CSV")), key=scan_number)

x_ref = None
metamict_spectra = []
replacement_spectra = []

for filepath in files:
    scan = scan_number(filepath)
    df = pd.read_csv(filepath, header=None, names=["x", "y"])
    if x_ref is None:
        x_ref = df["x"].to_numpy()
    if scan in METAMICT_RANGE:
        metamict_spectra.append(df["y"].to_numpy())
    elif scan in REPLACEMENT_RANGE:
        replacement_spectra.append(df["y"].to_numpy())

metamict_avg = np.mean(metamict_spectra, axis=0)
replacement_avg = np.mean(replacement_spectra, axis=0)

mud_tank_df = pd.read_csv(MUD_TANK_FILE)
mud_tank_x = mud_tank_df.iloc[:, 0].to_numpy()
mud_tank_y = mud_tank_df.iloc[:, 1].to_numpy()

series = {
    "metamict zone": (x_ref, metamict_avg),
    "replacement zone": (x_ref, replacement_avg),
    "Mud Tank": (mud_tank_x, mud_tank_y),
}

peaks = {name: find_series_peaks(x, y) for name, (x, y) in series.items()}
ref_x, ref_y = peaks["Mud Tank"]

rows = []
for name in ["metamict zone", "replacement zone", "Mud Tank"]:
    x, y = series[name]
    px, py = peaks[name]

    for i, (shift, intensity) in enumerate(zip(px, py)):
        # Size the local fit window from the gap to the nearest neighboring peak
        # so overlapping bands don't contaminate each other's fit.
        neighbor_gaps = [abs(shift - other) for j, other in enumerate(px) if j != i]
        half_width = min(neighbor_gaps) / 2.0 if neighbor_gaps else MAX_FIT_HALF_WIDTH
        half_width = float(np.clip(half_width, MIN_FIT_HALF_WIDTH, MAX_FIT_HALF_WIDTH))

        beta_s = fit_measured_fwhm(x, y, shift, half_width)
        beta = irmer_corrected_fwhm(beta_s)
        # Irmer (1985)/Nasdala et al. (2001) note that the correction is only
        # reliable if the corrected FWHM b >= 2*s.
        reliable = (not np.isnan(beta)) and (beta >= 2 * SPECTRAL_RESOLUTION)

        is_nu3_band = NU3_BAND_WINDOW[0] <= shift <= NU3_BAND_WINDOW[1]
        if is_nu3_band and not np.isnan(beta):
            alpha_dose = (beta - 1.2) / 140.0  # 1e16 alpha/mg
        else:
            alpha_dose = np.nan

        if name == "Mud Tank":
            d_shift, d_intensity = 0.0, 0.0
        else:
            diffs = np.abs(ref_x - shift)
            j = np.argmin(diffs)
            if diffs[j] <= MATCH_TOLERANCE:
                d_shift = shift - ref_x[j]
                d_intensity = intensity - ref_y[j]
            else:
                d_shift, d_intensity = np.nan, np.nan

        rows.append(
            {
                "Series": name,
                "Raman shift (cm-1)": round(float(shift), 2),
                "Intensity": round(float(intensity), 2),
                "Delta Raman shift vs Mud Tank (cm-1)": round(float(d_shift), 2) if not np.isnan(d_shift) else "N/A",
                "Delta Intensity vs Mud Tank": round(float(d_intensity), 2) if not np.isnan(d_intensity) else "N/A",
                "FWHM measured (cm-1)": round(float(beta_s), 2) if not np.isnan(beta_s) else "N/A",
                "FWHM, Irmer 1985 corrected (cm-1)": round(float(beta), 2) if not np.isnan(beta) else "undefined (bs<s*sqrt2)",
                "Reliable (b>=2s)": ("yes" if reliable else "no") if not np.isnan(beta) else "N/A",
                "Alpha dose, nu3(SiO4) band, Eq.6 (1e16 alpha/mg)": round(float(alpha_dose), 3) if not np.isnan(alpha_dose) else "N/A",
            }
        )

table = pd.DataFrame(rows)
table.to_csv("Raman_Peak_Table_Zir016_Run_Products.csv", index=False)
print(table.to_string(index=False))
