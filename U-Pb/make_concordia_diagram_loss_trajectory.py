"""
U-Pb concordia diagram with a modeled U-Pb loss trajectory from SMMZ to RPHZR.

Physical model
---------------
Each analysis gives ppm U and, now, actual Pb data: ppm 206Pb*, ppm 208Pb*,
and the total (i.e. not radiogenic-only) 207Pb/206Pb ratio. Total ppm Pb for
an analysis is:
    ppm Pb = ppm206Pb* + (Total 207Pb/206Pb)*ppm206Pb* + ppm208Pb*
i.e. 206Pb* + 207Pb (back-calculated from the total 207Pb/206Pb ratio) + 208Pb.

Taking the SMMZ class average as the "parent" composition and the RPHZR class
average as the fully U-Pb-depleted "daughter" composition, two total
fractional losses are solved for:
    f_U  = total fractional loss of U  (from average ppm U)
    f_Pb = total fractional loss of Pb (from average ppm Pb, above)

A hypothetical sample that has experienced fraction s (0-1) of the total
SMMZ -> RPHZR loss has U(s) = U_SMMZ*(1 - s*f_U) and Pb(s) = Pb_SMMZ*(1 - s*f_Pb),
with no isotopic fractionation assumed within either element, so both
concordia ratios scale by the same factor:
    x(s) = x_SMMZ * (1 - s*f_Pb) / (1 - s*f_U)
    y(s) = y_SMMZ * (1 - s*f_Pb) / (1 - s*f_U)

x(0),y(0) = SMMZ average by construction. Because this is a genuine 2-
parameter (f_U, f_Pb) mass-balance model rather than a curve fit forced
through both averages, x(1),y(1) approximates but will not exactly equal the
RPHZR average -- the small offset is the part of the real SMMZ->RPHZR change
that uniform, non-fractionating U and Pb loss alone cannot explain. This
analytic curve (evaluated finely in s) is plotted as the fitted loss
trajectory, and is sampled at s = 0.1, 0.2, ..., 1.0 for the hypothetical
10%-increment samples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path

plt.rcParams.update({"font.size": 13})

# ---------------------------------------------------------------- settings
CSV_PATH = Path(__file__).parent / "U-Pb_DiscordiaDiagram.csv"
OUT_PATH = Path(__file__).parent / "U-Pb_ConcordiaDiagram_loss_trajectory.png"
OUT_CSV = Path(__file__).parent / "U-Pb_LossTrajectory_hypothetical_samples.csv"

LAMBDA235 = 9.8485e-10  # 1/yr
LAMBDA238 = 1.55125e-10  # 1/yr

T_MAX = 3.5e9  # years
AGE_TICKS_GA = np.arange(0, 3.5 + 0.001, 0.5)  # Ga, tick marks along concordia

MAX_ERROR_PCT = 10  # exclude analyses with either %error above this

START_CLASS = "SMMZ"
END_CLASS = "RPHZR"
STEPS = np.arange(0.1, 1.0001, 0.1)  # 10% increments, 10%-100%

# ---------------------------------------------------------------- load data
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
df["Class"] = df["Class"].str.strip()

u_col = "ppm U"
x_col, xerr_col = "207Pb*/235U", "207Pb*/235U ±%"
y_col, yerr_col = "206Pb*/238U", "206Pb*/238U ±%"
rho_col = "err corr"
pb206_col = "ppm 206Pb*"
pb208_col = "ppm 208Pb*"
ratio207_206_col = "Total 207Pb/206Pb"

df = df.dropna(subset=["Class", x_col, y_col])

n_before = len(df)
df = df[(df[xerr_col] <= MAX_ERROR_PCT) & (df[yerr_col] <= MAX_ERROR_PCT)]
print(f"Excluded {n_before - len(df)} of {n_before} analyses with >{MAX_ERROR_PCT}% error")

# total ppm Pb = 206Pb* + 207Pb (back-calculated from the total 207Pb/206Pb
# ratio and ppm 206Pb*) + 208Pb*
df["ppm_Pb_total"] = (
    df[pb206_col] + df[ratio207_206_col] * df[pb206_col] + df[pb208_col]
)

# ---------------------------------------------------------------- SMMZ -> RPHZR loss model
start = df[df["Class"] == START_CLASS]
end = df[df["Class"] == END_CLASS]

U_start, x_start, y_start = start[u_col].mean(), start[x_col].mean(), start[y_col].mean()
U_end, x_end, y_end = end[u_col].mean(), end[x_col].mean(), end[y_col].mean()
Pb_start, Pb_end = start["ppm_Pb_total"].mean(), end["ppm_Pb_total"].mean()

print(f"\n{START_CLASS} average (n={len(start)}): U={U_start:.2f} ppm, Pb={Pb_start:.2f} ppm, "
      f"x={x_start:.5f}, y={y_start:.5f}")
print(f"{END_CLASS} average (n={len(end)}): U={U_end:.2f} ppm, Pb={Pb_end:.2f} ppm, "
      f"x={x_end:.5f}, y={y_end:.5f}")

f_U = 1 - U_end / U_start
f_Pb = 1 - Pb_end / Pb_start

print(f"\nAverage % U loss, {START_CLASS} -> {END_CLASS}:  {100*f_U:.1f}%")
print(f"Average % Pb loss, {START_CLASS} -> {END_CLASS}: {100*f_Pb:.1f}%")
print("(Pb = ppm 206Pb* + Total-207Pb/206Pb * ppm 206Pb* + ppm 208Pb*)")


def traj_xy(s):
    """Concordia coordinates of the hypothetical sample at loss fraction s."""
    x = x_start * (1 - s * f_Pb) / (1 - s * f_U)
    y = y_start * (1 - s * f_Pb) / (1 - s * f_U)
    return x, y


s_fit = np.linspace(0, 1, 400)
x_fit, y_fit = traj_xy(s_fit)

x_steps, y_steps = traj_xy(STEPS)

hyp_df = pd.DataFrame({
    "loss_fraction_of_total": STEPS,
    "pct_of_total_SMMZ_to_RPHZR_loss": STEPS * 100,
    "x_207Pb235U": x_steps,
    "y_206Pb238U": y_steps,
})
hyp_df.to_csv(OUT_CSV, index=False)
print(f"\nHypothetical sample coordinates saved to {OUT_CSV}")
print(hyp_df.to_string(index=False))

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


def plot_concordia(ax, tick_fontsize=14, tick_ms=8, label_offsets=None, show_label=True):
    label_offsets = label_offsets or {}
    ax.plot(x_conc, y_conc, color="black", lw=1.5, zorder=3,
            label="Concordia" if show_label else None)
    ax.plot(
        x_ticks, y_ticks, "o", markerfacecolor="black", markeredgecolor="black",
        ms=tick_ms, mew=1.3, zorder=4,
    )
    for tga, xt, yt in zip(AGE_TICKS_GA, x_ticks, y_ticks):
        if tga == 0:
            continue
        xytext = label_offsets.get(tga, (0, 10))
        ax.annotate(
            f"{tga:g}", (xt, yt), textcoords="offset points", xytext=xytext,
            ha="center", fontsize=tick_fontsize, color="black",
        )


def plot_ellipses(ax, alpha=0.55, lw=1.2):
    for cls in classes:
        sub = df[df["Class"] == cls]
        color = class_colors[cls]
        for _, row in sub.iterrows():
            w, h, ang = error_ellipse_params(
                row[x_col], row[y_col], row[xerr_col], row[yerr_col], row[rho_col]
            )
            ell = Ellipse(
                (row[x_col], row[y_col]), width=w, height=h, angle=ang,
                facecolor="none", edgecolor=color, linewidth=lw, alpha=alpha, zorder=5,
            )
            ax.add_patch(ell)


def draw_ticks(ax, xs, ys, half_length_px=7, **kwargs):
    """Draw short tick marks perpendicular to the local curve direction at
    each (x, y), computed in display (pixel) space so they look perpendicular
    regardless of the axes' data aspect ratio. Requires ax's data limits to
    already be final (transData must not change after this is called)."""
    xy = np.column_stack([xs, ys])
    disp = ax.transData.transform(xy)
    tangent = np.gradient(disp, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    norm[norm == 0] = 1
    tangent_unit = tangent / norm
    perp = np.column_stack([-tangent_unit[:, 1], tangent_unit[:, 0]])
    p1 = ax.transData.inverted().transform(disp - perp * half_length_px)
    p2 = ax.transData.inverted().transform(disp + perp * half_length_px)
    for (x1, y1), (x2, y2) in zip(p1, p2):
        ax.plot([x1, x2], [y1, y2], **kwargs)


# ---------------------------------------------------------------- figure (main, no trajectory)
fig, ax = plt.subplots(figsize=(9, 7))

data_xmax = df[x_col].max()
data_ymax = df[y_col].max()

plot_concordia(ax, label_offsets={0.5: (-28, 12)})
plot_ellipses(ax)

ax.set_xlabel(r"$^{207}$Pb*/$^{235}$U", fontsize=16)
ax.set_ylabel(r"$^{206}$Pb*/$^{238}$U", fontsize=16)
ax.tick_params(labelsize=13)

pad_x = 0.15 * data_xmax
pad_y = 0.15 * data_ymax
ax.set_xlim(0, data_xmax + pad_x)
ax.set_ylim(0, data_ymax + pad_y)

class_handles = [
    plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=class_colors[c], markeredgewidth=1.5, ms=10, label=c)
    for c in classes
]
main_handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=class_handles + main_handles, loc="upper left", fontsize=11.5, framealpha=0.9)

# lock in the main axes' layout now, before the inset is added, so the
# perpendicular tick-mark geometry computed below (via transData) stays
# valid at save time -- a later tight_layout() could shift ax's bbox and
# throw off the ticks
fig.tight_layout()

# ---------------------------------------------------------------- inset: loss trajectory, zoomed
x_range = x_start - x_end
y_range = y_start - y_end
pad_frac = 0.28
inset_xlim = (x_end - pad_frac * x_range, x_start + pad_frac * x_range)
inset_ylim = (y_end - pad_frac * y_range, y_start + pad_frac * y_range)

ax_inset = ax.inset_axes([0.54, 0.05, 0.44, 0.42])
ax_inset.set_xlim(*inset_xlim)
ax_inset.set_ylim(*inset_ylim)
ax_inset.set_autoscale_on(False)

plot_concordia(ax_inset, tick_fontsize=11, tick_ms=6, show_label=False)
plot_ellipses(ax_inset, alpha=0.7, lw=1.3)

ax_inset.plot(x_fit, y_fit, "-", color="purple", lw=2, zorder=7,
              label=f"{START_CLASS}→{END_CLASS} U-Pb loss trajectory (model fit)")
draw_ticks(ax_inset, x_steps, y_steps, half_length_px=7, color="purple", lw=1.8, zorder=8)

# the s=0.1-0.4 points sit close together (the trajectory is nonlinear and
# compresses near the SMMZ end), so only every other point is text-labeled
# (all ten still get a tick mark) and labels alternate below/above the line,
# clear of both the SMMZ diamond and the SMMZ/RPRZC ellipses -- offsets are
# perpendicular to the trajectory line (not along it), so neighboring labels
# move apart rather than converging on each other
LABELED_STEPS = {0.1, 0.3, 0.5, 0.7, 0.9}
STEP_LABEL_OFFSETS = {0.1: (9, -13), 0.3: (-9, 13), 0.5: (9, -13),
                       0.7: (-9, 13), 0.9: (9, -13)}
for s, xs, ys in zip(STEPS, x_steps, y_steps):
    s = round(s, 1)  # guard against float accumulation in np.arange (e.g. 0.30000000000000004)
    if s not in LABELED_STEPS:
        continue
    dx, dy = STEP_LABEL_OFFSETS[s]
    ha = "left" if dx > 0 else "right"
    ax_inset.annotate(f"{s*100:.0f}%", (xs, ys), textcoords="offset points",
                       xytext=(dx, dy), ha=ha, va="center", fontsize=10.5,
                       color="purple", zorder=10)

ax_inset.plot(x_start, y_start, "D", markerfacecolor="none", markeredgecolor="blue",
              ms=13, mew=2.2, zorder=9, label=f"{START_CLASS} average (0% loss)")
ax_inset.plot(x_end, y_end, "D", markerfacecolor="none", markeredgecolor="red",
              ms=13, mew=2.2, zorder=9, label=f"{END_CLASS} average (100% loss)")

ax_inset.tick_params(labelsize=10.5)
ax_inset.set_facecolor("white")

inset_handles, _ = ax_inset.get_legend_handles_labels()
ax_inset.legend(handles=inset_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                 fontsize=9.5, framealpha=0.9)

ax.indicate_inset_zoom(ax_inset, edgecolor="gray")

fig.savefig(OUT_PATH, dpi=300)
print(f"\nSaved figure to {OUT_PATH}")
