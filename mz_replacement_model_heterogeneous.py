"""
Mass-balance model of metamict zircon (MZ) replacement by crystalline zircon (CZ)
during an isotope-doping experiment (O and Si tracer exchange).

MODEL OVERVIEW
---------------
The experiment reacts a metamict zircon crystal (a cube) with an isotopically
doped aqueous fluid (H2O + a 29Si-enriched tracer). As the reaction proceeds,
MZ is progressively dissolved and replaced, mass-for-mass, by newly
precipitated crystalline zircon (CZ). Dissolved MZ releases oxygen and
silicon (with MZ's isotopic signature) into the fluid; new CZ precipitates
using the fluid's *current* isotopic composition (no fractionation, no back
reaction). Once precipitated, a CZ increment is isolated from the fluid and
never re-equilibrates ("physically segregated").

Two isotope systems are tracked independently, each as a well-mixed,
fixed-size fluid reservoir that receives isotopic input from dissolving MZ
and passes its instantaneous composition to precipitating CZ:

  Oxygen reservoir  = H2O mass + the Si-tracer's own oxygen mass
                      (m0_H2O + m0_29Si), composition = mass-weighted mix of
                      delta18O_H2O and delta18O_Si.
  Silicon reservoir = the 29Si-tracer mass (m0_29Si) only, since water
                      contributes no structural Si. Composition =
                      delta29Si_Si. This matches the stated initial condition
                      delta29Si_FL = delta29Si_Si at eps = 0.

Because MZ mass added to the fluid each step is balanced by an equal mass
removed into precipitating CZ, the two fluid reservoir masses stay constant
through the run; only their isotopic composition evolves (a Rayleigh-type
mixing/washout toward the MZ composition as more MZ dissolves).

The experimentally provided delta18O_CZ (-35.70 permil) is a *measured
reference* value (not an input used by the mass-balance equations, since the
model's CZ composition is entirely determined by fluid history under the
"no fractionation" assumption). It is reported alongside the model output
for comparison only.

Geometry uses a shrinking-cube ("shrinking core") model: MZ is replaced
uniformly inward from all six faces, so the remaining MZ core is a smaller
cube whose edge length scales with (remaining MZ volume)^(1/3). The
reaction-front distance from the original edge is half the total edge
shrinkage (the front advances from both sides of each face pair).

All simplifying assumptions requested in the task are honored:
heterogeneous replacement, no re-equilibration, no isotopic fractionation,
perfect mass conservation, stoichiometric (mass-for-mass) replacement, and a
perfectly mixed fluid at all times.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

# Oxygen isotope compositions (permil, delta notation)
DELTA18O_H2O = -50.55   # fluid water
DELTA18O_SI = -19.25    # oxygen carried by the Si-tracer reagent
DELTA18O_CZ_REFERENCE = -35.70  # measured bulk CZ (reference only, not a model input)
DELTA18O_MZ = 0.45      # metamict zircon (unreacted)

# Silicon isotope compositions (permil, delta29Si notation)
DELTA29SI_SI = 1.95e6   # fluid / Si-tracer reagent (== initial delta29Si_FL == delta29Si_CZ)
DELTA29SI_MZ = 1.18     # metamict zircon (unreacted)

# Initial masses (mg)
M0_MZ = 35.0
M0_CZ = 0.0
M0_H2O = 120.0
M0_29SI = 1.7

# Geometry
DENSITY_MZ = 4.0  # g/cm^3

# Reaction progress step size (Delta eps), as specified in the task
DEPS = 0.01

CM_TO_UM = 1.0e4
MG_TO_G = 1.0e-3

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def cube_geometry(mass_mg: float, density_g_cm3: float) -> dict:
    """Volume and edge length of a cube of given mass and density."""
    mass_g = mass_mg * MG_TO_G
    volume_cm3 = mass_g / density_g_cm3
    edge_cm = volume_cm3 ** (1.0 / 3.0)
    return {
        "volume_cm3": volume_cm3,
        "edge_cm": edge_cm,
        "edge_um": edge_cm * CM_TO_UM,
    }


def reaction_front_distance_um(edge0_cm: float, remaining_mz_fraction: float) -> float:
    """
    Distance (um) the reaction front has advanced inward from the original
    cube edge, for a cube shrinking uniformly inward from all six faces.

    remaining_mz_fraction = mMZ(t) / m0_MZ  (equals remaining volume
    fraction, since MZ density is constant).
    """
    remaining_edge_cm = edge0_cm * (remaining_mz_fraction ** (1.0 / 3.0))
    distance_cm = (edge0_cm - remaining_edge_cm) / 2.0
    return distance_cm * CM_TO_UM


# ---------------------------------------------------------------------------
# Isotope mixing
# ---------------------------------------------------------------------------

def mass_weighted_mix(comp_a: float, mass_a: float, comp_b: float, mass_b: float) -> float:
    """Mass-weighted average isotopic composition of two combined reservoirs."""
    return (comp_a * mass_a + comp_b * mass_b) / (mass_a + mass_b)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(deps: float = DEPS) -> pd.DataFrame:
    """
    Step reaction progress eps = mCZ / (mCZ + mMZ) from 0 to 1.

    At each step:
      1. A mass increment dm of MZ is dissolved and removed from MZ.
      2. dm is added to the fluid, mixing MZ's isotopic signature into the
         (fixed-size) fluid reservoirs -> updates delta18O_FL, delta29Si_FL.
      3. A CZ increment of mass dm precipitates, inheriting the fluid's
         current composition exactly (no fractionation). This increment is
         then isolated from the fluid (removed from the exchangeable pool),
         which is mass-for-mass replenished by the MZ input in step 2, so
         reservoir masses stay constant while composition evolves.
      4. Masses of MZ, CZ, and the geometry of the shrinking MZ core are
         updated, and mass conservation is verified.
    """
    n_steps = int(round(1.0 / deps))
    eps_grid = np.linspace(0.0, 1.0, n_steps + 1)

    geom0 = cube_geometry(M0_MZ, DENSITY_MZ)
    edge0_cm = geom0["edge_cm"]

    # Fixed-size, well-mixed fluid reservoirs (oxygen pool includes the
    # Si-tracer's own oxygen; silicon pool is the Si-tracer alone).
    mass_O_reservoir = M0_H2O + M0_29SI
    mass_Si_reservoir = M0_29SI

    delta18O_FL = mass_weighted_mix(DELTA18O_H2O, M0_H2O, DELTA18O_SI, M0_29SI)
    delta29Si_FL = DELTA29SI_SI  # matches given initial condition delta29Si_FL = delta29Si_Si

    total_system_mass = M0_MZ + M0_H2O + M0_29SI

    records = []

    # eps = 0 starting state (no CZ has precipitated yet)
    records.append({
        "eps": 0.0,
        "mCZ_mg": M0_CZ,
        "mMZ_mg": M0_MZ,
        "mass_increment_mg": 0.0,
        "delta18O_FL": delta18O_FL,
        "delta18O_CZ_bulk": np.nan,
        "delta18O_CZ_increment": np.nan,
        "delta29Si_FL": delta29Si_FL,
        "delta29Si_CZ_bulk": np.nan,
        "delta29Si_CZ_increment": np.nan,
        "remaining_MZ_edge_um": geom0["edge_um"],
        "distance_from_edge_um": 0.0,
        "mass_balance_error_mg": 0.0,
    })

    mCZ_prev = 0.0
    sum_mass_CZ = 0.0
    sum_d18O_CZ = 0.0
    sum_d29Si_CZ = 0.0

    for eps in eps_grid[1:]:
        mCZ_total = M0_MZ * eps
        mMZ_total = M0_MZ * (1.0 - eps)
        dm = mCZ_total - mCZ_prev  # mass increment converted this step

        # --- MZ dissolution mixes its isotopic signature into the fluid ---
        delta18O_FL = mass_weighted_mix(delta18O_FL, mass_O_reservoir, DELTA18O_MZ, dm)
        delta29Si_FL = mass_weighted_mix(delta29Si_FL, mass_Si_reservoir, DELTA29SI_MZ, dm)

        # --- CZ precipitates with the fluid's current composition (no fractionation) ---
        delta18O_CZ_increment = delta18O_FL
        delta29Si_CZ_increment = delta29Si_FL

        # --- bulk CZ running (mass-weighted) average ---
        sum_mass_CZ += dm
        sum_d18O_CZ += dm * delta18O_CZ_increment
        sum_d29Si_CZ += dm * delta29Si_CZ_increment
        delta18O_CZ_bulk = sum_d18O_CZ / sum_mass_CZ
        delta29Si_CZ_bulk = sum_d29Si_CZ / sum_mass_CZ

        # --- geometry: shrinking MZ core / reaction front position ---
        remaining_mz_fraction = 1.0 - eps
        distance_um = reaction_front_distance_um(edge0_cm, remaining_mz_fraction)
        remaining_edge_um = edge0_cm * (remaining_mz_fraction ** (1.0 / 3.0)) * CM_TO_UM

        # --- mass conservation check (MZ + CZ must equal the original MZ mass) ---
        mass_balance_error = (mCZ_total + mMZ_total) - M0_MZ

        records.append({
            "eps": eps,
            "mCZ_mg": mCZ_total,
            "mMZ_mg": mMZ_total,
            "mass_increment_mg": dm,
            "delta18O_FL": delta18O_FL,
            "delta18O_CZ_bulk": delta18O_CZ_bulk,
            "delta18O_CZ_increment": delta18O_CZ_increment,
            "delta29Si_FL": delta29Si_FL,
            "delta29Si_CZ_bulk": delta29Si_CZ_bulk,
            "delta29Si_CZ_increment": delta29Si_CZ_increment,
            "remaining_MZ_edge_um": remaining_edge_um,
            "distance_from_edge_um": distance_um,
            "mass_balance_error_mg": mass_balance_error,
        })

        mCZ_prev = mCZ_total

    df = pd.DataFrame.from_records(records)

    # Whole-system (MZ + CZ + fluid) mass conservation, constant by construction.
    # Fluid physical mass is H2O + Si-tracer (the two isotope reservoirs overlap
    # in the Si-tracer, so they are not simply summed here).
    fluid_mass_mg = M0_H2O + M0_29SI
    df["total_system_mass_mg"] = df["mCZ_mg"] + df["mMZ_mg"] + fluid_mass_mg

    df.attrs["total_system_mass_initial_mg"] = total_system_mass
    df.attrs["edge0_cm"] = edge0_cm
    df.attrs["edge0_um"] = geom0["edge_um"]
    df.attrs["volume0_cm3"] = geom0["volume_cm3"]
    df.attrs["mass_O_reservoir_mg"] = mass_O_reservoir
    df.attrs["mass_Si_reservoir_mg"] = mass_Si_reservoir

    return df


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csvs(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    oxygen_vs_eps = df[["eps", "delta18O_FL", "delta18O_CZ_bulk"]].rename(columns={
        "eps": "reaction_progress_eps",
        "delta18O_FL": "delta18O_fluid_permil",
        "delta18O_CZ_bulk": "delta18O_bulk_CZ_permil",
    })
    oxygen_vs_eps.to_csv(os.path.join(output_dir, "oxygen_vs_eps.csv"), index=False)

    silicon_vs_eps = df[["eps", "delta29Si_FL", "delta29Si_CZ_bulk"]].rename(columns={
        "eps": "reaction_progress_eps",
        "delta29Si_FL": "delta29Si_fluid_permil",
        "delta29Si_CZ_bulk": "delta29Si_bulk_CZ_permil",
    })
    silicon_vs_eps.to_csv(os.path.join(output_dir, "silicon_vs_eps.csv"), index=False)

    oxygen_spatial = df[["eps", "distance_from_edge_um", "delta18O_CZ_increment"]].dropna().rename(columns={
        "eps": "reaction_progress_eps",
        "distance_from_edge_um": "distance_from_edge_um",
        "delta18O_CZ_increment": "delta18O_CZ_increment_permil",
    })
    oxygen_spatial.to_csv(os.path.join(output_dir, "oxygen_spatial_profile.csv"), index=False)

    silicon_spatial = df[["eps", "distance_from_edge_um", "delta29Si_CZ_increment"]].dropna().rename(columns={
        "eps": "reaction_progress_eps",
        "distance_from_edge_um": "distance_from_edge_um",
        "delta29Si_CZ_increment": "delta29Si_CZ_increment_permil",
    })
    silicon_spatial.to_csv(os.path.join(output_dir, "silicon_spatial_profile.csv"), index=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_plots(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })

    # Plot 1: oxygen vs eps
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(df["eps"], df["delta18O_FL"], color="#1f77b4", lw=2, label=r"Fluid ($\delta^{18}O_{FL}$)")
    ax.plot(df["eps"], df["delta18O_CZ_bulk"], color="#d62728", lw=2, label=r"Bulk CZ ($\delta^{18}O_{CZ,bulk}$)")
    ax.axhline(DELTA18O_CZ_REFERENCE, color="#d62728", lw=1.2, ls="--", alpha=0.7,
               label=r"Measured bulk CZ (reference, %.2f‰)" % DELTA18O_CZ_REFERENCE)
    ax.axhline(DELTA18O_MZ, color="gray", lw=1, ls=":", label=r"MZ ($\delta^{18}O_{MZ}$)")
    ax.set_xlabel("Reaction progress, " + r"$\varepsilon = m_{CZ}/(m_{CZ}+m_{MZ})$")
    ax.set_ylabel(r"$\delta^{18}O$ (‰, VSMOW)")
    ax.set_title("Oxygen isotope evolution during MZ$\\rightarrow$CZ replacement")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot1_oxygen_vs_eps.png"), bbox_inches="tight")

    # Plot 2: silicon vs eps
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(df["eps"], df["delta29Si_FL"], color="#1f77b4", lw=2, label=r"Fluid ($\delta^{29}Si_{FL}$)")
    ax.plot(df["eps"], df["delta29Si_CZ_bulk"], color="#d62728", lw=2, label=r"Bulk CZ ($\delta^{29}Si_{CZ,bulk}$)")
    ax.axhline(DELTA29SI_MZ, color="gray", lw=1, ls=":", label=r"MZ ($\delta^{29}Si_{MZ}$)")
    ax.set_xlabel("Reaction progress, " + r"$\varepsilon = m_{CZ}/(m_{CZ}+m_{MZ})$")
    ax.set_ylabel(r"$\delta^{29}Si$ (‰)")
    ax.set_title("Silicon isotope evolution during MZ$\\rightarrow$CZ replacement")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot2_silicon_vs_eps.png"), bbox_inches="tight")

    # Plot 3: spatial oxygen profile
    spatial = df.dropna(subset=["delta18O_CZ_increment"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(spatial["distance_from_edge_um"], spatial["delta18O_CZ_increment"],
            color="#2ca02c", lw=1.5, marker="o", ms=2.5, label=r"CZ increment $\delta^{18}O$")
    ax.set_xlabel(r"Distance from original crystal edge, $\mu m$")
    ax.set_ylabel(r"$\delta^{18}O_{CZ}$ increment (‰, VSMOW)")
    ax.set_title("Spatial oxygen isotope profile of replacement CZ")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot3_oxygen_spatial_profile.png"), bbox_inches="tight")

    # Plot 4: spatial silicon profile
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(spatial["distance_from_edge_um"], spatial["delta29Si_CZ_increment"],
            color="#9467bd", lw=1.5, marker="o", ms=2.5, label=r"CZ increment $\delta^{29}Si$")
    ax.set_xlabel(r"Distance from original crystal edge, $\mu m$")
    ax.set_ylabel(r"$\delta^{29}Si_{CZ}$ increment (‰)")
    ax.set_title("Spatial silicon isotope profile of replacement CZ")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot4_silicon_spatial_profile.png"), bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    edge0_um = df.attrs["edge0_um"]
    volume0_cm3 = df.attrs["volume0_cm3"]
    final = df.iloc[-1]
    max_abs_error = df["mass_balance_error_mg"].abs().max()

    print("=" * 70)
    print("METAMICT ZIRCON (MZ) -> CRYSTALLINE ZIRCON (CZ) REPLACEMENT MODEL")
    print("=" * 70)
    print("\n--- Initial cube geometry ---")
    print(f"  Initial MZ mass            : {M0_MZ:.4f} mg")
    print(f"  MZ density                 : {DENSITY_MZ:.2f} g/cm^3")
    print(f"  Initial cube volume         : {volume0_cm3:.6e} cm^3")
    print(f"  Initial cube edge length    : {edge0_um:.3f} um")

    print("\n--- Final masses (eps = 1) ---")
    print(f"  MZ mass remaining           : {final['mMZ_mg']:.6f} mg")
    print(f"  CZ mass formed               : {final['mCZ_mg']:.6f} mg")
    print(f"  Fluid (H2O + Si-tracer) mass : {M0_H2O + M0_29SI:.4f} mg (buffered, constant)")

    print("\n--- Isotope compositions at eps = 1 ---")
    print(f"  delta18O_FL (fluid)          : {final['delta18O_FL']:.3f} permil")
    print(f"  delta18O_CZ_bulk (model)     : {final['delta18O_CZ_bulk']:.3f} permil")
    print(f"  delta18O_CZ (measured, ref.) : {DELTA18O_CZ_REFERENCE:.3f} permil")
    print(f"  delta29Si_FL (fluid)         : {final['delta29Si_FL']:.3f} permil")
    print(f"  delta29Si_CZ_bulk (model)    : {final['delta29Si_CZ_bulk']:.3f} permil")

    print("\n--- Mass conservation ---")
    print(f"  Max |mCZ + mMZ - m0_MZ|      : {max_abs_error:.3e} mg")

    print("\nOutputs written to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = run_simulation(DEPS)
    export_csvs(df, OUTPUT_DIR)
    print_summary(df)
    make_plots(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
