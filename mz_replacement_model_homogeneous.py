"""
Mass-balance model of metamict zircon (MZ) replacement by crystalline zircon (CZ)
during an isotope-doping experiment (O and Si tracer exchange) -- HOMOGENEOUS
(fully re-equilibrating) end-member.

MODEL OVERVIEW
---------------
This script uses the same replacement schedule, geometry, and initial
conditions as the heterogeneous ("segregated increment") model
(mz_replacement_model.py), but replaces its core isotopic assumption:

  Heterogeneous model : a CZ increment inherits the fluid's composition at
                         the moment it precipitates, then is isolated from
                         the fluid and never changes again (preserves
                         growth zoning).

  Homogeneous model    : bulk CZ isotopically RE-EQUILIBRATES with the fluid
                         at every timestep, i.e. delta_CZ_bulk(t) =
                         delta_FL(t) at all times (no zoning is preserved;
                         intracrystalline diffusion/recrystallization is
                         fast enough to keep the whole crystal homogenized
                         with the fluid).

Practically, this means previously precipitated CZ is not isotopically
isolated: it stays in continuous isotopic communication with the fluid, so
it acts as an additional, growing buffering mass alongside the fluid
reservoir. At each timestep:

  1. A mass increment dm of MZ is dissolved and removed from MZ (same
     structural/mass replacement schedule as the heterogeneous model).
  2. dm mixes into a single combined "fluid + existing bulk CZ" isotope
     pool (mass = fluid reservoir + current CZ mass), because the two are
     always at the same composition under full re-equilibration.
  3. The resulting composition becomes both the new fluid composition AND
     the new bulk CZ composition (delta_CZ_bulk = delta_FL, by definition).
  4. Masses of MZ, CZ, and the shrinking-cube geometry update exactly as in
     the heterogeneous model; mass conservation is verified.

As in the companion script, oxygen and silicon are tracked as two
independent reservoirs (oxygen = H2O + the Si-tracer's own oxygen; silicon =
the 29Si tracer), and delta18O_CZ = -35.70 permil is reported only as a
measured reference value, not used as a model input.

SPATIAL PROFILES UNDER FULL RE-EQUILIBRATION
----------------------------------------------
Because bulk CZ continuously re-equilibrates with the fluid, CZ formed early
in the reaction does NOT retain its formation-time composition -- it keeps
being pulled toward the fluid's composition at every later timestep, right
up to eps = 1. The measurable end state is therefore isotopically
HOMOGENEOUS throughout the crystal: every position records the same, final
bulk composition, regardless of when it structurally formed. This is the
key diagnostic contrast with the heterogeneous model's preserved growth
zoning, and is the intended, physically meaningful result of this script
(flat spatial profiles), not a modeling artifact.

All other simplifying assumptions requested in the original task are
honored: no isotopic fractionation, perfect mass conservation, stoichiometric
(mass-for-mass) replacement, and a perfectly mixed fluid at all times.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Initial conditions (identical to the heterogeneous model, for comparison)
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

# Reaction-progress snapshots for the intermediate spatial-profile plots
EPS_SNAPSHOTS = [0.1, 0.2, 0.3, 0.4, 0.5]

CM_TO_UM = 1.0e4
MG_TO_G = 1.0e-3

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# ---------------------------------------------------------------------------
# Geometry (unchanged from the heterogeneous model)
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
    Step reaction progress eps = mCZ / (mCZ + mMZ) from 0 to 1, with bulk CZ
    fully re-equilibrating with the fluid at every step.

    At each step:
      1. A mass increment dm of MZ is dissolved and removed from MZ.
      2. dm mixes into the combined "fluid + existing bulk CZ" pool (both
         already share the same composition under full re-equilibration).
      3. The resulting composition is assigned to BOTH the fluid and the
         bulk CZ (delta_CZ_bulk = delta_FL, always).
      4. Masses of MZ, CZ, and the shrinking-cube geometry update, and mass
         conservation is verified.

    Because previously formed CZ keeps re-equilibrating, the growing CZ mass
    itself becomes part of the isotope-buffering pool -- unlike the
    heterogeneous model, where the fluid reservoir size stays fixed.
    """
    n_steps = int(round(1.0 / deps))
    eps_grid = np.linspace(0.0, 1.0, n_steps + 1)

    geom0 = cube_geometry(M0_MZ, DENSITY_MZ)
    edge0_cm = geom0["edge_cm"]

    # Fluid-only reservoir masses (before any CZ has formed)
    mass_O_fluid = M0_H2O + M0_29SI
    mass_Si_fluid = M0_29SI

    delta18O = mass_weighted_mix(DELTA18O_H2O, M0_H2O, DELTA18O_SI, M0_29SI)
    delta29Si = DELTA29SI_SI  # matches given initial condition delta29Si_FL = delta29Si_Si

    total_system_mass = M0_MZ + M0_H2O + M0_29SI

    records = []

    # eps = 0 starting state (no CZ has precipitated yet; fluid == bulk CZ pool)
    records.append({
        "eps": 0.0,
        "mCZ_mg": M0_CZ,
        "mMZ_mg": M0_MZ,
        "mass_increment_mg": 0.0,
        "delta18O_FL": delta18O,
        "delta18O_CZ_bulk": np.nan,
        "delta18O_CZ_increment": np.nan,
        "delta29Si_FL": delta29Si,
        "delta29Si_CZ_bulk": np.nan,
        "delta29Si_CZ_increment": np.nan,
        "remaining_MZ_edge_um": geom0["edge_um"],
        "distance_from_edge_um": 0.0,
        "mass_balance_error_mg": 0.0,
    })

    mCZ_prev = 0.0

    for eps in eps_grid[1:]:
        mCZ_total = M0_MZ * eps
        mMZ_total = M0_MZ * (1.0 - eps)
        dm = mCZ_total - mCZ_prev  # mass increment converted this step

        # --- combined "fluid + existing bulk CZ" pool re-equilibrates with
        #     the dissolving MZ increment (full back-reaction/homogenization) ---
        pool_mass_O = mass_O_fluid + mCZ_prev
        pool_mass_Si = mass_Si_fluid + mCZ_prev
        delta18O = mass_weighted_mix(delta18O, pool_mass_O, DELTA18O_MZ, dm)
        delta29Si = mass_weighted_mix(delta29Si, pool_mass_Si, DELTA29SI_MZ, dm)

        # --- bulk CZ equals fluid at every timestep (homogeneous assumption) ---
        delta18O_FL = delta18O
        delta18O_CZ_bulk = delta18O
        delta18O_CZ_increment = delta18O  # newly formed material starts at this value too
        delta29Si_FL = delta29Si
        delta29Si_CZ_bulk = delta29Si
        delta29Si_CZ_increment = delta29Si

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
    fluid_mass_mg = M0_H2O + M0_29SI
    df["total_system_mass_mg"] = df["mCZ_mg"] + df["mMZ_mg"] + fluid_mass_mg

    # Under full re-equilibration, every point in the crystal shares the SAME
    # composition as the final bulk CZ/fluid value once the experiment ends
    # (continuous re-equilibration erases any earlier gradient). These
    # "final equilibrated" columns are what a real homogenized crystal would
    # actually show if probed spatially post-experiment, and are what the
    # spatial-profile plots/CSVs use.
    final_delta18O = df["delta18O_CZ_bulk"].iloc[-1]
    final_delta29Si = df["delta29Si_CZ_bulk"].iloc[-1]
    df["delta18O_CZ_final_equilibrated"] = np.where(df["delta18O_CZ_bulk"].isna(), np.nan, final_delta18O)
    df["delta29Si_CZ_final_equilibrated"] = np.where(df["delta29Si_CZ_bulk"].isna(), np.nan, final_delta29Si)

    df.attrs["total_system_mass_initial_mg"] = total_system_mass
    df.attrs["edge0_cm"] = edge0_cm
    df.attrs["edge0_um"] = geom0["edge_um"]
    df.attrs["volume0_cm3"] = geom0["volume_cm3"]
    df.attrs["mass_O_fluid_mg"] = mass_O_fluid
    df.attrs["mass_Si_fluid_mg"] = mass_Si_fluid

    return df


def get_spatial_snapshots(df: pd.DataFrame, eps_snapshots: list) -> pd.DataFrame:
    """
    Build the spatial isotope profile as it would appear at intermediate
    stages of reaction progress (not just the final state).

    Under full re-equilibration, ALL CZ that exists at a given instant
    shares one composition (= the bulk/fluid composition at that instant),
    so the "profile" at reaction progress eps is a flat segment spanning
    from the original crystal edge (position 0) out to the reaction front
    at that eps, at height delta_CZ_bulk(eps). Two rows per snapshot (the
    segment's two endpoints) are stored so the snapshot can be drawn as a
    horizontal line directly from the CSV.
    """
    rows = []
    for eps_snap in eps_snapshots:
        idx = (df["eps"] - eps_snap).abs().idxmin()
        row = df.loc[idx]
        front_um = row["distance_from_edge_um"]
        for position_um in (0.0, front_um):
            rows.append({
                "eps_snapshot": eps_snap,
                "position_um": position_um,
                "delta18O_CZ_permil": row["delta18O_CZ_bulk"],
                "delta29Si_CZ_permil": row["delta29Si_CZ_bulk"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csvs(df: pd.DataFrame, output_dir: str, snapshots: pd.DataFrame) -> None:
    os.makedirs(output_dir, exist_ok=True)

    oxygen_vs_eps = df[["eps", "delta18O_FL", "delta18O_CZ_bulk"]].rename(columns={
        "eps": "reaction_progress_eps",
        "delta18O_FL": "delta18O_fluid_permil",
        "delta18O_CZ_bulk": "delta18O_bulk_CZ_permil",
    })
    oxygen_vs_eps.to_csv(os.path.join(output_dir, "oxygen_vs_eps_homogeneous.csv"), index=False)

    silicon_vs_eps = df[["eps", "delta29Si_FL", "delta29Si_CZ_bulk"]].rename(columns={
        "eps": "reaction_progress_eps",
        "delta29Si_FL": "delta29Si_fluid_permil",
        "delta29Si_CZ_bulk": "delta29Si_bulk_CZ_permil",
    })
    silicon_vs_eps.to_csv(os.path.join(output_dir, "silicon_vs_eps_homogeneous.csv"), index=False)

    # Spatial profiles reflect the final, fully re-equilibrated (homogeneous)
    # crystal: a single composition at every position (see docstring).
    oxygen_spatial = df[["eps", "distance_from_edge_um", "delta18O_CZ_final_equilibrated"]].dropna().rename(columns={
        "eps": "reaction_progress_eps",
        "distance_from_edge_um": "distance_from_edge_um",
        "delta18O_CZ_final_equilibrated": "delta18O_CZ_increment_permil",
    })
    oxygen_spatial.to_csv(os.path.join(output_dir, "oxygen_spatial_profile_homogeneous.csv"), index=False)

    silicon_spatial = df[["eps", "distance_from_edge_um", "delta29Si_CZ_final_equilibrated"]].dropna().rename(columns={
        "eps": "reaction_progress_eps",
        "distance_from_edge_um": "distance_from_edge_um",
        "delta29Si_CZ_final_equilibrated": "delta29Si_CZ_increment_permil",
    })
    silicon_spatial.to_csv(os.path.join(output_dir, "silicon_spatial_profile_homogeneous.csv"), index=False)

    # Intermediate reaction-progress snapshots (eps = 0.1 ... 0.5): the
    # crystal's uniform composition at each stage, spanning the edge (0 um)
    # out to that stage's reaction front.
    oxygen_snapshots = snapshots[["eps_snapshot", "position_um", "delta18O_CZ_permil"]].rename(columns={
        "eps_snapshot": "reaction_progress_eps",
        "position_um": "distance_from_edge_um",
        "delta18O_CZ_permil": "delta18O_CZ_permil",
    })
    oxygen_snapshots.to_csv(os.path.join(output_dir, "oxygen_spatial_profile_snapshots_homogeneous.csv"), index=False)

    silicon_snapshots = snapshots[["eps_snapshot", "position_um", "delta29Si_CZ_permil"]].rename(columns={
        "eps_snapshot": "reaction_progress_eps",
        "position_um": "distance_from_edge_um",
        "delta29Si_CZ_permil": "delta29Si_CZ_permil",
    })
    silicon_snapshots.to_csv(os.path.join(output_dir, "silicon_spatial_profile_snapshots_homogeneous.csv"), index=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_snapshot_plots(snapshots: pd.DataFrame, output_dir: str, eps_snapshots: list) -> None:
    """
    Spatial profiles at intermediate reaction progress (eps = 0.1 ... 0.5).
    Each snapshot is a flat segment (0 um to that stage's reaction front),
    since the whole crystal is isotopically uniform at every instant under
    full re-equilibration. Plotting several snapshots together shows how
    that uniform composition, and the extent of the crystal it applies to,
    both evolve as the reaction proceeds.
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(eps_snapshots)))

    # Oxygen snapshots
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for eps_snap, color in zip(eps_snapshots, colors):
        seg = snapshots[snapshots["eps_snapshot"] == eps_snap]
        ax.plot(seg["position_um"], seg["delta18O_CZ_permil"], color=color, lw=2.5, marker="o", ms=5,
                label=fr"$\varepsilon$ = {eps_snap:.1f}")
    ax.set_xlabel(r"Distance from original crystal edge, $\mu m$")
    ax.set_ylabel(r"$\delta^{18}O_{CZ}$ (‰, VSMOW)")
    ax.set_title("Spatial oxygen isotope profile at intermediate reaction progress\n(homogeneous re-equilibration model)")
    ax.legend(loc="best", frameon=True, title="Reaction progress")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot5_oxygen_spatial_profile_snapshots_homogeneous.png"), bbox_inches="tight")

    # Silicon snapshots
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for eps_snap, color in zip(eps_snapshots, colors):
        seg = snapshots[snapshots["eps_snapshot"] == eps_snap]
        ax.plot(seg["position_um"], seg["delta29Si_CZ_permil"], color=color, lw=2.5, marker="o", ms=5,
                label=fr"$\varepsilon$ = {eps_snap:.1f}")
    ax.set_xlabel(r"Distance from original crystal edge, $\mu m$")
    ax.set_ylabel(r"$\delta^{29}Si_{CZ}$ (‰)")
    ax.set_title("Spatial silicon isotope profile at intermediate reaction progress\n(homogeneous re-equilibration model)")
    ax.legend(loc="best", frameon=True, title="Reaction progress")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot6_silicon_spatial_profile_snapshots_homogeneous.png"), bbox_inches="tight")


def make_plots(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })

    # Plot 1: oxygen vs eps (fluid and bulk CZ coincide exactly -- the
    # defining feature of full re-equilibration -- so a dashed overlay is
    # used to make both series visible).
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(df["eps"], df["delta18O_FL"], color="#1f77b4", lw=3, label=r"Fluid ($\delta^{18}O_{FL}$)")
    ax.plot(df["eps"], df["delta18O_CZ_bulk"], color="#d62728", lw=1.5, ls="--",
            label=r"Bulk CZ ($\delta^{18}O_{CZ,bulk}$, = fluid)")
    ax.axhline(DELTA18O_CZ_REFERENCE, color="#d62728", lw=1.2, ls=":", alpha=0.7,
               label=r"Measured bulk CZ (reference, %.2f‰)" % DELTA18O_CZ_REFERENCE)
    ax.axhline(DELTA18O_MZ, color="gray", lw=1, ls=":", label=r"MZ ($\delta^{18}O_{MZ}$)")
    ax.set_xlabel("Reaction progress, " + r"$\varepsilon = m_{CZ}/(m_{CZ}+m_{MZ})$")
    ax.set_ylabel(r"$\delta^{18}O$ (‰, VSMOW)")
    ax.set_title("Oxygen isotope evolution (homogeneous re-equilibration model)")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot1_oxygen_vs_eps_homogeneous.png"), bbox_inches="tight")

    # Plot 2: silicon vs eps
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(df["eps"], df["delta29Si_FL"], color="#1f77b4", lw=3, label=r"Fluid ($\delta^{29}Si_{FL}$)")
    ax.plot(df["eps"], df["delta29Si_CZ_bulk"], color="#d62728", lw=1.5, ls="--",
            label=r"Bulk CZ ($\delta^{29}Si_{CZ,bulk}$, = fluid)")
    ax.axhline(DELTA29SI_MZ, color="gray", lw=1, ls=":", label=r"MZ ($\delta^{29}Si_{MZ}$)")
    ax.set_xlabel("Reaction progress, " + r"$\varepsilon = m_{CZ}/(m_{CZ}+m_{MZ})$")
    ax.set_ylabel(r"$\delta^{29}Si$ (‰)")
    ax.set_title("Silicon isotope evolution (homogeneous re-equilibration model)")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot2_silicon_vs_eps_homogeneous.png"), bbox_inches="tight")

    # Plot 3: spatial oxygen profile (flat -- see docstring: full
    # re-equilibration erases any preserved growth zoning)
    spatial = df.dropna(subset=["delta18O_CZ_final_equilibrated"])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(spatial["distance_from_edge_um"], spatial["delta18O_CZ_final_equilibrated"],
            color="#2ca02c", lw=2, marker="o", ms=2.5,
            label=r"CZ (final, homogenized) $\delta^{18}O$")
    ax.set_xlabel(r"Distance from original crystal edge, $\mu m$")
    ax.set_ylabel(r"$\delta^{18}O_{CZ}$ (‰, VSMOW)")
    ax.set_title("Spatial oxygen isotope profile (homogeneous re-equilibration model)")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot3_oxygen_spatial_profile_homogeneous.png"), bbox_inches="tight")

    # Plot 4: spatial silicon profile (flat, same reasoning)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(spatial["distance_from_edge_um"], spatial["delta29Si_CZ_final_equilibrated"],
            color="#9467bd", lw=2, marker="o", ms=2.5,
            label=r"CZ (final, homogenized) $\delta^{29}Si$")
    ax.set_xlabel(r"Distance from original crystal edge, $\mu m$")
    ax.set_ylabel(r"$\delta^{29}Si_{CZ}$ (‰)")
    ax.set_title("Spatial silicon isotope profile (homogeneous re-equilibration model)")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot4_silicon_spatial_profile_homogeneous.png"), bbox_inches="tight")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    edge0_um = df.attrs["edge0_um"]
    volume0_cm3 = df.attrs["volume0_cm3"]
    final = df.iloc[-1]
    max_abs_error = df["mass_balance_error_mg"].abs().max()

    print("=" * 70)
    print("MZ -> CZ REPLACEMENT MODEL (HOMOGENEOUS RE-EQUILIBRATION)")
    print("=" * 70)
    print("\n--- Initial cube geometry ---")
    print(f"  Initial MZ mass            : {M0_MZ:.4f} mg")
    print(f"  MZ density                 : {DENSITY_MZ:.2f} g/cm^3")
    print(f"  Initial cube volume         : {volume0_cm3:.6e} cm^3")
    print(f"  Initial cube edge length    : {edge0_um:.3f} um")

    print("\n--- Final masses (eps = 1) ---")
    print(f"  MZ mass remaining           : {final['mMZ_mg']:.6f} mg")
    print(f"  CZ mass formed               : {final['mCZ_mg']:.6f} mg")
    print(f"  Fluid (H2O + Si-tracer) mass : {M0_H2O + M0_29SI:.4f} mg")

    print("\n--- Isotope compositions at eps = 1 (fluid == bulk CZ, by definition) ---")
    print(f"  delta18O_FL = delta18O_CZ_bulk  : {final['delta18O_FL']:.3f} permil")
    print(f"  delta18O_CZ (measured, ref.)    : {DELTA18O_CZ_REFERENCE:.3f} permil")
    print(f"  delta29Si_FL = delta29Si_CZ_bulk: {final['delta29Si_FL']:.3f} permil")

    print("\n--- Mass conservation ---")
    print(f"  Max |mCZ + mMZ - m0_MZ|      : {max_abs_error:.3e} mg")

    print("\nNote: spatial profiles (Plots 3-4) are flat by construction -- full")
    print("re-equilibration homogenizes the whole crystal to the final bulk")
    print("composition, erasing any growth zoning preserved in the")
    print("heterogeneous (segregated-increment) model. Plots 5-6 show the same")
    print(f"effect at intermediate stages (eps = {EPS_SNAPSHOTS}): a uniform")
    print("composition across the crystal reacted so far, that keeps shifting")
    print("with reaction progress.")

    print("\nOutputs written to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = run_simulation(DEPS)
    snapshots = get_spatial_snapshots(df, EPS_SNAPSHOTS)
    export_csvs(df, OUTPUT_DIR, snapshots)
    print_summary(df)
    make_plots(df, OUTPUT_DIR)
    make_snapshot_plots(snapshots, OUTPUT_DIR, EPS_SNAPSHOTS)
    plt.show()


if __name__ == "__main__":
    main()
