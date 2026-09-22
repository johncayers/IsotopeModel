"""Isotope-dilution estimate of hydrothermal zircon (hz) replacing metamict zircon (mz).

m_hz = C_SiO2 * m_SiO2 * (X29_SiO2 - R_hz * X28_SiO2)
       / (C_mz * (R_hz * X28_mz - X29_mz))
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

# --- constants -------------------------------------------------------------
N_A = 6.02214076e23          # Avogadro's number (1/mol)
C_SIO2 = 1 / 3               # atomic conc. of Si in SiO2
C_MZ = 1 / 6                 # atomic conc. of Si in ZrSiO4 (6 atoms/formula unit)
X29_SIO2 = 0.99908           # 29Si atomic fraction in spike SiO2
X28_SIO2 = 1 - X29_SIO2      # 28Si atomic fraction in spike SiO2
# Natural Si abundance in metamict zircon (ASSUMPTION - edit if measured)
X28_MZ = 0.92223
X29_MZ = 0.04685
R_STD = 0.05074              # NBS28 29Si/28Si
GFW_29SIO2 = 61.0            # g/mol
GFW_ZIRCON = 183.31          # g/mol ZrSiO4 (91.224 + 28.086 + 4*15.999)
KG_H2O = 120e-6

# --- experiments: id, duration (d), d29Si_hz (permil), w_29SiO2 (g) ---------
experiments = [
    (14, 3, 6380, 1.768e-3),
    (16, 7, 17891, 2.073e-3),
    (15, 19, 13253, 1.695e-3),
]


def r_hz(delta29):
    """29Si/28Si of hydrothermal zircon from delta notation."""
    return (delta29 / 1000 + 1) * R_STD


def molecules_sio2(w_g):
    return w_g / GFW_29SIO2 * N_A


def molecules_hz(m_sio2, r):
    num = C_SIO2 * m_sio2 * (X29_SIO2 - r * X28_SIO2)
    den = C_MZ * (r * X28_MZ - X29_MZ)
    return num / den


results = []
rows = []
print(f"{'Exp':>4} {'t(d)':>5} {'R_hz':>8} {'m_SiO2':>11} {'m_hz':>11} "
      f"{'rate (mol/kgH2O/s)':>20} {'w_hz (g)':>11}")
for exp, days, delta, w_si in experiments:
    r = r_hz(delta)
    m_sio2 = molecules_sio2(w_si)
    m_hz = molecules_hz(m_sio2, r)
    mol_hz = m_hz / N_A
    seconds = days * 86400
    rate = mol_hz / KG_H2O / seconds            # mol hz / kg H2O / s
    w_hz = rate * KG_H2O * seconds * GFW_ZIRCON  # g (rate * kg H2O * t * GFW)
    results.append((days, w_hz))
    rows.append([exp, days, delta, w_si, r, m_sio2, m_hz, rate, w_hz])
    print(f"{exp:>4} {days:>5} {r:>8.4f} {m_sio2:>11.3e} {m_hz:>11.3e} "
          f"{rate:>20.3e} {w_hz:>11.3e}")

with open("isotope_dilution_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "Duration_d", "delta29Si_hz_permil",
                     "w_29SiO2_g", "R_hz", "m_SiO2_molecules",
                     "m_hz_molecules", "rate_mol_hz_per_kgH2O_per_s",
                     "w_hz_g"])
    writer.writerows(rows)

# --- plot ------------------------------------------------------------------
t, w = np.array(results).T
order = np.argsort(t)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t[order], w[order] * 1e3, "o-", color="tab:blue")
for (exp, *_), ti, wi in zip(experiments, t, w):
    ax.annotate(f"Exp {exp}", (ti, wi * 1e3), textcoords="offset points",
                xytext=(6, -12))
ax.set_xlabel("Time (days)")
ax.set_ylabel("$w_{hz}$ (mg)")
ax.set_title("Hydrothermal zircon formed vs. time")
fig.tight_layout()
fig.savefig("w_hz_vs_time.png", dpi=200)
plt.show()
