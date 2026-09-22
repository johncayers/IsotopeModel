"""Plot Raman spectra of starting materials (Mud Tank, AFG, Norway zircons)."""

import pandas as pd
import matplotlib.pyplot as plt

FILES = [
    "Mud_Tank_20231201.CSV",
    "AFG_Zir05_SM_Avg.csv",
    "Norway_Zircon_2-2_Avg.csv",
]

fig, ax = plt.subplots(figsize=(8, 5))

for filename in FILES:
    df = pd.read_csv(filename)
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    label = filename.rsplit(".", 1)[0]
    ax.plot(x, y, label=label, linewidth=0.8)

ax.set_xlim(300, 1100)
ax.set_xlabel("Raman Shift (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.legend()

fig.tight_layout()
fig.savefig("Raman_Spectra_Starting_Materials.png", dpi=300)
plt.show()
