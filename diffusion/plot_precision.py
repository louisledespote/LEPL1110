import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

summary_path = Path("precision_results/summary_precision_2026-04-01_MAY_2026.csv")
df = pd.read_csv(summary_path)

out_dir = Path("precision_plots")
out_dir.mkdir(exist_ok=True)

# 1) Influence du temps
df_time = df[df["experiment"] == "raffinement_temps"].sort_values("nsteps")
plt.figure(figsize=(7, 4.5))
plt.plot(df_time["nsteps"], df_time["analytic_mae"], marker="o")
plt.xlabel("Nombre de pas de temps")
plt.ylabel("MAE FEM / analytique")
plt.title("Convergence temporelle")
plt.grid(True)
plt.tight_layout()
time_path = out_dir / "convergence_temps.png"
plt.savefig(time_path, dpi=200)
plt.close()

# 2) Influence de theta
df_theta = df[df["experiment"] == "theta"].sort_values("theta")
plt.figure(figsize=(7, 4.5))
plt.bar(df_theta["theta"].astype(str), df_theta["analytic_mae"])
plt.xlabel(r"$\theta$")
plt.ylabel("MAE FEM / analytique")
plt.title("Influence du schéma temporel")
plt.grid(True, axis="y")
plt.tight_layout()
theta_path = out_dir / "influence_theta.png"
plt.savefig(theta_path, dpi=200)
plt.close()

# 3) Influence de l'ordre
df_order = df[df["experiment"] == "ordre"].sort_values("order")
plt.figure(figsize=(7, 4.5))
plt.plot(df_order["order"], df_order["analytic_mae"], marker="o")
plt.xlabel("Ordre FEM")
plt.ylabel("MAE FEM / analytique")
plt.title("Influence de l'ordre des éléments finis")
plt.grid(True)
plt.xticks(df_order["order"])
plt.tight_layout()
order_path = out_dir / "influence_ordre.png"
plt.savefig(order_path, dpi=200)
plt.close()

# 4) Raffinement maillage
df_mesh = df[df["experiment"] == "raffinement_maillage"].sort_values("cl1", ascending=False)
plt.figure(figsize=(7, 4.5))
plt.plot(df_mesh["cl1"], df_mesh["analytic_mae"], marker="o")
plt.xlabel(r"$cl_1=cl_2$")
plt.ylabel("MAE FEM / analytique")
plt.title("Influence du raffinement spatial")
plt.grid(True)
plt.tight_layout()
mesh_path = out_dir / "raffinement_maillage.png"
plt.savefig(mesh_path, dpi=200)
plt.close()

print("Fichiers créés :")
print(time_path)
print(theta_path)
print(order_path)
print(mesh_path)
