import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

summary_path = Path("precision_results/summary_precision_2026-04-01_MAY_2026.csv")
df = pd.read_csv(summary_path)

out_dir = Path("precision_plots")
out_dir.mkdir(exist_ok=True)


def savefig(name):
    path = out_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"saved: {path}")


def get_experiment(name):
    d = df[df["experiment"] == name].sort_values("nsteps").copy()
    if d.empty:
        print(f"Expérience absente du CSV : {name}")
    return d


def plot_time_convergence(exp_name, label, expected_order=None):
    d = get_experiment(exp_name)
    if d.empty:
        return None

    n = d["nsteps"].to_numpy(dtype=float)
    err = d["analytic_mae"].to_numpy(dtype=float)

    mask = (n > 0) & (err > 0)
    n = n[mask]
    err = err[mask]

    if len(n) < 2:
        print(f"Pas assez de points pour {exp_name}")
        return None

    b, a = np.polyfit(np.log(n), np.log(err), 1)
    p_obs = -b

    plt.figure(figsize=(8, 5))
    plt.loglog(n, err, "o-", linewidth=2, markersize=8, label="Erreur FEM / analytique")
    plt.loglog(n, np.exp(a) * n**b, "--", linewidth=2,
               label=fr"Fit log-log, ordre observé $\approx {p_obs:.2f}$")

    if expected_order is not None:
        ref = err[0] * (n / n[0]) ** (-expected_order)
        plt.loglog(n, ref, ":", linewidth=2,
                   label=fr"Référence ordre {expected_order}")

    plt.xlabel(r"Nombre de pas de temps $nsteps$ (échelle log)")
    plt.ylabel(r"Erreur MAE (échelle log)")
    plt.title(f"Convergence temporelle - {label}")
    plt.grid(True, which="both")
    plt.legend()

    savefig(f"convergence_{exp_name}.png")

    print(f"{label} : ordre observé = {p_obs:.4f}")
    return n, err, p_obs


# ============================================================
# 1) Convergence Euler implicite
# ============================================================

euler_data = plot_time_convergence(
    "raffinement_temps_euler",
    r"Euler implicite $\theta=1$",
    expected_order=1
)


# ============================================================
# 2) Convergence Crank-Nicolson
# ============================================================

cn_data = plot_time_convergence(
    "raffinement_temps_crank_nicolson",
    r"Crank-Nicolson $\theta=0.5$",
    expected_order=2
)


# ============================================================
# 3) Comparaison Euler vs Crank-Nicolson
# ============================================================

plt.figure(figsize=(8, 5))

for exp_name, label in [
    ("raffinement_temps_euler", r"Euler implicite $\theta=1$"),
    ("raffinement_temps_crank_nicolson", r"Crank-Nicolson $\theta=0.5$"),
]:
    d = get_experiment(exp_name)
    if d.empty:
        continue

    n = d["nsteps"].to_numpy(dtype=float)
    err = d["analytic_mae"].to_numpy(dtype=float)

    mask = (n > 0) & (err > 0)
    n = n[mask]
    err = err[mask]

    if len(n) < 2:
        continue

    b, _ = np.polyfit(np.log(n), np.log(err), 1)
    p_obs = -b

    plt.loglog(n, err, "o-", linewidth=2, markersize=8,
               label=fr"{label}, ordre $\approx {p_obs:.2f}$")

plt.xlabel(r"Nombre de pas de temps $nsteps$ (échelle log)")
plt.ylabel(r"Erreur MAE (échelle log)")
plt.title("Comparaison des schémas temporels")
plt.grid(True, which="both")
plt.legend()
savefig("comparaison_schemas_temporels.png")


# ============================================================
# 4) Ordre observé + gain Euler
# ============================================================

if euler_data is not None:
    n, err, _ = euler_data

    orders = []
    gains = []
    labels = []

    for i in range(len(n) - 1):
        n1, n2 = n[i], n[i + 1]
        e1, e2 = err[i], err[i + 1]

        labels.append(f"{int(n1)}→{int(n2)}")
        gains.append(e1 / e2)
        orders.append(np.log(e1 / e2) / np.log(n2 / n1))

    plt.figure(figsize=(8, 5))
    plt.bar(labels, orders)
    plt.axhline(1.0, linestyle="--", linewidth=2, label="Ordre 1 attendu")
    plt.xlabel("Raffinement temporel")
    plt.ylabel("Ordre observé")
    plt.title(r"Ordre temporel observé - Euler implicite $\theta=1$")
    plt.grid(True, axis="y")
    plt.legend()
    savefig("ordre_observe_temps_euler.png")

    plt.figure(figsize=(8, 5))
    plt.bar(labels, gains)
    plt.axhline(2.0, linestyle="--", linewidth=2, label="Gain ×2 attendu")
    plt.xlabel("Augmentation de nsteps")
    plt.ylabel(r"Gain de précision $e_1/e_2$")
    plt.title(r"Gain de précision - Euler implicite $\theta=1$")
    plt.grid(True, axis="y")
    plt.legend()
    savefig("gain_precision_nsteps_euler.png")


# ============================================================
# 5) Influence du schéma theta
# ============================================================

df_theta = df[df["experiment"] == "theta"].sort_values("theta").copy()

if not df_theta.empty:
    plt.figure(figsize=(8, 5))
    plt.semilogy(
        df_theta["theta"],
        df_theta["analytic_mae"],
        "o-",
        linewidth=2,
        markersize=8
    )
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"Erreur MAE (échelle log)")
    plt.title("Influence du schéma temporel")
    plt.grid(True, which="both")
    savefig("influence_theta_semilogy.png")


# ============================================================
# 6) Raffinement spatial
# ============================================================

df_mesh = df[df["experiment"] == "raffinement_maillage"].sort_values("cl1").copy()

if not df_mesh.empty:
    cl = df_mesh["cl1"].to_numpy(dtype=float)
    err_mesh = df_mesh["analytic_mae"].to_numpy(dtype=float)

    plt.figure(figsize=(8, 5))
    plt.loglog(cl, err_mesh, "o-", linewidth=2, markersize=8)
    plt.gca().invert_xaxis()
    plt.xlabel(r"Précision du maillage $ cl1 =  cl2$ (échelle log)")
    plt.ylabel(r"Erreur MAE (échelle log)")
    plt.title("Raffinement spatial")
    plt.grid(True, which="both")
    savefig("raffinement_maillage_loglog.png")


# ============================================================
# 7) Influence de l'ordre FEM
# ============================================================

df_order = df[df["experiment"] == "ordre"].sort_values("order").copy()

if not df_order.empty:
    plt.figure(figsize=(8, 5))
    plt.semilogy(
        df_order["order"],
        df_order["analytic_mae"],
        "o-",
        linewidth=2,
        markersize=8
    )
    plt.xlabel("Ordre des éléments finis")
    plt.ylabel(r"Erreur MAE (échelle log)")
    plt.title("Influence de l'ordre FEM")
    plt.xticks(df_order["order"])
    plt.grid(True, which="both")
    savefig("influence_ordre_semilogy.png")


# ============================================================
# 8) Export tableau convergence Euler + Crank-Nicolson
# ============================================================

rows = []

for exp_name, scheme in [
    ("raffinement_temps_euler", "Euler implicite"),
    ("raffinement_temps_crank_nicolson", "Crank-Nicolson"),
]:
    d = get_experiment(exp_name)
    if d.empty:
        continue

    n = d["nsteps"].to_numpy(dtype=float)
    err = d["analytic_mae"].to_numpy(dtype=float)

    for i in range(len(n)):
        gain = np.nan
        p = np.nan

        if i > 0:
            gain = err[i - 1] / err[i]
            p = np.log(err[i - 1] / err[i]) / np.log(n[i] / n[i - 1])

        rows.append({
            "scheme": scheme,
            "experiment": exp_name,
            "nsteps": n[i],
            "analytic_mae": err[i],
            "gain_vs_previous": gain,
            "observed_order_vs_previous": p,
        })

df_conv = pd.DataFrame(rows)
csv_path = out_dir / "table_convergence_temporelle.csv"
df_conv.to_csv(csv_path, index=False)
print(f"saved: {csv_path}")