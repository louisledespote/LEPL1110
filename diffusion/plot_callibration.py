import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

maturity_path = Path("calibration_results/summary_sigma_by_maturity.csv")
strike_path = Path("calibration_results/summary_sigma_by_strike.csv")

df_maturity = pd.read_csv(maturity_path)
df_strike = pd.read_csv(strike_path)

out_dir = Path("calibration_plots")
out_dir.mkdir(exist_ok=True)

# -------- MAE par maturité --------

maturity_order = [
    "APR 2026", "MAY 2026", "JUN 2026", "JUL 2026",
    "SEP 2026", "DEC 2026", "MAR 2027", "JUN 2027"
]

df_maturity["maturity"] = pd.Categorical(
    df_maturity["maturity"],
    categories=maturity_order,
    ordered=True
)

df_maturity = df_maturity.sort_values("maturity")

plt.figure(figsize=(8, 4.5))
plt.plot(df_maturity["maturity"].astype(str), df_maturity["train_mae"], "o-", label="Train MAE")
plt.plot(df_maturity["maturity"].astype(str), df_maturity["test_mae"], "s-", label="Test MAE")
plt.xlabel("Maturité")
plt.ylabel("MAE (€)")
plt.title("Erreur de calibration par maturité")
plt.xticks(rotation=35)
plt.grid(True, axis="y", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "mae_par_maturite.png", dpi=250)
plt.close()


# -------- Sigma par maturité --------

plt.figure(figsize=(8, 4.5))
plt.plot(df_maturity["maturity"].astype(str), df_maturity["sigma_calibrated"], "o-")
plt.xlabel("Maturité")
plt.ylabel(r"Volatilité calibrée $\sigma$")
plt.title(r"Volatilité calibrée par maturité")
plt.xticks(rotation=35)
plt.grid(True, axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(out_dir / "sigma_par_maturite.png", dpi=250)
plt.close()


# -------- MAE par strike --------

df_strike = df_strike.sort_values("strike")

plt.figure(figsize=(8, 4.5))
plt.plot(df_strike["strike"], df_strike["train_mae"], "o-", label="Train MAE")
plt.plot(df_strike["strike"], df_strike["test_mae"], "s-", label="Test MAE")
plt.xlabel("Strike")
plt.ylabel("MAE (€)")
plt.title("Erreur de calibration par strike")
plt.grid(True, axis="y", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "mae_par_strike.png", dpi=250)
plt.close()


# -------- Sigma par strike --------

plt.figure(figsize=(8, 4.5))
plt.plot(df_strike["strike"], df_strike["sigma_calibrated"], "o-")
plt.xlabel("Strike")
plt.ylabel(r"Volatilité calibrée $\sigma$")
plt.title(r"Volatilité calibrée par strike")
plt.grid(True, axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(out_dir / "sigma_par_strike.png", dpi=250)
plt.close()

print("Graphes créés dans :", out_dir)