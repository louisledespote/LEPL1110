import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm

from read_data_csv import load_market_data
from main_black_scholes import compute_black_scholes_fem_prices


def black_scholes_call_price(S0, K, tau, sigma, r):
    if tau <= 0:
        return max(S0 - K, 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    return S0 * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def run_fem_vs_analytic(
    df_opt,
    S0,
    tau,
    S_max,
    sigma,
    r,
    order,
    cl1,
    cl2,
    theta,
    nsteps,
    experiment_name
):
    df_res = compute_black_scholes_fem_prices(
        df_opt=df_opt,
        S0=S0,
        tau=tau,
        S_max=S_max,
        sigma=sigma,
        r=r,
        order=order,
        cl1=cl1,
        cl2=cl2,
        theta=theta,
        nsteps=nsteps,
        plot_debug=False
    )

    analytic_prices = []

    for _, row in df_res.iterrows():
        K = float(row["strike"])
        analytic_prices.append(
            black_scholes_call_price(S0, K, tau, sigma, r)
        )

    df_res["analytic_price"] = analytic_prices
    df_res["analytic_abs_error"] = abs(df_res["fem_price"] - df_res["analytic_price"])
    df_res["analytic_sq_error"] = (df_res["fem_price"] - df_res["analytic_price"]) ** 2

    df_res["experiment"] = experiment_name
    df_res["order"] = order
    df_res["cl1"] = cl1
    df_res["cl2"] = cl2
    df_res["theta"] = theta
    df_res["nsteps"] = nsteps
    df_res["sigma"] = sigma
    df_res["r"] = r
    df_res["tau"] = tau
    df_res["S0"] = S0
    df_res["S_max"] = S_max

    summary = {
        "experiment": experiment_name,
        "order": order,
        "cl1": cl1,
        "cl2": cl2,
        "theta": theta,
        "nsteps": nsteps,
        "sigma": sigma,
        "r": r,
        "tau": tau,
        "S0": S0,
        "S_max": S_max,
        "analytic_mae": float(df_res["analytic_abs_error"].mean()),
        "analytic_mse": float(df_res["analytic_sq_error"].mean()),
        "market_mae": float(df_res["abs_error"].mean()),
        "market_mse": float(df_res["sq_error"].mean()),
        "n_options": len(df_res)
    }

    return df_res, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--options_csv", type=str, default="../data/daily_clean/2026-04-01_options.csv")
    parser.add_argument("--underlying_csv", type=str, default="../data/daily_clean/2026-04-01_underlying.csv")
    parser.add_argument("--maturity", type=str, default="MAY 2026")

    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--r", type=float, default=0.02)

    parser.add_argument("--out_dir", type=str, default="precision_results")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_opt, S0, tau, t0, expiry, Kmax_market, S_max = load_market_data(
        args.options_csv,
        args.underlying_csv,
        args.maturity
    )

    experiments = []

    # 1) Influence de l'ordre FEM
    for order in [1, 2, 3]:
        experiments.append({
            "experiment_name": "ordre",
            "order": order,
            "cl1": 0.05,
            "cl2": 0.05,
            "theta": 1.0,
            "nsteps": 500
        })

    # 2) Influence du schéma temporel theta
    for theta in [0.5, 1.0]:
        experiments.append({
            "experiment_name": "theta",
            "order": 2,
            "cl1": 0.05,
            "cl2": 0.05,
            "theta": theta,
            "nsteps": 500
        })

    # 3) Influence du raffinement spatial
    for cl in [0.10, 0.05, 0.02]:
        experiments.append({
            "experiment_name": "raffinement_maillage",
            "order": 2,
            "cl1": cl,
            "cl2": cl,
            "theta": 1.0,
            "nsteps": 500
        })

    # 4) Influence du nombre de pas de temps
    for nsteps in [100, 250, 500, 1000]:
        experiments.append({
            "experiment_name": "raffinement_temps",
            "order": 2,
            "cl1": 0.05,
            "cl2": 0.05,
            "theta": 1.0,
            "nsteps": nsteps
        })

    all_details = []
    all_summaries = []

    for exp in experiments:
        print(
            f"\n=== {exp['experiment_name']} | "
            f"order={exp['order']} | "
            f"theta={exp['theta']} | "
            f"cl={exp['cl1']} | "
            f"nsteps={exp['nsteps']} ==="
        )

        df_res, summary = run_fem_vs_analytic(
            df_opt=df_opt,
            S0=S0,
            tau=tau,
            S_max=S_max,
            sigma=args.sigma,
            r=args.r,
            order=exp["order"],
            cl1=exp["cl1"],
            cl2=exp["cl2"],
            theta=exp["theta"],
            nsteps=exp["nsteps"],
            experiment_name=exp["experiment_name"]
        )

        all_details.append(df_res)
        all_summaries.append(summary)

        print(
            f"MAE analytique={summary['analytic_mae']:.6e} | "
            f"MAE marché={summary['market_mae']:.6e}"
        )

    df_details = pd.concat(all_details, ignore_index=True)
    df_summary = pd.DataFrame(all_summaries)

    maturity_str = args.maturity.replace(" ", "_")
    date_str = str(t0.date())

    details_path = os.path.join(
        args.out_dir,
        f"details_precision_{date_str}_{maturity_str}.csv"
    )

    summary_path = os.path.join(
        args.out_dir,
        f"summary_precision_{date_str}_{maturity_str}.csv"
    )

    df_details.to_csv(details_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print("\n=== Étude de précision terminée ===")
    print(df_summary[
        [
            "experiment",
            "order",
            "cl1",
            "cl2",
            "theta",
            "nsteps",
            "analytic_mae",
            "market_mae"
        ]
    ])

    print(f"\nExport détails : {details_path}")
    print(f"Export résumé  : {summary_path}")


if __name__ == "__main__":
    main()