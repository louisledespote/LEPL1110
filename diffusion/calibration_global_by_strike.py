import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from read_data_csv import load_market_data
from main_black_scholes import compute_black_scholes_fem_prices


def build_global_dataset(options_pattern, underlying_pattern, maturities):
    """
    Construit un dataset brut global à partir de plusieurs jours et plusieurs maturités.
    """

    option_files = sorted(glob.glob(options_pattern))
    underlying_files = sorted(glob.glob(underlying_pattern))

    if len(option_files) != len(underlying_files):
        raise ValueError("Nombre de fichiers options et underlying différent.")

    all_rows = []

    for options_csv, underlying_csv in zip(option_files, underlying_files):
        for maturity in maturities:
            try:
                df_opt, S0, tau, t0, expiry, Kmax_market, S_max = load_market_data(
                    options_csv,
                    underlying_csv,
                    maturity
                )

                df_opt = df_opt.copy()
                df_opt["date"] = str(t0.date())
                df_opt["maturity"] = maturity
                df_opt["expiry"] = str(expiry.date())
                df_opt["tau"] = tau
                df_opt["S0"] = S0
                df_opt["S_max"] = S_max
                df_opt["moneyness"] = df_opt["strike"] / S0
                df_opt["options_csv"] = options_csv
                df_opt["underlying_csv"] = underlying_csv

                all_rows.append(df_opt)

                print(f"Chargé : {t0.date()} | {maturity} | {len(df_opt)} options")

            except Exception as e:
                print(f"Ignoré : {options_csv} | {maturity} | {e}")

    if not all_rows:
        raise ValueError("Aucune donnée chargée.")

    return pd.concat(all_rows, ignore_index=True)


def compute_group_error(
    df_group,
    sigma,
    r,
    order,
    cl1,
    cl2,
    theta,
    nsteps
):
    """
    Calcule l'erreur moyenne pour un groupe d'options avec un sigma donné.
    Le groupe peut contenir plusieurs jours et plusieurs maturités.
    """

    all_results = []

    group_keys = ["date", "maturity"]

    for (_, _), df_sub in df_group.groupby(group_keys):
        S0 = float(df_sub["S0"].iloc[0])
        tau = float(df_sub["tau"].iloc[0])
        S_max = float(df_sub["S_max"].iloc[0])

        df_res = compute_black_scholes_fem_prices(
            df_opt=df_sub,
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

        df_res["date"] = df_sub["date"].iloc[0]
        df_res["maturity"] = df_sub["maturity"].iloc[0]
        df_res["tau"] = tau
        df_res["S0"] = S0
        df_res["S_max"] = S_max

        all_results.append(df_res)

    df_all = pd.concat(all_results, ignore_index=True)

    return float(df_all["sq_error"].mean()), df_all


def calibrate_sigma_for_strike_global(
    df_strike,
    r,
    order,
    cl1,
    cl2,
    theta,
    nsteps,
    sigma_min=0.01,
    sigma_max=1.0
):
    """
    Pour un strike donné, calibre un sigma unique en utilisant
    toutes les observations disponibles : jours + maturités.
    """

    def objective(sigma):
        mse, _ = compute_group_error(
            df_group=df_strike,
            sigma=sigma,
            r=r,
            order=order,
            cl1=cl1,
            cl2=cl2,
            theta=theta,
            nsteps=nsteps
        )
        return mse

    sol = minimize_scalar(
        objective,
        bounds=(sigma_min, sigma_max),
        method="bounded",
        options={"xatol": 1e-3}
    )

    sigma_star = float(sol.x)
    train_mse, df_results = compute_group_error(
        df_group=df_strike,
        sigma=sigma_star,
        r=r,
        order=order,
        cl1=cl1,
        cl2=cl2,
        theta=theta,
        nsteps=nsteps
    )

    return sigma_star, train_mse, df_results


def run_global_calibration_by_strike(
    df_global,
    r,
    order,
    cl1,
    cl2,
    theta,
    nsteps
):
    """
    Calibre un sigma par strike en utilisant toutes les dates
    et toutes les maturités disponibles dans df_global.
    """

    all_details = []
    summaries = []

    for strike, df_strike in df_global.groupby("strike"):
        print(f"\n=== Calibration strike K={strike:.2f} ===")

        sigma_star, mse, df_res = calibrate_sigma_for_strike_global(
            df_strike=df_strike,
            r=r,
            order=order,
            cl1=cl1,
            cl2=cl2,
            theta=theta,
            nsteps=nsteps
        )

        df_res["group_strike"] = float(strike)
        df_res["sigma_calibrated"] = sigma_star

        all_details.append(df_res)

        summaries.append({
            "strike": float(strike),
            "sigma_calibrated": sigma_star,
            "mse": mse,
            "mae": float(df_res["abs_error"].mean()),
            "n_obs": len(df_res),
            "n_dates": df_strike["date"].nunique(),
            "n_maturities": df_strike["maturity"].nunique(),
            "mean_moneyness": float(df_strike["moneyness"].mean()),
            "min_moneyness": float(df_strike["moneyness"].min()),
            "max_moneyness": float(df_strike["moneyness"].max())
        })

        print(
            f"K={strike:.2f} | "
            f"sigma={sigma_star:.4f} | "
            f"MAE={summaries[-1]['mae']:.4f} | "
            f"n={summaries[-1]['n_obs']} | "
            f"dates={summaries[-1]['n_dates']} | "
            f"maturités={summaries[-1]['n_maturities']}"
        )

    df_details = pd.concat(all_details, ignore_index=True)
    df_summary = pd.DataFrame(summaries)

    return df_details, df_summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--options_pattern",
        type=str,
        default="../data/daily_clean/*_options.csv"
    )

    parser.add_argument(
        "--underlying_pattern",
        type=str,
        default="../data/daily_clean/*_underlying.csv"
    )

    parser.add_argument(
        "--maturities",
        nargs="+",
        default=["MAY 2026", "JUN 2026", "SEP 2026"]
    )

    parser.add_argument("--r", type=float, default=0.02)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--cl1", type=float, default=0.05)
    parser.add_argument("--cl2", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--nsteps", type=int, default=500)

    parser.add_argument("--out_dir", type=str, default="calibration_results")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Construction du dataset brut global
    df_global = build_global_dataset(
        options_pattern=args.options_pattern,
        underlying_pattern=args.underlying_pattern,
        maturities=args.maturities
    )

    raw_path = os.path.join(args.out_dir, "global_raw_dataset.csv")
    df_global.to_csv(raw_path, index=False)

    print("\n=== Dataset brut global construit ===")
    print(f"Nombre de lignes : {len(df_global)}")
    print(f"Export brut      : {raw_path}")

    # 2) Calibration globale par strike
    df_details, df_summary = run_global_calibration_by_strike(
        df_global=df_global,
        r=args.r,
        order=args.order,
        cl1=args.cl1,
        cl2=args.cl2,
        theta=args.theta,
        nsteps=args.nsteps
    )

    details_path = os.path.join(
        args.out_dir,
        "details_global_by_strike.csv"
    )

    summary_path = os.path.join(
        args.out_dir,
        "summary_global_by_strike.csv"
    )

    df_details.to_csv(details_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print("\n=== Calibration globale par strike terminée ===")
    print(f"Export détails : {details_path}")
    print(f"Export résumé  : {summary_path}")


if __name__ == "__main__":
    main()