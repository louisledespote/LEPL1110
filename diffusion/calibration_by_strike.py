import os
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from read_data_csv import load_market_data
from main_black_scholes import compute_black_scholes_fem_prices


def calibrate_sigma_for_group(
    df_group,
    S0,
    tau,
    S_max,
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
    Calibre un sigma unique pour un groupe d'options.
    Ici, le groupe correspond typiquement à un strike donné.
    """

    def objective(sigma):
        df_res = compute_black_scholes_fem_prices(
            df_opt=df_group,
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

        return float(df_res["sq_error"].mean())

    sol = minimize_scalar(
        objective,
        bounds=(sigma_min, sigma_max),
        method="bounded",
        options={"xatol": 1e-3}
    )

    return float(sol.x), float(sol.fun)


def run_calibration_by_strike(
    options_csv,
    underlying_csv,
    maturity,
    r,
    order,
    cl1,
    cl2,
    theta,
    nsteps,
    train_frac,
    random_state
):
    """
    Pour une date et une maturité données :
    - on groupe les options par strike
    - on calibre sigma pour chaque strike
    - on calcule les erreurs train/test
    """

    df_opt, S0, tau, t0, expiry, Kmax_market, S_max = load_market_data(
        options_csv,
        underlying_csv,
        maturity
    )

    all_details = []
    summaries = []

    for strike, df_strike in df_opt.groupby("strike"):

        if len(df_strike) < 2:
            df_train = df_strike.copy()
            df_test = df_strike.iloc[0:0].copy()
        else:
            df_train = df_strike.sample(frac=train_frac, random_state=random_state)
            df_test = df_strike.drop(df_train.index)

        sigma_star, train_mse = calibrate_sigma_for_group(
            df_train,
            S0,
            tau,
            S_max,
            r,
            order,
            cl1,
            cl2,
            theta,
            nsteps
        )

        df_train_res = compute_black_scholes_fem_prices(
            df_opt=df_train,
            S0=S0,
            tau=tau,
            S_max=S_max,
            sigma=sigma_star,
            r=r,
            order=order,
            cl1=cl1,
            cl2=cl2,
            theta=theta,
            nsteps=nsteps,
            plot_debug=False
        )

        df_train_res["set"] = "train"

        if len(df_test) > 0:
            df_test_res = compute_black_scholes_fem_prices(
                df_opt=df_test,
                S0=S0,
                tau=tau,
                S_max=S_max,
                sigma=sigma_star,
                r=r,
                order=order,
                cl1=cl1,
                cl2=cl2,
                theta=theta,
                nsteps=nsteps,
                plot_debug=False
            )
            df_test_res["set"] = "test"
        else:
            df_test_res = pd.DataFrame()

        df_details = pd.concat([df_train_res, df_test_res], ignore_index=True)

        df_details["date"] = t0.date()
        df_details["maturity"] = maturity
        df_details["expiry"] = expiry.date()
        df_details["tau"] = tau
        df_details["S0"] = S0
        df_details["S_max"] = S_max
        df_details["sigma_calibrated"] = sigma_star
        df_details["group_strike"] = strike
        df_details["group_moneyness"] = strike / S0

        all_details.append(df_details)

        summaries.append({
            "date": t0.date(),
            "maturity": maturity,
            "expiry": expiry.date(),
            "tau": tau,
            "S0": S0,
            "strike": float(strike),
            "moneyness": float(strike / S0),
            "sigma_calibrated": sigma_star,
            "train_mse": train_mse,
            "train_mae": float(df_train_res["abs_error"].mean()),
            "test_mae": float(df_test_res["abs_error"].mean()) if len(df_test_res) > 0 else np.nan,
            "n_train": len(df_train_res),
            "n_test": len(df_test_res)
        })

        print(
            f"K={strike:.2f} | "
            f"K/S0={strike/S0:.4f} | "
            f"sigma={sigma_star:.4f} | "
            f"train_mae={summaries[-1]['train_mae']:.4f} | "
            f"test_mae={summaries[-1]['test_mae']:.4f}"
        )

    df_details_all = pd.concat(all_details, ignore_index=True)
    df_summary = pd.DataFrame(summaries)

    return df_details_all, df_summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--options_csv", type=str, default="../data/daily_clean/2026-04-01_options.csv")
    parser.add_argument("--underlying_csv", type=str, default="../data/daily_clean/2026-04-01_underlying.csv")
    parser.add_argument("--maturity", type=str, default="MAY 2026")

    parser.add_argument("--r", type=float, default=0.02)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--cl1", type=float, default=0.05)
    parser.add_argument("--cl2", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--nsteps", type=int, default=500)

    parser.add_argument("--train_frac", type=float, default=0.75)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--out_dir", type=str, default="calibration_results")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_details, df_summary = run_calibration_by_strike(
        options_csv=args.options_csv,
        underlying_csv=args.underlying_csv,
        maturity=args.maturity,
        r=args.r,
        order=args.order,
        cl1=args.cl1,
        cl2=args.cl2,
        theta=args.theta,
        nsteps=args.nsteps,
        train_frac=args.train_frac,
        random_state=args.random_state
    )

    date_str = str(df_summary["date"].iloc[0])
    maturity_str = args.maturity.replace(" ", "_")

    details_path = os.path.join(
        args.out_dir,
        f"details_by_strike_{date_str}_{maturity_str}.csv"
    )

    summary_path = os.path.join(
        args.out_dir,
        f"summary_by_strike_{date_str}_{maturity_str}.csv"
    )

    df_details.to_csv(details_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print("\n=== Calibration par strike terminée ===")
    print(f"Export détails : {details_path}")
    print(f"Export résumé  : {summary_path}")


if __name__ == "__main__":
    main()