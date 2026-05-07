import os
import argparse
import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from main_black_scholes import compute_black_scholes_fem_prices


def compute_group_error(df_group, sigma, r, order, cl1, cl2, theta, nsteps):
    all_results = []

    for (_, _), df_sub in df_group.groupby(["date", "maturity"]):
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


def calibrate_sigma_for_group(
    df_group,
    r,
    order,
    cl1,
    cl2,
    theta,
    nsteps,
    sigma_min=0.01,
    sigma_max=1.00
):
    def objective(sigma):
        mse, _ = compute_group_error(
            df_group=df_group,
            sigma=sigma,
            r=r,
            order=order,
            cl1=cl1,
            cl2=cl2,
            theta=theta,
            nsteps=nsteps
        )
        return mse

    res = minimize_scalar(
        objective,
        bounds=(sigma_min, sigma_max),
        method="bounded",
        options={"xatol": 1e-3}
    )

    sigma_opt = float(res.x)

    train_mse, train_prices = compute_group_error(
        df_group=df_group,
        sigma=sigma_opt,
        r=r,
        order=order,
        cl1=cl1,
        cl2=cl2,
        theta=theta,
        nsteps=nsteps
    )

    return sigma_opt, train_mse, train_prices


def run_one_strike_from_dataset(
    df_global,
    dataset_csv,
    strike,
    r,
    order,
    cl1,
    cl2,
    theta,
    nsteps,
    train_frac,
    random_state
):
    df_strike = df_global[df_global["strike"] == strike].copy()

    if df_strike.empty:
        raise ValueError(f"Aucune option trouvée pour le strike {strike}")

    df_train = df_strike.sample(frac=train_frac, random_state=random_state)
    df_test = df_strike.drop(df_train.index)

    sigma_opt, train_mse, train_prices = calibrate_sigma_for_group(
        df_group=df_train,
        r=r,
        order=order,
        cl1=cl1,
        cl2=cl2,
        theta=theta,
        nsteps=nsteps
    )

    if len(df_test) > 0:
        _, test_prices = compute_group_error(
            df_group=df_test,
            sigma=sigma_opt,
            r=r,
            order=order,
            cl1=cl1,
            cl2=cl2,
            theta=theta,
            nsteps=nsteps
        )
    else:
        test_prices = pd.DataFrame()

    train_prices["set"] = "train"
    test_prices["set"] = "test"

    df_out = pd.concat([train_prices, test_prices], ignore_index=True)
    df_out["sigma_calibrated"] = sigma_opt
    df_out["r"] = r
    df_out["group_strike"] = strike

    summary = {
        "strike": float(strike),
        "sigma_calibrated": sigma_opt,
        "train_mse": train_mse,
        "train_mae": float(train_prices["abs_error"].mean()),
        "test_mae": float(test_prices["abs_error"].mean()) if len(test_prices) > 0 else np.nan,
        "n_train": len(train_prices),
        "n_test": len(test_prices),
        "n_obs": len(df_strike),
        "n_dates": df_strike["date"].nunique(),
        "n_maturities": df_strike["maturity"].nunique(),
        "mean_moneyness": float(df_strike["moneyness"].mean()),
        "min_moneyness": float(df_strike["moneyness"].min()),
        "max_moneyness": float(df_strike["moneyness"].max()),
        "dataset_csv": dataset_csv
    }

    return df_out, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_csv",
        type=str,
        default="calibration_results/global_raw_dataset.csv"
    )

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

    df_global = pd.read_csv(args.dataset_csv)

    strikes = sorted(df_global["strike"].dropna().unique())

    print("\nStrikes trouvés dans le dataset :")
    for strike in strikes:
        print(" -", strike)

    all_results = []
    all_summaries = []

    for strike in strikes:
        print(f"\n=== Calibration strike : {strike:.2f} ===")

        df_results, summary = run_one_strike_from_dataset(
            df_global=df_global,
            dataset_csv=args.dataset_csv,
            strike=strike,
            r=args.r,
            order=args.order,
            cl1=args.cl1,
            cl2=args.cl2,
            theta=args.theta,
            nsteps=args.nsteps,
            train_frac=args.train_frac,
            random_state=args.random_state
        )

        all_results.append(df_results)
        all_summaries.append(summary)

        print(
            f"K={strike:.2f} | "
            f"sigma={summary['sigma_calibrated']:.6f} | "
            f"train MAE={summary['train_mae']:.6f} | "
            f"test MAE={summary['test_mae']:.6f} | "
            f"n={summary['n_obs']} | "
            f"dates={summary['n_dates']} | "
            f"maturities={summary['n_maturities']}"
        )

    df_all_results = pd.concat(all_results, ignore_index=True)
    df_all_summary = pd.DataFrame(all_summaries)

    results_path = os.path.join(args.out_dir, "details_sigma_by_strike.csv")
    summary_path = os.path.join(args.out_dir, "summary_sigma_by_strike.csv")

    df_all_results.to_csv(results_path, index=False)
    df_all_summary.to_csv(summary_path, index=False)

    print("\n=== Calibration par strike terminée ===")
    print(
        df_all_summary[
            [
                "strike",
                "sigma_calibrated",
                "train_mae",
                "test_mae",
                "n_train",
                "n_test",
                "n_dates",
                "n_maturities"
            ]
        ]
    )

    print(f"\nExport détails : {results_path}")
    print(f"Export résumé  : {summary_path}")


if __name__ == "__main__":
    main()