import os
import glob
import argparse
import numpy as np
import pandas as pd
import gmsh

from scipy.optimize import minimize_scalar

from read_data_csv import load_market_data
from gmsh_utils import (
    gmsh_init, gmsh_finalize, build_1d_mesh,
    prepare_quadrature_and_basis, get_jacobians
)
from stiffness import assemble_black_scholes_operator
from mass import assemble_mass
from dirichlet import theta_step
from fem_eval import evaluate_fe_solution_1d


def price_options_for_sigma(
    df_opt, S0, tau, S_max, sigma, r,
    order=1, cl1=0.05, cl2=0.05,
    theta=1.0, nsteps=500
):
    """
    Calcule les prix FEM pour toutes les options d'une même date et maturité.
    """

    gmsh_init("black_scholes_calibration")

    L = S_max

    _, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags = build_1d_mesh(
        L=L, cl1=cl1, cl2=cl2, order=order
    )

    unique_dofs_tags = np.unique(elemNodeTags)

    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))

    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    dof_coords = np.zeros((num_dofs, 3), dtype=float)
    all_coords = nodeCoords.reshape(-1, 3)

    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i

    for i, tag in enumerate(nodeTags):
        dof_idx = tag_to_dof[int(tag)]
        if dof_idx != -1:
            dof_coords[dof_idx] = all_coords[i]

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords = get_jacobians(elemType, xi)

    S_nodes = dof_coords[:, 0]

    K_lil, F = assemble_black_scholes_operator(
        elemTags, elemNodeTags, jac, det, coords, w, N, gN,
        sigma, r, tag_to_dof
    )

    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)

    Kmat = K_lil.tocsr()
    Mmat = M_lil.tocsr()

    left = int(np.argmin(S_nodes))
    right = int(np.argmax(S_nodes))
    dir_dofs = [left, right]

    dt = tau / nsteps

    results = []

    for _, row in df_opt.iterrows():
        K_strike = float(row["strike"])
        market_price = float(row["settlement"])

        U = np.maximum(S_nodes - K_strike, 0.0)
        U[left] = 0.0
        U[right] = max(L - K_strike, 0.0)

        for step in range(nsteps):
            tau_np1 = (step + 1) * dt

            dir_vals = np.array([
                0.0,
                L - K_strike * np.exp(-r * tau_np1)
            ], dtype=float)

            U = theta_step(
                Mmat, Kmat, F, F, U,
                dt=dt,
                theta=theta,
                dirichlet_dofs=dir_dofs,
                dir_vals_np1=dir_vals
            )

        fem_price = evaluate_fe_solution_1d(
            S0, elemType, elemTags, elemNodeTags,
            dof_coords, U, tag_to_dof
        )

        results.append({
            "strike": K_strike,
            "market_price": market_price,
            "fem_price": fem_price,
            "abs_error": abs(fem_price - market_price),
            "sq_error": (fem_price - market_price) ** 2,
            "moneyness": K_strike / S0
        })

    gmsh_finalize()

    return pd.DataFrame(results)


def calibration_error(
    sigma, df_train, S0, tau, S_max, r,
    order, cl1, cl2, theta, nsteps
):
    df_prices = price_options_for_sigma(
        df_train, S0, tau, S_max,
        sigma=sigma,
        r=r,
        order=order,
        cl1=cl1,
        cl2=cl2,
        theta=theta,
        nsteps=nsteps
    )

    return float(df_prices["sq_error"].mean())


def calibrate_sigma(
    df_train, S0, tau, S_max, r,
    order=1, cl1=0.05, cl2=0.05,
    theta=1.0, nsteps=500,
    sigma_min=0.01, sigma_max=1.00
):
    """
    Trouve sigma qui minimise l'erreur quadratique moyenne.
    """

    res = minimize_scalar(
        calibration_error,
        bounds=(sigma_min, sigma_max),
        method="bounded",
        args=(df_train, S0, tau, S_max, r, order, cl1, cl2, theta, nsteps),
        options={"xatol": 1e-3}
    )

    return float(res.x), float(res.fun)


def run_one_file(
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
    df_opt, S0, tau, t0, expiry, Kmax_market, S_max = load_market_data(
        options_csv,
        underlying_csv,
        maturity
    )

    df_train = df_opt.sample(frac=train_frac, random_state=random_state)
    df_test = df_opt.drop(df_train.index)

    sigma_opt, train_mse = calibrate_sigma(
        df_train, S0, tau, S_max, r,
        order=order,
        cl1=cl1,
        cl2=cl2,
        theta=theta,
        nsteps=nsteps
    )

    train_prices = price_options_for_sigma(
        df_train, S0, tau, S_max,
        sigma=sigma_opt,
        r=r,
        order=order,
        cl1=cl1,
        cl2=cl2,
        theta=theta,
        nsteps=nsteps
    )

    test_prices = price_options_for_sigma(
        df_test, S0, tau, S_max,
        sigma=sigma_opt,
        r=r,
        order=order,
        cl1=cl1,
        cl2=cl2,
        theta=theta,
        nsteps=nsteps
    )

    train_prices["set"] = "train"
    test_prices["set"] = "test"

    df_out = pd.concat([train_prices, test_prices], ignore_index=True)

    df_out["date"] = t0.date()
    df_out["maturity"] = maturity
    df_out["expiry"] = expiry.date()
    df_out["tau"] = tau
    df_out["S0"] = S0
    df_out["S_max"] = S_max
    df_out["sigma_calibrated"] = sigma_opt
    df_out["r"] = r
    df_out["options_csv"] = options_csv

    summary = {
        "date": t0.date(),
        "maturity": maturity,
        "expiry": expiry.date(),
        "tau": tau,
        "S0": S0,
        "S_max": S_max,
        "sigma_calibrated": sigma_opt,
        "train_mse": train_mse,
        "train_mae": float(train_prices["abs_error"].mean()),
        "test_mae": float(test_prices["abs_error"].mean()) if len(test_prices) > 0 else np.nan,
        "n_train": len(train_prices),
        "n_test": len(test_prices),
        "options_csv": options_csv
    }

    return df_out, summary


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

    df_results, summary = run_one_file(
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

    date_str = str(summary["date"])
    maturity_str = args.maturity.replace(" ", "_")

    results_path = os.path.join(
        args.out_dir,
        f"details_{date_str}_{maturity_str}.csv"
    )

    summary_path = os.path.join(
        args.out_dir,
        f"summary_{date_str}_{maturity_str}.csv"
    )

    df_results.to_csv(results_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n=== Calibration terminée ===")
    print(f"Date              : {summary['date']}")
    print(f"Maturité          : {summary['maturity']}")
    print(f"Sigma calibré     : {summary['sigma_calibrated']:.6f}")
    print(f"Train MAE         : {summary['train_mae']:.6f}")
    print(f"Test MAE          : {summary['test_mae']:.6f}")
    print(f"Export détails    : {results_path}")
    print(f"Export résumé     : {summary_path}")


if __name__ == "__main__":
    main()