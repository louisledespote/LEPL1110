# Import
import argparse
from html import parser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import plot_prices_vs_strike, plot_payoff, plot_boundary_conditions, plot_final_solution
from stiffness import assemble_black_scholes_operator
from mass import assemble_mass
from dirichlet import theta_step
from fem_eval import evaluate_fe_solution_1d
from read_data_csv import load_market_data

from gmsh_utils import (
    gmsh_init, gmsh_finalize, build_1d_mesh,
    prepare_quadrature_and_basis, get_jacobians, end_dofs_from_nodes
)

def compute_black_scholes_fem_prices(
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
    maturity_label="",
    plot_debug=False
):

    gmsh_init("diffusion_1d")

    # L va fixer le domaine [0, S_max]
    L = S_max

    # On construit le maillage 1D
    # elemtype : type d'élément (1D, 2D, etc.)
    # nodeTags : tags des noeuds
    # nodeCoords : coordonnées des noeuds
    # elemTags : tags des éléments
    # elemNodeTags : pour chaque élément, les tags des noeuds qui lui sont associés
    _, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags = build_1d_mesh(
        L=L, cl1=cl1, cl2=cl2, order=order
    )

    # On construit le mapping entre les tags de noeuds gmsh et les indices de degrés (ij )de liberté (dof) dans notre système linéaire
    unique_dofs_tags = np.unique(elemNodeTags)

    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    dof_coords = np.zeros((num_dofs, 3), dtype=float)

    all_coords = nodeCoords.reshape(-1, 3)

    # 1) construire le mapping tag gmsh -> dof compact
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i

    # 2) remplir correctement les coordonnées des dofs
    for i, tag in enumerate(nodeTags):
        dof_idx = tag_to_dof[int(tag)]
        if dof_idx != -1:
            dof_coords[dof_idx] = all_coords[i]

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords = get_jacobians(elemType, xi)

    # On associe les element du maillage au prix
    S_nodes = dof_coords[:, 0]

    # on assemble les matrices et le vecteur de charge
    K_lil, F = assemble_black_scholes_operator(
        elemTags, elemNodeTags, jac, det, coords, w, N, gN, sigma, r, tag_to_dof
    )
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)

    K = K_lil.tocsr()
    M = M_lil.tocsr()

    # On récupère les indices des noeuds de frontière (S=0 et S=S_max) pour imposer les conditions de Dirichlet
    left = int(np.argmin(S_nodes))
    right = int(np.argmax(S_nodes))

    # les conditions de Dirichlet sont imposées à S=0 et S=S_max
    # On stocke les indices des degrés de liberté de ces noeuds dans dir_dofs et les valeurs correspondantes dans dir_vals (qui seront mises à jour à chaque pas de temps)
    dir_dofs = [left, right]

    # on definit le pas de temps pour l'integragration
    dt = tau / nsteps

    results = []

    for _, row in df_opt.iterrows():
        K_strike = float(row["strike"])
        market_price = float(row["settlement"])

        # condition initiale = payoff U[i] = max(S_i − K, 0)
        # Si l'option nous permet de gagner 10 euro elle ne peut pas valoir plus 
        U = np.maximum(S_nodes - K_strike, 0.0)

        # conditions initiales à S=0 et S=S_max donc sur les bords      
        U[left] = 0.0
        U[right] = max(L - K_strike, 0.0)

        # boucle en temps
        for step in range(nsteps):
            # le temps à maturité au pas n+1
            tau_np1 = (step + 1) * dt

            dir_vals = np.array([
                0.0,
                L - K_strike * np.exp(-r * tau_np1)
            ], dtype=float)

            U = theta_step(
                M, K, F, F, U,
                dt=dt,
                theta=theta,
                dirichlet_dofs=dir_dofs,
                dir_vals_np1=dir_vals
            )

        # évaluation FEM exacte (pas interpolation)
        fem_price = evaluate_fe_solution_1d(
            S0, elemType, elemTags, elemNodeTags, dof_coords, U, tag_to_dof
        )

        results.append({
            "strike": K_strike,
            "market_price": market_price,
            "fem_price": fem_price,
            "abs_error": abs(fem_price - market_price),
            "sq_error": (fem_price - market_price)**2,
            "moneyness": K_strike / S0
        })

        if plot_debug and abs(K_strike - 40.0) < 1e-12:
            plot_payoff(S_nodes, K_strike)
            plot_boundary_conditions(L, K_strike, r, tau)
            plot_final_solution(S_nodes, U, S0, fem_price, K_strike, maturity_label)

    gmsh_finalize()

    return pd.DataFrame(results)


def run_from_csv(
    options_csv,
    underlying_csv,
    maturity,
    sigma=0.2,
    r=0.02,
    order=1,
    cl1=0.05,
    cl2=0.05,
    theta=1.0,
    nsteps=500,
    plot_debug=False
):
    """
    Charge les données CSV puis appelle compute_black_scholes_fem_prices.
    """

    df_opt, S0, tau, t0, expiry, Kmax_market, S_max = load_market_data(
        options_csv,
        underlying_csv,
        maturity
    )

    df_results = compute_black_scholes_fem_prices(
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
        plot_debug=plot_debug,
        maturity_label=maturity
    )

    metadata = {
        "date": t0.date(),
        "maturity": maturity,
        "expiry": expiry.date(),
        "S0": S0,
        "tau": tau,
        "S_max": S_max,
        "Kmax_market": Kmax_market,
        "sigma": sigma,
        "r": r
    }

    return df_results, metadata

def main():
    parser = argparse.ArgumentParser(
        description="Pricing Black-Scholes par FEM"
    )

    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-cl1", type=float, default=0.05)
    parser.add_argument("-cl2", type=float, default=0.05)

    parser.add_argument("--options_csv", type=str, default="../data/daily_clean/2026-04-01_options.csv")
    parser.add_argument("--underlying_csv", type=str, default="../data/daily_clean/2026-04-01_underlying.csv")
    parser.add_argument("--maturity", type=str, default="MAY 2026")

    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--r", type=float, default=0.02)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--nsteps", type=int, default=500)
    parser.add_argument("--plot_debug", action="store_true")

    args = parser.parse_args()

    df_results, metadata = run_from_csv(
        options_csv=args.options_csv,
        underlying_csv=args.underlying_csv,
        maturity=args.maturity,
        sigma=args.sigma,
        r=args.r,
        order=args.order,
        cl1=args.cl1,
        cl2=args.cl2,
        theta=args.theta,
        nsteps=args.nsteps,
        plot_debug=args.plot_debug
    )

    print("=== Données marché ===")
    for key, value in metadata.items():
        print(f"{key:15s}: {value}")

    print("\n=== Comparaison FEM / Marché ===")
    print(f"Erreur moyenne : {df_results['abs_error'].mean():.4f}")

    for _, row in df_results.iterrows():
        print(
            f"K={row['strike']:.2f} | "
            f"Marché={row['market_price']:.4f} | "
            f"FEM={row['fem_price']:.4f} | "
            f"Erreur={row['abs_error']:.4f}"
        )


if __name__ == "__main__":
    main()