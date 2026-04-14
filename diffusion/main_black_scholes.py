# main_diffusion_1d.py
import argparse
from html import parser
from turtle import left, right
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from plot_utils import plot_prices_vs_strike
from stiffness import assemble_stiffness_and_rhs, assemble_rhs_neumann


sigma = 0.2
r = 0.02
K_strike = 40
T = 1.0
S_max = 100


# Les options europééen arrive a matu le s3 vendredi du moi
def third_friday(year, month):
    d = datetime(year, month, 1)
    while d.weekday() != 4:  # 4 = vendredi
        d += timedelta(days=1)
    d += timedelta(weeks=2)
    return d

from gmsh_utils import (
    gmsh_init, gmsh_finalize, build_1d_mesh,
    prepare_quadrature_and_basis, get_jacobians, end_dofs_from_nodes
)
from stiffness import assemble_black_scholes_operator
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_fe_solution_high_order, setup_interactive_figure



def maturity_label_to_expiry(label):
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }

    mmm, yyyy = label.strip().split()
    month = month_map[mmm.upper()]
    year = int(yyyy)
    return third_friday(year, month)

def load_market_data(options_csv, underlying_csv, maturity_label):
    df_opt = pd.read_csv(options_csv)
    df_und = pd.read_csv(underlying_csv)

    # garder seulement les calls de la maturité demandée
    df_opt = df_opt[df_opt["option_type"] == "CALL"].copy()
    df_opt = df_opt[df_opt["maturity"] == maturity_label].copy()
    df_opt = df_opt.sort_values("strike").reset_index(drop=True)

    if df_opt.empty:
        raise ValueError(f"Aucune option trouvée pour la maturité {maturity_label}")

    if df_und.empty:
        raise ValueError("Le fichier underlying est vide")

    # spot = dernier spot disponible
    S0 = float(df_und["spot"].iloc[-1])

    # date de collecte
    date_str = str(df_opt["date"].iloc[0])
    t0 = datetime.fromisoformat(date_str)

    # date d'échéance
    expiry = maturity_label_to_expiry(maturity_label)

    # temps à maturité en années
    tau = max((expiry - t0).days, 0) / 365.0
    
    #
    Kmax_market = float(df_opt["strike"].max())
    Smax = max(3.0 * Kmax_market, 2.0 * S0)

    return df_opt, S0, tau, t0, expiry, Kmax_market, Smax



def main():
    parser = argparse.ArgumentParser(description="Diffusion 1D with theta-scheme (Gmsh high-order FE)")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-cl1", type=float, default=0.05)
    parser.add_argument("-cl2", type=float, default=0.05)
    parser.add_argument("-L", type=float, default=1.0)
    parser.add_argument("--options_csv", type=str, default="../data/daily_clean/2026-04-01_options.csv")
    parser.add_argument("--underlying_csv", type=str, default="../data/daily_clean/2026-04-01_underlying.csv")
    parser.add_argument("--maturity", type=str, default="MAY 2026") 
    parser.add_argument("--theta", type=float, default=1.0, help="1: implicit Euler, 0.5: Crank-Nicolson, 0: explicit")
    parser.add_argument("--dt", type=float, default=1.0e-04)
    parser.add_argument("--nsteps", type=int, default=500)
    
    
    args = parser.parse_args()
    df_opt, S0, tau, t0, expiry, Kmax_market, S_max = load_market_data(
        args.options_csv,
        args.underlying_csv,
        args.maturity
    )

    print("=== Données marché ===")
    print("Maturité choisie :", args.maturity)
    print("Spot S0          :", S0)
    print("Date collecte    :", t0.date())
    print("Date échéance    :", expiry.date())
    print("Tau              :", tau)
    print("S_max            :", S_max)
    print("K_max            :", Kmax_market)
    print(df_opt[["strike", "settlement"]])
    
    gmsh_init("diffusion_1d")

    L = S_max

    _, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags = build_1d_mesh(
        L=L, cl1=args.cl1, cl2=args.cl2, order=args.order
    )

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
            
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    sigma = 0.2
    r = 0.02
    S_nodes = dof_coords[:, 0]

    print("S_nodes min =", np.min(S_nodes))
    print("S_nodes max =", np.max(S_nodes))
    print("Premiers S_nodes =", S_nodes[:10])
    K_lil, F = assemble_black_scholes_operator(
        elemTags, elemNodeTags, jac, det, coords, w, N, gN, sigma, r, tag_to_dof
        )
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)

    K = K_lil.tocsr()
    M = M_lil.tocsr()
    print("Test signe K :", K[0, 0], K[1, 1])

    n = len(F)
    S_nodes = dof_coords[:, 0]
    left = int(np.argmin(S_nodes))
    right = int(np.argmax(S_nodes))
    dir_dofs = [left, right]
    
    
    dt = tau / args.nsteps

    results = []

    for _, row in df_opt.iterrows():
        K_strike = float(row["strike"])
        market_price = float(row["settlement"])

        # condition initiale = payoff
        U = np.maximum(S_nodes - K_strike, 0.0)
        U[left] = 0.0
        U[right] = max(L - K_strike, 0.0)

        # boucle en temps
        for step in range(args.nsteps):
            tau_np1 = (step + 1) * dt

            dir_vals = np.array([
                0.0,
                L - K_strike * np.exp(-r * tau_np1)
            ], dtype=float)

            U = theta_step(
                M, K, F, F, U,
                dt=dt,
                theta=args.theta,
                dirichlet_dofs=dir_dofs,
                dir_vals_np1=dir_vals
            )

        # prix FEM évalué au spot
        order =  np.argsort(S_nodes)
        S_nodes_sorted = S_nodes[order]
        U_sorted = U[order]
        fem_price = np.interp(S0, S_nodes_sorted, U_sorted)
        print("S0 =", S0)
        print("U min =", np.min(U), "U max =", np.max(U))
        print("S_sorted first =", S_nodes_sorted[:10])
        results.append({
            "strike": K_strike,
            "market_price": market_price,
            "fem_price": fem_price,
            "abs_error": abs(fem_price - market_price)
            })
        
    print("\n=== Comparaison FEM / Marché ===")
    for row in results:
        print(
            f"K={row['strike']:.2f} | "
            f"Marché={row['market_price']:.4f} | "
            f"FEM={row['fem_price']:.4f} | "
            f"Erreur={row['abs_error']:.4f}"
        )
        plot_prices_vs_strike(results, args.maturity)
    gmsh_finalize()



if __name__ == "__main__":
    main()
