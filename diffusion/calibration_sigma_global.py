import os
import glob
import argparse
import pandas as pd

from calibration_sigma import run_one_file


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="../data/daily_clean")
    parser.add_argument("--maturities", nargs="+", default=["MAY 2026", "JUN 2026", "SEP 2026"])

    parser.add_argument("--r", type=float, default=0.02)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--cl1", type=float, default=0.05)
    parser.add_argument("--cl2", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--nsteps", type=int, default=500)

    parser.add_argument("--train_frac", type=float, default=0.75)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--out_dir", type=str, default="calibration_results")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    option_files = sorted(glob.glob(os.path.join(args.data_dir, "*_options.csv")))

    all_details = []
    all_summaries = []

    for options_csv in option_files:
        underlying_csv = options_csv.replace("_options.csv", "_underlying.csv")

        if not os.path.exists(underlying_csv):
            print(f"Underlying manquant pour {options_csv}, ignoré.")
            continue

        for maturity in args.maturities:
            print(f"\n=== Calibration : {options_csv} | {maturity} ===")

            try:
                df_details, summary = run_one_file(
                    options_csv=options_csv,
                    underlying_csv=underlying_csv,
                    maturity=maturity,
                    r=args.r,
                    order=args.order,
                    cl1=args.cl1,
                    cl2=args.cl2,
                    theta=args.theta,
                    nsteps=args.nsteps,
                    train_frac=args.train_frac,
                    random_state=args.random_state
                )

                all_details.append(df_details)
                all_summaries.append(summary)

            except Exception as e:
                print(f"Ignoré : {options_csv} | {maturity} | erreur : {e}")

    if not all_summaries:
        raise RuntimeError("Aucune calibration réussie.")

    df_all_details = pd.concat(all_details, ignore_index=True)
    df_all_summary = pd.DataFrame(all_summaries)

    details_path = os.path.join(args.out_dir, "global_details_sigma.csv")
    summary_path = os.path.join(args.out_dir, "global_summary_sigma.csv")

    df_all_details.to_csv(details_path, index=False)
    df_all_summary.to_csv(summary_path, index=False)

    print("\n=== Calibration globale terminée ===")
    print(f"Export détails : {details_path}")
    print(f"Export résumé  : {summary_path}")


if __name__ == "__main__":
    main()