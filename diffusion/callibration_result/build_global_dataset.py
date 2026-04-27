import os
import glob
import pandas as pd

from read_data_csv import load_market_data


def build_dataset(data_dir):
    option_files = sorted(glob.glob(os.path.join(data_dir, "*_options.csv")))

    if not option_files:
        raise RuntimeError(f"Aucun fichier *_options.csv trouvé dans {data_dir}")

    all_rows = []

    for opt_file in option_files:
        und_file = opt_file.replace("_options.csv", "_underlying.csv")

        if not os.path.exists(und_file):
            print(f"Missing underlying for {opt_file}")
            continue

        df_raw = pd.read_csv(opt_file)
        available_maturities = sorted(df_raw["maturity"].dropna().unique())

        print(f"\nFichier : {opt_file}")
        print(f"Maturités trouvées : {available_maturities}")

        for maturity in available_maturities:
            try:
                df_opt, S0, tau, t0, expiry, Kmax, S_max = load_market_data(
                    opt_file,
                    und_file,
                    maturity
                )

                df_opt = df_opt.copy()

                df_opt["date"] = str(t0.date())
                df_opt["maturity"] = maturity
                df_opt["tau"] = tau
                df_opt["S0"] = S0
                df_opt["S_max"] = S_max
                df_opt["moneyness"] = df_opt["strike"] / S0
                df_opt["options_csv"] = opt_file
                df_opt["underlying_csv"] = und_file

                all_rows.append(df_opt)

                print(f"OK: {t0.date()} | {maturity} | {len(df_opt)}")

            except Exception as e:
                print(f"Skip: {opt_file} | {maturity} | {e}")

    if not all_rows:
        raise RuntimeError("Aucune donnée chargée.")

    return pd.concat(all_rows, ignore_index=True)


if __name__ == "__main__":
    data_dir = "data/daily_clean"

    df = build_dataset(data_dir)

    os.makedirs("calibration_results", exist_ok=True)

    out_path = "calibration_results/global_raw_dataset.csv"
    df.to_csv(out_path, index=False)

    print("\n=== DONE ===")
    print(df.groupby("maturity").size())
    print(f"Saved to {out_path}")