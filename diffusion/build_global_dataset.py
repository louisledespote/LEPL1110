import glob
import pandas as pd

pattern = "calibration_results/summary_by_strike_*.csv"

files = glob.glob(pattern)

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

df_all.to_csv("global_sigma_dataset.csv", index=False)

print("Dataset construit :", len(df_all), "lignes")
print(df_all.head())