from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# Les options europééen arrive a matu le s3 vendredi du moi
def third_friday(year, month):
    d = datetime(year, month, 1)
    while d.weekday() != 4:  # 4 = vendredi
        d += timedelta(days=1)
    d += timedelta(weeks=2)
    return d




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
