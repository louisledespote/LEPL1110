import sqlite3
import os
from datetime import datetime

# chemin vers la base
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "market_data.db")

def save_underlying_price(ticker, spot, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec="seconds")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO underlying_prices (timestamp, ticker, spot)
    VALUES (?, ?, ?)
    """, (timestamp, ticker, spot))

    conn.commit()
    conn.close()


def save_option_quote(ticker, option_type, maturity, strike, last, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec="seconds")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO option_quotes (timestamp, ticker, option_type, maturity, strike, last)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, ticker, option_type, maturity, strike, last))

    conn.commit()
    conn.close()
def save_spot(conn, timestamp, ticker, spot, bid=None, ask=None):
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO underlying_prices (timestamp, ticker, spot)
    VALUES (?, ?, ?)
    """, (timestamp, ticker, spot))
    conn.commit()
if __name__ == "__main__":
    # exemple test : prix spot
    save_underlying_price("AD", 33.82)

    # exemple test : une option call
    save_option_quote(
        ticker="AD",
        option_type="CALL",
        maturity="2026-06-19",
        strike=34.0,
        last=1.25
    )

    print("Données enregistrées avec succès")