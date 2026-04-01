import sqlite3
import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "market_data.db")
CSV_PATH = os.path.join(BASE_DIR, "data", "option_quotes.csv")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

rows = cursor.execute("""
SELECT timestamp, ticker, option_type, maturity, strike, last
FROM option_quotes
ORDER BY maturity, strike
""").fetchall()

conn.close()

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "ticker", "option_type", "maturity", "strike", "settlement"])
    writer.writerows(rows)

print(f"CSV créé : {CSV_PATH}")