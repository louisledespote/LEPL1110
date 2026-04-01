import sqlite3
import os
import csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "market_data.db")
OUT_DIR = os.path.join(BASE_DIR, "data", "daily_clean")

os.makedirs(OUT_DIR, exist_ok=True)

today = datetime.now().date().isoformat()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

rows = cursor.execute("""
SELECT substr(timestamp, 1, 10) as date, ticker, spot
FROM underlying_prices
WHERE substr(timestamp, 1, 10) = ?
ORDER BY timestamp
""", (today,)).fetchall()

conn.close()

csv_path = os.path.join(OUT_DIR, f"{today}_underlying.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "ticker", "spot"])
    writer.writerows(rows)

print(f"CSV créé : {csv_path}")
print(f"{len(rows)} lignes exportées")