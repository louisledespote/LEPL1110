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
SELECT
    substr(timestamp, 1, 10) as date,
    ticker,
    option_type,
    maturity,
    strike,
    last
FROM option_quotes
WHERE substr(timestamp, 1, 10) = ?
  AND option_type = 'CALL'
  AND maturity IS NOT NULL
  AND maturity != ''
  AND strike IS NOT NULL
  AND last IS NOT NULL
ORDER BY maturity, strike
""", (today,)).fetchall()

conn.close()

seen = set()
clean_rows = []
for row in rows:
    key = (row[0], row[1], row[2], row[3], row[4], row[5])
    if key not in seen:
        seen.add(key)
        clean_rows.append(row)

csv_path = os.path.join(OUT_DIR, f"{today}_options.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "ticker", "option_type", "maturity", "strike", "settlement"])
    writer.writerows(clean_rows)

print(f"CSV journalier créé : {csv_path}")
print(f"{len(clean_rows)} lignes exportées")