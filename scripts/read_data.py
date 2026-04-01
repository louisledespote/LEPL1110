import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "market_data.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=== underlying_prices ===")
for row in cursor.execute("SELECT * FROM underlying_prices"):
    print(row)

print("\n=== option_quotes ===")
for row in cursor.execute("SELECT * FROM option_quotes"):
    print(row)

conn.close()