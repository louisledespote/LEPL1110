import sqlite3
import os

# chemin vers la base de données
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "market_data.db")

# crée le dossier data/ si pas existant
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# connexion (crée le fichier automatiquement)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# table prix action (spot)
cursor.execute("""
CREATE TABLE IF NOT EXISTS underlying_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    spot REAL NOT NULL
)
""")


# table options
cursor.execute("""
CREATE TABLE IF NOT EXISTS option_quotes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    option_type TEXT NOT NULL,
    maturity TEXT NOT NULL,
    strike REAL NOT NULL,
    last REAL
)
""")

conn.commit()
conn.close()

print("Base de données créée avec succès")

