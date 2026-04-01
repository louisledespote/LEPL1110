import sqlite3
import os
import re
from datetime import datetime
from playwright.sync_api import sync_playwright

from save_data import save_spot

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "market_data.db")

URL = "https://live.euronext.com/fr/product/stock-options/AH9-DAMS"

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def to_float(text):
    text = text.strip().replace(",", ".")
    if text in {"", "-", "--"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def save_option(conn, row):
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO option_quotes (timestamp, ticker, option_type, maturity, strike, last)
    VALUES (?, ?, ?, ?, ?, ?)
    """, row)
    conn.commit()


def dismiss_cookie_popup(page):
    for text in ["I Accept", "Accepter", "Tout accepter", "Accept"]:
        try:
            page.get_by_text(text, exact=False).click(timeout=2000)
            print(f"Popup cookies géré via : {text}")
            page.wait_for_timeout(2000)
            return
        except Exception:
            pass

    try:
        if page.locator("#onetrust-consent-sdk").count() > 0:
            page.evaluate("""
                () => {
                    const el = document.querySelector('#onetrust-consent-sdk');
                    if (el) el.remove();
                }
            """)
            print("Popup cookies supprimé du DOM")
            page.wait_for_timeout(1000)
    except Exception as e:
        print("Suppression popup impossible :", e)


def get_maturity_options(page):
    """
    Lit directement les options du <select>.
    Retourne une liste de tuples (label_visible, value_html).
    Exemple: [("Apr 2026", "01-04-2026"), ...]
    """
    options = page.locator("select option")
    results = []

    for i in range(options.count()):
        text = options.nth(i).inner_text().strip()
        value = options.nth(i).get_attribute("value")
        if not text or not value:
            continue
        if re.fullmatch(r"[A-Za-z]{3}\s+\d{4}", text):
            results.append((text, value))

    # suppression doublons éventuels
    unique = []
    seen = set()
    for item in results:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def extract_current_header_maturity(page):
    body_text = page.locator("body").inner_text()
    lines = [line.strip() for line in body_text.splitlines() if line.strip()]

    for line in lines:
        upper = line.upper()
        if "COURS -" in upper and any(upper.startswith(m + " ") for m in MONTHS):
            return line.split("COURS")[0].strip().upper()
    return ""


def scrape_current_table(page, conn):
    now = datetime.now().isoformat(timespec="seconds")
    maturity = extract_current_header_maturity(page)

    if not maturity:
        print("Maturité introuvable pour la table courante")
        return

    print(f"\n--- Scraping maturité: {maturity} ---")

    tables = page.locator("table")
    ntables = tables.count()
    if ntables < 4:
        print(f"Table principale introuvable. Nombre de tables = {ntables}")
        return

    # Dans ton diagnostic, la bonne table est la table 3
    table_text = tables.nth(3).inner_text()
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]

    if len(lines) <= 1:
        print("Table vide ou non exploitable")
        return

    saved = 0
    for line in lines[1:]:
        parts = line.split()

        # format attendu :
        # comp_call last_call bid_call ask_call C strike P bid_put ask_put last_put comp_put
        if len(parts) < 11:
            continue
        if parts[4] != "C":
            continue

        comp_call = to_float(parts[0])
        strike = to_float(parts[5])

        if comp_call is None or strike is None:
            continue

        save_option(conn, (
            now,
            "AD",
            "CALL",
            maturity,
            strike,
            comp_call
        ))
        saved += 1
        print(f"Saved CALL: maturity={maturity}, strike={strike}, comp={comp_call}")

    print(f"{saved} lignes sauvées pour {maturity}")


def set_maturity_and_submit(page, maturity_value, maturity_label):
    print(f"\nSélection de la maturité : {maturity_label} ({maturity_value})")

    # 1) changer directement la valeur du select HTML
    page.select_option("select", value=maturity_value)
    page.wait_for_timeout(1000)

    # 2) cliquer sur le bouton soumettre réel
    clicked = False
    buttons = page.locator("button, input[type='submit'], button[type='submit']")
    for i in range(buttons.count()):
        try:
            txt = buttons.nth(i).inner_text(timeout=500).strip().upper()
        except Exception:
            txt = ""
        try:
            val = buttons.nth(i).get_attribute("value")
            val = val.upper() if val else ""
        except Exception:
            val = ""

        if txt == "SOUMETTRE" or val == "SOUMETTRE":
            buttons.nth(i).click()
            clicked = True
            break

    if not clicked:
        # fallback JS
        page.evaluate("""
            () => {
                const els = [...document.querySelectorAll('button, input[type="submit"]')];
                const target = els.find(el =>
                    (el.innerText && el.innerText.trim().toUpperCase() === 'SOUMETTRE') ||
                    (el.value && el.value.trim().toUpperCase() === 'SOUMETTRE')
                );
                if (target) target.click();
            }
        """)

    page.wait_for_timeout(3000)

    # 3) contrôle
    header = extract_current_header_maturity(page)
    print("Header courant après submit :", header)


def extract_spot(page):
    bid = None
    last = None
    ask = None

    tables = page.locator("table")
    ntables = tables.count()

    for i in range(min(ntables, 3)):
        try:
            txt = tables.nth(i).inner_text()
        except:
            continue

        lines = [line.strip() for line in txt.splitlines() if line.strip()]

        for j, line in enumerate(lines):
            if line == "Achat":
                for k in range(j + 1, min(j + 4, len(lines))):
                    try:
                        bid = float(lines[k].replace(",", "."))
                        break
                    except:
                        pass

            if line == "Dernier":
                for k in range(j + 1, min(j + 4, len(lines))):
                    try:
                        last = float(lines[k].replace(",", "."))
                        break
                    except:
                        pass

            if line == "Vente":
                for k in range(j + 1, min(j + 4, len(lines))):
                    try:
                        ask = float(lines[k].replace(",", "."))
                        break
                    except:
                        pass

    return last, bid, ask

def main():
    conn = sqlite3.connect(DB_PATH)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        
        
        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_timeout(8000)

        dismiss_cookie_popup(page)
        spot, bid, ask = extract_spot(page)
        print("SPOT:", spot, "BID:", bid, "ASK:", ask)
        now = datetime.now().isoformat(timespec="seconds")

        if spot is not None:
            save_spot(conn, now, "AD", spot, bid, ask)

        maturity_options = get_maturity_options(page)
        print("Maturités trouvées :", maturity_options)

        if not maturity_options:
            print("Aucune maturité trouvée")
            browser.close()
            conn.close()
            return

        for label, value in maturity_options:
            try:
                set_maturity_and_submit(page, value, label)
                scrape_current_table(page, conn)
            except Exception as e:
                print(f"Erreur sur {label}: {e}")

        browser.close()

    conn.close()


if __name__ == "__main__":
    main()