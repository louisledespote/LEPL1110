import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

def run(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    print(f"Lancement de {script_name} ...")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        raise RuntimeError(f"Erreur dans {script_name}")

if __name__ == "__main__":
    run("clear_options.py")
    run("clear_underlying.py")
    run("collect_euronext.py")
    run("read_data.py")
    run("clean_export_daily.py")
    run("export_underlying.py")
    print("Pipeline journalier terminé")