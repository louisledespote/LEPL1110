# Black–Scholes FEM — Guide d’utilisation

Ce projet résout le modèle de Black–Scholes par éléments finis, compare la solution FEM à la solution analytique, puis calibre une volatilité constante sur des données de marché.

---

# Structure du projet

```text
PROJET/
├── data/                              # Données CSV marché
│
├── diffusion/                         # Code principal du projet
│   ├── main_black_scholes.py          # Lance une simulation FEM Black–Scholes
│   ├── precision_fem.py               # Génère les données d'étude de précision
│   ├── plot_precision.py              # Trace les graphes de précision FEM
│   ├── plot_callibration.py           # Trace les graphes de calibration
│   │
│   ├── calibration_script/            # Scripts de calibration
│   ├── calibration_results/           # Résultats CSV de calibration
│   ├── calibration_plots/             # Graphes de calibration
│   ├── precision_results/             # Résultats CSV de précision
│   ├── precision_plots/               # Graphes de précision
│   │
│   ├── stiffness.py                   # Assemblage opérateur Black–Scholes
│   ├── mass.py                        # Assemblage matrice de masse
│   ├── dirichlet.py                   # Conditions de Dirichlet + schéma theta
│   ├── gmsh_utils.py                  # Fonctions utilitaires Gmsh
│   ├── fem_eval.py                    # Évaluation de la solution FEM
│   ├── errors.py                      # Calcul d'erreurs numériques
│   ├── plot_utils.py                  # Fonctions de visualisation
│   ├── read_data_csv.py               # Lecture des données marché
│   └── panpan.msh                     # Maillage Gmsh
│
├── scripts/                           # Scripts auxiliaires
├── README.md
└── documents PDF                      # Ressources théoriques
```

---

# Installation

Créer un environnement virtuel :

```bash
python3 -m venv femvenv
source femvenv/bin/activate
```

Installer les dépendances :

```bash
pip install numpy scipy pandas matplotlib gmsh
```

---

# Lancer une simulation FEM Black–Scholes

Depuis le dossier `diffusion/`, utiliser :

```bash
python3 main_black_scholes.py \
    --options_csv ../data/daily_clean/2026-04-01_options.csv \
    --underlying_csv ../data/daily_clean/2026-04-01_underlying.csv \
    --maturity "MAY 2026" \
    --sigma 0.2 \
    --r 0.02 \
    -order 2 \
    -cl1 0.05 \
    -cl2 0.05 \
    --theta 1.0 \
    --nsteps 500
```

## Paramètres importants

| Paramètre | Description |
|---|---|
| `--maturity` | Maturité étudiée |
| `--sigma` | Volatilité |
| `--r` | Taux sans risque |
| `-order` | Ordre des éléments finis |
| `-cl1`, `-cl2` | Raffinement du maillage |
| `--theta` | Schéma temporel |
| `--nsteps` | Nombre de pas de temps |

## Schémas temporels

| Valeur de `theta` | Schéma |
|---|---|
| `1.0` | Euler implicite |
| `0.5` | Crank–Nicolson |

---

# Étude de précision numérique

Générer les données de convergence :

```bash
python3 precision_fem.py
```

Tracer les graphes :

```bash
python3 plot_precision.py
```

Les figures sont générées dans :

```text
precision_plots/
```

Graphes générés :

```text
comparaison_schemas_temporels.png
influence_ordre_semilogy.png
raffinement_maillage_loglog.png
```

---

# Calibration de la volatilité

## Calibration par maturité

```bash
python3 calibration_script/calibration_global_by_maturity.py
```

## Calibration par strike

```bash
python3 calibration_script/calibration_global_by_strike.py
```

Les résultats CSV sont stockés dans :

```text
calibration_results/
```

---

# Génération des graphes de calibration

```bash
python3 plot_callibration.py
```

Les figures sont générées dans :

```text
calibration_plots/
```

Graphes générés :

```text
mae_par_maturite.png
sigma_par_maturite.png
mae_par_strike.png
sigma_par_strike.png
```

---

# Workflow complet

Depuis le dossier `diffusion/` :

```bash
python3 precision_fem.py
python3 plot_precision.py

python3 calibration_script/calibration_global_by_maturity.py
python3 calibration_script/calibration_global_by_strike.py

python3 plot_callibration.py
```
