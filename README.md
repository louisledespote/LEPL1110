#  Projet LEPL1110 – Black–Scholes & Éléments Finis

## Objectif du projet

Ce projet vise à résoudre l’équation de Black–Scholes à l’aide de la méthode des éléments finis (FEM), et à comparer les résultats avec les données de marché réelles.

Le projet est structuré en deux parties :

---

#  PARTIE 1 — Black–Scholes avec volatilité constante

## Objectifs

- Implémenter le modèle Black–Scholes (PDE)
- Résoudre l’équation avec FEM
- Comparer :
  - solution analytique
  - solution FEM
  - prix de marché
- Estimer la volatilité σ à partir des données (volatilité implicite)

---

## Équation étudiée

\[
\frac{\partial V}{\partial t}
+ \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+ r S \frac{\partial V}{\partial S}
- rV = 0
\]

---

## Travail à faire

### 🔹 1. Adapter le code FEM existant

Le code actuel résout une équation de diffusion 1D :

👉 `diffusion/main_diffusion_1d.py`

À modifier :

- passer de :

👉 modifier `stiffness.py` pour ajouter :

- diffusion :
\[
\frac{1}{2}\sigma^2 S^2
\]

- convection :
\[
r S \frac{\partial V}{\partial S}
\]

- réaction :
\[
-rV
\]

---

### 🔹 2. Ajouter la dimension temporelle

Utiliser le schéma en temps déjà implémenté (`theta_step`).

À faire :
- boucle en temps
- condition initiale (payoff)

\[
V(S,0) = \max(S-K,0)
\]

---

### 🔹 3. Conditions aux limites

- \(S=0\) → \(V=0\)
- \(S=S_{max}\) →
\[
V(S,t) = S - K e^{-r(T-t)}
\]

---

### 🔹 4. Comparaison analytique

Implémenter la formule fermée Black–Scholes

Comparer :
- FEM vs analytique
- erreur en fonction du maillage

---

### 🔹 5. Estimation de la volatilité σ

À partir des données collectées :

- lire CSV (`data/`)
- pour chaque option :
- calculer σ implicite
- construire :
- σ(K)
- σ moyenne

---

### 🔹 6. Comparaison avec le marché

Comparer :
- prix FEM
- prix Black–Scholes
- prix marché (Euronext)

---

#  PARTIE 2 — Black–Scholes non linéaire

## Objectif

Rendre le modèle plus réaliste :

\[
\sigma = \sigma(S,t)
\]

---

## Approche proposée

### Option simple

\[
\sigma(S) = \sigma_0 (1 + \alpha S)
\]

---

## Modifications nécessaires

- remplacer σ constante dans `stiffness.py`
- recalculer diffusion :

\[
\kappa(S) = \frac{1}{2} \sigma(S)^2 S^2
\]

---

## Intérêt

- plus de solution analytique ❌
- FEM devient indispensable ✅

---

# 📁 Structure du projet

projet/
│
├── data/
│ ├── raw/
│ ├── cleaned/
│ ├── daily_clean/
│
├── scripts/
│ ├── collect_euronext.py
│ ├── clean_export_daily.py
│ ├── run_daily.py
│ ├── export_csv.py
│
├── diffusion/
│ ├── main_diffusion_1d.py
│ ├── stiffness.py
│ ├── mass.py
│ ├── dirichlet.py
│ ├── gmsh_utils.py
│ ├── plot_utils.py
│ ├── errors.py
│
├── README.md
