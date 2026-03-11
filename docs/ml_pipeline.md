# Pipeline ML — Condition Monitoring Hydraulique

## 1. Ingestion des données

**Source :** UCI – *Condition monitoring of hydraulic systems* (2205 cycles × 17 capteurs)

Chaque fichier capteur est une matrice `(2205, N)` où chaque ligne est un cycle de 60 s et chaque colonne un échantillon temporel (fréquence variable selon le capteur : 1 Hz à 100 Hz).

**Capteurs retenus :**

| Groupe | Capteurs | Fréquence |
|--------|----------|-----------|
| Pression | PS1–PS6 | 100 Hz |
| Puissance moteur | EPS1 | 100 Hz |
| Débit | FS1, FS2 | 10 Hz |
| Température | TS1–TS4 | 1 Hz |
| Vibration | VS1 | 1 Hz |
| Indicateurs calculés | CE, CP, SE | 1 Hz |

**Agrégation :** chaque cycle est résumé par la **moyenne temporelle** de chaque capteur → vecteur de 17 features par cycle. Ce choix de simplicité est suffisant ici car la variance intra-cycle est faible et les signaux sont stationnaires sur 60 s.

---

## 2. Prétraitement

**Filtrage des cycles instables :** les cycles avec `stable_flag = 1` sont retirés (756 cycles supprimés, 1449 conservés). Ces cycles correspondent à des transitoires système non représentatifs du comportement en régime établi.

**Labels (multi-output) :** 4 variables cibles issues de `profile.txt` :

| Target | Classes | Interprétation |
|--------|---------|----------------|
| `cooler_condition` | 3, 20, 100 | panne / réduit / ok |
| `valve_condition` | 73, 80, 90, 100 | panne / lag sévère / lag léger / ok |
| `pump_leakage` | 0, 1, 2 | aucune / faible / sévère |
| `accumulator_pressure` | 90, 100, 115, 130 | panne / sévère / réduit / ok |

**Choix de formulation :** classification multi-output supervisée plutôt que détection d'anomalies non-supervisée. Une approche binaire globale (`anomalie si au moins un composant dégradé`) a été écartée car elle produit 99.3% d'anomalies sur ce dataset — les chercheurs ayant conçu les expériences en faisant varier les états de chaque composant indépendamment, le système entièrement sain n'apparaît que dans 0.7% des cycles.

**Distribution des classes :** toutes les classes sont équilibrées (~360 exemples chacune), aucun rééchantillonnage nécessaire.

---

## 3. Modèle

**Architecture :** `MultiOutputClassifier(RandomForestClassifier)` — scikit-learn

**Justification :**
- Le Random Forest est robuste aux features non normalisées et gère nativement les classes non consécutives (ex: `{3, 20, 100}`)
- `MultiOutputClassifier` entraîne un classifieur indépendant par target, ce qui est adapté ici car les 4 composants ne sont pas corrélés dans leurs modes de défaillance
- Pas de one-hot encoding des sorties nécessaire : contrairement aux réseaux de neurones, les arbres traitent les labels entiers comme des classes catégorielles

**Hyperparamètres :** `n_estimators=100`, `random_state=42` (valeurs par défaut, non optimisées)

**Split :** 80% train / 20% test, stratifié sur `cooler_condition`

---

## 4. Métriques d'évaluation

- **F1 macro** : métrique principale — traite toutes les classes à égalité indépendamment de leur fréquence. Pertinent pour un contexte maintenance où détecter une panne rare est aussi important que détecter l'état normal.
- **Precision / Recall par classe** : le recall est particulièrement critique en maintenance prédictive — un faux négatif (panne non détectée) est plus coûteux qu'une fausse alarme.
- **Matrice de confusion** : permet d'identifier quelles classes sont confondues entre elles, ce qui a une signification physique (états voisins = signaux similaires).

---

## 5. Résultats

| Composant | Accuracy | F1 macro |
|-----------|----------|----------|
| `cooler_condition` | 100% | 1.000 |
| `pump_leakage` | 99.3% | 0.993 |
| `accumulator_pressure` | 99.0% | 0.990 |
| `valve_condition` | 94.8% | 0.947 |

**Analyse :**

- `cooler_condition` est parfaitement classifié : les 3 états (panne / réduit / ok) produisent des signatures thermiques et de pression très distinctes.
- `valve_condition` est le composant le plus difficile à diagnostiquer. La confusion se concentre entre les états **90** (lag léger) et **100** (ok) : 6 cycles sur 67 sont mal classés. Ces deux états sont physiquement proches — la valve fonctionne mais avec un léger retard, ce qui génère des signaux capteurs similaires à l'état nominal. Un feature engineering plus fin (std, percentiles) ou un modèle plus expressif pourrait réduire cette erreur.
- `pump_leakage` et `accumulator_pressure` affichent des F1 > 0.99 avec des confusions marginales uniquement entre états adjacents.

**Conclusion :** le pipeline de features agrégées + Random Forest est suffisant pour un baseline haute performance sur ce dataset. La marge d'amélioration réside principalement sur la `valve_condition`, candidat prioritaire pour une V2 avec feature engineering enrichi (domaine fréquentiel) ou un CNN 1D sur les séries brutes.
