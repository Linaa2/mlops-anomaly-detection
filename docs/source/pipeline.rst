Pipeline ML — Condition Monitoring Hydraulique
===============================================

1. Ingestion des données
-------------------------

**Source :** UCI – *Condition monitoring of hydraulic systems* (2205 cycles × 17 capteurs)

Chaque fichier capteur est une matrice ``(2205, N)`` où chaque ligne est un cycle de 60 s.
L'agrégation par **moyenne temporelle** produit un vecteur de 17 features par cycle.

.. list-table:: Capteurs
   :header-rows: 1

   * - Groupe
     - Capteurs
     - Fréquence
   * - Pression
     - PS1–PS6
     - 100 Hz
   * - Puissance moteur
     - EPS1
     - 100 Hz
   * - Débit
     - FS1, FS2
     - 10 Hz
   * - Température
     - TS1–TS4
     - 1 Hz
   * - Vibration
     - VS1
     - 1 Hz
   * - Indicateurs calculés
     - CE, CP, SE
     - 1 Hz

2. Prétraitement
-----------------

- **Filtrage** des cycles instables (``stable_flag = 1``) — 756 cycles supprimés, 1449 conservés
- **Labels multi-output** issus de ``profile.txt`` :

.. list-table::
   :header-rows: 1

   * - Target
     - Classes
     - Interprétation
   * - ``cooler_condition``
     - 3, 20, 100
     - panne / réduit / ok
   * - ``valve_condition``
     - 73, 80, 90, 100
     - panne / lag sévère / lag léger / ok
   * - ``pump_leakage``
     - 0, 1, 2
     - aucune / faible / sévère
   * - ``accumulator_pressure``
     - 90, 100, 115, 130
     - panne / sévère / réduit / ok

3. Modèle
----------

``MultiOutputClassifier(RandomForestClassifier)`` — scikit-learn

- Pas de normalisation (arbres, seuils relatifs)
- Split 80/20 stratifié sur ``cooler_condition``
- Un classifieur indépendant par composant

4. Métriques
-------------

- **F1 macro** : métrique principale, traite toutes les classes à égalité
- **Precision / Recall** : recall critique en maintenance (faux négatif = panne non détectée)
- **Matrice de confusion** : identifier les classes confondues entre elles

5. Résultats
-------------

.. list-table::
   :header-rows: 1

   * - Composant
     - Accuracy
     - F1 macro
   * - ``cooler_condition``
     - 100%
     - 1.000
   * - ``pump_leakage``
     - 99.3%
     - 0.993
   * - ``accumulator_pressure``
     - 99.0%
     - 0.990
   * - ``valve_condition``
     - 94.8%
     - 0.947

La confusion principale est entre les états **90** (lag léger) et **100** (ok) de la valve —
physiquement proches, signaux capteurs similaires.
