# Roadmap — Juste des Ventilateurs

Projet M2 Data/IA — LaPlateforme_  
Version 1.2 — Juin 2026

---

## Vue d'ensemble

Le projet est organisé en 6 phases successives, chacune livrant des artefacts exploitables. Les phases 1 à 3 sont des fondations ; les phases 4 à 6 constituent le cœur ML et l'évaluation comparative.

```
Phase 1 : Prise en main          [Semaine 1]
Phase 2 : Ingestion & stockage   [Semaine 1-2]
Phase 3 : Feature engineering    [Semaine 2-3]
Phase 4 : Modèle prédictif       [Semaine 3-4]
Phase 5 : Contrôleur prescriptif [Semaine 4-5]
Phase 6 : Boucle fermée & éval   [Semaine 5-6]
```

---

## Phase 1 — Prise en main et compréhension ✅

**Objectif :** Comprendre l'environnement jumeaux-chauds et valider la connectivité.

### Tâches

- [x] Structure du projet créée (répertoires, packages Python, Docker)
- [x] Documentation initiale (README, roadmap, specifications)
- [x] Supervisor placeholder opérationnel (Docker tourne sans erreur)
- [x] Analyse complète des topics MQTT de jumeaux-chauds
  - Root : `dt/`, cluster : `cluster_alpha`
  - Topics : `.../telemetry` (QoS 0, 1/s), `.../status` (QoS 1), `.../fault` (QoS 1), `.../summary` (QoS 1)
- [x] Analyse des endpoints REST de pilotage des ventilateurs
- [x] Identification des seuils thermiques par rôle (base.yaml)
- [ ] `notebooks/01_exploration.ipynb` : exploration MQTT et API interactive

### Livrables ✅
- Structure du projet, Docker, supervisor placeholder
- `documents/roadmap.md`, `documents/specifications.md`, `README.md`
- `.env.example`, `build-clean-app.bat`

---

## Phase 2 — Ingestion et stockage des données ✅

**Objectif :** Collecter en continu la télémétrie et constituer des datasets reproductibles.

### Tâches

- [x] `ingest/mqtt_subscriber.py` : subscriber MQTT async avec reconnexion automatique (backoff exponentiel)
  - Souscription : `dt/cluster_alpha/+/telemetry`, `.../status`, `.../fault`, `.../summary`
  - Mode CLI : `--duration N` (épisode borné) ou `--continuous` (daemon)
- [x] `ingest/normalizer.py` : parsing et normalisation des payloads
  - Schéma unifié : timestamp, cluster_id, machine_id, role, status, temperature_c, power_w, fan_rpm_mean/std, load_estimated, has_fault, fault_types...
  - Gestion des 4 types de messages : telemetry, status_event, fault_event, cluster_summary
- [x] `ingest/dataset_exporter.py` : export Parquet partitionné par machine et épisode
  - Fallback CSV si pandas/pyarrow non disponibles
  - `metadata.json` par épisode (scenario, seed, durée, n_records)
- [x] `tests/test_ingest.py` : 19 tests unitaires (Normalizer + DatasetExporter)
- [x] `data/schema.md` : schéma unifié documenté
- [ ] `notebooks/01_exploration.ipynb` : collecte interactive et visualisation

### Livrables ✅
- `ingest/mqtt_subscriber.py`, `ingest/normalizer.py`, `ingest/dataset_exporter.py`
- `ingest_mqtt_simulations.bat` : collecte automatisée des 6 scénarios à x60
- `ingest_gen_features.bat` : feature engineering en batch sur tous les épisodes
- `tests/test_ingest.py`
- `data/schema.md`

### Commandes de collecte
```bash
# Collecte multi-scenarios automatisee (recommande)
ingest_mqtt_simulations.bat

# Collecte manuelle d'un episode
python -m ingest.mqtt_subscriber --duration 600 --episode 001 --scenario nominal

# Collecte continue
python -m ingest.mqtt_subscriber --continuous --episode 001 --scenario stress
```

---

## Phase 3 — Feature engineering

**Objectif :** Construire des features pertinentes pour la prédiction de pannes et le contrôle.

### Tâches

- [x] `features/temporal.py` : dérivées de température (5s/15s/30s), rolling means, marge au shutdown, RPM variance/CV
- [x] `features/contextual.py` : durée en zone chaude, compteurs shutdowns/degraded, indicateurs de pannes, changements de consigne RPM, flag récupération
- [x] `features/energy.py` : puissance fans (loi cubique RPM³), fan_energy_ratio, pue_estimated, energy_fans_kwh_cumulated
- [x] `features/labeler.py` : failure_60s / failure_30s / hot_30s (forward-looking), time_to_failure_s, optimal_rpm / action_class
- [x] `features/pipeline.py` : pipeline complet CLI, traitement multi-machine, export Parquet
- [x] `tests/test_features.py` : 28 tests unitaires
- [ ] `notebooks/02_feature_analysis.ipynb` : analyse exploratoire interactive

### Livrables ✅
- `features/temporal.py`, `features/contextual.py`, `features/energy.py`
- `features/labeler.py`, `features/pipeline.py`
- `tests/test_features.py`

### Commandes
```bash
# Feature engineering en batch (tous les episodes)
ingest_gen_features.bat

# Episode specifique
ingest_gen_features.bat 003

# Manuel
python -m features.pipeline \
  --input data/raw/episode=001 \
  --output data/processed/episode=001 \
  --config data/raw/episode=001/metadata.json

pytest tests/test_features.py -v
```

---

## Phase 4 — Modèle d'anticipation de pannes

**Objectif :** Prédire avec suffisamment d'avance les incidents thermiques.

### Module : `models/failure_prediction/`

### Tâches

- [ ] **Baseline heuristique** (`models/failure_prediction/baseline_threshold.py`)
  - Règle : si `temperature_c > T_warn` depuis N secondes → panne probable
  - Paramètres : `T_warn` et `N` optimisés sur les données d'entraînement
- [ ] **Modèle 1 : Régression Logistique** (`models/failure_prediction/logistic_regression.py`)
  - Entraînement, calibration, seuil optimal
- [ ] **Modèle 2 : Random Forest** (`models/failure_prediction/random_forest.py`)
  - Importance des features, profondeur optimale
- [ ] **Modèle 3 : Gradient Boosting** (`models/failure_prediction/gradient_boosting.py`)
  - XGBoost ou LightGBM, early stopping, tuning
- [ ] **Évaluation comparative** (`evaluation/failure_prediction_eval.py`)
  - Métriques : Precision, Recall, F1, PR-AUC, ROC-AUC
  - **Métrique clé : temps moyen d'anticipation avant incident** (en secondes)
  - Analyse des faux négatifs (incidents non détectés)
  - Validation : séparation train/test par épisodes, test sur plusieurs seeds
- [ ] Sauvegarde des modèles : `models/failure_prediction/saved/` (joblib ou ONNX)
- [ ] Notebook : `notebooks/03_failure_prediction.ipynb`

### Métriques cibles
- Recall sur cas dangereux ≥ 0.85
- Temps moyen d'anticipation ≥ 30s avant incident
- F1-score > baseline heuristique

### Livrables
- `models/failure_prediction/` : code + modèles sauvegardés
- `evaluation/results/failure_prediction_results.json`
- `notebooks/03_failure_prediction.ipynb`

---

## Phase 5 — Contrôleur de régulation des ventilateurs

**Objectif :** Définir une politique d'actionnement des ventilateurs, sûre et sobre en énergie.

### Module : `models/fan_control/`

### Espace d'actions
Actions discrétisées : `RPM ∈ {0, 1500, 2500, 3500, 4500}` par ventilateur.

### Tâches

- [ ] **Baseline 1 : Ventilateur fixe** (`models/fan_control/baseline_fixed.py`)
  - RPM constant (plusieurs valeurs testées : 0, 2500, 4500)
- [ ] **Baseline 2 : Contrôle à seuils** (`models/fan_control/baseline_threshold.py`)
  - `if T > T1: RPM = high; elif T > T2: RPM = medium; else: RPM = low`
- [ ] **Baseline 3 : PID simple** (`models/fan_control/baseline_pid.py`)
  - Contrôleur proportionnel-intégral-dérivé, cible de température configurable
- [ ] **Contrôleur ML supervisé** (`models/fan_control/supervised_controller.py`)
  - Classifier qui apprend la "meilleure action" à partir des données offline
  - Labels d'action générés par simulation ou expert (baseline optimisée)
- [ ] **Contrôleur à score multi-objectif** (`models/fan_control/score_controller.py`)
  - Fonction de coût : `J(t) = α·risk(t) + β·heat(t) + γ·energy(t) + δ·|ΔRPM_t|`
  - Paramètres α, β, γ, δ optimisés par grid search ou Bayesian optimization
- [ ] **(Optionnel avancé) Bandit contextuel** (`models/fan_control/contextual_bandit.py`)
  - LinUCB ou similaire, exploration/exploitation adaptative
- [ ] **Évaluation comparative** (`evaluation/fan_control_eval.py`)
  - Métriques : nb shutdowns, température moyenne, énergie consommée, nb incidents évités
- [ ] Notebook : `notebooks/04_fan_control.ipynb`

### Métriques cibles
- Réduction du nombre de shutdowns vs baseline auto native ≥ 50%
- Consommation énergétique des fans ≤ baseline "full speed" (économie ≥ 20%)

### Livrables
- `models/fan_control/` : code + modèles sauvegardés
- `evaluation/results/fan_control_results.json`
- `notebooks/04_fan_control.ipynb`

---

## Phase 6 — Boucle fermée et évaluation comparative

**Objectif :** Coupler prédicteur + contrôleur en temps réel et mesurer l'impact réel.

### Tâches

- [ ] **Service de supervision** (`supervisor/supervisor.py`)
  - Boucle : lire état MQTT → évaluer risque (modèle prédictif) → décider action (contrôleur) → envoyer commande REST → logger
  - Fréquence de décision configurable (ex: toutes les 5s)
  - Passage des machines en mode `manual` avant activation du contrôleur
  - L'arrêt thermique automatique reste ACTIF (événement d'échec à éviter)
- [ ] **Logger de décisions** (`supervisor/decision_logger.py`)
  - Enregistrement de : timestamp, machine, état, risque prédit, action choisie, résultat observé
- [ ] **Protocole d'évaluation** (`evaluation/benchmark.py`)
  - Scénario unique, durée fixe, plusieurs runs :
    1. **Baseline native** : mode auto jumeaux-chauds, aucune intervention
    2. **Baseline seuils** : contrôleur à seuils externe
    3. **Couple ML** : prédicteur + contrôleur à score
  - Métriques globales : nb shutdowns, T_mean, T_max, énergie totale, énergie fans
- [ ] **Test de robustesse** (`evaluation/robustness.py`)
  - Variation de charge, bruit capteur augmenté, drift rapide, pannes fréquentes
  - Généralisation : tester sur scénarios `nominal`, `stress`, `heatwave`, `busy_weeks`
- [ ] **Rapport final** (`documents/rapport_analyse.md`)
  - Résultats chiffrés, graphiques comparatifs, analyse critique
- [ ] Notebook final : `notebooks/05_evaluation_comparative.ipynb`

### Livrables
- `supervisor/` : service de supervision temps réel
- `evaluation/` : protocole complet et résultats
- `documents/rapport_analyse.md`
- `notebooks/05_evaluation_comparative.ipynb`

---

## Récapitulatif des livrables

| Livrable | Phase | Priorité |
|----------|-------|----------|
| `ingest/` : subscriber MQTT + normaliser | 2 | Essentielle |
| `data/` : datasets reproductibles | 2-3 | Essentielle |
| `features/` : pipeline de features | 3 | Essentielle |
| `models/failure_prediction/` : 3 modèles + baseline | 4 | Essentielle |
| `models/fan_control/` : 3+ contrôleurs | 5 | Essentielle |
| `supervisor/` : boucle fermée temps réel | 6 | Essentielle |
| `evaluation/` : protocole + résultats | 4-6 | Essentielle |
| `documents/rapport_analyse.md` | 6 | Essentielle |
| `docker-compose.yml` | 2 | Recommandée |
| Bandit contextuel | 5 | Optionnelle |

---

## Couples prédicteur / contrôleur à comparer

| # | Prédicteur | Contrôleur | Objectif |
|---|-----------|------------|----------|
| 0 | Aucun (natif) | Auto jumeaux-chauds | Baseline de référence |
| 1 | Heuristique seuil | Seuils fixes | Baseline règles simples |
| 2 | Régression Logistique | PID | Approche classique |
| 3 | Random Forest | Score multi-objectif | Recommandé |
| 4 | Gradient Boosting | Score multi-objectif | Meilleure performance attendue |
| 5 | Gradient Boosting | Bandit contextuel | Option avancée |
