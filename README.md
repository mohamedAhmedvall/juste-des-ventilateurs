# Juste des Ventilateurs

**Service de maintenance prédictive et de régulation thermique** pour datacenter simulé.

Projet M2 Data/IA — LaPlateforme_ | Couplé à [jumeaux-chauds](https://github.com/TristanV/jumeaux-chauds)

---

## Présentation

*"Quand les choses se réchauffent et que les capteurs s'agitent, dans un système qui contrôle juste des ventilateurs, notre IA est activée pour refroidir les choses et faire baisser la pression."*

Ce projet implémente un service intelligent de supervision thermique qui s'appuie sur le jumeau numérique **jumeaux-chauds** pour :

1. **Anticiper** les pannes (surchauffe, mode dégradé, arrêt thermique) par machine learning
2. **Piloter** les ventilateurs de façon optimale (sécurité + sobriété énergétique)
3. **Évaluer** et comparer plusieurs couples (modèle prédictif, contrôleur prescriptif)

---

## Architecture

```
jumeaux-chauds (simulateur)
┌─────────────────────────────────────────┐
│  MQTT :1883                             │
│    dt/{cluster}/{machine}/telemetry  ───┼──► supervisor/mqtt_telemetry.py
│    dt/{cluster}/{machine}/status/fault ─┼──► ingest/mqtt_subscriber.py
│  REST API :8000                         │
│    GET /cluster/status               ◄──┼─── supervisor (fallback + résumé cycle)
│    PUT /machines/{id}/fan_speed       ◄──┼─── supervisor (commandes)
│    PUT /machines/{id}/fan_mode        ◄──┼─── supervisor (init manual/auto)
└─────────────────────────────────────────┘

juste-des-ventilateurs
┌──────────────────────────────────────────────────────────────┐
│  supervisor/mqtt_telemetry.py  ← abonné MQTT télémétrie      │
│    (1 msg/s simulé, correct à toute vitesse de simulation)   │
│         │                                                    │
│         ▼                                                    │
│  supervisor/online_features.py ← OnlineFeatureBuffer         │
│    (fenêtres glissantes 5/15/30/60s alignées entraînement)   │
│         │  (décision tous les decision_interval_ticks ticks) │
│         ▼                                                    │
│  supervisor/supervisor.py      ← boucle de décision          │
│    predict risk → decide RPM → PUT fan_speed                 │
│         │                                                    │
│  ingest/         → collecte dataset (MQTT subscriber)        │
│  features/       → feature engineering hors-ligne            │
│  models/         → prédicteur de pannes + contrôleur fans    │
│  evaluation/     → benchmark offline comparatif              │
└──────────────────────────────────────────────────────────────┘
```

---

## Structure du projet

```
juste-des-ventilateurs/
├── ingest/                     # Collecte MQTT et normalisation ✅
│   ├── mqtt_subscriber.py      # Subscriber MQTT async (reconnexion auto, backoff)
│   ├── normalizer.py           # Parsing payload → schéma unifié (4 types de messages)
│   └── dataset_exporter.py     # Export Parquet partitionné par machine/épisode
│
├── features/                   # Ingénierie des features temporelles ✅
│   ├── temporal.py             # Dérivées de température, rolling means, marge shutdown
│   ├── contextual.py           # Durée zone chaude, compteurs incidents, pannes actives
│   ├── energy.py               # Puissance fans (loi cubique), PUE, ratio énergie
│   ├── labeler.py              # Labels failure_60s/30s, hot_30s, action_class (oracle)
│   └── pipeline.py             # Pipeline complet : brut → features + labels (CLI)
│
├── models/
│   ├── failure_prediction/     # Modèles d'anticipation de pannes
│   │   ├── baseline_threshold.py
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── gradient_boosting.py
│   │   └── saved/              # Modèles sérialisés (joblib/ONNX)
│   └── fan_control/            # Contrôleurs de régulation
│       ├── baseline_fixed.py
│       ├── baseline_threshold.py
│       ├── baseline_pid.py
│       ├── supervised_controller.py
│       ├── score_controller.py
│       └── saved/
│
├── supervisor/                 # Service de supervision temps réel
│   ├── supervisor.py           # Boucle de décision (predict → decide → command)
│   ├── online_features.py      # OnlineFeatureBuffer — fenêtres glissantes par machine ✅
│   ├── mqtt_telemetry.py       # Consumer MQTT télémétrie (Phase 7) 🔲
│   └── decision_logger.py      # Logger JSONL des décisions
│
├── evaluation/                 # Protocole et métriques comparatives
│   ├── benchmark.py            # Comparaison offline des contrôleurs
│   ├── robustness.py           # Tests de robustesse multi-scénarios
│   ├── fan_control_eval.py     # Évaluation comparative (oracle v1 vs v2)
│   └── closed_loop_eval.py     # Évaluation boucle fermée + PUE (Phase 9) ✅
│
├── notebooks/                  # Analyses et explorations
│   ├── 01_ingestion_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_failure_prediction.ipynb
│   ├── 04_fan_control.ipynb
│   ├── 05_evaluation_comparative.ipynb
│   ├── 06_phase7_mqtt_supervision.ipynb
│   └── 07_closed_loop_evaluation.ipynb  # Phase 9 ✅
│
├── data/                       # Datasets (ignorés par git, sauf schéma)
│   ├── schema.md               # Description du schéma unifié
│   ├── raw/                    # Données brutes collectées
│   └── processed/              # Datasets avec features et labels
│
├── documents/                  # Documentation du projet
│   ├── roadmap.md              # Feuille de route par phases
│   ├── specifications.md       # Spécifications techniques détaillées
│   └── rapport_analyse.md      # Rapport final (Phase 6)
│
├── tests/                      # Tests unitaires
├── docker-compose.yml          # Orchestration (branchement au simulateur)
├── Dockerfile
├── requirements.txt
├── setup.py
└── .env.example                # Variables d'environnement
```

---

## Prérequis

- **jumeaux-chauds** lancé et accessible (MQTT :1883, API :8000)
- Python 3.11+
- Docker & Docker Compose (pour le déploiement)

---

## Installation

```bash
# 1. Cloner ce dépôt
git clone https://github.com/TristanV/juste-des-ventilateurs.git
cd juste-des-ventilateurs

# 2. Environnement Python
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

pip install -r requirements.txt

# 3. Configuration
cp .env.example .env
# Éditer .env selon votre configuration jumeaux-chauds
```

---

## Démarrage rapide

### Lancer jumeaux-chauds (prérequis)

```bash
cd ../jumeaux-chauds
docker compose up -d
# Vérifier que l'API est accessible : curl http://localhost:8000/cluster/status
```

### Collecter des données

```bash
# Collecte automatisée des 6 scénarios à x60 (recommandé)
ingest_mqtt_simulations.bat

# Collecte manuelle d'un épisode avec scénario explicite
python -m ingest.mqtt_subscriber --duration 600 --episode 001 --scenario nominal

# Collecte continue
python -m ingest.mqtt_subscriber --continuous --episode 001 --scenario stress
```

### Lancer les tests

```bash
pytest tests/ -v
```

### Construire les features

```bash
# Feature engineering en batch sur tous les épisodes (recommandé)
ingest_gen_features.bat

# Épisode spécifique
ingest_gen_features.bat 003

# Manuel
python -m features.pipeline \
  --input data/raw/episode=001 \
  --output data/processed/episode=001 \
  --config data/raw/episode=001/metadata.json
```

### Entraîner et évaluer les modèles de prédiction de pannes

Trois labels de prédiction sont disponibles :

| Label | Horizon | Description |
|-------|---------|-------------|
| `failure_60s` | 60 s | Panne dans la minute — **utilisé par le superviseur** |
| `failure_30s` | 30 s | Panne dans les 30 secondes (plus précis, moins de préavis) |
| `hot_30s`     | 30 s | Température > 95 % du seuil (alerte thermique pure) |

```bash
# Option recommandée : entraîner et benchmarker les 3 labels d'un coup
run_all_labels.bat

# Ou script par script, label par label :
03_train_models.bat                  # failure_60s par défaut
03_train_models.bat failure_30s
03_train_models.bat hot_30s

04_train_fan_controllers.bat         # contrôleurs (toujours sur failure_60s)

05_benchmark_offline_metrics.bat              # benchmark failure_60s
05_benchmark_offline_metrics.bat failure_30s  # benchmark failure_30s
05_benchmark_offline_metrics.bat hot_30s      # benchmark hot_30s

# Analyse exploratoire rapide (volumes de split, labels, corrélations)
python ingest_quick_EDA.py --processed-only
```

Les résultats sont produits dans `evaluation/results/` avec le label dans le nom :
```
failure_prediction_results_failure_60s.json
failure_prediction_results_failure_30s.json
failure_prediction_results_hot_30s.json
benchmark_results_failure_60s.json
robustness_results_failure_60s.json
fan_control_results_failure_60s.json
...
```

Pour visualiser et comparer les résultats, ouvrir `notebooks/05_evaluation_comparative.ipynb`
et modifier la variable `LABEL` en tête de notebook.

### Lancer le superviseur

```bash
# Mode ML (prédicteur logistique + contrôleur supervisé, recommandé)
python -m supervisor.supervisor --mode ml

# Avec durée limitée et dry-run (test sans commandes)
python -m supervisor.supervisor --mode ml --duration 300 --dry-run

# Via Docker (jumeaux-chauds doit tourner sur le même hôte)
docker compose up --build supervisor
```

**Variables d'environnement clés (`.env`) :**
```
API_BASE_URL=http://localhost:8000
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
DECISION_INTERVAL_TICKS=5     # décision toutes les 5 secondes simulées
RISK_LOG_THRESHOLD=0.05       # log machine seulement si risk > seuil
```

---

## Couples prédicteur / contrôleur comparés

| # | Prédicteur | Contrôleur | Description |
|---|-----------|------------|-------------|
| 0 | Aucun | Auto (natif) | Baseline de référence |
| 1 | Seuils | Seuils fixes | Baseline règles simples |
| 2 | Régression Logistique | PID | Approche classique |
| 3 | Random Forest | Score multi-objectif | **Recommandé** |
| 4 | Gradient Boosting | Score multi-objectif | Meilleure performance attendue |
| 5 | Gradient Boosting | Bandit contextuel | Option avancée |

---

## Métriques d'évaluation

- **Sécurité thermique** : nombre de shutdowns, épisodes dégradés évités
- **Anticipation** : temps moyen d'alerte avant incident (lead time, en secondes)
- **Efficacité énergétique** : consommation totale des fans (kWh), ratio PUE
- **Qualité ML** : Precision, Recall, F1, PR-AUC (modèle prédictif)

---

## Évaluation en boucle fermée — Phase 9

Les évaluations offline (phases 5-6) rejoueront des données historiques sans modifier la simulation. La **Phase 9** pilote effectivement jumeaux-chauds en temps réel et mesure l'impact causal de chaque contrôleur.

### Pourquoi la boucle fermée ?

Dans l'évaluation offline, `nb_shutdowns` et `T_mean` sont identiques pour tous les contrôleurs — ils correspondent aux valeurs du scénario enregistré. La boucle fermée laisse la physique du simulateur recalculer les températures en réponse aux RPM commandés, ce qui permet de mesurer :

- **Pannes évitées** : shutdowns qui n'auraient pas eu lieu avec un meilleur contrôle (distinctes des pannes inévitables causées par `fan_failure` active)
- **PUE réel** : `(power_compute + power_fans) / power_compute`, calculé tick par tick
- **Économie énergétique** : kWh fans économisés par rapport au baseline full-speed

### Lancer une évaluation boucle fermée

```bash
# Prérequis : jumeaux-chauds en cours d'exécution
docker compose -f ../jumeaux-chauds/docker-compose.yml up -d

# Comparer les contrôleurs sur le scénario stress
# (--speed accélère la simulation ; --dt = secondes simulées entre décisions)
python -m evaluation.closed_loop_eval \
  --scenario stress --duration 300 --dt 5 --speed 60 \
  --controllers native supervised score_controller baseline_pid baseline_fixed_4500

# Scénario heatwave (montée T progressive)
python -m evaluation.closed_loop_eval --scenario heatwave --duration 300 --speed 60

# Résultats :
#   evaluation/results/closed_loop_results_stress.json
#   evaluation/results/closed_loop_results_heatwave.json

# Visualisation comparative (PUE, pannes, Pareto sécurité/énergie)
jupyter notebook notebooks/07_closed_loop_evaluation.ipynb
```

### Métriques Phase 9

| Métrique | Description |
|---------|-------------|
| `nb_shutdowns_cl` | Arrêts thermiques en boucle fermée |
| `nb_avoidable_avoided` | Pannes évitables réellement évitées vs natif |
| `pue_mean` | PUE moyen sur l'épisode (dérivé de `energy_kwh_cumulated`) |
| `energy_fans_kwh` | Énergie fans cumulée |
| `energy_saved_vs_max_pct` | Économie vs ventilation à fond 4500 RPM (%) |

---

## Documentation

- [Roadmap](documents/roadmap.md) — Phases et livrables (Phases 1–9)
- [Spécifications techniques](documents/specifications.md) — Architecture, interfaces, métriques Phase 9
- [Rapport d'analyse](documents/rapport_analyse.md) — Résultats et conclusions (Phase 6)

---

## Ressources

- [jumeaux-chauds](https://github.com/TristanV/jumeaux-chauds) — Jumeau numérique
- [Sujet M2](https://docs.google.com/document/d/1c4AP-FDV5l1hVE1tYLIS8y81e29oh9Y1Y6KCIY_kFBg) — Description complète du projet
- [MQTT Guide](https://mqtt.org/) | [scikit-learn](https://scikit-learn.org/) | [XGBoost](https://xgboost.readthedocs.io/)
