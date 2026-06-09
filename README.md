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
jumeaux-chauds (simulateur)          juste-des-ventilateurs (ce projet)
┌─────────────────────────┐         ┌──────────────────────────────────┐
│  MQTT :1883             │◄────────│  ingest/ : subscriber MQTT       │
│  REST API :8000         │◄────────│  supervisor/ : commandes REST    │
│  (fans, machines)       │         │                                  │
└─────────────────────────┘         │  features/ : feature engineering │
                                    │  models/failure_prediction/      │
                                    │  models/fan_control/             │
                                    │  evaluation/ : benchmark         │
                                    └──────────────────────────────────┘
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
│   ├── supervisor.py           # Boucle de décision principale
│   └── decision_logger.py      # Logger des décisions et résultats
│
├── evaluation/                 # Protocole et métriques comparatives
│   ├── benchmark.py            # Comparaison des couples prédicteur/contrôleur
│   └── robustness.py           # Tests de robustesse multi-scénarios
│
├── notebooks/                  # Analyses et explorations
│   ├── 01_exploration.ipynb    # Exploration MQTT et API
│   ├── 02_feature_analysis.ipynb
│   ├── 03_failure_prediction.ipynb
│   ├── 04_fan_control.ipynb
│   └── 05_evaluation_comparative.ipynb
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

```bash
# Entraînement + évaluation comparative (tous les modèles, label failure_60s)
train_models.bat

# Label spécifique
train_models.bat failure_30s

# Analyse exploratoire rapide (volumes de split, labels, corrélations)
python ingest_quick_EDA.py --processed-only
```

### Lancer le superviseur

```bash
python -m supervisor.supervisor --predictor gradient_boosting --controller score_controller
```

### Via Docker

```bash
docker compose up -d
# Le superviseur se connecte automatiquement à jumeaux-chauds
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

## Documentation

- [Roadmap](documents/roadmap.md) — Phases et livrables
- [Spécifications techniques](documents/specifications.md) — Architecture détaillée, schémas, interfaces
- [Rapport d'analyse](documents/rapport_analyse.md) — Résultats et conclusions (Phase 6)

---

## Ressources

- [jumeaux-chauds](https://github.com/TristanV/jumeaux-chauds) — Jumeau numérique
- [Sujet M2](https://docs.google.com/document/d/1c4AP-FDV5l1hVE1tYLIS8y81e29oh9Y1Y6KCIY_kFBg) — Description complète du projet
- [MQTT Guide](https://mqtt.org/) | [scikit-learn](https://scikit-learn.org/) | [XGBoost](https://xgboost.readthedocs.io/)
