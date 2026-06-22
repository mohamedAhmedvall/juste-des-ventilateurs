# CLAUDE.md

Guide pour les assistants IA (Claude Code et autres) travaillant dans ce dépôt.

## Présentation du projet

**Juste des Ventilateurs** est un **service de maintenance prédictive et de
régulation thermique** pour un datacenter *simulé*. Il consomme la télémétrie du
jumeau numérique **jumeaux-chauds** (projet séparé), anticipe les pannes
(surchauffe / mode dégradé / arrêt thermique) par ML, et pilote les ventilateurs
de refroidissement vers un compromis optimal entre **sécurité thermique** et
**sobriété énergétique**.

- **La langue du domaine est le français.** Commentaires de code, docstrings,
  messages de commit et documentation sont rédigés en français. Respecte cette
  convention — garde les nouveaux commentaires et docs en français. Les
  identifiants (fonctions, variables) sont majoritairement en anglais.
- **Projet académique** (M2 Data/IA, LaPlateforme_). Organisé en *phases*
  numérotées ; voir `documents/roadmap.md`.
- Le service est **lecture-via-MQTT, commande-via-REST** : il ne fait jamais
  tourner la physique lui-même — c'est jumeaux-chauds qui possède la simulation.

## Prérequis & dépendance externe

Ce service ne sert à rien seul. **jumeaux-chauds doit être lancé** et accessible :
- Broker MQTT sur `:1883` — publie telemetry/status/fault/summary sur les topics
  `dt/{cluster}/{machine}/...`
- API REST sur `:8000` — `GET /cluster/status`, `PUT /machines/{id}/fan_speed`,
  `PUT /machines/{id}/fan_mode`

Python **3.11+** requis. La configuration est lue depuis des variables
d'environnement (`.env`, voir `.env.example`).

## Structure du dépôt

```
ingest/        Phase 2 — collecte MQTT → datasets Parquet normalisés
  mqtt_subscriber.py   subscriber async (reconnexion auto, backoff) · point d'entrée CLI
  normalizer.py        payload → schéma unifié (4 types de messages)
  dataset_exporter.py  export Parquet partitionné par épisode/machine
features/      Phase 3 — feature engineering hors-ligne
  temporal.py · contextual.py · energy.py   constructeurs de features
  labeler.py           failure_60s / failure_30s / hot_30s + action_class (oracle)
  pipeline.py          brut → features+labels · point d'entrée CLI
models/
  failure_prediction/  baseline_threshold, logistic_regression, random_forest,
                       gradient_boosting, splitter
  fan_control/         baseline_fixed/threshold/pid, supervised_controller,
                       score_controller
supervisor/    Service de supervision temps réel
  supervisor.py        boucle de décision (predict → decide → command) · point d'entrée CLI
  online_features.py   OnlineFeatureBuffer — fenêtres glissantes par machine
  mqtt_telemetry.py    consumer de télémétrie en direct
  decision_logger.py   journal JSONL des décisions
evaluation/    Benchmarks comparatifs hors-ligne + boucle fermée
  benchmark.py · robustness.py · fan_control_eval.py · failure_prediction_eval.py
  closed_loop_eval.py  Phase 9 — pilotage live de jumeaux-chauds (impact causal, PUE)
notebooks/     Analyses Jupyter 01..06 (ingestion, features, prédiction,
               contrôle, comparaison, supervision MQTT)
data/          datasets (ignorés par git sauf schema.md) — raw/ et processed/
documents/     roadmap.md, specifications.md, rapport_analyse.md
tests/         suite pytest (par phase)
*.bat          runners de workflow Windows (01..05, voir plus bas)
```

> **État Phase 9 :** `evaluation/closed_loop_eval.py` **existe désormais**
> (évaluation boucle fermée + `tests/test_phase9_closed_loop.py` et
> `tests/test_phase9b_methodology.py`). En revanche
> `notebooks/07_closed_loop_evaluation.ipynb` mentionné dans le `README.md`
> n'existe **pas encore** — vérifie avant d'y faire référence. La prédiction de
> panne expose maintenant des **métriques anticipatoires** (recall/precision sur
> machines `status=on`) et un **embargo** de split (voir `failure_prediction_eval`
> et `TemporalSplitter`), suite à l'audit d'intégrité (cf. `rapport_analyse.md`).

## Flux de données principal

1. **Ingest** — `mqtt_subscriber` collecte la télémétrie → `normalizer` la mappe
   vers le schéma unifié (voir `data/schema.md`) → `dataset_exporter` écrit
   `data/raw/episode=NNN/machine=mXX/part-*.parquet` + `metadata.json`.
2. **Features** — `features.pipeline` transforme la télémétrie brute en dataset
   ML dans `data/processed/episode=NNN/`, en appliquant les features temporal →
   contextual → energy, puis les labels failure/control, en retirant les lignes
   de warmup et les lignes à NaN critiques.
3. **Train/eval** — `evaluation.failure_prediction_eval` entraîne les modèles de
   panne ; `evaluation.fan_control_eval` évalue les contrôleurs ;
   `evaluation.benchmark` / `evaluation.robustness` produisent les métriques
   comparatives dans `evaluation/results/*.json` (suffixe label dans le nom).
4. **Supervise** — `supervisor.supervisor` exécute la boucle temps réel :
   télémétrie MQTT → `OnlineFeatureBuffer` → prédire le risque → décider le RPM →
   `PUT fan_speed`, avec un fallback REST `GET /cluster/status` si MQTT est
   indisponible.

### Labels de prédiction de panne

| Label         | Horizon | Usage |
|---------------|---------|-------|
| `failure_60s` | 60 s | **par défaut**, utilisé par le superviseur |
| `failure_30s` | 30 s | plus précis, préavis plus court |
| `hot_30s`     | 30 s | alerte thermique (temp > 95 % du seuil d'arrêt) |

La plupart des scripts/outils de workflow prennent un argument `--label` valant
`failure_60s` par défaut.

### Constantes de décision du superviseur (`supervisor/supervisor.py`)

- `RPM_LEVELS = [800, 1500, 2500, 3500, 4500]` — consignes de ventilation
  discrètes.
- `RISK_THRESHOLD = 0.60` — au-dessus de ce score de risque, override vers
  `RPM_HIGH` (4500).
- `HOT30S_THRESHOLD` (env, défaut 0.5) — override surchauffe.
- `RISK_LOG_THRESHOLD` (env, défaut 0.05) — ne logue une machine qu'au-dessus de
  ce risque.
- Les décisions se déclenchent tous les `DECISION_INTERVAL_TICKS` ticks
  *simulés* (1 tick = 1 s simulée), donc le comportement est indépendant de la
  vitesse de simulation.

## Commandes courantes

À lancer depuis la racine du dépôt. Les fichiers `*.bat` sont des wrappers de
confort pour Windows ; sous Linux/Mac, appelle directement les commandes
`python -m ...` sous-jacentes.

```bash
# Installation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env            # puis éditer pour ton hôte jumeaux-chauds

# Tests (markers : tests d'intégration "slow" exclus par défaut)
pytest tests/ -v
pytest tests/ -v -m "not slow"
pytest tests/test_phase4_models.py -v

# Ingérer un épisode
python -m ingest.mqtt_subscriber --duration 600 --episode 001 --scenario nominal
python -m ingest.mqtt_subscriber --continuous --episode 001 --scenario stress

# Feature engineering
python -m features.pipeline \
  --input data/raw/episode=001 \
  --output data/processed/episode=001 \
  --config data/raw/episode=001/metadata.json

# Entraînement / évaluation (paramétré par label)
python -m evaluation.failure_prediction_eval --label failure_60s   # 03_train_models.bat
python -m evaluation.fan_control_eval --label failure_60s --models all --output evaluation/results/fan_control_results_failure_60s.json
python -m evaluation.benchmark --label failure_60s                 # 05_benchmark_offline_metrics.bat
python -m evaluation.robustness --label failure_60s
python ingest_quick_EDA.py --processed-only                        # EDA rapide

# Lancer le superviseur
python -m supervisor.supervisor --mode ml --duration 300 --dry-run
python -m supervisor.supervisor --mode threshold

# Docker (jumeaux-chauds doit être accessible ; utilise host.docker.internal)
docker compose up --build supervisor
```

### Correspondance des scripts .bat (runners Windows)

| Script | Étape sous-jacente |
|--------|--------------------|
| `01_ingest_mqtt_simulations.bat` | collecte les 6 scénarios via MQTT |
| `02_ingest_gen_features.bat [NNN]` | lance `features.pipeline` par épisode |
| `03_train_models.bat [label]` | entraînement/éval de prédiction de panne |
| `04_train_fan_controllers.bat` | entraînement/éval des contrôleurs |
| `05_benchmark_offline_metrics.bat [label]` | benchmark + robustesse |
| `03_04_05_run_all_labels.bat` | lance 03–05 sur les 3 labels |
| `build-clean-app.bat` | rebuild Docker complet |

## Conventions & pièges

- **Points d'entrée des modules :** les modules exécutables utilisent
  `python -m <pkg>.<module>` avec `argparse`. Modules CLI :
  `ingest.mqtt_subscriber`, `features.pipeline`, `supervisor.supervisor`, et les
  quatre scripts `evaluation.*`.
- **`from __future__ import annotations`** est utilisé partout — garde-le sur les
  nouveaux modules.
- **Modèles sérialisés** sous `models/*/saved/` (joblib). Ces répertoires sont
  ignorés par git et produits par les scripts d'entraînement ; ne suppose pas
  qu'ils sont présents dans un clone neuf — lance d'abord l'entraînement, sinon
  les tests qui en dépendent échoueront/seront ignorés.
- **Isolation du stub xgboost :** `tests/conftest.py` retire un faux stub
  `xgboost` que `test_phase7_supervisor.py` peut injecter, pour que le
  dépicklage joblib fonctionne dans les autres phases. Ne supprime pas cette
  fixture ; si tu touches aux imports xgboost dans les tests, préserve ce
  nettoyage du stub.
- **Logging :** le superviseur met `httpx`/`httpcore` en WARNING pour éviter le
  bruit par requête. Suis ce pattern plutôt que de réactiver les logs HTTP
  verbeux.
- **Données reproductibles & ignorées par git :** tout ce qui est sous `data/`
  (sauf `data/schema.md`) est régénéré depuis l'ingestion + les features. Ne
  commite jamais de datasets ni de modèles entraînés.
- **Sémantique du temps :** les "ticks" sont des secondes *simulées*, pas du
  temps réel. Garde la cadence de décision en ticks (`DECISION_INTERVAL_TICKS`),
  en réservant les secondes réelles (`DECISION_INTERVAL_S`) au seul chemin de
  fallback REST.
- **Features online vs offline :** le superviseur ne peut calculer en direct
  qu'un sous-ensemble de features (`ONLINE_FEATURES` dans `supervisor.py`). En
  ajoutant des features au modèle, assure-toi qu'elles sont dérivables depuis
  `OnlineFeatureBuffer`, sinon le prédicteur cassera à l'exécution même si les
  métriques offline semblent correctes.

## Workflow git

- Développe sur la branche de feature assignée ; ne pousse jamais sur `main` sans
  permission explicite. Pousse avec `git push -u origin <branche>`.
- Les messages de commit sont en français et suivent des préfixes de style
  Conventional Commits (`feat`, `fix`, `docs`, souvent scopés par phase, ex.
  `feat(phase8): ...`). Respecte ce style.
- N'ouvre **pas** de pull request sauf demande explicite.

## Pour aller plus loin

- `documents/roadmap.md` — plan phase par phase et livrables.
- `documents/specifications.md` — specs techniques, interfaces, métriques.
- `documents/rapport_analyse.md` — rapport d'analyse / résultats.
- `data/schema.md` — schéma unifié de télémétrie et partitionnement Parquet.
- `README.md` — quickstart utilisateur (attention à la note Phase 9 ci-dessus).
