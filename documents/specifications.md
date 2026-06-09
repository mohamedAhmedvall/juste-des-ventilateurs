# Spécifications Techniques — Juste des Ventilateurs

Projet M2 Data/IA — LaPlateforme_  
Version 1.2 — Juin 2026

---

## 1. Contexte et objectifs

### 1.1 Positionnement

**Juste des Ventilateurs** est un microservice de **maintenance prédictive et de régulation thermique** conçu pour fonctionner en parallèle du jumeau numérique [jumeaux-chauds](https://github.com/TristanV/jumeaux-chauds). Il s'insère dans la boucle opérationnelle d'un datacenter simulé pour :

1. **Anticiper** les pannes thermiques (degraded, shutdown) avant qu'elles surviennent
2. **Piloter** intelligemment les ventilateurs pour maintenir la sécurité thermique tout en limitant la consommation énergétique
3. **Évaluer** et **comparer** plusieurs couples (modèle prédictif, contrôleur prescriptif) contre des baselines

### 1.2 Contraintes système

- **Sécurité absolue** : ne jamais inhiber l'arrêt thermique automatique de jumeaux-chauds
- **Latence** : décision de contrôle en < 1s, fréquence de décision ≥ 1 Hz (configurable)
- **Reproductibilité** : tous les résultats reproductibles (seeds fixés, splits documentés)
- **Indépendance** : le service fonctionne même si jumeaux-chauds redémarre (reconnexion MQTT)

---

## 2. Architecture générale

```
┌─────────────────────────────────────────────────────────────┐
│                    jumeaux-chauds                           │
│  MQTT :1883  ←──────────────────────────────────────────   │
│  REST API :8000  ←──────────────────────────────────────── │
└──────────────────────────┬──────────────────────────────────┘
                           │ télémétrie (MQTT sub)
                           │ commandes (HTTP PUT)
┌──────────────────────────▼──────────────────────────────────┐
│                 juste-des-ventilateurs                       │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────────┐  │
│  │  Ingest  │──▶│ Features │──▶│  Failure Predictor     │  │
│  │  (MQTT)  │   │ Pipeline │   │  (RF / GBM / LogReg)   │  │
│  └──────────┘   └──────────┘   └──────────┬─────────────┘  │
│       │                                    │ risk_score      │
│       ▼                                    ▼                 │
│  ┌──────────┐                  ┌────────────────────────┐  │
│  │  Storage │                  │  Fan Controller        │  │
│  │  (TS/PQ) │                  │  (Score / Supervised)  │  │
│  └──────────┘                  └──────────┬─────────────┘  │
│                                            │ RPM command     │
│                                            ▼                 │
│                               ┌────────────────────────┐  │
│                               │  Supervisor / Logger   │  │
│                               │  (Decision loop)       │  │
│                               └────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Module Ingest (`ingest/`)

### 3.1 Subscriber MQTT (`ingest/mqtt_subscriber.py`)

**Connexion :**
- Broker : `localhost:1883` (configurable via `.env`)
- Client : `aiomqtt` v2 (async), avec `WindowsSelectorEventLoopPolicy` sur Windows
- Reconnexion automatique avec backoff exponentiel (1s → 30s)

**Topics souscrits (convention jumeaux-chauds) :**

| Topic | QoS | Fréquence | Description |
|-------|-----|-----------|-------------|
| `dt/cluster_alpha/+/telemetry` | 0 | 1/s par machine | Snapshot complet machine |
| `dt/cluster_alpha/+/status` | 1 | événementiel | Changements d'état (on/degraded/off) |
| `dt/cluster_alpha/+/fault` | 1 | événementiel | Injections et recovery de pannes |
| `dt/cluster_alpha/summary` | 1 | ~1/5s | KPI agrégés du cluster |

**CLI :**
```bash
python -m ingest.mqtt_subscriber --duration 600 --episode 001 --scenario stress
```
- `--duration N` : collecte bornée (secondes réelles)
- `--continuous` : mode daemon sans limite
- `--episode ID` : identifiant de l'épisode (ex: `001`)
- `--scenario NOM` : nom du scénario jumeaux-chauds actif (priorité sur `.env` et API)

**Résolution du scénario actif :**
Priorité : `--scenario CLI` > variable `SCENARIO` dans `.env` > réponse API > `"unknown"`

**Payload reçu (topic telemetry, format jumeaux-chauds) :**
```json
{
  "id": "srv-worker-01",
  "role": "worker",
  "status": "on",
  "temperature_c": 67.3,
  "power_w": 342.1,
  "energy_kwh_cumulated": 1.23,
  "fans": [{"idx": 0, "rpm": 2800, "mode": "auto"}, {"idx": 1, "rpm": 2750, "mode": "auto"}],
  "sensors": {"s1": {"temp_c": 67.8, "bias_c": 0.5}, "s2": {"temp_c": 66.9, "bias_c": -0.3}},
  "faults": []
}
```

### 3.2 Normalizer (`ingest/normalizer.py`)

**Schéma unifié de sortie :**

| Champ | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | Horodatage UTC de réception |
| `cluster_id` | str | ID du cluster (extrait du topic MQTT) |
| `machine_id` | str | ID de la machine |
| `status` | enum | `on` / `degraded` / `off` |
| `temperature_c` | float | Température en °C |
| `power_w` | float | Puissance électrique totale (W) |
| `energy_kwh` | float | Énergie cumulée (kWh) |
| `fan_rpm_mean` | float | RPM moyen de tous les fans |
| `fan_rpm_std` | float | Écart-type des RPM |
| `fan_count` | int | Nombre de ventilateurs |
| `sensor_temp_max` | float | Température max parmi les sondes |
| `sensor_temp_mean` | float | Température moyenne des sondes |
| `has_fault` | bool | Présence d'une panne active |
| `fault_types` | str | Types de pannes actives (CSV) |
| `load_estimated` | float | Charge estimée depuis power_w |

### 3.3 Dataset Exporter (`ingest/dataset_exporter.py`)

- Format : **Parquet** (recommandé pour ML), fallback CSV si pyarrow absent
- Partitionnement : `data/raw/episode={N}/machine={id}/part-0.parquet`
- Les résumés cluster vont dans `machine=_cluster/`
- Métadonnées par épisode : `data/raw/episode={N}/metadata.json`

**Contenu de `metadata.json` :**
```json
{
  "episode_id": "001",
  "scenario": "stress",
  "n_records": 54000,
  "duration_s": 180.3,
  "ts_start_real": "2026-06-09T01:00:00Z",
  "ts_end_real": "2026-06-09T01:03:00Z",
  "ts_sim_start": "2005-01-01T00:00:00Z",
  "ts_sim_end": "2005-01-01T03:00:00Z",
  "sim_duration_s": 10800,
  "machines": {
    "srv-master-01": {"role": "master", "t_shutdown_c": 90.0, "t_restart_c": 55.0, "fan_max_rpm": 5000, "fan_count": 2},
    "srv-worker-01": {"role": "worker", "t_shutdown_c": 88.0, "t_restart_c": 50.0, "fan_max_rpm": 5000, "fan_count": 2}
  },
  "cluster_id": "cluster_alpha"
}
```

### 3.4 Scripts de collecte

**`ingest_mqtt_simulations.bat`** : collecte automatisée multi-scénarios
- Passe la simulation à x60 (`PUT /simulation/speed`)
- Pour chaque scénario : change le scénario, vérifie/corrige l'état (running/paused/stopped), stabilise 5s, lance le subscriber 180s
- Scénarios couverts : `basic`, `busy_weeks`, `heatwave`, `nominal`, `stress`, `trace_replay`
- Épisodes produits : `001` à `006`

**`ingest_gen_features.bat`** : feature engineering en batch
```bash
ingest_gen_features.bat          # tous les épisodes
ingest_gen_features.bat 003      # épisode spécifique
```

---

## 4. Module Features (`features/`)

### 4.1 Features temporelles (`features/temporal.py`)

| Feature | Calcul | Fenêtre |
|---------|--------|---------|
| `temp_delta_5s` | `temp(t) - temp(t-5s)` | 5s |
| `temp_delta_15s` | `temp(t) - temp(t-15s)` | 15s |
| `temp_delta_30s` | `temp(t) - temp(t-30s)` | 30s |
| `temp_rolling_mean_30s` | Moyenne glissante température | 30s |
| `temp_rolling_mean_60s` | Moyenne glissante température | 60s |
| `temp_rolling_std_30s` | Écart-type glissant température | 30s |
| `margin_to_shutdown` | `t_shutdown - temperature_c` | instant |
| `margin_pct` | `margin_to_shutdown / t_shutdown * 100` | instant |
| `margin_delta_30s` | Variation de la marge sur 30s | 30s |
| `load_rolling_mean_30s` | Moyenne charge estimée | 30s |
| `load_rolling_mean_60s` | Moyenne charge estimée | 60s |
| `rpm_variance` | Variance des RPM des fans | instant |
| `rpm_cv` | Coefficient de variation RPM | instant |
| `rpm_delta_15s` | Variation RPM moyen sur 15s | 15s |
| `rpm_rolling_mean_30s` | Moyenne RPM moyen | 30s |
| `power_rolling_mean_30s` | Moyenne puissance totale | 30s |
| `power_delta_30s` | Variation puissance sur 30s | 30s |
| `sensor_max_delta_15s` | Variation temp max capteurs sur 15s | 15s |
| `sensor_max_rolling_mean_30s` | Moyenne temp max capteurs | 30s |

### 4.2 Features contextuelles (`features/contextual.py`)

Seuil zone chaude : `T > 0.80 × t_shutdown`

| Feature | Description |
|---------|-------------|
| `time_in_hot_zone_s` | Durée continue en zone chaude (reset si T redescend) |
| `time_in_degraded_s` | Durée continue en mode degraded |
| `nb_shutdowns_episode` | Nombre de shutdowns depuis début épisode |
| `nb_degraded_episode` | Nombre de passages en degraded depuis début épisode |
| `ticks_since_last_shutdown` | Ticks depuis le dernier shutdown |
| `ticks_since_last_fault` | Ticks depuis la dernière panne injectée |
| `has_fan_fault` | Fan failure active (bool) |
| `has_power_surge` | Power surge active (bool) |
| `has_sensor_drift` | Sensor drift active (bool) |
| `fan_mode_manual` | Au moins un fan en mode manual (bool) |
| `rpm_changes_last_60s` | Nombre de changements de consigne ventilateur sur 60s |
| `is_recovering` | Machine repassée à `on` après `degraded`/`off` récemment (bool) |

### 4.3 Features énergétiques (`features/energy.py`)

Loi cubique (alignée sur `physics.py` de jumeaux-chauds) : `P_fan = P_nominal × (RPM/RPM_max)³ × fan_count`

| Feature | Description |
|---------|-------------|
| `power_fans_w` | Puissance consommée par les fans (W, loi cubique) |
| `power_compute_w` | Puissance de calcul (W, hors fans) |
| `fan_energy_ratio` | `power_fans / power_total` |
| `pue_estimated` | PUE estimé (1 + fan_energy / compute_energy) |
| `energy_per_temp_unit` | kWh / °C (efficacité du refroidissement) |
| `energy_fans_kwh_cumulated` | Énergie fans cumulée depuis début épisode (kWh) |
| `power_fans_rolling_mean_30s` | Moyenne puissance fans sur 30s |
| `pue_rolling_mean_30s` | Moyenne PUE sur 30s |

### 4.4 Labeler (`features/labeler.py`)

**Labels pour le modèle prédictif :**

| Label | Définition | Usage |
|-------|-----------|-------|
| `failure_60s` | `1` si status=degraded ou off(overheat) dans les 60s suivantes | Prédiction principale |
| `failure_30s` | `1` si status=degraded ou off(overheat) dans les 30s suivantes | Prédiction courte portée |
| `hot_30s` | `1` si température > `0.95 * t_shutdown` dans les 30s | Alerte température |

**Labels pour le contrôleur supervisé :**

| Label | Définition |
|-------|-----------|
| `time_to_failure_s` | Temps en secondes avant le prochain incident (`None` si aucun) |
| `action_class` | Index dans `RPM_LEVELS = [0, 1500, 2500, 3500, 4500]` — oracle heuristique basé sur `temp_ratio × n_levels`, forcé au max si `status=degraded` |

---

## 5. Module Failure Prediction (`models/failure_prediction/`)

### 5.1 Interface commune

Tous les modèles implémentent l'interface suivante :

```python
class FailurePredictor:
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...       # labels binaires
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ... # shape (n, 2), colonne 1 = P(failure)
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> "FailurePredictor": ...
```

### 5.2 Splitter temporel (`splitter.py`)

Stratégie **Option A — fenêtre temporelle (70/15/15) par épisode** :

```
Pour chaque épisode (trié chronologiquement) :
  train ← 70% initial
  val   ← 15% médian
  test  ← 15% final

X_train_global = concat(train_ep001, ..., train_ep006)
X_val_global   = concat(val_ep001,   ..., val_ep006)
X_test_global  = concat(test_ep001,  ..., test_ep006)
```

Garanties : pas de leakage temporel, chaque scénario représenté dans les 3 splits.

```python
splitter = TemporalSplitter(processed_dir="data/processed")
X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(label_col="failure_60s")
# → train=212 864  val=45 611  test=45 621  features=47
```

Colonnes exclues des features : timestamps, IDs, labels, colonnes 100% NaN (`machines_total`, `machines_on`).

### 5.3 Modèles implémentés

**Baseline heuristique (`baseline_threshold.py`) :**
- Règle : `failure = 1 si temperature_c > T_warn ET time_in_hot_zone_s > N`
- Grid search sur `T_warn ∈ [60, 85]°C` et `N ∈ [0, 30]s`
- Score continu : `0.7 × (T/T_warn) + 0.3 × (hot/N)`

**Régression Logistique (`logistic_regression.py`) :**
- `StandardScaler` + `CalibratedClassifierCV` (calibration Platt, cv=3)
- `C` sélectionné par validation sur `[0.001, 0.01, 0.1, 1, 10, 100]`
- Seuil de décision optimisé sur Recall ≥ 0.85

**Random Forest (`random_forest.py`) :**
- `class_weight="balanced"`, `random_state=42`
- Grid search : `n_estimators ∈ [100, 200]`, `max_depth ∈ [10, 15, 20]`
- Feature importance extraite et loggée, seuil optimisé

**Gradient Boosting (`gradient_boosting.py`) :**
- Backend automatique : XGBoost → LightGBM → sklearn GBM
- Early stopping sur val (`n_rounds=30`), `scale_pos_weight` automatique
- Paramètres par défaut : `lr=0.05`, `max_depth=6`, `subsample=0.8`

### 5.4 Protocole d'évaluation (`evaluation/failure_prediction_eval.py`)

**Métriques :**
- Precision, Recall, F1, PR-AUC, ROC-AUC
- **Lead time moyen** : `mean(t_incident - t_première_alerte)` sur fenêtre 120s
- Taux de faux négatifs sur shutdowns thermiques

**Cibles :**
- Recall ≥ 0.85 sur cas dangereux
- Lead time moyen ≥ 30s
- F1 > baseline heuristique

**Lancement :**
```bash
python -m evaluation.failure_prediction_eval --label failure_60s
python -m evaluation.failure_prediction_eval --models gradient_boosting --label failure_30s
```

Résultats exportés dans `evaluation/results/failure_prediction_results.json`.

---

## 6. Module Fan Control (`models/fan_control/`)

### 6.1 Interface commune

```python
class FanController:
    def decide(self, state: MachineState, risk_score: float) -> int:
        """Retourne le RPM cible pour le prochain pas de temps."""
        ...
    def fit(self, X_train, y_train) -> None: ...  # pour les méthodes supervisées
```

### 6.2 Contrôleurs implémentés

**Baseline fixe (`baseline_fixed.py`) :**
- RPM constant, plusieurs niveaux : 0 (off), 1500 (low), 2500 (medium), 3500 (high), 4500 (max)

**Baseline seuils (`baseline_threshold.py`) :**
```python
if temperature_c > T_high: rpm = 4500
elif temperature_c > T_medium: rpm = 3500
elif temperature_c > T_low: rpm = 2500
else: rpm = 1500
```
Seuils `T_low`, `T_medium`, `T_high` optimisés par grid search.

**PID simple (`baseline_pid.py`) :**
- Cible : `T_target = 0.80 * t_shutdown`
- Erreur : `e(t) = temperature_c - T_target`
- Commande : `rpm(t) = rpm_min + Kp*e + Ki*∫e + Kd*Δe` (clampé dans `[0, 4500]`)

**Contrôleur ML supervisé (`supervised_controller.py`) :**
- Classifier multiclasse (5 classes = 5 niveaux RPM)
- Features : état courant + risk_score du prédicteur
- Labels : `action_class` généré par simulation avec oracle (baseline PID optimisée)

**Contrôleur à score multi-objectif (`score_controller.py`) :**
```python
J(t) = α·risk(t) + β·heat(t) + γ·energy(t) + δ·|ΔRPM_t|
```
- `risk(t)` : probabilité de panne prédite par le modèle
- `heat(t)` : `temperature_c / t_shutdown` (proportion du seuil atteint)
- `energy(t)` : `rpm / rpm_max` (proxy consommation ventilateur)
- `|ΔRPM_t|` : pénalité de changement brusque de consigne
- Pour chaque action candidate, on choisit celle qui minimise J(t)
- Paramètres α, β, γ, δ optimisés offline

---

## 7. Module Supervisor (`supervisor/`)

### 7.1 Boucle de supervision (`supervisor/supervisor.py`)

**Cycle de décision (fréquence : toutes les 5s par défaut) :**

```
1. Lire l'état de chaque machine (via MQTT ou GET /machines/{id})
2. Calculer les features (pipeline features/)
3. Évaluer le risque (modèle prédictif → risk_score ∈ [0,1])
4. Décider la consigne RPM (contrôleur → rpm_target)
5. Si machine en mode auto : passer en mode manual (PUT /machines/{id}/fan_mode)
6. Envoyer la consigne (PUT /machines/{id}/fan_speed)
7. Logger la décision et les métriques observées
8. Attendre le prochain cycle
```

**Garanties :**
- L'arrêt thermique automatique de jumeaux-chauds reste ACTIF (non court-circuité)
- Si le superviseur crash, les machines restent dans leur dernier mode (safe by default)
- Timeout REST : 500ms max par commande

### 7.2 Logger de décisions (`supervisor/decision_logger.py`)

Chaque décision est loggée avec :
- `timestamp`, `machine_id`, `temperature_c`, `status`
- `risk_score`, `failure_predicted` (bool)
- `rpm_before`, `rpm_decided`, `fan_mode`
- `event` : `shutdown`, `degraded`, `recovery`, `normal`

Stockage : Parquet ou TimescaleDB selon configuration.

---

## 8. Module Evaluation (`evaluation/`)

### 8.1 Métriques globales de benchmark

| Métrique | Description | Sens |
|---------|-------------|------|
| `nb_shutdowns` | Nombre total d'arrêts thermiques | ↓ mieux |
| `nb_degraded_episodes` | Nombre de passages en mode dégradé | ↓ mieux |
| `T_mean` | Température moyenne sur l'épisode (°C) | ↓ mieux |
| `T_max` | Température maximale observée (°C) | ↓ mieux |
| `energy_total_kwh` | Énergie totale consommée | ↓ mieux |
| `energy_fans_kwh` | Énergie consommée par les fans | ↓ mieux |
| `fan_energy_pct` | Part des fans dans l'énergie totale (%) | info |
| `incidents_avoided_pct` | Shutdowns évités vs baseline native (%) | ↑ mieux |

### 8.2 Scénarios d'évaluation

| Scénario | Description | Durée |
|----------|-------------|-------|
| `nominal` | Charge sine_wave standard | 30 min |
| `stress` | Charge élevée + pannes fréquentes | 30 min |
| `heatwave` | Température ambiante croissante | 30 min |
| `busy_weeks` | Cycles jour/semaine réalistes | 60 min |

---

## 9. Configuration et déploiement

### 9.1 Variables d'environnement (`.env`)

```env
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_TOPIC_ROOT=dt
CLUSTER_ID=cluster_alpha
API_BASE_URL=http://localhost:8000
T_SHUTDOWN_DEFAULT_C=88.0
STORAGE_BACKEND=parquet
PARQUET_DATA_DIR=./data
DECISION_INTERVAL_S=5
PREDICTOR_MODEL=gradient_boosting
CONTROLLER_MODEL=score_controller
RISK_THRESHOLD=0.6
ROLLING_WINDOW_S=60
```

### 9.2 Structure Docker

```yaml
# docker-compose.yml
services:
  supervisor:
    build: .
    depends_on: [mosquitto]
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    network_mode: host  # pour rejoindre le réseau jumeaux-chauds
```

### 9.3 Requirements principaux

```
# Core
paho-mqtt>=1.6
httpx>=0.27          # client HTTP async pour l'API jumeaux-chauds
pandas>=2.0
numpy>=1.24
pyarrow>=14.0        # Parquet

# ML
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.0
joblib>=1.3

# Storage (optionnel)
psycopg2-binary>=2.9

# Config
python-dotenv>=1.0
omegaconf>=2.3       # cohérence avec jumeaux-chauds
```

---

## 10. Intégration avec jumeaux-chauds

### 10.1 Connexion MQTT

Topics suivis :
- `cluster/{cluster_id}/machine/{machine_id}` → payload snapshot complet

### 10.2 Commandes REST utilisées

| Endpoint | Usage dans juste-des-ventilateurs |
|----------|----------------------------------|
| `GET /cluster/status` | Initialisation : découverte des machines et paramètres thermiques |
| `GET /machines/{id}` | Fallback si MQTT indisponible |
| `PUT /machines/{id}/fan_mode` | Passage en mode manual avant contrôle |
| `PUT /machines/{id}/fan_speed` | Envoi de la consigne RPM |

### 10.3 Récupération des paramètres thermiques

Au démarrage, le superviseur requête `GET /cluster/status` pour récupérer les seuils `t_shutdown_c` et `t_restart_c` de chaque machine, utilisés pour le calcul des features `margin_to_shutdown` et des labels.

---

## 11. Conventions de code

- Python 3.11+, typage statique (type hints complets)
- Formatage : `black` + `ruff`
- Tests : `pytest`, couverture ≥ 80% sur les modules critiques
- Logging : module `logging` standard, niveau configurable
- Nommage : `snake_case` pour fonctions/variables, `PascalCase` pour classes
- Docstrings : format Google style
