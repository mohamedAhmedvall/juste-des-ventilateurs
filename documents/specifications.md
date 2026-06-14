# Spécifications Techniques — Juste des Ventilateurs

Projet M2 Data/IA — LaPlateforme_  
Version 1.4 — Juin 2026

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
| `action_class` | Oracle v1 — index dans `RPM_LEVELS` basé sur `temp_ratio` instantané. Classe 0 = 800 RPM. **Myope : ne tient pas compte de la trajectoire.** |
| `action_class_v2` | Oracle v2 (Phase 8) — index enrichi combinant position thermique, vitesse thermique et urgence panne. Voir §4.5. |

#### §4.5 Oracle trajectoire v2 (`add_control_labels_v2`) — Phase 8

**Motivation :** L'oracle v1 est myope — deux machines à T=75°C reçoivent le même label qu'elles montent ou descendent. Les features de trajectoire (`temp_delta_30s`, `time_to_failure_s`) sont disponibles mais ignorées des labels, créant une incohérence features/labels.

**Formule oracle v2 :**

```
score(t) = α · temp_ratio(t)
         + β · clip(temp_delta_30s(t) / DELTA_MAX, 0, 1)
         + γ · urgency(t)

action_class_v2 = floor(score × n_levels).clip(0, n_levels-1)
```

Avec :
- `temp_ratio = clip((T - 0.5·T_sd) / (0.5·T_sd), 0, 1)` — position thermique normalisée
- `temp_delta_30s` — variation de T sur 30s (positif = montée, négatif = descente)
- `DELTA_MAX = 5.0°C` — normalisation de la vitesse thermique
- `urgency = clip(1 - time_to_failure_s / HORIZON_S, 0, 1)` si `time_to_failure_s` disponible, sinon 0
- `HORIZON_S = 60s` — horizon d'urgence

**Paramètres par défaut :** α=0.5, β=0.3, γ=0.2

**Propriétés :**
- Montée rapide (+2°C/s sur 30s = +60°C, normalisé à 1.0) → score plus élevé → RPM plus élevé
- Refroidissement (temp_delta_30s < 0) → contribution β nulle (clip à 0) → RPM peut descendre
- Panne imminente (time_to_failure_s=20s, HORIZON=60s) → urgency=0.67 → pousse vers RPM_HIGH
- Machine froide sans trajectoire notable → score ≈ 0 → action_class_v2 = 0 (800 RPM)

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

Trois labels de prédiction sont disponibles — le nom du label est inclus dans le fichier de sortie :

| Label | Description | Fichier résultat |
|-------|-------------|-----------------|
| `failure_60s` | Panne dans 60s — **label superviseur** | `failure_prediction_results_failure_60s.json` |
| `failure_30s` | Panne dans 30s | `failure_prediction_results_failure_30s.json` |
| `hot_30s` | T > 95% seuil dans 30s | `failure_prediction_results_hot_30s.json` |

```bash
# Entraîner et benchmarker les 3 labels d'un coup (recommandé)
run_all_labels.bat

# Ou label par label :
03_train_models.bat                  # failure_60s (defaut)
03_train_models.bat failure_30s
03_train_models.bat hot_30s

# Evaluation seule (sans re-entrainement)
python -m evaluation.failure_prediction_eval --label failure_60s
```

**Résultats obtenus** (`evaluation/results/failure_prediction_results_failure_60s.json`) :

| Modèle | F1 | Recall | PR-AUC | Lead time médian |
|--------|----|--------|--------|-----------------|
| baseline | 0.141 | 0.130 | 0.171 | 8.7s |
| logistic | 0.851 | 0.930 | 0.812 | 72s |
| random_forest | 0.877 | 0.930 | 0.759 | 5876s |
| gradient_boosting | **0.877** | **0.931** | 0.757 | 5876s |

**Recommandation :** régression logistique pour le superviseur (meilleur taux de détection d'incidents 12/14, PR-AUC supérieur).

**Notebook :** `notebooks/03_failure_prediction.ipynb` — courbes PR/ROC, matrices de confusion, feature importance, analyse des résultats.

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
- Features : état courant + risk_score du prédicteur + features de trajectoire (`temp_delta_*`)
- Labels oracle v1 : `action_class` — basé sur température instantanée (myope)
- Labels oracle v2 : `action_class_v2` — enrichi trajectoire thermique + urgence panne (Phase 8, voir §4.5)

**Contrôleur à score multi-objectif (`score_controller.py`) :**
```python
J(a) = α·risk(t)·(1 - RPM(a)/RPM_MAX)   # risque résiduel pondéré par cooling
      + β·heat(t)                          # contrainte thermique pure
      + γ·energy(a)                        # coût énergétique (RPM/RPM_MAX)
      + δ·|ΔRPM(a)|/RPM_MAX               # pénalité changement brusque
```
- `cooling = 1 - RPM/RPM_MAX` : un RPM élevé réduit le risque résiduel
- `heat` non multiplié par cooling — évite le biais RPM_MAX à basse température
- `RPM_LEVELS = [800, 1500, 2500, 3500, 4500]` — plancher 800 RPM
- Paramètres par défaut : α=0.60, β=0.15, γ=0.20, δ=0.05 (recalibrés)

---

## 7. Module Supervisor (`supervisor/`)

### 7.1 Architecture Phase 7 (cible)

Le superviseur est structuré en deux boucles indépendantes :

```
MQTT broker :1883
  dt/{cluster}/+/telemetry
       │
       ▼  (1 msg/s simulé, quelle que soit la vitesse de simulation)
supervisor/mqtt_telemetry.py  ← MqttTelemetryConsumer
  - normalise le payload (même schéma que ingest/normalizer.py)
  - appelle OnlineFeatureBuffer.update(machine_id, snapshot)
  - fallback silencieux si MQTT indisponible
       │
       ▼  OnlineFeatureBuffer (fenêtre glissante 70 ticks par machine)
       │  recalcule : temp_delta_5/15/30s, temp_rolling_mean/std_30/60s,
       │              margin_*, rpm_*, power_*, pue_*, time_in_hot_zone_s,
       │              nb_shutdowns_episode, ticks_since_last_fault, ...
       │
       ▼  (tous les decision_interval_ticks ticks simulés reçus, défaut=5)
supervisor/supervisor.py  ← Supervisor._decision_loop()
  - feat_buffer.get_features(machine_id) → pd.Series
  - predictor.predict_proba(X) → risk_score
  - controller.decide_batch(X) → rpm
  - max(rpm, RPM_MIN=800)  ← plancher ventilation
  - si risk >= RISK_THRESHOLD (0.60) → override RPM_HIGH=4500  [risk_override]
  - si hot30s_score >= HOT30S_THRESHOLD (0.50) → override RPM_HIGH=4500  [hot30s_override]
  - PUT /machines/{id}/fan_speed (si rpm != prev_rpm)
  - decision_logger.log(entry)
  - fallback REST GET /cluster/status si MQTT indisponible
```

**Correctif de sous-échantillonnage (complément à Option E)**

L'Option E (MQTT) résout l'alimentation du buffer à la bonne cadence simulée. Sans sous-échantillonnage complémentaire, le loop de décision tournerait à `events_per_sec × speed_multiplier` fois/seconde réelle (60 décisions/s à `speed=60x`). Le mécanisme `decision_interval_ticks` compte les ticks MQTT reçus et ne déclenche une décision que toutes les N ticks — équivalent à décider toutes les N secondes *simulées*, robuste à toute vitesse.

**Garanties :**
- L'arrêt thermique automatique de jumeaux-chauds reste ACTIF (non court-circuité)
- Si le superviseur crash, les machines restent dans leur dernier mode (safe by default)
- Si MQTT indisponible : fallback sur GET /cluster/status REST (comportement Phase 6)
- `RPM_MIN=800` : les fans ne s'arrêtent jamais complètement en mode supervisé
- Timeout REST : 5s max par commande (configurable)

### 7.2 Boucle de supervision (`supervisor/supervisor.py`)

**Modes de supervision :**

| Mode | Prédicteur | Contrôleur | Description |
|------|-----------|------------|-------------|
| `ml` | LogisticPredictor | SupervisedController + override | **Recommandé** |
| `threshold` | — | ThresholdFanController | Règles simples sans ML |
| `native` | — | — | Aucune intervention (observation seule) |

**Paramètres CLI principaux :**
```
--mode ml|threshold|native   Mode de supervision (défaut: ml)
--duration N                 Durée max en secondes (défaut: infini)
--risk-threshold 0.60        Seuil risk_score → RPM_HIGH override
--dry-run                    Simuler sans envoyer de commandes REST
```

**Variables d'environnement (`.env`) :**
```
RISK_THRESHOLD=0.60         Seuil failure_60s → RPM_HIGH
HOT30S_THRESHOLD=0.50       Seuil hot_30s → RPM_HIGH (override surchauffe)
RISK_LOG_THRESHOLD=0.05     Seuil log INFO par machine
RPM_MIN=800                 Plancher ventilation minimale
```

### 7.3 Buffer de features en ligne (`supervisor/online_features.py`) ✅

`OnlineFeatureBuffer` maintient un historique de 70 ticks par machine (fenêtre max = 60s simulées + marge). À chaque tick MQTT reçu, il recalcule :

| Feature | Formule | Fenêtre |
|---------|---------|---------|
| `temp_delta_{5,15,30}s` | `temp[t] - temp[t-w]` | 5/15/30 ticks |
| `temp_rolling_mean_{30,60}s` | moyenne glissante | 30/60 ticks |
| `temp_rolling_std_30s` | écart-type glissant | 30 ticks |
| `margin_to_shutdown` | `88 - temp_c` | instantané |
| `margin_delta_30s` | `-temp_delta_30s` | 30 ticks |
| `power_fans_w` | `P_nom × (RPM/RPM_max)³ × n_fans` | instantané |
| `pue_estimated` | `1 + P_fans / P_compute` | instantané |
| `time_in_hot_zone_s` | durée cumulée T > 70.4°C | cumulatif |
| `nb_shutdowns_episode` | transitions → "off" | cumulatif |
| `ticks_since_last_fault` | ticks depuis dernière panne | cumulatif |

Ces features sont strictement alignées sur `features/temporal.py`, `features/energy.py` et `features/contextual.py` utilisés à l'entraînement.

### 7.4 Consumer MQTT télémétrie (`supervisor/mqtt_telemetry.py`) ✅

`MqttTelemetryConsumer` s'abonne à `dt/{cluster}/+/telemetry` via `aiomqtt` et alimente le buffer à chaque message reçu. Tourne en coroutine asyncio dans le même process que le superviseur. Fallback silencieux si MQTT indisponible.

### 7.5 Override hot_30s — surchauffe imminente (`supervisor/supervisor.py`) ✅

Second prédicteur `logistic_hot_30s` chargé en parallèle du prédicteur principal `failure_60s`.

**Logique de décision (ordre de priorité) :**
1. `risk_score >= RISK_THRESHOLD (0.60)` → RPM_HIGH, `risk_override=True`
2. `hot30s_score >= HOT30S_THRESHOLD (0.50)` → RPM_HIGH, `hot30s_override=True`
3. Sinon → décision normale du contrôleur (RPM_MIN=800 garanti)

**Champs JSONL ajoutés :** `hot30s_score` (float), `hot30s_override` (bool)

### 7.6 Logger de décisions (`supervisor/decision_logger.py`)

Chaque décision est loggée en JSONL avec :
- `ts`, `machine_id`, `temperature_c`, `status`, `fan_rpm_mean`
- `risk_score`, `hot30s_score`
- `risk_override` (bool), `hot30s_override` (bool)
- `rpm_previous`, `rpm_decided`, `command_sent` (bool)
- `mode`

### 7.7 Logs superviseur — niveaux et format ✅

| Niveau | Contenu |
|--------|---------|
| `DEBUG` | Requêtes HTTP 200 OK (httpx), cycles sans événement |
| `INFO` | Une ligne par cycle : `[t=120s speed=1x] cluster — 5 on  T_max=67.3°C  risk_max=0.12` |
| `INFO` | Une ligne par machine si `risk > RISK_LOG_THRESHOLD` ou RPM change : `srv-worker-01  T=67.3°C  risk=0.42  RPM 1500→3500 [RISK OVERRIDE]` |
| `INFO` | Connexions initiales, passage manual/auto |
| `WARNING` | Erreurs réseau, timeout — dédupliquées : `GET /cluster/status échoué (×12 depuis 60s)` |
| `ERROR` | Exceptions inattendues |

`RISK_LOG_THRESHOLD` configurable via `.env` (défaut `0.05`).

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

## 9. Module Évaluation boucle fermée (`evaluation/closed_loop_eval.py`) — Phase 9

### 9.1 Motivation

L'évaluation offline (Phase 5-6) mesure la fidélité d'un contrôleur à l'oracle sur des données historiques figées. Elle ne peut pas mesurer l'impact réel sur la thermique, car le simulateur n'est pas piloté. Deux conséquences :

- `nb_shutdowns` identique pour tous les contrôleurs (les arrêts thermiques passés ne changent pas)
- `T_mean` identique (les températures ne dépendent pas des RPM commandés a posteriori)
- PUE non calculable (nécessite les puissances mesurées en temps réel)

L'évaluation en boucle fermée pilote effectivement jumeaux-chauds, laisse la physique recalculer les températures en réponse aux RPM commandés, et mesure les métriques réelles.

### 9.2 Architecture du module

```
evaluation/
└── closed_loop_eval.py
    ├── ClosedLoopRunner          # orchestre un épisode complet
    │   ├── reset_episode()       # change scénario via PUT /simulation/scenario
    │   ├── decision_step()       # un tick : GET status → décider → PUT fan_speed → log
    │   └── aggregate_metrics()   # calcule PUE, shutdowns, économies à la fin
    ├── FaultClassifier           # distingue pannes évitables / inévitables
    └── main()                    # CLI : compare contrôleurs sur scénario
```

**Dépendances :**
- `supervisor/online_features.py` — OnlineFeatureBuffer (réutilisé tel quel)
- `models/failure_prediction/logistic_regression.py` — prédicteur risk_score
- `models/fan_control/*.py` — contrôleurs à comparer
- `httpx` — appels REST jumeaux-chauds

### 9.3 Interface `ClosedLoopRunner`

```python
class ClosedLoopRunner:
    def __init__(
        self,
        controller: FanController,
        predictor: FailurePredictor | None = None,
        api_url: str = "http://localhost:8000",
        dt_s: float = 5.0,           # intervalle de décision (secondes réelles)
        duration_s: float = 600.0,   # durée de l'épisode (secondes réelles)
        scenario: str = "stress",
        dry_run: bool = False,
    ): ...

    def run(self) -> dict:
        """Pilote jumeaux-chauds pendant duration_s et retourne les métriques."""
        ...

    def aggregate_metrics(self, records: list[dict]) -> dict:
        """Calcule les métriques à partir des enregistrements tick-par-tick."""
        ...
```

**Enregistrement par tick :**

```json
{
  "ts":               "2026-06-14T10:00:05Z",
  "machine_id":       "srv-worker-01",
  "temperature_c":    74.2,
  "status":           "on",
  "has_fan_fault":    false,
  "power_w":          342.1,
  "power_fans_w":     47.3,
  "rpm_commanded":    3500,
  "rpm_previous":     2500,
  "risk_score":       0.41,
  "risk_override":    false,
  "hot30s_override":  false
}
```

### 9.4 Interface `FaultClassifier`

```python
class FaultClassifier:
    @staticmethod
    def is_avoidable(record: dict) -> bool:
        """
        Retourne True si le shutdown/degraded est potentiellement évitable.
        Critères d'inévitabilité :
          - has_fan_fault == True au moment du shutdown (RPM forcé à 0 par le simulateur)
          - ou fan_fault active dans les 30s précédant le shutdown
        """
        ...

    @staticmethod
    def classify_episode(records: list[dict]) -> dict:
        """
        Retourne :
          nb_shutdowns_total      : tous arrêts thermiques
          nb_shutdowns_avoidable  : ceux sans fan_fault active
          nb_shutdowns_inevitable : ceux avec fan_fault active
        """
        ...
```

**Principe de classification :**

```
Pour chaque tick où status passe à "degraded" ou "off" :
  Si has_fan_fault == True dans les 30 ticks précédents → inévitable
  Sinon → évitable
```

### 9.5 Métriques calculées

**Métriques de sécurité :**

| Métrique | Formule | Sens |
|---------|---------|------|
| `nb_shutdowns_cl` | Compte des `status == "off"` après "degraded" | ↓ mieux |
| `nb_avoidable_cl` | Shutdowns sans fan_fault dans les 30s précédentes | ↓ mieux |
| `nb_inevitable_cl` | Shutdowns avec fan_fault active | référence fixe |
| `avoidable_avoided_pct` | `(nb_avoidable_natif - nb_avoidable_cl) / nb_avoidable_natif` | ↑ mieux |

**Métriques énergétiques :**

| Métrique | Formule | Sens |
|---------|---------|------|
| `pue_mean` | `mean( (power_compute_w + power_fans_w) / power_compute_w )` | ↓ mieux |
| `pue_p95` | 95e percentile du PUE | ↓ mieux |
| `energy_fans_kwh` | `sum(power_fans_w × dt_s) / 3 600 000` | ↓ mieux |
| `energy_saved_pct` | `(energy_baseline_4500 - energy_fans) / energy_baseline_4500` | ↑ mieux |
| `rpm_mean_cl` | Moyenne des RPM commandés | info |

**Métriques thermiques :**

| Métrique | Formule | Sens |
|---------|---------|------|
| `T_mean_cl` | Température moyenne toutes machines | ↓ mieux |
| `T_max_cl` | Température maximale observée | ↓ mieux |
| `pct_time_critical` | % ticks avec `T > 0.95 × t_shutdown` | ↓ mieux |

### 9.6 Calcul du PUE

```python
# Par tick, pour une machine :
power_it_w    = snapshot["power_w"]          # puissance totale mesurée par jumeaux-chauds
power_fans_w  = (rpm / RPM_MAX)**3 * 300.0   # loi cubique (300W à RPM_MAX)
power_compute = max(power_it_w - power_fans_w, 1.0)   # éviter division par zéro
pue_tick      = (power_compute + power_fans_w) / power_compute

# Sur l'épisode :
pue_mean = mean(pue_tick for all ticks all machines)
```

**Valeurs de référence PUE :**

| Contrôleur | PUE attendu |
|-----------|------------|
| `native` (auto jumeaux-chauds) | ~1.05–1.10 |
| `baseline_fixed_4500` | ~1.25–1.35 (fans toujours à fond) |
| `baseline_pid` | ~1.07–1.12 |
| `supervised_v2` | cible < 1.12 |

### 9.7 CLI et sorties

**Invocation :**

```bash
python -m evaluation.closed_loop_eval \
  --scenario stress \
  --duration 600 \
  --dt 5 \
  --controllers native supervised supervised_v2 score_controller baseline_pid \
  --output evaluation/results/closed_loop_results_stress.json \
  --api-url http://localhost:8000
```

**Arguments :**

| Argument | Défaut | Description |
|---------|--------|-------------|
| `--scenario` | `stress` | Scénario jumeaux-chauds |
| `--duration` | `600` | Durée de chaque épisode (s réelles) |
| `--dt` | `5` | Intervalle de décision (s réelles) |
| `--controllers` | `all` | Liste des contrôleurs à comparer |
| `--output` | auto | Fichier JSON de sortie |
| `--dry-run` | False | Simule sans envoyer de commandes REST |
| `--no-reset` | False | Ne pas changer de scénario entre épisodes |

**Format de sortie JSON :**

```json
{
  "scenario": "stress",
  "duration_s": 600,
  "dt_s": 5,
  "timestamp": "2026-06-14T10:30:00Z",
  "results": [
    {
      "controller": "supervised_v2",
      "nb_shutdowns_cl": 3,
      "nb_avoidable_cl": 1,
      "nb_inevitable_cl": 2,
      "avoidable_avoided_pct": 0.67,
      "pue_mean": 1.09,
      "pue_p95": 1.21,
      "energy_fans_kwh": 0.048,
      "energy_saved_pct": 0.83,
      "T_mean_cl": 66.4,
      "T_max_cl": 84.1,
      "rpm_mean_cl": 2180
    }
  ]
}
```

### 9.8 Notebook d'analyse (`notebooks/07_closed_loop_evaluation.ipynb`)

**Contenu :**

1. Chargement des résultats `closed_loop_results_{scenario}.json`
2. Tableau comparatif toutes métriques par contrôleur
3. Graphe Pareto sécurité / énergie (x = `energy_fans_kwh`, y = `nb_avoidable_cl`)
4. Évolution temporelle de T, RPM et PUE pour chaque contrôleur (courbes superposées)
5. Répartition pannes évitables / inévitables par scénario
6. Comparaison oracle v1 vs oracle v2 : gain sur `avoidable_avoided_pct` et `pue_mean`

### 9.9 Tests unitaires (`tests/test_phase9_closed_loop.py`)

**Couverture minimale :**

- `FaultClassifier.is_avoidable()` : shutdown avec fan_fault active → inévitable
- `FaultClassifier.is_avoidable()` : shutdown sans fan_fault → évitable
- `ClosedLoopRunner.aggregate_metrics()` : PUE calculé correctement depuis records mock
- `ClosedLoopRunner.aggregate_metrics()` : `energy_fans_kwh` = somme cohérente
- Aucune commande REST envoyée en mode `dry_run=True`
- `main()` CLI : fonctionne avec `--dry-run` (sans jumeaux-chauds actif)

---

## 10. Configuration et déploiement

### 9.1 Variables d'environnement (`.env`)

```env
# API jumeaux-chauds
API_BASE_URL=http://localhost:8000
# Sur Docker Desktop Windows, utiliser host.docker.internal à la place de localhost

# MQTT (Phase 7)
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_TOPIC_ROOT=dt
CLUSTER_ID=cluster_alpha

# Superviseur
SUPERVISOR_MODE=ml
PREDICTOR_MODEL=logistic
CONTROLLER_MODEL=supervised
RISK_THRESHOLD=0.6
DECISION_INTERVAL_S=5          # Phase 6 : intervalle REST en secondes
DECISION_INTERVAL_TICKS=5      # Phase 7 : décision toutes les N ticks simulés MQTT
RISK_LOG_THRESHOLD=0.05        # Seuil minimum pour logger une machine en INFO

# Données
T_SHUTDOWN_DEFAULT_C=88.0
STORAGE_BACKEND=parquet
PARQUET_DATA_DIR=./data
```

### 9.2 Structure Docker

```yaml
# docker-compose.yml
services:
  supervisor:
    build: .
    env_file: .env
    environment:
      # Docker Desktop Windows : host.docker.internal → hôte Windows
      - API_BASE_URL=http://host.docker.internal:8000
      - MQTT_BROKER_HOST=host.docker.internal
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./supervisor/logs:/app/supervisor/logs
    # Note : network_mode:host ignoré silencieusement sur Docker Desktop Windows
    #        Utiliser host.docker.internal pour joindre les services de l'hôte
```

**Note Docker Desktop Windows :** `network_mode: host` est silencieusement ignoré sur Docker Desktop Windows. Pour joindre jumeaux-chauds (API REST et MQTT) depuis un container, utiliser `host.docker.internal` comme hostname.

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
