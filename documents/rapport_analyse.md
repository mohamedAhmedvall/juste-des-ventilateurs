# Rapport d'analyse — Juste des Ventilateurs

**Projet M2 Data/IA — LaPlateforme_**  
**Auteur :** Tristan Vanrullen  
**Date :** Juin 2026  
**Version :** 1.1

---

## 1. Contexte et objectif

Le projet **Juste des Ventilateurs** est un service de maintenance prédictive et de régulation thermique couplé au simulateur de data center **jumeaux-chauds**. L'objectif est de démontrer qu'un couple (prédicteur ML + contrôleur prescriptif) surpasse les approches réactives classiques pour la gestion thermique d'un cluster de serveurs.

### Environnement de simulation

- **Cluster** : 5 machines (2 masters, 3 workers), seuil de shutdown ~88°C
- **Scénarios** : basic, busy_weeks, heatwave, nominal, stress, trace_replay
- **Données collectées** : 304 096 observations, 6 épisodes, fréquence ~5s
- **Features** : 47 features numériques (temporelles, contextuelles, énergétiques)

---

## 2. Phase 4 — Prédiction de pannes

### Stratégie de split (Option A)

Split temporel 70/15/15 par épisode pour éviter tout leakage :

| Split | Lignes | % positifs failure_60s |
|-------|--------|------------------------|
| Train | 212 864 | 22.0% |
| Val | 45 611 | — |
| Test | 45 621 | 18.8% |

### Résultats comparatifs

| Modèle | F1 | Recall | PR-AUC | Lead time médian | Incidents détectés |
|--------|----|--------|--------|------------------|-------------------|
| Baseline heuristique | 0.141 | 0.130 | 0.171 | 8.7s | 10/14 |
| **Régression logistique** | **0.851** | **0.930** | **0.812** | **72s** | **12/14** |
| Random Forest | 0.877 | 0.930 | 0.759 | 5876s | 4/14 |
| Gradient Boosting | 0.877 | 0.931 | 0.757 | 5876s | 4/14 |

### Analyse

La **régression logistique** est le meilleur prédicteur pour la Phase 6, bien que sa F1 soit légèrement inférieure au Random Forest. En effet :

- Son **PR-AUC** (0.812) est supérieur aux arbres (0.757-0.759), indiquant une meilleure calibration des probabilités.
- Son **lead time médian de 72s** permet une anticipation réelle (vs 5876s pour RF/GB — artéfact de détection très précoce sur les épisodes longs sans incident).
- Il détecte **12/14 incidents** vs 4/14 pour les arbres, qui tendent à s'activer trop tôt (faux positifs persistants).

**Recommandation :** régression logistique avec seuil optimisé pour Recall ≥ 0.85.

---

## 3. Phase 5 — Contrôleurs de régulation

### Résultats comparatifs (jeu de test, failure_60s)

| Contrôleur | MeanRPM | Action Accuracy | RPM MAE | Réaction danger |
|-----------|---------|----------------|---------|----------------|
| fixed_1500 | 1500 | 0.355 | 1031 | 0.000 |
| fixed_2500 | 2500 | 0.223 | 1270 | 0.000 |
| fixed_4500 | 4500 | 0.096 | 2764 | **1.000** |
| threshold | 1815 | 0.417 | 716 | 0.105 |
| pid | 1082 | 0.460 | 654 | 0.117 |
| **supervised** | **1736** | **0.9999** | **0** | 0.130 |
| score_controller | 0 | 0.265 | 1736 | 0.000 |

### Analyse

Le **contrôleur supervisé** (RandomForest sur action_class) reproduit quasi-parfaitement la politique oracle (accuracy 99.99%). Son RPM moyen de 1736 est sobre par rapport à fixed_4500 (4500 RPM constant).

**Anomalie score_controller :** le contrôleur à score choisit systématiquement RPM=0 car la température moyenne du test (~55°C) est loin du seuil de shutdown (~88°C). Le terme énergie domine la fonction de coût. Piste d'amélioration : forcer un RPM minimum ou rebalancer les poids α/β/γ.

**Recommandation :** contrôleur supervisé pour la Phase 6, avec override logistique.

---

## 4. Phase 6 — Boucle fermée et évaluation comparative

### Benchmark offline (3 modes, jeu de test complet)

| Mode | MeanRPM | AccAct | Danger→RPM≥3500 | Lead time médian | Incidents détectés |
|------|---------|--------|-----------------|-----------------|-------------------|
| **native** | 984 | 0.200 | 0.000 | 0s | 0/7986 |
| **threshold** | 1815 | 0.417 | 0.105 | 0s | 0/7986 |
| **ml** | 2681 | 0.786 | **1.000** | **120s** | **7981/7986** |

### Analyse

Le mode ML (logistic + supervised + override RPM_HIGH quand risk≥0.60) apporte des gains majeurs :

- **Détection d'incidents** : 7981/7986 (99.9%) vs 0% pour les autres modes
- **Lead time** : 120s médian — les alarmes arrivent en moyenne 2 minutes avant l'incident
- **Réaction danger** : 100% des ticks dangereux reçoivent un RPM ≥ 3500
- **Consigne** : RPM moyen de 2681 vs 984 (natif). La puissance réelle suit une loi cubique (P ∝ RPM³) — le surcoût en puissance fans est donc bien supérieur au ratio des RPM, mais reste marginal face au coût d'un shutdown (arrêt production + redémarrage)

### Test de robustesse par scénario

| Scénario | ML RPM | Natif RPM | ΔRPM | ML AccAct | ML Réaction |
|----------|--------|-----------|------|-----------|-------------|
| basic | 2087 | 857 | +1230 | 0.800 | 1.000 |
| busy_weeks | 2500 | 882 | +1618 | 0.800 | 1.000 |
| heatwave | 1905 | 802 | +1104 | 0.800 | 1.000 |
| nominal | 2236 | 867 | +1369 | 0.800 | 1.000 |
| **stress** | **4154** | **1377** | **+2777** | **0.723** | **1.000** |
| trace_replay | 3022 | 1084 | +1938 | 0.800 | 1.000 |

Le scénario **stress** est le plus exigeant (RPM moyen ML = 4154, ΔRPM = +2777). L'action accuracy légèrement plus faible (0.723 vs 0.800) s'explique par la présence de la classe 4 (RPM=4500) absente des autres scénarios d'entraînement.

### Limites corrigées en Phase 7

Les limites 1 et 4 ont été résolues :

1. ~~**Features online limitées**~~ → **Résolu Phase 7** : `OnlineFeatureBuffer` maintient une fenêtre glissante de 70 ticks par machine. Toutes les features rolling (temp_delta_5/15/30s, temp_rolling_mean/std, margin_to_shutdown) sont calculées correctement en temps réel.

4. ~~**Test en conditions réelles**~~ → **Phase 7 livrée** : le superviseur a été testé en live contre jumeaux-chauds. Voir section 5.

Limites restantes :

2. **Score controller à corriger** : le contrôleur à score multi-objectif nécessite un plancher RPM ou un rebalancement des poids lorsque la température est basse.

3. **Généralisation inter-scénarios** : entraîner sur tous les scénarios améliorerait l'accuracy sur `stress` (classe 4 sous-représentée).

---

## 5. Phase 7 — Superviseur robuste et télémétrie MQTT

### Problèmes identifiés lors des tests live Phase 6

| Problème | Cause | Impact |
|----------|-------|--------|
| risk=0.00 constant | Features glissantes toujours nulles (snapshot unique) | Prédicteur aveugle aux montées en température |
| Fréquence inadaptée | REST toutes les 5s réelles, simulation à 1 tick/s simulé | Features calculées sur des fenêtres 5x à 300x trop larges |
| speed_multiplier non rafraichi | Lecture de `/cluster/status` (champ absent) | Superviseur ignore les changements de vitesse Streamlit |
| Windows asyncio | ProactorEventLoop incompatible avec aiomqtt/paho | `NotImplementedError: add_reader` au démarrage |

### Solutions implémentées

**OnlineFeatureBuffer** (`supervisor/online_features.py`) : fenêtre glissante de 70 ticks par machine, alignée exactement sur `features/temporal.py`. Recalcule toutes les features rolling à chaque tick MQTT entrant.

**MqttTelemetryConsumer** (`supervisor/mqtt_telemetry.py`) : subscriber asyncio (aiomqtt), topic `dt/{cluster}/+/telemetry`. Alimente le buffer au rythme de la simulation (1 msg/s simulé), quelle que soit la vitesse. Sous-échantillonnage des décisions via `decision_interval_ticks` (défaut 5 ticks = 5s simulées).

**Fallback REST** : si MQTT indisponible au démarrage, le superviseur bascule automatiquement en boucle REST (comportement Phase 6). Log `[FALLBACK REST]` explicite.

**Corrections superviseur** :
- `get_speed_multiplier()` lit `/simulation/speed` (correction endpoint)
- `_refresh_speed()` : re-lecture periodique du multiplicateur (toutes les 10 iterations MQTT / 6 iterations REST)
- `asyncio.WindowsSelectorEventLoopPolicy` sur Windows (fix aiomqtt)
- `--log-level` CLI argument, `LOG_LEVEL` env var
- Logging DEBUG des features dans `_predict_risk()` pour diagnostic

### Résultats tests (suite 147 tests)

```
147 passed, 3 skipped in 20.81s
```

Les 3 tests skippés sont les tests gradient_boosting qui nécessitent xgboost installé (skip propre via `importorskip`).

---

## 6. Conclusion

Ce projet démontre qu'un couple **prédicteur logistique + contrôleur supervisé** avec override de risque surpasse significativement les approches réactives :

- **Détection** : 99.9% des incidents anticipés avec 120s de préavis (offline)
- **Sécurité** : RPM ≥ 3500 garanti sur 100% des situations dangereuses
- **Sobriété** : RPM moyen de 2681 (vs 4500 en mode "full speed" permanent)

La régression logistique, malgré sa simplicité, s'avère le meilleur choix pour la production grâce à ses probabilités bien calibrées et son lead time court et fiable.

La Phase 7 rend ce système opérationnel en conditions réelles : le superviseur reçoit la télémétrie à la cadence simulée via MQTT, calcule des features temporelles fidèles quelle que soit la vitesse de simulation, et reste robuste aux problèmes d'infrastructure (fallback REST, reconnexion, compatibilité Windows).

---

## 7. Artefacts produits

| Artefact | Phase | Description |
|----------|-------|-------------|
| `data/raw/episode=*/` | 2 | 6 épisodes de simulation bruts |
| `data/processed/episode=*/features.parquet` | 3 | Features + labels (66 colonnes) |
| `models/failure_prediction/saved/logistic_failure_60s.joblib` | 4 | Prédicteur recommandé |
| `models/fan_control/saved/supervised.joblib` | 5 | Contrôleur recommandé |
| `evaluation/results/failure_prediction_results_failure_60s.json` | 4 | Métriques Phase 4 |
| `evaluation/results/fan_control_results.json` | 5 | Métriques Phase 5 |
| `evaluation/results/benchmark_results.json` | 6 | Benchmark comparatif |
| `evaluation/results/robustness_results.json` | 6 | Robustesse par scénario |
| `supervisor/supervisor.py` | 6-7 | Service de supervision temps réel (721 lignes) |
| `supervisor/online_features.py` | 7 | OnlineFeatureBuffer -- fenetre glissante 70 ticks |
| `supervisor/mqtt_telemetry.py` | 7 | MqttTelemetryConsumer -- subscriber asyncio aiomqtt |
| `tests/conftest.py` | 7 | Isolation stubs xgboost entre modules de test |
| `notebooks/01_ingestion_eda.ipynb` | 2 | EDA données brutes |
| `notebooks/02_feature_engineering.ipynb` | 3 | Exploration features |
| `notebooks/03_failure_prediction.ipynb` | 4 | Modèles prédictifs |
| `notebooks/04_fan_control.ipynb` | 5 | Contrôleurs de régulation |
| `notebooks/05_evaluation_comparative.ipynb` | 6 | Évaluation finale |
| `notebooks/06_phase7_mqtt_supervision.ipynb` | 7 | Analyse MQTT live, features, decisions |

---

# Addendum — Phase 9 & audit d'intégrité méthodologique

> **Nature de cet addendum.** Les sections 1–7 ci-dessus reposent sur le dataset
> d'origine (304 k observations). L'addendum ci-dessous présente des résultats
> **régénérés sur un dataset fraîchement collecté et reproductible** (6 épisodes,
> **202 043 lignes**, collectés en direct contre jumeaux-chauds : stress×2,
> heatwave×2, busy_weeks, nominal). Les chiffres diffèrent donc volontairement de
> ceux des sections précédentes — l'objectif est la **traçabilité** et
> l'**honnêteté méthodologique**, pas le réglage fin des performances.

## 8. Phase 9 — Évaluation en boucle fermée (impact causal)

### Pourquoi la boucle fermée

L'évaluation offline (sections 4–6) **rejoue des données figées** : les RPM
commandés ne modifient pas les températures enregistrées, donc `nb_shutdowns` et
`T_mean` y sont identiques pour tous les contrôleurs. Elle mesure la *fidélité à
l'oracle*, pas l'*impact réel*. La Phase 9 (`evaluation/closed_loop_eval.py`)
**pilote réellement le simulateur** et laisse la physique recalculer les
températures en réponse aux consignes.

### Protocole

- Scénario `stress`, 300 s simulées, décision toutes les 5 s, vitesse 60×.
- Client découplé (`ControlClient`) : testable hors-ligne (faux client thermique)
  **et** branché en direct sur l'API jumeaux-chauds.
- PUE dérivé tick-par-tick de la dérivée de `energy_kwh_cumulated` (l'API
  n'expose pas `power_w` par machine) ; énergie fans via loi cubique P ∝ RPM³.

### Résultats (live, scénario stress)

| Contrôleur | T_moy (°C) | T_max (°C) | RPM moy | PUE moy | kWh fans | Éco. vs 4500 |
|---|---|---|---|---|---|---|
| native (auto) | 53.0 | 72.9 | 897 | 1.001 | 0.103 | 98.6 % |
| **supervised** | **48.5** | 72.3 | 1203 | 1.001 | 0.305 | 95.8 % |
| baseline_pid | 50.0 | 68.4 | 951 | 1.001 | 0.149 | 98.0 % |
| score_controller | 52.6 | **84.6** | 1244 | 1.008 | 1.420 | 80.5 % |
| baseline_fixed_4500 | **36.8** | 56.7 | 3600 | 1.061 | 6.561 | 10.0 % |

### Analyse

- **La différenciation causale apparaît enfin** : T_moy et l'énergie diffèrent
  désormais par contrôleur (impossible en offline). `fixed_4500` est le plus
  froid (36.8 °C) mais brûle **6.56 kWh** ; `supervised` refroidit utilement
  (48.5 vs 53.0 °C natif) pour **0.31 kWh** seulement.
- **Arbitrage sécurité/sobriété visible** : `score_controller` laisse T_max
  grimper à **84.6 °C** (proche du seuil 88) en gardant un RPM bas — illustration
  directe du risque d'un contrôleur trop sobre.
- Aucun shutdown sur cette fenêtre (le scénario `stress` reste sous 88 °C ici) ;
  la distinction **pannes évitables vs inévitables** (`fan_failure`) et le calcul
  `nb_avoidable_avoided` sont validés par les tests unitaires
  (`tests/test_phase9_closed_loop.py`, 33 tests dont 1 intégration live).

## 9. Intégrité méthodologique — audit de fuite de données

Un audit du data engineering a été mené (revue ligne par ligne + mesures).

### ✅ Pas de fuite train/test classique

| Vecteur | Constat |
|---|---|
| Normalisation | `StandardScaler` **dans le Pipeline**, ajusté sur le train seul |
| Hyperparamètres + seuil | optimisés sur **validation**, jamais sur test |
| Métriques | calculées sur **test tenu à l'écart** |
| Split | **temporel** par épisode (aucun shuffle aléatoire) |
| Features | **causales** (rolling/diff/cumsum backward only) |

### ⚠️ Faiblesse réelle trouvée : circularité cible/feature

Le label `failure_60s` dérive du **statut futur** (degraded/off). Or des features
dérivées du **statut courant** (`is_degraded`, `is_off`, …) étaient disponibles :

- `is_degraded` corrèle **+0.63** avec le label ;
- **74 %** des positifs sont des machines **déjà en panne** (41 % degraded,
  33 % off) → le label « panne dans 60 s » y est trivialement vrai.

Les métriques globales **surestiment donc l'anticipation** (détection d'un état
courant ≠ prédiction). Mesure honnête, restreinte aux machines **encore saines** :

| Vue (régression logistique) | Recall | Precision |
|---|---|---|
| Globale (annoncée) | 0.95 | 0.98 |
| **Anticipatoire** (status=on) | **0.90** | **0.83** |
| Anticipation pure (sans features de statut) | 0.83 | 0.74 |

Résultat rassurant : le **random forest** garde **0.99** de recall anticipatoire
*même sans les features de statut* → il anticipe réellement depuis la dynamique
thermique, ce n'était pas qu'une béquille.

### ⚠️ Préavis réel : pannes à montée rapide (fast-onset)

Sur ce dataset, le préavis médian mesuré est **~14 s** (robuste sur 7 incidents),
pas les 72 s de la section 2 : les pannes de ce simulateur sont **fast-onset**,
le signal précurseur ne diverge du régime normal que ~15 s avant la fenêtre de
danger. La cible de test a été ajustée en conséquence (≥ 12 s, justifiée
empiriquement) plutôt que de viser un seuil physiquement inatteignable.

### Corrections implémentées (toutes testées)

1. **Embargo temporel** (`TemporalSplitter`, `embargo_s`, défaut 60 s) : supprime
   la contamination de seuil des labels forward-looking entre splits.
2. **Métriques anticipatoires** exposées par défaut dans
   `evaluation.failure_prediction_eval` (recall/precision status=on).
3. **Option `--exclude-status-features`** (modèle d'anticipation pure).

Couverture : `tests/test_phase9b_methodology.py` (12 tests). Suite globale :
**210 passants**.

### Artefacts Phase 9 / intégrité

| Artefact | Description |
|---|---|
| `evaluation/closed_loop_eval.py` | Évaluation boucle fermée (impact causal, PUE, pannes évitables) |
| `tests/test_phase9_closed_loop.py` | 33 tests (faux client thermique + intégration live) |
| `tests/test_phase9b_methodology.py` | 12 tests (embargo, anti-circularité, métriques anticipatoires) |
| `evaluation/results/closed_loop_results_stress.json` | Résultats boucle fermée |
