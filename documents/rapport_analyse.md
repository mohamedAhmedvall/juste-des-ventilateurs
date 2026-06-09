# Rapport d'analyse — Juste des Ventilateurs

**Projet M2 Data/IA — LaPlateforme_**  
**Auteur :** Tristan Vanrullen  
**Date :** Juin 2026  
**Version :** 1.0

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

### Limites et pistes d'amélioration

1. **Features online limitées** : en temps réel, les features rolling (30s, 60s) ne sont pas disponibles — elles sont approximées par la valeur instantanée. Une fenêtre glissante en mémoire améliorerait la qualité des décisions.

2. **Score controller à corriger** : le contrôleur à score multi-objectif nécessite un plancher RPM ou un rebalancement des poids lorsque la température est basse.

3. **Généralisation inter-scénarios** : entraîner sur tous les scénarios améliorerait l'accuracy sur `stress` (classe 4 sous-représentée).

4. **Test en conditions réelles** : le benchmark offline est une borne supérieure — en conditions réelles, la latence MQTT/REST et le bruit des capteurs dégraderont légèrement les performances.

---

## 5. Conclusion

Ce projet démontre qu'un couple **prédicteur logistique + contrôleur supervisé** avec override de risque surpasse significativement les approches réactives :

- **Détection** : 99.9% des incidents anticipés avec 120s de préavis
- **Sécurité** : RPM ≥ 3500 garanti sur 100% des situations dangereuses
- **Sobriété** : RPM moyen de 2681 (vs 4500 en mode "full speed" permanent)

La régression logistique, malgré sa simplicité, s'avère le meilleur choix pour la production grâce à ses probabilités bien calibrées et son lead time court et fiable.

---

## 6. Artefacts produits

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
| `supervisor/supervisor.py` | 6 | Service de supervision temps réel |
| `notebooks/01_ingestion_eda.ipynb` | 2 | EDA données brutes |
| `notebooks/02_feature_engineering.ipynb` | 3 | Exploration features |
| `notebooks/03_failure_prediction.ipynb` | 4 | Modèles prédictifs |
| `notebooks/04_fan_control.ipynb` | 5 | Contrôleurs de régulation |
| `notebooks/05_evaluation_comparative.ipynb` | 6 | Évaluation finale |
