# Récapitulatif des travaux & guide de test local

Ce document trace **tout ce qui a été réalisé** (au-delà du socle Phases 1–8
pré-existant) et explique **comment tester le produit en local**.

---

## 1. Ce qui a été réalisé

### 1.1 Évaluation en boucle fermée (Phase 9)
- `evaluation/closed_loop_eval.py` : pilote **réellement** jumeaux-chauds pour mesurer
  l'**impact causal** des contrôleurs (vs rejeu offline figé).
  - `ClosedLoopRunner` (lire → décider → commander → enregistrer), découplé du
    transport (client injectable → testable sans simulateur).
  - `FaultClassifier` : pannes **évitables** vs **inévitables** (`fan_failure`).
  - **PUE réel** dérivé de la dérivée de `energy_kwh_cumulated` (l'API n'expose pas
    `power_w` par machine).
  - Métriques : `nb_shutdowns_cl`, `nb_avoidable`, `nb_avoidable_avoided`,
    `pue_mean`, `energy_fans_kwh`, `energy_saved_vs_max_pct`, `T_*_cl`, `rpm_mean_cl`.
- **Démonstration causale** (heatwave, sims fraîches) : sans refroidissement = **4
  arrêts évitables** ; natif / supervisé / plein régime les **évitent tous**.
  Le supervisé : 72 °C @ 0,34 kWh vs plein régime 57,9 °C @ 157 kWh.

### 1.2 Données & modèles (pipeline exécutée)
- Ingestion de **6 épisodes** via MQTT (202 043 lignes), features générées.
- **Prédiction de pannes** entraînée sur 3 labels (`failure_60s/30s`, `hot_30s`) :
  baseline, logistique, random forest, gradient boosting.
- **Contrôleurs** entraînés : supervisé, score, PID, seuils.

### 1.3 Audit d'intégrité du data engineering
- **Vérifié : aucune fuite train/test** (scaler sur train, hyperparamètres + seuil
  sur validation, métriques sur test isolé, features causales, split temporel).
- **Circularité détectée** : `is_degraded` corrèle +0,63 avec le label ; 74 % des
  positifs sont des machines **déjà** en panne (détection ≠ anticipation).
- **Correctifs** (tous testés) :
  - **Embargo temporel** (`TemporalSplitter`, `embargo_s`, défaut 60 s).
  - **Métriques anticipatoires** (recall/precision sur machines `status=on`),
    exposées par défaut : logistique **0.90 / 0.83**, RF **0.99** (robuste même sans
    features de statut, via `--exclude-status-features`).
  - Cible de lead-time du test alignée sur la dynamique réelle (pannes *fast-onset*,
    préavis médian ~14 s).

### 1.4 Dashboard de pilotage NOC (`dashboard/`)
- `noc.html` : dashboard **autonome** (HTML/CSS/JS + SVG, zéro dépendance/CDN), thème
  futuriste, **serveurs rackmount réalistes** (2 ventilateurs animés/machine, baies,
  PSU, grille, surchauffe visuelle), **jauge de risque ML + explicabilité**, **consigne
  IA**, **contrôles** (boutons RPM/mode + AUTO-PILOTE). Fallback démo hors-ligne.
- `noc_bridge.py` : pont (stdlib) qui sert le dashboard et expose :
  - `GET /api/live` : télémétrie réelle + **risque ML** + **explicabilité** + **rpm_reco**
    + métriques cluster ;
  - `POST /api/command` (RPM ou mode), `POST /api/autopilot` (toggle) ;
  - une **boucle de fond** qui, si l'auto-pilote est actif, applique en continu la
    consigne du contrôleur → **boucle de décision fermée, live**.

### 1.5 Documentation
- `CLAUDE.md` (guide assistants), addendum `documents/rapport_analyse.md`
  (sections 8 Phase 9 + 9 intégrité), `notebooks/07_closed_loop_evaluation.ipynb`,
  `documents/presentation_soutenance.md`, ce fichier.

### 1.6 Tests
- **221 tests** automatisés verts (+ 4 « slow » d'intégration).
  - `test_phase9_closed_loop.py` (boucle fermée, faux client thermique + live),
    `test_phase9b_methodology.py` (embargo, anti-circularité, anticipatoire),
    `test_phase9c_dashboard.py` (pont : risque, explicabilité, reco, commandes, auto-pilote).

---

## 2. Tester le produit en local

> **Prérequis** : Python 3.11+, un broker MQTT (mosquitto), et le dépôt
> **jumeaux-chauds** cloné en sibling (`../jumeaux-chauds`).

### 2.1 Installation

```bash
# Projet
cd juste-des-ventilateurs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# jumeaux-chauds (dans un autre dossier, sibling)
cd ../jumeaux-chauds
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Lancer la suite de tests (aucune dépendance externe)

```bash
cd juste-des-ventilateurs && source .venv/bin/activate
pytest tests/ -v -m "not slow"     # ~220 tests, < 15 s
```
Les tests de boucle fermée et du dashboard utilisent des **faux clients** → ils
valident toute la logique **sans** jumeaux-chauds.

### 2.3 Lancer jumeaux-chauds (pour tout ce qui est « live »)

```bash
# Broker MQTT
mosquitto -p 1883 &
# API + simulateur (autostart)
cd ../jumeaux-chauds && source .venv/bin/activate
SCENARIO=stress SIMULATION_AUTOSTART=1 MQTT_BROKER_HOST=localhost \
  uvicorn api.main:app --host 0.0.0.0 --port 8000
# Vérifier : curl http://localhost:8000/cluster/status
```

### 2.4 Régénérer données, features, modèles

```bash
cd juste-des-ventilateurs && source .venv/bin/activate
export MQTT_BROKER_HOST=localhost CLUSTER_ID=cluster_alpha API_BASE_URL=http://localhost:8000

# Ingestion (à vitesse accélérée côté jumeaux-chauds : PUT /simulation/speed)
python -m ingest.mqtt_subscriber --duration 100 --episode 001 --scenario stress
# Features
python -m features.pipeline --input data/raw/episode=001 \
  --output data/processed/episode=001 --config data/raw/episode=001/metadata.json
# Entraînement + métriques (annoncées + anticipatoires)
python -m evaluation.failure_prediction_eval --label failure_60s
python -m evaluation.fan_control_eval --label failure_60s --models all \
  --output evaluation/results/fan_control_results_failure_60s.json
```
> Les `data/` et `models/saved/` sont **versionnés** dans le dépôt : on peut
> sauter cette étape et utiliser directement l'existant.

### 2.5 Évaluation en boucle fermée (impact causal)

```bash
# jumeaux-chauds doit tourner
python -m evaluation.closed_loop_eval --scenario heatwave --duration 7200 \
  --dt 5 --speed 120 \
  --controllers baseline_fixed_0 native supervised baseline_fixed_4500
# Résultats : evaluation/results/closed_loop_results_heatwave.json
```

### 2.6 Le superviseur autonome (CLI)

```bash
python -m supervisor.supervisor --mode ml --duration 300        # pilote en ML
python -m supervisor.supervisor --mode ml --duration 60 --dry-run  # sans commande
```

### 2.7 Le dashboard de pilotage NOC

```bash
# jumeaux-chauds + modèles entraînés requis
python -m dashboard.noc_bridge --api-url http://localhost:8000 --port 8080
# Ouvrir http://localhost:8080
#  - badge vert "LIVE" = données réelles
#  - boutons RPM/AUTO par machine = commande manuelle
#  - toggle AUTO-PILOTE = boucle de décision live (le contrôleur commande)
```
Sans le pont, ouvrir `dashboard/noc.html` directement → mode **SIM** (démo).

### 2.8 Notebooks

```bash
jupyter notebook notebooks/07_closed_loop_evaluation.ipynb   # visualisation boucle fermée
```

---

## 3. Notes & limites

- **Tout le « live » dépend de jumeaux-chauds** (broker + API). Sans lui : tests
  unitaires + dashboard en mode SIM uniquement.
- `data/`, `models/saved/`, `evaluation/results/` sont **versionnés** (snapshot
  reproductible) ; ils peuvent être régénérés via la pipeline (§2.4).
- **Préavis ~14 s** (pannes fast-onset du simulateur) : limite assumée, documentée
  dans `rapport_analyse.md` (§9).
- **Équité boucle fermée** : obtenue en relançant le simulateur par contrôleur
  (le `soft_reset` ne remet pas les températures) — Phase 9bis = l'intégrer au runner.
