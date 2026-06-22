# Dashboard NOC — Juste des Ventilateurs

Dashboard de supervision thermique (thème NOC sombre) branché sur les données
**réelles** de jumeaux-chauds + le **risque de panne du prédicteur ML**.

## Fichiers

- `noc.html` — dashboard **autonome, sans aucune dépendance** (HTML/CSS/JS + SVG,
  pas de React/CDN). Il interroge `/api/live` toutes les 2 s :
  - **pont disponible** → données live, badge vert « ● LIVE » : par machine,
    température vs seuil, jauge de risque ML, charge, RPM des ventilateurs ;
    KPIs cluster (PUE, énergie, coût, risque max, machines actives) ;
  - **pont absent** → jeu de démonstration interne, badge gris « SIM ».
  On peut donc l'ouvrir directement dans un navigateur (mode SIM) ou via le pont
  (mode LIVE).
- `noc_bridge.py` — pont (stdlib `http.server`) : sert `noc.html` et expose
  `/api/live` = `GET /cluster/status` de jumeaux-chauds + risque via la régression
  logistique + métriques cluster.

## Lancer

```bash
# 1. jumeaux-chauds (API :8000)
#    cd ../jumeaux-chauds && SCENARIO=stress SIMULATION_AUTOSTART=1 uvicorn api.main:app --port 8000
# 2. le pont (venv du projet, modèle logistic_failure_60s entraîné)
python -m dashboard.noc_bridge --api-url http://localhost:8000 --port 8080
# 3. ouvrir http://localhost:8080   -> badge vert "● LIVE"
```

## `/api/live`

```json
{"source":"cluster_alpha","ts":"...",
 "metrics":{"pue_effective":1.18,"energy_kwh_total":3.2,"cost_eur_total":0.9},
 "byId":{"srv-worker-01":{"temp":71.5,"load":0.6,"on":true,"status":"on",
                          "role":"worker","fans":[3200,3100],"risk":42.0}}}
```

`risk` = probabilité de panne (%) du prédicteur ; `null` si le modèle est absent.
La logique d'agrégation (`build_live`) est testée dans
`tests/test_phase9c_dashboard.py`.
