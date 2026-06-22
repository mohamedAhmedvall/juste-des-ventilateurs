# Dashboard NOC — Juste des Ventilateurs

Dashboard de supervision (design NOC, React/D3) branché sur les données **réelles**
de jumeaux-chauds + le **risque de panne du prédicteur ML**.

## Fichiers

- `noc.html` — le dashboard (design standalone). En l'absence de pont de données,
  il tourne sur sa **simulation interne** (badge « SIM »). Quand le pont répond,
  il affiche les **données live** (badge « LIVE ») : températures, RPM, statut et
  risque ML par machine.
- `noc_bridge.py` — pont : sert `noc.html` et expose `/api/live` (agrège
  `GET /cluster/status` de jumeaux-chauds + risque via la régression logistique).

## Lancer

```bash
# 1. jumeaux-chauds doit tourner (API :8000)
#    cd ../jumeaux-chauds && SCENARIO=stress SIMULATION_AUTOSTART=1 uvicorn api.main:app --port 8000

# 2. le pont (dans le venv du projet, modèle logistic_failure_60s entraîné)
python -m dashboard.noc_bridge --api-url http://localhost:8000 --port 8080

# 3. ouvrir http://localhost:8080
```

Sans le pont, on peut aussi ouvrir `noc.html` directement dans un navigateur
(mode SIM, pour la démo visuelle).

## Données exposées par `/api/live`

```json
{"source": "cluster_alpha", "ts": "...",
 "byId": {"srv-worker-01": {"temp": 71.5, "load": 0.6, "on": true,
                            "status": "on", "fans": [3200, 3100], "risk": 42.0}}}
```

`risk` est la probabilité de panne (%) du prédicteur ; `null` si le modèle est
absent (lancer l'entraînement d'abord). La logique d'agrégation
(`build_live`) est testée dans `tests/test_phase9c_dashboard.py`.
