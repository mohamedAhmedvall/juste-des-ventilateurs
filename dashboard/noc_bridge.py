"""Pont de données pour le dashboard NOC — Juste des Ventilateurs.

Sert le dashboard `dashboard/noc.html` et expose `/api/live`, qui agrège en
temps réel :
  - la télémétrie de jumeaux-chauds (`GET /cluster/status`)
  - le **risque de panne calculé par notre prédicteur ML** (régression logistique
    sur les features online)

Le dashboard (React, design NOC) interroge `/api/live` toutes les 2 s et remplace
sa simulation interne par ces données réelles (badge « LIVE » à l'écran). Si le
pont est indisponible, le dashboard retombe sur sa simulation (badge « SIM »).

Tout est en bibliothèque standard (http.server) côté serveur ; le predictor et
l'OnlineFeatureBuffer viennent du projet — lancer dans le venv du projet.

Usage :
    python -m dashboard.noc_bridge --api-url http://localhost:8000 --port 8080
    # puis ouvrir http://localhost:8080
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("noc_bridge")
NOC_HTML = Path(__file__).resolve().parent / "noc.html"


# Libellés lisibles pour quelques features (explicabilité)
_FEAT_LABELS = {
    "temperature_c": "Température", "sensor_temp_max": "Capteur max",
    "temp_delta_5s": "Δ temp 5s", "temp_delta_15s": "Δ temp 15s",
    "temp_delta_30s": "Δ temp 30s", "temp_rolling_std_30s": "Volatilité 30s",
    "temp_rolling_mean_30s": "Temp moy 30s", "margin_to_shutdown": "Marge au seuil",
    "margin_delta_30s": "Vitesse vers seuil", "load_estimated": "Charge",
    "load_rolling_mean_30s": "Charge moy 30s", "time_in_hot_zone_s": "Durée zone chaude",
    "power_w": "Puissance", "fan_rpm_mean": "RPM moyen",
    "rpm_rolling_mean_30s": "RPM moy 30s",
}

# Explications lisibles (légende) de chaque feature, envoyées au dashboard
_FEAT_DESC = {
    "temperature_c": "Température CPU courante (°C)",
    "sensor_temp_max": "Température max parmi les sondes (°C)",
    "temp_delta_5s": "Variation de température sur 5 s",
    "temp_delta_15s": "Variation de température sur 15 s",
    "temp_delta_30s": "Variation de température sur 30 s (vitesse de montée)",
    "temp_rolling_std_30s": "Instabilité thermique : écart-type de T sur 30 s",
    "temp_rolling_mean_30s": "Température moyenne glissante sur 30 s",
    "margin_to_shutdown": "Marge avant l'arrêt thermique (°C) — bas = risqué",
    "margin_delta_30s": "Vitesse d'approche du seuil d'arrêt",
    "load_estimated": "Charge de calcul estimée (0–1)",
    "load_rolling_mean_30s": "Charge moyenne sur 30 s",
    "time_in_hot_zone_s": "Temps passé en zone chaude (>80 % du seuil), en s",
    "power_w": "Puissance électrique consommée (W)",
    "fan_rpm_mean": "Vitesse moyenne des ventilateurs (RPM)",
    "rpm_rolling_mean_30s": "RPM moyen glissant sur 30 s",
}


def _explain(predictor, X, topn: int = 5) -> list[dict]:
    """Contributions des features au risque (régression logistique).

    contribution_i = coef_i × feature_i_standardisée. Positif = pousse vers la
    panne, négatif = rassure. Retourne les `topn` plus fortes (en valeur absolue).
    Best-effort : [] si le modèle n'expose pas de coefficients linéaires.
    """
    try:
        import numpy as np
        pipe = getattr(predictor, "_pipeline", None)
        if pipe is None:
            return []
        scaler = pipe.named_steps.get("scaler")
        clf = pipe.named_steps.get("clf")
        coefs = []
        if hasattr(clf, "calibrated_classifiers_"):
            for cc in clf.calibrated_classifiers_:
                est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
                if est is not None and hasattr(est, "coef_"):
                    coefs.append(np.asarray(est.coef_[0]))
        elif hasattr(clf, "coef_"):
            coefs.append(np.asarray(clf.coef_[0]))
        if not coefs:
            return []
        coef = np.mean(coefs, axis=0)
        x = X.fillna(0.0).values[0]
        z = scaler.transform(X.fillna(0.0).values)[0] if scaler is not None else x
        contrib = coef * z
        names = list(X.columns)
        order = np.argsort(np.abs(contrib))[::-1][:topn]
        out = []
        for i in order:
            if abs(float(contrib[i])) < 1e-6:
                continue
            out.append({
                "feature": _FEAT_LABELS.get(names[i], names[i]),
                "desc": _FEAT_DESC.get(names[i], "Feature du modèle de prédiction"),
                "value": round(float(x[i]), 2),
                "contribution": round(float(contrib[i]), 3),
            })
        return out
    except Exception:  # pragma: no cover - robustesse runtime
        return []


def build_live(cluster: dict, buffer, predictor, feature_order, controller=None) -> dict:
    """Transforme un snapshot cluster en payload pour le dashboard.

    Fonction pure (testable) : met à jour le buffer par machine, calcule le
    risque ML + son explicabilité + la consigne RPM recommandée, et retourne ::

        {"source", "ts", "metrics", "byId": {machine_id: {temp, load, on, status,
         role, fans:[rpm,...], risk, rpm_reco, explain:[{feature,value,contribution}]}}}
    """
    from evaluation.closed_loop_eval import machines_from_cluster

    by_id: dict[str, dict] = {}
    for mid, snap in machines_from_cluster(cluster).items():
        status = str(snap.get("status", "on"))
        fans = snap.get("fans", [])
        rpms = [int(f.get("rpm", 0)) for f in fans if isinstance(f, dict)] if isinstance(fans, list) else []

        risk = None
        explain: list[dict] = []
        rpm_reco = None
        feats = {}
        if buffer is not None:
            buffer.update(mid, snap)
            feats = buffer.get_features(mid)
            import pandas as pd
            X = pd.DataFrame([feats])
            if feature_order is not None:
                X = X.reindex(columns=feature_order, fill_value=0.0)
            if predictor is not None:
                try:
                    risk = round(float(predictor.predict_proba(X)[0, 1]) * 100.0, 1)
                    explain = _explain(predictor, X)
                except Exception as e:  # pragma: no cover
                    logger.debug("risk %s: %s", mid, e)
            if controller is not None:
                try:
                    import numpy as np
                    rr = controller.decide_batch(X, risk_scores=np.array([(risk or 0) / 100.0]))
                    rpm_reco = int(rr[0])
                except Exception:
                    try:
                        rpm_reco = int(controller.decide_batch(X)[0])
                    except Exception:  # pragma: no cover
                        rpm_reco = None

        by_id[mid] = {
            "temp": round(float(snap.get("temperature_c", feats.get("temperature_c", 0.0))), 2),
            "load": round(float(snap.get("load_estimated", feats.get("load_estimated", 0.0))), 3),
            "on":   status == "on",
            "status": status,
            "role": str(snap.get("role", "worker")),
            "fans": rpms,
            "risk": risk,
            "rpm_reco": rpm_reco,
            "explain": explain,
        }

    return {
        "source": cluster.get("cluster_id", "jumeaux-chauds"),
        "ts": cluster.get("ts"),
        "metrics": cluster.get("metrics", {}),
        "byId": by_id,
    }


def change_scenario(client, scenario: str, speed: float | None = None) -> bool:
    """Change le scénario du simulateur jumeaux-chauds (injection depuis le dashboard).

    PUT /simulation/scenario, remet le temps/énergie à zéro, et règle la vitesse.
    Utilise la session httpx du client si dispo (testable via un faux client).
    """
    base = getattr(client, "base_url", "")
    sess = getattr(client, "_client", None)

    def _send(method: str, path: str, body: dict | None = None) -> bool:
        url = f"{base}{path}"
        try:
            if sess is not None:
                return sess.request(method, url, json=body).status_code < 300
            import httpx
            return httpx.request(method, url, json=body, timeout=10.0).status_code < 300
        except Exception as e:  # pragma: no cover - réseau
            logger.warning("%s %s : %s", method, path, e)
            return False

    ok = _send("PUT", "/simulation/scenario", {"scenario": scenario})
    _send("POST", "/simulation/speed/reset")
    if speed:
        _send("PUT", "/simulation/speed", {"speed_multiplier": speed})
    return ok


def command_machine(client, machine_id: str, rpm: int | None = None, mode: str | None = None) -> bool:
    """Envoie une commande de ventilation à jumeaux-chauds (pilotage actif).

    - mode="auto"  : repasse les fans en régulation native.
    - rpm fourni   : bascule en manuel puis applique la consigne RPM.
    Fonction fine et testable (client mockable).
    """
    ok = True
    if mode is not None:
        ok = client.set_fan_mode(machine_id, mode) and ok
    if rpm is not None:
        ok = client.set_fan_mode(machine_id, "manual") and ok
        ok = client.set_fan_speed(machine_id, int(rpm)) and ok
    return ok


class BridgeState:
    """État partagé du pont : modèles, dernier instantané, mode auto-pilote."""
    def __init__(self, client, buffer, predictor, feature_order, controller):
        self.client = client
        self.buffer = buffer
        self.predictor = predictor
        self.feature_order = feature_order
        self.controller = controller
        self.latest: dict = {"byId": {}}
        self.autopilot = False
        self.lock = threading.Lock()

    def refresh(self) -> dict:
        cluster = self.client.get_cluster_status()
        payload = build_live(cluster, self.buffer, self.predictor,
                             self.feature_order, controller=self.controller)
        payload["autopilot"] = self.autopilot
        with self.lock:
            self.latest = payload
        return payload

    def apply_autopilot(self, payload: dict) -> int:
        """Applique la consigne du contrôleur (rpm_reco) à chaque machine active.
        Retourne le nombre de commandes envoyées."""
        n = 0
        for mid, m in payload.get("byId", {}).items():
            if m.get("on") and m.get("rpm_reco") is not None:
                command_machine(self.client, mid, rpm=m["rpm_reco"])
                n += 1
        return n


def run_loop(state: BridgeState, interval: float = 2.0):
    """Boucle de fond : rafraîchit l'état et, si l'auto-pilote est actif,
    applique les consignes du contrôleur (la boucle de décision live)."""
    while True:
        try:
            payload = state.refresh()
            if state.autopilot:
                state.apply_autopilot(payload)
        except Exception as e:  # pragma: no cover
            logger.debug("loop: %s", e)
        time.sleep(interval)


class _Handler(BaseHTTPRequestHandler):
    state: BridgeState = None

    def log_message(self, *a):  # silence per-request logs
        pass

    def do_GET(self):
        if self.path.startswith("/api/live"):
            self._json(self.state.latest if self.state.latest.get("byId") else self.state.refresh())
        elif self.path in ("/", "/index.html", "/noc.html"):
            self._html()
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if self.path.startswith("/api/command"):
            mid = body.get("machine_id")
            if not mid:
                self._json({"ok": False, "error": "machine_id requis"}, 400); return
            ok = command_machine(self.state.client, mid,
                                 rpm=body.get("rpm"), mode=body.get("mode"))
            self._json({"ok": bool(ok)})
        elif self.path.startswith("/api/autopilot"):
            self.state.autopilot = bool(body.get("enabled"))
            logger.info("auto-pilote %s", "ACTIVÉ" if self.state.autopilot else "désactivé")
            self._json({"ok": True, "autopilot": self.state.autopilot})
        elif self.path.startswith("/api/scenario"):
            sc = body.get("scenario")
            if not sc:
                self._json({"ok": False, "error": "scenario requis"}, 400); return
            ok = change_scenario(self.state.client, sc, speed=body.get("speed"))
            logger.info("scénario injecté : %s (ok=%s)", sc, ok)
            self._json({"ok": bool(ok), "scenario": sc})
        else:
            self.send_error(404)

    def _html(self):
        if not NOC_HTML.exists():
            self.send_error(500, "noc.html introuvable"); return
        body = NOC_HTML.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Pont de données dashboard NOC")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--label", default="failure_60s")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="cadence (s) de la boucle de rafraîchissement / pilotage")
    args = parser.parse_args()

    from supervisor.supervisor import JumeauxClient, load_predictor
    from supervisor.online_features import OnlineFeatureBuffer

    from supervisor.supervisor import load_controller

    predictor = load_predictor("logistic", args.label)
    controller = load_controller("supervised")
    feature_order = None
    if predictor is not None:
        try:
            from models.failure_prediction.splitter import TemporalSplitter
            feature_order = list(TemporalSplitter().split()[0].columns)
        except Exception as e:
            logger.warning("ordre des features indisponible : %s", e)

    _Handler.state = BridgeState(
        client=JumeauxClient(args.api_url), buffer=OnlineFeatureBuffer(),
        predictor=predictor, feature_order=feature_order, controller=controller,
    )
    threading.Thread(target=run_loop, args=(_Handler.state, args.interval), daemon=True).start()

    logger.info("predictor %s | controller %s | dashboard sur http://localhost:%d (pilotage actif)",
                "chargé" if predictor else "absent (risk=null)",
                "chargé" if controller else "absent", args.port)
    ThreadingHTTPServer(("0.0.0.0", args.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
