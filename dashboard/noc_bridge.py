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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("noc_bridge")
NOC_HTML = Path(__file__).resolve().parent / "noc.html"


def build_live(cluster: dict, buffer, predictor, feature_order) -> dict:
    """Transforme un snapshot cluster en payload pour le dashboard.

    Fonction pure (testable) : met à jour le buffer par machine, calcule le
    risque via le prédicteur si disponible, et retourne ::

        {"source": "...", "ts": "...", "byId": {machine_id: {temp, load, on,
         fans:[rpm,...], risk}}}
    """
    from evaluation.closed_loop_eval import machines_from_cluster

    by_id: dict[str, dict] = {}
    for mid, snap in machines_from_cluster(cluster).items():
        status = str(snap.get("status", "on"))
        fans = snap.get("fans", [])
        rpms = [int(f.get("rpm", 0)) for f in fans if isinstance(f, dict)] if isinstance(fans, list) else []

        risk = None
        if buffer is not None:
            buffer.update(mid, snap)
            if predictor is not None:
                try:
                    import pandas as pd
                    feats = buffer.get_features(mid)
                    X = pd.DataFrame([feats])
                    if feature_order is not None:
                        X = X.reindex(columns=feature_order, fill_value=0.0)
                    risk = round(float(predictor.predict_proba(X)[0, 1]) * 100.0, 1)
                except Exception as e:  # pragma: no cover - robustesse runtime
                    logger.debug("risk %s: %s", mid, e)

        feats = buffer.get_features(mid) if buffer is not None else {}
        by_id[mid] = {
            "temp": round(float(snap.get("temperature_c", feats.get("temperature_c", 0.0))), 2),
            "load": round(float(snap.get("load_estimated", feats.get("load_estimated", 0.0))), 3),
            "on":   status == "on",
            "status": status,
            "fans": rpms,
            "risk": risk,
        }

    return {
        "source": cluster.get("cluster_id", "jumeaux-chauds"),
        "ts": cluster.get("ts"),
        "byId": by_id,
    }


class _Handler(BaseHTTPRequestHandler):
    client = None
    buffer = None
    predictor = None
    feature_order = None

    def log_message(self, *a):  # silence per-request logs
        pass

    def do_GET(self):
        if self.path.startswith("/api/live"):
            self._live()
        elif self.path in ("/", "/index.html", "/noc.html"):
            self._html()
        elif self.path.startswith("/vendor/"):
            self._vendor()
        else:
            self.send_error(404)

    def _vendor(self):
        # sert dashboard/vendor/* (React/ReactDOM/Babel inlinés localement)
        name = Path(self.path.split("?", 1)[0]).name
        fpath = NOC_HTML.parent / "vendor" / name
        if not fpath.exists() or fpath.suffix != ".js":
            self.send_error(404)
            return
        body = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/javascript")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        if not NOC_HTML.exists():
            self.send_error(500, "noc.html introuvable")
            return
        body = NOC_HTML.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _live(self):
        try:
            cluster = self.client.get_cluster_status()
            payload = build_live(cluster, self.buffer, self.predictor, self.feature_order)
            body = json.dumps(payload).encode()
            self.send_response(200)
        except Exception as e:
            body = json.dumps({"error": str(e), "byId": {}}).encode()
            self.send_response(502)
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
    args = parser.parse_args()

    from supervisor.supervisor import JumeauxClient, load_predictor
    from supervisor.online_features import OnlineFeatureBuffer

    _Handler.client = JumeauxClient(args.api_url)
    _Handler.buffer = OnlineFeatureBuffer()
    _Handler.predictor = load_predictor("logistic", args.label)
    if _Handler.predictor is not None:
        try:
            from models.failure_prediction.splitter import TemporalSplitter
            _Handler.feature_order = list(TemporalSplitter().split()[0].columns)
        except Exception as e:
            logger.warning("ordre des features indisponible : %s", e)
    logger.info("predictor %s | dashboard sur http://localhost:%d",
                "chargé" if _Handler.predictor else "absent (risk=null)", args.port)

    ThreadingHTTPServer(("0.0.0.0", args.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
