"""Superviseur de régulation thermique -- Juste des Ventilateurs.

Boucle principale de décision en temps réel :
  1. Recevoir la télémétrie via MQTT (consumer asyncio) -> OnlineFeatureBuffer
  2. Tous les decision_interval_ticks ticks simulés :
     a. Extraire les features enrichies (fenêtres glissantes)
     b. Évaluer le risque de panne (prédicteur logistique)
     c. Décider la consigne RPM (contrôleur supervisé)
     d. Override RPM_HIGH si risk_score >= RISK_THRESHOLD
     e. Envoyer la commande via REST (PUT /machines/{id}/fan_speed)
     f. Logger la décision
  3. Fallback : si MQTT indisponible, lire GET /cluster/status REST

Ce module peut aussi être utilisé en mode "offline replay" pour
rejouer un dataset et comparer les décisions du superviseur avec
les décisions oracle (voir evaluation/benchmark.py).

Usage :
    python -m supervisor.supervisor
    python -m supervisor.supervisor --mode ml --duration 300
    python -m supervisor.supervisor --mode threshold --duration 300
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supervisor.decision_logger import DecisionLogger
from supervisor.online_features import OnlineFeatureBuffer

# ---------------------------------------------------------------------------
# Configuration des logs -- silencer httpx, reformater le superviseur
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
)
logger = logging.getLogger("supervisor")

# httpx est très verbeux (une ligne par requête HTTP 200) -- passer en WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

RPM_LEVELS      = [800, 1500, 2500, 3500, 4500]
RPM_HIGH        = 4500    # RPM d'urgence quand risk_score > threshold
RPM_DEFAULT     = 2500    # RPM de sécurité si aucune décision possible
RPM_MIN         = 800     # Plancher : ventilation minimale même à froid
RISK_THRESHOLD  = 0.60    # Seuil de surcharge risque -> RPM_HIGH
HOT30S_THRESHOLD = float(os.environ.get("HOT30S_THRESHOLD", "0.5"))  # Override surchauffe

# Seuil de risk_score à partir duquel on logue une machine en INFO
RISK_LOG_THRESHOLD = float(os.environ.get("RISK_LOG_THRESHOLD", "0.05"))

# Features attendues par le prédicteur -- sous-ensemble disponible online
ONLINE_FEATURES = [
    "temperature_c", "sensor_temp_max", "sensor_temp_mean",
    "power_w", "energy_kwh", "fan_rpm_mean",
    "load_estimated",
]


# ---------------------------------------------------------------------------
# Client REST jumeaux-chauds
# ---------------------------------------------------------------------------

class JumeauxClient:
    """Client REST léger vers l'API jumeaux-chauds."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        try:
            import httpx
            self._client = httpx.Client(timeout=timeout)
        except ImportError:
            self._client = None
        logger.info("JumeauxClient -> %s", self.base_url)

    def get_cluster_status(self) -> dict:
        return self._get("/cluster/status")

    def get_speed_multiplier(self) -> float:
        """Lit le speed_multiplier depuis /simulation/speed.
        Fallback sur /cluster/status pour retrocompatibilite.
        """
        info = self._get("/simulation/speed")
        if info and "speed_multiplier" in info:
            return float(info["speed_multiplier"])
        status = self._get("/cluster/status")
        return float(status.get("speed_multiplier", 1.0))

    def set_fan_speed(self, machine_id: str, rpm: int, fan_indices: list[int] | None = None) -> bool:
        if fan_indices is None:
            fan_indices = [0, 1]
        ok = True
        for idx in fan_indices:
            try:
                url  = f"{self.base_url}/machines/{machine_id}/fan_speed"
                body = {"fan_idx": idx, "rpm": rpm}
                if self._client is not None:
                    resp = self._client.put(url, json=body)
                    ok = ok and resp.status_code < 300
                else:
                    import json as _json, urllib.request
                    data = _json.dumps(body).encode()
                    req  = urllib.request.Request(url, data=data, method="PUT",
                                                  headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        ok = ok and r.status < 300
            except Exception as e:
                logger.warning("set_fan_speed(%s, fan=%d, %d) échoué : %s", machine_id, idx, rpm, e)
                ok = False
        return ok

    def set_fan_mode(self, machine_id: str, mode: str, fan_indices: list[int] | None = None) -> bool:
        if fan_indices is None:
            fan_indices = [0, 1]
        ok = True
        for idx in fan_indices:
            try:
                url  = f"{self.base_url}/machines/{machine_id}/fan_mode"
                body = {"fan_idx": idx, "mode": mode}
                if self._client is not None:
                    resp = self._client.put(url, json=body)
                    ok = ok and resp.status_code < 300
                else:
                    import json as _json, urllib.request
                    data = _json.dumps(body).encode()
                    req  = urllib.request.Request(url, data=data, method="PUT",
                                                  headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        ok = ok and r.status < 300
            except Exception as e:
                logger.warning("set_fan_mode(%s, fan=%d, %s) échoué : %s", machine_id, idx, mode, e)
                ok = False
        return ok

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        try:
            if self._client is not None:
                resp = self._client.get(url)
                resp.raise_for_status()
                return resp.json()
            else:
                import json as _json, urllib.request
                with urllib.request.urlopen(url, timeout=self.timeout) as r:
                    return _json.loads(r.read())
        except Exception as e:
            logger.warning("GET %s échoué : %s", path, e)
            return {}

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------------------------------

def load_predictor(model_name: str = "logistic", label: str = "failure_60s"):
    import joblib as _joblib
    saved_path = ROOT / "models" / "failure_prediction" / "saved" / f"{model_name}_{label}.joblib"
    if not saved_path.exists():
        logger.warning("Prédicteur absent : %s. risk_score=0 partout.", saved_path)
        return None
    from models.failure_prediction.logistic_regression import LogisticPredictor
    predictor = LogisticPredictor()
    raw = _joblib.load(str(saved_path))
    if isinstance(raw, dict):
        pipeline_val = (raw.get("pipeline") or raw.get("model") or raw.get("estimator")
                        or next((v for v in raw.values() if hasattr(v, "predict_proba")), None))
        if pipeline_val is None:
            raise KeyError(f"Aucun objet sklearn dans le joblib. Clés: {list(raw.keys())}")
        predictor._pipeline = pipeline_val
        predictor.threshold = raw.get("threshold", predictor.threshold)
        predictor.C         = raw.get("C", predictor.C)
    else:
        # Objet direct : sklearn pipeline/estimator ou XGBClassifier wrapper
        if hasattr(raw, "predict_proba"):
            predictor._pipeline = raw
        else:
            # Booster XGBoost natif ou autre objet sans predict_proba
            logger.warning(
                "Prédicteur '%s' : objet de type %s sans predict_proba. "
                "Utilisez le modèle 'logistic' pour le superviseur temps-réel.",
                model_name, type(raw).__name__,
            )
            return None
    logger.info("Prédicteur chargé : %s", saved_path.name)
    return predictor


def load_controller(model_name: str = "supervised"):
    saved_dir = ROOT / "models" / "fan_control" / "saved"
    paths = {
        "supervised":          (saved_dir / "supervised.joblib",          "models.fan_control.supervised_controller", "SupervisedController", "load"),
        "score_controller":    (saved_dir / "score_controller.json",       "models.fan_control.score_controller",      "ScoreController",      "load"),
        "baseline_pid":        (saved_dir / "baseline_pid.json",           "models.fan_control.baseline_pid",          "PIDController",        "load"),
        "baseline_threshold":  (saved_dir / "baseline_threshold.json",     "models.fan_control.baseline_threshold",    "ThresholdFanController","load"),
    }
    if model_name not in paths:
        logger.warning("Contrôleur inconnu : %s. RPM_DEFAULT utilisé.", model_name)
        return None
    path, mod, cls, meth = paths[model_name]
    if not path.exists():
        logger.warning("Contrôleur absent : %s. RPM_DEFAULT utilisé.", path)
        return None
    import importlib
    m = importlib.import_module(mod)
    ctrl = getattr(m, cls).load(str(path))
    logger.info("Contrôleur chargé : %s", path.name)
    return ctrl


# ---------------------------------------------------------------------------
# Helpers REST fallback (lecture cluster en mode sans MQTT)
# ---------------------------------------------------------------------------

def _machines_from_cluster(cluster: dict) -> dict[str, dict]:
    """Normalise machines dict ou liste -> {machine_id: snapshot}."""
    machines_raw = cluster.get("machines", {})
    if isinstance(machines_raw, list):
        return {m.get("machine_id", m.get("id", f"machine_{i}")): m
                for i, m in enumerate(machines_raw)}
    return machines_raw


def _fan_indices(snapshot: dict) -> list[int]:
    fans = snapshot.get("fans", [])
    if isinstance(fans, list):
        return [f["idx"] for f in fans if "idx" in f]
    return list(range(len(fans))) if fans else [0, 1]


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

_DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"


def _DEFAULT_BROKER_HOST() -> str:
    return os.environ.get("MQTT_BROKER_HOST", "localhost")

def _DEFAULT_BROKER_PORT() -> int:
    return int(os.environ.get("MQTT_BROKER_PORT", "1883"))

def _DEFAULT_DECISION_INTERVAL() -> int:
    return int(os.environ.get("DECISION_INTERVAL_TICKS", "5"))


class Supervisor:
    """Superviseur temps réel couplant prédicteur + contrôleur.

    Parameters
    ----------
    mode                    : "ml" | "threshold" | "native"
    predictor_name          : nom du modèle prédictif
    controller_name         : nom du contrôleur
    label                   : label de panne cible
    risk_threshold          : seuil risk_score -> RPM_HIGH
    api_url                 : URL de l'API jumeaux-chauds
    mqtt_host / mqtt_port   : broker MQTT (Option E)
    decision_interval_ticks : décision toutes les N ticks simulés (MQTT)
    decision_interval_s     : intervalle fallback REST (secondes)
    log_dir / run_name      : log JSONL
    dry_run                 : ne pas envoyer de commandes REST
    """

    def __init__(
        self,
        mode:                    str   = "ml",
        predictor_name:          str   = "logistic",
        controller_name:         str   = "supervised",
        label:                   str   = "failure_60s",
        risk_threshold:          float = RISK_THRESHOLD,
        api_url:                 str   = "http://localhost:8000",
        mqtt_host:               str   = _DEFAULT_BROKER_HOST(),
        mqtt_port:               int   = _DEFAULT_BROKER_PORT(),
        decision_interval_ticks: int   = _DEFAULT_DECISION_INTERVAL(),
        decision_interval_s:     float = 5.0,
        log_dir:                 str | Path = _DEFAULT_LOG_DIR,
        run_name:                str   = "supervisor",
        dry_run:                 bool  = False,
    ) -> None:
        self.mode                    = mode
        self.label                   = label
        self.risk_threshold          = risk_threshold
        self.decision_interval_s     = decision_interval_s
        self.decision_interval_ticks = decision_interval_ticks
        self.dry_run                 = dry_run

        self.client     = JumeauxClient(api_url)
        self.dec_logger = DecisionLogger(log_dir=log_dir, run_name=run_name)

        self.predictor  = load_predictor(predictor_name, label) if mode == "ml" else None
        self.controller = load_controller(controller_name)       if mode == "ml" else None

        # Prédicteur hot_30s pour l'override surchauffe imminente (Phase 7.4)
        self.hot30s_predictor = load_predictor(predictor_name, "hot_30s") if mode == "ml" else None
        if self.hot30s_predictor is not None:
            logger.info("Prédicteur hot_30s chargé (override surchauffe >= %.2f)", HOT30S_THRESHOLD)
        else:
            logger.info("Prédicteur hot_30s absent -- override surchauffe désactivé")

        # Ordre des features attendu par le modele (depuis le splitter d'entrainement)
        self._feature_order: list[str] | None = None
        if self.predictor is not None:
            try:
                from models.failure_prediction.splitter import TemporalSplitter
                r = TemporalSplitter().split()
                self._feature_order = list(r[0].columns)
                logger.info("Feature order charge : %d features", len(self._feature_order))
            except Exception as e:
                logger.warning("Impossible de charger l'ordre des features : %s", e)

        self._feat_buffer = OnlineFeatureBuffer()
        self._prev_rpm: dict[str, int] = {}

        # Consumer MQTT (Option E)
        from supervisor.mqtt_telemetry import MqttTelemetryConsumer
        self._mqtt = MqttTelemetryConsumer(
            buffer=self._feat_buffer,
            broker_host=mqtt_host,
            broker_port=mqtt_port,
            decision_interval_ticks=decision_interval_ticks,
        )

        # Déduplication des warnings répétitifs
        self._warn_counts: dict[str, int] = {}

        logger.info("Supervisor prêt -- mode=%s  dry_run=%s", mode, dry_run)

    # ------------------------------------------------------------------
    # Prédiction et décision
    # ------------------------------------------------------------------

    def _predict_risk(self, state_series: "pd.Series") -> float:
        if self.predictor is None:
            return 0.0
        import pandas as pd
        X = pd.DataFrame([state_series])
        # Reordonner les colonnes selon l'ordre d'entrainement
        if self._feature_order is not None:
            X = X.reindex(columns=self._feature_order, fill_value=0.0)
        try:
            proba = self.predictor.predict_proba(X)
            risk  = float(proba[0, 1])
            logger.debug(
                "predict_proba  T=%.1f  d5s=%.3f  d30s=%.3f  margin=%.1f  load=%.2f  -> risk=%.4f",
                float(state_series.get("temperature_c", 0)),
                float(state_series.get("temp_delta_5s", 0)),
                float(state_series.get("temp_delta_30s", 0)),
                float(state_series.get("margin_to_shutdown", 0)),
                float(state_series.get("load_estimated", 0)),
                risk,
            )
            return risk
        except Exception as e:
            logger.warning("predict_proba echoue : %s", e)
            return 0.0

    def _predict_hot30s(self, state_series: "pd.Series") -> float:
        """Score de surchauffe imminente (hot_30s). Retourne 0.0 si prédicteur absent."""
        if self.hot30s_predictor is None:
            return 0.0
        import pandas as pd
        X = pd.DataFrame([state_series])
        if self._feature_order is not None:
            X = X.reindex(columns=self._feature_order, fill_value=0.0)
        try:
            proba = self.hot30s_predictor.predict_proba(X)
            return float(proba[0, 1])
        except Exception as e:
            logger.debug("predict_proba hot_30s echoue : %s", e)
            return 0.0

    def _decide_rpm(self, state_series: "pd.Series", risk_score: float,
                    hot30s_score: float = 0.0) -> tuple[int, bool, bool]:
        """Retourne (rpm, risk_override, hot30s_override)."""
        if risk_score >= self.risk_threshold:
            return RPM_HIGH, True, False
        if hot30s_score >= HOT30S_THRESHOLD:
            logger.debug("hot30s override : score=%.3f >= %.2f -> RPM_HIGH", hot30s_score, HOT30S_THRESHOLD)
            return RPM_HIGH, False, True
        if self.mode == "native":
            return -1, False, False
        if self.controller is None:
            return RPM_DEFAULT, False, False
        import pandas as pd
        X = pd.DataFrame([state_series])
        risk_arr = np.array([risk_score])
        try:
            rpms = self.controller.decide_batch(X, risk_scores=risk_arr)
            return max(int(rpms[0]), RPM_MIN), False, False
        except TypeError:
            try:
                rpms = self.controller.decide_batch(X)
                return max(int(rpms[0]), RPM_MIN), False, False
            except Exception as e:
                logger.debug("decide_batch echoue : %s", e)
                return RPM_DEFAULT, False, False

    # ------------------------------------------------------------------
    # Traitement d'une machine
    # ------------------------------------------------------------------

    def _process_machine(self, machine_id: str, snapshot: dict | None = None) -> dict:
        """Prédit, décide, envoie, logue pour une machine.

        Si snapshot est fourni (fallback REST), il alimente d'abord le buffer.
        En mode MQTT, le buffer est déjà alimenté par le consumer.
        """
        if snapshot is not None:
            self._feat_buffer.update(machine_id, snapshot)

        state      = self._feat_buffer.get_features(machine_id)
        risk       = self._predict_risk(state)
        hot30s     = self._predict_hot30s(state)
        rpm, risk_ov, hot30s_ov = self._decide_rpm(state, risk, hot30s)
        prev_rpm   = self._prev_rpm.get(machine_id, RPM_DEFAULT)

        temp_c = float(state.get("temperature_c", 0.0))

        entry = {
            "ts":              datetime.now(timezone.utc).isoformat(),
            "machine_id":      machine_id,
            "temperature_c":   temp_c,
            "status":          str(state.get("status", "unknown") if snapshot is None
                                   else snapshot.get("status", "unknown")),
            "fan_rpm_mean":    float(state.get("fan_rpm_mean", 0.0)),
            "risk_score":      round(risk, 4),
            "hot30s_score":    round(hot30s, 4),
            "rpm_decided":     rpm,
            "rpm_previous":    prev_rpm,
            "mode":            self.mode,
            "risk_override":   risk_ov,
            "hot30s_override": hot30s_ov,
        }

        # Envoyer la commande seulement si RPM change
        if rpm >= 0 and rpm != prev_rpm:
            if not self.dry_run:
                indices = _fan_indices(snapshot) if snapshot else [0, 1]
                ok = self.client.set_fan_speed(machine_id, rpm, fan_indices=indices)
                entry["command_sent"] = ok
            else:
                entry["command_sent"] = None
            self._prev_rpm[machine_id] = rpm

            # Log INFO : changement RPM ou risque élevé
            override_tag = (" [RISK OVERRIDE]"  if risk_ov   else
                            " [HOT30S OVERRIDE]" if hot30s_ov else "")
            dry_tag = " [DRY RUN]" if self.dry_run else ""
            logger.info(
                "  %-20s  T=%5.1f degC  risk=%.2f  hot30s=%.2f  RPM %d->%d%s%s",
                machine_id, temp_c, risk, hot30s, prev_rpm, rpm, override_tag, dry_tag,
            )
        elif risk > RISK_LOG_THRESHOLD or hot30s > HOT30S_THRESHOLD * 0.7:
            # Log INFO : risque ou surchauffe notable même sans changement RPM
            logger.info(
                "  %-20s  T=%5.1f degC  risk=%.2f  hot30s=%.2f  RPM %d (stable)",
                machine_id, temp_c, risk, hot30s, rpm if rpm >= 0 else prev_rpm,
            )
        else:
            logger.debug(
                "  %-20s  T=%5.1f degC  risk=%.2f  RPM %d",
                machine_id, temp_c, risk, rpm if rpm >= 0 else prev_rpm,
            )

        entry["command_sent"] = entry.get("command_sent")
        self.dec_logger.log(entry)
        return entry

    # ------------------------------------------------------------------
    # Cycle de décision (commun MQTT et REST)
    # ------------------------------------------------------------------

    def _decision_cycle_mqtt(self) -> list[dict]:
        """Décision sur les machines connues du buffer (mode MQTT)."""
        results = []
        for machine_id in self._feat_buffer.machines():
            try:
                result = self._process_machine(machine_id)
                results.append(result)
            except Exception as e:
                logger.error("Erreur traitement %s : %s", machine_id, e)

        # Résumé cluster en INFO
        if results:
            t_max    = max(r["temperature_c"] for r in results)
            risk_max = max(r["risk_score"] for r in results)
            n_on     = len(results)
            elapsed  = getattr(self, "_t_elapsed", 0)
            speed    = getattr(self, "_speed_multiplier", 1.0)
            logger.info(
                "[t=%ds speed=%.0fx]  cluster -- %d on  T_max=%.1f degC  risk_max=%.2f",
                elapsed, speed, n_on, t_max, risk_max,
            )
        return results

    def _decision_cycle_rest(self) -> list[dict]:
        """Cycle de décision en fallback REST (alimenter buffer + décider)."""
        cluster = self.client.get_cluster_status()
        if not cluster:
            self._log_warning_dedup("cluster_empty", "cluster/status vide -- skip cycle")
            return []

        machines = _machines_from_cluster(cluster)
        if not machines:
            return []

        results = []
        for machine_id, snapshot in machines.items():
            if snapshot.get("status") == "off":
                continue
            try:
                result = self._process_machine(machine_id, snapshot=snapshot)
                results.append(result)
            except Exception as e:
                logger.error("Erreur traitement %s : %s", machine_id, e)

        if results:
            t_max    = max(r["temperature_c"] for r in results)
            risk_max = max(r["risk_score"] for r in results)
            n_on     = len(results)
            elapsed  = getattr(self, "_t_elapsed", 0)
            speed    = getattr(self, "_speed_multiplier", 1.0)
            logger.info(
                "[t=%ds speed=%.0fx]  cluster -- %d on  T_max=%.1f degC  risk_max=%.2f  [REST]",
                elapsed, speed, n_on, t_max, risk_max,
            )
        return results

    # ------------------------------------------------------------------
    # Boucle principale asynchrone
    # ------------------------------------------------------------------

    async def _run_async(self, duration_s: float | None = None) -> None:
        """Boucle principale async : consumer MQTT + loop de décision."""
        logger.info("=== Supervisor démarré -- mode=%s ===", self.mode)
        if self.dry_run:
            logger.info("  [DRY RUN] Aucune commande ne sera envoyée")

        # Lire speed_multiplier depuis l'API
        try:
            self._speed_multiplier = self.client.get_speed_multiplier()
            logger.info("  speed_multiplier=%.0fx (simulateur)", self._speed_multiplier)
        except Exception:
            self._speed_multiplier = 1.0

        # Passage en mode manual
        if self.mode != "native" and not self.dry_run:
            self._set_all_manual()

        # Démarrer le consumer MQTT en arrière-plan
        mqtt_task = asyncio.create_task(self._mqtt.run())
        mqtt_ready = await self._mqtt.wait_ready(timeout=10.0)

        if mqtt_ready:
            logger.info("  Mode MQTT -- buffer alimenté à la cadence simulée")
            await self._loop_mqtt(duration_s)
        else:
            logger.info("  Mode REST fallback -- lecture API toutes les %.0fs", self.decision_interval_s)
            await self._loop_rest(duration_s)

        # Arrêt propre
        self._mqtt.stop()
        mqtt_task.cancel()
        try:
            await mqtt_task
        except asyncio.CancelledError:
            pass

        if self.mode != "native" and not self.dry_run:
            self._set_all_auto()
        self.client.close()
        self.dec_logger.close()
        logger.info("=== Supervisor arrêté ===")

    def _refresh_speed(self) -> None:
        """Relit speed_multiplier depuis /simulation/speed et logue si changement."""
        try:
            new_speed = self.client.get_speed_multiplier()
        except Exception:
            return
        if new_speed != self._speed_multiplier:
            logger.info("  speed_multiplier change : %.0fx -> %.0fx",
                        self._speed_multiplier, new_speed)
            self._speed_multiplier = new_speed

    async def _loop_mqtt(self, duration_s: float | None) -> None:
        """Boucle de decision pilotee par les ticks MQTT."""
        t_start  = time.monotonic()
        self._t_elapsed = 0
        _cycle = 0
        _speed_refresh_interval = 10

        logger.info("  En attente des premiers ticks MQTT...")
        await asyncio.sleep(2.0)

        try:
            while True:
                elapsed = time.monotonic() - t_start
                self._t_elapsed = int(elapsed * self._speed_multiplier)
                if duration_s is not None and elapsed >= duration_s:
                    break

                _cycle += 1
                if _cycle % _speed_refresh_interval == 0:
                    self._refresh_speed()

                results = []
                for machine_id in list(self._feat_buffer.machines()):
                    if self._mqtt.should_decide(machine_id):
                        try:
                            result = self._process_machine(machine_id)
                            results.append(result)
                        except Exception as e:
                            logger.error("Erreur traitement %s : %s", machine_id, e)

                if results:
                    t_max    = max(r["temperature_c"] for r in results)
                    risk_max = max(r["risk_score"] for r in results)
                    logger.info(
                        "[t=%ds speed=%.0fx]  cluster -- %d machines  T_max=%.1f C  risk_max=%.2f",
                        self._t_elapsed, self._speed_multiplier, len(results), t_max, risk_max,
                    )

                await asyncio.sleep(1.0 / max(1.0, self._speed_multiplier))

        except asyncio.CancelledError:
            pass

    async def _loop_rest(self, duration_s: float | None) -> None:
        """Boucle de decision en fallback REST."""
        t_start = time.monotonic()
        self._t_elapsed = 0
        _cycle = 0
        _speed_refresh_interval = 6
        try:
            while True:
                elapsed = time.monotonic() - t_start
                self._t_elapsed = int(elapsed)
                if duration_s is not None and elapsed >= duration_s:
                    break
                _cycle += 1
                if _cycle % _speed_refresh_interval == 0:
                    self._refresh_speed()
                self._decision_cycle_rest()
                await asyncio.sleep(self.decision_interval_s)
        except asyncio.CancelledError:
            pass

    def run(self, duration_s: float | None = None) -> None:
        """Point d'entree synchrone -- lance la boucle async.
        Sur Windows, force SelectorEventLoop (aiomqtt/paho incompatible avec ProactorEventLoop).
        """
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        try:
            asyncio.run(self._run_async(duration_s))
        except KeyboardInterrupt:
            logger.info("Arret demande (Ctrl+C)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _machines_iter(self, cluster: dict):
        machines_raw = cluster.get("machines", {})
        if isinstance(machines_raw, list):
            for i, m in enumerate(machines_raw):
                yield m.get("machine_id", m.get("id", f"machine_{i}")), m
        else:
            yield from machines_raw.items()

    def _set_all_manual(self) -> None:
        cluster = self.client.get_cluster_status()
        for machine_id, snapshot in self._machines_iter(cluster):
            indices = _fan_indices(snapshot)
            self.client.set_fan_mode(machine_id, "manual", fan_indices=indices)
            logger.info("  %s -> mode manual (fans %s)", machine_id, indices)

    def _set_all_auto(self) -> None:
        cluster = self.client.get_cluster_status()
        for machine_id, snapshot in self._machines_iter(cluster):
            indices = _fan_indices(snapshot)
            self.client.set_fan_mode(machine_id, "auto", fan_indices=indices)
            logger.info("  %s -> mode auto (fans %s)", machine_id, indices)

    def _log_warning_dedup(self, key: str, msg: str) -> None:
        """Log un warning en dedupliquant les repetitions."""
        count = self._warn_counts.get(key, 0) + 1
        self._warn_counts[key] = count
        if count == 1 or count % 12 == 0:
            suffix = " (x%d)" % count if count > 1 else ""
            logger.warning("%s%s", msg, suffix)


# ---------------------------------------------------------------------------
# Point d'entree CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Superviseur ML -- Juste des Ventilateurs")
    parser.add_argument("--mode",        default=os.getenv("SUPERVISOR_MODE", "ml"),
                        choices=["ml", "threshold", "native"])
    parser.add_argument("--predictor",   default=os.getenv("PREDICTOR_MODEL", "logistic"))
    parser.add_argument("--controller",  default=os.getenv("CONTROLLER_MODEL", "supervised"))
    parser.add_argument("--label",       default="failure_60s")
    parser.add_argument("--api-url",     default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--mqtt-host",   default=os.getenv("MQTT_BROKER_HOST", "localhost"))
    parser.add_argument("--mqtt-port",   type=int, default=int(os.getenv("MQTT_BROKER_PORT", "1883")))
    parser.add_argument("--interval-ticks", type=int,
                        default=int(os.getenv("DECISION_INTERVAL_TICKS", "5")),
                        help="Decision toutes les N ticks simules MQTT (defaut: 5)")
    parser.add_argument("--interval",    type=float,
                        default=float(os.getenv("DECISION_INTERVAL_S", "5")),
                        help="Intervalle REST fallback en secondes (defaut: 5)")
    parser.add_argument("--duration",    type=float, default=None)
    parser.add_argument("--risk-threshold", type=float, default=RISK_THRESHOLD)
    parser.add_argument("--run-name",    default="supervisor")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--log-level",   default=os.getenv("LOG_LEVEL", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Niveau de log (defaut: INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
        force=True,
    )

    sup = Supervisor(
        mode                    = args.mode,
        api_url                 = args.api_url,
        mqtt_host               = args.mqtt_host,
        mqtt_port               = args.mqtt_port,
        predictor_name          = args.predictor,
        controller_name         = args.controller,
        label                   = args.label,
        decision_interval_ticks = args.interval_ticks,
        decision_interval_s     = args.interval,
        risk_threshold          = float(os.getenv("RISK_THRESHOLD", "0.6")),
        dry_run                 = args.dry_run,
    )
    sup.run(duration_s=args.duration)


if __name__ == "__main__":
    main()
