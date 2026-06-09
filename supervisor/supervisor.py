"""Superviseur de régulation thermique — Juste des Ventilateurs.

Boucle principale de décision en temps réel :
  1. Lire l'état du cluster via REST (GET /cluster/status)
  2. Pour chaque machine active : extraire les features online
  3. Évaluer le risque de panne (prédicteur logistic ou autre)
  4. Décider la consigne RPM (contrôleur supervisé ou autre)
  5. Appliquer un override si risk_score > risk_threshold
  6. Envoyer la commande via REST (PUT /machines/{id}/fan_speed)
  7. Logger la décision

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("supervisor")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

RPM_LEVELS     = [0, 1500, 2500, 3500, 4500]
RPM_HIGH       = 4500   # RPM d'urgence quand risk_score > threshold
RPM_DEFAULT    = 2500   # RPM de sécurité si aucune décision possible
RISK_THRESHOLD = 0.60   # Seuil de surcharge risque → RPM_HIGH

# Features attendues par le prédicteur (doit correspondre aux 47 features du splitter)
# On utilise un sous-ensemble disponible dans l'état REST
ONLINE_FEATURES = [
    "temperature_c", "sensor_temp_max", "sensor_temp_mean",
    "power_w", "energy_kwh", "fan_rpm_mean",
    "load_estimated",
]


# ---------------------------------------------------------------------------
# Client REST jumeaux-chauds
# ---------------------------------------------------------------------------

class JumeauxClient:
    """Client REST léger vers l'API jumeaux-chauds.

    Parameters
    ----------
    base_url : URL de base de l'API (ex: "http://localhost:8000")
    timeout  : timeout HTTP en secondes
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        try:
            import httpx
            self._client = httpx.Client(timeout=timeout)
        except ImportError:
            import urllib.request
            self._client = None
        logger.info(f"JumeauxClient → {self.base_url}")

    def get_cluster_status(self) -> dict:
        """GET /cluster/status — état complet du cluster."""
        return self._get("/cluster/status")

    def get_machine(self, machine_id: str) -> dict:
        """GET /machines/{id} — état d'une machine."""
        return self._get(f"/machines/{machine_id}")

    def set_fan_speed(self, machine_id: str, rpm: int, fan_indices: list[int] | None = None) -> bool:
        """PUT /machines/{id}/fan_speed — fixe le RPM sur tous les fans (fan_idx requis par l'API)."""
        if fan_indices is None:
            fan_indices = [0, 1]  # 2 fans par machine par défaut
        ok = True
        for idx in fan_indices:
            try:
                url  = f"{self.base_url}/machines/{machine_id}/fan_speed"
                body = {"fan_idx": idx, "rpm": rpm}
                if self._client is not None:
                    resp = self._client.put(url, json=body)
                    ok = ok and resp.status_code < 300
                else:
                    import json, urllib.request
                    data = json.dumps(body).encode()
                    req  = urllib.request.Request(url, data=data, method="PUT",
                                                  headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        ok = ok and r.status < 300
            except Exception as e:
                logger.warning(f"set_fan_speed({machine_id}, fan={idx}, {rpm}) échoué : {e}")
                ok = False
        return ok

    def set_fan_mode(self, machine_id: str, mode: str, fan_indices: list[int] | None = None) -> bool:
        """PUT /machines/{id}/fan_mode — mode auto/manual sur tous les fans."""
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
                    import json, urllib.request
                    data = json.dumps(body).encode()
                    req  = urllib.request.Request(url, data=data, method="PUT",
                                                  headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        ok = ok and r.status < 300
            except Exception as e:
                logger.warning(f"set_fan_mode({machine_id}, fan={idx}, {mode}) échoué : {e}")
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
                import json, urllib.request
                with urllib.request.urlopen(url, timeout=self.timeout) as r:
                    return json.loads(r.read())
        except Exception as e:
            logger.warning(f"GET {path} échoué : {e}")
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
    """Charge le prédicteur de pannes sauvegardé."""
    import joblib as _joblib
    saved_path = ROOT / "models" / "failure_prediction" / "saved" / f"{model_name}_{label}.joblib"
    if not saved_path.exists():
        logger.warning(f"Prédicteur absent : {saved_path}. risk_score=0 partout.")
        return None
    from models.failure_prediction.logistic_regression import LogisticPredictor
    predictor = LogisticPredictor()
    # Chargement defensif : compatibilite format dict ou pipeline direct
    raw = _joblib.load(str(saved_path))
    logger.info(f"  joblib type={type(raw).__name__}  keys={list(raw.keys()) if isinstance(raw, dict) else 'n/a'}")
    if isinstance(raw, dict):
        # Chercher le pipeline sous les cles connues (pipeline, model, estimator...)
        pipeline_val = (raw.get("pipeline") or raw.get("model") or raw.get("estimator")
                        or next((v for v in raw.values() if hasattr(v, "predict_proba")), None))
        if pipeline_val is None:
            raise KeyError(f"Aucun objet sklearn trouve dans le joblib. Cles: {list(raw.keys())}")
        predictor._pipeline = pipeline_val
        predictor.threshold = raw.get("threshold", predictor.threshold)
        predictor.C         = raw.get("C", predictor.C)
    else:
        predictor._pipeline = raw
    logger.info(f"Prédicteur chargé : {saved_path.name}")
    return predictor


def load_controller(model_name: str = "supervised"):
    """Charge le contrôleur de ventilateurs sauvegardé."""
    saved_dir = ROOT / "models" / "fan_control" / "saved"
    if model_name == "supervised":
        path = saved_dir / "supervised.joblib"
        if not path.exists():
            logger.warning(f"Contrôleur absent : {path}. RPM_DEFAULT utilisé.")
            return None
        from models.fan_control.supervised_controller import SupervisedController
        ctrl = SupervisedController.load(str(path))
    elif model_name == "score_controller":
        path = saved_dir / "score_controller.json"
        if not path.exists():
            logger.warning(f"Contrôleur absent : {path}. RPM_DEFAULT utilisé.")
            return None
        from models.fan_control.score_controller import ScoreController
        ctrl = ScoreController.load(str(path))
    elif model_name == "baseline_pid":
        path = saved_dir / "baseline_pid.json"
        if not path.exists():
            logger.warning(f"Contrôleur absent : {path}. RPM_DEFAULT utilisé.")
            return None
        from models.fan_control.baseline_pid import PIDController
        ctrl = PIDController.load(str(path))
    elif model_name == "baseline_threshold":
        path = saved_dir / "baseline_threshold.json"
        if not path.exists():
            logger.warning(f"Contrôleur absent : {path}. RPM_DEFAULT utilisé.")
            return None
        from models.fan_control.baseline_threshold import ThresholdFanController
        ctrl = ThresholdFanController.load(str(path))
    else:
        logger.warning(f"Contrôleur inconnu : {model_name}. RPM_DEFAULT utilisé.")
        return None
    logger.info(f"Contrôleur chargé : {path.name}")
    return ctrl


# ---------------------------------------------------------------------------
# Extraction des features online (depuis snapshot REST)
# ---------------------------------------------------------------------------

def snapshot_to_series(snapshot: dict) -> "pd.Series":
    """Transforme un snapshot REST machine en pd.Series compatible features."""
    import pandas as pd

    sensors = snapshot.get("sensors", {})
    fans    = snapshot.get("fans", {})

    # Fans : liste [{idx, rpm, mode}] OU dict {fan_id: {rpm, mode}}
    if isinstance(fans, list):
        rpms = [f.get("rpm", 0) for f in fans if isinstance(f, dict)]
    else:
        rpms = [v.get("rpm", 0) for v in fans.values() if isinstance(v, dict)]
    fan_rpm_mean = float(np.mean(rpms)) if rpms else 0.0

    # Sensors : {"temp_cpu": {"temp_c": X}, ...} OU {"temp_max": X, ...}
    def _sensor_val(key_nested: str, key_flat: str, default: float) -> float:
        """Extrait une valeur de sensor quel que soit le format."""
        if key_nested in sensors and isinstance(sensors[key_nested], dict):
            return float(sensors[key_nested].get("temp_c", default))
        return float(sensors.get(key_flat, default))

    temp_c   = float(snapshot.get("temperature_c", 60.0))
    temp_max  = _sensor_val("temp_cpu",     "temp_max",  temp_c)
    temp_mean = _sensor_val("temp_chassis", "temp_mean", temp_c)

    row = {
        "temperature_c":    temp_c,
        "sensor_temp_max":  temp_max,
        "sensor_temp_mean": temp_mean,
        "power_w":          float(snapshot.get("power_w", snapshot.get("power_watts", 0.0))),
        "energy_kwh":       float(snapshot.get("energy_kwh", snapshot.get("energy_kwh_cumulated", 0.0))),
        "fan_rpm_mean":     fan_rpm_mean,
        "load_estimated":   float(snapshot.get("load_estimated", snapshot.get("load", 0.5))),
        # Features rolling non disponibles online → approx avec valeur courante
        "temp_delta_5s":                0.0,
        "temp_delta_15s":               0.0,
        "temp_delta_30s":               0.0,
        "temp_rolling_mean_30s":        temp_c,
        "temp_rolling_mean_60s":        temp_c,
        "temp_rolling_std_30s":         0.0,
        "margin_to_shutdown":           88.0 - temp_c,  # t_shutdown fixe 88°C
        "margin_pct":                   max(0.0, (88.0 - temp_c) / 88.0),
        "margin_delta_30s":             0.0,
        "load_rolling_mean_30s":        float(snapshot.get("load_estimated", 0.5)),
        "load_rolling_mean_60s":        float(snapshot.get("load_estimated", 0.5)),
        "rpm_delta_15s":                0.0,
        "rpm_rolling_mean_30s":         fan_rpm_mean,
        "power_rolling_mean_30s":       float(snapshot.get("power_w", 0.0)),
        "power_delta_30s":              0.0,
        "sensor_max_delta_15s":         0.0,
        "sensor_max_rolling_mean_30s":  temp_max,
        "power_fans_rolling_mean_30s":  0.0,
        "pue_rolling_mean_30s":         1.0,
        "time_in_hot_zone_s":           0.0,
        "nb_shutdowns_episode":         0,
        "nb_degraded_episode":          0,
        "ticks_since_last_shutdown":    9999,
        "has_fan_fault":                0,
        "has_power_surge":              0,
        "ticks_since_last_fault":       9999,
        "is_recovering":                0,
        "power_fans_w":                 0.0,
        "fan_energy_ratio":             0.0,
        "pue_estimated":                1.0,
        "energy_fans_kwh_cumulated":    0.0,
        # Contexte épisode (non disponible online)
        "time_in_degraded_s":           0.0,
        "time_to_failure_s":            999.0,
        "fan_count":                    len(fans),
        "fan_rpm_std":                  float(np.std(rpms)) if len(rpms) > 1 else 0.0,
        "fault_count":                  0,
    }
    import pandas as pd
    return pd.Series(row)


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

_DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"


class Supervisor:
    """Superviseur temps réel couplant prédicteur + contrôleur.

    Parameters
    ----------
    mode            : "ml" | "threshold" | "native"
    predictor_name  : nom du modèle prédictif ("logistic", "random_forest", ...)
    controller_name : nom du contrôleur ("supervised", "score_controller", ...)
    label           : label de panne cible ("failure_60s" | "failure_30s")
    risk_threshold  : seuil de surcharge risk_score → RPM_HIGH
    api_url         : URL de l'API jumeaux-chauds
    decision_interval_s : fréquence de décision (secondes)
    log_dir         : répertoire des logs
    run_name        : préfixe du fichier log
    dry_run         : si True, ne pas envoyer les commandes REST
    """

    def __init__(
        self,
        mode:                 str   = "ml",
        predictor_name:       str   = "logistic",
        controller_name:      str   = "supervised",
        label:                str   = "failure_60s",
        risk_threshold:       float = RISK_THRESHOLD,
        api_url:              str   = "http://localhost:8000",
        decision_interval_s:  float = 5.0,
        log_dir:              str | Path = _DEFAULT_LOG_DIR,
        run_name:             str   = "supervisor",
        dry_run:              bool  = False,
    ) -> None:
        self.mode                 = mode
        self.label                = label
        self.risk_threshold       = risk_threshold
        self.decision_interval_s  = decision_interval_s
        self.dry_run              = dry_run

        self.client     = JumeauxClient(api_url)
        self.dec_logger = DecisionLogger(log_dir=log_dir, run_name=run_name)

        # Modèles (None si mode != "ml" ou si fichiers absents)
        self.predictor  = load_predictor(predictor_name, label) if mode == "ml" else None
        self.controller = load_controller(controller_name) if mode == "ml" else None

        # État interne : RPM précédent par machine
        self._prev_rpm: dict[str, int] = {}

        logger.info(f"Supervisor prêt — mode={mode}  dry_run={dry_run}")

    # ------------------------------------------------------------------

    def _predict_risk(self, state_series: "pd.Series") -> float:
        """Retourne un risk_score ∈ [0, 1]."""
        if self.predictor is None:
            return 0.0
        import pandas as pd
        X = pd.DataFrame([state_series])
        # Garder seulement les colonnes connues du modèle
        try:
            proba = self.predictor.predict_proba(X)
            return float(proba[0, 1])
        except Exception as e:
            logger.debug(f"predict_proba échoué : {e}")
            return 0.0

    def _decide_rpm(self, state_series: "pd.Series", risk_score: float) -> int:
        """Retourne le RPM décidé pour une machine."""
        # Surcharge risque élevé
        if risk_score >= self.risk_threshold:
            return RPM_HIGH

        if self.mode == "native":
            return -1  # Ne pas intervenir

        if self.controller is None:
            return RPM_DEFAULT

        import pandas as pd
        X = pd.DataFrame([state_series])
        risk_arr = np.array([risk_score])
        try:
            rpms = self.controller.decide_batch(X, risk_scores=risk_arr)
            return int(rpms[0])
        except TypeError:
            try:
                rpms = self.controller.decide_batch(X)
                return int(rpms[0])
            except Exception as e:
                logger.debug(f"decide_batch échoué : {e}")
                return RPM_DEFAULT

    def _process_machine(self, machine_id: str, snapshot: dict) -> dict:
        """Traite une machine : prédit, décide, envoie, logue."""
        state = snapshot_to_series(snapshot)
        risk  = self._predict_risk(state)
        rpm   = self._decide_rpm(state, risk)

        prev_rpm = self._prev_rpm.get(machine_id, RPM_DEFAULT)

        entry = {
            "ts":             datetime.now(timezone.utc).isoformat(),
            "machine_id":     machine_id,
            "temperature_c":  float(snapshot.get("temperature_c", 0.0)),
            "status":         str(snapshot.get("status", "unknown")),
            "fan_rpm_mean":   float(state.get("fan_rpm_mean", 0.0)),
            "risk_score":     round(risk, 4),
            "rpm_decided":    rpm,
            "rpm_previous":   prev_rpm,
            "mode":           self.mode,
            "risk_override":  risk >= self.risk_threshold,
        }

        if rpm >= 0 and rpm != prev_rpm:
            if not self.dry_run:
                fan_indices = self._fan_indices(snapshot)
                ok = self.client.set_fan_speed(machine_id, rpm, fan_indices=fan_indices)
                entry["command_sent"] = ok
            else:
                entry["command_sent"] = None  # dry_run
            self._prev_rpm[machine_id] = rpm
            logger.info(
                f"  {machine_id:<20} T={state['temperature_c']:.1f}°C  "
                f"risk={risk:.2f}  RPM {prev_rpm}→{rpm}"
                + (" [RISK OVERRIDE]" if entry["risk_override"] else "")
                + (" [DRY RUN]" if self.dry_run else "")
            )
        else:
            entry["command_sent"] = None  # Pas de changement

        self.dec_logger.log(entry)
        return entry

    def step(self) -> list[dict]:
        """Un cycle de décision : lit le cluster, traite toutes les machines."""
        cluster = self.client.get_cluster_status()
        if not cluster:
            logger.warning("cluster/status vide — skip cycle")
            return []

        machines_raw = cluster.get("machines", {})
        if not machines_raw:
            logger.warning("Aucune machine dans le cluster — skip cycle")
            return []

        # Normaliser : l'API peut retourner un dict {id: snap} ou une liste [{machine_id, ...}]
        if isinstance(machines_raw, list):
            machines_data = {m.get("machine_id", m.get("id", f"machine_{i}")): m
                             for i, m in enumerate(machines_raw)}
        else:
            machines_data = machines_raw

        results = []
        for machine_id, snapshot in machines_data.items():
            if snapshot.get("status") == "off":
                continue
            try:
                result = self._process_machine(machine_id, snapshot)
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur traitement {machine_id} : {e}")
        return results

    def run(self, duration_s: float | None = None) -> None:
        """Lance la boucle principale.

        Parameters
        ----------
        duration_s : durée max en secondes (None = infini jusqu'à Ctrl+C)
        """
        logger.info(f"=== Supervisor démarré — mode={self.mode} ===")
        if self.dry_run:
            logger.info("  [DRY RUN] Aucune commande ne sera envoyée")

        # Mettre les machines en mode manual avant de prendre la main
        if self.mode != "native" and not self.dry_run:
            self._set_all_manual()

        t_start = time.monotonic()
        cycle   = 0
        try:
            while True:
                cycle += 1
                elapsed = time.monotonic() - t_start
                if duration_s is not None and elapsed >= duration_s:
                    break
                logger.info(f"[cycle {cycle}] t={elapsed:.0f}s")
                self.step()
                time.sleep(self.decision_interval_s)
        except KeyboardInterrupt:
            logger.info("Arrêt demandé (Ctrl+C)")
        finally:
            if self.mode != "native" and not self.dry_run:
                self._set_all_auto()
            self.client.close()
            self.dec_logger.close()
            logger.info(f"=== Supervisor arrêté après {cycle} cycles ===")

    def _fan_indices(self, snapshot: dict) -> list[int]:
        """Retourne la liste des indices de fans d'une machine depuis son snapshot."""
        fans = snapshot.get("fans", [])
        if isinstance(fans, list):
            return [f["idx"] for f in fans if "idx" in f]
        return list(range(len(fans))) if fans else [0, 1]

    def _set_all_manual(self) -> None:
        """Passe toutes les machines en mode manual fan."""
        cluster = self.client.get_cluster_status()
        for machine_id, snapshot in cluster.get("machines", {}).items():
            indices = self._fan_indices(snapshot)
            self.client.set_fan_mode(machine_id, "manual", fan_indices=indices)
            logger.info(f"  {machine_id} → mode manual (fans {indices})")

    def _set_all_auto(self) -> None:
        """Repasse toutes les machines en mode auto fan."""
        cluster = self.client.get_cluster_status()
        for machine_id, snapshot in cluster.get("machines", {}).items():
            indices = self._fan_indices(snapshot)
            self.client.set_fan_mode(machine_id, "auto", fan_indices=indices)
            logger.info(f"  {machine_id} → mode auto (fans {indices})")


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Superviseur ML — Juste des Ventilateurs")
    parser.add_argument("--mode",        default=os.getenv("SUPERVISOR_MODE", "ml"),
                        choices=["ml", "threshold", "native"],
                        help="Mode de supervision (défaut: ml)")
    parser.add_argument("--predictor",   default=os.getenv("PREDICTOR_MODEL", "logistic"),
                        help="Modèle prédictif (défaut: logistic)")
    parser.add_argument("--controller",  default=os.getenv("CONTROLLER_MODEL", "supervised"),
                        help="Contrôleur (défaut: supervised)")
    parser.add_argument("--label",       default="failure_60s",
                        help="Label de panne (défaut: failure_60s)")
    parser.add_argument("--api-url",     default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--interval",    type=float, default=float(os.getenv("DECISION_INTERVAL_S", "5")),
                        help="Intervalle de décision en secondes (défaut: 5)")
    parser.add_argument("--duration",    type=float, default=None,
                        help="Durée max en secondes (None = infini)")
    parser.add_argument("--risk-threshold", type=float, default=RISK_THRESHOLD,
                        help=f"Seuil risk_score pour surcharge RPM_HIGH (défaut: {RISK_THRESHOLD})")
    parser.add_argument("--run-name",    default="supervisor",
                        help="Préfixe du fichier log")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Simuler sans envoyer de commandes REST")
    args = parser.parse_args()

    sup = Supervisor(
        mode                = args.mode,
        predictor_name      = args.predictor,
        controller_name     = args.controller,
        label               = args.label,
        risk_threshold      = args.risk_threshold,
        api_url             = args.api_url,
        decision_interval_s = args.interval,
        log_dir             = _DEFAULT_LOG_DIR,
        run_name            = args.run_name,
        dry_run             = args.dry_run,
    )
    sup.run(duration_s=args.duration)


if __name__ == "__main__":
    main()