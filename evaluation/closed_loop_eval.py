"""Évaluation en boucle fermée — Phase 9.

Pilote *réellement* le simulateur jumeaux-chauds en temps réel : pour chaque
contrôleur, applique ses consignes RPM pendant un épisode, laisse la physique
du simulateur recalculer les températures, et mesure l'impact causal.

Contrairement au benchmark offline (`evaluation/benchmark.py`) qui rejoue des
données figées — où `nb_shutdowns` et `T_mean` sont identiques pour tous les
contrôleurs — la boucle fermée mesure :

  - les pannes réellement évitées (en distinguant pannes *évitables* des pannes
    *inévitables* causées par une `fan_failure` active) ;
  - le PUE réel, calculé tick par tick à partir des RPM commandés ;
  - l'énergie fans consommée et l'économie vs une ventilation à fond.

Le module est **découplé du transport** : `ClosedLoopRunner` ne dépend que d'un
client respectant le protocole `ControlClient` (lire l'état, commander les fans).
En production on injecte `supervisor.JumeauxClient` ; en test on injecte un faux
client doté d'un mini-modèle thermique, ce qui rend toute la logique (boucle,
métriques, classification des pannes) testable sans simulateur.

Usage CLI :
    python -m evaluation.closed_loop_eval \\
        --scenario stress --duration 600 --dt 5 \\
        --controllers native baseline_pid baseline_threshold \\
        --output evaluation/results/closed_loop_results_stress.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"

logger = logging.getLogger("closed_loop")

# ---------------------------------------------------------------------------
# Constantes (alignées sur supervisor.supervisor et features/energy.py)
# ---------------------------------------------------------------------------

RPM_LEVELS     = [800, 1500, 2500, 3500, 4500]
RPM_HIGH       = 4500     # consigne d'urgence (override risque)
RPM_DEFAULT    = 2500     # consigne de repli
RPM_MIN        = 800      # plancher de ventilation
RISK_THRESHOLD = 0.60     # au-dessus -> override RPM_HIGH

# Modèle de puissance des fans (loi cubique P ∝ RPM³, cf. features/energy.py)
FAN_MAX_RPM     = 5000
FAN_P_WORKER_W  = 12.0
FAN_P_MASTER_W  = 15.0
PUE_BASELINE    = 1.40

# Contrôleur de référence pour l'économie d'énergie : ventilation à fond
_ENERGY_REF_RPM = 4500


# ---------------------------------------------------------------------------
# Protocole client (permet l'injection d'un faux client en test)
# ---------------------------------------------------------------------------

@runtime_checkable
class ControlClient(Protocol):
    """Surface minimale attendue d'un client jumeaux-chauds.

    `supervisor.supervisor.JumeauxClient` la satisfait déjà. Les méthodes de
    contrôle de scénario/vitesse sont optionnelles (détectées via hasattr).
    """

    def get_cluster_status(self) -> dict: ...
    def set_fan_speed(self, machine_id: str, rpm: int,
                      fan_indices: list[int] | None = None) -> bool: ...
    def set_fan_mode(self, machine_id: str, mode: str,
                     fan_indices: list[int] | None = None) -> bool: ...


# ---------------------------------------------------------------------------
# Helpers physiques / parsing
# ---------------------------------------------------------------------------

def estimate_power_fans_w(rpm: float, role: str = "worker", fan_count: float = 2.0) -> float:
    """Puissance fans estimée (W) par la loi cubique, cohérente avec features/energy.py."""
    p_nom = FAN_P_MASTER_W if role == "master" else FAN_P_WORKER_W
    ratio = min(1.0, max(0.0, rpm / FAN_MAX_RPM))
    return p_nom * (ratio ** 3) * fan_count


def machines_from_cluster(cluster: dict) -> dict[str, dict]:
    """Normalise machines (dict ou liste) -> {machine_id: snapshot}."""
    raw = cluster.get("machines", {})
    if isinstance(raw, list):
        out = {}
        for i, m in enumerate(raw):
            mid = m.get("machine_id") or m.get("id") or f"machine_{i}"
            out[mid] = m
        return out
    return dict(raw)


def _fan_indices(snapshot: dict) -> list[int]:
    fans = snapshot.get("fans", [])
    if isinstance(fans, list):
        idx = [f["idx"] for f in fans if isinstance(f, dict) and "idx" in f]
        return idx or list(range(len(fans))) or [0, 1]
    return [0, 1]


def _has_fan_fault(snapshot: dict) -> bool:
    faults = snapshot.get("faults", []) or []
    return any("fan_failure" in str(f.get("type", "")) for f in faults if isinstance(f, dict))


def _role(snapshot: dict) -> str:
    return str(snapshot.get("role", "worker"))


def _fan_count(snapshot: dict) -> float:
    fans = snapshot.get("fans", [])
    return float(len(fans)) if isinstance(fans, list) and fans else 2.0


# ---------------------------------------------------------------------------
# Enregistrement d'un tick et d'un événement de panne
# ---------------------------------------------------------------------------

@dataclass
class TickRecord:
    """État observé d'une machine à un instant de décision."""
    sim_time_s:      float
    machine_id:      str
    temperature_c:   float
    rpm_commanded:   int
    status:          str
    has_fan_fault:   bool
    power_fans_w:    float
    power_compute_w: float

    @property
    def power_total_w(self) -> float:
        return self.power_fans_w + self.power_compute_w

    @property
    def pue(self) -> float:
        return (self.power_total_w / self.power_compute_w
                if self.power_compute_w > 0 else PUE_BASELINE)


@dataclass
class ShutdownEvent:
    """Transition vers 'off' (panne) détectée pendant l'épisode."""
    sim_time_s:  float
    machine_id:  str
    inevitable:  bool   # True si fan_failure active au moment de la panne


# ---------------------------------------------------------------------------
# Classification des pannes
# ---------------------------------------------------------------------------

class FaultClassifier:
    """Distingue les pannes évitables des pannes inévitables.

    Une panne est *inévitable* quand une `fan_failure` est active : le RPM est
    forcé à 0 quelle que soit la commande, la machine surchauffe quoi qu'on
    fasse. Toutes les autres pannes (accumulation thermique mal anticipée) sont
    *évitables* par un meilleur contrôle.
    """

    @staticmethod
    def is_inevitable(snapshot: dict) -> bool:
        return _has_fan_fault(snapshot)


# ---------------------------------------------------------------------------
# Décision de consigne RPM pour une machine
# ---------------------------------------------------------------------------

def _risk_score(predictor, feats: pd.Series, feature_order: list[str] | None) -> float:
    """Probabilité de panne prédite ; 0.0 si pas de prédicteur ou erreur."""
    if predictor is None:
        return 0.0
    X = pd.DataFrame([feats])
    if feature_order is not None:
        X = X.reindex(columns=feature_order, fill_value=0.0)
    try:
        return float(predictor.predict_proba(X)[0, 1])
    except Exception:
        return 0.0


def decide_rpm(
    controller,
    feats: pd.Series,
    risk: float,
    *,
    mode: str,
    risk_threshold: float = RISK_THRESHOLD,
    prev_rpm: int = RPM_DEFAULT,
) -> int:
    """Calcule la consigne RPM pour une machine.

    - mode "native"          : -1 (aucune intervention, fans laissés en auto).
    - risk >= risk_threshold : override RPM_HIGH.
    - sinon                  : décision du contrôleur (plancher RPM_MIN),
                               ou RPM_DEFAULT si aucun contrôleur.
    """
    if mode == "native":
        return -1
    if risk >= risk_threshold:
        return RPM_HIGH
    if controller is None:
        return RPM_DEFAULT
    X = pd.DataFrame([feats])
    if hasattr(controller, "_prev_rpm"):
        controller._prev_rpm = prev_rpm
    try:
        rpms = controller.decide_batch(X, risk_scores=np.array([risk]))
    except TypeError:
        rpms = controller.decide_batch(X)
    return max(int(rpms[0]), RPM_MIN)


# ---------------------------------------------------------------------------
# Runner d'un épisode pour un contrôleur donné
# ---------------------------------------------------------------------------

@dataclass
class ClosedLoopRunner:
    """Exécute un épisode en boucle fermée pour un contrôleur.

    Parameters
    ----------
    client          : client respectant ControlClient (réel ou fake).
    name            : nom du contrôleur (clé de résultat).
    controller      : objet exposant decide_batch (None en mode native).
    predictor       : prédicteur de risque optionnel (predict_proba).
    mode            : "native" (observation seule) ou "control".
    decision_dt_s   : secondes simulées entre deux décisions.
    risk_threshold  : seuil d'override RPM_HIGH.
    feature_order   : ordre de colonnes attendu par le prédicteur (optionnel).
    buffer_factory  : fabrique de OnlineFeatureBuffer (injectable en test).
    sleep_fn        : fonction d'attente réelle entre ticks (no-op en test).
    """
    client:         ControlClient
    name:           str
    controller:     Any = None
    predictor:      Any = None
    mode:           str = "control"
    decision_dt_s:  float = 5.0
    risk_threshold: float = RISK_THRESHOLD
    feature_order:  list[str] | None = None
    buffer_factory: Callable[[], Any] | None = None
    sleep_fn:       Callable[[float], None] = time.sleep

    records:  list[TickRecord]   = field(default_factory=list, init=False)
    events:   list[ShutdownEvent] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.buffer_factory is None:
            from supervisor.online_features import OnlineFeatureBuffer
            self.buffer_factory = OnlineFeatureBuffer
        self._buffer = self.buffer_factory()
        self._prev_status: dict[str, str] = {}
        self._prev_rpm: dict[str, int] = {}
        self._prev_energy: dict[str, float] = {}
        self._manual_set: set[str] = set()

    # -- contrôle d'épisode (best-effort, dépend des capacités du client) ----

    def reset_episode(self, scenario: str | None = None,
                      speed_multiplier: float | None = None) -> None:
        """Prépare l'épisode : (ré)applique le scénario, remet le temps à zéro,
        règle la vitesse. Chaque action est best-effort selon le client."""
        if scenario is not None and hasattr(self.client, "change_scenario"):
            try:
                self.client.change_scenario(scenario)
            except Exception as e:  # pragma: no cover - dépend du client réel
                logger.warning("change_scenario(%s) échoué : %s", scenario, e)
        if hasattr(self.client, "soft_reset"):
            try:
                self.client.soft_reset()
            except Exception as e:  # pragma: no cover
                logger.warning("soft_reset échoué : %s", e)
        if speed_multiplier is not None and hasattr(self.client, "set_speed"):
            try:
                self.client.set_speed(speed_multiplier)
            except Exception as e:  # pragma: no cover
                logger.warning("set_speed(%s) échoué : %s", speed_multiplier, e)
        # Repartir d'états propres
        self._buffer = self.buffer_factory()
        self._prev_status.clear()
        self._prev_rpm.clear()
        self._prev_energy.clear()
        self._manual_set.clear()

    # -- un pas de décision --------------------------------------------------

    def step(self, sim_time_s: float) -> list[TickRecord]:
        """Lit l'état, décide et commande pour chaque machine ; enregistre."""
        cluster = self.client.get_cluster_status()
        machines = machines_from_cluster(cluster)
        step_records: list[TickRecord] = []

        for mid, snap in machines.items():
            self._buffer.update(mid, snap)
            feats   = self._buffer.get_features(mid)
            status  = str(snap.get("status", "on"))
            fault   = FaultClassifier.is_inevitable(snap)

            # Détection de transition vers 'off' (événement de panne).
            # À la première observation, on mémorise l'état SANS compter :
            # une machine déjà 'off' au début de l'épisode n'est pas une panne
            # que l'on a provoquée/subie pendant la fenêtre de mesure.
            prev = self._prev_status.get(mid)
            if prev is not None and status == "off" and prev != "off":
                self.events.append(ShutdownEvent(sim_time_s, mid, inevitable=fault))
            self._prev_status[mid] = status

            # Décision + commande
            risk = _risk_score(self.predictor, feats, self.feature_order)
            prev_rpm = self._prev_rpm.get(mid, RPM_DEFAULT)
            rpm = decide_rpm(
                self.controller, feats, risk,
                mode=self.mode, risk_threshold=self.risk_threshold,
                prev_rpm=prev_rpm,
            )

            commanded = rpm
            if self.mode != "native" and rpm >= 0 and status != "off":
                if mid not in self._manual_set:
                    self.client.set_fan_mode(mid, "manual", fan_indices=_fan_indices(snap))
                    self._manual_set.add(mid)
                if rpm != prev_rpm:
                    self.client.set_fan_speed(mid, rpm, fan_indices=_fan_indices(snap))
                    self._prev_rpm[mid] = rpm
            else:
                # native : on observe le RPM réel pour l'énergie
                commanded = int(feats.get("fan_rpm_mean", 0.0))

            # Énergie : puissance fans estimée depuis le RPM effectif (loi cubique).
            role  = _role(snap)
            nfan  = _fan_count(snap)
            p_fan = estimate_power_fans_w(commanded if commanded >= 0 else 0, role, nfan)

            # Puissance totale : directe si le snapshot l'expose, sinon dérivée
            # de la dérivée de l'énergie cumulée (cas jumeaux-chauds : /cluster/status
            # ne fournit que energy_kwh_cumulated, pas power_w).
            p_tot = float(snap.get("power_w", snap.get("power_watts", 0.0)) or 0.0)
            energy_now = float(snap.get("energy_kwh",
                                        snap.get("energy_kwh_cumulated", 0.0)) or 0.0)
            if p_tot <= 0.0:
                prev_e = self._prev_energy.get(mid)
                if prev_e is not None and self.decision_dt_s > 0:
                    # kWh -> W : ΔkWh * 3.6e6 / Δt[s]
                    p_tot = max(0.0, (energy_now - prev_e) * 3.6e6 / self.decision_dt_s)
            self._prev_energy[mid] = energy_now
            p_cmp = max(0.0, p_tot - p_fan)

            rec = TickRecord(
                sim_time_s=sim_time_s, machine_id=mid,
                temperature_c=float(feats.get("temperature_c", 0.0)),
                rpm_commanded=int(max(commanded, 0)),
                status=status, has_fan_fault=fault,
                power_fans_w=p_fan, power_compute_w=p_cmp,
            )
            self.records.append(rec)
            step_records.append(rec)

        return step_records

    # -- boucle complète -----------------------------------------------------

    def run(self, duration_s: float, speed_multiplier: float = 1.0) -> dict:
        """Exécute l'épisode sur `duration_s` secondes simulées.

        Avance par pas de `decision_dt_s` secondes simulées ; entre deux pas,
        attend `decision_dt_s / speed_multiplier` secondes réelles (no-op en
        test via sleep_fn). Retourne les métriques agrégées.
        """
        n_steps = max(1, int(duration_s / self.decision_dt_s))
        for i in range(n_steps):
            sim_t = i * self.decision_dt_s
            try:
                self.step(sim_t)
            except Exception as e:  # pragma: no cover - robustesse boucle live
                logger.error("[%s] erreur au tick %.0fs : %s", self.name, sim_t, e)
            if i < n_steps - 1:
                self.sleep_fn(self.decision_dt_s / max(1.0, speed_multiplier))
        return self.aggregate_metrics()

    # -- agrégation des métriques -------------------------------------------

    def aggregate_metrics(self) -> dict:
        """Calcule les métriques de l'épisode à partir des ticks enregistrés."""
        if not self.records:
            return {"name": self.name, "n_ticks": 0}

        temps = np.array([r.temperature_c for r in self.records], dtype=float)
        rpms  = np.array([r.rpm_commanded for r in self.records], dtype=float)
        # PUE : moyenné uniquement sur les ticks où la puissance de calcul est
        # connue (> 0). Sinon on retombe sur le PUE de référence.
        pue_vals = [r.pue for r in self.records if r.power_compute_w > 0]
        pue_mean = float(np.mean(pue_vals)) if pue_vals else PUE_BASELINE

        # Énergie fans : chaque tick couvre decision_dt_s secondes simulées
        energy_fans_kwh = float(
            sum(r.power_fans_w for r in self.records) * self.decision_dt_s / 3600.0
        )

        nb_shutdowns   = len(self.events)
        nb_inevitable  = sum(1 for e in self.events if e.inevitable)
        nb_avoidable   = nb_shutdowns - nb_inevitable

        return {
            "name":            self.name,
            "mode":            self.mode,
            "n_ticks":         len(self.records),
            "n_machines":      len({r.machine_id for r in self.records}),
            "nb_shutdowns_cl": nb_shutdowns,
            "nb_inevitable":   nb_inevitable,
            "nb_avoidable":    nb_avoidable,
            "T_mean_cl":       float(temps.mean()),
            "T_max_cl":        float(temps.max()),
            "rpm_mean_cl":     float(rpms.mean()),
            "pue_mean":        pue_mean,
            "energy_fans_kwh": energy_fans_kwh,
        }


# ---------------------------------------------------------------------------
# Comparaison de plusieurs contrôleurs
# ---------------------------------------------------------------------------

def compute_relative_metrics(results: list[dict]) -> list[dict]:
    """Enrichit chaque résultat de métriques relatives au natif / au plein régime.

    - `nb_avoidable_avoided` : pannes évitables en moins vs le contrôleur natif.
    - `energy_saved_vs_max_pct` : économie d'énergie fans vs la consommation
      d'une ventilation à fond (RPM=4500) sur le même nombre de ticks/machines.
    """
    native = next((r for r in results if r.get("mode") == "native"), None)
    native_avoidable = native["nb_avoidable"] if native else None

    for r in results:
        if not r.get("n_ticks"):
            continue
        # Pannes évitables évitées vs natif
        if native_avoidable is not None:
            r["nb_avoidable_avoided"] = max(0, native_avoidable - r["nb_avoidable"])
        # Référence énergie : tous les fans à _ENERGY_REF_RPM sur les mêmes ticks
        ref_p = estimate_power_fans_w(_ENERGY_REF_RPM)  # worker, 2 fans
        # approx : n_ticks couvre l'ensemble machine×temps -> même base que mesuré
        ref_energy = ref_p * r["n_ticks"] * _step_dt_from(r) / 3600.0
        if ref_energy > 0:
            r["energy_saved_vs_max_pct"] = round(
                100.0 * (1.0 - r["energy_fans_kwh"] / ref_energy), 1
            )
    return results


def _step_dt_from(result: dict) -> float:
    """Récupère decision_dt_s encodé dans le résultat (défaut 5s)."""
    return float(result.get("decision_dt_s", 5.0))


def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 96)
    print("ÉVALUATION BOUCLE FERMÉE — PHASE 9")
    print("=" * 96)
    hdr = (f"{'Contrôleur':<22} {'Shutd':>6} {'Évit.':>6} {'Inévit':>7} "
           f"{'ÉvitéVsNat':>11} {'T_mean':>7} {'T_max':>7} {'RPM':>6} "
           f"{'PUE':>6} {'kWh_fans':>9}")
    print(hdr)
    print("-" * 96)
    for r in results:
        if not r.get("n_ticks"):
            print(f"{r['name']:<22} (aucun tick)")
            continue
        print(
            f"{r['name']:<22} {r['nb_shutdowns_cl']:>6} {r['nb_avoidable']:>6} "
            f"{r['nb_inevitable']:>7} {r.get('nb_avoidable_avoided', '-'):>11} "
            f"{r['T_mean_cl']:>7.1f} {r['T_max_cl']:>7.1f} {r['rpm_mean_cl']:>6.0f} "
            f"{r['pue_mean']:>6.3f} {r['energy_fans_kwh']:>9.4f}"
        )
    print("=" * 96)


# ---------------------------------------------------------------------------
# Construction des contrôleurs (réutilise les loaders du benchmark)
# ---------------------------------------------------------------------------

def build_controller(name: str):
    """Instancie un contrôleur par nom. Retourne (mode, controller, predictor).

    Les baselines ne nécessitent aucun modèle entraîné. Les contrôleurs ML
    (supervised/score) sont chargés depuis models/*/saved si présents, sinon
    on retombe sur un baseline pour ne pas planter.
    """
    from models.fan_control.baseline_fixed import FixedController
    from models.fan_control.baseline_threshold import ThresholdFanController
    from models.fan_control.baseline_pid import PIDController

    if name == "native":
        return "native", None, None
    if name == "baseline_threshold":
        return "control", ThresholdFanController(), None
    if name == "baseline_pid":
        return "control", PIDController(), None
    if name.startswith("baseline_fixed"):
        # ex : baseline_fixed_4500 / baseline_fixed_0
        from models.fan_control.baseline_fixed import RPM_LEVELS as FIXED_LEVELS
        rpm = 2500
        parts = name.split("_")
        if parts[-1].isdigit():
            # snap aux niveaux VALIDES de FixedController (≠ RPM_LEVELS boucle fermée)
            rpm = min(FIXED_LEVELS, key=lambda x: abs(x - int(parts[-1])))
        return "control", FixedController(rpm=rpm), None

    # Contrôleurs ML : chargement best-effort
    saved = ROOT / "models" / "fan_control" / "saved"
    if name == "supervised":
        p = saved / "supervised.joblib"
        if p.exists():
            from models.fan_control.supervised_controller import SupervisedController
            return "control", SupervisedController.load(str(p)), _load_predictor()
    if name == "score_controller":
        p = saved / "score_controller.json"
        if p.exists():
            from models.fan_control.score_controller import ScoreController
            return "control", ScoreController.load(str(p)), _load_predictor()

    logger.warning("Contrôleur '%s' indisponible -> repli baseline_pid", name)
    return "control", PIDController(), None


def _load_predictor(label: str = "failure_60s"):
    p = ROOT / "models" / "failure_prediction" / "saved" / f"logistic_{label}.joblib"
    if not p.exists():
        return None
    try:
        from models.failure_prediction.logistic_regression import LogisticPredictor
        return LogisticPredictor().load(str(p))
    except Exception as e:  # pragma: no cover
        logger.warning("Prédicteur non chargé : %s", e)
        return None


def _feature_order() -> list[str] | None:
    try:
        from models.failure_prediction.splitter import TemporalSplitter
        return list(TemporalSplitter().split()[0].columns)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Client de contrôle enrichi (scénario / vitesse / reset)
# ---------------------------------------------------------------------------

class ClosedLoopClient:
    """Enveloppe `JumeauxClient` avec le contrôle d'épisode (scénario, vitesse,
    soft reset) requis par la boucle fermée."""

    def __init__(self, api_url: str = "http://localhost:8000") -> None:
        from supervisor.supervisor import JumeauxClient
        self._c = JumeauxClient(api_url)
        self.base_url = self._c.base_url

    # délégation lecture/commande
    def get_cluster_status(self) -> dict:
        return self._c.get_cluster_status()

    def set_fan_speed(self, machine_id, rpm, fan_indices=None) -> bool:
        return self._c.set_fan_speed(machine_id, rpm, fan_indices=fan_indices)

    def set_fan_mode(self, machine_id, mode, fan_indices=None) -> bool:
        return self._c.set_fan_mode(machine_id, mode, fan_indices=fan_indices)

    # contrôle d'épisode
    def change_scenario(self, scenario: str) -> bool:
        return self._put("/simulation/scenario", {"scenario": scenario})

    def set_speed(self, speed_multiplier: float) -> bool:
        return self._put("/simulation/speed", {"speed_multiplier": speed_multiplier})

    def soft_reset(self) -> bool:
        return self._post("/simulation/speed/reset")

    def get_speed_multiplier(self) -> float:
        return self._c.get_speed_multiplier()

    def close(self) -> None:
        self._c.close()

    def _put(self, path: str, body: dict) -> bool:
        return self._request("PUT", path, body)

    def _post(self, path: str, body: dict | None = None) -> bool:
        return self._request("POST", path, body or {})

    def _request(self, method: str, path: str, body: dict) -> bool:
        url = f"{self.base_url}{path}"
        try:
            import httpx
            r = httpx.request(method, url, json=body, timeout=10.0)
            return r.status_code < 300
        except Exception as e:  # pragma: no cover - réseau
            logger.warning("%s %s échoué : %s", method, path, e)
            return False


# ---------------------------------------------------------------------------
# Orchestration : comparer plusieurs contrôleurs
# ---------------------------------------------------------------------------

def compare_controllers(
    client: ControlClient,
    controller_names: list[str],
    *,
    scenario: str,
    duration_s: float,
    decision_dt_s: float,
    speed_multiplier: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """Exécute séquentiellement chaque contrôleur et retourne leurs métriques."""
    feature_order = _feature_order()
    results: list[dict] = []

    for name in controller_names:
        mode, controller, predictor = build_controller(name)
        logger.info("=== Contrôleur '%s' (mode=%s) ===", name, mode)
        runner = ClosedLoopRunner(
            client=client, name=name, controller=controller, predictor=predictor,
            mode=mode, decision_dt_s=decision_dt_s, feature_order=feature_order,
            sleep_fn=sleep_fn,
        )
        runner.reset_episode(scenario=scenario, speed_multiplier=speed_multiplier)
        metrics = runner.run(duration_s, speed_multiplier=speed_multiplier)
        metrics["decision_dt_s"] = decision_dt_s
        results.append(metrics)

    compute_relative_metrics(results)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Évaluation boucle fermée — Phase 9")
    parser.add_argument("--scenario", default="stress")
    parser.add_argument("--duration", type=float, default=600.0,
                        help="durée de l'épisode en secondes simulées")
    parser.add_argument("--dt", type=float, default=5.0,
                        help="secondes simulées entre deux décisions")
    parser.add_argument("--speed", type=float, default=60.0,
                        help="multiplicateur de vitesse de simulation")
    parser.add_argument("--controllers", nargs="+",
                        default=["native", "baseline_pid", "baseline_threshold"])
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    client = ClosedLoopClient(args.api_url)
    try:
        results = compare_controllers(
            client, args.controllers,
            scenario=args.scenario, duration_s=args.duration,
            decision_dt_s=args.dt, speed_multiplier=args.speed,
        )
    finally:
        client.close()

    print_comparison(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or str(RESULTS_DIR / f"closed_loop_results_{args.scenario}.json")
    payload = {
        "scenario":   args.scenario,
        "duration_s": args.duration,
        "dt_s":       args.dt,
        "speed":      args.speed,
        "results":    results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nRésultats sauvegardés : {out}")


if __name__ == "__main__":
    main()
