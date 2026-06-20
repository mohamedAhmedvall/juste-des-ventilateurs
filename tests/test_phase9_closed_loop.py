"""Tests de validation Phase 9 — Évaluation en boucle fermée.

Toute la logique (boucle de décision, métriques, classification des pannes,
métriques relatives) est testée sans simulateur réel grâce à un faux client
`FakeThermalClient` doté d'un mini-modèle thermique déterministe :

    T(t+1) = T(t) + chaleur - k_cool · (rpm_effectif / RPM_max)

où `rpm_effectif = 0` si une `fan_failure` est active (panne inévitable).

Un test d'intégration marqué `slow` pilote l'API jumeaux-chauds live si elle
est accessible sur :8000 (sinon il est ignoré).

Usage :
    pytest tests/test_phase9_closed_loop.py -v
    pytest tests/test_phase9_closed_loop.py -v -m "not slow"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.closed_loop_eval import (
    ClosedLoopRunner,
    FaultClassifier,
    TickRecord,
    ShutdownEvent,
    estimate_power_fans_w,
    machines_from_cluster,
    decide_rpm,
    compute_relative_metrics,
    compare_controllers,
    build_controller,
    RPM_HIGH,
    RPM_MIN,
    RPM_DEFAULT,
    FAN_MAX_RPM,
)
from models.fan_control.baseline_fixed import FixedController


# ===========================================================================
# Faux client avec modèle thermique déterministe
# ===========================================================================

class FakeThermalClient:
    """Simulateur thermique minimal pilotable par RPM, conforme à ControlClient.

    Chaque appel à `get_cluster_status()` avance la physique d'un pas :
    les machines chauffent et sont refroidies proportionnellement au RPM
    *effectif* (0 si `fan_failure`). Au-delà de `t_shutdown`, la machine
    passe `off` et y reste (un seul événement de panne par machine).
    """

    def __init__(self, machines: dict, *, k_cool: float = 5.0,
                 t_shutdown: float = 88.0, auto_rpm: int = 1500,
                 dt_sim: float = 5.0, compute_w: float = 200.0):
        # machines : {id: {role, heat, load, has_fan_fault, temp}}
        self._spec = machines
        self._k_cool = k_cool
        self._t_shutdown = t_shutdown
        self._auto_rpm = auto_rpm
        self._dt_sim = dt_sim          # secondes simulées par appel get_cluster_status
        self._compute_w = compute_w    # puissance de calcul (W) d'une machine 'on'
        self._temp = {m: s.get("temp", 70.0) for m, s in machines.items()}
        self._status = {m: "on" for m in machines}
        self._mode = {m: "auto" for m in machines}
        self._cmd_rpm = {m: auto_rpm for m in machines}
        self._energy = {m: 0.0 for m in machines}  # kWh cumulés
        self.fan_speed_calls: list[tuple] = []
        self.fan_mode_calls: list[tuple] = []

    # -- API ControlClient ---------------------------------------------------

    def get_cluster_status(self) -> dict:
        self._advance()
        machines = []
        for mid, spec in self._spec.items():
            eff_rpm = 0 if spec.get("has_fan_fault") else self._cmd_rpm[mid]
            if self._status[mid] == "off":
                eff_rpm = 0
            faults = [{"type": "fan_failure"}] if spec.get("has_fan_fault") else []
            machines.append({
                "id": mid,
                "role": spec.get("role", "worker"),
                "status": self._status[mid],
                "temperature_c": self._temp[mid],
                # comme jumeaux-chauds : pas de power_w, seulement l'énergie cumulée
                "energy_kwh_cumulated": self._energy[mid],
                "load_estimated": spec.get("load", 0.5),
                "fans": [{"idx": 0, "rpm": eff_rpm, "mode": self._mode[mid]},
                         {"idx": 1, "rpm": eff_rpm, "mode": self._mode[mid]}],
                "sensors": {"temp_cpu": {"temp_c": self._temp[mid]}},
                "faults": faults,
            })
        return {"cluster_id": "fake", "machines": machines}

    def set_fan_speed(self, machine_id, rpm, fan_indices=None) -> bool:
        self.fan_speed_calls.append((machine_id, rpm))
        self._cmd_rpm[machine_id] = int(rpm)
        return True

    def set_fan_mode(self, machine_id, mode, fan_indices=None) -> bool:
        self.fan_mode_calls.append((machine_id, mode))
        self._mode[machine_id] = mode
        return True

    # -- contrôle d'épisode (comme ClosedLoopClient réel) --------------------

    def change_scenario(self, scenario: str) -> bool:
        return True

    def set_speed(self, speed_multiplier: float) -> bool:
        return True

    def soft_reset(self) -> bool:
        """Réinitialise l'état thermique : indispensable entre deux contrôleurs."""
        self._temp = {m: s.get("temp", 70.0) for m, s in self._spec.items()}
        self._status = {m: "on" for m in self._spec}
        self._mode = {m: "auto" for m in self._spec}
        self._cmd_rpm = {m: self._auto_rpm for m in self._spec}
        self._energy = {m: 0.0 for m in self._spec}
        return True

    # -- physique ------------------------------------------------------------

    def _advance(self) -> None:
        for mid, spec in self._spec.items():
            if self._status[mid] == "off":
                continue
            eff_rpm = 0 if spec.get("has_fan_fault") else self._cmd_rpm[mid]
            cooling = self._k_cool * (eff_rpm / FAN_MAX_RPM)
            self._temp[mid] += spec.get("heat", 1.5) - cooling
            # énergie cumulée : (compute + fans) sur dt_sim
            fan_w = estimate_power_fans_w(eff_rpm, spec.get("role", "worker"), 2.0)
            self._energy[mid] += (self._compute_w + fan_w) * self._dt_sim / 3.6e6
            if self._temp[mid] >= self._t_shutdown:
                self._status[mid] = "off"


def _three_machines():
    """m_ok stable, m_hot évitable (forte chaleur), m_fault inévitable."""
    return {
        "m_ok":    {"role": "worker", "heat": 1.5, "load": 0.5, "has_fan_fault": False, "temp": 70.0},
        "m_hot":   {"role": "worker", "heat": 3.0, "load": 0.9, "has_fan_fault": False, "temp": 70.0},
        "m_fault": {"role": "worker", "heat": 3.0, "load": 0.9, "has_fan_fault": True,  "temp": 70.0},
    }


# ===========================================================================
# Helpers purs
# ===========================================================================

class TestEnergyModel:

    def test_power_zero_at_zero_rpm(self):
        assert estimate_power_fans_w(0) == 0.0

    def test_cubic_law(self):
        # P(RPM_max) = P_nom * 1^3 * fan_count
        assert estimate_power_fans_w(FAN_MAX_RPM, "worker", 2.0) == pytest.approx(24.0)

    def test_master_higher_than_worker(self):
        assert (estimate_power_fans_w(FAN_MAX_RPM, "master")
                > estimate_power_fans_w(FAN_MAX_RPM, "worker"))

    def test_monotonic_increasing(self):
        vals = [estimate_power_fans_w(r) for r in (0, 1500, 2500, 3500, 4500)]
        assert vals == sorted(vals)


class TestParsing:

    def test_machines_from_list(self):
        cluster = {"machines": [{"id": "a"}, {"machine_id": "b"}]}
        out = machines_from_cluster(cluster)
        assert set(out) == {"a", "b"}

    def test_machines_from_dict(self):
        cluster = {"machines": {"a": {}, "b": {}}}
        assert set(machines_from_cluster(cluster)) == {"a", "b"}

    def test_empty(self):
        assert machines_from_cluster({}) == {}


class TestFaultClassifier:

    def test_inevitable_when_fan_failure(self):
        assert FaultClassifier.is_inevitable({"faults": [{"type": "fan_failure"}]}) is True

    def test_avoidable_when_other_fault(self):
        assert FaultClassifier.is_inevitable({"faults": [{"type": "power_surge"}]}) is False

    def test_avoidable_when_no_fault(self):
        assert FaultClassifier.is_inevitable({"faults": []}) is False


class TestDecideRpm:

    def _state(self, temp=70.0):
        return pd.Series({"temperature_c": temp, "margin_to_shutdown": 88.0 - temp,
                          "fan_rpm_mean": 1500.0})

    def test_native_returns_minus_one(self):
        assert decide_rpm(None, self._state(), 0.0, mode="native") == -1

    def test_risk_override(self):
        rpm = decide_rpm(FixedController(rpm=1500), self._state(), 0.99, mode="control")
        assert rpm == RPM_HIGH

    def test_controller_decision_floored(self):
        rpm = decide_rpm(FixedController(rpm=0), self._state(), 0.0, mode="control")
        assert rpm == RPM_MIN  # plancher appliqué

    def test_no_controller_default(self):
        assert decide_rpm(None, self._state(), 0.0, mode="control") == RPM_DEFAULT


class TestTickRecord:

    def test_pue_computation(self):
        r = TickRecord(0, "m", 70.0, 2500, "on", False,
                       power_fans_w=20.0, power_compute_w=180.0)
        assert r.power_total_w == pytest.approx(200.0)
        assert r.pue == pytest.approx(200.0 / 180.0)

    def test_pue_baseline_when_no_compute(self):
        r = TickRecord(0, "m", 70.0, 0, "off", False, 0.0, 0.0)
        assert r.pue == pytest.approx(1.40)


# ===========================================================================
# ClosedLoopRunner — comportement causal
# ===========================================================================

class TestRunnerCausal:

    def _run(self, controller_name, duration_s=120.0):
        mode, ctrl, pred = build_controller(controller_name)
        client = FakeThermalClient(_three_machines())
        runner = ClosedLoopRunner(
            client=client, name=controller_name, controller=ctrl, predictor=pred,
            mode=mode, decision_dt_s=5.0, sleep_fn=lambda _s: None,
        )
        return runner.run(duration_s), client, runner

    def test_native_lets_hot_machine_shutdown(self):
        """En natif (auto 1500 RPM), m_hot et m_fault surchauffent."""
        m, _, _ = self._run("native")
        # 2 pannes : m_hot (évitable) + m_fault (inévitable)
        assert m["nb_shutdowns_cl"] == 2
        assert m["nb_inevitable"] == 1
        assert m["nb_avoidable"] == 1

    def test_full_speed_saves_avoidable_machine(self):
        """À 4500 RPM, m_hot est sauvée ; m_fault reste inévitable."""
        m, _, _ = self._run("baseline_fixed_4500")
        assert m["nb_inevitable"] == 1          # fan_failure -> inévitable
        assert m["nb_avoidable"] == 0           # m_hot sauvée
        assert m["nb_shutdowns_cl"] == 1

    def test_full_speed_cooler_than_native(self):
        m_native, _, _ = self._run("native")
        m_full, _, _ = self._run("baseline_fixed_4500")
        assert m_full["T_mean_cl"] < m_native["T_mean_cl"]
        assert m_full["rpm_mean_cl"] > m_native["rpm_mean_cl"]

    def test_native_does_not_command(self):
        _, client, _ = self._run("native")
        assert client.fan_speed_calls == []
        assert client.fan_mode_calls == []

    def test_control_sets_manual_then_speed(self):
        _, client, _ = self._run("baseline_fixed_4500")
        assert any(mode == "manual" for _, mode in client.fan_mode_calls)
        assert any(rpm == 4500 for _, rpm in client.fan_speed_calls)

    def test_full_speed_more_energy_than_native(self):
        m_native, _, _ = self._run("native")
        m_full, _, _ = self._run("baseline_fixed_4500")
        assert m_full["energy_fans_kwh"] > m_native["energy_fans_kwh"]

    def test_pue_derived_from_energy_delta(self):
        """PUE calculé depuis la dérivée d'énergie cumulée (pas de power_w fourni)."""
        # machine toujours 'on' (heat=0), ventilée à 4500 -> fans=24W, compute=200W
        spec = {"m": {"role": "worker", "heat": 0.0, "load": 0.5,
                      "has_fan_fault": False, "temp": 70.0}}
        client = FakeThermalClient(spec, dt_sim=5.0, compute_w=200.0)
        runner = ClosedLoopRunner(
            client=client, name="baseline_fixed_4500",
            controller=FixedController(rpm=4500), mode="control",
            decision_dt_s=5.0, sleep_fn=lambda _s: None,
        )
        m = runner.run(120.0)
        # PUE attendu = (200 + 24) / 200 = 1.12
        assert m["pue_mean"] == pytest.approx(1.12, rel=0.05)

    def test_machine_already_off_not_counted(self):
        """Une machine déjà 'off' au démarrage ne compte pas comme une panne."""
        spec = {"m_dead": {"role": "worker", "heat": 0.0, "load": 0.0,
                           "has_fan_fault": False, "temp": 70.0}}
        client = FakeThermalClient(spec)
        client._status["m_dead"] = "off"        # déjà éteinte avant l'épisode
        runner = ClosedLoopRunner(client=client, name="native", mode="native",
                                  decision_dt_s=5.0, sleep_fn=lambda _s: None)
        m = runner.run(50.0)
        assert m["nb_shutdowns_cl"] == 0

    def test_records_and_machines_count(self):
        m, _, runner = self._run("native", duration_s=50.0)
        assert m["n_machines"] == 3
        assert m["n_ticks"] == len(runner.records)
        assert m["n_ticks"] == 3 * int(50.0 / 5.0)


# ===========================================================================
# Métriques relatives
# ===========================================================================

class TestRelativeMetrics:

    def test_avoidable_avoided_vs_native(self):
        results = [
            {"name": "native", "mode": "native", "n_ticks": 30, "nb_avoidable": 2,
             "nb_shutdowns_cl": 3, "nb_inevitable": 1, "energy_fans_kwh": 0.01,
             "decision_dt_s": 5.0},
            {"name": "ml", "mode": "control", "n_ticks": 30, "nb_avoidable": 0,
             "nb_shutdowns_cl": 1, "nb_inevitable": 1, "energy_fans_kwh": 0.05,
             "decision_dt_s": 5.0},
        ]
        compute_relative_metrics(results)
        assert results[1]["nb_avoidable_avoided"] == 2
        assert results[0]["nb_avoidable_avoided"] == 0

    def test_energy_saved_pct_present(self):
        results = [
            {"name": "low", "mode": "control", "n_ticks": 10, "nb_avoidable": 0,
             "nb_shutdowns_cl": 0, "nb_inevitable": 0, "energy_fans_kwh": 0.0001,
             "decision_dt_s": 5.0},
        ]
        compute_relative_metrics(results)
        # consommation faible -> forte économie vs plein régime
        assert results[0]["energy_saved_vs_max_pct"] > 50.0

    def test_skips_empty_results(self):
        results = [{"name": "x", "n_ticks": 0}]
        compute_relative_metrics(results)  # ne doit pas lever
        assert "nb_avoidable_avoided" not in results[0]


# ===========================================================================
# Orchestration compare_controllers (avec fake)
# ===========================================================================

class TestCompareControllers:

    def test_compare_demonstrates_causal_gain(self):
        client = FakeThermalClient(_three_machines())
        results = compare_controllers(
            client, ["native", "baseline_fixed_4500"],
            scenario="fake", duration_s=120.0, decision_dt_s=5.0,
            speed_multiplier=1.0, sleep_fn=lambda _s: None,
        )
        by_name = {r["name"]: r for r in results}
        # Le plein régime évite la panne évitable du natif
        assert by_name["baseline_fixed_4500"]["nb_avoidable_avoided"] >= 1
        # mais ne peut rien contre la panne inévitable
        assert by_name["baseline_fixed_4500"]["nb_inevitable"] == 1


# ===========================================================================
# build_controller
# ===========================================================================

class TestBuildController:

    def test_native(self):
        mode, ctrl, pred = build_controller("native")
        assert mode == "native" and ctrl is None

    def test_baselines_need_no_training(self):
        for name in ("baseline_pid", "baseline_threshold", "baseline_fixed_2500"):
            mode, ctrl, _ = build_controller(name)
            assert mode == "control" and ctrl is not None

    def test_fixed_rpm_snapped_to_levels(self):
        _, ctrl, _ = build_controller("baseline_fixed_4500")
        assert ctrl.rpm == 4500

    def test_unknown_falls_back(self):
        mode, ctrl, _ = build_controller("does_not_exist")
        assert mode == "control" and ctrl is not None  # repli baseline_pid


# ===========================================================================
# Intégration live (jumeaux-chauds réel) — marquée slow
# ===========================================================================

def _api_live(url="http://localhost:8000") -> bool:
    try:
        import httpx
        return httpx.get(f"{url}/cluster/status", timeout=2.0).status_code == 200
    except Exception:
        return False


@pytest.mark.slow
class TestLiveIntegration:

    @pytest.fixture(autouse=True)
    def _skip_if_no_api(self):
        if not _api_live():
            pytest.skip("API jumeaux-chauds non accessible sur :8000")

    def test_native_then_full_speed_real_sim(self):
        from evaluation.closed_loop_eval import ClosedLoopClient
        client = ClosedLoopClient("http://localhost:8000")
        try:
            results = compare_controllers(
                client, ["native", "baseline_fixed_4500"],
                scenario="stress", duration_s=60.0, decision_dt_s=5.0,
                speed_multiplier=60.0,
            )
        finally:
            client.close()
        assert len(results) == 2
        for r in results:
            assert r["n_ticks"] > 0
            assert r["T_mean_cl"] > 0
            assert "pue_mean" in r
