"""Tests du pont de données dashboard NOC (dashboard/noc_bridge.build_live).

Valide le mapping snapshot cluster -> payload dashboard, sans serveur ni API
réelle (OnlineFeatureBuffer réel + faux prédicteur).

    pytest tests/test_phase9c_dashboard.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.noc_bridge import build_live, command_machine, change_scenario, BridgeState
from supervisor.online_features import OnlineFeatureBuffer


class _RecordClient:
    """Faux client jumeaux-chauds qui enregistre les commandes."""
    def __init__(self):
        self.speed_calls = []
        self.mode_calls = []
    def set_fan_speed(self, mid, rpm, fan_indices=None):
        self.speed_calls.append((mid, int(rpm))); return True
    def set_fan_mode(self, mid, mode, fan_indices=None):
        self.mode_calls.append((mid, mode)); return True
    def get_cluster_status(self):
        return {}


def _cluster():
    return {
        "cluster_id": "cluster_alpha",
        "ts": "2025-01-01T00:00:00Z",
        "metrics": {"pue_effective": 1.18, "energy_kwh_total": 3.2, "cost_eur_total": 0.9},
        "machines": [
            {"id": "srv-01", "role": "worker", "status": "on",
             "temperature_c": 71.5, "load_estimated": 0.6,
             "fans": [{"idx": 0, "rpm": 3200}, {"idx": 1, "rpm": 3100}], "faults": []},
            {"id": "srv-02", "role": "worker", "status": "off",
             "temperature_c": 22.0, "load_estimated": 0.0,
             "fans": [{"idx": 0, "rpm": 0}], "faults": []},
        ],
    }


class _FakePredictor:
    def predict_proba(self, X):
        return np.array([[0.27, 0.73]])  # risque 73 %


class _FakeController:
    def decide_batch(self, X, risk_scores=None):
        return np.array([3500])


class TestBuildLive:

    def test_maps_core_fields(self):
        out = build_live(_cluster(), OnlineFeatureBuffer(), None, None)
        assert out["source"] == "cluster_alpha"
        assert set(out["byId"]) == {"srv-01", "srv-02"}
        a = out["byId"]["srv-01"]
        assert a["temp"] == pytest.approx(71.5)
        assert a["on"] is True and a["status"] == "on"
        assert a["role"] == "worker"
        assert a["fans"] == [3200, 3100]
        assert a["risk"] is None          # pas de prédicteur -> risk None
        assert a["explain"] == []         # pas de modèle linéaire -> pas d'explicabilité
        assert a["rpm_reco"] is None      # pas de contrôleur
        # métriques cluster propagées (pour les KPI du dashboard)
        assert out["metrics"]["pue_effective"] == pytest.approx(1.18)
        assert out["metrics"]["energy_kwh_total"] == pytest.approx(3.2)

    def test_rpm_reco_from_controller(self):
        out = build_live(_cluster(), OnlineFeatureBuffer(), None, None, controller=_FakeController())
        assert out["byId"]["srv-01"]["rpm_reco"] == 3500

    def test_off_machine(self):
        out = build_live(_cluster(), OnlineFeatureBuffer(), None, None)
        b = out["byId"]["srv-02"]
        assert b["on"] is False and b["status"] == "off"
        assert b["fans"] == [0]

    def test_risk_from_predictor(self):
        out = build_live(_cluster(), OnlineFeatureBuffer(), _FakePredictor(), None)
        assert out["byId"]["srv-01"]["risk"] == pytest.approx(73.0)

    def test_empty_cluster(self):
        out = build_live({}, OnlineFeatureBuffer(), None, None)
        assert out["byId"] == {}

    def test_robust_to_predictor_error(self):
        class Boom:
            def predict_proba(self, X):
                raise ValueError("boom")
        out = build_live(_cluster(), OnlineFeatureBuffer(), Boom(), None)
        # l'erreur du prédicteur ne casse pas le payload -> risk None
        assert out["byId"]["srv-01"]["risk"] is None
        assert out["byId"]["srv-01"]["temp"] == pytest.approx(71.5)


class TestPilotage:

    def test_command_rpm_sets_manual_then_speed(self):
        c = _RecordClient()
        assert command_machine(c, "srv-01", rpm=3500) is True
        assert ("srv-01", "manual") in c.mode_calls
        assert ("srv-01", 3500) in c.speed_calls

    def test_command_mode_auto(self):
        c = _RecordClient()
        command_machine(c, "srv-01", mode="auto")
        assert c.mode_calls == [("srv-01", "auto")]
        assert c.speed_calls == []

    def test_autopilot_applies_reco_to_on_machines(self):
        c = _RecordClient()
        st = BridgeState(c, None, None, None, None)
        payload = {"byId": {
            "srv-01": {"on": True,  "rpm_reco": 4500},
            "srv-02": {"on": False, "rpm_reco": 800},   # off -> ignorée
            "srv-03": {"on": True,  "rpm_reco": None},  # pas de reco -> ignorée
        }}
        n = st.apply_autopilot(payload)
        assert n == 1
        assert ("srv-01", 4500) in c.speed_calls
        assert all(mid != "srv-02" for mid, _ in c.speed_calls)


class _ScenClient:
    """Faux client avec base_url + session httpx-like enregistrant les requêtes."""
    def __init__(self):
        self.base_url = "http://x:8000"
        self.calls = []
        self._client = self
    def request(self, method, url, json=None):
        self.calls.append((method, url, json))
        return type("R", (), {"status_code": 200})()


class TestScenario:

    def test_change_scenario_hits_api(self):
        c = _ScenClient()
        ok = change_scenario(c, "heatwave", speed=60)
        assert ok is True
        paths = [u for _, u, _ in c.calls]
        assert any(u.endswith("/simulation/scenario") for u in paths)
        assert any(u.endswith("/simulation/speed/reset") for u in paths)
        assert any(u.endswith("/simulation/speed") for u in paths)
        # le scénario est bien transmis
        scen = next(j for _, u, j in c.calls if u.endswith("/simulation/scenario"))
        assert scen == {"scenario": "heatwave"}

    def test_change_scenario_no_speed(self):
        c = _ScenClient()
        change_scenario(c, "stress")
        paths = [u for _, u, _ in c.calls]
        assert any(u.endswith("/simulation/scenario") for u in paths)
        assert not any(u.endswith("/simulation/speed") and not u.endswith("reset") for u in paths)
