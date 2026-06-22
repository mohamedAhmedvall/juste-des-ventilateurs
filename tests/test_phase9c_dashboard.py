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

from dashboard.noc_bridge import build_live
from supervisor.online_features import OnlineFeatureBuffer


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
