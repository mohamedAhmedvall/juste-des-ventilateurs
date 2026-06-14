"""Tests unitaires — Phase 6 : Supervisor et DecisionLogger.

pytest tests/test_phase6_supervisor.py -v
pytest tests/test_phase6_supervisor.py -v -m "not slow"
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supervisor.decision_logger import DecisionLogger
from supervisor.supervisor import (
    Supervisor,
    JumeauxClient,
    RISK_THRESHOLD,
    RPM_HIGH,
    RPM_DEFAULT,
)
from supervisor.online_features import OnlineFeatureBuffer


# ---------------------------------------------------------------------------
# DecisionLogger
# ---------------------------------------------------------------------------

class TestDecisionLogger:

    def test_creates_jsonl_file(self, tmp_path):
        with DecisionLogger(log_dir=tmp_path, run_name="test") as dl:
            dl.log({"machine_id": "srv-01", "rpm": 2500})
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1

    def test_entries_are_valid_json(self, tmp_path):
        with DecisionLogger(log_dir=tmp_path, run_name="test") as dl:
            dl.log({"machine_id": "srv-01", "rpm": 2500, "risk_score": 0.3})
            dl.log({"machine_id": "srv-02", "rpm": 3500, "risk_score": 0.8})
        entries = DecisionLogger.load(list(tmp_path.glob("*.jsonl"))[0])
        assert len(entries) == 2
        assert entries[0]["rpm"] == 2500
        assert entries[1]["risk_score"] == 0.8

    def test_timestamp_auto_added(self, tmp_path):
        with DecisionLogger(log_dir=tmp_path, run_name="test") as dl:
            dl.log({"machine_id": "srv-01"})
        entries = DecisionLogger.load(list(tmp_path.glob("*.jsonl"))[0])
        assert "ts" in entries[0]

    def test_to_dataframe(self, tmp_path):
        with DecisionLogger(log_dir=tmp_path, run_name="test") as dl:
            for i in range(5):
                dl.log({"machine_id": f"srv-{i:02d}", "rpm": 1500 * (i + 1)})
        log_path = list(tmp_path.glob("*.jsonl"))[0]
        df = DecisionLogger.to_dataframe(log_path)
        assert len(df) == 5
        assert "machine_id" in df.columns

    def test_count_matches_entries(self, tmp_path):
        dl = DecisionLogger(log_dir=tmp_path, run_name="count_test")
        for _ in range(7):
            dl.log({"x": 1})
        assert dl._count == 7
        dl.close()


# ---------------------------------------------------------------------------
# OnlineFeatureBuffer (remplacement de snapshot_to_series — Phase 7)
# ---------------------------------------------------------------------------

class TestOnlineFeatureBridge:
    """Tests de compatibilite OnlineFeatureBuffer avec les attentes Phase 6."""

    def _make_snap(self, temp=65.0, n_fans=2, rpm=2500, status="on"):
        return {
            "machine_id":       "m0",
            "role":             "worker",
            "status":           status,
            "temperature_c":    temp,
            "sensor_temp_max":  temp + 2.0,
            "sensor_temp_mean": temp,
            "power_w":          100.0,
            "energy_kwh":       0.3,
            "load_estimated":   0.5,
            "fans":             [{"idx": i, "rpm": rpm} for i in range(n_fans)],
            "faults":           [],
        }

    def _filled_buffer(self, temp=65.0, n=10, **kw):
        buf = OnlineFeatureBuffer()
        for _ in range(n):
            buf.update("m0", self._make_snap(temp=temp, **kw))
        return buf

    def test_returns_series(self):
        buf = self._filled_buffer()
        s = buf.get_features("m0")
        assert isinstance(s, pd.Series)

    def test_temperature_correct(self):
        buf = self._filled_buffer(temp=72.0)
        s = buf.get_features("m0")
        assert s["temperature_c"] == pytest.approx(72.0)

    def test_margin_to_shutdown(self):
        buf = self._filled_buffer(temp=65.0)
        s = buf.get_features("m0")
        assert s["margin_to_shutdown"] == pytest.approx(88.0 - 65.0)

    def test_rpm_rolling_mean(self):
        buf = self._filled_buffer(rpm=3500, n_fans=3)
        s = buf.get_features("m0")
        assert s["rpm_rolling_mean_30s"] == pytest.approx(3500.0, rel=0.01)

    def test_empty_fans(self):
        buf = self._filled_buffer(n_fans=0)
        s = buf.get_features("m0")
        assert s["rpm_rolling_mean_30s"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# JumeauxClient (sans API réelle — test des erreurs gracieuses)
# ---------------------------------------------------------------------------

class TestJumeauxClientOffline:

    def test_get_cluster_returns_empty_on_failure(self):
        client = JumeauxClient(base_url="http://localhost:19999", timeout=0.5)
        result = client.get_cluster_status()
        assert result == {}

    def test_set_fan_speed_returns_false_on_failure(self):
        client = JumeauxClient(base_url="http://localhost:19999", timeout=0.5)
        ok = client.set_fan_speed("srv-01", 3500)
        assert ok is False

    def test_set_fan_mode_returns_false_on_failure(self):
        client = JumeauxClient(base_url="http://localhost:19999", timeout=0.5)
        ok = client.set_fan_mode("srv-01", "manual")
        assert ok is False


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class TestSupervisor:

    def _make_supervisor(self, tmp_path, mode="ml", dry_run=True):
        return Supervisor(
            mode           = mode,
            dry_run        = dry_run,
            api_url        = "http://localhost:19999",
            log_dir        = tmp_path,
            run_name       = "test",
        )

    def test_init_ml_mode(self, tmp_path):
        sup = self._make_supervisor(tmp_path, mode="ml")
        assert sup.mode == "ml"
        assert sup.dry_run is True
        sup.dec_logger.close()

    def test_init_native_mode(self, tmp_path):
        sup = self._make_supervisor(tmp_path, mode="native")
        assert sup.predictor is None
        assert sup.controller is None
        sup.dec_logger.close()

    def _make_state(self, temp=65.0):
        """Cree un pd.Series minimal compatible avec _predict_risk / _decide_rpm."""
        buf = OnlineFeatureBuffer()
        snap = {
            "machine_id": "m0", "role": "worker", "status": "on",
            "temperature_c": temp, "sensor_temp_max": temp + 2,
            "sensor_temp_mean": temp, "power_w": 100.0, "energy_kwh": 0.3,
            "load_estimated": 0.5, "fans": [], "faults": [],
        }
        for _ in range(10):
            buf.update("m0", snap)
        return buf.get_features("m0")

    def test_predict_risk_no_predictor(self, tmp_path):
        sup = self._make_supervisor(tmp_path, mode="native")
        state = self._make_state()
        risk = sup._predict_risk(state)
        assert risk == 0.0
        sup.dec_logger.close()

    def test_risk_override(self, tmp_path):
        """Quand risk_score >= threshold, le RPM doit etre RPM_HIGH."""
        sup = self._make_supervisor(tmp_path, mode="ml")
        state = self._make_state()
        rpm, _, _ = sup._decide_rpm(state, risk_score=RISK_THRESHOLD + 0.01)
        assert rpm == RPM_HIGH
        sup.dec_logger.close()

    def test_native_mode_returns_minus_one(self, tmp_path):
        """Mode native -> decision = -1 (pas d'intervention)."""
        sup = self._make_supervisor(tmp_path, mode="native")
        state = self._make_state()
        rpm, _, _ = sup._decide_rpm(state, risk_score=0.0)
        assert rpm == -1
        sup.dec_logger.close()

    def test_decision_cycle_rest_empty_cluster(self, tmp_path):
        """_decision_cycle_rest sur un cluster vide ne leve pas d'exception."""
        sup = self._make_supervisor(tmp_path, mode="ml")
        # API inexistante -> cluster vide, aucun plantage attendu
        try:
            sup._decision_cycle_rest()
        except Exception as e:
            assert False, f"_decision_cycle_rest a leve une exception inattendue : {e}"
        sup.dec_logger.close()

    def test_decision_logged_after_process(self, tmp_path):
        """Vérifier qu'une décision est loggée dans _process_machine."""
        sup = self._make_supervisor(tmp_path, mode="ml")
        snap = {
            "temperature_c": 72.0, "power_w": 150.0, "energy_kwh": 0.5,
            "load_estimated": 0.8, "t_shutdown_c": 88.0, "status": "on",
            "fans": {"fan_0": {"rpm": 2500}}, "sensors": {"temp_max": 74.0, "temp_mean": 72.0},
        }
        sup._process_machine("srv-worker-01", snap)
        assert sup.dec_logger._count == 1
        sup.dec_logger.close()

    @pytest.mark.slow
    def test_supervisor_run_short(self, tmp_path):
        """Lancer le superviseur pendant 3 cycles (15s) en dry_run."""
        sup = Supervisor(
            mode="ml", dry_run=True,
            api_url="http://localhost:19999",
            decision_interval_s=0.1,
            log_dir=tmp_path, run_name="short_run",
        )
        sup.run(duration_s=0.35)  # ~3 cycles de 0.1s
        assert sup.dec_logger._count >= 0  # Pas d'erreur
        sup.dec_log