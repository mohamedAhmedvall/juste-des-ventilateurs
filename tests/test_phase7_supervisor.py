"""Tests Phase 7 -- OnlineFeatureBuffer, MqttTelemetryConsumer, boucle supervisor.

Couvre :
- OnlineFeatureBuffer : calcul des features incrementales (tick par tick)
- MqttTelemetryConsumer : normalisation payload, compteur should_decide
- supervisor._log_warning_dedup : deduplication des warnings
- supervisor._machines_iter : gestion liste et dict
"""
from __future__ import annotations

import asyncio
import sys
import types
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stubs minimalistes pour les dependances lourdes
# ---------------------------------------------------------------------------

def _make_stub(name: str):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

# Ne stubber que les modules reellement absents du systeme de test.
# IMPORTANT : ne pas stubber xgboost si le vrai package est disponible,
# car cela casserait test_phase4_models qui charge des modeles .joblib xgboost.
_STUB_ONLY_IF_MISSING = ["aiomqtt"]
for _m in _STUB_ONLY_IF_MISSING:
    if _m not in sys.modules:
        _make_stub(_m)

# xgboost : stubber uniquement si vraiment absent (evite de polluer test_phase4)
try:
    import xgboost  # noqa: F401
except ImportError:
    _make_stub("xgboost")


# ---------------------------------------------------------------------------
# Tests OnlineFeatureBuffer
# ---------------------------------------------------------------------------

class TestOnlineFeatureBuffer(unittest.TestCase):

    def _make_snapshot(self, temp=60.0, load=0.5, power=300.0,
                       status="on", fans=None, faults=None):
        if fans is None:
            fans = [{"idx": 0, "rpm": 2000}, {"idx": 1, "rpm": 2000}]
        return {
            "machine_id":       "m0",
            "role":             "worker",
            "status":           status,
            "temperature_c":    temp,
            "sensor_temp_max":  temp + 2.0,
            "sensor_temp_mean": temp + 1.0,
            "power_w":          power,
            "energy_kwh":       0.0,
            "load_estimated":   load,
            "fans":             fans,
            "faults":           faults or [],
        }

    def test_empty_buffer_returns_zero_features(self):
        from supervisor.online_features import OnlineFeatureBuffer
        buf = OnlineFeatureBuffer()
        # Aucun tick => pas de machine enregistree
        self.assertEqual(buf.machines(), [])

    def test_single_tick_registered(self):
        from supervisor.online_features import OnlineFeatureBuffer
        buf = OnlineFeatureBuffer()
        snap = self._make_snapshot(temp=65.0)
        buf.update("m0", snap)
        self.assertIn("m0", buf.machines())

    def test_features_shape(self):
        """Apres 70 ticks, get_features retourne une Series avec toutes les features attendues."""
        from supervisor.online_features import OnlineFeatureBuffer
        buf = OnlineFeatureBuffer()
        for i in range(70):
            temp = 60.0 + i * 0.1   # montee progressive
            buf.update("m0", self._make_snapshot(temp=temp))
        feat = buf.get_features("m0")
        self.assertIsNotNone(feat)
        # Features temporelles cles
        for col in [
            "temperature_c", "temp_delta_5s", "temp_delta_15s", "temp_delta_30s",
            "temp_rolling_mean_30s", "temp_rolling_std_30s",
            "margin_to_shutdown", "load_rolling_mean_30s",
            "rpm_rolling_mean_30s", "power_rolling_mean_30s",
            "power_fans_w", "pue_estimated",
            "nb_shutdowns_episode", "ticks_since_last_shutdown",
        ]:
            self.assertIn(col, feat.index, msg=f"Feature manquante : {col}")

    def test_temp_delta_increases_with_rising_temp(self):
        """temp_delta_5s doit etre positif quand la temperature monte."""
        from supervisor.online_features import OnlineFeatureBuffer
        buf = OnlineFeatureBuffer()
        for i in range(30):
            buf.update("m0", self._make_snapshot(temp=60.0 + i * 0.5))
        feat = buf.get_features("m0")
        self.assertGreater(feat["temp_delta_5s"], 0.0)

    def test_margin_to_shutdown(self):
        """margin_to_shutdown = T_SHUTDOWN - temp_c."""
        from supervisor.online_features import OnlineFeatureBuffer, _T_SHUTDOWN_C
        buf = OnlineFeatureBuffer()
        temp = 80.0
        for _ in range(10):
            buf.update("m0", self._make_snapshot(temp=temp))
        feat = buf.get_features("m0")
        expected = _T_SHUTDOWN_C - temp
        self.assertAlmostEqual(feat["margin_to_shutdown"], expected, places=1)

    def test_power_fans_w_cubic(self):
        """Loi cubique : P_fans = (rpm/rpm_max)^3 * P_nom."""
        from supervisor.online_features import (
            OnlineFeatureBuffer, _FAN_MAX_RPM, _FAN_P_WORKER_W
        )
        buf = OnlineFeatureBuffer()
        rpm = 2500
        fans = [{"idx": 0, "rpm": rpm}]
        for _ in range(10):
            buf.update("m0", self._make_snapshot(fans=fans))
        feat = buf.get_features("m0")
        expected = ((rpm / _FAN_MAX_RPM) ** 3) * _FAN_P_WORKER_W * 1
        self.assertAlmostEqual(feat["power_fans_w"], expected, places=2)

    def test_nb_shutdowns_increments(self):
        """nb_shutdowns_episode s'incremente quand status passe a shutdown."""
        from supervisor.online_features import OnlineFeatureBuffer
        buf = OnlineFeatureBuffer()
        for _ in range(5):
            buf.update("m0", self._make_snapshot(status="on"))
        buf.update("m0", self._make_snapshot(status="off"))
        buf.update("m0", self._make_snapshot(status="on"))
        feat = buf.get_features("m0")
        self.assertGreaterEqual(feat["nb_shutdowns_episode"], 1)


# ---------------------------------------------------------------------------
# Tests MqttTelemetryConsumer
# ---------------------------------------------------------------------------

class TestMqttTelemetryConsumer(unittest.TestCase):

    def _make_buffer_mock(self):
        buf = MagicMock()
        buf.machines.return_value = []
        return buf

    def test_normalize_valid_payload(self):
        from supervisor.mqtt_telemetry import MqttTelemetryConsumer
        buf = self._make_buffer_mock()
        consumer = MqttTelemetryConsumer(buf, cluster_id="cluster_alpha")

        payload = {
            "role":          "worker",
            "status":        "on",
            "temperature_c": 72.5,
            "power_w":       350.0,
            "energy_kwh_cumulated": 0.12,
            "load_estimated": 0.65,
            "fans": [{"idx": 0, "rpm": 2200}],
            "sensors": {
                "temp_cpu":    {"temp_c": 74.0},
                "temp_inlet":  {"temp_c": 68.0},
            },
            "faults": [],
        }
        result = consumer._normalize("m0", payload)
        self.assertIsNotNone(result)
        self.assertEqual(result["machine_id"], "m0")
        self.assertAlmostEqual(result["temperature_c"], 72.5)
        self.assertAlmostEqual(result["sensor_temp_max"], 74.0)
        self.assertAlmostEqual(result["load_estimated"], 0.65)
        self.assertIsInstance(result["fans"], list)

    def test_normalize_none_on_bad_payload(self):
        from supervisor.mqtt_telemetry import MqttTelemetryConsumer
        buf = self._make_buffer_mock()
        consumer = MqttTelemetryConsumer(buf)
        self.assertIsNone(consumer._normalize("m0", "not a dict"))

    def test_should_decide_interval(self):
        """should_decide retourne True exactement tous les N ticks."""
        from supervisor.mqtt_telemetry import MqttTelemetryConsumer
        buf = self._make_buffer_mock()
        consumer = MqttTelemetryConsumer(buf, decision_interval_ticks=5)
        results = [consumer.should_decide("m0") for _ in range(10)]
        true_ticks = [i for i, v in enumerate(results, 1) if v]
        # Ticks 5 et 10 doivent declencher
        self.assertIn(5, true_ticks)
        self.assertIn(10, true_ticks)
        self.assertEqual(len(true_ticks), 2)

    def test_should_decide_interval_1(self):
        """Avec interval=1, toutes les decisions sont declenchees."""
        from supervisor.mqtt_telemetry import MqttTelemetryConsumer
        buf = self._make_buffer_mock()
        consumer = MqttTelemetryConsumer(buf, decision_interval_ticks=1)
        results = [consumer.should_decide("m0") for _ in range(5)]
        self.assertTrue(all(results))

    def test_handle_feeds_buffer(self):
        """_handle doit appeler buffer.update avec le snapshot normalise."""
        from supervisor.mqtt_telemetry import MqttTelemetryConsumer
        buf = self._make_buffer_mock()
        consumer = MqttTelemetryConsumer(buf, cluster_id="cluster_alpha")
        import json
        payload = json.dumps({
            "role": "worker", "status": "on",
            "temperature_c": 65.0, "power_w": 300.0,
            "energy_kwh_cumulated": 0.0, "load_estimated": 0.5,
            "fans": [], "sensors": {}, "faults": [],
        }).encode()
        asyncio.run(consumer._handle("dt/cluster_alpha/m0/telemetry", payload))
        buf.update.assert_called_once()
        call_args = buf.update.call_args[0]
        self.assertEqual(call_args[0], "m0")

    def test_handle_invalid_json_ignored(self):
        """Payload invalide ne doit pas lever d'exception."""
        from supervisor.mqtt_telemetry import MqttTelemetryConsumer
        buf = self._make_buffer_mock()
        consumer = MqttTelemetryConsumer(buf)
        asyncio.run(consumer._handle("dt/cluster_alpha/m0/telemetry", b"invalid json"))
        buf.update.assert_not_called()


# ---------------------------------------------------------------------------
# Tests supervisor helpers (sans demarrer la boucle async)
# ---------------------------------------------------------------------------

class TestSupervisorHelpers(unittest.TestCase):

    def _make_supervisor(self):
        """Instancie un Supervisor minimal (mode threshold pour eviter joblib)."""
        # On patche JumeauxClient et les modeles pour ne pas avoir besoin du simulateur
        with patch("supervisor.supervisor.JumeauxClient") as MockClient, \
             patch("supervisor.supervisor.MqttTelemetryConsumer", create=True):
            MockClient.return_value.get_speed_multiplier.return_value = 1.0
            from supervisor.supervisor import Supervisor
            sup = Supervisor(
                mode="threshold",
                api_url="http://localhost:8000",
                mqtt_host="localhost",
                mqtt_port=1883,
            )
        return sup

    def test_log_warning_dedup_first_occurrence(self):
        """Premier appel : toujours logue."""
        sup = self._make_supervisor()
        with patch("supervisor.supervisor.logger") as mock_log:
            sup._log_warning_dedup("key1", "message test")
            mock_log.warning.assert_called_once()

    def test_log_warning_dedup_suppresses_repeat(self):
        """2e a 11e appel : supprime le log."""
        sup = self._make_supervisor()
        with patch("supervisor.supervisor.logger") as mock_log:
            for _ in range(11):
                sup._log_warning_dedup("key1", "message test")
            # 1er appel logue, les 10 suivants non => 1 appel total
            self.assertEqual(mock_log.warning.call_count, 1)

    def test_log_warning_dedup_12th_occurrence(self):
        """12e appel : logue de nouveau (rappel periodique)."""
        sup = self._make_supervisor()
        with patch("supervisor.supervisor.logger") as mock_log:
            for _ in range(12):
                sup._log_warning_dedup("key1", "message test")
            # Appel 1 + appel 12
            self.assertEqual(mock_log.warning.call_count, 2)

    def test_machines_iter_list(self):
        """_machines_iter gere un cluster avec machines en liste."""
        sup = self._make_supervisor()
        cluster = {"machines": [
            {"machine_id": "m0", "temperature_c": 60.0},
            {"id": "m1", "temperature_c": 65.0},
        ]}
        result = list(sup._machines_iter(cluster))
        ids = [r[0] for r in result]
        self.assertIn("m0", ids)
        self.assertIn("m1", ids)

    def test_machines_iter_dict(self):
        """_machines_iter gere un cluster avec machines en dict."""
        sup = self._make_supervisor()
        cluster = {"machines": {
            "m0": {"temperature_c": 60.0},
            "m1": {"temperature_c": 65.0},
        }}
        result = list(sup._machines_iter(cluster))
        ids = [r[0] for r in result]
        self.assertIn("m0", ids)
        self.assertIn("m1", ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
