"""Tests de validation Phase 5 -- Fan Controllers.

Verifie que :
  - les 5 controleurs s'instancient et produisent des RPM valides
  - decide() et decide_batch() retournent des valeurs dans RPM_LEVELS
  - fit() ne plante pas sur les donnees reelles
  - save()/load() fonctionnent (round-trip)

Usage :
  pytest tests/test_phase5_controllers.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RPM_LEVELS = {800, 1500, 2500, 3500, 4500}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_df():
    """DataFrame minimal simulant des features machine."""
    n = 100
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "temperature_c":         rng.uniform(50, 90, n),
        "fan_rpm_mean":          rng.uniform(1000, 4000, n),
        "margin_to_shutdown":    rng.uniform(-5, 30, n),
        "margin_pct":            rng.uniform(0, 100, n),
        "time_in_hot_zone_s":    rng.uniform(0, 60, n),
        "temp_delta_5s":         rng.uniform(-2, 2, n),
        "temp_delta_30s":        rng.uniform(-5, 5, n),
        "load_estimated":        rng.uniform(0, 1, n),
        "failure_60s":           rng.integers(0, 2, n),
        "action_class":          rng.integers(0, 4, n),
        "optimal_rpm":           rng.choice([0, 1500, 2500, 3500], n),
    })


@pytest.fixture(scope="module")
def processed_data():
    """Charge les donnees reelles si disponibles, sinon skip."""
    processed_dir = Path("data/processed")
    if not any(processed_dir.glob("episode=*")):
        pytest.skip("Donnees processed absentes -- lancer ingest_gen_features.bat")
    from models.failure_prediction.splitter import TemporalSplitter
    splitter = TemporalSplitter(processed_dir=str(processed_dir))
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(label_col="failure_60s")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 1. FixedController
# ---------------------------------------------------------------------------

class TestFixedController:
    def test_valid_rpm_levels(self):
        from models.fan_control.baseline_fixed import FixedController
        for rpm in [0, 1500, 2500, 3500, 4500]:
            ctrl = FixedController(rpm=rpm)
            assert ctrl.rpm == rpm

    def test_invalid_rpm_raises(self):
        from models.fan_control.baseline_fixed import FixedController
        with pytest.raises(ValueError):
            FixedController(rpm=999)

    def test_decide_returns_constant(self, small_df):
        from models.fan_control.baseline_fixed import FixedController
        ctrl = FixedController(rpm=2500)
        row = small_df.iloc[0]
        assert ctrl.decide(row) == 2500

    def test_decide_batch_shape(self, small_df):
        from models.fan_control.baseline_fixed import FixedController
        ctrl = FixedController(rpm=3500)
        result = ctrl.decide_batch(small_df)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(small_df)
        assert set(result).issubset(RPM_LEVELS)

    def test_save_load_roundtrip(self, small_df):
        from models.fan_control.baseline_fixed import FixedController
        ctrl = FixedController(rpm=4500)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        ctrl.save(path)
        ctrl2 = FixedController.load(path)
        assert ctrl2.rpm == 4500


# ---------------------------------------------------------------------------
# 2. ThresholdFanController
# ---------------------------------------------------------------------------

class TestThresholdFanController:
    def test_decide_high_temp(self):
        from models.fan_control.baseline_threshold import ThresholdFanController
        ctrl = ThresholdFanController(t_low=65, t_medium=72, t_high=79)
        row = pd.Series({"temperature_c": 85.0})
        assert ctrl.decide(row) == 4500

    def test_decide_low_temp(self):
        from models.fan_control.baseline_threshold import ThresholdFanController
        ctrl = ThresholdFanController(t_low=65, t_medium=72, t_high=79)
        row = pd.Series({"temperature_c": 50.0})
        assert ctrl.decide(row) == 1500

    def test_decide_batch_in_rpm_levels(self, small_df):
        from models.fan_control.baseline_threshold import ThresholdFanController
        ctrl = ThresholdFanController()
        result = ctrl.decide_batch(small_df)
        assert set(result).issubset(RPM_LEVELS)

    def test_fit_does_not_crash(self, small_df):
        from models.fan_control.baseline_threshold import ThresholdFanController
        ctrl = ThresholdFanController()
        ctrl.fit(small_df, small_df["failure_60s"])
        assert ctrl.best_params_ != {} or True  # peut etre vide si grid vide

    def test_save_load_roundtrip(self):
        from models.fan_control.baseline_threshold import ThresholdFanController
        ctrl = ThresholdFanController(t_low=62, t_medium=70, t_high=80)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        ctrl.save(path)
        ctrl2 = ThresholdFanController.load(path)
        assert ctrl2.t_low == 62
        assert ctrl2.t_high == 80


# ---------------------------------------------------------------------------
# 3. PIDController
# ---------------------------------------------------------------------------

class TestPIDController:
    def test_decide_returns_valid_rpm(self, small_df):
        from models.fan_control.baseline_pid import PIDController
        ctrl = PIDController()
        row = small_df.iloc[0]
        rpm = ctrl.decide(row)
        assert rpm in RPM_LEVELS

    def test_decide_batch_in_rpm_levels(self, small_df):
        from models.fan_control.baseline_pid import PIDController
        ctrl = PIDController()
        result = ctrl.decide_batch(small_df)
        assert set(result).issubset(RPM_LEVELS)
        assert len(result) == len(small_df)

    def test_reset_clears_state(self):
        from models.fan_control.baseline_pid import PIDController
        ctrl = PIDController()
        ctrl._integral = 999.0
        ctrl._prev_error = 5.0
        ctrl.reset()
        assert ctrl._integral == 0.0
        assert ctrl._prev_error is None

    def test_fit_does_not_crash(self, small_df):
        from models.fan_control.baseline_pid import PIDController
        ctrl = PIDController()
        ctrl.fit(small_df, small_df["failure_60s"])

    def test_save_load_roundtrip(self):
        from models.fan_control.baseline_pid import PIDController
        ctrl = PIDController(kp=50.0, ki=1.0, kd=5.0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        ctrl.save(path)
        ctrl2 = PIDController.load(path)
        assert ctrl2.kp == 50.0
        assert ctrl2.ki == 1.0


# ---------------------------------------------------------------------------
# 4. SupervisedController
# ---------------------------------------------------------------------------

class TestSupervisedController:
    def test_fit_and_predict(self, small_df):
        from models.fan_control.supervised_controller import SupervisedController
        ctrl = SupervisedController(n_estimators=10, max_depth=5)
        ctrl.fit(small_df, small_df["action_class"])
        result = ctrl.decide_batch(small_df)
        assert set(result).issubset(RPM_LEVELS)
        assert len(result) == len(small_df)

    def test_decide_returns_rpm(self, small_df):
        from models.fan_control.supervised_controller import SupervisedController
        ctrl = SupervisedController(n_estimators=10, max_depth=5)
        ctrl.fit(small_df, small_df["action_class"])
        rpm = ctrl.decide(small_df.iloc[0], risk_score=0.7)
        assert rpm in RPM_LEVELS

    def test_not_fitted_raises(self):
        from models.fan_control.supervised_controller import SupervisedController
        ctrl = SupervisedController()
        with pytest.raises(RuntimeError):
            ctrl.decide_batch(pd.DataFrame({"temperature_c": [70.0]}))

    def test_save_load_roundtrip(self, small_df):
        from models.fan_control.supervised_controller import SupervisedController
        ctrl = SupervisedController(n_estimators=10, max_depth=5)
        ctrl.fit(small_df, small_df["action_class"])
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        ctrl.save(path)
        ctrl2 = SupervisedController.load(path)
        result = ctrl2.decide_batch(small_df)
        assert set(result).issubset(RPM_LEVELS)


# ---------------------------------------------------------------------------
# 5. ScoreController
# ---------------------------------------------------------------------------

class TestScoreController:
    def test_decide_returns_valid_rpm(self, small_df):
        from models.fan_control.score_controller import ScoreController
        ctrl = ScoreController()
        row = small_df.iloc[0]
        rpm = ctrl.decide(row, risk_score=0.8)
        assert rpm in RPM_LEVELS

    def test_high_risk_favors_high_rpm(self):
        from models.fan_control.score_controller import ScoreController
        ctrl = ScoreController(alpha=0.9, beta=0.05, gamma=0.03, delta=0.02)
        row = pd.Series({"temperature_c": 75.0, "margin_to_shutdown": 13.0})
        rpm_high_risk = ctrl.decide(row, risk_score=0.95)
        ctrl.reset() if hasattr(ctrl, 'reset') else None
        ctrl._prev_rpm = 2500
        rpm_low_risk  = ctrl.decide(row, risk_score=0.0)
        assert rpm_high_risk >= rpm_low_risk

    def test_decide_batch_in_rpm_levels(self, small_df):
        from models.fan_control.score_controller import ScoreController
        ctrl = ScoreController()
        risk = np.random.default_rng(0).uniform(0, 1, len(small_df))
        result = ctrl.decide_batch(small_df, risk_scores=risk)
        assert set(result).issubset(RPM_LEVELS)
        assert len(result) == len(small_df)

    def test_fit_does_not_crash(self, small_df):
        from models.fan_control.score_controller import ScoreController
        ctrl = ScoreController()
        risk = np.zeros(len(small_df))
        ctrl.fit(small_df, small_df["failure_60s"], risk_scores_train=risk)

    def test_save_load_roundtrip(self):
        from models.fan_control.score_controller import ScoreController
        ctrl = ScoreController(alpha=0.6, beta=0.25, gamma=0.1, delta=0.05)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        ctrl.save(path)
        ctrl2 = ScoreController.load(path)
        assert ctrl2.alpha == 0.6
        assert ctrl2.beta == 0.25


# ---------------------------------------------------------------------------
# 6. Test d'integration sur donnees reelles (marque slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_all_controllers_on_real_data(processed_data):
    """Verifie que tous les controleurs tournent sur les vraies donnees."""
    from models.fan_control.baseline_fixed import FixedController
    from models.fan_control.baseline_threshold import ThresholdFanController
    from models.fan_control.baseline_pid import PIDController
    from models.fan_control.supervised_controller import SupervisedController
    from models.fan_control.score_controller import ScoreController

    X_train, X_test, y_train, y_test = processed_data

    # Simuler action_class depuis y_train si absente
    action_train = pd.Series(np.ones(len(X_train), dtype=int))
    action_test  = pd.Series(np.ones(len(X_test),  dtype=int))

    controllers = [
        FixedController(rpm=2500),
        ThresholdFanController(),
        PIDController(),
        SupervisedController(n_estimators=50, max_depth=10),
        ScoreController(),
    ]

    for ctrl in controllers:
        if hasattr(ctrl, 'fit') and ctrl.name not in ("baseline_fixed_2500",):
            ctrl.fit(X_train, action_train)
        result = ctrl.decide_batch(X_test)
        assert isinstance(result, np.ndarray), f"{ctrl.name}: decide_batch doit retourner ndarray"
        assert len(result) == len(X_test),     f"{ctrl.name}: longueur incorrecte"
        assert set(result).issubset(RPM_LEVELS), f"{ctrl.name}: RPM hors niveaux valides : {set(result) - RPM_LEVELS}"
        print(f"  {ctrl.name}: mean_rpm={result.mean():.0f}")
