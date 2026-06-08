"""Tests unitaires — module features.

Teste temporal.py, contextual.py, energy.py, labeler.py et pipeline.py
sur des DataFrames synthétiques reproductibles.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from features.temporal import add_temporal_features, feature_names_temporal
from features.contextual import add_contextual_features, feature_names_contextual
from features.energy import add_energy_features, feature_names_energy
from features.labeler import (
    add_failure_labels,
    add_control_labels,
    RPM_LEVELS,
    _is_dangerous_status,
)
from features.pipeline import build_feature_dataset, all_feature_names, all_label_names


# ---------------------------------------------------------------------------
# Helpers — génération de DataFrames synthétiques
# ---------------------------------------------------------------------------

T_SHUTDOWN = 88.0
N_ROWS = 120  # 2 minutes à 1 Hz


def _make_base_df(
    n: int = N_ROWS,
    temp_start: float = 60.0,
    temp_end: float = 80.0,
    status: str = "on",
    fan_rpm: float = 2500.0,
) -> pd.DataFrame:
    """Crée un DataFrame de télémétrie synthétique pour une machine."""
    ts = [datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(seconds=i) for i in range(n)]
    temps = np.linspace(temp_start, temp_end, n)
    return pd.DataFrame({
        "timestamp": ts,
        "cluster_id": "cluster_alpha",
        "machine_id": "srv-worker-01",
        "role": "worker",
        "msg_type": "telemetry",
        "status": status,
        "temperature_c": temps,
        "sensor_temp_max": temps + 0.5,
        "sensor_temp_mean": temps + 0.2,
        "power_w": 800.0 + temps * 2,
        "energy_kwh": np.arange(n) * 0.001,
        "fan_rpm_mean": fan_rpm,
        "fan_rpm_std": 50.0,
        "fan_count": 2,
        "fan_modes": "auto,auto",
        "load_estimated": 0.5,
        "has_fault": False,
        "fault_types": "",
        "fault_count": 0,
    })


def _make_incident_df(n: int = N_ROWS) -> pd.DataFrame:
    """Crée un DataFrame avec un incident thermique à mi-parcours."""
    df = _make_base_df(n=n, temp_start=60.0, temp_end=95.0)
    # Simuler un passage en degraded à 70% puis off à 90%
    shutdown_idx = int(n * 0.85)
    degraded_idx = int(n * 0.70)
    statuses = ["on"] * n
    for i in range(degraded_idx, shutdown_idx):
        statuses[i] = "degraded"
    for i in range(shutdown_idx, n):
        statuses[i] = "off"
    df["status"] = statuses
    df["status_cause"] = ["normal"] * degraded_idx + ["overheat_partial"] * (shutdown_idx - degraded_idx) + ["overheat"] * (n - shutdown_idx)
    return df


# ---------------------------------------------------------------------------
# Tests features temporelles
# ---------------------------------------------------------------------------

class TestTemporalFeatures:
    def test_output_columns_present(self):
        df = _make_base_df()
        df_out = add_temporal_features(df, t_shutdown_c=T_SHUTDOWN)
        for col in ["temp_delta_5s", "temp_delta_30s", "margin_to_shutdown",
                    "margin_pct", "temp_rolling_mean_30s", "temp_rolling_mean_60s"]:
            assert col in df_out.columns, f"Colonne manquante : {col}"

    def test_margin_to_shutdown_correct(self):
        df = _make_base_df(temp_start=70.0, temp_end=70.0)
        df_out = add_temporal_features(df, t_shutdown_c=T_SHUTDOWN)
        expected_margin = T_SHUTDOWN - 70.0
        assert df_out["margin_to_shutdown"].iloc[-1] == pytest.approx(expected_margin, abs=0.1)

    def test_margin_pct_bounded(self):
        df = _make_base_df(temp_start=20.0, temp_end=90.0)
        df_out = add_temporal_features(df, t_shutdown_c=T_SHUTDOWN)
        assert (df_out["margin_pct"] >= 0).all()

    def test_temp_delta_5s_first_rows_nan(self):
        df = _make_base_df()
        df_out = add_temporal_features(df, t_shutdown_c=T_SHUTDOWN)
        # Les 5 premières lignes de temp_delta_5s doivent être NaN
        assert df_out["temp_delta_5s"].iloc[:5].isna().all()

    def test_temp_delta_30s_positive_when_rising(self):
        df = _make_base_df(temp_start=60.0, temp_end=85.0)
        df_out = add_temporal_features(df, t_shutdown_c=T_SHUTDOWN)
        # Après 30 lignes, delta doit être positif (temp monte)
        assert df_out["temp_delta_30s"].iloc[35:].mean() > 0

    def test_rolling_mean_30s_shape(self):
        df = _make_base_df()
        df_out = add_temporal_features(df, t_shutdown_c=T_SHUTDOWN)
        assert len(df_out) == len(df)
        assert df_out["temp_rolling_mean_30s"].notna().all()

    def test_rpm_features_present(self):
        df = _make_base_df(fan_rpm=3000.0)
        df_out = add_temporal_features(df, t_shutdown_c=T_SHUTDOWN)
        assert "rpm_variance" in df_out.columns
        assert "rpm_cv" in df_out.columns
        assert "rpm_rolling_mean_30s" in df_out.columns

    def test_feature_names_list_nonempty(self):
        names = feature_names_temporal()
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# Tests features contextuelles
# ---------------------------------------------------------------------------

class TestContextualFeatures:
    def test_time_in_hot_zone_accumulates(self):
        # Température juste au-dessus de 80% du seuil = 70.4°C
        df = _make_base_df(temp_start=72.0, temp_end=72.0)
        df_out = add_contextual_features(df, t_shutdown_c=T_SHUTDOWN)
        # Doit augmenter monotoniquement
        assert (df_out["time_in_hot_zone_s"].diff().iloc[1:] >= 0).all()
        assert df_out["time_in_hot_zone_s"].iloc[-1] > 0

    def test_time_in_hot_zone_zero_when_cold(self):
        df = _make_base_df(temp_start=50.0, temp_end=55.0)
        df_out = add_contextual_features(df, t_shutdown_c=T_SHUTDOWN)
        assert (df_out["time_in_hot_zone_s"] == 0).all()

    def test_nb_shutdowns_counts_correctly(self):
        df = _make_incident_df()
        df_out = add_contextual_features(df, t_shutdown_c=T_SHUTDOWN)
        # Il doit y avoir au moins 1 shutdown comptabilisé
        assert df_out["nb_shutdowns_episode"].iloc[-1] >= 1

    def test_nb_degraded_counts_correctly(self):
        df = _make_incident_df()
        df_out = add_contextual_features(df, t_shutdown_c=T_SHUTDOWN)
        assert df_out["nb_degraded_episode"].iloc[-1] >= 1

    def test_has_fan_fault_detected(self):
        df = _make_base_df()
        df["fault_types"] = ["fan_failure" if i > 50 else "" for i in range(len(df))]
        df["has_fault"] = df["fault_types"] != ""
        df_out = add_contextual_features(df, t_shutdown_c=T_SHUTDOWN)
        assert df_out["has_fan_fault"].iloc[60] == 1
        assert df_out["has_fan_fault"].iloc[10] == 0

    def test_fan_mode_manual_detected(self):
        df = _make_base_df()
        df["fan_modes"] = "manual,manual"
        df_out = add_contextual_features(df, t_shutdown_c=T_SHUTDOWN)
        assert (df_out["fan_mode_manual"] == 1).all()

    def test_rpm_changes_counted(self):
        df = _make_base_df(fan_rpm=2500.0)
        # Introduire des changements de RPM
        df.loc[10, "fan_rpm_mean"] = 4500.0
        df.loc[30, "fan_rpm_mean"] = 1500.0
        df.loc[60, "fan_rpm_mean"] = 3500.0
        df_out = add_contextual_features(df, t_shutdown_c=T_SHUTDOWN)
        # La fenêtre de 60s doit capturer les changements
        assert df_out["rpm_changes_last_60s"].max() >= 1

    def test_feature_names_list_nonempty(self):
        names = feature_names_contextual()
        assert len(names) > 0


# ---------------------------------------------------------------------------
# Tests features énergétiques
# ---------------------------------------------------------------------------

class TestEnergyFeatures:
    def test_power_fans_cubic_law(self):
        df = _make_base_df(fan_rpm=5000.0)  # RPM max → puissance nominale × 2 fans
        df_out = add_energy_features(df, fan_max_rpm=5000, fan_power_nominal_w=12.0)
        # À RPM max : P_fans = 12.0 × (5000/5000)³ × 2 = 24.0 W
        assert df_out["power_fans_w"].iloc[0] == pytest.approx(24.0, abs=0.5)

    def test_power_fans_zero_at_zero_rpm(self):
        df = _make_base_df(fan_rpm=0.0)
        df_out = add_energy_features(df, fan_max_rpm=5000, fan_power_nominal_w=12.0)
        assert (df_out["power_fans_w"] == 0.0).all()

    def test_fan_energy_ratio_bounded(self):
        df = _make_base_df(fan_rpm=2500.0)
        df_out = add_energy_features(df, fan_max_rpm=5000, fan_power_nominal_w=12.0)
        assert (df_out["fan_energy_ratio"] >= 0).all()
        assert (df_out["fan_energy_ratio"] <= 1).all()

    def test_pue_estimated_above_one(self):
        df = _make_base_df(fan_rpm=2500.0)
        df_out = add_energy_features(df, fan_max_rpm=5000, fan_power_nominal_w=12.0)
        assert (df_out["pue_estimated"] >= 1.0).all()

    def test_energy_fans_kwh_monotonic(self):
        df = _make_base_df(fan_rpm=2500.0)
        df_out = add_energy_features(df, fan_max_rpm=5000, fan_power_nominal_w=12.0)
        assert (df_out["energy_fans_kwh_cumulated"].diff().iloc[1:] >= 0).all()

    def test_power_compute_non_negative(self):
        df = _make_base_df(fan_rpm=1500.0)
        df_out = add_energy_features(df, fan_max_rpm=5000, fan_power_nominal_w=12.0)
        assert (df_out["power_compute_w"] >= 0).all()

    def test_feature_names_list_nonempty(self):
        names = feature_names_energy()
        assert len(names) > 0


# ---------------------------------------------------------------------------
# Tests labeler
# ---------------------------------------------------------------------------

class TestLabeler:
    def test_failure_60s_detected(self):
        df = _make_incident_df()
        df_out = add_failure_labels(df, t_shutdown_c=T_SHUTDOWN)
        # Les lignes avant le shutdown doivent avoir failure_60s=1
        shutdown_idx = int(N_ROWS * 0.85)
        # À 60s avant le shutdown
        idx_before = max(0, shutdown_idx - 60)
        assert df_out["failure_60s"].iloc[idx_before] == 1

    def test_failure_0s_after_incident(self):
        df = _make_incident_df()
        df_out = add_failure_labels(df, t_shutdown_c=T_SHUTDOWN)
        # Après le shutdown, pas d'incident dans les 60s suivantes (fin de série)
        assert df_out["failure_60s"].iloc[-1] == 0

    def test_failure_30s_subset_of_60s(self):
        df = _make_incident_df()
        df_out = add_failure_labels(df, t_shutdown_c=T_SHUTDOWN)
        # Tout ce qui est failure_30s=1 doit aussi être failure_60s=1
        assert (df_out.loc[df_out["failure_30s"] == 1, "failure_60s"] == 1).all()

    def test_time_to_failure_decreasing_before_incident(self):
        df = _make_incident_df()
        df_out = add_failure_labels(df, t_shutdown_c=T_SHUTDOWN)
        # Avant le premier incident, time_to_failure doit décroître
        degraded_idx = int(N_ROWS * 0.70)
        ttf = df_out["time_to_failure_s"].iloc[:degraded_idx].dropna()
        if len(ttf) > 1:
            # Globalement décroissant
            assert ttf.iloc[0] > ttf.iloc[-1]

    def test_no_false_positives_on_stable_machine(self):
        df = _make_base_df(temp_start=55.0, temp_end=60.0, status="on")
        df_out = add_failure_labels(df, t_shutdown_c=T_SHUTDOWN)
        # Machine stable → pas d'alerte
        assert df_out["failure_60s"].sum() == 0
        assert df_out["failure_30s"].sum() == 0

    def test_action_class_bounded(self):
        df = _make_base_df(temp_start=40.0, temp_end=90.0)
        df_out = add_control_labels(df, t_shutdown_c=T_SHUTDOWN)
        assert df_out["action_class"].between(0, len(RPM_LEVELS) - 1).all()

    def test_optimal_rpm_in_levels(self):
        df = _make_base_df()
        df_out = add_control_labels(df, t_shutdown_c=T_SHUTDOWN)
        assert df_out["optimal_rpm"].isin(RPM_LEVELS).all()

    def test_action_class_max_when_degraded(self):
        df = _make_base_df(temp_start=85.0, temp_end=87.0, status="degraded")
        df_out = add_control_labels(df, t_shutdown_c=T_SHUTDOWN)
        # En mode dégradé → classe max
        assert (df_out["action_class"] == len(RPM_LEVELS) - 1).all()

    def test_is_dangerous_status(self):
        df = pd.DataFrame({"status": ["on", "degraded", "off", "on"]})
        result = _is_dangerous_status(df)
        assert result[0] == False
        assert result[1] == True
        assert result[2] == True  # off sans cause = dangereux par défaut


# ---------------------------------------------------------------------------
# Tests pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_pipeline_runs_end_to_end(self):
        df = _make_base_df(n=200)
        df_out = build_feature_dataset(df, machine_config={"t_shutdown_c": T_SHUTDOWN})
        assert len(df_out) > 0
        assert "failure_60s" in df_out.columns
        assert "action_class" in df_out.columns

    def test_pipeline_filters_non_telemetry(self):
        df = _make_base_df(n=100)
        # Ajouter des lignes de type status_event (doivent être ignorées)
        extra = df.head(10).copy()
        extra["msg_type"] = "status_event"
        df_mixed = pd.concat([df, extra], ignore_index=True)
        df_out = build_feature_dataset(df_mixed, machine_config={"t_shutdown_c": T_SHUTDOWN})
        # Seules les lignes telemetry doivent être traitées
        assert (df_out["msg_type"] == "telemetry").all()

    def test_pipeline_drops_warmup_rows(self):
        n = 200
        df = _make_base_df(n=n)
        df_out = build_feature_dataset(df, machine_config={"t_shutdown_c": T_SHUTDOWN}, drop_warmup_rows=60)
        assert len(df_out) <= n - 60

    def test_pipeline_multi_machine(self):
        df1 = _make_base_df(n=150)
        df2 = _make_base_df(n=150)
        df2["machine_id"] = "srv-master-01"
        df2["role"] = "master"
        df_both = pd.concat([df1, df2], ignore_index=True)
        cfg = {
            "srv-worker-01": {"t_shutdown_c": 88.0},
            "srv-master-01": {"t_shutdown_c": 90.0},
        }
        df_out = build_feature_dataset(df_both, machine_config=cfg)
        assert df_out["machine_id"].nunique() == 2

    def test_all_feature_names_nonempty(self):
        names = all_feature_names()
        assert len(names) > 10

    def test_all_label_names_nonempty(self):
        names = all_label_names()
        assert "failure_60s" in names
        assert "action_class" in names
