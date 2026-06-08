"""Features temporelles — Juste des Ventilateurs.

Calcule les dérivées, moyennes glissantes et indicateurs de marge thermique
sur un DataFrame de télémétrie normalisée (sortie du Normalizer).

Toutes les fonctions opèrent sur un DataFrame **trié par timestamp**,
groupé par machine_id. Le DataFrame d'entrée est celui produit par
ingest/normalizer.py (schéma unifié).

Usage typique :
    df_raw = pd.read_parquet("data/raw/episode=001/machine=srv-worker-01/")
    df_feat = add_temporal_features(df_raw, t_shutdown_c=88.0)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fenêtres temporelles en secondes
_WINDOWS_S = [5, 15, 30, 60]

# Fréquence nominale de publication jumeaux-chauds : 1 message/s
_TICK_HZ = 1.0


def add_temporal_features(
    df: pd.DataFrame,
    t_shutdown_c: float = 88.0,
    tick_hz: float = _TICK_HZ,
) -> pd.DataFrame:
    """Ajoute toutes les features temporelles au DataFrame.

    Le DataFrame doit être filtré sur un seul machine_id et trié par timestamp.

    Parameters
    ----------
    df          : DataFrame de télémétrie normalisée (une machine, trié par ts)
    t_shutdown_c: seuil de shutdown thermique de la machine (en °C)
    tick_hz     : fréquence de publication (messages/seconde)

    Returns
    -------
    DataFrame enrichi (nouvelles colonnes ajoutées, index préservé)
    """
    df = df.copy()

    if "temperature_c" not in df.columns:
        logger.warning("Colonne 'temperature_c' manquante — features temporelles ignorées.")
        return df

    # Nombre de lignes correspondant à chaque fenêtre temporelle
    w = {s: max(1, int(s * tick_hz)) for s in _WINDOWS_S}

    # ------------------------------------------------------------------
    # 1. Dérivées de température (différences décalées)
    # ------------------------------------------------------------------
    for s, rows in w.items():
        if s in (5, 15, 30):
            df[f"temp_delta_{s}s"] = df["temperature_c"].diff(periods=rows)

    # ------------------------------------------------------------------
    # 2. Moyennes glissantes de température
    # ------------------------------------------------------------------
    for s in (30, 60):
        rows = w[s]
        df[f"temp_rolling_mean_{s}s"] = (
            df["temperature_c"]
            .rolling(window=rows, min_periods=1)
            .mean()
        )

    # ------------------------------------------------------------------
    # 3. Écart-type glissant de température (volatilité thermique)
    # ------------------------------------------------------------------
    df["temp_rolling_std_30s"] = (
        df["temperature_c"]
        .rolling(window=w[30], min_periods=2)
        .std()
        .fillna(0.0)
    )

    # ------------------------------------------------------------------
    # 4. Marge au shutdown
    # ------------------------------------------------------------------
    df["margin_to_shutdown"] = t_shutdown_c - df["temperature_c"]
    df["margin_pct"] = (df["margin_to_shutdown"] / t_shutdown_c * 100).clip(lower=0.0)

    # Dérivée de la marge (vitesse d'approche du seuil, positif = danger croissant)
    df["margin_delta_30s"] = -df["temp_delta_30s"] if "temp_delta_30s" in df.columns else np.nan

    # ------------------------------------------------------------------
    # 5. Moyennes glissantes de charge
    # ------------------------------------------------------------------
    if "load_estimated" in df.columns:
        for s in (30, 60):
            rows = w[s]
            df[f"load_rolling_mean_{s}s"] = (
                df["load_estimated"]
                .rolling(window=rows, min_periods=1)
                .mean()
            )

    # ------------------------------------------------------------------
    # 6. Features sur les RPM ventilateurs
    # ------------------------------------------------------------------
    if "fan_rpm_mean" in df.columns:
        df["rpm_variance"] = df["fan_rpm_std"] ** 2 if "fan_rpm_std" in df.columns else 0.0
        # Coefficient de variation : std / mean (0 si mean=0)
        df["rpm_cv"] = np.where(
            df["fan_rpm_mean"] > 0,
            df["fan_rpm_std"] / df["fan_rpm_mean"],
            0.0,
        )
        # Dérivée RPM : taux de changement des ventilateurs
        df["rpm_delta_15s"] = df["fan_rpm_mean"].diff(periods=w[15])
        # Moyenne glissante RPM
        df["rpm_rolling_mean_30s"] = (
            df["fan_rpm_mean"]
            .rolling(window=w[30], min_periods=1)
            .mean()
        )

    # ------------------------------------------------------------------
    # 7. Features sur la puissance
    # ------------------------------------------------------------------
    if "power_w" in df.columns:
        df["power_rolling_mean_30s"] = (
            df["power_w"]
            .rolling(window=w[30], min_periods=1)
            .mean()
        )
        df["power_delta_30s"] = df["power_w"].diff(periods=w[30])

    # ------------------------------------------------------------------
    # 8. Température max capteur (si disponible)
    # ------------------------------------------------------------------
    if "sensor_temp_max" in df.columns:
        df["sensor_max_delta_15s"] = df["sensor_temp_max"].diff(periods=w[15])
        df["sensor_max_rolling_mean_30s"] = (
            df["sensor_temp_max"]
            .rolling(window=w[30], min_periods=1)
            .mean()
        )

    return df


def feature_names_temporal() -> list[str]:
    """Retourne la liste des noms de features temporelles produites."""
    return [
        "temp_delta_5s", "temp_delta_15s", "temp_delta_30s",
        "temp_rolling_mean_30s", "temp_rolling_mean_60s",
        "temp_rolling_std_30s",
        "margin_to_shutdown", "margin_pct", "margin_delta_30s",
        "load_rolling_mean_30s", "load_rolling_mean_60s",
        "rpm_variance", "rpm_cv", "rpm_delta_15s", "rpm_rolling_mean_30s",
        "power_rolling_mean_30s", "power_delta_30s",
        "sensor_max_delta_15s", "sensor_max_rolling_mean_30s",
    ]
