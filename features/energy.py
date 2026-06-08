"""Features énergétiques — Juste des Ventilateurs.

Calcule les indicateurs de consommation des ventilateurs, le PUE estimé
et l'efficacité du refroidissement.

Modèle de puissance des fans (cohérent avec jumeaux-chauds/physics.py) :
    P_fan(RPM) = P_nominal × (RPM / RPM_max)³   [loi cubique]

Usage typique :
    df = add_energy_features(df, fan_max_rpm=5000, fan_power_nominal_w=12.0)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Valeurs par défaut issues de base.yaml
_FAN_MAX_RPM_DEFAULT = 5000
_FAN_POWER_NOMINAL_W_WORKER = 12.0   # power_per_fan_w worker
_FAN_POWER_NOMINAL_W_MASTER = 15.0   # power_per_fan_w master
_PUE_BASELINE = 1.40                 # PUE global cluster (base.yaml)


def add_energy_features(
    df: pd.DataFrame,
    fan_max_rpm: int = _FAN_MAX_RPM_DEFAULT,
    fan_power_nominal_w: float | None = None,
    pue_baseline: float = _PUE_BASELINE,
    tick_hz: float = 1.0,
) -> pd.DataFrame:
    """Ajoute les features énergétiques au DataFrame.

    Parameters
    ----------
    df                  : DataFrame de télémétrie normalisée (une machine, trié par ts)
    fan_max_rpm         : RPM maximum des ventilateurs de la machine
    fan_power_nominal_w : Puissance nominale d'un fan à RPM max (W).
                          Si None, déduit depuis la colonne 'role'.
    pue_baseline        : PUE de référence du cluster
    tick_hz             : fréquence de publication (messages/seconde)

    Returns
    -------
    DataFrame enrichi
    """
    df = df.copy()

    if "power_w" not in df.columns or "fan_rpm_mean" not in df.columns:
        logger.warning("Colonnes power_w ou fan_rpm_mean manquantes — features énergie ignorées.")
        return df

    # Résoudre la puissance nominale par rôle si non fournie
    if fan_power_nominal_w is None:
        if "role" in df.columns:
            role = df["role"].mode()[0] if not df["role"].empty else "worker"
            fan_power_nominal_w = (
                _FAN_POWER_NOMINAL_W_MASTER if role == "master"
                else _FAN_POWER_NOMINAL_W_WORKER
            )
        else:
            fan_power_nominal_w = _FAN_POWER_NOMINAL_W_WORKER

    fan_count = df["fan_count"].fillna(2).astype(float)

    # ------------------------------------------------------------------
    # 1. Puissance estimée des fans (loi cubique : P ∝ RPM³)
    # ------------------------------------------------------------------
    rpm_ratio = (df["fan_rpm_mean"] / fan_max_rpm).clip(0.0, 1.0)
    df["power_fans_w"] = fan_power_nominal_w * (rpm_ratio ** 3) * fan_count

    # ------------------------------------------------------------------
    # 2. Puissance de calcul (totale - fans)
    # ------------------------------------------------------------------
    df["power_compute_w"] = (df["power_w"] - df["power_fans_w"]).clip(lower=0.0)

    # ------------------------------------------------------------------
    # 3. Ratio énergie fans / énergie totale
    # ------------------------------------------------------------------
    total_nonzero = df["power_w"].replace(0, np.nan)
    df["fan_energy_ratio"] = (df["power_fans_w"] / total_nonzero).fillna(0.0).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # 4. PUE estimé : 1 + P_fans / P_compute
    #    (overhead de refroidissement par rapport à la charge utile)
    # ------------------------------------------------------------------
    compute_nonzero = df["power_compute_w"].replace(0, np.nan)
    df["pue_estimated"] = (1.0 + df["power_fans_w"] / compute_nonzero).fillna(pue_baseline)

    # ------------------------------------------------------------------
    # 5. Efficacité du refroidissement : °C refroidi par Watt de fan
    #    Plus élevé = meilleur rapport efficacité/énergie
    # ------------------------------------------------------------------
    if "temperature_c" in df.columns and "margin_to_shutdown" in df.columns:
        # kWh consommé par °C de marge au shutdown (proxy d'efficacité)
        margin_nonzero = df["margin_to_shutdown"].replace(0, np.nan)
        df["energy_per_temp_unit"] = (df["power_fans_w"] / margin_nonzero).fillna(0.0).clip(lower=0.0)

    # ------------------------------------------------------------------
    # 6. Énergie cumulée des fans sur l'épisode (kWh)
    #    Intégration trapézoïdale : P × Δt / 3600
    # ------------------------------------------------------------------
    dt = 1.0 / tick_hz
    df["energy_fans_kwh_cumulated"] = (df["power_fans_w"] * dt / 3600.0).cumsum()

    # ------------------------------------------------------------------
    # 7. Rolling means énergie (30s)
    # ------------------------------------------------------------------
    window_30 = max(1, int(30 * tick_hz))
    df["power_fans_rolling_mean_30s"] = (
        df["power_fans_w"].rolling(window=window_30, min_periods=1).mean()
    )
    df["pue_rolling_mean_30s"] = (
        df["pue_estimated"].rolling(window=window_30, min_periods=1).mean()
    )

    return df


def feature_names_energy() -> list[str]:
    """Retourne la liste des noms de features énergétiques produites."""
    return [
        "power_fans_w",
        "power_compute_w",
        "fan_energy_ratio",
        "pue_estimated",
        "energy_per_temp_unit",
        "energy_fans_kwh_cumulated",
        "power_fans_rolling_mean_30s",
        "pue_rolling_mean_30s",
    ]
