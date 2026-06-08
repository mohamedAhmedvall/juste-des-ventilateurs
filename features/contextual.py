"""Features contextuelles — Juste des Ventilateurs.

Calcule les indicateurs d'historique : durée en zone chaude, compteurs
d'incidents, types de pannes actives, changements de consigne ventilateur.

Ces features capturent le contexte opérationnel que les features temporelles
pures ne voient pas : "la machine est en zone chaude depuis combien de temps ?",
"combien de shutdowns depuis le début de l'épisode ?".

Usage typique :
    df = add_contextual_features(df, t_shutdown_c=88.0, tick_hz=1.0)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Seuil "zone chaude" : 80% du shutdown
_HOT_ZONE_RATIO = 0.80

# Fenêtre pour comptage des changements RPM (en secondes)
_RPM_CHANGE_WINDOW_S = 60


def add_contextual_features(
    df: pd.DataFrame,
    t_shutdown_c: float = 88.0,
    tick_hz: float = 1.0,
) -> pd.DataFrame:
    """Ajoute toutes les features contextuelles au DataFrame.

    Le DataFrame doit être filtré sur un seul machine_id et trié par timestamp.

    Parameters
    ----------
    df          : DataFrame de télémétrie normalisée (une machine, trié par ts)
    t_shutdown_c: seuil de shutdown thermique (°C)
    tick_hz     : fréquence de publication (messages/seconde)

    Returns
    -------
    DataFrame enrichi (nouvelles colonnes ajoutées)
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        return df

    hot_threshold = t_shutdown_c * _HOT_ZONE_RATIO
    dt = 1.0 / tick_hz  # durée d'un tick en secondes

    # ------------------------------------------------------------------
    # 1. Durée cumulée en zone chaude (T > 80% du seuil shutdown)
    # ------------------------------------------------------------------
    if "temperature_c" in df.columns:
        in_hot_zone = (df["temperature_c"] > hot_threshold).astype(float)
        # Durée cumulée : somme des ticks en zone chaude × dt
        # Réinitialisé quand on sort de la zone chaude
        time_in_hot = np.zeros(n)
        cumulative = 0.0
        for i, hot in enumerate(in_hot_zone):
            if hot:
                cumulative += dt
            else:
                cumulative = 0.0
            time_in_hot[i] = cumulative
        df["time_in_hot_zone_s"] = time_in_hot

    # ------------------------------------------------------------------
    # 2. Durée en mode dégradé (status == "degraded")
    # ------------------------------------------------------------------
    if "status" in df.columns:
        in_degraded = (df["status"] == "degraded").astype(float)
        time_in_degraded = np.zeros(n)
        cumulative = 0.0
        for i, deg in enumerate(in_degraded):
            if deg:
                cumulative += dt
            else:
                cumulative = 0.0
            time_in_degraded[i] = cumulative
        df["time_in_degraded_s"] = time_in_degraded

        # ------------------------------------------------------------------
        # 3. Compteurs d'événements depuis le début de l'épisode
        # ------------------------------------------------------------------
        statuses = df["status"].values
        nb_shutdowns = np.zeros(n, dtype=int)
        nb_degraded = np.zeros(n, dtype=int)
        prev_status = statuses[0] if n > 0 else "on"
        s_count = 0
        d_count = 0

        for i in range(n):
            cur = statuses[i]
            # Shutdown : transition vers "off" depuis un état actif
            if cur == "off" and prev_status in ("on", "degraded"):
                s_count += 1
            # Dégradé : transition vers "degraded"
            if cur == "degraded" and prev_status == "on":
                d_count += 1
            nb_shutdowns[i] = s_count
            nb_degraded[i] = d_count
            prev_status = cur

        df["nb_shutdowns_episode"] = nb_shutdowns
        df["nb_degraded_episode"] = nb_degraded

        # Ticks depuis le dernier shutdown
        ticks_since_shutdown = np.zeros(n, dtype=int)
        counter = 0
        for i in range(n):
            if nb_shutdowns[i] > (nb_shutdowns[i - 1] if i > 0 else 0):
                counter = 0
            else:
                counter += 1
            ticks_since_shutdown[i] = counter
        df["ticks_since_last_shutdown"] = ticks_since_shutdown

    # ------------------------------------------------------------------
    # 4. Indicateurs de pannes actives (depuis fault_types)
    # ------------------------------------------------------------------
    if "fault_types" in df.columns:
        fault_col = df["fault_types"].fillna("").astype(str)
        df["has_fan_fault"] = fault_col.str.contains("fan_failure", na=False).astype(int)
        df["has_power_surge"] = fault_col.str.contains("power_surge", na=False).astype(int)
        df["has_sensor_drift"] = fault_col.str.contains("sensor_drift", na=False).astype(int)

        # Ticks depuis la dernière panne (toutes pannes confondues)
        has_any_fault = df["has_fault"].fillna(False).astype(bool).values if "has_fault" in df.columns else (fault_col != "").values
        ticks_since_fault = np.zeros(n, dtype=int)
        counter = 0
        for i in range(n):
            if has_any_fault[i]:
                counter = 0
            else:
                counter += 1
            ticks_since_fault[i] = counter
        df["ticks_since_last_fault"] = ticks_since_fault

    # ------------------------------------------------------------------
    # 5. Mode ventilateur : au moins un fan en mode manual
    # ------------------------------------------------------------------
    if "fan_modes" in df.columns:
        df["fan_mode_manual"] = (
            df["fan_modes"].fillna("").str.contains("manual", na=False).astype(int)
        )

    # ------------------------------------------------------------------
    # 6. Nombre de changements de consigne RPM sur les 60 dernières secondes
    # ------------------------------------------------------------------
    if "fan_rpm_mean" in df.columns:
        window = max(1, int(_RPM_CHANGE_WINDOW_S * tick_hz))
        # Un changement = variation absolue de RPM mean > seuil minimal (50 RPM)
        rpm_changed = (df["fan_rpm_mean"].diff().abs() > 50).astype(int)
        df["rpm_changes_last_60s"] = (
            rpm_changed
            .rolling(window=window, min_periods=1)
            .sum()
            .astype(int)
        )

    # ------------------------------------------------------------------
    # 7. Indicateur de récupération (machine revenue en "on" après incident)
    # ------------------------------------------------------------------
    if "status" in df.columns:
        statuses = df["status"].values
        recovering = np.zeros(n, dtype=int)
        for i in range(1, n):
            if statuses[i] == "on" and statuses[i - 1] in ("off", "degraded"):
                recovering[i] = 1
        df["is_recovering"] = recovering

    return df


def feature_names_contextual() -> list[str]:
    """Retourne la liste des noms de features contextuelles produites."""
    return [
        "time_in_hot_zone_s",
        "time_in_degraded_s",
        "nb_shutdowns_episode",
        "nb_degraded_episode",
        "ticks_since_last_shutdown",
        "has_fan_fault",
        "has_power_surge",
        "has_sensor_drift",
        "ticks_since_last_fault",
        "fan_mode_manual",
        "rpm_changes_last_60s",
        "is_recovering",
    ]
