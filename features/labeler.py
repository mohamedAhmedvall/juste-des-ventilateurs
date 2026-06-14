"""Labeler — Juste des Ventilateurs.

Génère les labels supervisés pour :
  - Le modèle de prédiction de pannes (failure_60s, failure_30s, hot_30s)
  - Le contrôleur supervisé (action_class, action_class_v2)

Les labels sont calculés en regardant "vers l'avenir" (forward-looking) :
à chaque pas de temps t, on regarde ce qui se passe dans les N secondes suivantes.

Convention des labels de panne :
  - 1 = incident prévu dans l'horizon
  - 0 = pas d'incident prévu

Convention des labels de contrôle :
  - action_class   : oracle v1 — basé sur temperature_c instantanée uniquement (myope)
  - action_class_v2: oracle v2 (Phase 8) — trajectoire thermique + urgence panne
    ∈ {0, 1, 2, 3, 4} correspondant à RPM ∈ {800, 1500, 2500, 3500, 4500}

Usage typique :
    df = add_failure_labels(df, t_shutdown_c=88.0, tick_hz=1.0)
    df = add_control_labels(df, t_shutdown_c=88.0)           # oracle v1
    df = add_control_labels_v2(df, t_shutdown_c=88.0)        # oracle v2 (Phase 8)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Niveaux RPM discrets pour le contrôleur (doit correspondre aux specs)
RPM_LEVELS = [800, 1500, 2500, 3500, 4500]  # plancher 800 RPM (ventilation minimale)

# Seuil "trop chaud" pour le label hot_30s : 95% du shutdown
_HOT_RATIO = 0.95

# Seuil "zone sûre" pour l'oracle du contrôleur : 85% du shutdown
_SAFE_RATIO = 0.85


def add_failure_labels(
    df: pd.DataFrame,
    t_shutdown_c: float = 88.0,
    tick_hz: float = 1.0,
    horizons_s: tuple[int, ...] = (30, 60),
) -> pd.DataFrame:
    """Ajoute les labels de prédiction de pannes.

    Parameters
    ----------
    df          : DataFrame d'une machine, trié par timestamp
    t_shutdown_c: seuil de shutdown thermique (°C)
    tick_hz     : fréquence de publication
    horizons_s  : horizons de prédiction en secondes (défaut: 30s et 60s)

    Colonnes ajoutées
    -----------------
    failure_{H}s : 1 si status=degraded ou off(overheat) dans les H secondes suivantes
    hot_{H}s     : 1 si température > 0.95*t_shutdown dans les H secondes suivantes
                   (uniquement pour H=30)
    """
    df = df.copy()
    n = len(df)
    hot_threshold = t_shutdown_c * _HOT_RATIO

    # Statuts dangereux : degraded ou off causé par surchauffe
    # On identifie les indices où une panne est imminente
    is_dangerous = _is_dangerous_status(df, t_shutdown_c=t_shutdown_c)
    is_hot = (df["temperature_c"] > hot_threshold).values if "temperature_c" in df.columns else np.zeros(n, dtype=bool)

    for horizon_s in horizons_s:
        horizon_rows = max(1, int(horizon_s * tick_hz))
        label_failure = np.zeros(n, dtype=int)
        label_hot = np.zeros(n, dtype=int)

        for i in range(n):
            look_ahead = slice(i + 1, min(i + 1 + horizon_rows, n))
            if is_dangerous[look_ahead].any():
                label_failure[i] = 1
            if is_hot[look_ahead].any():
                label_hot[i] = 1

        df[f"failure_{horizon_s}s"] = label_failure
        if horizon_s == 30:
            df[f"hot_{horizon_s}s"] = label_hot

    # ------------------------------------------------------------------
    # Label auxiliaire : temps avant le prochain incident (en secondes)
    # Utile pour l'analyse du lead time du prédicteur
    # ------------------------------------------------------------------
    time_to_failure = np.full(n, np.nan)
    next_failure_idx = n  # pas d'incident par défaut
    for i in range(n - 1, -1, -1):
        if is_dangerous[i]:
            next_failure_idx = i
        if next_failure_idx < n:
            time_to_failure[i] = (next_failure_idx - i) / tick_hz
    df["time_to_failure_s"] = time_to_failure

    return df


def add_control_labels(
    df: pd.DataFrame,
    t_shutdown_c: float = 88.0,
    rpm_levels: list[int] | None = None,
) -> pd.DataFrame:
    """Ajoute les labels pour l'apprentissage supervisé du contrôleur.

    L'oracle détermine la consigne RPM minimale qui aurait suffi à maintenir
    la température en dessous du seuil sûr (85% du shutdown).

    En pratique, on utilise une heuristique basée sur la marge thermique :
    plus la marge est faible, plus le RPM cible est élevé.

    Parameters
    ----------
    df          : DataFrame d'une machine avec features temporelles
    t_shutdown_c: seuil de shutdown thermique (°C)
    rpm_levels  : niveaux RPM discrets (défaut: [0, 1500, 2500, 3500, 4500])

    Colonnes ajoutées
    -----------------
    optimal_rpm   : RPM cible selon l'oracle
    action_class  : index dans rpm_levels (0..4)
    """
    df = df.copy()
    levels = rpm_levels if rpm_levels is not None else RPM_LEVELS
    n_levels = len(levels)
    safe_threshold = t_shutdown_c * _SAFE_RATIO

    if "temperature_c" not in df.columns:
        logger.warning("'temperature_c' manquant — labels contrôle ignorés.")
        return df

    # Oracle : mapping marge thermique → classe RPM
    # Principe : diviser l'espace [safe_threshold, t_shutdown] en n_levels zones
    # Plus on approche du shutdown, plus le RPM est élevé
    temp = df["temperature_c"].values
    margin = t_shutdown_c - temp

    # Zone froide (T << safe) → RPM faible ; zone chaude (T → shutdown) → RPM max
    # Normalisation de la température dans [0, 1] par rapport au seuil
    temp_ratio = np.clip((temp - (t_shutdown_c * 0.5)) / (t_shutdown_c * 0.5), 0.0, 1.0)

    # Classe RPM = floor(temp_ratio × n_levels), clampé dans [0, n_levels-1]
    action_class = np.floor(temp_ratio * n_levels).astype(int).clip(0, n_levels - 1)
    optimal_rpm = np.array(levels)[action_class]

    # Forcer RPM max si en zone dégradée ou très proche du shutdown
    if "status" in df.columns:
        critical = (df["status"].isin(["degraded"])).values | (margin < (t_shutdown_c * 0.05))
        action_class[critical] = n_levels - 1
        optimal_rpm[critical] = levels[-1]

    df["optimal_rpm"] = optimal_rpm
    df["action_class"] = action_class

    return df


def add_control_labels_v2(
    df: pd.DataFrame,
    t_shutdown_c: float = 88.0,
    rpm_levels: list[int] | None = None,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta_max_c: float = 5.0,
    horizon_s: float = 60.0,
) -> pd.DataFrame:
    """Oracle trajectoire enrichi — Phase 8.

    Corrige la myopie de add_control_labels() en intégrant :
      - la position thermique (comme v1)
      - la vitesse de montée/descente thermique (temp_delta_30s)
      - l'urgence panne (time_to_failure_s)

    score(t) = alpha * temp_ratio
             + beta  * clip(temp_delta_30s / delta_max_c, 0, 1)
             + gamma * urgency

    action_class_v2 = floor(score * n_levels).clip(0, n_levels-1)

    Parameters
    ----------
    df          : DataFrame d'une machine avec features temporelles
    t_shutdown_c: seuil de shutdown thermique (degC)
    rpm_levels  : niveaux RPM discrets (defaut: RPM_LEVELS)
    alpha       : poids position thermique (defaut: 0.5)
    beta        : poids vitesse thermique (defaut: 0.3)
    gamma       : poids urgence panne (defaut: 0.2)
    delta_max_c : normalisation de temp_delta_30s en degC (defaut: 5.0)
    horizon_s   : horizon d'urgence en secondes (defaut: 60.0)

    Colonnes ajoutees
    -----------------
    action_class_v2 : index dans rpm_levels (0..n_levels-1)
    """
    df = df.copy()
    levels = rpm_levels if rpm_levels is not None else RPM_LEVELS
    n_levels = len(levels)

    if "temperature_c" not in df.columns:
        logger.warning("'temperature_c' manquant — labels controle v2 ignores.")
        return df

    temp = df["temperature_c"].values

    # --- Composante 1 : position thermique (identique oracle v1) ---
    temp_ratio = np.clip(
        (temp - (t_shutdown_c * 0.5)) / (t_shutdown_c * 0.5), 0.0, 1.0
    )

    # --- Composante 2 : vitesse thermique ---
    # Utilise temp_delta_30s si disponible, sinon temp_delta_5s, sinon 0
    if "temp_delta_30s" in df.columns:
        delta = df["temp_delta_30s"].fillna(0.0).values
    elif "temp_delta_5s" in df.columns:
        delta = df["temp_delta_5s"].fillna(0.0).values
        # Normaliser sur l'horizon equivalent (5s vs 30s)
        delta = delta * (30.0 / 5.0)
    else:
        logger.debug("Aucune feature temp_delta_* disponible — composante vitesse ignoree.")
        delta = np.zeros(len(df))

    # Clip [0, 1] : on penalise la montee, pas la descente (beta=0 si descente)
    velocity_score = np.clip(delta / max(delta_max_c, 1e-6), 0.0, 1.0)

    # --- Composante 3 : urgence panne ---
    if "time_to_failure_s" in df.columns:
        ttf = df["time_to_failure_s"].values
        # NaN = aucune panne prevue -> urgence 0
        urgency = np.where(
            np.isfinite(ttf),
            np.clip(1.0 - ttf / max(horizon_s, 1.0), 0.0, 1.0),
            0.0,
        )
    else:
        logger.debug("'time_to_failure_s' absent — composante urgence ignoree.")
        urgency = np.zeros(len(df))

    # --- Score composite ---
    score = alpha * temp_ratio + beta * velocity_score + gamma * urgency
    # score est dans [0, alpha+beta+gamma] = [0, 1.0] avec les defauts
    # On le norme pour garantir que score=1.0 -> classe max
    score_norm = np.clip(score, 0.0, 1.0)

    action_class_v2 = np.floor(score_norm * n_levels).astype(int).clip(0, n_levels - 1)

    # Forcer RPM max si en zone degradee, tres proche du shutdown, ou panne imminente
    critical = np.zeros(len(df), dtype=bool)
    if "status" in df.columns:
        margin = t_shutdown_c - temp
        critical |= (df["status"].isin(["degraded"])).values | (margin < (t_shutdown_c * 0.05))
    # Panne dans moins de 1/3 de l'horizon -> RPM_HIGH obligatoire
    if "time_to_failure_s" in df.columns:
        ttf = df["time_to_failure_s"].values
        critical |= np.isfinite(ttf) & (ttf < (horizon_s / 3.0))
    action_class_v2[critical] = n_levels - 1

    df["action_class_v2"] = action_class_v2

    return df


def label_names_failure() -> list[str]:
    """Labels de prédiction de pannes produits par add_failure_labels."""
    return ["failure_60s", "failure_30s", "hot_30s", "time_to_failure_s"]


def label_names_control() -> list[str]:
    """Labels de contrôle produits par add_control_labels (oracle v1)."""
    return ["optimal_rpm", "action_class"]


def label_names_control_v2() -> list[str]:
    """Labels de contrôle produits par add_control_labels_v2 (oracle v2, Phase 8)."""
    return ["action_class_v2"]


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _is_dangerous_status(
    df: pd.DataFrame,
    t_shutdown_c: float = 88.0,
    ticks_since_shutdown_col: str = "ticks_since_last_shutdown",
    hot_off_min_ticks: int = 60,
) -> np.ndarray:
    """Retourne un tableau bool : True si le statut est dangereux (dégradé ou shutdown thermique).

    Un row "off" n'est considéré dangereux que si :
      - temperature_c > 50% du seuil de shutdown (arrêt par surchauffe), OU
      - ticks_since_last_shutdown < hot_off_min_ticks (machine vient juste de s'éteindre,
        température encore potentiellement élevée)

    Ceci évite de labeler comme dangereux les rows "off" de longue durée à froid
    (machines éteintes volontairement ou déjà refroidies).
    """
    n = len(df)
    if "status" not in df.columns:
        return np.zeros(n, dtype=bool)

    statuses = df["status"].values
    dangerous = np.zeros(n, dtype=bool)

    # Seuil temperature : 50% du shutdown
    hot_off_threshold = t_shutdown_c * 0.5
    temps = df["temperature_c"].values if "temperature_c" in df.columns else np.zeros(n)
    ticks_since = (
        df[ticks_since_shutdown_col].values
        if ticks_since_shutdown_col in df.columns
        else np.full(n, hot_off_min_ticks + 1, dtype=int)
    )

    for i, s in enumerate(statuses):
        if s == "degraded":
            dangerous[i] = True
        elif s == "off":
            # "off" dangereux seulement si surchauffe probable
            if "status_cause" in df.columns:
                cause = df["status_cause"].iloc[i]
                if pd.notna(cause) and "overheat" in str(cause):
                    dangerous[i] = True
            # Heuristique : T > seuil basse OU shutdown recent
            if temps[i] > hot_off_threshold or int(ticks_since[i]) < hot_off_min_ticks:
                dangerous[i] = True

    return dangerous
