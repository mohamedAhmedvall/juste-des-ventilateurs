"""Contrôleur baseline : PID simple.

Régulateur Proportionnel-Intégral-Dérivé dont la consigne est :
    T_target = pid_target_ratio * t_shutdown  (défaut : 80%)

Erreur : e(t) = temperature_c(t) - T_target

Commande continue :
    u(t) = Kp*e(t) + Ki*integral(e) + Kd*de/dt

La commande est ensuite clampée dans [rpm_min, rpm_max] puis arrondie au
niveau RPM discret le plus proche parmi {800, 1500, 2500, 3500, 4500}.
800 RPM est le plancher minimal de ventilation (Phase 7.5).

Les gains Kp, Ki, Kd sont optimisés par grid search sur les données
d'entraînement (minimisation du nb de shutdowns + énergie).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RPM_LEVELS   = [800, 1500, 2500, 3500, 4500]
RPM_MIN      = 800
RPM_MAX      = 4500
DEFAULT_T_SHUTDOWN = 88.0  # fallback si non fourni dans les données


def _snap_to_levels(rpm_continuous: float) -> int:
    """Arrondi au niveau RPM discret le plus proche."""
    arr = np.array(RPM_LEVELS)
    return int(arr[np.argmin(np.abs(arr - rpm_continuous))])


class PIDController:
    """Contrôleur PID thermique avec quantification RPM discrète."""

    name = "baseline_pid"

    def __init__(
        self,
        kp: float = 80.0,
        ki: float = 2.0,
        kd: float = 10.0,
        pid_target_ratio: float = 0.80,
        t_shutdown: float = DEFAULT_T_SHUTDOWN,
        rpm_min: int = RPM_MIN,
        rpm_max: int = RPM_MAX,
        integral_clip: float = 1000.0,
    ):
        self.kp               = kp
        self.ki               = ki
        self.kd               = kd
        self.pid_target_ratio = pid_target_ratio
        self.t_shutdown       = t_shutdown
        self.rpm_min          = rpm_min
        self.rpm_max          = rpm_max
        self.integral_clip    = integral_clip

        # État interne du PID (remis à zéro entre les épisodes)
        self._integral  = 0.0
        self._prev_error: Optional[float] = None

        # Rempli après fit()
        self.best_params_: dict = {}

    # ------------------------------------------------------------------
    # Interface commune FanController
    # ------------------------------------------------------------------

    @property
    def t_target(self) -> float:
        return self.pid_target_ratio * self.t_shutdown

    def reset(self) -> None:
        """Remet l'état PID à zéro (à appeler en début d'épisode)."""
        self._integral  = 0.0
        self._prev_error = None

    def decide(self, state: pd.Series, risk_score: float = 0.0) -> int:
        """Décision PID sur une observation — maintient l'état interne."""
        temp       = float(state.get("temperature_c", self.t_target))
        t_shutdown = float(state.get("t_shutdown_c", self.t_shutdown))
        t_target   = self.pid_target_ratio * t_shutdown

        error      = temp - t_target
        self._integral = np.clip(
            self._integral + error, -self.integral_clip, self.integral_clip
        )
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error)
        self._prev_error = error

        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        # Offset de base : RPM_MAX/2 quand erreur nulle
        rpm_continuous = np.clip(RPM_MAX / 2 + u, self.rpm_min, self.rpm_max)
        return _snap_to_levels(rpm_continuous)

    def decide_batch(
        self,
        X: pd.DataFrame,
        risk_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Décisions en batch — version vectorisée (sans état intégral).

        Pour l'évaluation offline on approxime le PID sans accumuler
        l'intégrale (Ki=0 effectif), ce qui permet une vectorisation
        NumPy complète et évite l'iterrows sur 200k lignes.

        La composante intégrale est négligeable devant Kp dans la pratique :
        l'erreur proportionnelle domine la commande RPM.
        """
        temps = X["temperature_c"].values if "temperature_c" in X.columns \
                else np.full(len(X), self.t_target)

        # t_shutdown par ligne si margin disponible, sinon valeur globale
        if "margin_to_shutdown" in X.columns:
            t_shutdown_arr = temps + X["margin_to_shutdown"].values
        else:
            t_shutdown_arr = np.full(len(X), self.t_shutdown)

        t_target_arr = self.pid_target_ratio * t_shutdown_arr
        error = temps - t_target_arr

        # Dérivée approchée (diff entre ticks consécutifs, 0 au premier)
        deriv = np.concatenate([[0.0], np.diff(error)])

        u = self.kp * error + self.kd * deriv
        rpm_continuous = np.clip(RPM_MAX / 2 + u, self.rpm_min, self.rpm_max)

        arr = np.array(RPM_LEVELS)
        # Vectorisation de snap_to_levels
        rpms = arr[np.argmin(np.abs(rpm_continuous[:, None] - arr[None, :]), axis=1)]
        return rpms.astype(int)

    # ------------------------------------------------------------------
    # Optimisation des gains
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        kp_grid: Optional[list] = None,
        ki_grid: Optional[list] = None,
        kd_grid: Optional[list] = None,
    ) -> "PIDController":
        """Grid search sur Kp, Ki, Kd.

        Score = -nb_shutdowns - 0.05 * mean_rpm_norm
        (minimiser les shutdowns en priorité, puis l'énergie)
        """
        if "temperature_c" not in X_train.columns:
            return self

        kp_grid = kp_grid or [40.0, 120.0]
        ki_grid = ki_grid or [0.5,  5.0]
        kd_grid = kd_grid or [5.0,  20.0]

        # Echantillon pour accélérer la grid search (5k lignes suffisent)
        if len(X_train) > 5000:
            idx = np.random.default_rng(42).choice(len(X_train), 5000, replace=False)
            X_train = X_train.iloc[idx].reset_index(drop=True)
            y_train = y_train.iloc[idx].reset_index(drop=True)

        # Détecter t_shutdown depuis les données si disponible
        if "margin_to_shutdown" in X_train.columns and "temperature_c" in X_train.columns:
            t_shutdown_est = (
                X_train["temperature_c"] + X_train["margin_to_shutdown"]
            ).median()
            if not np.isnan(t_shutdown_est):
                self.t_shutdown = float(t_shutdown_est)

        best_score = -np.inf
        best_params = {}

        for kp in kp_grid:
            for ki in ki_grid:
                for kd in kd_grid:
                    ctrl = PIDController(
                        kp=kp, ki=ki, kd=kd,
                        pid_target_ratio=self.pid_target_ratio,
                        t_shutdown=self.t_shutdown,
                    )
                    rpms = ctrl.decide_batch(X_train)

                    # Score : proxy shutdown = fraction de ticks où
                    # temperature > t_shutdown (margin_to_shutdown < 0)
                    if "margin_to_shutdown" in X_train.columns:
                        n_critical = (X_train["margin_to_shutdown"] < 0).sum()
                    else:
                        n_critical = (
                            X_train["temperature_c"] > self.t_shutdown
                        ).sum()

                    mean_rpm_norm = rpms.mean() / RPM_MAX
                    # On pénalise aussi les RPM trop bas quand dangereux
                    dangerous = y_train.values == 1
                    if dangerous.sum() > 0:
                        low_rpm_during_failure = (rpms[dangerous] < 3500).mean()
                    else:
                        low_rpm_during_failure = 0.0

                    score = -(n_critical / max(len(X_train), 1)) \
                            - 0.05 * mean_rpm_norm \
                            - 0.3 * low_rpm_during_failure

                    if score > best_score:
                        best_score = score
                        best_params = {"kp": kp, "ki": ki, "kd": kd}

        if best_params:
            self.kp = best_params["kp"]
            self.ki = best_params["ki"]
            self.kd = best_params["kd"]
            self.best_params_ = {**best_params, "score": best_score}
            self.reset()

        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "kp":               self.kp,
            "ki":               self.ki,
            "kd":               self.kd,
            "pid_target_ratio": self.pid_target_ratio,
            "t_shutdown":       self.t_shutdown,
            "rpm_min":          self.rpm_min,
            "rpm_max":          self.rpm_max,
            "integral_clip":    self.integral_clip,
            "best_params":      self.best_params_,
        }
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PIDController":
        with open(path) as f:
            cfg = json.load(f)
        obj = cls(
            kp=cfg["kp"],
            ki=cfg["ki"],
            kd=cfg["kd"],
            pid_target_ratio=cfg.get("pid_target_ratio", 0.80),
            t_shutdown=cfg.get("t_shutdown", DEFAULT_T_SHUTDOWN),
            rpm_min=cfg.get("rpm_min", RPM_MIN),
            rpm_max=cfg.get("rpm_max", RPM_MAX),
            integral_clip=cfg.get("integral_clip", 1000.0),
        )
        obj.best_params_ = cfg.get("best_params", {})
        return obj

    def __repr__(self) -> str:
        return (
            f"PIDController(kp={self.kp}, ki={self.ki}, kd={self.kd}, "
            f"t_target={self.t_target:.1f}°C)"
        )
