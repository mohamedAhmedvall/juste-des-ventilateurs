"""
baseline_threshold.py — Prédicteur heuristique à seuils.

Règle : failure = 1 si temperature_c > T_warn ET time_in_hot_zone_s > N_seconds.
Paramètres optimisés par grid search sur le jeu de validation.

Usage :
    from models.failure_prediction.baseline_threshold import ThresholdPredictor
    model = ThresholdPredictor()
    model.fit(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)
    proba  = model.predict_proba(X_test)
"""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

logger = logging.getLogger(__name__)

# Grille de recherche par défaut
DEFAULT_T_WARN_GRID = [60.0, 65.0, 70.0, 75.0, 78.0, 80.0, 83.0, 85.0]
DEFAULT_N_SECONDS_GRID = [0, 5, 10, 15, 20, 30]


class ThresholdPredictor:
    """Prédicteur à seuils — baseline heuristique.

    Parameters
    ----------
    t_warn_c       : seuil de température d'alerte (°C)
    n_seconds      : durée minimale en zone chaude (s) avant d'alerter
    recall_target  : Recall cible pour l'optimisation (défaut 0.85)
    """

    def __init__(
        self,
        t_warn_c: float = 78.0,
        n_seconds: float = 10.0,
        recall_target: float = 0.85,
    ) -> None:
        self.t_warn_c = t_warn_c
        self.n_seconds = n_seconds
        self.recall_target = recall_target
        self._fitted = False

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        t_warn_grid: list[float] | None = None,
        n_seconds_grid: list[float] | None = None,
    ) -> "ThresholdPredictor":
        """Grid search sur X_val/y_val, fallback sur X_train si pas de val."""
        X_opt = X_val if X_val is not None else X_train
        y_opt = y_val if y_val is not None else y_train

        t_grid = t_warn_grid or DEFAULT_T_WARN_GRID
        n_grid = n_seconds_grid or DEFAULT_N_SECONDS_GRID

        best_f1 = -1.0
        best_params = (self.t_warn_c, self.n_seconds)

        logger.info(
            "Grid search seuils : %d × %d = %d combinaisons",
            len(t_grid), len(n_grid), len(t_grid) * len(n_grid),
        )

        for t_warn, n_sec in itertools.product(t_grid, n_grid):
            y_pred = self._apply_rule(X_opt, t_warn, n_sec)
            rec = recall_score(y_opt, y_pred, zero_division=0)
            if rec >= self.recall_target:
                f1 = f1_score(y_opt, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (t_warn, n_sec)

        # Si aucune combinaison n'atteint le recall cible, prendre le meilleur recall
        if best_f1 < 0:
            logger.warning(
                "Aucune combinaison n'atteint recall >= %.2f, "
                "sélection du meilleur recall disponible.",
                self.recall_target,
            )
            best_rec = -1.0
            for t_warn, n_sec in itertools.product(t_grid, n_grid):
                y_pred = self._apply_rule(X_opt, t_warn, n_sec)
                rec = recall_score(y_opt, y_pred, zero_division=0)
                if rec > best_rec:
                    best_rec = rec
                    best_params = (t_warn, n_sec)

        self.t_warn_c, self.n_seconds = best_params
        self._fitted = True
        logger.info(
            "Meilleurs paramètres : T_warn=%.1f°C  N=%.0fs  (val F1=%.3f)",
            self.t_warn_c, self.n_seconds, best_f1 if best_f1 >= 0 else float("nan"),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._apply_rule(X, self.t_warn_c, self.n_seconds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Score continu ∈ [0, 1] basé sur la proximité du seuil."""
        temp = self._get_col(X, "temperature_c", default=0.0)
        hot  = self._get_col(X, "time_in_hot_zone_s", default=0.0)
        score_temp = np.clip(temp / max(self.t_warn_c, 1e-6), 0, 1)
        score_hot  = np.clip(hot  / max(self.n_seconds, 1.0), 0, 1) if self.n_seconds > 0 else np.ones(len(X))
        score = 0.7 * score_temp + 0.3 * score_hot
        return np.column_stack([1 - score, score])

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        params = {"t_warn_c": self.t_warn_c, "n_seconds": self.n_seconds,
                  "recall_target": self.recall_target}
        Path(path).write_text(json.dumps(params, indent=2))
        logger.info("Modèle sauvegardé : %s", path)

    def load(self, path: str) -> "ThresholdPredictor":
        params = json.loads(Path(path).read_text())
        self.t_warn_c = params["t_warn_c"]
        self.n_seconds = params["n_seconds"]
        self.recall_target = params.get("recall_target", 0.85)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _apply_rule(
        self, X: pd.DataFrame, t_warn: float, n_sec: float
    ) -> np.ndarray:
        temp = self._get_col(X, "temperature_c", default=0.0)
        hot  = self._get_col(X, "time_in_hot_zone_s", default=0.0)
        return ((temp > t_warn) & (hot > n_sec)).astype(int).values

    @staticmethod
    def _get_col(X: pd.DataFrame, col: str, default: float) -> np.ndarray:
        if col in X.columns:
            return X[col].fillna(default).values
        return np.full(len(X), default)
