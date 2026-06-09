"""
random_forest.py — Prédicteur Random Forest.

- class_weight="balanced" pour gérer le déséquilibre de classes
- Seuil de décision optimisé sur Recall >= recall_target
- Feature importance loggée

Usage :
    from models.failure_prediction.random_forest import RandomForestPredictor
    model = RandomForestPredictor()
    model.fit(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_test)
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

logger = logging.getLogger(__name__)

# Grille légère pour limiter le temps d'entraînement
PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 100, "max_depth": 15},
    {"n_estimators": 200, "max_depth": 15},
    {"n_estimators": 200, "max_depth": 20},
]
THRESHOLD_GRID = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]


class RandomForestPredictor:
    """Random Forest avec optimisation du seuil sur Recall cible.

    Parameters
    ----------
    n_estimators   : nombre d'arbres
    max_depth      : profondeur max (None = illimitée)
    threshold      : seuil de décision pour predict()
    recall_target  : Recall cible pour le choix du seuil (défaut 0.85)
    n_jobs         : parallélisme (-1 = tous les cœurs)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 15,
        threshold: float = 0.5,
        recall_target: float = 0.85,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.threshold = threshold
        self.recall_target = recall_target
        self.n_jobs = n_jobs
        self._model: RandomForestClassifier | None = None
        self.feature_importances_: pd.Series | None = None

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "RandomForestPredictor":
        X_tr = self._to_array(X_train)
        y_tr = y_train.values

        # Sélection des hyperparamètres sur val
        if X_val is not None and y_val is not None:
            best_params = self._select_params(X_tr, y_tr, X_val, y_val)
            self.n_estimators = best_params["n_estimators"]
            self.max_depth = best_params["max_depth"]
            logger.info(
                "Meilleurs params : n_estimators=%d  max_depth=%s",
                self.n_estimators, self.max_depth,
            )

        # Entraînement final
        self._model = self._build(self.n_estimators, self.max_depth)
        self._model.fit(X_tr, y_tr)

        # Feature importances
        if hasattr(X_train, "columns"):
            self.feature_importances_ = pd.Series(
                self._model.feature_importances_,
                index=X_train.columns,
            ).sort_values(ascending=False)
            top5 = self.feature_importances_.head(5)
            logger.info("Top 5 features : %s", top5.to_dict())

        # Seuil optimal
        if X_val is not None and y_val is not None:
            self.threshold = self._select_threshold(X_val, y_val)
            logger.info("Seuil optimal : %.2f", self.threshold)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Modèle non entraîné — appeler fit() d'abord.")
        return self._model.predict_proba(self._to_array(X))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "threshold": self.threshold,
            "feature_importances": self.feature_importances_,
        }, path)
        logger.info("Modèle sauvegardé : %s", path)

    def load(self, path: str) -> "RandomForestPredictor":
        data = joblib.load(path)
        self._model = data["model"]
        self.threshold = data["threshold"]
        self.feature_importances_ = data.get("feature_importances")
        return self

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _build(self, n_estimators: int, max_depth: int | None) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=self.n_jobs,
            random_state=42,
        )

    def _select_params(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        X_v = self._to_array(X_val)
        y_v = y_val.values
        best_params = PARAM_GRID[0]
        best_f1 = -1.0
        for params in PARAM_GRID:
            m = self._build(params["n_estimators"], params["max_depth"])
            m.fit(X_tr, y_tr)
            y_pred = (m.predict_proba(X_v)[:, 1] >= 0.5).astype(int)
            rec = recall_score(y_v, y_pred, zero_division=0)
            if rec >= self.recall_target:
                f1 = f1_score(y_v, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_params = f1, params
        return best_params

    def _select_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        proba = self.predict_proba(X_val)[:, 1]
        y_v = y_val.values
        best_thr, best_f1 = 0.5, -1.0
        for thr in THRESHOLD_GRID:
            y_pred = (proba >= thr).astype(int)
            rec = recall_score(y_v, y_pred, zero_division=0)
            if rec >= self.recall_target:
                f1 = f1_score(y_v, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
        return best_thr if best_f1 >= 0 else 0.5

    @staticmethod
    def _to_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.fillna(0).values
        return X
