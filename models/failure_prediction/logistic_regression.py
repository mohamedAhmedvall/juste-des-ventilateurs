"""
logistic_regression.py — Prédicteur par régression logistique.

- Normalisation StandardScaler
- Régularisation L2, C optimisé par validation
- Calibration Platt (CalibratedClassifierCV) pour des probabilités fiables

Usage :
    from models.failure_prediction.logistic_regression import LogisticPredictor
    model = LogisticPredictor()
    model.fit(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_test)
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
THRESHOLD_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


class LogisticPredictor:
    """Régression logistique avec calibration Platt.

    Parameters
    ----------
    C              : force de régularisation inverse (défaut optimisé sur val)
    threshold      : seuil de décision pour predict() (défaut 0.5)
    recall_target  : Recall cible pour le choix du seuil (défaut 0.85)
    max_iter       : nombre max d'itérations du solveur
    """

    def __init__(
        self,
        C: float = 1.0,
        threshold: float = 0.5,
        recall_target: float = 0.85,
        max_iter: int = 1000,
    ) -> None:
        self.C = C
        self.threshold = threshold
        self.recall_target = recall_target
        self.max_iter = max_iter
        self._pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LogisticPredictor":
        X_tr = self._to_array(X_train)
        y_tr = y_train.values

        # Sélection de C sur la validation
        if X_val is not None and y_val is not None:
            best_C, best_f1 = self._select_C(X_tr, y_tr, X_val, y_val)
            self.C = best_C
            logger.info("Meilleur C=%.4f (val F1=%.3f)", self.C, best_f1)

        # Entraînement final avec calibration
        self._pipeline = self._build_pipeline(self.C)
        self._pipeline.fit(X_tr, y_tr)

        # Optimisation du seuil sur val
        if X_val is not None and y_val is not None:
            self.threshold = self._select_threshold(X_val, y_val)
            logger.info("Seuil optimal : %.2f", self.threshold)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Modèle non entraîné — appeler fit() d'abord.")
        return self._pipeline.predict_proba(self._to_array(X))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "threshold": self.threshold,
                     "C": self.C}, path)
        logger.info("Modèle sauvegardé : %s", path)

    def load(self, path: str) -> "LogisticPredictor":
        data = joblib.load(path)
        if isinstance(data, dict):
            pipeline_val = (data.get("pipeline") or data.get("model") or data.get("estimator")
                            or next((v for v in data.values() if hasattr(v, "predict_proba")), None))
            if pipeline_val is None:
                raise KeyError(f"Aucun pipeline sklearn dans le joblib. Cles: {list(data.keys())}")
            self._pipeline = pipeline_val
            self.threshold = data.get("threshold", self.threshold)
            self.C         = data.get("C", self.C)
        else:
            self._pipeline = data
        return self

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _build_pipeline(self, C: float) -> Pipeline:
        lr = LogisticRegression(
            C=C, penalty="l2", solver="lbfgs",
            max_iter=self.max_iter, class_weight="balanced",
        )
        calibrated = CalibratedClassifierCV(lr, method="sigmoid", cv=3)
        return Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])

    def _select_C(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[float, float]:
        X_v = self._to_array(X_val)
        y_v = y_val.values
        best_C, best_f1 = C_GRID[0], -1.0
        for c in C_GRID:
            pipe = self._build_pipeline(c)
            pipe.fit(X_tr, y_tr)
            y_pred = (pipe.predict_proba(X_v)[:, 1] >= 0.5).astype(int)
            rec = recall_score(y_v, y_pred, zero_division=0)
            if rec >= self.recall_target:
                f1 = f1_score(y_v, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_C = f1, c
        if best_f1 < 0:
            # Fallback : meilleur F1 sans contrainte recall
            for c in C_GRID:
                pipe = self._build_pipeline(c)
                pipe.fit(X_tr, y_tr)
                y_pred = (pipe.predict_proba(X_v)[:, 1] >= 0.5).astype(int)
                f1 = f1_score(y_v, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_C = f1, c
        return best_C, best_f1

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
