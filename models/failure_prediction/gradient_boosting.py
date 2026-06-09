"""
gradient_boosting.py — Prédicteur Gradient Boosting (XGBoost ou LightGBM).

- Préférence XGBoost, fallback automatique sur LightGBM, puis sklearn GBM
- Early stopping sur le jeu de validation
- Seuil de décision optimisé sur Recall >= recall_target

Usage :
    from models.failure_prediction.gradient_boosting import GradientBoostingPredictor
    model = GradientBoostingPredictor()
    model.fit(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_test)
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

logger = logging.getLogger(__name__)

THRESHOLD_GRID = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

# Hyperparamètres par défaut (raisonnables sans tuning)
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,   # XGBoost / LightGBM
    "early_stopping_rounds": 30,
}


def _import_backend() -> tuple[str, type]:
    """Retourne (backend_name, ModelClass) selon ce qui est installé."""
    try:
        from xgboost import XGBClassifier
        return "xgboost", XGBClassifier
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        return "lightgbm", LGBMClassifier
    except ImportError:
        pass
    from sklearn.ensemble import GradientBoostingClassifier
    logger.warning(
        "XGBoost et LightGBM non installés — fallback sklearn GradientBoosting "
        "(pas d'early stopping, plus lent)."
    )
    return "sklearn", GradientBoostingClassifier


class GradientBoostingPredictor:
    """Gradient Boosting avec early stopping et optimisation du seuil.

    Parameters
    ----------
    threshold      : seuil de décision pour predict()
    recall_target  : Recall cible pour le choix du seuil (défaut 0.85)
    params         : dict d'hyperparamètres (surcharge les valeurs par défaut)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        recall_target: float = 0.85,
        params: dict | None = None,
    ) -> None:
        self.threshold = threshold
        self.recall_target = recall_target
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self._model = None
        self._backend: str = "unknown"
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
    ) -> "GradientBoostingPredictor":
        backend, ModelClass = _import_backend()
        self._backend = backend
        logger.info("Backend gradient boosting : %s", backend)

        X_tr = self._to_array(X_train)
        y_tr = y_train.values
        scale_pos_weight = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1.0)

        if backend == "xgboost":
            self._model = self._fit_xgboost(
                ModelClass, X_tr, y_tr, X_val, y_val, scale_pos_weight
            )
        elif backend == "lightgbm":
            self._model = self._fit_lightgbm(
                ModelClass, X_tr, y_tr, X_val, y_val, scale_pos_weight
            )
        else:
            self._model = self._fit_sklearn(ModelClass, X_tr, y_tr)

        # Feature importances
        if hasattr(self._model, "feature_importances_") and hasattr(X_train, "columns"):
            self.feature_importances_ = pd.Series(
                self._model.feature_importances_,
                index=X_train.columns,
            ).sort_values(ascending=False)
            logger.info("Top 5 features : %s", self.feature_importances_.head(5).to_dict())

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
        arr = self._to_array(X)
        if self._backend == "xgboost":
            import xgboost as xgb
            dmat = xgb.DMatrix(arr)
            p = self._model.predict(dmat)
            return np.column_stack([1 - p, p])
        proba = self._model.predict_proba(arr)
        return proba

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "threshold": self.threshold,
            "backend": self._backend,
            "feature_importances": self.feature_importances_,
        }, path)
        logger.info("Modèle sauvegardé : %s", path)

    def load(self, path: str) -> "GradientBoostingPredictor":
        data = joblib.load(path)
        self._model = data["model"]
        self.threshold = data["threshold"]
        self._backend = data.get("backend", "unknown")
        self.feature_importances_ = data.get("feature_importances")
        return self

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _fit_xgboost(self, ModelClass, X_tr, y_tr, X_val, y_val, scale_pos_weight):
        import xgboost as xgb
        p = self._params
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        eval_list = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(self._to_array(X_val), label=y_val.values)
            eval_list.append((dval, "val"))
        model = xgb.train(
            {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "learning_rate": p["learning_rate"],
                "max_depth": p["max_depth"],
                "subsample": p["subsample"],
                "colsample_bytree": p["colsample_bytree"],
                "scale_pos_weight": scale_pos_weight,
                "seed": 42,
                "verbosity": 0,
            },
            dtrain,
            num_boost_round=p["n_estimators"],
            evals=eval_list,
            early_stopping_rounds=p["early_stopping_rounds"] if len(eval_list) > 1 else None,
            verbose_eval=False,
        )
        logger.info("XGBoost best iteration : %d", model.best_iteration)
        return model

    def _fit_lightgbm(self, ModelClass, X_tr, y_tr, X_val, y_val, scale_pos_weight):
        p = self._params
        callbacks = []
        try:
            import lightgbm as lgb
            callbacks = [lgb.early_stopping(p["early_stopping_rounds"], verbose=False),
                         lgb.log_evaluation(period=-1)]
        except Exception:
            pass
        model = ModelClass(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        )
        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(self._to_array(X_val), y_val.values)]
            fit_kwargs["callbacks"] = callbacks
        model.fit(X_tr, y_tr, **fit_kwargs)
        return model

    def _fit_sklearn(self, ModelClass, X_tr, y_tr):
        p = self._params
        model = ModelClass(
            n_estimators=min(p["n_estimators"], 200),  # sklearn GBM est lent
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            subsample=p["subsample"],
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        return model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
