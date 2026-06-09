"""
failure_prediction_eval.py — Évaluation comparative des modèles de prédiction.

Métriques calculées :
  - Precision, Recall, F1, PR-AUC, ROC-AUC
  - Lead time moyen : temps entre la première alerte et l'incident (secondes)
  - Taux de faux négatifs sur shutdowns (cas les plus critiques)

Usage :
    python -m evaluation.failure_prediction_eval \
        --label failure_60s \
        --models baseline threshold logistic random_forest gradient_boosting \
        --output evaluation/results/failure_prediction_results.json
"""
from __future__ import annotations

from evaluation import _compat  # noqa: F401 — force UTF-8 stdout Windows

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.failure_prediction.splitter import TemporalSplitter

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("evaluation/results")
MODELS_DIR  = Path("models/failure_prediction/saved")


# ---------------------------------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------------------------------

def load_model(name: str):
    """Charge et retourne un modèle par son nom court."""
    if name in ("baseline", "threshold"):
        from models.failure_prediction.baseline_threshold import ThresholdPredictor
        return ThresholdPredictor(), name

    if name == "logistic":
        from models.failure_prediction.logistic_regression import LogisticPredictor
        return LogisticPredictor(), name

    if name == "random_forest":
        from models.failure_prediction.random_forest import RandomForestPredictor
        return RandomForestPredictor(), name

    if name == "gradient_boosting":
        from models.failure_prediction.gradient_boosting import GradientBoostingPredictor
        return GradientBoostingPredictor(), name

    raise ValueError(f"Modèle inconnu : '{name}'. "
                     f"Valeurs : baseline, logistic, random_forest, gradient_boosting")


# ---------------------------------------------------------------------------
# Calcul du lead time
# ---------------------------------------------------------------------------

def compute_lead_time(
    df_test_meta: pd.DataFrame,
    y_pred: np.ndarray,
    label_col: str,
    tick_hz: float = 1.0,
) -> dict:
    """Calcule le lead time moyen (temps entre première alerte et incident).

    Returns dict avec : mean_s, median_s, min_s, max_s, n_incidents, n_detected
    """
    if "timestamp" not in df_test_meta.columns or "machine_id" not in df_test_meta.columns:
        return {"mean_s": None, "median_s": None, "n_incidents": 0, "n_detected": 0}

    df = df_test_meta.copy()
    df["_pred"] = y_pred
    df["_label"] = df[label_col].fillna(0).astype(int)

    lead_times = []
    for machine_id, grp in df.groupby("machine_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        # Repérer les débuts d'incidents (label passe de 0 à 1)
        label_series = grp["_label"].values
        pred_series  = grp["_pred"].values
        ts_series    = grp["timestamp"].values

        incident_starts = np.where(
            (label_series[1:] == 1) & (label_series[:-1] == 0)
        )[0] + 1

        for idx in incident_starts:
            # Chercher la première alerte avant cet incident (dans une fenêtre de 120s)
            window_start = max(0, idx - int(120 * tick_hz))
            alerts_before = np.where(pred_series[window_start:idx] == 1)[0]
            if len(alerts_before) > 0:
                first_alert_idx = window_start + alerts_before[0]
                ts_alert   = pd.Timestamp(ts_series[first_alert_idx])
                ts_incident = pd.Timestamp(ts_series[idx])
                lead_s = (ts_incident - ts_alert).total_seconds()
                lead_times.append(lead_s)

    n_incidents = sum(
        ((grp["_label"].values[1:] == 1) & (grp["_label"].values[:-1] == 0)).sum()
        for _, grp in df.groupby("machine_id")
    )

    if not lead_times:
        return {
            "mean_s": 0.0, "median_s": 0.0, "min_s": 0.0, "max_s": 0.0,
            "n_incidents": int(n_incidents), "n_detected": 0,
        }

    return {
        "mean_s":    float(np.mean(lead_times)),
        "median_s":  float(np.median(lead_times)),
        "min_s":     float(np.min(lead_times)),
        "max_s":     float(np.max(lead_times)),
        "n_incidents": int(n_incidents),
        "n_detected":  len(lead_times),
    }


# ---------------------------------------------------------------------------
# Évaluation d'un modèle
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    label_col: str,
    df_test_meta: pd.DataFrame,
) -> dict:
    """Entraîne, prédit et calcule toutes les métriques pour un modèle."""
    logger.info("=== Évaluation : %s ===", model_name)

    # -- Entraînement
    if hasattr(model, "fit"):
        kwargs: dict = {}
        import inspect
        sig = inspect.signature(model.fit)
        if "X_val" in sig.parameters:
            kwargs = {"X_val": X_val, "y_val": y_val}
        model.fit(X_train, y_train, **kwargs)

    # -- Prédictions
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # -- Métriques standard
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    try:
        pr_auc  = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        pr_auc = roc_auc = float("nan")

    # -- Taux de faux négatifs sur shutdowns
    fn_rate = float("nan")
    if "nb_shutdowns_episode" in df_test_meta.columns:
        shutdown_mask = df_test_meta["nb_shutdowns_episode"] > 0
        if shutdown_mask.sum() > 0:
            y_test_shut = y_test[shutdown_mask]
            y_pred_shut = y_pred[shutdown_mask]
            fn = ((y_test_shut == 1) & (y_pred_shut == 0)).sum()
            fn_rate = fn / max(len(y_test_shut), 1)

    # -- Lead time
    lead_time = compute_lead_time(df_test_meta, y_pred, label_col)

    result = {
        "model":     model_name,
        "label":     label_col,
        "n_train":   len(X_train),
        "n_val":     len(X_val),
        "n_test":    len(X_test),
        "pos_rate_test": float((y_test == 1).mean()),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "pr_auc":    round(pr_auc, 4) if not np.isnan(pr_auc) else None,
        "roc_auc":   round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "fn_rate_shutdown": round(fn_rate, 4) if not np.isnan(fn_rate) else None,
        "lead_time": lead_time,
    }

    # -- Sauvegarde du modèle
    save_path = MODELS_DIR / f"{model_name}_{label_col}.joblib"
    if hasattr(model, "save"):
        model.save(str(save_path))
    else:
        import joblib
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)

    logger.info(
        "%s — Precision=%.3f  Recall=%.3f  F1=%.3f  PR-AUC=%s  "
        "LeadTime=%.1fs  (%d/%d incidents détectés)",
        model_name, precision, recall, f1,
        f"{pr_auc:.3f}" if not np.isnan(pr_auc) else "N/A",
        lead_time["mean_s"] or 0,
        lead_time["n_detected"], lead_time["n_incidents"],
    )

    return result


# ---------------------------------------------------------------------------
# Affichage du tableau comparatif
# ---------------------------------------------------------------------------

def print_comparison(results: list[dict]) -> None:
    print()
    print("=" * 90)
    print("  COMPARAISON DES MODÈLES")
    print("=" * 90)
    header = (f"  {'Modèle':<22} {'Precision':>9} {'Recall':>7} {'F1':>7} "
              f"{'PR-AUC':>8} {'ROC-AUC':>8} {'LeadTime':>10} {'Détectés':>10}")
    print(header)
    print("-" * 90)
    for r in sorted(results, key=lambda x: x.get("f1", 0), reverse=True):
        lt = r["lead_time"]
        lt_s = f"{lt['mean_s']:.1f}s" if lt.get("mean_s") is not None else "N/A"
        det  = f"{lt['n_detected']}/{lt['n_incidents']}" if lt else "N/A"
        print(
            f"  {r['model']:<22} {r['precision']:>9.3f} {r['recall']:>7.3f} "
            f"{r['f1']:>7.3f} {r['pr_auc'] or 0:>8.3f} {r['roc_auc'] or 0:>8.3f} "
            f"{lt_s:>10} {det:>10}"
        )
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Évaluation comparative des modèles de prédiction de pannes"
    )
    parser.add_argument(
        "--label", default="failure_60s",
        choices=["failure_60s", "failure_30s", "hot_30s"],
        help="Label cible (défaut: failure_60s)",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["baseline", "logistic", "random_forest", "gradient_boosting"],
        help="Modèles à évaluer",
    )
    parser.add_argument(
        "--data", default="data/processed",
        help="Répertoire des données processed",
    )
    parser.add_argument(
        "--output", default="evaluation/results/failure_prediction_results.json",
        help="Fichier de sortie JSON",
    )
    args = parser.parse_args()

    # -- Split
    splitter = TemporalSplitter(processed_dir=args.data)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(label_col=args.label)
    _, _, df_test_meta = splitter.split_with_meta(label_col=args.label)

    logger.info(
        "Split — train: %d  val: %d  test: %d  features: %d",
        len(X_train), len(X_val), len(X_test), len(splitter.feature_cols),
    )
    logger.info(
        "Taux positifs — train: %.1f%%  val: %.1f%%  test: %.1f%%",
        100 * (y_train == 1).mean(),
        100 * (y_val   == 1).mean(),
        100 * (y_test  == 1).mean(),
    )

    # -- Évaluation de chaque modèle
    results = []
    for model_name in args.models:
        try:
            model, name = load_model(model_name)
            result = evaluate_model(
                model, name,
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                args.label, df_test_meta,
            )
            results.append(result)
        except Exception as exc:
            logger.error("Erreur pour le modèle %s : %s", model_name, exc, exc_info=True)

    # -- Résumé
    print_comparison(results)

    # -- Sauvegarde JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"label": args.label, "results": results}, indent=2, ensure_ascii=False)
    )
    logger.info("Résultats sauvegardés : %s", output_path)


if __name__ == "__main__":
    main()
