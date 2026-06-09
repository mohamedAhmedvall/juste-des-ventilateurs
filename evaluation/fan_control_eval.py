"""Évaluation comparative des contrôleurs de ventilateurs — Phase 5.

Pour chaque contrôleur :
  1. Entraîner sur le split train (avec risk_scores du prédicteur logistic)
  2. Évaluer sur le split test
  3. Calculer les métriques : shutdowns évités, T_mean, énergie fans, accuracy action

Usage :
    python -m evaluation.fan_control_eval
    python -m evaluation.fan_control_eval --label failure_60s --models all
    python -m evaluation.fan_control_eval --models baseline_fixed baseline_pid
"""
from __future__ import annotations

from evaluation import _compat  # noqa: F401 — force UTF-8 stdout Windows

import argparse
import json
import sys
import warnings
from pathlib import Path

# Supprimer les warnings de compatibilite sklearn entre versions
warnings.filterwarnings("ignore", category=UserWarning, message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.failure_prediction.logistic_regression import LogisticPredictor
from models.failure_prediction.splitter import TemporalSplitter
from models.fan_control.baseline_fixed import FixedController
from models.fan_control.baseline_threshold import ThresholdFanController
from models.fan_control.baseline_pid import PIDController
from models.fan_control.supervised_controller import SupervisedController
from models.fan_control.score_controller import ScoreController

SAVED_DIR   = ROOT / "models" / "fan_control" / "saved"
RESULTS_DIR = ROOT / "evaluation" / "results"
RPM_MAX     = 4500

# Contrôleurs à évaluer (nom → classe)
CONTROLLERS = {
    "baseline_fixed_1500":  lambda: FixedController(rpm=1500),
    "baseline_fixed_2500":  lambda: FixedController(rpm=2500),
    "baseline_fixed_4500":  lambda: FixedController(rpm=4500),
    "baseline_threshold":   lambda: ThresholdFanController(),
    "baseline_pid":         lambda: PIDController(),
    "supervised":           lambda: SupervisedController(),
    "score_controller":     lambda: ScoreController(),
}


# ---------------------------------------------------------------------------
# Chargement des risk_scores depuis le prédicteur logistic
# ---------------------------------------------------------------------------

def _load_risk_scores(
    X: pd.DataFrame,
    label_col: str,
) -> np.ndarray:
    """Charge le modèle logistic entraîné et calcule les probabilités de panne."""
    model_path = ROOT / "models" / "failure_prediction" / "saved" / f"logistic_{label_col}.joblib"
    if not model_path.exists():
        print(f"  [WARN] Predicteur logistic absent ({model_path}). risk_score=0.")
        return np.zeros(len(X))
    model = LogisticPredictor().load(str(model_path))
    proba = model.predict_proba(X)
    return proba[:, 1]


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def _compute_metrics(
    X_test: pd.DataFrame,
    y_rpms: np.ndarray,       # RPM décidés par le contrôleur
    y_action: pd.Series,      # action_class oracle
    label_col: str = "failure_60s",
) -> dict:
    """Calcule les métriques d'évaluation d'un contrôleur."""

    # 1. Energie fans (loi cubique RPM^3, normalisée)
    power_fans_w = (y_rpms / RPM_MAX) ** 3 * 300.0   # proxy watts
    mean_power_fans = float(power_fans_w.mean())
    mean_rpm = float(y_rpms.mean())

    # 2. Température moyenne et % temps en zone critique
    if "temperature_c" in X_test.columns:
        t_mean = float(X_test["temperature_c"].mean())
        if "margin_to_shutdown" in X_test.columns:
            pct_critical = float((X_test["margin_to_shutdown"] < 0).mean())
        else:
            pct_critical = 0.0
    else:
        t_mean = 0.0
        pct_critical = 0.0

    # 3. Nb shutdowns approximé (ticks où status passe à off après degraded)
    if "nb_shutdowns_episode" in X_test.columns:
        nb_shutdowns = int(X_test["nb_shutdowns_episode"].max())
    else:
        nb_shutdowns = -1

    # 4. Action accuracy vs oracle
    if y_action is not None and len(y_action) == len(y_rpms):
        from models.fan_control.baseline_fixed import RPM_LEVELS
        action_to_rpm = {0: 0, 1: 1500, 2: 2500, 3: 3500, 4: 4500}
        y_oracle_rpm = y_action.fillna(1).astype(int).map(action_to_rpm).values
        action_accuracy = float((y_rpms == y_oracle_rpm).mean())
        # Rack par rapport à optimal_rpm si présent
        if "optimal_rpm" in X_test.columns:
            y_opt = X_test["optimal_rpm"].fillna(1500).astype(int).values
            rpm_mae = float(np.abs(y_rpms - y_opt).mean())
        else:
            rpm_mae = float(np.abs(y_rpms - y_oracle_rpm).mean())
    else:
        action_accuracy = -1.0
        rpm_mae = -1.0

    # 5. Recall sur cas dangereux (failure_60s=1)
    if label_col in X_test.columns:
        dangerous = X_test[label_col].fillna(0).astype(int).values == 1
    elif hasattr(y_action, 'values'):
        dangerous = np.zeros(len(y_rpms), dtype=bool)
    else:
        dangerous = np.zeros(len(y_rpms), dtype=bool)

    if dangerous.sum() > 0:
        # On considère une alerte correcte si RPM >= 3500 quand dangereux
        high_rpm_when_dangerous = (y_rpms[dangerous] >= 3500).mean()
    else:
        high_rpm_when_dangerous = -1.0

    return {
        "mean_rpm":               mean_rpm,
        "mean_power_fans_w":      mean_power_fans,
        "t_mean":                 t_mean,
        "pct_critical":           pct_critical,
        "nb_shutdowns":           nb_shutdowns,
        "action_accuracy":        action_accuracy,
        "rpm_mae":                rpm_mae,
        "high_rpm_when_dangerous": high_rpm_when_dangerous,
    }


# ---------------------------------------------------------------------------
# Évaluation d'un contrôleur
# ---------------------------------------------------------------------------

def evaluate_controller(
    name: str,
    ctrl_factory,
    X_train: pd.DataFrame,
    y_train_action: pd.Series,
    X_test: pd.DataFrame,
    y_test_action: pd.Series,
    risk_scores_train: np.ndarray,
    risk_scores_test: np.ndarray,
    label_col: str,
) -> dict:
    print(f"\n--- {name} ---")
    ctrl = ctrl_factory()

    # Entraînement (les baselines fixes ignorent fit())
    if name in ("supervised", "score_controller", "baseline_threshold", "baseline_pid"):
        print(f"  Entrainement...")
        if name == "score_controller":
            ctrl.fit(X_train, y_train_action, risk_scores_train=risk_scores_train)
        else:
            ctrl.fit(X_train, y_train_action)

    # Prédictions sur le test
    print(f"  Prediction sur {len(X_test):,} lignes...")
    if name == "score_controller":
        y_rpms = ctrl.decide_batch(X_test, risk_scores=risk_scores_test)
    elif name == "supervised":
        y_rpms = ctrl.decide_batch(X_test, risk_scores=risk_scores_test)
    else:
        y_rpms = ctrl.decide_batch(X_test)

    # Métriques
    metrics = _compute_metrics(X_test, y_rpms, y_test_action, label_col=label_col)

    # Sauvegarde du modèle
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVED_DIR / f"{name}.joblib"
    try:
        if hasattr(ctrl, 'save'):
            suffix = ".json" if name.startswith("baseline_fixed") or \
                                name == "baseline_threshold" or \
                                name == "baseline_pid" or \
                                name == "score_controller" else ".joblib"
            save_path = SAVED_DIR / f"{name}{suffix}"
            ctrl.save(str(save_path))
            print(f"  Sauvegarde : {save_path.name}")
    except Exception as e:
        print(f"  [WARN] Sauvegarde echouee : {e}")

    result = {
        "controller": name,
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        **metrics,
    }
    print(f"  mean_rpm={metrics['mean_rpm']:.0f}  "
          f"t_mean={metrics['t_mean']:.1f}C  "
          f"action_acc={metrics['action_accuracy']:.3f}  "
          f"rpm_mae={metrics['rpm_mae']:.0f}")
    return result


# ---------------------------------------------------------------------------
# Tableau comparatif
# ---------------------------------------------------------------------------

def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("COMPARAISON DES CONTROLEURS DE VENTILATEURS")
    print("=" * 80)
    header = f"{'Controleur':<24} {'MeanRPM':>8} {'T_mean':>8} {'%Crit':>7} {'AccAct':>8} {'RPM_MAE':>8} {'DangHigh':>9}"
    print(header)
    print("-" * 80)
    for r in sorted(results, key=lambda x: x.get("action_accuracy", 0), reverse=True):
        name    = r["controller"][:23]
        mrpm    = f"{r['mean_rpm']:.0f}"
        tmean   = f"{r['t_mean']:.1f}"
        crit    = f"{r['pct_critical']*100:.1f}%"
        acc     = f"{r['action_accuracy']:.3f}" if r['action_accuracy'] >= 0 else "N/A"
        mae     = f"{r['rpm_mae']:.0f}" if r['rpm_mae'] >= 0 else "N/A"
        dang    = f"{r['high_rpm_when_dangerous']:.3f}" if r['high_rpm_when_dangerous'] >= 0 else "N/A"
        print(f"{name:<24} {mrpm:>8} {tmean:>8} {crit:>7} {acc:>8} {mae:>8} {dang:>9}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluation des controleurs Phase 5")
    parser.add_argument("--label",   default="failure_60s",
                        help="Label de panne a utiliser pour risk_score")
    parser.add_argument("--models",  nargs="+", default=["all"],
                        help="Controleurs a evaluer (all ou liste de noms)")
    parser.add_argument("--output",  default=None,
                        help="Fichier JSON de sortie (auto si absent)")
    args = parser.parse_args()

    # Sélection des contrôleurs
    if args.models == ["all"]:
        selected = CONTROLLERS
    else:
        selected = {k: v for k, v in CONTROLLERS.items() if k in args.models}
        unknown = set(args.models) - set(CONTROLLERS)
        if unknown:
            print(f"[WARN] Controleurs inconnus ignores : {unknown}")

    # Chargement des données
    print("Chargement des donnees...")
    splitter = TemporalSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(label_col=args.label)

    # Action_class depuis split_with_meta
    try:
        df_train, df_val, df_test = splitter.split_with_meta(label_col=args.label)
        y_train_action = df_train["action_class"].fillna(1).astype(int) \
            if "action_class" in df_train.columns else pd.Series(np.ones(len(X_train), dtype=int))
        y_test_action  = df_test["action_class"].fillna(1).astype(int) \
            if "action_class" in df_test.columns else pd.Series(np.ones(len(X_test), dtype=int))
    except Exception as e:
        print(f"[WARN] split_with_meta echoue ({e}), action_class=1 partout")
        y_train_action = pd.Series(np.ones(len(X_train), dtype=int))
        y_test_action  = pd.Series(np.ones(len(X_test), dtype=int))

    print(f"Train : {len(X_train):,}  Test : {len(X_test):,}")

    # Risk scores du prédicteur logistic
    # IMPORTANT : calculer AVANT d'enrichir X_test avec des colonnes métriques
    # (le prédicteur attend exactement les 47 features du splitter)
    print("Calcul des risk_scores (predicteur logistic)...")
    risk_train = _load_risk_scores(X_train, args.label)
    risk_test  = _load_risk_scores(X_test,  args.label)

    # Enrichir X_test avec les colonnes métriques (après calcul risk_scores)
    try:
        df_train, df_val, df_test = splitter.split_with_meta(label_col=args.label)
        X_test_meta = X_test.copy()
        for col in [args.label, "optimal_rpm", "margin_to_shutdown",
                    "nb_shutdowns_episode", "temperature_c"]:
            if col in df_test.columns and col not in X_test_meta.columns:
                X_test_meta[col] = df_test[col].values
            elif col in df_test.columns:
                X_test_meta[col] = df_test[col].values
        X_test = X_test_meta
    except Exception:
        pass  # X_test reste sans colonnes métriques, _compute_metrics gérera

    # Évaluation
    results = []
    for name, factory in selected.items():
        try:
            result = evaluate_controller(
                name, factory,
                X_train, y_train_action,
                X_test, y_test_action,
                risk_train, risk_test,
                label_col=args.label,
            )
            results.append(result)
        except Exception as e:
            print(f"  [ERREUR] {name} : {e}")
            import traceback; traceback.print_exc()

    print_comparison(results)

    # Export JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(RESULTS_DIR / "fan_control_results.json")
    payload = {
        "label":       args.label,
        "n_train":     len(X_train),
        "n_test":      len(X_test),
        "controllers": [r["controller"] for r in results],
        "results":     results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResultats sauvegardes : {out_path}")


if __name__ == "__main__":
    main()
