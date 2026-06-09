"""Benchmark comparatif offline — Phase 6.

Compare 3 modes de supervision sur le jeu de test :

  0. native   : pas d'intervention (RPM oracle auto du simulateur)
  1. threshold : contrôleur à seuils (baseline_threshold)
  2. ml        : prédicteur logistic + contrôleur supervisé (recommandé Phase 4/5)

Pour chaque mode, on rejoue le dataset de test offline et on calcule :
  - Nombre de ticks en zone critique (margin_to_shutdown < 0)
  - Temperature moyenne et max
  - RPM moyen (proxy énergie fans)
  - Taux de réaction en situation de danger (risk_score > 0.5 → RPM >= 3500)
  - Nombre d'incidents détectés avant shutdown (lead time > 0)
  - Action accuracy vs oracle (action_class)

Usage :
    python -m evaluation.benchmark
    python -m evaluation.benchmark --label failure_60s --output results/benchmark.json
"""
from __future__ import annotations

from evaluation import _compat  # noqa: F401 — force UTF-8 stdout Windows

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.failure_prediction.splitter import TemporalSplitter
from models.failure_prediction.logistic_regression import LogisticPredictor
from models.fan_control.baseline_threshold import ThresholdFanController
from models.fan_control.baseline_pid import PIDController
from models.fan_control.supervised_controller import SupervisedController
from models.fan_control.score_controller import ScoreController
from models.fan_control.baseline_fixed import FixedController

SAVED_PRED = ROOT / "models" / "failure_prediction" / "saved"
SAVED_CTRL = ROOT / "models" / "fan_control" / "saved"
RESULTS_DIR = ROOT / "evaluation" / "results"

RPM_HIGH    = 4500
RPM_DEFAULT = 2500
RISK_THR    = 0.60   # Seuil de surcharge risque → RPM_HIGH


# ---------------------------------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------------------------------

def _load_logistic(label: str):
    path = SAVED_PRED / f"logistic_{label}.joblib"
    if not path.exists():
        print(f"  [WARN] {path.name} absent — risk_score=0")
        return None
    return LogisticPredictor().load(str(path))


def _load_ctrl(name: str):
    if name == "supervised":
        p = SAVED_CTRL / "supervised.joblib"
        return SupervisedController.load(str(p)) if p.exists() else None
    if name == "score_controller":
        p = SAVED_CTRL / "score_controller.json"
        return ScoreController.load(str(p)) if p.exists() else None
    if name == "baseline_threshold":
        p = SAVED_CTRL / "baseline_threshold.json"
        return ThresholdFanController.load(str(p)) if p.exists() else None
    if name == "baseline_pid":
        p = SAVED_CTRL / "baseline_pid.json"
        return PIDController.load(str(p)) if p.exists() else None
    return None


# ---------------------------------------------------------------------------
# Simulation offline d'un mode
# ---------------------------------------------------------------------------

def _risk_scores(predictor, X: pd.DataFrame) -> np.ndarray:
    if predictor is None:
        return np.zeros(len(X))
    try:
        proba = predictor.predict_proba(X)
        return proba[:, 1]
    except Exception:
        return np.zeros(len(X))


def run_mode(
    name: str,
    X_test: pd.DataFrame,
    df_test_meta: pd.DataFrame,
    label_col: str,
    predictor=None,
    controller=None,
    risk_thr: float = RISK_THR,
) -> dict:
    """Rejoue le dataset de test avec un mode donné et calcule les métriques."""
    print(f"\n--- {name} ---")

    # Risk scores
    risk = _risk_scores(predictor, X_test)

    # Décisions RPM
    if name == "native":
        # Mode natif : on utilise le RPM oracle (fan_rpm_mean du simulateur)
        if "fan_rpm_mean" in df_test_meta.columns:
            rpms = df_test_meta["fan_rpm_mean"].fillna(RPM_DEFAULT).astype(int).values
        else:
            rpms = np.full(len(X_test), RPM_DEFAULT, dtype=int)
    elif controller is None:
        rpms = np.full(len(X_test), RPM_DEFAULT, dtype=int)
    else:
        try:
            rpms = controller.decide_batch(X_test, risk_scores=risk)
        except TypeError:
            rpms = controller.decide_batch(X_test)

        # Surcharge risque élevé (override ML)
        if predictor is not None:
            high_risk = risk >= risk_thr
            rpms[high_risk] = RPM_HIGH

    # ---- Métriques ----
    # 1. Energie fans (loi cubique RPM^3)
    mean_rpm       = float(rpms.mean())
    mean_power_w   = float(((rpms / 4500) ** 3 * 300).mean())

    # 2. Température
    t_col = "temperature_c"
    if t_col in df_test_meta.columns:
        t_mean = float(df_test_meta[t_col].mean())
        t_max  = float(df_test_meta[t_col].max())
    else:
        t_mean = t_max = 0.0

    # 3. Temps en zone critique (margin < 0)
    if "margin_to_shutdown" in df_test_meta.columns:
        pct_critical = float((df_test_meta["margin_to_shutdown"] < 0).mean())
        n_critical   = int((df_test_meta["margin_to_shutdown"] < 0).sum())
    else:
        pct_critical = 0.0
        n_critical   = 0

    # 4. Shutdowns
    if "nb_shutdowns_episode" in df_test_meta.columns:
        nb_shutdowns = int(df_test_meta["nb_shutdowns_episode"].max())
    else:
        nb_shutdowns = -1

    # 5. Réaction en situation dangereuse
    if label_col in df_test_meta.columns:
        dangerous = df_test_meta[label_col].fillna(0).astype(int).values == 1
        if dangerous.sum() > 0:
            high_rpm_when_dangerous = float((rpms[dangerous] >= 3500).mean())
            n_dangerous_ticks = int(dangerous.sum())
        else:
            high_rpm_when_dangerous = -1.0
            n_dangerous_ticks = 0
    else:
        high_rpm_when_dangerous = -1.0
        n_dangerous_ticks = 0

    # 6. Action accuracy vs oracle
    if "action_class" in df_test_meta.columns:
        action_to_rpm = {0: 0, 1: 1500, 2: 2500, 3: 3500, 4: 4500}
        y_oracle = df_test_meta["action_class"].fillna(1).astype(int).map(action_to_rpm).values
        action_accuracy = float((rpms == y_oracle).mean())
        rpm_mae         = float(np.abs(rpms - y_oracle).mean())
    else:
        action_accuracy = -1.0
        rpm_mae         = -1.0

    # 7. Lead time : détection d'incidents via risk_score avant shutdown
    # On cherche les incidents (passages en shutdown) et on calcule le temps
    # d'anticipation moyen de la première alerte (risk > 0.5) avant l'incident
    lead_times = _compute_lead_times(df_test_meta, risk, label_col)
    mean_lead_time   = float(np.mean(lead_times)) if lead_times else -1.0
    median_lead_time = float(np.median(lead_times)) if lead_times else -1.0
    n_incidents_detected = len([lt for lt in lead_times if lt > 0])
    n_incidents_total    = len(lead_times)

    result = {
        "mode":                   name,
        "n_test":                 len(X_test),
        "mean_rpm":               mean_rpm,
        "mean_power_fans_w":      mean_power_w,
        "t_mean":                 t_mean,
        "t_max":                  t_max,
        "pct_critical":           pct_critical,
        "n_critical_ticks":       n_critical,
        "nb_shutdowns":           nb_shutdowns,
        "n_dangerous_ticks":      n_dangerous_ticks,
        "high_rpm_when_dangerous": high_rpm_when_dangerous,
        "action_accuracy":        action_accuracy,
        "rpm_mae":                rpm_mae,
        "mean_lead_time_s":       mean_lead_time,
        "median_lead_time_s":     median_lead_time,
        "n_incidents_detected":   n_incidents_detected,
        "n_incidents_total":      n_incidents_total,
    }

    print(f"  mean_rpm={mean_rpm:.0f}  T_mean={t_mean:.1f}°C  T_max={t_max:.1f}°C")
    print(f"  pct_critical={pct_critical*100:.1f}%  nb_shutdowns={nb_shutdowns}")
    print(f"  action_acc={action_accuracy:.3f}  high_risk_react={high_rpm_when_dangerous:.3f}")
    print(f"  incidents détectés={n_incidents_detected}/{n_incidents_total}  "
          f"lead_time_med={median_lead_time:.0f}s")
    return result


def _compute_lead_times(
    df_meta: pd.DataFrame,
    risk_scores: np.ndarray,
    label_col: str,
    risk_alert_thr: float = 0.50,
    window_s: int = 120,
) -> list[float]:
    """Calcule le lead time de chaque incident détecté.

    Pour chaque transition positive→négatif sur label_col (fin d'une fenêtre de danger),
    cherche le premier tick dans les `window_s` secondes précédentes où risk_score > thr.
    Retourne la liste des lead times en secondes (un par incident).
    """
    if label_col not in df_meta.columns:
        return []

    labels = df_meta[label_col].fillna(0).astype(int).values
    # Estimer le timestamp si disponible
    if "timestamp" in df_meta.columns:
        try:
            ts = pd.to_datetime(df_meta["timestamp"], utc=True)
            dt_s = ts.diff().dt.total_seconds().fillna(5).values
        except Exception:
            dt_s = np.full(len(labels), 5.0)
    else:
        dt_s = np.full(len(labels), 5.0)

    lead_times = []
    in_danger  = False
    danger_start = 0

    for i in range(len(labels)):
        if labels[i] == 1 and not in_danger:
            in_danger    = True
            danger_start = i
        elif labels[i] == 0 and in_danger:
            # Fin d'une fenêtre de danger : chercher la première alerte avant
            in_danger = False
            # Chercher dans les window_s secondes avant danger_start
            t_accum = 0.0
            alert_idx = None
            for j in range(danger_start - 1, -1, -1):
                t_accum += dt_s[j + 1]
                if t_accum > window_s:
                    break
                if risk_scores[j] > risk_alert_thr:
                    alert_idx = j
            if alert_idx is not None:
                # Lead time = temps depuis alerte jusqu'à danger_start
                lead = float(sum(dt_s[alert_idx + 1: danger_start + 1]))
                lead_times.append(lead)
            else:
                lead_times.append(0.0)  # Incident non détecté

    return lead_times


# ---------------------------------------------------------------------------
# Tableau comparatif
# ---------------------------------------------------------------------------

def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("BENCHMARK COMPARATIF — PHASE 6")
    print("=" * 90)
    hdr = (f"{'Mode':<20} {'MeanRPM':>8} {'T_mean':>7} {'T_max':>7} "
           f"{'%Crit':>7} {'AccAct':>8} {'DangHigh':>9} {'LeadMed':>9} {'Detect':>8}")
    print(hdr)
    print("-" * 90)
    for r in results:
        print(
            f"{r['mode']:<20} {r['mean_rpm']:>8.0f} {r['t_mean']:>7.1f} {r['t_max']:>7.1f} "
            f"{r['pct_critical']*100:>6.1f}% {r['action_accuracy']:>8.3f} "
            f"{r['high_rpm_when_dangerous']:>9.3f} {r['median_lead_time_s']:>9.0f} "
            f"{r['n_incidents_detected']:>4}/{r['n_incidents_total']:<3}"
        )
    print("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark comparatif Phase 6")
    parser.add_argument("--label",    default="failure_60s")
    parser.add_argument("--output",   default=None)
    parser.add_argument("--risk-thr", type=float, default=RISK_THR)
    args = parser.parse_args()

    print("Chargement des données...")
    splitter = TemporalSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(label_col=args.label)
    df_train, df_val, df_test = splitter.split_with_meta(label_col=args.label)

    print(f"Test : {len(X_test):,} lignes  ({y_test.mean():.1%} positifs)")

    print("\nChargement des modèles...")
    predictor  = _load_logistic(args.label)
    ctrl_ml    = _load_ctrl("supervised")
    ctrl_thr   = _load_ctrl("baseline_threshold")

    # Risk scores sur X_test (47 features)
    risk_test = _risk_scores(predictor, X_test)

    # Modes à comparer
    modes = [
        ("native",    None,      None),
        ("threshold", None,      ctrl_thr),
        ("ml",        predictor, ctrl_ml),
    ]

    results = []
    for mode_name, pred, ctrl in modes:
        r = run_mode(
            name          = mode_name,
            X_test        = X_test,
            df_test_meta  = df_test,
            label_col     = args.label,
            predictor     = pred,
            controller    = ctrl,
            risk_thr      = args.risk_thr,
        )
        results.append(r)

    print_comparison(results)

    # Export JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(RESULTS_DIR / "benchmark_results.json")
    payload = {
        "label":    args.label,
        "n_test":   len(X_test),
        "risk_thr": args.risk_thr,
        "results":  results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nRésultats sauvegardés : {out_path}")


if __name__ == "__main__":
    main()
