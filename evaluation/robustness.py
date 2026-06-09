"""Test de robustesse par scénario — Phase 6.

Évalue le couple ML (logistic + supervised) sur chaque scénario séparément
pour mesurer la généralisation et identifier les points de faiblesse.

Pour chaque scénario (basic, busy_weeks, heatwave, nominal, stress, trace_replay) :
  - Rejouer le segment test du scénario avec le mode ML
  - Calculer les métriques
  - Identifier les dégradations vs le mode natif

Usage :
    python -m evaluation.robustness
    python -m evaluation.robustness --label failure_60s
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

from models.failure_prediction.splitter import TemporalSplitter, TRAIN_RATIO, VAL_RATIO
from models.failure_prediction.logistic_regression import LogisticPredictor
from models.fan_control.supervised_controller import SupervisedController

SAVED_PRED  = ROOT / "models" / "failure_prediction" / "saved"
SAVED_CTRL  = ROOT / "models" / "fan_control" / "saved"
RESULTS_DIR = ROOT / "evaluation" / "results"
PROC_DIR    = ROOT / "data" / "processed"

RPM_HIGH    = 4500
RISK_THR    = 0.60

NON_FEATURE_COLS = {
    "timestamp", "cluster_id", "machine_id", "role", "msg_type", "status",
    "fault_types", "fan_modes",
    "failure_30s", "failure_60s", "hot_30s", "time_to_failure_s",
    "optimal_rpm", "action_class", "time_in_degraded_s",
    "machines_total", "machines_on", "status_cause",
}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes features (même logique que TemporalSplitter)."""
    return [
        c for c in df.select_dtypes(include="number").columns
        if c not in NON_FEATURE_COLS and df[c].isnull().mean() < 1.0
    ]


def _load_episode_test(ep_path: Path, label_col: str):
    """Charge le segment test (15% final) d'un épisode."""
    import json as _json
    df = pd.read_parquet(ep_path / "features.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    n      = len(df)
    n_test = int(n * (1 - TRAIN_RATIO - VAL_RATIO))
    n_val  = int(n * VAL_RATIO)
    df_test = df.iloc[n - n_test:].reset_index(drop=True)

    # Métadonnées
    meta_path = ROOT / "data" / "raw" / ep_path.name / "metadata.json"
    scenario  = "unknown"
    if meta_path.exists():
        scenario = _json.loads(meta_path.read_text()).get("scenario", "unknown")

    return df_test, scenario


def _risk_scores(predictor, X: pd.DataFrame) -> np.ndarray:
    if predictor is None:
        return np.zeros(len(X))
    try:
        return predictor.predict_proba(X)[:, 1]
    except Exception:
        return np.zeros(len(X))


def run_scenario(
    scenario: str,
    df_test: pd.DataFrame,
    label_col: str,
    predictor,
    controller,
    feature_cols: list[str],
) -> dict:
    """Évalue un scénario en mode ML et en mode natif."""
    # Features disponibles dans ce scénario
    feat_present = [c for c in feature_cols if c in df_test.columns]
    X = df_test[feat_present].copy()
    # Aligner sur les features du modèle (compléter avec 0 si manquant)
    try:
        all_feats = predictor.feature_names_in_ if hasattr(predictor, "feature_names_in_") else feature_cols
        for c in all_feats:
            if c not in X.columns:
                X[c] = 0.0
        X = X[all_feats]
    except Exception:
        pass

    risk = _risk_scores(predictor, X)

    # Décisions ML
    try:
        rpms_ml = controller.decide_batch(X, risk_scores=risk)
    except TypeError:
        rpms_ml = controller.decide_batch(X)
    # Override risque élevé
    rpms_ml[risk >= RISK_THR] = RPM_HIGH

    # Décisions natives (fan_rpm_mean du simulateur)
    if "fan_rpm_mean" in df_test.columns:
        rpms_native = df_test["fan_rpm_mean"].fillna(2500).astype(int).values
    else:
        rpms_native = np.full(len(df_test), 2500, dtype=int)

    def _metrics(rpms: np.ndarray, tag: str) -> dict:
        mean_rpm     = float(rpms.mean())
        mean_power   = float(((rpms / 4500) ** 3 * 300).mean())
        if label_col in df_test.columns:
            danger = df_test[label_col].fillna(0).astype(int).values == 1
            react  = float((rpms[danger] >= 3500).mean()) if danger.sum() > 0 else -1.0
        else:
            react = -1.0
        if "margin_to_shutdown" in df_test.columns:
            pct_crit = float((df_test["margin_to_shutdown"] < 0).mean())
        else:
            pct_crit = 0.0
        if "temperature_c" in df_test.columns:
            t_mean = float(df_test["temperature_c"].mean())
            t_max  = float(df_test["temperature_c"].max())
        else:
            t_mean = t_max = 0.0
        if "nb_shutdowns_episode" in df_test.columns:
            nb_shut = int(df_test["nb_shutdowns_episode"].max())
        else:
            nb_shut = -1
        if "action_class" in df_test.columns:
            action_to_rpm = {0: 0, 1: 1500, 2: 2500, 3: 3500, 4: 4500}
            y_oracle = df_test["action_class"].fillna(1).astype(int).map(action_to_rpm).values
            acc  = float((rpms == y_oracle).mean())
            mae  = float(np.abs(rpms - y_oracle).mean())
        else:
            acc = mae = -1.0
        return {
            f"{tag}_mean_rpm":    mean_rpm,
            f"{tag}_power_w":     mean_power,
            f"{tag}_t_mean":      t_mean,
            f"{tag}_t_max":       t_max,
            f"{tag}_pct_crit":    pct_crit,
            f"{tag}_nb_shut":     nb_shut,
            f"{tag}_react":       react,
            f"{tag}_acc":         acc,
            f"{tag}_mae":         mae,
        }

    row = {"scenario": scenario, "n_test": len(df_test)}
    row.update(_metrics(rpms_ml,     "ml"))
    row.update(_metrics(rpms_native, "nat"))

    # Delta énergie : économie ML vs natif
    row["delta_rpm"]    = row["ml_mean_rpm"] - row["nat_mean_rpm"]
    row["delta_power_w"] = row["ml_power_w"]  - row["nat_power_w"]

    print(f"  {scenario:<15} n={len(df_test):>6,}  "
          f"ML rpm={row['ml_mean_rpm']:.0f}  nat rpm={row['nat_mean_rpm']:.0f}  "
          f"ML acc={row['ml_acc']:.3f}  ML react={row['ml_react']:.3f}")
    return row


def print_robustness(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("ROBUSTESSE PAR SCENARIO — Mode ML vs Natif")
    print("=" * 100)
    hdr = (f"{'Scenario':<15} {'n':>6} {'ML_RPM':>7} {'Nat_RPM':>7} "
           f"{'ΔRPM':>6} {'ML_acc':>7} {'ML_react':>9} {'ML_crit%':>9} {'Nat_crit%':>9}")
    print(hdr)
    print("-" * 100)
    for r in results:
        print(
            f"{r['scenario']:<15} {r['n_test']:>6,} {r['ml_mean_rpm']:>7.0f} "
            f"{r['nat_mean_rpm']:>7.0f} {r['delta_rpm']:>+6.0f} "
            f"{r['ml_acc']:>7.3f} {r['ml_react']:>9.3f} "
            f"{r['ml_pct_crit']*100:>8.1f}% {r['nat_pct_crit']*100:>8.1f}%"
        )
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test de robustesse par scénario — Phase 6")
    parser.add_argument("--label",  default="failure_60s")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("Chargement des modèles...")
    pred_path = SAVED_PRED / f"logistic_{args.label}.joblib"
    ctrl_path = SAVED_CTRL / "supervised.joblib"

    if not pred_path.exists():
        print(f"[ERREUR] Prédicteur absent : {pred_path}")
        sys.exit(1)
    if not ctrl_path.exists():
        print(f"[ERREUR] Contrôleur absent : {ctrl_path}")
        sys.exit(1)

    predictor  = LogisticPredictor().load(str(pred_path))
    controller = SupervisedController.load(str(ctrl_path))

    # Feature cols de référence (depuis le splitter)
    splitter = TemporalSplitter()
    X_train, *_ = splitter.split(label_col=args.label)
    feature_cols = list(X_train.columns)
    print(f"Features du modèle : {len(feature_cols)}")

    # Itérer sur les épisodes
    episodes = sorted(p for p in PROC_DIR.glob("episode=*") if p.is_dir())
    print(f"\nÉvaluation sur {len(episodes)} épisodes...\n")

    results = []
    for ep in episodes:
        try:
            df_test, scenario = _load_episode_test(ep, args.label)
            if len(df_test) == 0:
                continue
            row = run_scenario(
                scenario     = scenario,
                df_test      = df_test,
                label_col    = args.label,
                predictor    = predictor,
                controller   = controller,
                feature_cols = feature_cols,
            )
            results.append(row)
        except Exception as e:
            print(f"  [ERREUR] {ep.name} : {e}")
            import traceback; traceback.print_exc()

    print_robustness(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(RESULTS_DIR / "robustness_results.json")
    with open(out_path, "w") as f:
        json.dump({"label": args.label, "results": results}, f, indent=2)
    print(f"\nRésultats sauvegardés : {out_path}")


if __name__ == "__main__":
    main()
