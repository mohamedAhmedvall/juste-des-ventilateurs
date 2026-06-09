"""
ingest_quick_EDA.py — Analyse rapide des datasets raw et processed

Usage :
    python ingest_quick_EDA.py                     # tous les épisodes
    python ingest_quick_EDA.py --episode 003       # épisode spécifique
    python ingest_quick_EDA.py --raw-only          # raw uniquement
    python ingest_quick_EDA.py --processed-only    # processed uniquement
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

LABEL_COLS = ["failure_30s", "failure_60s", "hot_30s", "action_class"]
FEATURE_GROUPS = {
    "Temporelles": ["temp_delta_5s", "temp_delta_30s", "margin_to_shutdown", "margin_pct",
                    "temp_rolling_mean_30s", "temp_rolling_std_30s", "margin_delta_30s"],
    "Contextuelles": ["time_in_hot_zone_s", "nb_shutdowns_episode", "nb_degraded_episode",
                      "has_fan_fault", "has_power_surge", "is_recovering"],
    "Energetiques": ["power_fans_w", "fan_energy_ratio", "pue_estimated",
                     "energy_fans_kwh_cumulated"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(char="─", width=70):
    print(char * width)

def header(title: str):
    sep("═")
    print(f"  {title}")
    sep("═")

def section(title: str):
    print()
    sep()
    print(f"  {title}")
    sep()


def fmt_pct(n, total):
    if total == 0:
        return "N/A"
    return f"{n:,} ({100*n/total:.1f}%)"


def load_raw_episode(ep_dir: Path) -> pd.DataFrame:
    """Charge tous les parquet machine (hors _cluster) d'un épisode raw."""
    frames = []
    for machine_dir in sorted(ep_dir.iterdir()):
        if not machine_dir.is_dir():
            continue
        if machine_dir.name == "machine=_cluster":
            continue
        for f in machine_dir.glob("*.parquet"):
            frames.append(pd.read_parquet(f))
        for f in machine_dir.glob("*.csv"):
            frames.append(pd.read_csv(f, parse_dates=["timestamp"]))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def load_processed_episode(ep_dir: Path) -> pd.DataFrame:
    """Charge le features.parquet d'un épisode processed."""
    pq = ep_dir / "features.parquet"
    csv = ep_dir / "features.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv, parse_dates=["timestamp"])
    else:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def load_metadata(ep_dir: Path) -> dict:
    meta_path = ep_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}

# ---------------------------------------------------------------------------
# Analyse RAW
# ---------------------------------------------------------------------------

def analyse_raw(ep_id: str, ep_dir: Path, meta: dict):
    section(f"[RAW] Episode {ep_id}  —  scenario: {meta.get('scenario', '?')}")

    df = load_raw_episode(ep_dir)
    if df.empty:
        print("  ⚠  Aucune donnée trouvée.")
        return

    n = len(df)
    machines = sorted(df["machine_id"].unique()) if "machine_id" in df.columns else []

    # -- Résumé général
    print(f"  Enregistrements : {n:,}")
    print(f"  Colonnes        : {len(df.columns)}")
    print(f"  Machines        : {', '.join(machines)}")

    if meta:
        print(f"  Durée réelle    : {meta.get('duration_s', '?'):.1f}s")
        print(f"  Durée simulée   : {meta.get('sim_duration_s', '?'):.0f}s "
              f"(~{meta.get('sim_duration_s', 0)/3600:.1f}h)")

    # -- Couverture temporelle
    if "timestamp" in df.columns:
        ts = df["timestamp"].dropna()
        if len(ts):
            print(f"  Période sim     : {ts.min()} → {ts.max()}")

    # -- Répartition par machine
    if "machine_id" in df.columns and "msg_type" in df.columns:
        print()
        print("  Répartition par machine :")
        for mid in machines:
            sub = df[df["machine_id"] == mid]
            telemetry = (sub["msg_type"] == "telemetry").sum()
            events = (sub["msg_type"] != "telemetry").sum()
            print(f"    {mid:<20} {len(sub):>6,} lignes  "
                  f"(telemetry={telemetry:,}, events={events:,})")

    # -- Distribution des statuts
    if "status" in df.columns:
        print()
        print("  Distribution des statuts :")
        vc = df["status"].value_counts()
        for status, count in vc.items():
            print(f"    {status:<12} {fmt_pct(count, n)}")

    # -- Pannes
    if "has_fault" in df.columns:
        n_fault = df["has_fault"].sum()
        print()
        print(f"  Ticks avec panne active : {fmt_pct(n_fault, n)}")
        if "fault_types" in df.columns and n_fault > 0:
            fault_types = df[df["has_fault"] == True]["fault_types"].dropna()
            if len(fault_types):
                from collections import Counter
                all_types = Counter()
                for ft in fault_types:
                    for t in str(ft).split(","):
                        t = t.strip()
                        if t:
                            all_types[t] += 1
                for ftype, cnt in all_types.most_common():
                    print(f"    {ftype:<25} {cnt:,}")

    # -- Température
    if "temperature_c" in df.columns:
        temps = df["temperature_c"].dropna()
        print()
        print(f"  Température (°C) : min={temps.min():.1f}  "
              f"mean={temps.mean():.1f}  "
              f"max={temps.max():.1f}  "
              f"p95={temps.quantile(0.95):.1f}")

    # -- RPM
    if "fan_rpm_mean" in df.columns:
        rpm = df["fan_rpm_mean"].dropna()
        print(f"  RPM moyen fans   : min={rpm.min():.0f}  "
              f"mean={rpm.mean():.0f}  "
              f"max={rpm.max():.0f}")

    # -- Qualité données
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.05]
    if len(high_null):
        print()
        print("  Colonnes avec >5% NaN :")
        for col, pct in high_null.items():
            print(f"    {col:<30} {100*pct:.1f}%")
    else:
        print()
        print("  Qualité données  : OK (aucune colonne >5% NaN)")


# ---------------------------------------------------------------------------
# Analyse PROCESSED
# ---------------------------------------------------------------------------

def analyse_processed(ep_id: str, ep_dir: Path, meta: dict):
    section(f"[PROCESSED] Episode {ep_id}  —  scenario: {meta.get('scenario', '?')}")

    df = load_processed_episode(ep_dir)
    if df.empty:
        print("  ⚠  Aucune donnée trouvée.")
        return

    n = len(df)
    machines = sorted(df["machine_id"].unique()) if "machine_id" in df.columns else []
    feature_cols = [c for c in df.columns if c not in LABEL_COLS
                    and c not in ["timestamp", "cluster_id", "machine_id", "role",
                                  "msg_type", "status", "fault_types", "fan_modes"]]
    label_cols_present = [c for c in LABEL_COLS if c in df.columns]

    print(f"  Enregistrements : {n:,}")
    print(f"  Features        : {len(feature_cols)}")
    print(f"  Labels          : {', '.join(label_cols_present)}")
    print(f"  Machines        : {', '.join(machines)}")

    # -- Répartition par machine
    if "machine_id" in df.columns:
        print()
        print("  Lignes par machine :")
        for mid in machines:
            print(f"    {mid:<20} {len(df[df['machine_id']==mid]):>7,}")

    # -- Labels
    if label_cols_present:
        print()
        print("  Distribution des labels :")
        for col in label_cols_present:
            s = df[col].dropna()
            if col == "action_class":
                vc = s.value_counts().sort_index()
                print(f"    {col} :")
                for cls, cnt in vc.items():
                    print(f"      classe {int(cls)} : {fmt_pct(cnt, len(s))}")
            else:
                pos = (s == 1).sum()
                print(f"    {col:<15} positifs={fmt_pct(pos, len(s))}  "
                      f"(imbalance ratio 1:{len(s)/max(pos,1):.0f})")

    # -- Statistiques par groupe de features
    print()
    print("  Statistiques des features clés :")
    for group_name, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        print(f"    [{group_name}]")
        stats = df[present].describe(percentiles=[0.05, 0.95]).T[
            ["mean", "std", "5%", "95%", "max"]
        ]
        for col, row in stats.iterrows():
            print(f"      {col:<35} mean={row['mean']:>8.2f}  "
                  f"std={row['std']:>7.2f}  "
                  f"p95={row['95%']:>8.2f}  "
                  f"max={row['max']:>8.2f}")

    # -- Corrélation features → failure_60s
    if "failure_60s" in df.columns:
        numeric = df[feature_cols].select_dtypes(include=[np.number])
        corr = numeric.corrwith(df["failure_60s"]).abs().sort_values(ascending=False)
        top10 = corr.head(10)
        print()
        print("  Top 10 features corrélées avec failure_60s :")
        for feat, val in top10.items():
            bar = "█" * int(val * 20)
            print(f"    {feat:<35} |{bar:<20}| {val:.3f}")

    # -- NaN dans les features
    null_pct = df[feature_cols].isnull().mean()
    high_null = null_pct[null_pct > 0.05]
    if len(high_null):
        print()
        print("  Features avec >5% NaN :")
        for col, pct in high_null.items():
            print(f"    {col:<35} {100*pct:.1f}%")
    else:
        print()
        print("  Qualité features : OK (aucune colonne >5% NaN)")


# ---------------------------------------------------------------------------
# Résumé global multi-épisodes
# ---------------------------------------------------------------------------

def global_summary(episodes: list[tuple[str, dict, pd.DataFrame | None]]):
    header("RÉSUMÉ GLOBAL")

    rows = []
    for ep_id, meta, df_proc in episodes:
        if df_proc is None or df_proc.empty:
            continue
        row = {
            "Episode": ep_id,
            "Scenario": meta.get("scenario", "?"),
            "Lignes": len(df_proc),
            "Sim (h)": round(meta.get("sim_duration_s", 0) / 3600, 1),
        }
        for col in ["failure_60s", "failure_30s", "hot_30s"]:
            if col in df_proc.columns:
                s = df_proc[col].dropna()
                pos = (s == 1).sum()
                row[f"{col} (%)"] = round(100 * pos / max(len(s), 1), 1)
        if "nb_shutdowns_episode" in df_proc.columns:
            row["Shutdowns"] = int(df_proc.groupby("machine_id")["nb_shutdowns_episode"].max().sum())
        if "temperature_c" in df_proc.columns:
            row["T_max (°C)"] = round(df_proc["temperature_c"].max(), 1)
        rows.append(row)

    if not rows:
        print("  Aucun épisode processed disponible.")
        return

    summary_df = pd.DataFrame(rows).set_index("Episode")
    print(summary_df.to_string())

    # Split Option A — fenêtre temporelle par épisode
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    # TEST_RATIO  = 0.15 (implicite)

    print()
    sep()
    print("  Split Option A — fenêtre temporelle (70 / 15 / 15) par épisode")
    sep()
    print(f"  {'Episode':<10} {'Scenario':<14} {'Total':>8}  "
          f"{'Train (70%)':>12}  {'Val (15%)':>10}  {'Test (15%)':>10}  "
          f"{'f60 train%':>10}  {'f60 test%':>10}")
    sep("─")

    train_total = val_total = test_total = 0
    for ep_id, meta, df_proc in episodes:
        if df_proc is None or df_proc.empty:
            continue
        # Tri chronologique par machine puis global
        df_s = df_proc.sort_values("timestamp") if "timestamp" in df_proc.columns else df_proc
        n = len(df_s)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        n_test  = n - n_train - n_val

        train_total += n_train
        val_total   += n_val
        test_total  += n_test

        # Taux de positifs failure_60s dans train vs test
        f60_train = f60_test = "N/A"
        if "failure_60s" in df_s.columns:
            s_train = df_s.iloc[:n_train]["failure_60s"]
            s_test  = df_s.iloc[n_train + n_val:]["failure_60s"]
            pos_tr  = (s_train == 1).sum()
            pos_te  = (s_test  == 1).sum()
            f60_train = f"{100*pos_tr/max(len(s_train),1):.1f}%"
            f60_test  = f"{100*pos_te/max(len(s_test), 1):.1f}%"

        print(f"  {ep_id:<10} {meta.get('scenario','?'):<14} {n:>8,}  "
              f"{n_train:>12,}  {n_val:>10,}  {n_test:>10,}  "
              f"{f60_train:>10}  {f60_test:>10}")

    sep("─")
    grand_total = train_total + val_total + test_total
    print(f"  {'TOTAL':<10} {'':<14} {grand_total:>8,}  "
          f"{train_total:>12,}  {val_total:>10,}  {test_total:>10,}")
    print()
    print("  Stratégie : couper chaque épisode chronologiquement,")
    print("  puis concaténer : train_ep001..006 → X_train global,")
    print("                    val_ep001..006   → X_val global,")
    print("                    test_ep001..006  → X_test global.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDA rapide datasets juste-des-ventilateurs")
    parser.add_argument("--episode", type=str, default=None,
                        help="Traiter uniquement cet épisode (ex: 003)")
    parser.add_argument("--raw-only", action="store_true", help="Analyser uniquement le raw")
    parser.add_argument("--processed-only", action="store_true",
                        help="Analyser uniquement le processed")
    args = parser.parse_args()

    do_raw = not args.processed_only
    do_processed = not args.raw_only

    header("JUSTE DES VENTILATEURS — Quick EDA")

    # Lister les épisodes disponibles
    raw_eps = {p.name.replace("episode=", ""): p
               for p in sorted(RAW_DIR.glob("episode=*")) if p.is_dir()
               and p.name != "episode="}
    proc_eps = {p.name.replace("episode=", ""): p
                for p in sorted(PROCESSED_DIR.glob("episode=*")) if p.is_dir()
                and p.name != "episode="
                and not p.name.endswith("=data")}

    all_ep_ids = sorted(set(raw_eps) | set(proc_eps))
    if args.episode:
        all_ep_ids = [args.episode] if args.episode in all_ep_ids else []
        if not all_ep_ids:
            print(f"Episode '{args.episode}' introuvable.")
            sys.exit(1)

    print(f"  Episodes raw      : {', '.join(sorted(raw_eps)) or 'aucun'}")
    print(f"  Episodes processed: {', '.join(sorted(proc_eps)) or 'aucun'}")
    print(f"  Episodes analysés : {', '.join(all_ep_ids)}")

    # Collecte pour le résumé global
    global_data = []

    for ep_id in all_ep_ids:
        raw_meta = load_metadata(raw_eps[ep_id]) if ep_id in raw_eps else {}
        proc_meta = load_metadata(raw_eps[ep_id]) if ep_id in raw_eps else {}

        if do_raw and ep_id in raw_eps:
            analyse_raw(ep_id, raw_eps[ep_id], raw_meta)

        df_proc = None
        if do_processed and ep_id in proc_eps:
            analyse_processed(ep_id, proc_eps[ep_id], proc_meta)
            df_proc = load_processed_episode(proc_eps[ep_id])

        global_data.append((ep_id, raw_meta, df_proc))

    if do_processed and not args.episode and len(all_ep_ids) > 1:
        global_summary(global_data)

    print()
    sep("═")
    print("  EDA terminée.")
    sep("═")


if __name__ == "__main__":
    main()
