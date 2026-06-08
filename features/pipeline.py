"""Pipeline de feature engineering — Juste des Ventilateurs.

Point d'entrée unique pour transformer un DataFrame de télémétrie brute
en un DataFrame prêt pour l'entraînement ML.

Applique dans l'ordre :
    1. Filtrage des lignes de télémétrie machine (msg_type == "telemetry")
    2. Tri par timestamp
    3. Features temporelles (temporal.py)
    4. Features contextuelles (contextual.py)
    5. Features énergétiques (energy.py)
    6. Labels de panne (labeler.py)
    7. Labels de contrôle (labeler.py)
    8. Suppression des lignes avec NaN sur les features critiques

Usage CLI :
    python -m features.pipeline --input data/raw/episode=001 --output data/processed/episode=001

Usage Python :
    from features.pipeline import build_feature_dataset
    df_feat = build_feature_dataset(df_raw, machine_config={"t_shutdown_c": 88.0})
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from features.temporal import add_temporal_features, feature_names_temporal
from features.contextual import add_contextual_features, feature_names_contextual
from features.energy import add_energy_features, feature_names_energy
from features.labeler import (
    add_failure_labels,
    add_control_labels,
    label_names_failure,
    label_names_control,
    RPM_LEVELS,
)

logger = logging.getLogger(__name__)

# Features retirées du dataset ML final (identifiants, non prédictifs)
_DROP_FOR_ML = ["cluster_id", "msg_type", "fault_event", "fault_type_event"]

# Features critiques : une ligne avec NaN sur ces colonnes est retirée
_CRITICAL_FEATURES = ["temperature_c", "fan_rpm_mean", "status"]


def build_feature_dataset(
    df_raw: pd.DataFrame,
    machine_config: dict | None = None,
    add_labels: bool = True,
    drop_warmup_rows: int = 60,
) -> pd.DataFrame:
    """Construit le dataset de features à partir d'un DataFrame brut.

    Parameters
    ----------
    df_raw          : DataFrame de télémétrie normalisée (une ou plusieurs machines)
    machine_config  : config thermique par machine_id
                      ex: {"srv-worker-01": {"t_shutdown_c": 88.0, "fan_max_rpm": 5000}}
                      ou un dict global {"t_shutdown_c": 88.0, "fan_max_rpm": 5000}
    add_labels      : si True, calcule les labels de panne et de contrôle
    drop_warmup_rows: nombre de lignes initiales à supprimer (fenêtres glissantes invalides)

    Returns
    -------
    DataFrame avec toutes les features et labels, prêt pour le ML
    """
    cfg = machine_config or {}

    # Garder uniquement les messages de télémétrie machine
    if "msg_type" in df_raw.columns:
        df = df_raw[df_raw["msg_type"] == "telemetry"].copy()
    else:
        df = df_raw.copy()

    if df.empty:
        logger.warning("DataFrame vide après filtrage msg_type=telemetry.")
        return df

    # Tri par timestamp
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Traitement par machine
    machine_ids = df["machine_id"].dropna().unique() if "machine_id" in df.columns else [None]
    results = []

    for machine_id in machine_ids:
        if machine_id is not None:
            df_m = df[df["machine_id"] == machine_id].copy().reset_index(drop=True)
        else:
            df_m = df.copy()

        # Résoudre la config thermique pour cette machine
        if machine_id and machine_id in cfg:
            mcfg = cfg[machine_id]
        elif "t_shutdown_c" in cfg:
            mcfg = cfg  # config globale
        else:
            mcfg = {}

        t_shutdown = float(mcfg.get("t_shutdown_c", 88.0))
        fan_max_rpm = int(mcfg.get("fan_max_rpm", 5000))
        fan_power_w = mcfg.get("fan_power_nominal_w", None)

        logger.info(
            "Machine %s : %d lignes, t_shutdown=%.1f°C, fan_max=%d RPM",
            machine_id, len(df_m), t_shutdown, fan_max_rpm,
        )

        # Appliquer le pipeline de features
        df_m = add_temporal_features(df_m, t_shutdown_c=t_shutdown)
        df_m = add_contextual_features(df_m, t_shutdown_c=t_shutdown)
        df_m = add_energy_features(df_m, fan_max_rpm=fan_max_rpm, fan_power_nominal_w=fan_power_w)

        if add_labels:
            df_m = add_failure_labels(df_m, t_shutdown_c=t_shutdown)
            df_m = add_control_labels(df_m, t_shutdown_c=t_shutdown)

        # Supprimer les lignes de chauffe (fenêtres glissantes non pleines)
        if drop_warmup_rows > 0 and len(df_m) > drop_warmup_rows:
            df_m = df_m.iloc[drop_warmup_rows:].reset_index(drop=True)

        results.append(df_m)

    if not results:
        return pd.DataFrame()

    df_out = pd.concat(results, ignore_index=True)

    # Supprimer les lignes avec NaN sur les features critiques
    before = len(df_out)
    df_out = df_out.dropna(subset=[c for c in _CRITICAL_FEATURES if c in df_out.columns])
    dropped = before - len(df_out)
    if dropped > 0:
        logger.info("Supprimé %d lignes avec NaN sur features critiques.", dropped)

    logger.info(
        "Dataset final : %d lignes × %d colonnes.", len(df_out), len(df_out.columns)
    )
    return df_out


def all_feature_names() -> list[str]:
    """Retourne la liste complète des features produites par le pipeline."""
    return (
        feature_names_temporal()
        + feature_names_contextual()
        + feature_names_energy()
    )


def all_label_names() -> list[str]:
    """Retourne la liste complète des labels produits."""
    return label_names_failure() + label_names_control()


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Pipeline feature engineering — Juste des Ventilateurs"
    )
    parser.add_argument("--input", required=True,
                        help="Répertoire Parquet d'entrée (ex: data/raw/episode=001)")
    parser.add_argument("--output", required=True,
                        help="Répertoire de sortie (ex: data/processed/episode=001)")
    parser.add_argument("--config", default=None,
                        help="Chemin vers metadata.json avec config thermique par machine")
    parser.add_argument("--no-labels", action="store_true",
                        help="Ne pas calculer les labels (mode inférence)")
    args = parser.parse_args()

    # Charger la config thermique depuis metadata.json si disponible
    machine_config: dict = {}
    if args.config:
        with open(args.config) as f:
            meta = json.load(f)
            machine_config = meta.get("machines", {})

    # Charger toutes les partitions Parquet de l'épisode
    input_path = Path(args.input)
    parquet_files = list(input_path.rglob("*.parquet"))
    if not parquet_files:
        logger.error("Aucun fichier Parquet trouvé dans %s", input_path)
        return

    logger.info("Chargement de %d fichiers Parquet...", len(parquet_files))
    df_raw = pd.concat(
        [pd.read_parquet(f) for f in parquet_files],
        ignore_index=True,
    )
    logger.info("Dataset brut : %d lignes × %d colonnes", len(df_raw), len(df_raw.columns))

    # Construire le dataset de features
    df_feat = build_feature_dataset(
        df_raw,
        machine_config=machine_config,
        add_labels=not args.no_labels,
    )

    # Sauvegarder
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / "features.parquet"
    df_feat.to_parquet(out_file, index=False, compression="snappy")
    logger.info("Features sauvegardées : %s", out_file)

    # Résumé des colonnes
    feature_cols = [c for c in df_feat.columns if c in all_feature_names()]
    label_cols = [c for c in df_feat.columns if c in all_label_names()]
    logger.info("Features : %d | Labels : %d", len(feature_cols), len(label_cols))


if __name__ == "__main__":
    main()
