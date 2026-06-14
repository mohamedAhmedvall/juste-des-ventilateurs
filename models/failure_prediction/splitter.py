"""
splitter.py — Split temporel Option A (70/15/15) par épisode.

Stratégie :
  - Chaque épisode est coupé chronologiquement en train/val/test
  - Les morceaux sont concaténés : X_train = concat(train_ep001, ..., train_ep006)
  - Aucun leakage temporel (on ne voit jamais le futur pendant l'entraînement)

Usage :
    from models.failure_prediction.splitter import TemporalSplitter
    splitter = TemporalSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
        data_dir="data/processed", label_col="failure_60s"
    )
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Colonnes à exclure des features ML
NON_FEATURE_COLS = {
    "timestamp", "cluster_id", "machine_id", "role", "msg_type", "status",
    "fault_types", "fan_modes",
    # Labels (ne pas leaker)
    "failure_30s", "failure_60s", "hot_30s", "time_to_failure_s",
    "optimal_rpm", "action_class", "action_class_v2",
    # Colonnes 100% NaN observées
    "machines_total", "machines_on",
}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15 (implicite)


class TemporalSplitter:
    """Split temporel 70/15/15 par épisode, concaténation globale.

    Parameters
    ----------
    train_ratio : fraction pour le train (défaut 0.70)
    val_ratio   : fraction pour la validation (défaut 0.15)
    processed_dir : répertoire contenant les episode=* processed
    """

    def __init__(
        self,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        processed_dir: str = "data/processed",
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.processed_dir = Path(processed_dir)
        self._feature_cols: list[str] | None = None

    @property
    def feature_cols(self) -> list[str]:
        if self._feature_cols is None:
            raise RuntimeError("Appeler split() avant d'accéder à feature_cols.")
        return self._feature_cols

    def split(
        self,
        label_col: str = "failure_60s",
        episode_ids: list[str] | None = None,
        drop_na_label: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series]:
        """Charge les épisodes processed et retourne les splits (X_train, X_val, X_test,
        y_train, y_val, y_test).

        Parameters
        ----------
        label_col      : colonne cible ('failure_60s', 'failure_30s', 'hot_30s', 'action_class')
        episode_ids    : liste d'IDs à charger (None = tous)
        drop_na_label  : supprimer les lignes où le label est NaN
        """
        ep_dirs = self._list_episodes(episode_ids)
        if not ep_dirs:
            raise FileNotFoundError(
                f"Aucun épisode trouvé dans {self.processed_dir}. "
                "Lance ingest_gen_features.bat d'abord."
            )

        trains, vals, tests = [], [], []

        for ep_id, ep_dir in ep_dirs:
            df = self._load_episode(ep_dir)
            if df.empty:
                logger.warning("Episode %s : vide, ignoré.", ep_id)
                continue
            if label_col not in df.columns:
                logger.warning("Episode %s : colonne '%s' absente, ignoré.", ep_id, label_col)
                continue

            # Tri chronologique
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)

            if drop_na_label:
                df = df.dropna(subset=[label_col]).reset_index(drop=True)

            n = len(df)
            n_train = int(n * self.train_ratio)
            n_val   = int(n * self.val_ratio)

            trains.append(df.iloc[:n_train])
            vals.append(df.iloc[n_train:n_train + n_val])
            tests.append(df.iloc[n_train + n_val:])

            pos_train = (df.iloc[:n_train][label_col] == 1).sum()
            pos_test  = (df.iloc[n_train + n_val:][label_col] == 1).sum()
            logger.info(
                "Episode %s : %d lignes → train=%d (pos=%.1f%%)  "
                "val=%d  test=%d (pos=%.1f%%)",
                ep_id, n, n_train,
                100 * pos_train / max(n_train, 1),
                n_val,
                n - n_train - n_val,
                100 * pos_test / max(n - n_train - n_val, 1),
            )

        if not trains:
            raise ValueError(f"Aucune donnée valide trouvée pour le label '{label_col}'.")

        df_train = pd.concat(trains, ignore_index=True)
        df_val   = pd.concat(vals,   ignore_index=True)
        df_test  = pd.concat(tests,  ignore_index=True)

        # Déterminer les colonnes features
        all_cols = set(df_train.columns)
        self._feature_cols = sorted(
            c for c in all_cols
            if c not in NON_FEATURE_COLS
            and df_train[c].dtype in [np.float64, np.float32, np.int64, np.int32, bool]
        )

        logger.info(
            "Split final — train: %d  val: %d  test: %d  features: %d",
            len(df_train), len(df_val), len(df_test), len(self._feature_cols),
        )

        X_train = df_train[self._feature_cols]
        X_val   = df_val[self._feature_cols]
        X_test  = df_test[self._feature_cols]
        y_train = df_train[label_col].astype(int)
        y_val   = df_val[label_col].astype(int)
        y_test  = df_test[label_col].astype(int)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def split_with_meta(
        self,
        label_col: str = "failure_60s",
        episode_ids: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Comme split() mais retourne les DataFrames complets (avec metadata)
        pour le calcul du lead time et l'analyse post-hoc."""
        ep_dirs = self._list_episodes(episode_ids)
        trains, vals, tests = [], [], []

        for ep_id, ep_dir in ep_dirs:
            df = self._load_episode(ep_dir)
            if df.empty or label_col not in df.columns:
                continue
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)
            df = df.dropna(subset=[label_col]).reset_index(drop=True)
            n = len(df)
            n_train = int(n * self.train_ratio)
            n_val   = int(n * self.val_ratio)
            trains.append(df.iloc[:n_train])
            vals.append(df.iloc[n_train:n_train + n_val])
            tests.append(df.iloc[n_train + n_val:])

        return (
            pd.concat(trains, ignore_index=True) if trains else pd.DataFrame(),
            pd.concat(vals,   ignore_index=True) if vals   else pd.DataFrame(),
            pd.concat(tests,  ignore_index=True) if tests  else pd.DataFrame(),
        )

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _list_episodes(
        self, episode_ids: list[str] | None
    ) -> list[tuple[str, Path]]:
        result = []
        for p in sorted(self.processed_dir.glob("episode=*")):
            if not p.is_dir():
                continue
            ep_id = p.name[len("episode="):]
            if not ep_id or ep_id == "data":
                continue
            if episode_ids is not None and ep_id not in episode_ids:
                continue
            result.append((ep_id, p))
        return result

    def _load_episode(self, ep_dir: Path) -> pd.DataFrame:
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
