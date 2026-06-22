"""Tests de validation Phase 9b — Méthodologie anti-fuite.

Couvre les corrections d'intégrité du data engineering :
  1. Embargo temporel aux frontières de split (labels forward-looking)
  2. Exclusion optionnelle des features dérivées du statut (anti-circularité)
  3. Métriques anticipatoires (recall/precision sur machines encore saines)

Usage :
    pytest tests/test_phase9b_methodology.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.failure_prediction.splitter import (
    TemporalSplitter,
    STATUS_DERIVED_COLS,
    _apply_embargo,
)


# ---------------------------------------------------------------------------
# Fixture : épisode processed synthétique sur disque
# ---------------------------------------------------------------------------

def _make_episode(tmp_path: Path, n: int = 100) -> Path:
    """Crée data/processed/episode=001/features.parquet synthétique (1 machine)."""
    ts = pd.date_range("2025-01-01", periods=n, freq="1s", tz="UTC")
    df = pd.DataFrame({
        "timestamp":   ts,
        "machine_id":  "m0",
        "status":      ["on"] * n,
        "temperature_c": np.linspace(60, 85, n),
        "is_on":       1,
        "is_degraded": 0,
        "is_off":      0,
        "nb_shutdowns_episode": 0,
        "failure_60s": ([0] * (n - 10)) + ([1] * 10),  # pannes en fin d'épisode
    })
    ep = tmp_path / "episode=001"
    ep.mkdir(parents=True)
    df.to_parquet(ep / "features.parquet")
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Embargo
# ---------------------------------------------------------------------------

class TestApplyEmbargo:

    def _seg(self, n=20):
        ts = pd.date_range("2025-01-01", periods=n, freq="1s", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "v": range(n)})

    def test_drops_last_seconds(self):
        seg = self._seg(20)              # ts 0..19s
        out = _apply_embargo(seg, embargo_s=5)
        # garde ts <= max - 5 = 14s -> 15 lignes (0..14)
        assert len(out) == 15

    def test_zero_embargo_noop(self):
        seg = self._seg(20)
        assert len(_apply_embargo(seg, 0)) == 20

    def test_empty_segment(self):
        out = _apply_embargo(pd.DataFrame(), 5)
        assert out.empty

    def test_no_timestamp_noop(self):
        seg = pd.DataFrame({"v": range(5)})
        assert len(_apply_embargo(seg, 5)) == 5


class TestSplitterEmbargo:

    def test_test_set_unchanged_by_embargo(self, tmp_path):
        d = _make_episode(tmp_path, n=100)
        sp0 = TemporalSplitter(processed_dir=str(d))
        *_, X_test0, _, _, y_test0 = sp0.split(label_col="failure_60s", embargo_s=0)
        sp1 = TemporalSplitter(processed_dir=str(d))
        *_, X_test1, _, _, y_test1 = sp1.split(label_col="failure_60s", embargo_s=10)
        # Le test n'est jamais tronqué par l'embargo
        assert len(X_test0) == len(X_test1)

    def test_embargo_shrinks_train(self, tmp_path):
        d = _make_episode(tmp_path, n=100)
        sp0 = TemporalSplitter(processed_dir=str(d))
        X_tr0, *_ = sp0.split(label_col="failure_60s", embargo_s=0)
        sp1 = TemporalSplitter(processed_dir=str(d))
        X_tr1, *_ = sp1.split(label_col="failure_60s", embargo_s=10)
        assert len(X_tr1) < len(X_tr0)

    def test_no_temporal_overlap_after_embargo(self, tmp_path):
        d = _make_episode(tmp_path, n=100)
        sp = TemporalSplitter(processed_dir=str(d))
        df_tr, df_val, df_te = sp.split_with_meta(label_col="failure_60s", embargo_s=10)
        # gap d'au moins 10s entre fin du train et début de la val
        gap = (df_val["timestamp"].min() - df_tr["timestamp"].max()).total_seconds()
        assert gap >= 10


# ---------------------------------------------------------------------------
# 2. Exclusion des features de statut
# ---------------------------------------------------------------------------

class TestExcludeStatusFeatures:

    def test_status_features_present_by_default(self, tmp_path):
        d = _make_episode(tmp_path, n=100)
        sp = TemporalSplitter(processed_dir=str(d))
        sp.split(label_col="failure_60s")
        assert "is_degraded" in sp.feature_cols

    def test_status_features_excluded_on_request(self, tmp_path):
        d = _make_episode(tmp_path, n=100)
        sp = TemporalSplitter(processed_dir=str(d))
        sp.split(label_col="failure_60s", extra_exclude=STATUS_DERIVED_COLS)
        for c in ("is_degraded", "is_off", "nb_shutdowns_episode"):
            assert c not in sp.feature_cols
        # une feature non-statut reste présente
        assert "temperature_c" in sp.feature_cols


# ---------------------------------------------------------------------------
# 3. Métriques anticipatoires
# ---------------------------------------------------------------------------

class TestAnticipatoryMetrics:

    def _fn(self):
        from evaluation.failure_prediction_eval import _anticipatory_metrics
        return _anticipatory_metrics

    def test_restricts_to_status_on(self):
        f = self._fn()
        # 4 lignes : 2 saines (on), 2 déjà off. Le modèle rate le positif sain.
        y_true = pd.Series([1, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        meta = pd.DataFrame({"status": ["on", "on", "off", "off"]})
        a = f(y_true, y_pred, meta)
        assert a["n"] == 2 and a["n_pos"] == 1
        assert a["recall"] == 0.0          # le seul positif sain est raté

    def test_uses_is_on_column(self):
        f = self._fn()
        y_true = pd.Series([1, 1])
        y_pred = np.array([1, 1])
        meta = pd.DataFrame({"is_on": [1, 1]})
        a = f(y_true, y_pred, meta)
        assert a["recall"] == 1.0 and a["n_pos"] == 2

    def test_none_when_no_status_col(self):
        f = self._fn()
        assert f(pd.Series([1]), np.array([1]), pd.DataFrame({"x": [0]})) is None
