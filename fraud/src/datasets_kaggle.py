"""Kaggle cc-fraud dataset — the public benchmark branch of the fraud pipeline.

Data: the 2013 European cardholder set (Kaggle ``creditcardfraud`` /
``ealtman2019/credit-card-transactions``): 284,807 transactions, 492 fraud
(0.17%), features Time (seconds since 01/01/2013 00:00), V1-V28 (PCA-
transformed, anonymized), Amount, Class.

Why a separate branch instead of extending the IBM one:
  * no User/identity column -> no GroupKFold; splits are stratified random.
  * V1-V28 are PCA-processed values -> the plain-text explanation wedge of the
    IBM vertical degrades here ("V13 in bucket 7" is not a readable rationale);
    this branch is the *paper* benchmark, not the product dataset.
  * the IBM feature/quantizer specs (features.derive_base / QUANT_SPECS) are
    frozen against existing checkpoints; this branch defines its own.

Role of the features:
  * V1-V28 stay verbatim (they are the usable signal; each = 1 cloud point).
  * Time -> hour / dayofweek / is_weekend (cyclical calendar derived, cheap).
  * Amount -> quantile buckets directly (the distribution is skewed; the
    quantile kind handles it, no log transform needed for bucketing).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

KAGGLE_CSV = os.path.join(config.DATA_DIR_KAGGLE, "creditcard.csv")

LABEL_COL = "Class"
TIME_EPOCH = pd.Timestamp("2013-01-01 00:00:00")


def load_kaggle(n_rows: int | None = None, path: str | None = None) -> pd.DataFrame:
    p = path or KAGGLE_CSV
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Kaggle cc-fraud csv not found at {p}. "
            "Download it (Kaggle: machinelearninginvite/creditcardfraud) and place "
            "the unzipped creditcard.csv in data/kaggle_cc/."
        )
    return pd.read_csv(p, nrows=n_rows)


def get_label(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL_COL].to_numpy(dtype=np.int64)


def derive_base_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap derived calendar features for the cloud quantizer (no fitting)."""
    out = df.copy()
    t = pd.to_numeric(out[config.KAGGLE_TIME_COL], errors="coerce").fillna(0.0)
    dt = TIME_EPOCH + pd.to_timedelta(t, unit="s")
    out["hour"] = dt.dt.hour.fillna(12).astype(np.int16)
    dow = dt.dt.dayofweek  # 0=Mon .. 6=Sun
    out["dayofweek"] = dow.fillna(0).astype(np.int16)
    out["is_weekend"] = (dow.fillna(0) >= 5).astype(np.int8)
    return out


def split_kaggle(
    df: pd.DataFrame,
    n_train: int,
    n_val: int,
    n_test: int,
    train_frac: float,
    test_frac: float,
    seed: int = config.SEED,
):
    """Deterministic stratified random split (no identity grouping available).

    Mirrors train_quann.build_data's sampling contract: train/val are
    class-enriched to ~train_frac, test keeps the natural base rate at
    test_frac enrichment. Returns (tr_sel, va_sel, te_sel) position arrays.
    """
    y = get_label(df)
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)

    valtest, test_pool = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    train_pool, val_pool = train_test_split(valtest, test_size=0.25, stratify=y[valtest],
                                            random_state=seed)

    def _sample(pool: np.ndarray, n_total: int, frac: float, s: int) -> np.ndarray:
        sub = y[pool]
        f = pool[sub == 1]
        n_ = pool[sub == 0]
        rng2 = np.random.default_rng(s)
        n_fraud = min(int(round(n_total * frac)), len(f))
        n_norm = min(n_total - n_fraud, len(n_))
        sel = np.concatenate([rng2.choice(f, size=n_fraud, replace=False),
                              rng2.choice(n_, size=n_norm, replace=False)])
        rng2.shuffle(sel)
        return sel

    tr_sel = _sample(train_pool, n_train, train_frac, seed)
    va_sel = _sample(val_pool, n_val, train_frac, seed + 1)
    te_sel = _sample(test_pool, n_test, test_frac, seed + 2)
    return np.sort(tr_sel), np.sort(va_sel), np.sort(te_sel)
