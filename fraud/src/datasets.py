"""Data loading + splitting + sampling for the fraud sub-project.

Two data scales:
  * SMALL (bundled, fraud/data/User0_*.csv): ~20K rows, 1 user. Quick dev.
  * BIG (in place, ~/Downloads/archive/credit_card_transactions-ibm_v2.csv):
    24.4M rows, 2000 users, 29,757 fraud. The real dataset.

Splits use GroupKFold on the User identity so the same user never spans
train/val/test (avoids the classic fraud-detection leakage where a user's
fraud and normal transactions land on opposite sides of a random split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config


def load_small() -> pd.DataFrame:
    return pd.read_csv(config.SMALL_CSV)


def load_big(n_rows: int | None = None) -> pd.DataFrame:
    """Load the full 24.4M-row dataset (or the first n_rows for a quick check)."""
    return pd.read_csv(config.BIG_CSV, nrows=n_rows)


def get_label(df: pd.DataFrame) -> np.ndarray:
    """Is Fraud? Yes/No -> 1/0 int array."""
    return (df[config.LABEL_COL].astype(str).str.strip().str.lower() == "yes").to_numpy(dtype=np.int64)


def get_groups(df: pd.DataFrame) -> np.ndarray:
    return df[config.GROUP_COL].to_numpy()


def split_by_user(df: pd.DataFrame, n_splits: int = 5, seed: int = config.SEED):
    """Return (train_idx, val_idx, test_idx) as position arrays, grouped by User.

    fold 0 -> test, fold 1 -> val, remaining folds -> train (so train is large).
    Users (groups) are permuted with a seeded RNG before folding for determinism.
    """
    groups = get_groups(df)
    uniq, _ = np.unique(groups, return_inverse=True)
    n_groups = len(uniq)

    # Fallback: not enough distinct groups for a group split (e.g. the bundled
    # single-user User0 file) -> stratified random split instead.
    if n_groups < max(3, n_splits):
        from sklearn.model_selection import train_test_split
        y = get_label(df)
        idx = np.arange(len(df))
        valtest, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        train, val = train_test_split(valtest, test_size=0.25,
                                      stratify=y[valtest], random_state=seed)
        return (np.sort(train), np.sort(val), np.sort(test))

    gkf = GroupKFold(n_splits=n_splits)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)
    folds = list(gkf.split(np.arange(n_groups), groups=uniq[perm]))
    test_g = set(uniq[perm][folds[0][1]].tolist())
    val_g = set(uniq[perm][folds[1][1]].tolist())
    mask_test = np.isin(groups, list(test_g))
    mask_val = np.isin(groups, list(val_g))
    mask_train = ~(mask_test | mask_val)
    return (np.flatnonzero(mask_train), np.flatnonzero(mask_val), np.flatnonzero(mask_test))


def enrich_sample(df: pd.DataFrame, n_total: int, fraud_frac: float = 0.5,
                  seed: int = config.SEED) -> np.ndarray:
    """Position indices of a class-enriched sample.

    Takes (as many as exist) fraud rows + a random fraction of non-fraud rows so
    the sample has ~`fraud_frac` fraud. Balanced-ish sampling gives the slow
    QUANN 0th-order search a strong, learnable reward signal. (Probabilities on
    such a sample are not base-rate calibrated — evaluate on a base-rate set.)
    """
    y = get_label(df)
    fraud_idx = np.flatnonzero(y == 1)
    normal_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    n_fraud_target = int(round(n_total * fraud_frac))
    n_fraud = min(n_fraud_target, len(fraud_idx))
    n_normal = min(n_total - n_fraud, len(normal_idx))
    sel_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
    sel_normal = rng.choice(normal_idx, size=n_normal, replace=False)
    sel = np.concatenate([sel_fraud, sel_normal])
    rng.shuffle(sel)
    return sel


def base_rate_sample(df: pd.DataFrame, n_total: int, seed: int = config.SEED) -> np.ndarray:
    """A random sample at the NATURAL base rate (for honest evaluation)."""
    rng = np.random.default_rng(seed)
    return rng.choice(len(df), size=min(n_total, len(df)), replace=False)
