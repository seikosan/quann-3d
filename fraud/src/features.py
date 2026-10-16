"""Feature engineering for the IBM Credit Card Fraud dataset.

One shared, leakage-safe pipeline used by BOTH the standard-ML layer and the
QUANN 0th-order layer, so the two are compared on identical inputs.

Raw columns (IBM schema):
    User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name,
    Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?

Design:
  * Numeric features are parsed from the raw strings (Amount "$134.09" -> 134.09,
    Time "19:40" -> hour, etc.).
  * Low-cardinality categoricals (Use Chip, error types) are one-hot encoded.
  * High-cardinality categoricals (MCC, Merchant City/Name/State) are frequency
    encoded. The frequency table is fit on the TRAIN split only, then applied to
    val/test -> no target or membership leakage.

All transforms fit on the training rows only (see FeatureTransformer.fit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config  # same package (fraud/src)

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
# Raw categorical columns and how each is encoded.
ONEHOT_COLS = ["Use Chip"]                       # low-cardinality -> one-hot
ERR_COL = "Errors?"                             # multi-label -> exploded one-hot
FREQ_COLS = ["MCC", "Merchant City", "Merchant Name", "Merchant State"]  # high-card -> freq

# Numeric feature names produced by derive_base().
NUMERIC = [
    "amount", "amount_abs_log", "amount_neg",
    "hour", "hour_sin", "hour_cos",
    "month", "day", "dayofweek", "is_weekend", "year",
    "zip", "zip_missing",
]

ERR_TOKENS = [
    "Insufficient Balance", "Technical Glitch", "Bad PIN",
    "Bad Expiration", "Bad Card Number", "Bad CVV", "Bad Zipcode",
]


# ---------------------------------------------------------------------------
# Base derivation (cheap, vectorized, no fitting)
# ---------------------------------------------------------------------------
def _parse_amount(s: pd.Series) -> pd.Series:
    """'$134.09' / '$-373.00' -> float. Robust to odd prefixes."""
    return (
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
    )


def derive_base(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numeric/categorical columns to a copy of the raw frame."""
    out = df.copy()

    # --- money ---
    amt = _parse_amount(out[config.AMOUNT_COL])
    out["amount"] = amt
    out["amount_abs_log"] = np.log1p(amt.abs())
    out["amount_neg"] = (amt < 0).astype(np.int8)

    # --- time of day (cyclical) ---
    hh = out["Time"].astype(str).str[:2]
    out["hour"] = pd.to_numeric(hh, errors="coerce").fillna(12).astype(np.int16)
    ang = np.deg2rad(out["hour"] / 24.0 * 2 * np.pi)
    out["hour_sin"] = np.sin(ang)
    out["hour_cos"] = np.cos(ang)

    # --- calendar / day-of-week ---
    out["month"] = pd.to_numeric(out["Month"], errors="coerce")
    out["day"] = pd.to_numeric(out["Day"], errors="coerce")
    out["year"] = pd.to_numeric(out["Year"], errors="coerce")
    dt = pd.to_datetime(
        {
            "year": out["Year"].astype("int32").clip(1900, 2100),
            "month": out["Month"].astype("int32").clip(1, 12),
            "day": out["Day"].astype("int32").clip(1, 31),
        },
        errors="coerce",
    )
    dow = dt.dt.dayofweek  # 0=Mon .. 6=Sun  (Series -> .dt accessor)
    out["dayofweek"] = dow.fillna(0).astype(np.int16)
    out["is_weekend"] = (dow.fillna(0) >= 5).astype(np.int8)

    # --- zip (missing == online) ---
    z = pd.to_numeric(out["Zip"], errors="coerce")
    out["zip"] = z
    out["zip_missing"] = z.isna().astype(np.int8)
    out["zip"] = z.fillna(0.0)

    # --- merchant state: NaN (online) -> sentinel so freq-encode has a category
    out["Merchant State"] = out["Merchant State"].fillna("ONLINE")

    # --- error tokens (multi-label) ---
    has_err = out[ERR_COL].notna()
    out["has_error"] = has_err.astype(np.int8)
    for tok in ERR_TOKENS:
        out[f"err_{_slug(tok)}"] = out[ERR_COL].fillna("").str.contains(tok, regex=False).astype(np.int8)

    # --- single 'error_type' for the point-cloud quantizer (1 feature = 1 point) ---
    # primary error token, 'multi' when several, else 'none'
    err_str = out[ERR_COL].fillna("").astype(str)
    def _primary(s: str) -> str:
        s = s.strip()
        if not s:
            return "none"
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) > 1:
            return "multi"
        return _slug(parts[0]) if parts else "none"
    out["error_type"] = err_str.map(_primary)

    return out


def _slug(s: str) -> str:
    return s.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Fitting transformer (leakage-safe)
# ---------------------------------------------------------------------------
class FeatureTransformer:
    """Fit on the TRAIN split; transform any split into a float32 matrix."""

    def __init__(self, onehot_cols=None, freq_cols=None, numeric_cols=None,
                 err_tokens=None):
        self.onehot_cols = list(onehot_cols or ONEHOT_COLS)
        self.freq_cols = list(freq_cols or FREQ_COLS)
        self.numeric_cols = list(numeric_cols or NUMERIC)
        self.err_tokens = list(err_tokens or ERR_TOKENS)
        self._freq: dict[str, dict] = {}
        self._onehot_cats: dict[str, list] = {}
        self.feature_names: list[str] = []
        self._fitted = False

    # -- fit: learn encoding tables from training rows only ----------------
    def fit(self, raw: pd.DataFrame) -> "FeatureTransformer":
        base = derive_base(raw)
        n = len(base)
        for c in self.freq_cols:
            vc = base[c].astype(str).value_counts()
            self._freq[c] = {k: v / n for k, v in vc.items()}
        for c in self.onehot_cols:
            self._onehot_cats[c] = sorted(base[c].astype(str).unique().tolist())

        self.feature_names = (
            self.numeric_cols
            + [f"{_slug(c)}={cat}" for c in self.onehot_cols for cat in self._onehot_cats[c]]
            + [f"err_{_slug(t)}" for t in self.err_tokens]
            + [f"{_slug(c)}_freq" for c in self.freq_cols]
        )
        self._fitted = True
        return self

    # -- transform: encode any split using the fit tables ------------------
    def transform(self, raw: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureTransformer.fit() must be called first")
        base = derive_base(raw)
        n = len(base)
        cols: list[pd.Series] = [base[c] for c in self.numeric_cols]

        for c in self.onehot_cols:
            s = base[c].astype(str)
            for cat in self._onehot_cats[c]:
                cols.append((s == cat).astype(np.float32))

        for t in self.err_tokens:
            cols.append(base[f"err_{_slug(t)}"].astype(np.float32))

        for c in self.freq_cols:
            table = self._freq[c]
            cols.append(base[c].astype(str).map(table).fillna(0.0).astype(np.float32))

        mat = np.stack([c.values if hasattr(c, "values") else np.asarray(c) for c in cols], axis=1)
        return mat.astype(np.float32, copy=False)

    def fit_transform(self, raw: pd.DataFrame) -> np.ndarray:
        return self.fit(raw).transform(raw)

    def __len__(self) -> int:
        return len(self.feature_names)


def feature_names(transformer: FeatureTransformer) -> list[str]:
    return transformer.feature_names
