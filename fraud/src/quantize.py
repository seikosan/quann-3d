"""FeatureQuantizer — the point-cloud INPUT for the QUANN fraud layer.

The QUANN cloud representation is identity-based: one D-dim point per discrete
token identity. A tabular feature has many *values*, so we QUANTIZE each feature
into buckets and let the token identity = (feature, bucket). A transaction then
becomes a fixed-size point cloud of K points (one per feature), which the
permutation-invariant CloudEncoderND reads as geometry.

This is the "feature -> point, transaction -> cloud" mapping (the research's
signature), shared by all three QUANN training configs so they are compared on
identical inputs.

Fitting (bucket edges / top-N categories) happens on the TRAIN split only, so
val/test quantization is leakage-safe — same principle as FeatureTransformer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config  # noqa: F401  (also sets up repo-root import path)
from features import derive_base, _slug

# Quantization spec: (feature_key, kind, n)
#   kind 'quantile'  : n quantile bins of the numeric column (missing -> extra bucket)
#   kind 'int'       : column already an integer in [0, n) -> bucket = value
#   kind 'cat'       : every distinct category gets a bucket (+ 'other' for unseen)
#   kind 'cat_topn'  : top-n most-frequent categories + 'other'
# feature_key is a column name in derive_base() output (or a raw column for cat*).
QUANT_SPECS = [
    ("amount",         "quantile", 8),
    ("hour",           "int", 24),
    ("month",          "int", 12),
    ("dayofweek",      "int", 7),
    ("is_weekend",     "int", 2),
    ("year",           "quantile", 4),
    ("zip",            "quantile", 16),          # +1 missing bucket at fit
    ("Use Chip",       "cat", None),             # 3
    ("MCC",            "cat_topn", 32),
    ("Merchant State", "cat_topn", 32),
    ("Merchant City",  "cat_topn", 32),
    ("Merchant Name",  "cat_topn", 32),
    ("error_type",     "cat", None),             # none/multi/~7 = 9
]
LEGACY_QUANT_SPECS = list(QUANT_SPECS)

# Rich-point experiment additions. Keep the original 13 entries above byte-for-
# byte stable: their token offsets are part of existing QUANN checkpoint data.
# These are all outputs of features.derive_base(); no raw-column features are
# engineered here. Current nominal vocabulary growth is only 28 buckets.
QUANT_SPECS += [
    ("amount_abs_log", "quantile", 16),
    ("amount_neg",    "int", 2),
    ("zip_missing",   "int", 2),
    ("has_error",     "int", 2),
    ("err_bad_pin",   "int", 2),
    ("err_insufficient_balance", "int", 2),
    ("err_bad_zipcode", "int", 2),
]


class FeatureQuantizer:
    """Map a raw frame to an (N, K) int64 token matrix (1 token per feature)."""

    def __init__(self, specs=None):
        self.specs = list(specs or QUANT_SPECS)
        self.feature_order = [s[0] for s in self.specs]
        self.offsets: dict[str, int] = {}
        self.n_per_feature: dict[str, int] = {}
        self.bins: dict[str, np.ndarray] = {}      # quantile edges
        self.cat_map: dict[str, dict] = {}         # cat value -> bucket
        self.vocab_size = 0
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, raw: pd.DataFrame) -> "FeatureQuantizer":
        base = derive_base(raw)
        offset = 2  # 0 = pad, 1 = eos (reserved, mirroring QCloudNetwork)
        for key, kind, n in self.specs:
            if kind == "quantile":
                s = base[key] if key in base.columns else raw[key]
                s = pd.to_numeric(s, errors="coerce")
                has_missing = bool(s.isna().any())
                edges = np.quantile(s.dropna().values, np.linspace(0, 1, n + 1)[1:-1])
                self.bins[key] = edges
                nb = n + (1 if has_missing else 0)
            elif kind == "int":
                self.bins[key] = None
                nb = n
            elif kind == "cat":
                vals = (base[key] if key in base.columns else raw[key]).astype(str)
                cats = sorted(vals.unique().tolist())
                self.cat_map[key] = {c: i for i, c in enumerate(cats)}
                self.bins[key] = None
                nb = len(cats) + 1  # + 'other'
            elif kind == "cat_topn":
                vals = (base[key] if key in base.columns else raw[key]).astype(str)
                top = list(vals.value_counts().head(n).index)
                self.cat_map[key] = {c: i for i, c in enumerate(top)}
                self.bins[key] = None
                nb = n + 1  # + 'other'
            else:
                raise ValueError(f"unknown quant kind: {kind}")
            self.offsets[key] = offset
            self.n_per_feature[key] = nb
            offset += nb
        self.vocab_size = offset
        self._fitted = True
        return self

    # ------------------------------------------------------------- transform
    def _bucket(self, raw: pd.DataFrame, base: pd.DataFrame, key: str,
                kind: str, n: int) -> np.ndarray:
        s = base[key] if key in base.columns else raw[key]
        if kind == "quantile":
            s = pd.to_numeric(s, errors="coerce")
            b = np.digitize(s.values, self.bins[key])  # 0..n
            miss = s.isna().values
            b = b.astype(np.int64)
            if miss.any():
                b[miss] = self.n_per_feature[key] - 1  # dedicated missing bucket
            return b
        if kind == "int":
            return np.clip(s.astype("int64"), 0, n - 1).values
        # cat / cat_topn
        v = s.astype(str).values
        cmap = self.cat_map[key]
        out = np.full(len(v), self.n_per_feature[key] - 1, dtype=np.int64)  # 'other'
        for i, val in enumerate(v):
            if val in cmap:
                out[i] = cmap[val]
        return out

    def transform(self, raw: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureQuantizer.fit() must be called first")
        base = derive_base(raw)
        cols = []
        for key, kind, n in self.specs:
            b = self._bucket(raw, base, key, kind, n)
            cols.append(b + self.offsets[key])
        return np.stack(cols, axis=1).astype(np.int64)

    def fit_transform(self, raw: pd.DataFrame) -> np.ndarray:
        return self.fit(raw).transform(raw)

    @property
    def n_features(self) -> int:
        return len(self.specs)

    # ------------------------------------------------- (de)serialization
    def to_dict(self) -> dict:
        """JSON-able snapshot of the fit state (bins / cat maps / offsets).

        A QUANN cloud model is only reproducible & deployable together with the
        quantizer that produced its token ids — this snapshot is what makes the
        checkpoint self-contained (see qcloud_fraud.save_quann).
        """
        return {
            "specs": [list(s) for s in self.specs],
            "feature_order": list(self.feature_order),
            "offsets": {k: int(v) for k, v in self.offsets.items()},
            "n_per_feature": {k: int(v) for k, v in self.n_per_feature.items()},
            "bins": {k: (None if v is None else [float(x) for x in v])
                     for k, v in self.bins.items()},
            "cat_map": {k: dict(v) for k, v in self.cat_map.items()},
            "vocab_size": int(self.vocab_size),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureQuantizer":
        fq = cls(specs=[tuple(s) for s in d["specs"]])
        fq.feature_order = list(d["feature_order"])
        fq.offsets = {k: int(v) for k, v in d["offsets"].items()}
        fq.n_per_feature = {k: int(v) for k, v in d["n_per_feature"].items()}
        fq.bins = {k: (None if v is None else np.array(v, dtype=float))
                   for k, v in d["bins"].items()}
        fq.cat_map = {k: dict(v) for k, v in d["cat_map"].items()}
        fq.vocab_size = int(d["vocab_size"])
        fq._fitted = True
        return fq
