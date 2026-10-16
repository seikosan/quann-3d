"""Leakage-safe feature-to-token quantizer for AML account windows."""
from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureQuantizer:
    """Map one account window to one token per feature, fit on train only."""

    SPECS = {
        "amount_median": ("quantile", 8),
        "amount_p95": ("quantile", 8),
        "threshold_share": ("quantile", 5),
        "time_of_day_hour": ("quantile", 8),
        "inter_tx_interval_min": ("quantile", 8),
        "counterparty_count": ("quantile", 6),
        "counterparty_id": ("cat", None),
        "channel": ("cat", None),
        "window_count": ("quantile", 6),
        "rate_per_hour": ("quantile", 8),
        "chain_hops": ("quantile", 6),
        "window_duration_min": ("quantile", 8),
    }

    def __init__(self):
        self.feature_order = list(self.SPECS)
        self.offsets, self.bins, self.cat_maps, self.n_bins = {}, {}, {}, {}
        self.vocab_size = 2
        self._fitted = False

    def fit(self, frame: pd.DataFrame):
        offset = 2
        for feature, (kind, n) in self.SPECS.items():
            if kind == "quantile":
                values = pd.to_numeric(frame[feature], errors="coerce").to_numpy()
                edges = np.unique(np.quantile(values[np.isfinite(values)], np.linspace(0, 1, n + 1)[1:-1]))
                self.bins[feature] = edges
                count = len(edges) + 1
            else:
                categories = sorted(frame[feature].astype(str).unique())
                self.cat_maps[feature] = {value: i for i, value in enumerate(categories)}
                count = len(categories) + 1
            self.offsets[feature] = offset
            self.n_bins[feature] = count
            offset += count
        self.vocab_size = offset
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit the feature quantizer before transform")
        cols = []
        for feature, (kind, _) in self.SPECS.items():
            if kind == "quantile":
                values = pd.to_numeric(frame[feature], errors="coerce").to_numpy()
                bucket = np.digitize(np.nan_to_num(values, nan=0.0), self.bins[feature])
            else:
                cmap = self.cat_maps[feature]
                bucket = np.asarray([cmap.get(str(value), len(cmap)) for value in frame[feature]])
            cols.append(bucket.astype(np.int64) + self.offsets[feature])
        return np.stack(cols, axis=1)
