from __future__ import annotations

import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

import config
from e3_chain import BASE_SPECS, CHAIN_SPECS, E3Quantizer, add_chain_features
from generator import generate_transactions


def test_chain_features_are_window_aggregates_and_quantize_without_raw_ids():
    transactions, windows = generate_transactions(n_transactions=800, seed=config.SEED)
    engineered = add_chain_features(transactions, windows)

    assert len(engineered) == len(windows)
    assert set(CHAIN_SPECS).issubset(engineered.columns)
    assert "counterparty_id" not in E3Quantizer(include_chain=True).feature_order
    assert "counterparty_id" not in E3Quantizer(include_chain=False).feature_order
    assert engineered["observed_hop_count"].between(1, 10).all()
    assert engineered["counterparty_shared_ratio"].between(0, 1).all()
    assert engineered["amount_retention_median"].between(0, 1).all()

    train = engineered.iloc[:60]
    quantizer = E3Quantizer(include_chain=True).fit(train)
    tokens = quantizer.transform(engineered.iloc[60:])
    assert tokens.shape == (len(engineered) - 60, len(BASE_SPECS) + len(CHAIN_SPECS))
    assert np.issubdtype(tokens.dtype, np.integer)
