from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

import config
from e3_chain import BASE_SPECS, CHAIN_SPECS, E3Quantizer
from e3b_chain import PROBE_COUNTS, _probe_report, _split
from generator_hard import CLASS_SHARES, generate_hard_transactions


def test_hard_generator_is_deterministic_imbalanced_noisy_and_observed_hops():
    tx_a, windows_a = generate_hard_transactions(n_transactions=4_000, seed=731)
    tx_b, windows_b = generate_hard_transactions(n_transactions=4_000, seed=731)

    pd.testing.assert_frame_equal(tx_a, tx_b)
    pd.testing.assert_frame_equal(windows_a, windows_b)
    counts = windows_a.typology.value_counts(normalize=True)
    assert counts["normal"] == CLASS_SHARES["normal"]
    assert set(windows_a.typology.unique()) == set(config.CLASSES)
    observed_counts = tx_a.groupby("window_id").counterparty_id.nunique().sort_index().to_numpy()
    assert np.array_equal(observed_counts, windows_a.sort_values("window_id").chain_hops.to_numpy())
    assert windows_a.chain_hops.between(1, 10).all()
    assert tx_a.amount.between(1, 12_000).all()
    assert not np.allclose(windows_a.amount_median, windows_a.amount_p95)


def test_idless_quantizer_changes_only_by_the_chain_feature_family():
    tx, windows = generate_hard_transactions(n_transactions=4_000, seed=config.SEED)
    from e3_chain import add_chain_features

    windows = add_chain_features(tx, windows)
    base = E3Quantizer(include_chain=False)
    chain = E3Quantizer(include_chain=True)
    assert base.feature_order == list(BASE_SPECS)
    assert chain.feature_order == list(BASE_SPECS) + list(CHAIN_SPECS)
    assert "counterparty_id" not in base.feature_order
    assert "counterparty_id" not in chain.feature_order
    fitted = chain.fit(windows.iloc[:120])
    tokens = fitted.transform(windows.iloc[120:])
    assert tokens.shape == (len(windows) - 120, len(BASE_SPECS) + len(CHAIN_SPECS))


def test_e3b_split_matches_e0_seeded_account_window_split():
    labels = np.repeat(np.arange(4), 50)
    train, temp = train_test_split(np.arange(len(labels)), test_size=0.4, random_state=config.SEED, stratify=labels)
    val, test = train_test_split(temp, test_size=0.5, random_state=config.SEED + 1, stratify=labels[temp])
    actual = _split(labels)
    assert np.array_equal(actual[0], train)
    assert np.array_equal(actual[1], val)
    assert np.array_equal(actual[2], test)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))


def test_sar_probe_sanity_reports_observed_reasons_distances_and_non_determination():
    windows = pd.DataFrame({
        "amount_p95": [9500.0, 800.0, 1500.0, 100.0],
        "threshold_share": [0.8, 0.0, 0.0, 0.0],
        "inter_tx_interval_min": [20.0, 30.0, 24.0, 0.3],
        "rate_per_hour": [3.0, 2.0, 2.5, 120.0],
        "window_count": [3, 2, 3, 10],
        "chain_hops": [4, 7, 9, 3],
        "counterparty_count": [4, 7, 9, 3],
        "channel": ["cash", "wire", "wire", "digital"],
    })
    labels = np.arange(4)
    prototypes = {name: np.eye(4, dtype=float)[i] for i, name in enumerate(config.CLASSES)}
    probs = np.eye(4, dtype=float)
    probes = _probe_report(windows, labels, np.arange(4), probs, np.eye(4), prototypes)

    assert len(probes) == len(config.CLASSES)
    assert all(probe["observable_reason"] for probe in probes)
    assert all(probe["nearest_prototype_distance"] >= 0 for probe in probes)
    assert all("SAR kararı değildir" in probe["sar_draft"] for probe in probes)
    assert set(PROBE_COUNTS) == set(config.CLASSES)
