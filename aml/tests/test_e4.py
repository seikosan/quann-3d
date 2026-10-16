"""Focused invariants for the E4 synthetic drift generator and runner."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from generator_weeks import generate_adaptation_batch, generate_week0, generate_week_stream
from e4_adapt import E4Quantizer


def test_week0_and_post_change_have_expected_threshold_bands_and_determinism():
    week0 = generate_week0(n_per_class=24, seed=17)
    same_week0 = generate_week0(n_per_class=24, seed=17)
    post_change = generate_week_stream(1, n_per_class=40, seed=17, edge_fraction=0.5)

    assert week0.equals(same_week0)
    old = week0[week0.typology == "structuring"].amount_p95
    new = post_change[(post_change.typology == "structuring") & (post_change.edge_regime == "5k")].amount_p95
    legacy = post_change[(post_change.typology == "structuring") & (post_change.edge_regime == "10k")].amount_p95
    assert old.between(9_000, 10_000).all()
    assert new.between(4_000, 5_000).all()
    assert legacy.between(9_000, 10_000).all()
    assert (post_change.loc[new.index, "channel"] == "card").all()
    assert (post_change.loc[legacy.index, "channel"] == "cash_deposit").all()
    assert (post_change.week == 1).all()


def test_week0_quantizer_exposes_unseen_5k_amount_tokens():
    week0 = generate_week0(n_per_class=80, seed=29)
    quantizer = E4Quantizer().fit(week0.iloc[:240])
    week0_tokens = set(quantizer.transform(week0).reshape(-1).tolist())
    post_change = generate_week_stream(1, n_per_class=40, seed=29, edge_fraction=1.0)
    new_structuring = post_change[(post_change.typology == "structuring") & (post_change.edge_regime == "5k")]
    new_tokens = quantizer.transform(new_structuring)
    novel = set(new_tokens.reshape(-1).tolist()) - week0_tokens

    assert len(new_structuring) == 40
    assert novel, "the 5K regulatory shape should create at least one week0-unseen quantized feature token"
    assert np.all(new_tokens[:, quantizer.feature_order.index("amount_median")] ==
                  new_tokens[:, quantizer.feature_order.index("amount_p95")])


def test_adaptation_batch_is_fixed_size_balanced_and_contains_both_structuring_shapes():
    batch = generate_adaptation_batch(2, n=48, seed=31)

    assert len(batch) == 48
    assert batch.typology.value_counts().to_dict() == {"normal": 12, "structuring": 12, "layering": 12, "velocity": 12}
    structuring_edges = batch[batch.typology == "structuring"].edge_regime.value_counts().to_dict()
    assert structuring_edges == {"5k": 6, "10k": 6}
