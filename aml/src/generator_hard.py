"""Imbalanced, noisy synthetic AML windows for the E3' stress test."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


CLASS_SHARES = {
    "normal": 0.94,
    "structuring": 0.02,
    "layering": 0.02,
    "velocity": 0.02,
}
PROPORTIONAL_NOISE_LOG_SIGMA = 0.20


def _noisy(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply multiplicative lognormal feature noise (~20% log-space spread)."""
    sigma = PROPORTIONAL_NOISE_LOG_SIGMA
    return values * rng.lognormal(mean=-(sigma**2) / 2, sigma=sigma, size=len(values))


def _counterparties(unique_count: int, rng: np.random.Generator) -> list[str]:
    pool = rng.choice(10_000, unique_count, replace=False).tolist()
    values = pool + rng.choice(pool, 10 - unique_count, replace=True).tolist()
    rng.shuffle(values)
    return [f"cp-{value:05d}" for value in values]


def _window_profile(label: str, rng: np.random.Generator):
    """Sample overlapping class-conditional amounts, timings, channels and hops."""
    ordinary_amounts = rng.lognormal(np.log(240.0), 0.95, 10).clip(3.0, 8_500.0)
    if label == "normal":
        amounts = ordinary_amounts
        fast_burst = rng.lognormal(np.log(3.5), 0.9, 9).clip(0.08, 35.0)
        regular_gaps = rng.exponential(115.0, 9) + 4.0
        gaps = fast_burst if rng.random() < 0.18 else regular_gaps
        unique_count = int(rng.integers(6, 11)) if rng.random() < 0.72 else int(rng.integers(2, 8))
        channel = str(rng.choice(["card", "cash", "wire", "digital"], p=[0.60, 0.20, 0.10, 0.10]))
    elif label == "structuring":
        near_threshold = rng.uniform(7_600.0, 9_950.0, 10)
        if rng.random() < 0.38:
            near_mask = rng.random(10) < rng.uniform(0.25, 0.48)
            amounts = np.where(near_mask, near_threshold, ordinary_amounts)
        else:
            amounts = ordinary_amounts
        gaps = rng.exponential(78.0, 9) + 3.0
        unique_count = int(rng.integers(3, 10))
        channel = str(rng.choice(["cash", "card", "wire", "digital"], p=[0.38, 0.42, 0.12, 0.08]))
    elif label == "layering":
        layering_amounts = rng.lognormal(np.log(450.0), 0.9, 10).clip(10.0, 8_500.0)
        amounts = layering_amounts if rng.random() < 0.50 else ordinary_amounts
        chain_gaps = rng.lognormal(np.log(35.0), 0.82, 9).clip(2.0, 180.0)
        ordinary_gaps = rng.exponential(115.0, 9) + 4.0
        gaps = chain_gaps if rng.random() < 0.50 else ordinary_gaps
        unique_count = int(rng.integers(6, 11)) if rng.random() < 0.68 else int(rng.integers(2, 8))
        channel = str(rng.choice(["wire", "card", "cash", "digital"], p=[0.38, 0.34, 0.20, 0.08]))
    else:
        fast_gaps = rng.lognormal(np.log(3.5), 0.9, 9).clip(0.08, 35.0)
        ordinary_gaps = rng.exponential(115.0, 9) + 4.0
        gaps = fast_gaps if rng.random() < 0.45 else ordinary_gaps
        low_amounts = rng.lognormal(np.log(120.0), 0.9, 10).clip(2.0, 2_500.0)
        amounts = low_amounts if rng.random() < 0.55 else ordinary_amounts
        unique_count = int(rng.integers(2, 9))
        channel = str(rng.choice(["digital", "card", "cash", "wire"], p=[0.40, 0.38, 0.14, 0.08]))

    amounts = _noisy(amounts, rng).clip(1.0, 12_000.0)
    gaps = _noisy(gaps, rng).clip(0.02, 360.0)
    return amounts, gaps, unique_count, channel


def generate_hard_transactions(n_transactions: int = 20_000, seed: int = config.SEED):
    """Return transaction rows and account windows with a 94/2/2/2 class mix.

    Classes overlap by design: signals are mixtures rather than deterministic
    signatures, and observed ``chain_hops`` is computed from counterparties,
    never assigned directly from the label. Amounts and inter-event gaps receive
    multiplicative lognormal noise with a 20% log-space spread before clipping.
    """
    n_windows = max(40, int(n_transactions) // 10)
    class_counts = {name: int(n_windows * CLASS_SHARES[name]) for name in config.CLASSES}
    class_counts["normal"] += n_windows - sum(class_counts.values())
    labels = np.concatenate([
        np.full(class_counts[name], name, dtype=object) for name in config.CLASSES
    ])
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)

    tx_rows, windows = [], []
    for window_id, label in enumerate(labels):
        amounts, gaps, unique_count, channel = _window_profile(str(label), rng)
        counterparties = _counterparties(unique_count, rng)
        elapsed = np.concatenate([[0.0], np.cumsum(gaps)])
        start_minute = float(rng.uniform(0.0, 24.0 * 60.0))
        times = (start_minute + elapsed) % (24.0 * 60.0)
        duration = max(float(elapsed[-1]), 1e-4)
        duration_hours = duration / 60.0
        count_60m = int(np.sum(elapsed <= 60.0))
        rate_hour = 10.0 / max(duration_hours, 1e-4)
        distinct_count = len(set(counterparties))

        for index in range(10):
            tx_rows.append({
                "window_id": window_id,
                "amount": float(amounts[index]),
                "time_of_day_hour": float(times[index] / 60.0),
                "inter_tx_interval_min": float(0.0 if index == 0 else gaps[index - 1]),
                "counterparty_id": counterparties[index],
                "channel": channel,
                "window_count": count_60m,
                "rate_per_hour": rate_hour,
                "typology": str(label),
            })
        windows.append({
            "window_id": window_id,
            "amount_median": float(np.median(amounts)),
            "amount_p95": float(np.quantile(amounts, 0.95)),
            "threshold_share": float(np.mean((amounts >= 9_200.0) & (amounts < 10_000.0))),
            "time_of_day_hour": float(times.mean() / 60.0),
            "inter_tx_interval_min": float(np.median(gaps)),
            "counterparty_id": counterparties[-1],
            "counterparty_count": distinct_count,
            "channel": channel,
            "window_count": count_60m,
            "rate_per_hour": rate_hour,
            "chain_hops": distinct_count,
            "window_duration_min": duration,
            "typology": str(label),
        })
    return pd.DataFrame(tx_rows), pd.DataFrame(windows)


def generate_hard_aml_data(n_transactions: int = 20_000, seed: int = config.SEED):
    """Convenience entry point returning hard account-window rows."""
    return generate_hard_transactions(n_transactions=n_transactions, seed=seed)[1]
