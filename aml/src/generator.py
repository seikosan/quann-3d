"""Deterministic account-window generator for AML typology E0."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def generate_transactions(n_transactions: int = 20_000, seed: int = config.SEED):
    """Return (transaction rows, ten-transaction account-window rows).

    Typologies are generated at the account-window level, then each window is
    expanded into transactions. n_transactions is rounded down to a multiple
    of ten so each sample has a complete, fixed-size event cloud.
    """
    n_windows = max(4, int(n_transactions) // 10)
    n_windows -= n_windows % 4
    rng = np.random.default_rng(seed)
    tx_rows, windows = [], []
    for window_id in range(n_windows):
        label = config.CLASSES[window_id % len(config.CLASSES)]
        start_minute = float(rng.uniform(0, 24 * 60))
        if label == "normal":
            gaps = rng.exponential(180.0, 9) + 15.0
            amounts = rng.lognormal(np.log(95.0), 0.85, 10).clip(2, 2400)
            channel = rng.choice(["card", "cash", "wire"], p=[0.72, 0.20, 0.08])
            counterparties = [f"cp-{i:04d}" for i in rng.choice(1000, 10, replace=False)]
        elif label == "structuring":
            gaps = rng.exponential(28.0, 9) + 1.0
            amounts = rng.uniform(9200.0, 9900.0, 10)
            channel = "cash_deposit"
            pool = rng.choice(1000, 4, replace=False)
            counterparties = [f"cp-{i:04d}" for i in rng.choice(pool, 10, replace=True)]
        elif label == "layering":
            gaps = rng.uniform(4.0, 38.0, 9)
            amounts = rng.lognormal(np.log(1250.0), 0.42, 10).clip(150, 5000)
            channel = "wire"
            counterparties = [f"cp-{i:04d}" for i in rng.choice(1000, 10, replace=False)]
        else:
            gaps = rng.exponential(0.22, 9) + 0.01
            amounts = rng.lognormal(np.log(24.0), 0.55, 10).clip(1, 180)
            channel = "digital"
            pool = rng.choice(1000, 2, replace=False)
            counterparties = [f"cp-{i:04d}" for i in rng.choice(pool, 10, replace=True)]

        elapsed = np.concatenate([[0.0], np.cumsum(gaps)])
        times = (start_minute + elapsed) % (24 * 60)
        duration_hr = max(float(elapsed[-1]) / 60.0, 1e-4)
        count_60m = int(np.sum(elapsed <= 60.0))
        rate_hour = 10.0 / duration_hr
        for i in range(10):
            tx_rows.append({
                "window_id": window_id,
                "amount": float(amounts[i]),
                "time_of_day_hour": float(times[i] / 60.0),
                "inter_tx_interval_min": float(0.0 if i == 0 else gaps[i - 1]),
                "counterparty_id": counterparties[i],
                "channel": str(channel),
                "window_count": count_60m,
                "rate_per_hour": rate_hour,
                "typology": label,
            })
        windows.append({
            "window_id": window_id,
            "amount_median": float(np.median(amounts)),
            "amount_p95": float(np.quantile(amounts, 0.95)),
            "threshold_share": float(np.mean((amounts >= 9200) & (amounts < 10_000))),
            "time_of_day_hour": float(times.mean() / 60.0),
            "inter_tx_interval_min": float(np.median(gaps)),
            "counterparty_id": counterparties[-1],
            "counterparty_count": int(len(set(counterparties))),
            "channel": str(channel),
            "window_count": count_60m,
            "rate_per_hour": rate_hour,
            "chain_hops": int(len(set(counterparties))) if label == "layering" else 1,
            "window_duration_min": float(elapsed[-1]),
            "typology": label,
        })
    return pd.DataFrame(tx_rows), pd.DataFrame(windows)


def generate_aml_data(n_transactions: int = 20_000, seed: int = config.SEED):
    """Convenience entry point returning the modeled account windows."""
    return generate_transactions(n_transactions=n_transactions, seed=seed)[1]
