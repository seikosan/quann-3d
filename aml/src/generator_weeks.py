"""Small deterministic AML account-window streams for the E4 drift demo."""
from __future__ import annotations

import numpy as np
import pandas as pd


CLASSES = ("normal", "structuring", "layering", "velocity")
OLD_THRESHOLD = 10_000.0
NEW_THRESHOLD = 5_000.0


def _one_window(label: str, rng: np.random.Generator, *, edge: str, threshold: float):
    """Generate one ten-transaction summary and its observable evidence fields."""
    if label == "normal":
        amounts = rng.lognormal(np.log(180.0), 0.62, 10).clip(2.0, 2_400.0)
        gaps = rng.exponential(145.0, 9) + 20.0
        counterparties = int(rng.integers(7, 11))
        channel = "card"
    elif label == "structuring":
        if edge == "5k":
            amounts = rng.uniform(4_200.0, 4_900.0, 10)
            # Controlled drift: outside the newly shifted deposit amounts,
            # this subtype matches ordinary baseline activity.
            gaps = rng.exponential(145.0, 9) + 20.0
            counterparties = int(rng.integers(7, 11))
            channel = "card"
        else:
            amounts = rng.uniform(9_200.0, 9_900.0, 10)
            amounts *= rng.lognormal(0.0, 0.018, 10)
            amounts = amounts.clip(9_200.0, 9_900.0)
            gaps = rng.exponential(32.0, 9) + 2.0
            counterparties = int(rng.integers(3, 7))
            channel = "cash_deposit"
    elif label == "layering":
        amounts = rng.lognormal(np.log(680.0), 0.48, 10).clip(100.0, 2_200.0)
        gaps = rng.uniform(7.0, 38.0, 9)
        counterparties = int(rng.integers(7, 11))
        channel = "wire"
    else:
        amounts = rng.lognormal(np.log(55.0), 0.52, 10).clip(5.0, 500.0)
        gaps = rng.exponential(0.20, 9) + 0.02
        counterparties = int(rng.integers(2, 5))
        channel = "digital"

    elapsed = np.concatenate([[0.0], np.cumsum(gaps)])
    # This remains the week-0 10K feature. A frozen representation must not
    # silently recode the 5K regulatory change as an already-known edge.
    near_edge = (amounts >= OLD_THRESHOLD * 0.92) & (amounts < OLD_THRESHOLD)
    return {
        "amount_median": float(np.median(amounts)),
        "amount_p95": float(np.quantile(amounts, 0.95)),
        "threshold_share": float(np.mean(near_edge)),
        "time_of_day_hour": float(rng.uniform(0.0, 24.0)),
        "inter_tx_interval_min": float(np.median(gaps)),
        # This is a stable synthetic placeholder, not a customer identifier.
        "counterparty_id": "synthetic-counterparty-pool",
        "counterparty_count": counterparties,
        "channel": channel,
        "window_count": int(np.sum(elapsed <= 60.0)),
        "rate_per_hour": float(10.0 / max(float(elapsed[-1]) / 60.0, 1e-4)),
        "chain_hops": counterparties if label == "layering" else max(1, counterparties // 2),
        "window_duration_min": float(elapsed[-1]),
        "typology": label,
        "edge_regime": edge if label == "structuring" else "not_applicable",
        "regulatory_reporting_threshold": float(threshold),
    }


def generate_week0(n_per_class: int = 400, seed: int = 20261016) -> pd.DataFrame:
    """Generate week-0 data; structuring is clustered below the 10K threshold."""
    rng = np.random.default_rng(seed)
    rows = [
        _one_window(label, rng, edge="10k", threshold=OLD_THRESHOLD)
        for label in CLASSES
        for _ in range(int(n_per_class))
    ]
    rng.shuffle(rows)
    frame = pd.DataFrame(rows)
    frame.insert(0, "window_id", np.arange(len(frame), dtype=int))
    frame["week"] = 0
    return frame


def generate_week_stream(
    week: int,
    *,
    n_per_class: int = 200,
    seed: int = 20261016,
    edge_fraction: float = 0.5,
) -> pd.DataFrame:
    """Generate a post-change weekly stream with old/new structuring patterns."""
    if week < 1:
        raise ValueError("week must be >= 1")
    if not 0.0 <= edge_fraction <= 1.0:
        raise ValueError("edge_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed + week * 1009)
    n_per_class = int(n_per_class)
    n_new = int(round(n_per_class * edge_fraction))
    structuring_edges = ["5k"] * n_new + ["10k"] * (n_per_class - n_new)
    rng.shuffle(structuring_edges)
    rows = []
    for label in CLASSES:
        edge_modes = structuring_edges if label == "structuring" else ["none"] * n_per_class
        for edge in edge_modes:
            rows.append(_one_window(
                label,
                rng,
                edge=edge,
                threshold=NEW_THRESHOLD,
            ))
    rng.shuffle(rows)
    frame = pd.DataFrame(rows)
    frame.insert(0, "window_id", np.arange(len(frame), dtype=int) + week * 1_000_000)
    frame["week"] = int(week)
    return frame


def generate_adaptation_batch(week: int, n: int = 48, seed: int = 20261016) -> pd.DataFrame:
    """A fixed-size labeled weekly budget with 5K and legacy cases represented."""
    if n < 8 or n % 4:
        raise ValueError("n must be a multiple of four and at least eight")
    rng = np.random.default_rng(seed + week * 7919)
    rows = []
    per_class = n // 4
    for label in CLASSES:
        if label == "structuring":
            modes = ["5k"] * (per_class // 2) + ["10k"] * (per_class - per_class // 2)
            rng.shuffle(modes)
        else:
            modes = ["none"] * per_class
        for edge in modes:
            rows.append(_one_window(label, rng, edge=edge, threshold=NEW_THRESHOLD))
    rng.shuffle(rows)
    frame = pd.DataFrame(rows)
    frame.insert(0, "window_id", np.arange(len(frame), dtype=int) + week * 10_000_000)
    frame["week"] = int(week)
    return frame
