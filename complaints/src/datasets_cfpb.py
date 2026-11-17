"""Bounded-memory, reproducible sampling and chronological splitting of CFPB."""

import glob
import os
import random

import pandas as pd

from config import PARQUET_DIR, SEED

KEEP_COLUMNS = [
    "consumer_complaint_narrative", "date_received", "product", "issue",
]
OUTPUT_COLUMNS = ["narrative", "date", "week", "product", "issue"]


def _parquet_files(data_dir):
    """Accept either the snapshot root or its nested `data/` directory."""
    if os.path.isdir(os.path.join(data_dir, "data")):
        data_dir = os.path.join(data_dir, "data")
    return sorted(glob.glob(os.path.join(data_dir, "*.parquet")))


def load_cfpb(n_rows=None, n_sample=100_000, seed=SEED, data_dir=None):
    """Load valid narratives, reservoir-sample deterministically, then split by time.

    `n_rows`, when set, limits the number of eligible rows considered before
    sampling (useful for small fixture datasets). `n_sample` bounds memory to
    the requested pilot size even though all parquet fragments are scanned.
    The returned `split` column is `train` for the oldest 70% and `drift` for
    the most recent 30% of the sampled records.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("CFPB parquet loading requires pyarrow") from exc

    data_dir = data_dir or PARQUET_DIR
    files = _parquet_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CFPB parquet fragments under {data_dir}")
    if n_sample is not None and n_sample < 1:
        raise ValueError("n_sample must be positive")

    rng = random.Random(seed)
    reservoir = []
    eligible_seen = 0
    scan_limit = None if n_rows is None else max(0, int(n_rows))
    reservoir_limit = scan_limit if n_sample is None else int(n_sample)

    for path in files:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema.names)
        missing = set(KEEP_COLUMNS[:3]) - available
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        columns = [c for c in KEEP_COLUMNS if c in available]
        for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
            frame = batch.to_pandas()
            dates = pd.to_datetime(frame["date_received"], errors="coerce", utc=True)
            frame["date_received"] = dates.dt.tz_convert(None)
            narrative = frame["consumer_complaint_narrative"]
            valid = (
                narrative.notna()
                & narrative.astype(str).str.strip().ne("")
                & frame["date_received"].notna()
                & frame["product"].notna()
                & frame["product"].astype(str).str.strip().ne("")
            )
            rows = frame.loc[valid]
            for values in rows.itertuples(index=False, name=None):
                row = dict(zip(columns, values))
                if scan_limit is not None and eligible_seen >= scan_limit:
                    break
                eligible_seen += 1
                if reservoir_limit is None or len(reservoir) < reservoir_limit:
                    reservoir.append(row)
                else:
                    pick = rng.randrange(eligible_seen)
                    if pick < reservoir_limit:
                        reservoir[pick] = row
            if scan_limit is not None and eligible_seen >= scan_limit:
                break
        if scan_limit is not None and eligible_seen >= scan_limit:
            break

    if not reservoir:
        return pd.DataFrame(columns=OUTPUT_COLUMNS + ["split"])

    selected = pd.DataFrame(reservoir)
    selected = selected.rename(columns={
        "consumer_complaint_narrative": "narrative",
        "date_received": "date",
    })
    for column in ("product", "issue"):
        if column not in selected:
            selected[column] = None
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
    selected = selected.dropna(subset=["date"]).sort_values(
        ["date"], kind="mergesort"
    ).reset_index(drop=True)
    selected["week"] = selected["date"].dt.to_period("W").astype(str)
    split_at = int(len(selected) * 0.70)
    selected["split"] = "drift"
    selected.loc[selected.index < split_at, "split"] = "train"
    return selected[OUTPUT_COLUMNS + ["split"]]
