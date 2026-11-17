"""Append-only AML review records built on the E1 prototype explanation pipeline."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from e1_explain import _encode, _load_e0_model, _reason
from generator import generate_transactions
from quantize import FeatureQuantizer


TYPOLOGY_PROTOCOLS = {
    "normal": "No named typology indicator; this is not a clearance or no-risk determination.",
    "structuring": (
        "Structuring indicator reference: 31 U.S.C. § 5324 and FinCEN Advisory "
        "FIN-2010-A003. Smurfing is treated as a structuring-related term, not a "
        "separate E1 model class. A pattern is an indicator, not proof."
    ),
    "smurfing": (
        "Smurfing-related structuring indicator reference: 31 U.S.C. § 5324 and "
        "FinCEN Advisory FIN-2010-A003. E1 does not predict smurfing as a separate class."
    ),
    "layering": (
        "Layering typology reference: FATF, Money Laundering and Terrorist Financing "
        "Typologies; a multi-hop pattern is an indicator requiring source-record review."
    ),
    "velocity": (
        "Velocity is an E1 synthetic monitoring pattern, not a standalone statutory "
        "typology or a finding of suspicious activity."
    ),
}
ALLOWED_LABELS = frozenset(config.CLASSES)
FEATURE_COLUMNS = tuple(FeatureQuantizer.SPECS)
DEFAULT_AUDIT_TRAIL = os.path.join(config.RESULTS_DIR, "audit_trail.jsonl")
STABILITY_LIMITATION = (
    "E5 stability scope: only the coarse typology name is used as a stable explanation; "
    "continuous coordinates and attribution magnitudes are not retraining-stable."
)


def normalize_distance(raw_distance: float) -> float:
    """Map E1's unbounded Euclidean distance to a bounded, monotonic [0, 1) score."""
    distance = float(raw_distance)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("prototype distance must be a finite non-negative number")
    return distance / (1.0 + distance)


def make_audit_record(
    *,
    sample_id: str,
    typology_label: str,
    second_prototype: str,
    raw_distance: float,
    second_raw_distance: float,
    motif: str,
    model_version: str,
    predicted_class: str,
) -> dict:
    """Adapt E1's reason and nearest-centroid fields into a durable review record."""
    if typology_label not in ALLOWED_LABELS or second_prototype not in ALLOWED_LABELS:
        raise ValueError("prototype labels must be from the configured E1 class vocabulary")
    protocol_reference = TYPOLOGY_PROTOCOLS[typology_label]
    reviewer_note = (
        "Bu örüntü analist incelemesine yöneliktir; tek başına SAR kararı değildir."
        if typology_label != "normal"
        else "Bu sonuç SAR taslağı değildir ve risk yokluğu ya da aklanma kanıtı sayılmaz."
    )
    rationale = (
        f"E1 gözlenen örüntü: {motif}. {protocol_reference} {reviewer_note}"
    )
    record = {
        "record_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "sample_id": sample_id,
        "decision": "analyst_review" if typology_label != "normal" else "no_named_typology_indicator",
        "typology_label": typology_label,
        "model_prediction": predicted_class,
        "prototype": {
            "region": typology_label,
            "distance": normalize_distance(raw_distance),
            "distance_metric": "normalized_euclidean_d_over_1_plus_d",
            "raw_euclidean_distance": float(raw_distance),
            "second_region": second_prototype,
            "second_distance": normalize_distance(second_raw_distance),
            "second_raw_euclidean_distance": float(second_raw_distance),
        },
        "rationale": rationale,
        "explanation_scope": "name-scope",
        "stability_limitation": STABILITY_LIMITATION,
        "evidence_source": "E1 CloudEncoderND nearest-prototype explanation pipeline",
        "data_scope_limitation": (
            "E0/E1 AML typology evidence is synthetic. Elliptic E2/E2′ is a separate "
            "anonymized transaction-node proxy with one-hop/transductive limits; it "
            "does not validate this account-window SAR workflow."
        ),
        "performance_context": (
            "The best K10/B8 hybrid trailed the same-split GBT baseline; no performance "
            "superiority is claimed."
        ),
    }
    validate_record(record)
    return record


def validate_record(record: dict) -> None:
    """Validate the durable audit schema and its bounded decision fields."""
    if record.get("typology_label") not in ALLOWED_LABELS:
        raise ValueError("typology_label is outside the controlled vocabulary")
    if not str(record.get("rationale", "")).strip():
        raise ValueError("rationale must not be empty")
    prototype = record.get("prototype", {})
    if prototype.get("region") not in ALLOWED_LABELS:
        raise ValueError("prototype region is outside the controlled vocabulary")
    if record.get("model_prediction") not in ALLOWED_LABELS:
        raise ValueError("model prediction is outside the controlled vocabulary")
    for key in ("distance", "second_distance"):
        distance = prototype.get(key)
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise ValueError(f"prototype {key} must be numeric")
        if not math.isfinite(float(distance)) or not 0.0 <= float(distance) <= 1.0:
            raise ValueError(f"prototype {key} must be in [0, 1]")
    if record.get("explanation_scope") != "name-scope":
        raise ValueError("explanations must remain at the E5-supported name scope")
    if not record.get("stability_limitation"):
        raise ValueError("stability limitation must be included")


def append_records(records: list[dict], out_path: str) -> None:
    """Append complete JSON records; never truncate or rewrite existing trail rows."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            validate_record(record)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _reference_training_windows():
    """Reproduce E1's train-only reference split for legacy E0 checkpoints."""
    _, windows = generate_transactions(n_transactions=20_000, seed=config.SEED)
    labels = np.asarray([config.CLASSES.index(name) for name in windows.typology], dtype=int)
    train_ids, temp_ids = train_test_split(
        np.arange(len(windows)), test_size=0.4, random_state=config.SEED, stratify=labels,
    )
    # Keep this split identical to E1's deterministic test partition protocol.
    train_test_split(
        temp_ids, test_size=0.5, random_state=config.SEED + 1, stratify=labels[temp_ids],
    )
    return windows.iloc[train_ids].reset_index(drop=True)


def generate_report(checkpoint_path: str, data_path: str, out_path: str = DEFAULT_AUDIT_TRAIL) -> list[dict]:
    """Generate audit entries for CSV test rows using E1's encoder and reason templates.

    A CSV may contain ``split`` values ``train`` and ``test``. If it contains only
    test rows, E1's deterministic synthetic training split supplies the quantizer
    and train-only prototypes, matching the legacy E0/E1 checkpoint protocol.
    """
    frame = pd.read_csv(data_path)
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"data CSV is missing E1 window features: {', '.join(missing)}")

    if "split" in frame.columns and (frame["split"].astype(str).str.lower() == "train").any():
        split = frame["split"].astype(str).str.lower()
        train_rows = frame.loc[split == "train"].drop(columns=["split"]).reset_index(drop=True)
        test_rows = frame.loc[split == "test"].drop(columns=["split"]).reset_index(drop=True)
        if "typology" not in train_rows:
            raise ValueError("training rows must include typology labels to fit E1 prototypes")
    else:
        train_rows = _reference_training_windows()
        test_rows = frame.drop(columns=["split"], errors="ignore").reset_index(drop=True)
    if train_rows.empty or test_rows.empty:
        raise ValueError("CSV must provide non-empty training and test rows (or test-only rows)")
    unknown_labels = sorted(set(train_rows["typology"].astype(str)) - set(config.CLASSES))
    if unknown_labels:
        raise ValueError(f"unsupported training typology labels: {', '.join(unknown_labels)}")

    quantizer = FeatureQuantizer().fit(train_rows)
    model = _load_e0_model(checkpoint_path, quantizer.vocab_size)
    train_tokens = quantizer.transform(train_rows)
    test_tokens = quantizer.transform(test_rows)
    train_embeddings, _ = _encode(model, train_tokens)
    test_embeddings, test_logits = _encode(model, test_tokens)
    train_labels = train_rows["typology"].astype(str).to_numpy()
    prototypes = {
        name: train_embeddings[train_labels == name].mean(axis=0)
        for name in config.CLASSES
        if np.any(train_labels == name)
    }
    if not prototypes:
        raise ValueError("training rows contain no configured E1 prototype classes")

    prototype_names = list(prototypes)
    prototype_matrix = np.stack([prototypes[name] for name in prototype_names])
    distances = np.linalg.norm(test_embeddings[:, None, :] - prototype_matrix[None, :, :], axis=2)
    model_version = Path(checkpoint_path).stem
    records = []
    for row_index, row in test_rows.iterrows():
        order = np.argsort(distances[row_index], kind="stable")
        nearest_index = int(order[0])
        second_index = int(order[1]) if len(order) > 1 else nearest_index
        nearest = prototype_names[nearest_index]
        second = prototype_names[second_index]
        motif, _ = _reason(row)
        if "sample_id" in test_rows and pd.notna(row.get("sample_id")):
            sample_id = str(row["sample_id"])
        elif "window_id" in test_rows and pd.notna(row.get("window_id")):
            sample_id = str(row["window_id"])
        elif "transaction_node_id" in test_rows and pd.notna(row.get("transaction_node_id")):
            sample_id = str(row["transaction_node_id"])
        else:
            sample_id = f"row-{row_index}"
        records.append(make_audit_record(
            sample_id=sample_id,
            typology_label=nearest,
            second_prototype=second,
            raw_distance=float(distances[row_index, nearest_index]),
            second_raw_distance=float(distances[row_index, second_index]),
            motif=motif,
            model_version=model_version,
            predicted_class=config.CLASSES[int(np.argmax(test_logits[row_index]))],
        ))
    append_records(records, out_path)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="trained E0 AML checkpoint")
    parser.add_argument("--data", required=True, help="CSV of E1 account-window test rows")
    parser.add_argument("--out", default=DEFAULT_AUDIT_TRAIL, help="append-only JSONL output")
    args = parser.parse_args(argv)
    records = generate_report(args.ckpt, args.data, args.out)
    print(f"appended {len(records)} audit records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
