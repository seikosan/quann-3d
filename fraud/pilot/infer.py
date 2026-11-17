#!/usr/bin/env python3
"""Run the packaged QUANN fraud model on an IBM-schema transaction CSV.

From the repository root:
    python3 fraud/pilot/infer.py transactions.csv

From fraud/:
    python3 pilot/infer.py transactions.csv

Each stdout line is one transaction: decision, probability, and point-ablation
explanation. Diagnostics go to stderr so stdout contains only result rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRAUD_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = FRAUD_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


REQUIRED_COLUMNS = {
    "Year", "Month", "Day", "Time", "Amount", "Use Chip",
    "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
    "Errors?",
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="CSV with IBM transaction feature columns")
    p.add_argument("--deploy", type=Path, default=FRAUD_DIR / "results" / "deploy",
                   help="deploy package directory (default: fraud/results/deploy)")
    p.add_argument("--threshold", type=float, default=0.9969,
                   help="predict fraud when P(fraud) >= threshold (default: 0.9969)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="transactions per inference/explanation batch")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most this many input rows (useful for a smoke test)")
    p.add_argument("--top-k", type=int, default=5,
                   help="number of strongest point-ablation effects to print")
    return p


def _load_package(deploy: Path):
    try:
        import numpy as np
        import onnxruntime as ort
        import pandas as pd
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            f"Eksik bağımlılık ({missing}). Kurulum: "
            f"python3 -m pip install -r {Path(__file__).with_name('requirements.txt')}"
        ) from exc

    from quantize import FeatureQuantizer

    paths = {
        "model": deploy / "model.onnx",
        "quantizer": deploy / "quantizer.json",
        "decoder": deploy / "token_decoder.json",
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise RuntimeError("Deploy paketi eksik dosya içeriyor: " + ", ".join(missing_files))

    with paths["quantizer"].open(encoding="utf-8") as f:
        quantizer = FeatureQuantizer.from_dict(json.load(f))
    with paths["decoder"].open(encoding="utf-8") as f:
        decoder = json.load(f)
    session = ort.InferenceSession(str(paths["model"]), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    if not {"ids", "mask"}.issubset(input_names):
        raise RuntimeError(f"Beklenmeyen ONNX girdileri: {sorted(input_names)}; ids ve mask gerekli")
    return np, pd, quantizer, decoder, session


def _logit(np, probs):
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def _predict_chunk(np, session, tokens):
    masks = (tokens != 0) & (tokens != 1)
    probs = session.run(None, {"ids": tokens, "mask": masks})[0].reshape(-1)

    # Match explain_pilot.point_ablation: hide one point at a time, compare
    # base logit - masked logit. DeployPath applies exactly ids-table * mask.
    n_rows, n_features = tokens.shape
    variants = np.repeat(tokens[:, None, :], n_features, axis=1).reshape(
        n_rows * n_features, n_features
    )
    variant_masks = (variants != 0) & (variants != 1)
    row_ix = np.repeat(np.arange(n_rows), n_features)
    feature_ix = np.tile(np.arange(n_features), n_rows)
    variant_masks[row_ix, feature_ix] = False
    ablated = session.run(None, {"ids": variants, "mask": variant_masks})[0].reshape(
        n_rows, n_features
    )
    contributions = _logit(np, probs)[:, None] - _logit(np, ablated)
    return probs, contributions


def run(args) -> int:
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold değeri 0 ile 1 arasında olmalı")
    if args.batch_size < 1 or args.top_k < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("--batch-size, --top-k ve verilmişse --limit pozitif tam sayı olmalı")
    if not args.csv.is_file():
        raise RuntimeError(f"CSV bulunamadı: {args.csv}")

    deploy = args.deploy
    if not deploy.is_absolute() and not deploy.exists() and (ROOT / deploy).exists():
        deploy = ROOT / deploy
    np, pd, quantizer, decoder, session = _load_package(deploy)
    frame = pd.read_csv(args.csv, nrows=args.limit)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise RuntimeError("CSV'de gerekli IBM özellik kolonları eksik: " + ", ".join(missing))
    if frame.empty:
        return 0

    tokens = quantizer.transform(frame)
    expected_k = len(quantizer.feature_order)
    if tokens.shape[1] != expected_k:
        raise RuntimeError(f"Token boyutu hatalı: {tokens.shape[1]} != {expected_k}")
    if "p" not in {item.name for item in session.get_outputs()}:
        raise RuntimeError("ONNX çıktısı 'p' bulunamadı")

    for start in range(0, len(frame), args.batch_size):
        stop = min(start + args.batch_size, len(frame))
        probs, contribs = _predict_chunk(np, session, tokens[start:stop])
        for local, prob in enumerate(probs):
            token_row = tokens[start + local]
            order = np.argsort(-np.abs(contribs[local]), kind="stable")
            effects = []
            for j in order:
                token_id = str(int(token_row[j]))
                token_info = decoder.get(token_id)
                if token_info is None:
                    label = f"token_id={token_id}"
                else:
                    label = token_info["label"]
                delta = float(contribs[local, j])
                if abs(delta) < 1e-6:
                    continue
                direction = "fraud-destekliyor" if delta > 0 else "fraud-karşıtı"
                effects.append(f"{label} ({direction}, {delta:+.2f} logit)")
                if len(effects) == args.top_k:
                    break
            decision = "FRAUD" if prob >= args.threshold else "NORMAL"
            explanation = "; ".join(effects) if effects else "single-token etkisi ölçülemedi"
            print(f"{decision}\tp={prob:.8f}\t{explanation}", flush=True)
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"infer.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
