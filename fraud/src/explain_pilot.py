from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import datasets
import metrics
import quantize
import threshold_utils
import train_quann as tq
from qcloud_fraud import QCloudFraudClassifier
from quantize import FeatureQuantizer


def load_quantizer(checkpoint_path: str, fallback: FeatureQuantizer) -> FeatureQuantizer:
    raw = torch.load(checkpoint_path, map_location="cpu")
    if raw.get("quantizer"):
        return FeatureQuantizer.from_dict(raw["quantizer"])
    return fallback


def free_gb() -> float:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        free = int(re.search(r"Pages free:\s+(\d+)", out).group(1))
        inactive = int(re.search(r"Pages inactive:\s+(\d+)", out).group(1))
        return (free + inactive) * page / 1e9
    except Exception:
        return 1e9


def select_upper_slice(probs: np.ndarray, labels: np.ndarray,
                       n_slice: int) -> np.ndarray:
    """Fixed-size upper slice: all fraud (if it fits) + highest-prob normals."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    n = len(probs)
    if n_slice >= n:
        sel = np.arange(n)
    else:
        fraud_idx = np.flatnonzero(labels == 1)
        if len(fraud_idx) >= n_slice:
            sel = np.argsort(-probs, kind="stable")[:n_slice]
        else:
            normal_idx = np.flatnonzero(labels == 0)
            remaining = max(0, n_slice - len(fraud_idx))
            if remaining:
                top_normal = np.argsort(-probs[normal_idx], kind="stable")[:remaining]
                sel = np.concatenate([fraud_idx, normal_idx[top_normal]])
            else:
                sel = fraud_idx
    return np.sort(sel)


def decode_token(fq: FeatureQuantizer, feature_index: int, token_id: int) -> str:
    key, kind, n = fq.specs[feature_index]
    local = int(token_id) - fq.offsets[key]
    n_buckets = fq.n_per_feature[key]
    if kind == "quantile":
        edges = fq.bins[key]
        if n_buckets == n + 1 and local == n:
            return f"{key}=missing"
        if local == 0:
            return f"{key}<{float(edges[0]):.2f}"
        if local == n - 1:
            return f"{key}>{float(edges[-1]):.2f}"
        return f"{key}~[{float(edges[local - 1]):.2f},{float(edges[local]):.2f})"
    if kind == "int":
        return f"{key}={local}"
    inv = {v: k for k, v in fq.cat_map.get(key, {}).items()}
    if local == n_buckets - 1 and (n is None or local >= n):
        return f"{key}=other"
    return f"{key}={inv.get(local, str(local))}"


def decode_row(fq: FeatureQuantizer, tokens: np.ndarray) -> list[str]:
    return [decode_token(fq, j, int(t)) for j, t in enumerate(tokens)]


def point_ablation(model: QCloudFraudClassifier, tokens: torch.Tensor,
                   mask: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    model.eval()
    n, k = tokens.shape
    contribs = torch.zeros(n, k, device=tokens.device)
    with torch.no_grad():
        for s in range(0, n, chunk):
            ts = tokens[s:s + chunk]
            ms = mask[s:s + chunk]
            pts = model.points_for(ts, ms)
            base, _ = model.forward(ts, ms, points_override=pts)
            base = base.squeeze(-1)
            for j in range(k):
                ab_pts = pts.clone()
                ab_pts[:, j, :] = 0.0
                ab_mask = ms.clone()
                ab_mask[:, j] = False
                out, _ = model.forward(ts, ab_mask, points_override=ab_pts)
                contribs[s:s + chunk, j] = base - out.squeeze(-1)
    return contribs


def aggregate(fq: FeatureQuantizer, tokens: np.ndarray,
              contribs: np.ndarray, top_tokens: int = 20) -> dict:
    n, k = tokens.shape
    feature_abs = np.zeros(k)
    feature_signed = np.zeros(k)
    for i in range(n):
        feature_abs += np.abs(contribs[i])
        feature_signed += contribs[i]
    feature_rows = []
    for j in range(k):
        feature_rows.append({
            "feature": fq.feature_order[j],
            "mean_abs_logit_contrib": float(feature_abs[j] / n),
            "mean_signed_logit_contrib": float(feature_signed[j] / n),
        })
    feature_rows.sort(key=lambda r: r["mean_abs_logit_contrib"], reverse=True)

    acc: dict[int, dict] = {}
    for i in range(n):
        for j in range(k):
            tid = int(tokens[i, j])
            row = acc.setdefault(tid, {
                "token_id": tid,
                "feature": fq.feature_order[j],
                "label": decode_token(fq, j, tid),
                "sum_abs": 0.0,
                "sum_signed": 0.0,
                "count": 0,
            })
            row["sum_abs"] += float(abs(contribs[i, j]))
            row["sum_signed"] += float(contribs[i, j])
            row["count"] += 1
    token_rows = []
    for row in acc.values():
        mean_abs = row["sum_abs"] / row["count"]
        token_rows.append({
            "token_id": row["token_id"],
            "feature": row["feature"],
            "label": row["label"],
            "count": row["count"],
            "total_abs_logit_contrib": row["sum_abs"],
            "mean_abs_logit_contrib": mean_abs,
            "mean_signed_logit_contrib": row["sum_signed"] / row["count"],
            "impact_score": mean_abs * np.sqrt(row["count"]),
        })
    token_rows.sort(key=lambda r: r["impact_score"], reverse=True)
    positive = sorted((r for r in token_rows if r["mean_signed_logit_contrib"] > 0),
                      key=lambda r: r["mean_signed_logit_contrib"], reverse=True)
    negative = sorted((r for r in token_rows if r["mean_signed_logit_contrib"] < 0),
                      key=lambda r: r["mean_signed_logit_contrib"])
    return {
        "features": feature_rows,
        "top_tokens": token_rows[:top_tokens],
        "top_positive": positive[:top_tokens],
        "top_negative": negative[:top_tokens],
    }


def explanation_text(p_fraud: float, decoded: list[str],
                     top: list[dict]) -> str:
    positive = [r["label"] for r in top if r["logit_contrib"] > 0][:3]
    negative = [r["label"] for r in top if r["logit_contrib"] < 0][:2]
    parts = [f"P(fraud)={p_fraud:.3f}"]
    if positive:
        parts.append("fraud-supporting: " + "; ".join(positive))
    if negative:
        parts.append("fraud-protecting: " + "; ".join(negative))
    parts.append("context: " + "; ".join(decoded[:5]))
    return " | ".join(parts)


def select_cases(fq: FeatureQuantizer, labels: np.ndarray, probs: np.ndarray,
                 tokens: np.ndarray, contribs: np.ndarray,
                 n_cases: int = 5, top_k: int = 5) -> list[dict]:
    cases = []
    for cls, name in ((1, "fraud"), (0, "normal")):
        idx = np.flatnonzero(labels == cls)
        if not len(idx):
            continue
        order = idx[np.argsort(-probs[idx])][:n_cases]
        for i in order:
            top_idx = np.argsort(-np.abs(contribs[i]))[:top_k]
            top = [{
                "position": int(j),
                "feature": fq.feature_order[int(j)],
                "label": decode_token(fq, int(j), int(tokens[i, j])),
                "logit_contrib": float(contribs[i, j]),
            } for j in top_idx]
            decoded = decode_row(fq, tokens[i])
            cases.append({
                "true_class": name,
                "p_fraud": float(probs[i]),
                "decoded": decoded,
                "top_contributions": top,
                "text": explanation_text(float(probs[i]), decoded, top),
            })
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(config.RESULTS_DIR, "quann_full_hybrid.pt"))
    ap.add_argument("--data", choices=["small", "big-test"], default="small")
    ap.add_argument("--n-rows", type=int, default=None)
    ap.add_argument("--n-fraud", type=int, default=200)
    ap.add_argument("--n-normal", type=int, default=800)
    ap.add_argument("--n-explain", type=int, default=50000)
    ap.add_argument("--n-test", type=int, default=4700000)
    ap.add_argument("--n-train", type=int, default=1020000)
    ap.add_argument("--n-val", type=int, default=100000)
    ap.add_argument("--train-fraud-frac", type=float, default=0.02)
    ap.add_argument("--test-frac", type=float, default=0.0013)
    ap.add_argument("--top-tokens", type=int, default=20)
    ap.add_argument("--cases", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.chunk is None:
        args.chunk = 256 if args.data == "small" else 5000
    if args.out is None:
        name = "explain_pilot.json" if args.data == "small" else "explain_upper_slice.json"
        args.out = os.path.join(config.RESULTS_DIR, name)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    log_name = ("run_explain_pilot.log" if args.data == "small"
                else "run_explain_upper_slice.log")
    log_path = os.path.join(config.LOGS_DIR, log_name)
    log_fp = open(log_path, "a")

    def log(msg: str):
        line = f"=== [{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg} ==="
        print(line, flush=True)
        log_fp.write(line + "\n")

    title = ("FRAUD EXPLANATION PILOT — point-ablation attribution"
             if args.data == "small" else
             "FRAUD UPPER-SLICE EXPLANATION — full 4.7M test, bounded attribution")
    log(title)
    print(f"  checkpoint: {args.checkpoint}", flush=True)
    log_fp.write(f"  checkpoint: {args.checkpoint}\n")

    model = QCloudFraudClassifier.load_quann(args.checkpoint, torch.device("cpu"))

    full_test_info = None
    full_test_metrics = None
    if args.data == "small":
        frame = datasets.load_small()
        y = datasets.get_label(frame)
        rng = np.random.default_rng(args.seed)
        fraud_idx = np.flatnonzero(y == 1)
        normal_idx = np.flatnonzero(y == 0)
        n_fraud = min(args.n_fraud, len(fraud_idx))
        n_normal = min(args.n_normal, len(normal_idx))
        sel = np.concatenate([
            rng.choice(fraud_idx, size=n_fraud, replace=n_fraud > len(fraud_idx)),
            rng.choice(normal_idx, size=n_normal, replace=n_normal > len(normal_idx)),
        ])
        rng.shuffle(sel)
        fallback = quantize.FeatureQuantizer(specs=quantize.LEGACY_QUANT_SPECS).fit(frame.iloc[sel])
        fq = load_quantizer(args.checkpoint, fallback)
        tok = fq.transform(frame.iloc[sel])
        labels = y[sel]
        note = "stratified development sample; not the natural 4.7M held-out population"
        strategy = "stratified random sample"
    else:
        if free_gb() < 12:
            raise SystemExit(f"aborted: only {free_gb():.1f}GB reclaimable; big-test load needs >=12GB")
        usecols = ["User", "Is Fraud?", "Amount", "Time", "Month", "Day", "Year",
                   "Zip", "Use Chip", "MCC", "Merchant City", "Merchant Name",
                   "Merchant State", "Errors?"]
        print(f"  reading {config.BIG_CSV} (usecols={len(usecols)}) ...", flush=True)
        log_fp.write(f"  reading {config.BIG_CSV} (usecols={len(usecols)}) ...\n")
        frame = pd.read_csv(config.BIG_CSV, usecols=usecols)
        print(f"  rows={len(frame):,} free_gb={free_gb():.1f}", flush=True)
        log_fp.write(f"  rows={len(frame):,} free_gb={free_gb():.1f}\n")
        raw = torch.load(args.checkpoint, map_location="cpu")
        if not raw.get("quantizer"):
            raise SystemExit("big-test mode requires a checkpoint with a bundled quantizer")
        fq = FeatureQuantizer.from_dict(raw["quantizer"])
        data = tq.build_data(frame, args.n_train, args.n_val, args.n_test,
                             args.train_fraud_frac, args.test_frac,
                             seed=config.SEED, quantizer=fq)
        te = data["te_sel"]
        tok = data["tok_te"]
        labels = datasets.get_label(frame)[te]
        del data, frame
        gc.collect()

        t = torch.tensor(tok, dtype=torch.long)
        mask = (t != 0) & (t != 1)
        probs_full = model.predict_proba(t, use_quann_points=True,
                                         batch=args.chunk).cpu().numpy()
        
        # Threshold-independent metrics (AUC-PR/ROC)
        full_test_metrics = metrics.evaluate(labels, probs_full)
        
        # Val-based threshold selection would happen here in production
        # For now, document that threshold should be selected on val, not test
        full_test_info = {
            "n": int(len(labels)),
            "n_fraud": int(labels.sum()),
            "base_rate": float(labels.mean()),
            "prob_percentiles": {
                "p50": float(np.quantile(probs_full, 0.50)),
                "p90": float(np.quantile(probs_full, 0.90)),
                "p99": float(np.quantile(probs_full, 0.99)),
                "p999": float(np.quantile(probs_full, 0.999)),
                "p9999": float(np.quantile(probs_full, 0.9999)),
            },
            "threshold_protocol": "AUC-PR/ROC are threshold-independent; operational threshold MUST be selected on validation set, not test",
        }
        sel = select_upper_slice(probs_full, labels, args.n_explain)
        tok = tok[sel]
        labels = labels[sel]
        probs = probs_full[sel]
        del t, mask, probs_full
        gc.collect()
        note = ("upper slice of the full natural-rate test; enriched by construction, "
                "not a natural-prevalence estimate")
        strategy = "all fraud + highest-probability normal rows"

    print(f"  {model} | bundled quantizer vocab={fq.vocab_size} K={fq.n_features}", flush=True)
    log_fp.write(f"  {model} | bundled quantizer vocab={fq.vocab_size} K={fq.n_features}\n")

    t = torch.tensor(tok, dtype=torch.long)
    mask = (t != 0) & (t != 1)
    if args.data == "small":
        with torch.no_grad():
            base, _ = model.forward(t, mask, points_override=model.points_for(t, mask))
            probs = torch.sigmoid(base).squeeze(-1).cpu().numpy()
    contribs = point_ablation(model, t, mask, chunk=args.chunk).cpu().numpy()
    sample_metrics = metrics.evaluate(labels, probs)

    aggregate_by_class = {
        "all": aggregate(fq, tok, contribs, top_tokens=args.top_tokens),
        "true_fraud": aggregate(fq, tok[labels == 1], contribs[labels == 1], top_tokens=args.top_tokens),
        "true_normal": aggregate(fq, tok[labels == 0], contribs[labels == 0], top_tokens=args.top_tokens),
    }
    result = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "seed": args.seed,
        "sample": {
            "n": int(len(labels)),
            "n_fraud": int(labels.sum()),
            "n_normal": int((labels == 0).sum()),
            "sample_base_rate": float(labels.mean()),
            "strategy": strategy,
            "note": note,
        },
        "full_test": full_test_info,
        "full_test_metrics": full_test_metrics,
        "model": {
            "D": model.D,
            "d_cloud": model.d_cloud,
            "d_model": model.d_model,
            "vocab": model.vocab_size,
            "K": fq.n_features,
            "quann_points": len(model.token_weights),
        },
        "method": {
            "attribution": "leave-one-point-out logit difference (base logit - logit with point j masked)",
            "positive_means": "feature supports the fraud prediction",
            "caveat": "ablation sensitivity, not SHAP; pair-geometry makes contributions non-additive",
            "token_ranking": "impact_score = mean_abs_logit_contrib * sqrt(count)",
        },
        "sample_metrics": sample_metrics,
        "aggregate": aggregate_by_class,
        "cases": select_cases(fq, labels, probs, tok, contribs,
                              n_cases=args.cases, top_k=args.top_k),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(f"done -> {args.out}")
    log_fp.close()

    if full_test_metrics:
        print(f"  full test AUC-PR={full_test_metrics['auc_pr']:.4f} "
              f"ROC={full_test_metrics['auc_roc']:.4f} "
              f"n={full_test_metrics['n']:,} fraud={full_test_metrics['n_fraud']:,}")
    print(f"  sample AUC-PR={sample_metrics['auc_pr']:.4f} ROC={sample_metrics['auc_roc']:.4f}")
    print("  top features:", ", ".join(
        f"{r['feature']}({r['mean_abs_logit_contrib']:.3f})"
        for r in result["aggregate"]["all"]["features"][:5]))
    print("  top tokens:", ", ".join(
        f"{r['label']}({r['mean_signed_logit_contrib']:+.3f})"
        for r in result["aggregate"]["all"]["top_tokens"][:5]))
    print("  fraud-case positive tokens:", ", ".join(
        f"{r['label']}({r['mean_signed_logit_contrib']:+.3f})"
        for r in result["aggregate"]["true_fraud"]["top_positive"][:5]))
    for case in result["cases"][:3]:
        print(f"  [{case['true_class']}] {case['text']}")


if __name__ == "__main__":
    main()
