from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import datasets
import metrics
import quantize
from qcloud_fraud import QCloudFraudClassifier
from quantize import FeatureQuantizer


def load_quantizer(checkpoint_path: str, fallback: FeatureQuantizer) -> FeatureQuantizer:
    raw = torch.load(checkpoint_path, map_location="cpu")
    if raw.get("quantizer"):
        return FeatureQuantizer.from_dict(raw["quantizer"])
    return fallback


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
    ap.add_argument("--data", choices=["small", "big"], default="small")
    ap.add_argument("--n-rows", type=int, default=None)
    ap.add_argument("--n-fraud", type=int, default=200)
    ap.add_argument("--n-normal", type=int, default=800)
    ap.add_argument("--top-tokens", type=int, default=20)
    ap.add_argument("--cases", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "explain_pilot.json"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(config.LOGS_DIR, "run_explain_pilot.log")
    log_fp = open(log_path, "a")

    def log(msg: str):
        line = f"=== [{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg} ==="
        print(line, flush=True)
        log_fp.write(line + "\n")

    log("FRAUD EXPLANATION PILOT — point-ablation attribution")
    print(f"  checkpoint: {args.checkpoint}", flush=True)
    log_fp.write(f"  checkpoint: {args.checkpoint}\n")

    frame = datasets.load_small() if args.data == "small" else datasets.load_big(args.n_rows)
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

    model = QCloudFraudClassifier.load_quann(args.checkpoint, torch.device("cpu"))
    fallback = quantize.FeatureQuantizer(specs=quantize.LEGACY_QUANT_SPECS).fit(frame.iloc[sel])
    fq = load_quantizer(args.checkpoint, fallback)
    print(f"  {model} | bundled quantizer vocab={fq.vocab_size} K={fq.n_features}", flush=True)
    log_fp.write(f"  {model} | bundled quantizer vocab={fq.vocab_size} K={fq.n_features}\n")

    tok = fq.transform(frame.iloc[sel])
    labels = y[sel]
    t = torch.tensor(tok, dtype=torch.long)
    mask = (t != 0) & (t != 1)
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
            "n": int(len(sel)),
            "n_fraud": int(labels.sum()),
            "n_normal": int((labels == 0).sum()),
            "sample_base_rate": float(labels.mean()),
            "note": "stratified development sample; not the natural 4.7M held-out population",
        },
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
