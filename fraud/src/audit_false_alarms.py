from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import datasets
import explain_pilot
import metrics
import quantize
import train_quann as tq
from qcloud_fraud import QCloudFraudClassifier
from quantize import FeatureQuantizer

USECOLS = ["User", "Is Fraud?", "Amount", "Time", "Month", "Day", "Year",
           "Zip", "Use Chip", "MCC", "Merchant City", "Merchant Name",
           "Merchant State", "Errors?"]


def select_fp(probs: np.ndarray, labels: np.ndarray, thr: float,
              tail_frac: float = 0.1) -> tuple[np.ndarray, str]:
    """False-alarm population: normals at/above the operational threshold.

    Falls back to the top-`tail_frac` of normals by probability when the
    threshold flags no normal (small dev samples).
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    fp = np.flatnonzero((labels == 0) & (probs >= thr))
    if len(fp):
        return np.sort(fp), "normals predicted fraud at the operational threshold"
    normals = np.flatnonzero(labels == 0)
    k = max(1, int(tail_frac * len(normals)))
    top = np.argsort(-probs[normals], kind="stable")[:k]
    return np.sort(normals[top]), (
        f"no normals at/above threshold; fallback = top {tail_frac:.0%} normals by probability")


def select_tn(probs: np.ndarray, labels: np.ndarray, thr: float,
              n: int) -> np.ndarray:
    """Top-n true negatives (normals just below the threshold) for contrast."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    tn_all = np.flatnonzero((labels == 0) & (probs < thr))
    if len(tn_all) <= n:
        return tn_all
    top = np.argsort(-probs[tn_all], kind="stable")[:n]
    return np.sort(tn_all[top])


def feature_index_of(fq: FeatureQuantizer, token_id: int) -> int | None:
    for j, key in enumerate(fq.feature_order):
        lo = fq.offsets[key]
        if lo <= token_id < lo + fq.n_per_feature[key]:
            return j
    return None


def token_slices(fq: FeatureQuantizer, tok: np.ndarray, base: np.ndarray,
                 contribs: np.ndarray, probs: np.ndarray, thr: float,
                 rows: np.ndarray, top_n: int = 12) -> list[dict]:
    """Per-token cross analysis on `rows`: prevalence, prob split, counterfactual.

    Counterfactual = single-point mask: sigmoid(base - contrib_j) < thr means
    the model would no longer flag the row if that token were absent.
    """
    agg = explain_pilot.aggregate(fq, tok[rows], contribs[rows], top_tokens=top_n)
    base_r = base[rows]
    contribs_r = contribs[rows]
    probs_r = probs[rows]
    tok_r = tok[rows]
    out = []
    for r in agg["top_tokens"]:
        if r["token_id"] < 2:
            continue
        j = feature_index_of(fq, r["token_id"])
        if j is None:
            continue
        has = tok_r[:, j] == r["token_id"]
        if has.any():
            cf_prob = 1.0 / (1.0 + np.exp(-(base_r[has] - contribs_r[has, j])))
        else:
            cf_prob = np.zeros(0)
        cf = cf_prob < thr
        out.append({
            "token_id": r["token_id"],
            "feature": r["feature"],
            "label": r["label"],
            "n_in_rows": int(has.sum()),
            "frac_of_rows": float(has.mean()),
            "mean_prob_with": float(probs_r[has].mean()) if has.any() else None,
            "mean_prob_without": float(probs_r[~has].mean()) if has.any() else None,
            "mean_signed_logit_contrib": r["mean_signed_logit_contrib"],
            "n_unflagged_if_absent": int(cf.sum()),
            "frac_unflagged_if_absent": float(cf.mean()) if has.any() else None,
        })
    return out


def _masked_prob(model: QCloudFraudClassifier, t: torch.Tensor, mask: torch.Tensor,
                 idx: torch.Tensor, cols: list[int]) -> np.ndarray:
    pts = model.points_for(t[idx], mask[idx])
    ab = pts.clone()
    for c in cols:
        ab[:, c, :] = 0.0
    m2 = mask[idx].clone()
    for c in cols:
        m2[:, c] = False
    with torch.no_grad():
        o, _ = model.forward(t[idx], m2, points_override=ab)
    return torch.sigmoid(o).squeeze(-1).cpu().numpy()


def joint_counterfactual(model: QCloudFraudClassifier, fq: FeatureQuantizer,
                         t: torch.Tensor,
                         rows: np.ndarray, targets: list[tuple[int, int]],
                         thr: float, chunk: int = 5000) -> list[dict]:
    """Mask each pair of target tokens together (and each single) and count
    unflagged rows. Non-additive when the joint differs from the singles.
    """
    out = []
    mask = (t != 0) & (t != 1)
    for (ja, tida), (jb, tidb) in itertools.combinations(targets, 2):
        sub = [int(i) for i in rows
               if t[i, ja].item() == tida and t[i, jb].item() == tidb]
        if not sub:
            continue
        idx = torch.tensor(sub, dtype=torch.long)
        unflagged_joint = unflagged_a = unflagged_b = 0
        for s in range(0, len(idx), chunk):
            ii = idx[s:s + chunk]
            for key, cols in (("joint", [ja, jb]), ("a", [ja]), ("b", [jb])):
                p = _masked_prob(model, t, mask, ii, cols)
                n = int((p < thr).sum())
                if key == "joint":
                    unflagged_joint += n
                elif key == "a":
                    unflagged_a += n
                else:
                    unflagged_b += n
        out.append({
            "tokens": [explain_pilot.decode_token(fq, ja, tida),
                       explain_pilot.decode_token(fq, jb, tidb)],
            "n_rows_both": len(sub),
            "n_unflagged_joint": unflagged_joint,
            "frac_unflagged_joint": unflagged_joint / len(sub),
            "n_unflagged_single_a": unflagged_a,
            "frac_unflagged_single_a": unflagged_a / len(sub),
            "n_unflagged_single_b": unflagged_b,
            "frac_unflagged_single_b": unflagged_b / len(sub),
        })
    return out


def audit_cases(fq: FeatureQuantizer, tok: np.ndarray, contribs: np.ndarray,
                probs: np.ndarray, idx: np.ndarray, orig_idx: np.ndarray,
                n: int = 8, top_k: int = 5) -> list[dict]:
    cases = []
    for pos in np.argsort(-probs[idx], kind="stable")[:n]:
        i = int(idx[pos])
        top_idx = np.argsort(-np.abs(contribs[i]))[:top_k]
        top = [{
            "position": int(j),
            "feature": fq.feature_order[int(j)],
            "label": explain_pilot.decode_token(fq, int(j), int(tok[i, j])),
            "logit_contrib": float(contribs[i, j]),
        } for j in top_idx]
        decoded = explain_pilot.decode_row(fq, tok[i])
        cases.append({
            "row": int(orig_idx[i]),
            "p_fraud": float(probs[i]),
            "decoded": decoded,
            "top_contributions": top,
            "text": explain_pilot.explanation_text(float(probs[i]), decoded, top),
        })
    return cases


def run_model_part(model: QCloudFraudClassifier, t: torch.Tensor,
                   mask: torch.Tensor, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """Base logits (full cloud) + leave-one-point-out contributions for `t`."""
    with torch.no_grad():
        base, _ = model.forward(t, mask, points_override=model.points_for(t, mask))
    base = base.squeeze(-1).cpu().numpy()
    contribs = explain_pilot.point_ablation(model, t, mask, chunk=chunk).cpu().numpy()
    return base, contribs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(config.RESULTS_DIR, "quann_full_hybrid.pt"))
    ap.add_argument("--data", choices=["small", "big-test"], default="small")
    ap.add_argument("--n-rows", type=int, default=None)
    ap.add_argument("--n-fraud", type=int, default=200)
    ap.add_argument("--n-normal", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=4700000)
    ap.add_argument("--n-train", type=int, default=1020000)
    ap.add_argument("--n-val", type=int, default=100000)
    ap.add_argument("--train-fraud-frac", type=float, default=0.02)
    ap.add_argument("--test-frac", type=float, default=0.0013)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--fp-cap", type=int, default=None)
    ap.add_argument("--tn-n", type=int, default=2000)
    ap.add_argument("--tail-frac", type=float, default=0.1)
    ap.add_argument("--slices", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--dom-pairs", type=int, default=3,
                    help="top-impact tokens for the dominant joint counterfactual")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.chunk is None:
        args.chunk = 256 if args.data == "small" else 5000
    if args.out is None:
        name = "false_alarm_audit_dev.json" if args.data == "small" else "false_alarm_audit.json"
        args.out = os.path.join(config.RESULTS_DIR, name)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    log_name = ("run_false_alarm_audit_dev.log" if args.data == "small"
                else "run_false_alarm_audit.log")
    log_path = os.path.join(config.LOGS_DIR, log_name)
    log_fp = open(log_path, "a")

    def log(msg: str):
        line = f"=== [{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg} ==="
        print(line, flush=True)
        log_fp.write(line + "\n")

    log("FRAUD FALSE-ALARM AUDIT — high-prob normal population, per-token cross analysis")
    print(f"  checkpoint: {args.checkpoint}", flush=True)
    log_fp.write(f"  checkpoint: {args.checkpoint}\n")

    model = QCloudFraudClassifier.load_quann(args.checkpoint, torch.device("cpu"))
    raw = torch.load(args.checkpoint, map_location="cpu")
    fallback = quantize.FeatureQuantizer(specs=quantize.LEGACY_QUANT_SPECS)
    fq = explain_pilot.load_quantizer(args.checkpoint, fallback)

    if args.data == "small":
        frame = datasets.load_small()
        y = datasets.get_label(frame)
        rng = np.random.default_rng(args.seed)
        fraud_idx = np.flatnonzero(y == 1)
        normal_idx = np.flatnonzero(y == 0)
        sel = np.concatenate([
            rng.choice(fraud_idx, size=min(args.n_fraud, len(fraud_idx)),
                       replace=args.n_fraud > len(fraud_idx)),
            rng.choice(normal_idx, size=min(args.n_normal, len(normal_idx)),
                       replace=args.n_normal > len(normal_idx)),
        ])
        if args.n_rows is not None:
            sel = rng.choice(sel, size=min(args.n_rows, len(sel)), replace=False)
        rng.shuffle(sel)
        if not fq._fitted:
            fq = quantize.FeatureQuantizer(specs=quantize.LEGACY_QUANT_SPECS).fit(frame.iloc[sel])
        tok = fq.transform(frame.iloc[sel])
        labels = y[sel]
        del frame
        gc.collect()
        t = torch.tensor(tok, dtype=torch.long)
        mask = (t != 0) & (t != 1)
        with torch.no_grad():
            base, _ = model.forward(t, mask, points_override=model.points_for(t, mask))
        probs = torch.sigmoid(base).squeeze(-1).cpu().numpy()
    else:
        if explain_pilot.free_gb() < 12:
            raise SystemExit(f"aborted: only {explain_pilot.free_gb():.1f}GB reclaimable; "
                             "big-test load needs >=12GB")
        print(f"  reading {config.BIG_CSV} (usecols={len(USECOLS)}) ...", flush=True)
        log_fp.write(f"  reading {config.BIG_CSV} (usecols={len(USECOLS)}) ...\n")
        frame = pd.read_csv(config.BIG_CSV, usecols=USECOLS)
        print(f"  rows={len(frame):,} free_gb={explain_pilot.free_gb():.1f}", flush=True)
        log_fp.write(f"  rows={len(frame):,} free_gb={explain_pilot.free_gb():.1f}\n")
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
        probs = model.predict_proba(t, use_quann_points=True,
                                    batch=args.chunk).cpu().numpy()
        del t, mask
        gc.collect()

    if args.threshold is None:
        thr = metrics.best_threshold(labels, probs)
    else:
        thr = args.threshold
    flagged = (probs >= thr)

    fp_idx, fp_note = select_fp(probs, labels, thr, tail_frac=args.tail_frac)
    if args.fp_cap is not None and len(fp_idx) > args.fp_cap:
        order = np.argsort(-probs[fp_idx], kind="stable")[:args.fp_cap]
        fp_idx = np.sort(fp_idx[order])
        fp_note += f"; capped to top {args.fp_cap} by probability"
    tn_idx = select_tn(probs, labels, thr, args.tn_n)

    print(f"  thr={thr:.4f} fp={len(fp_idx):,} tn_sample={len(tn_idx):,} "
          f"flagged_total={int(flagged.sum()):,}", flush=True)
    log_fp.write(f"  thr={thr:.4f} fp={len(fp_idx):,} tn_sample={len(tn_idx):,} "
                 f"flagged_total={int(flagged.sum()):,}\n")

    rows = np.concatenate([fp_idx, tn_idx])
    class_of = np.concatenate([np.ones(len(fp_idx)), np.zeros(len(tn_idx))])
    t = torch.tensor(tok[rows], dtype=torch.long)
    mask = (t != 0) & (t != 1)
    base, contribs = run_model_part(model, t, mask, chunk=args.chunk)
    probs_rows = probs[rows]

    fp_rows = np.flatnonzero(class_of == 1)
    tn_rows = np.flatnonzero(class_of == 0)

    def pop_aggregate(idx: np.ndarray) -> dict:
        return explain_pilot.aggregate(fq, tok[rows][idx], contribs[idx],
                                       top_tokens=args.slices)

    fp_aggregate = pop_aggregate(fp_rows) if len(fp_rows) else {}
    tn_aggregate = pop_aggregate(tn_rows) if len(tn_rows) else {}

    slices = token_slices(fq, tok[rows], base, contribs, probs_rows, thr, fp_rows,
                          top_n=args.slices) if len(fp_rows) else []

    targets = []
    for r in fp_aggregate.get("top_positive", []):
        if r["token_id"] < 2:
            continue
        j = feature_index_of(fq, r["token_id"])
        if j is not None:
            targets.append((j, r["token_id"]))
        if len(targets) >= args.pairs:
            break
    joint = joint_counterfactual(model, fq, t, fp_rows, targets, thr,
                                  chunk=args.chunk) if len(fp_rows) and len(targets) >= 2 else []

    # dominant (high-population-share) tokens: same non-additivity test at scale.
    # `targets` above are rare-but-strong tokens (n~1); these drive the FP
    # population, so their joint carries a meaningful unflag rate.
    dom_targets = []
    for r in sorted(fp_aggregate.get("top_positive", []),
                    key=lambda r: r.get("count", 0), reverse=True):
        if r["token_id"] < 2:
            continue
        j = feature_index_of(fq, r["token_id"])
        if j is not None:
            dom_targets.append((j, r["token_id"]))
        if len(dom_targets) >= args.dom_pairs:
            break
    dom_joint = (joint_counterfactual(model, fq, t, fp_rows, dom_targets, thr,
                                      chunk=args.chunk)
                 if len(fp_rows) and len(dom_targets) >= 2 else [])

    tp = int(np.sum((labels == 1) & flagged))
    fp_tot = int(np.sum((labels == 0) & flagged))
    fn = int(np.sum((labels == 1) & ~flagged))
    result = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "seed": args.seed,
        "population": {
            "n": int(len(labels)),
            "n_fraud": int(labels.sum()),
            "base_rate": float(labels.mean()),
            "threshold": float(thr),
            "threshold_rule": ("best-F2 grid" if args.threshold is None
                               else "user-specified"),
            "flagged": int(flagged.sum()),
            "tp": tp, "fp": fp_tot, "fn": fn,
            "precision": tp / (tp + fp_tot) if tp + fp_tot else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "fp_population": {
                "n": int(len(fp_idx)),
                "note": fp_note,
                "prob_percentiles": {
                    "p50": float(np.quantile(probs[fp_idx], 0.50)) if len(fp_idx) else None,
                    "p90": float(np.quantile(probs[fp_idx], 0.90)) if len(fp_idx) else None,
                    "p99": float(np.quantile(probs[fp_idx], 0.99)) if len(fp_idx) else None,
                    "max": float(probs[fp_idx].max()) if len(fp_idx) else None,
                },
            },
            "tn_sample": {"n": int(len(tn_idx)),
                          "note": "top normals below the threshold, for contrast"},
        },
        "fp_aggregate": fp_aggregate,
        "tn_aggregate": tn_aggregate,
        "slices": slices,
        "joint": joint,
        "dom_joint": dom_joint,
        "cases": audit_cases(fq, tok[rows], contribs, probs_rows, fp_rows,
                             np.asarray(rows[fp_rows]),
                             n=args.cases, top_k=args.top_k) if len(fp_rows) else [],
        "method": {
            "attribution": "leave-one-point-out logit difference (base logit - logit with point j masked)",
            "positive_means": "feature supports the fraud prediction",
            "counterfactual": "single-point mask: sigmoid(base - contrib_j) < threshold => row unflagged",
            "joint": "both points masked in one forward (non-additive; not the sum of singles)",
            "dom_joint": "same joint test over the dominant (most-populous) positive tokens; statistically meaningful n vs the rare-token joint",
            "caveat": "ablation sensitivity, not SHAP; pair-geometry makes contributions non-additive",
            "population": fp_note,
        },
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(f"done -> {args.out}")
    log_fp.close()

    print(f"  thr={thr:.4f}: tp={tp:,} fp={fp_tot:,} fn={fn:,} "
          f"prec={result['population']['precision']} "
          f"recall={result['population']['recall']}")
    if slices:
        print("  fp slices (top):", ", ".join(
            f"{s['label']}(n={s['n_in_rows']},unflag={s['frac_unflagged_if_absent'] or 0:.2f})"
            for s in slices[:5]))
    if joint:
        print("  joint (rare-strong):", ", ".join(
            f"{' + '.join(j['tokens'])}(n={j['n_rows_both']},unflag={j['frac_unflagged_joint']:.2f})"
            for j in joint[:3]))
    if dom_joint:
        print("  joint (dominant):", ", ".join(
            f"{' + '.join(j['tokens'])}(n={j['n_rows_both']},unflag={j['frac_unflagged_joint']:.2f})"
            for j in dom_joint[:3]))
    for case in result["cases"][:3]:
        print(f"  [false-alarm] {case['text']}")


if __name__ == "__main__":
    main()
