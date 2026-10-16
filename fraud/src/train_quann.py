"""QUANN fraud layer — the research's point-cloud architecture, 3 training regimes.

All three configs share the SAME QCloudFraudClassifier + the SAME quantized
point-cloud input; only how the representation is learned differs:

    pure     : network frozen (random init) — QUANN 0th-order narrows the points.
               (the strongest "no-backprop" claim)
    hybrid   : network (cloud_encoder+class_head) backprop + QUANN narrows points.
    backprop : points from a learned emb->to_nd; everything trained by backprop
               (the fair reference / ceiling).

Data: the full IBM file, group-split by User (no leakage). Train/val are
class-enriched (strong signal for the slow 0th-order search); test is a
separate, group-disjoint, lower-fraud-rate set for honest evaluation.

Runs on CPU (MPS is held free for the live CloudLM 1.5M run).

Run:
    python3 fraud/src/train_quann.py --smoke                       # tiny, verify
    python3 fraud/src/train_quann.py                                # large (50-100K)
    python3 fraud/src/train_quann.py --configs pure hybrid --epochs 3
"""
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
import quantize
import metrics
from qcloud_fraud import QCloudFraudClassifier


def sample_from_users(df, users, n_total, fraud_frac, seed):
    """Rows belonging to `users`, class-enriched to ~fraud_frac. Position indices."""
    keep = np.isin(df[config.GROUP_COL].to_numpy(), list(users))
    sub = np.flatnonzero(keep)
    y = datasets.get_label(df)[sub]
    rng = np.random.default_rng(seed)
    fraud = sub[y == 1]
    normal = sub[y == 0]
    n_f = min(int(round(n_total * fraud_frac)), len(fraud))
    n_n = min(n_total - n_f, len(normal))
    sel = np.concatenate([rng.choice(fraud, n_f, replace=False),
                          rng.choice(normal, n_n, replace=False)])
    rng.shuffle(sel)
    return sel


def user_folds(n_groups, n_splits=10, seed=config.SEED):
    """Deterministic group -> fold assignment. Returns (fold_of_group, n_groups)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)
    fold = np.empty(n_groups, dtype=int)
    for i, g in enumerate(perm):
        fold[g] = i % n_splits
    return fold


def build_tensors(tokens: np.ndarray, y: np.ndarray, device):
    t = torch.tensor(tokens, dtype=torch.long, device=device)
    mask = (t != 0) & (t != 1)
    yy = torch.tensor(y.astype(np.float32), device=device)
    return t, mask, yy


def build_data(df, n_train, n_val, n_test, train_frac, test_frac, seed=config.SEED):
    """Deterministic group-disjoint splits + quantized token matrices.

    Returns a dict so both training and figure-making can rebuild the exact same
    test set from the full dataset (same seed -> same users/samples/quantizer).
    """
    y = datasets.get_label(df)
    groups = df[config.GROUP_COL].to_numpy()
    uniq = np.unique(groups)
    fold = user_folds(len(uniq), n_splits=10)
    fold_of_user = {int(u): int(f) for u, f in zip(uniq, fold)}
    tr_users = sorted({int(u) for u in uniq if fold_of_user[int(u)] in range(7)})
    va_users = sorted({int(u) for u in uniq if fold_of_user[int(u)] == 7})
    te_users = sorted({int(u) for u in uniq if fold_of_user[int(u)] >= 8})

    tr_sel = sample_from_users(df, tr_users, n_train, train_frac, seed)
    va_sel = sample_from_users(df, va_users, n_val, train_frac, seed + 1)
    te_sel = sample_from_users(df, te_users, n_test, test_frac, seed + 2)

    fq = quantize.FeatureQuantizer()
    fq.fit(df.iloc[tr_sel])
    return {
        "fq": fq,
        "tr_sel": tr_sel, "va_sel": va_sel, "te_sel": te_sel,
        "tok_tr": fq.transform(df.iloc[tr_sel]),
        "tok_va": fq.transform(df.iloc[va_sel]),
        "tok_te": fq.transform(df.iloc[te_sel]),
        "y": y,
        "users": {"tr": len(tr_users), "va": len(va_users), "te": len(te_users)},
    }


def train_config(cfg: str, model, ttr, mtr, ytr, tva, mva, yva, tte, mte, yte,
                 epochs, device, log):
    """Train one regime; returns (test_metrics, val_metrics, convergence_info)."""
    model.to(device)
    K = ttr.size(1)
    n = ttr.size(0)
    rng_np = np.random.default_rng(config.SEED)

    use_backprop = cfg in ("hybrid", "backprop")
    if use_backprop:
        params = (list(model.cloud_encoder.parameters()) + list(model.class_head.parameters())
                  if cfg == "hybrid" else list(model.parameters()))
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    # Class-balance the loss (capped) so the model/search stays sane on the
    # imbalanced full-data pool (~1.7% fraud). =1.0 (no-op) when balanced.
    raw_pw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    pos_weight = torch.tensor(min(raw_pw, 20.0), device=device)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        order = rng_np.permutation(n)
        ep_t = time.time()
        r = c = u = 0
        base_e = fin_e = 0.0
        for i in order:
            toks = ttr[i:i + 1]
            msk = mtr[i:i + 1]
            tgt = int(ytr[i])
            if cfg == "backprop":
                # batched backprop over the learned emb->to_nd path
                model.train()
                logits, _ = model(toks, msk)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), ytr[i:i + 1], pos_weight=pos_weight)
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                # QUANN 0th-order (points). hybrid also backprops the reader.
                base = model.points_for(toks, msk)
                if cfg == "hybrid":
                    model.train()
                    logits, _ = model(toks, msk, points_override=base.detach())
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.squeeze(-1), ytr[i:i + 1], pos_weight=pos_weight)
                    opt.zero_grad(); loss.backward(); opt.step()
                    model.eval()
                stats = model.update_token_weights(
                    toks, msk, tgt, model.points_for(toks, msk),
                    search_rule="search" if cfg == "hybrid" else "classic",
                    pos_weight=pos_weight)
                r += stats["rewarded"]; c += stats["censured"]; u += stats["unchanged"]
                base_e += stats.get("base_error", 0.0); fin_e += stats.get("final_error", 0.0)

        trials = r + c
        dt = time.time() - ep_t
        conv = model.convergence_info()
        log(f"    [{cfg}] ep{ep}: R={r} C={c} U={u} ratio="
            f"{(r/trials if trials else 0):.3f} base_err={base_e/max(n,1):.4f} "
            f"-> final={fin_e/max(n,1):.4f} width={conv['mean_width']:.3f} "
            f"conv={conv['converged_tokens']}/{conv['total_tokens']} ({dt:.0f}s)")

    # threshold on val, evaluate on test
    pva = model.predict_proba(tva, use_quann_points=(cfg != "backprop")).cpu().numpy()
    thr = metrics.best_threshold(yva.cpu().numpy(), pva)
    pte = model.predict_proba(tte, use_quann_points=(cfg != "backprop")).cpu().numpy()
    m_va = metrics.evaluate(yva.cpu().numpy(), pva, threshold=thr)
    m_te = metrics.evaluate(yte.cpu().numpy(), pte, threshold=thr)
    conv = model.convergence_info()
    m_te.update({"config": cfg, "train_sec": round(time.time() - t0, 1),
                 "quann_points": conv["total_tokens"],
                 "converged_points": conv["converged_tokens"],
                 "mean_width": conv["mean_width"]})
    return m_te, m_va, conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=60000, help="enriched train pool size")
    ap.add_argument("--n-val", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--train-fraud-frac", type=float, default=0.5)
    ap.add_argument("--test-fraud-frac", type=float, default=0.05)
    ap.add_argument("--configs", nargs="+", default=["pure", "hybrid", "backprop"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--D", type=int, default=8, help="d_cloud_dim (point dimension)")
    ap.add_argument("--d-cloud", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the loop")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.n_test = 1500, 400, 400
        args.epochs = 1
        args.D, args.d_cloud, args.d_model = 6, 32, 16
        args.configs = ["pure", "hybrid", "backprop"]

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cpu")  # MPS held free for the live CloudLM run

    print("Loading full dataset ...")
    df = datasets.load_big()
    data = build_data(df, args.n_train, args.n_val, args.n_test,
                      args.train_fraud_frac, args.test_fraud_frac)
    fq = data["fq"]; y = data["y"]
    tr_sel, va_sel, te_sel = data["tr_sel"], data["va_sel"], data["te_sel"]
    print(f"users | train={data['users']['tr']} val={data['users']['va']} "
          f"test={data['users']['te']}")
    print(f"train={len(tr_sel):,} (fraud {y[tr_sel].sum():,})  "
          f"val={len(va_sel):,} (fraud {y[va_sel].sum():,})  "
          f"test={len(te_sel):,} (fraud {y[te_sel].sum():,}, base {y[te_sel].mean():.3%})")
    print(f"quantized: vocab={fq.vocab_size} K={fq.n_features} points/tx")

    ytr, yva, yte = y[tr_sel], y[va_sel], y[te_sel]
    ttr, mtr, ytr_t = build_tensors(data["tok_tr"], ytr, device)
    tva, mva, yva_t = build_tensors(data["tok_va"], yva, device)
    tte, mte, yte_t = build_tensors(data["tok_te"], yte, device)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    results = {}
    for cfg in args.configs:
        print(f"\n{'='*64}\n  CONFIG: {cfg}\n{'='*64}")
        model = QCloudFraudClassifier(vocab_size=fq.vocab_size, d_model=args.d_model,
                                      d_cloud=args.d_cloud, d_cloud_dim=args.D).to(device)
        print(f"  {model} | params={sum(p.numel() for p in model.parameters()):,}")
        m_te, m_va, conv = train_config(
            cfg, model, ttr, mtr, ytr_t, tva, mva, yva_t, tte, mte, yte_t,
            args.epochs, device, log=print)
        results[cfg] = {"test": m_te, "val": m_va}
        print(f"  [{cfg}] VAL  : {metrics.format_metrics(m_va)}")
        print(f"  [{cfg}] TEST : {metrics.format_metrics(m_te)}")
        # save checkpoint
        cp = os.path.join(config.RESULTS_DIR, f"quann_{cfg}.pt")
        model.save_quann(cp, extra={"test": {k: v for k, v in m_te.items() if isinstance(v, (int, float))}})
        print(f"  saved -> {cp}")

    out = args.out or os.path.join(config.RESULTS_DIR,
                                   f"quann_{'smoke' if args.smoke else 'large'}.json")
    with open(out, "w") as f:
        json.dump({"configs": results,
                   "split": {"train": int(len(tr_sel)), "val": int(len(va_sel)), "test": int(len(te_sel))},
                   "test_base_rate": float(yte.mean()),
                   "model": {"D": args.D, "d_cloud": args.d_cloud, "d_model": args.d_model,
                             "epochs": args.epochs, "vocab": fq.vocab_size, "K": fq.n_features}},
                   f, indent=2)
    print(f"\nSaved -> {out}")
    print("\n=== SUMMARY (test) ===")
    for cfg in args.configs:
        print(f"  {cfg:10s} {metrics.format_metrics(results[cfg]['test'])}")


if __name__ == "__main__":
    main()
