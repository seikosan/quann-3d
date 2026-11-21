"""Standard ML layer — applied fraud baselines on the full IBM dataset.

Models (all on the same leakage-safe features, GroupKFold-by-User splits):
    lr     : LogisticRegression (balanced)
    rf     : RandomForest (balanced; row-subsampled at scale)
    gb     : HistGradientBoosting (scalable GBT; pos-weight for imbalance)
    mlp    : small torch MLP (balanced BCE; row-subsampled at scale)

Imbalance: class_weight / pos-weight + operational threshold tuned on val
(best F2). Primary metric: AUC-PR.

Run:
    python3 fraud/src/train_standard.py                 # full 24.4M, all models
    python3 fraud/src/train_standard.py --n-rows 300000 --models lr gb   # quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import datasets
import features
import metrics


def _torch_mlp(X, y, Xv, yv, seed=config.SEED, epochs=25, bs=512, lr=1e-3,
               hidden=64, device=None):
    """A plain small MLP (the *standard* layer — not the cloud). Returns val prob."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    device = device or ("cuda" if torch.cuda.is_available() else
                        "cpu")  # MPS left free for the live CloudLM run
    # Standardize with TRAIN stats, applied to ALL splits (fix: capture mu/sd
    # before overwriting X, so val/test use the same train scaling).
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    X = (X - mu) / sd
    Xv = (Xv - mu) / sd
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    # Cap the class weight: at ~1000:1 imbalance the raw ratio makes a small
    # MLP collapse to "predict all fraud". A capped weight trains a sane (if
    # weak) baseline; trees (RF/GBT) handle the raw imbalance better.
    raw_pw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    pos_weight = torch.tensor(min(raw_pw, 20.0), dtype=torch.float32, device=device)
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, 1),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = lambda lo, t: torch.nn.functional.binary_cross_entropy_with_logits(
        lo, t, pos_weight=pos_weight)
    dl = DataLoader(TensorDataset(xt, yt), batch_size=bs, shuffle=True)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(torch.tensor(Xv, dtype=torch.float32, device=device))).squeeze(-1).cpu().numpy()
    return pv


def run_model(name, Xtr, ytr, Xva, yva, Xte, yte, n_full, args):
    t0 = time.time()
    subsample = args.subsample and len(Xtr) > args.subsample
    si = np.arange(len(Xtr))
    if subsample:
        rng = np.random.default_rng(config.SEED)
        si = rng.choice(si, size=args.subsample, replace=False)
    Xt, yt = Xtr[si], ytr[si]

    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5)
        clf.fit(Xt, yt)
        pva = clf.predict_proba(Xva)[:, 1]
        pte = clf.predict_proba(Xte)[:, 1]
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=args.rf_trees, class_weight="balanced",
                                     n_jobs=-1, random_state=config.SEED)
        clf.fit(Xt, yt)
        pva = clf.predict_proba(Xva)[:, 1]
        pte = clf.predict_proba(Xte)[:, 1]
    elif name == "gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        pos_w = (yt == 0).sum() / max((yt == 1).sum(), 1)
        clf = HistGradientBoostingClassifier(
            max_iter=args.gb_iter, max_leaf_nodes=63, learning_rate=0.08,
            random_state=config.SEED)
        clf.fit(Xt, yt, sample_weight=np.where(yt == 1, pos_w, 1.0))
        pva = clf.predict_proba(Xva)[:, 1]
        pte = clf.predict_proba(Xte)[:, 1]
    elif name == "mlp":
        pva = _torch_mlp(Xt, yt, Xva, yva)
        pte = _torch_mlp(Xt, yt, Xte, yte)
    else:
        raise ValueError(name)

    thr = metrics.best_threshold(yva, pva)
    m_va = metrics.evaluate(yva, pva, threshold=thr)
    m_te = metrics.evaluate(yte, pte, threshold=thr)
    m_te["model"] = name
    m_te["trained_rows"] = int(len(Xt))
    m_te["subsampled"] = bool(subsample)
    m_te["fit_sec"] = round(time.time() - t0, 1)
    return m_te, m_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=None, help="limit big rows (quick)")
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "gb", "mlp"])
    ap.add_argument("--subsample", type=int, default=1_000_000,
                    help="max rows used by rf/mlp (0=off)")
    ap.add_argument("--rf-trees", type=int, default=400)
    ap.add_argument("--gb-iter", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models = list(args.models)
    print(f"Loading data (n_rows={args.n_rows}) ...")
    df = datasets.load_big(n_rows=args.n_rows)
    y = datasets.get_label(df)
    tr, va, te = datasets.split_by_user(df)
    print(f"n={len(df):,} fraud={y.sum():,} | train={len(tr):,} val={len(va):,} test={len(te):,} "
          f"(fraud {y[tr].sum()}/{y[va].sum()}/{y[te].sum()})")

    ft = features.FeatureTransformer().fit(df.iloc[tr])
    Xtr = ft.transform(df.iloc[tr]); Xva = ft.transform(df.iloc[va]); Xte = ft.transform(df.iloc[te])
    print(f"features: {Xtr.shape[1]} -> {ft.feature_names[:6]} ...")

    results = {}
    for name in models:
        m_te, m_va = run_model(name, Xtr, y[tr], Xva, y[va], Xte, y[te], len(df), args)
        results[name] = m_te
        print(f"[{name:4s}] val: {metrics.format_metrics(m_va)}")
        print(f"[{name:4s}] TEST: {metrics.format_metrics(m_te)}  "
              f"({m_te['fit_sec']}s, rows={m_te['trained_rows']:,})")

    out = args.out or os.path.join(config.RESULTS_DIR, "standard_full.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"models": results,
                   "split": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
                   "n_rows": int(len(df)), "features": ft.feature_names}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
