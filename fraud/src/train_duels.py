"""Head-to-head tree and FT-Transformer baselines for IBM fraud data.

The full run uses exactly the leakage-safe features and User-group split from
``train_standard.py``. ``--smoke`` takes deterministic, natural-rate samples
from each of those same partitions so the complete pipeline can be checked
without fitting or scoring millions of rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import datasets
import features
import metrics
import train_quann


MODEL_NAMES = ("xgboost", "lightgbm", "ftt")
METRIC_KEYS = ("auc_roc", "auc_pr", "fbeta", "precision", "recall", "lift")
QUANN_SPLIT = {"train": 1_020_000, "val": 100_000, "test": 4_700_000}


def _step(number: int, name: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== [{stamp}] {number}/5: {name} ===", flush=True)


def _sample_partition(indices: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(indices) <= n:
        return indices
    return np.sort(np.random.default_rng(seed).choice(indices, size=n, replace=False))


def _torch_threads() -> None:
    import torch

    # Use CPU explicitly and avoid OpenMP oversubscription on these small
    # transformer batches (the tree models independently use their n_jobs).
    torch.set_num_threads(1)


def _hashed_categoricals(X: np.ndarray, cat_idx: np.ndarray,
                         max_buckets: int = 500) -> np.ndarray:
    """Stable per-value hash buckets for frequency/one-hot feature columns."""
    if not len(cat_idx):
        return np.empty((len(X), 0), dtype=np.int64)
    vals = np.rint(X[:, cat_idx].astype(np.float64) * 1_000_000_000).astype(np.int64)
    # SplitMix64 finalizer gives neighboring frequency values well-distributed
    # buckets. Zero remains a dedicated missing/unknown bucket.
    hashed = vals.astype(np.uint64)
    hashed += np.uint64(0x9E3779B97F4A7C15)
    hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    hashed ^= hashed >> np.uint64(31)
    out = (hashed % np.uint64(max_buckets - 1)).astype(np.int64) + 1
    out[vals == 0] = 0
    return out


def _categorical_columns(feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Categorical outputs in FeatureTransformer's numeric feature matrix."""
    cat = np.asarray([
        i for i, name in enumerate(feature_names)
        if name.startswith("use_chip=") or name.startswith("err_") or name.endswith("_freq")
    ], dtype=np.int64)
    continuous = np.asarray([i for i in range(len(feature_names)) if i not in set(cat)],
                            dtype=np.int64)
    return continuous, cat


def _make_ftt(n_cont: int, n_cat: int, d_model: int = 128,
              n_layers: int = 2, n_heads: int = 8, cat_dim: int = 16,
              ff_dim: int = 64):
    import torch
    from torch import nn

    class FTTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_cont = n_cont
            self.n_cat = n_cat
            self.value_mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(),
                                           nn.Linear(32, d_model))
            self.cont_field = nn.Embedding(max(n_cont, 1), d_model)
            self.cat_values = nn.Embedding(500, cat_dim)
            self.cat_fields = nn.Embedding(max(n_cat, 1), cat_dim)
            self.cat_projection = nn.Linear(cat_dim, d_model)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)

        def forward(self, cont, cat):
            pieces = [self.value_mlp(cont.unsqueeze(-1)) +
                      self.cont_field.weight[:self.n_cont].unsqueeze(0)]
            if self.n_cat:
                fields = self.cat_fields.weight[:self.n_cat].unsqueeze(0)
                pieces.append(self.cat_projection(self.cat_values(cat) + fields))
            tokens = torch.cat([self.cls.expand(cont.shape[0], -1, -1), *pieces], dim=1)
            return self.head(self.norm(self.encoder(tokens)[:, 0])).squeeze(-1)

    return FTTransformer()


def _ftt_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray,
                     Xte: np.ndarray, feature_names: list[str], *, seed: int,
                     epochs: int = 3, batch_size: int = 2048, lr: float = 1e-3,
                     d_model: int = 128, n_layers: int = 2, n_heads: int = 8,
                     cat_dim: int = 16, ff_dim: int = 64,
                     pos_weight_cap: float = 20.0):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _torch_threads()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    cont_idx, cat_idx = _categorical_columns(feature_names)
    mu = Xtr[:, cont_idx].mean(axis=0) if len(cont_idx) else np.empty(0)
    sd = Xtr[:, cont_idx].std(axis=0) + 1e-6 if len(cont_idx) else np.empty(0)

    def encode(X):
        cont = ((X[:, cont_idx] - mu) / sd).astype(np.float32, copy=False)
        cats = _hashed_categoricals(X, cat_idx)
        return (torch.from_numpy(np.ascontiguousarray(cont)),
                torch.from_numpy(np.ascontiguousarray(cats)))

    xcont, xcat = encode(Xtr)
    xv_cont, xv_cat = encode(Xva)
    xt_cont, xt_cat = encode(Xte)
    model = _make_ftt(len(cont_idx), len(cat_idx), d_model, n_layers, n_heads,
                      cat_dim, ff_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    raw_pw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    pos_weight = torch.tensor(min(raw_pw, pos_weight_cap), dtype=torch.float32,
                              device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    target = torch.from_numpy(ytr.astype(np.float32, copy=False))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(xcont, xcat, target), batch_size=batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    start = time.time()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for cb, kb, yb in loader:
            logits = model(cb.to(device), kb.to(device))
            loss = lossf(logits, yb.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach())
        print(f"    FTT epoch {epoch + 1}/{epochs}: loss={loss_sum / max(len(loader), 1):.5f}",
              flush=True)

    def predict(c, k):
        model.eval()
        scores = []
        with torch.no_grad():
            for start_idx in range(0, len(c), 50_000):
                logits = model(c[start_idx:start_idx + 50_000].to(device),
                               k[start_idx:start_idx + 50_000].to(device))
                scores.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)

    pva, pte = predict(xv_cont, xv_cat), predict(xt_cont, xt_cat)
    details = {
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "d_model": d_model, "layers": n_layers, "heads": n_heads,
        "cat_embedding_dim": cat_dim, "ff_dim": ff_dim,
        "pos_weight": min(raw_pw, pos_weight_cap), "pos_weight_cap": pos_weight_cap,
        "params": sum(p.numel() for p in model.parameters()),
        "continuous_features": int(len(cont_idx)), "categorical_features": int(len(cat_idx)),
        "fit_sec": round(time.time() - start, 1),
    }
    return model, pva, pte, details, {"mu": mu, "sd": sd, "continuous_idx": cont_idx,
                                     "categorical_idx": cat_idx}


def _tree_fit_predict(name: str, Xtr: np.ndarray, ytr: np.ndarray,
                      Xva: np.ndarray, yva: np.ndarray, Xte: np.ndarray, *,
                      seed: int, n_estimators: int, early_stopping_rounds: int,
                      max_depth: int, learning_rate: float, n_jobs: int):
    pos = max(int(np.sum(ytr == 1)), 1)
    neg = int(np.sum(ytr == 0))
    scale_pos_weight = neg / pos
    start = time.time()
    if name == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            tree_method="hist", n_jobs=n_jobs, objective="binary:logistic",
            eval_metric="aucpr", scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=early_stopping_rounds, random_state=seed,
            subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        config_data = {"n_estimators": n_estimators, "best_iteration": int(clf.best_iteration),
                       "max_depth": max_depth, "learning_rate": learning_rate,
                       "tree_method": "hist", "n_jobs": n_jobs,
                       "scale_pos_weight": scale_pos_weight,
                       "early_stopping_rounds": early_stopping_rounds}
    else:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        # LightGBM's native parallel training path segfaulted on the 1.02M-row
        # QUANN matrix on macOS/ARM (n_jobs=4 and 18); serial training completed
        # on the same finite, contiguous float32 data, so keep this workaround.
        lgbm_n_jobs = 1
        # A 2M-row standard-split sweep selected no positive reweighting:
        # scale_pos_weight=1 gave val sklearn AP 0.505 (best iter 34), versus
        # 0.454 at 2 and 0.423 at 5; the raw ~808 ratio had collapsed to base
        # rate. Keep AP as the stopper because it directly matches our target.
        lgbm_pos_weight = 1.0
        clf = LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth, num_leaves=31,
            learning_rate=learning_rate, n_jobs=lgbm_n_jobs, objective="binary",
            metric="average_precision", scale_pos_weight=lgbm_pos_weight,
            random_state=seed, verbosity=-1, subsample=0.85, colsample_bytree=0.9,
        )
        clf.fit(Xtr, ytr, eval_X=Xva, eval_y=yva,
                eval_metric="average_precision",
                callbacks=[early_stopping(early_stopping_rounds, verbose=False),
                           log_evaluation(0)])
        config_data = {"n_estimators": n_estimators,
                       "best_iteration": int(clf.best_iteration_ or n_estimators),
                       "max_depth": max_depth, "learning_rate": learning_rate,
                       "n_jobs": lgbm_n_jobs, "n_jobs_requested": n_jobs,
                       "scale_pos_weight": lgbm_pos_weight,
                       "raw_scale_pos_weight": scale_pos_weight,
                       "early_stopping_rounds": early_stopping_rounds,
                       "eval_metric": "average_precision"}

    def predict(X):
        chunks = [clf.predict_proba(X[i:i + 50_000])[:, 1]
                  for i in range(0, len(X), 50_000)]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    pva, pte = predict(Xva), predict(Xte)
    config_data["fit_sec"] = round(time.time() - start, 1)
    return clf, pva, pte, config_data


def run_model(name: str, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray,
              yva: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
              feature_names: list[str], *, seed: int = config.SEED,
              model_options: dict | None = None, artifact_path: str | None = None):
    """Train one duel model and return test metrics plus val metrics/config."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")
    options = dict(model_options or {})
    preprocess = None
    if name == "ftt":
        model, pva, pte, model_config, preprocess = _ftt_fit_predict(
            Xtr, ytr, Xva, Xte, feature_names, seed=seed,
            epochs=options.pop("epochs", 3), batch_size=options.pop("batch_size", 2048),
            lr=options.pop("lr", 1e-3), d_model=options.pop("d_model", 128),
            n_layers=options.pop("n_layers", 2), n_heads=options.pop("n_heads", 8),
            cat_dim=options.pop("cat_dim", 16), ff_dim=options.pop("ff_dim", 64),
            pos_weight_cap=options.pop("pos_weight_cap", 20.0),
        )
        model_config.update(options)
    else:
        model, pva, pte, model_config = _tree_fit_predict(
            name, Xtr, ytr, Xva, yva, Xte, seed=seed,
            n_estimators=options.pop("n_estimators", 1200),
            early_stopping_rounds=options.pop("early_stopping_rounds", 80),
            max_depth=options.pop("max_depth", 8),
            learning_rate=options.pop("learning_rate", 0.05),
            n_jobs=options.pop("n_jobs", 18),
        )
        model_config.update(options)

    threshold = metrics.best_threshold(yva, pva)
    val_metrics = metrics.evaluate(yva, pva, threshold=threshold)
    test_metrics = metrics.evaluate(yte, pte, threshold=threshold)
    output = {key: test_metrics[key] for key in
              ("auc_roc", "auc_pr", "fbeta", "precision", "recall", "threshold")}
    output["lift"] = test_metrics["lift_at_threshold"]
    output.update({"trained_rows": int(len(Xtr)), "config": model_config})

    if artifact_path:
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        if name == "ftt":
            import torch

            torch.save({"state_dict": model.state_dict(), "config": model_config,
                        "feature_names": feature_names, "preprocessing": preprocess},
                       artifact_path)
        else:
            import joblib

            joblib.dump(model, artifact_path, compress=3)
    return output, val_metrics


def _sample_indices(y: np.ndarray, split: tuple[np.ndarray, np.ndarray, np.ndarray],
                    smoke: bool):
    tr, va, te = split
    if not smoke:
        return tr, va, te
    # Same group-disjoint train/val/test partitions as the canonical protocol;
    # bounded, natural-rate samples keep smoke metrics interpretable.
    return (_sample_partition(tr, 50_000, config.SEED),
            _sample_partition(va, 10_000, config.SEED + 1),
            _sample_partition(te, 10_000, config.SEED + 2))


def _split_indices(df, split_name: str, smoke: bool):
    """Return model indices, feature-fit indices, and the full split sizes."""
    if split_name == "standard":
        full_split = datasets.split_by_user(df)
        selected = _sample_indices(datasets.get_label(df), full_split, smoke)
        return selected, full_split[0], full_split

    quann_data = train_quann.build_data(
        df, QUANN_SPLIT["train"], QUANN_SPLIT["val"], QUANN_SPLIT["test"],
        train_frac=0.02, test_frac=0.05, seed=config.SEED,
    )
    full_split = (quann_data["tr_sel"], quann_data["va_sel"], quann_data["te_sel"])
    del quann_data  # duel models use the shared numeric features, not QUANN tokens
    selected = _sample_indices(datasets.get_label(df), full_split, smoke)
    return selected, full_split[0], full_split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="sample 50K/10K/10K rows")
    parser.add_argument("--split", choices=("standard", "quann"), default="standard",
                        help="standard User split (default) or QUANN-matched capped split")
    parser.add_argument("--models", default=",".join(MODEL_NAMES),
                        help="comma-separated: xgboost,lightgbm,ftt")
    parser.add_argument("--out", default=None)
    parser.add_argument("--artifact-dir", default=config.RESULTS_DIR,
                        help="directory for trained model artifacts")
    parser.add_argument("--n-jobs", type=int, default=18)
    parser.add_argument("--merge-existing", action="store_true",
                        help="merge selected model results into a compatible output JSON")
    args = parser.parse_args(argv)
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    unknown = set(models) - set(MODEL_NAMES)
    if unknown or not models:
        parser.error(f"unknown/empty --models selection: {sorted(unknown)}")

    _step(1, f"data load + {args.split} split/features")
    load_start = time.time()
    df = datasets.load_big()
    y = datasets.get_label(df)
    selected, feature_fit_indices, full_split = _split_indices(df, args.split, args.smoke)
    tr, va, te = selected
    print(f"rows={len(df):,} fraud={int(y.sum()):,} | sampled train={len(tr):,} "
          f"val={len(va):,} test={len(te):,} (fraud {int(y[tr].sum())}/"
          f"{int(y[va].sum())}/{int(y[te].sum())})", flush=True)

    # Standard mirrors train_standard.py; QUANN fits only on its own capped train
    # partition so the matched control does not see omitted QUANN training rows.
    transformer = features.FeatureTransformer().fit(df.iloc[feature_fit_indices])
    Xtr = transformer.transform(df.iloc[tr])
    Xva = transformer.transform(df.iloc[va])
    Xte = transformer.transform(df.iloc[te])
    ytr, yva, yte = y[tr], y[va], y[te]
    print(f"features={Xtr.shape[1]} fit/load/transform={time.time() - load_start:.1f}s",
          flush=True)

    os.makedirs(args.artifact_dir, exist_ok=True)
    results = {}
    val_results = {}
    for number, name in enumerate(models, start=2):
        _step(number, {"xgboost": "XGBoost", "lightgbm": "LightGBM", "ftt": "FT-Transformer"}[name])
        if name == "ftt":
            options = ({"epochs": 1, "batch_size": 2048, "d_model": 64,
                        "n_layers": 1, "n_heads": 4, "cat_dim": 8, "ff_dim": 32}
                       if args.smoke else {})
            artifact_name = (f"duels_{'smoke_' if args.smoke else ''}quann_ftt.pt"
                             if args.split == "quann" else
                             "duels_smoke_ftt.pt" if args.smoke else "duels_ftt.pt")
            artifact = os.path.join(args.artifact_dir, artifact_name)
        else:
            options = ({"n_estimators": 120, "early_stopping_rounds": 20}
                       if args.smoke else {})
            options["n_jobs"] = args.n_jobs
            artifact_name = (f"duels_{'smoke_' if args.smoke else ''}quann_{name}.joblib"
                             if args.split == "quann" else
                             f"duels_smoke_{name}.joblib" if args.smoke
                             else f"duels_{name}.joblib")
            artifact = os.path.join(args.artifact_dir, artifact_name)
        result, val = run_model(name, Xtr, ytr, Xva, yva, Xte, yte,
                                transformer.feature_names, model_options=options,
                                artifact_path=artifact)
        result["artifact"] = os.path.basename(artifact)
        results[name] = result
        val_results[name] = {key: val[key] for key in
                             ("auc_roc", "auc_pr", "fbeta", "precision", "recall",
                              "threshold", "lift_at_threshold")}
        print(f"  {name}: val AUC-PR={val['auc_pr']:.4f} AUC-ROC={val['auc_roc']:.4f}; "
              f"test AUC-PR={result['auc_pr']:.4f} AUC-ROC={result['auc_roc']:.4f}",
              flush=True)

    _step(5, "summary + save results")
    default_out = (
        f"duels_{'smoke_' if args.smoke else ''}quann.json"
        if args.split == "quann" else
        "duels_smoke.json" if args.smoke else "duels.json"
    )
    out = args.out or os.path.join(config.RESULTS_DIR, default_out)
    payload = {
        "models": results,
        "validation": val_results,
        "data": {
            "n_rows": int(len(df)), "fraud": int(y.sum()),
            "split_name": args.split,
            "split": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
            "split_full": {"train": int(len(full_split[0])),
                           "val": int(len(full_split[1])),
                           "test": int(len(full_split[2]))},
            "fraud_by_split": {"train": int(y[tr].sum()), "val": int(y[va].sum()),
                               "test": int(y[te].sum())},
            "fraud_by_split_full": {
                "train": int(y[full_split[0]].sum()),
                "val": int(y[full_split[1]].sum()),
                "test": int(y[full_split[2]].sum()),
            },
            "test_base_rate": float(yte.mean()),
            "feature_count": int(Xtr.shape[1]), "features": transformer.feature_names,
            "seed": config.SEED, "smoke": bool(args.smoke),
            "sampling": (
                "deterministic natural-rate sample of QUANN-capped User-group partitions"
                if args.split == "quann" and args.smoke else
                "full QUANN-capped User-group partitions" if args.split == "quann" else
                "deterministic natural-rate sample of canonical User-group partitions"
                if args.smoke else "full canonical User-group partitions"
            ),
        },
    }
    if args.merge_existing:
        if not os.path.isfile(out):
            parser.error(f"cannot merge: output does not exist: {out}")
        with open(out) as handle:
            existing = json.load(handle)
        old_data = existing.get("data", {})
        new_data = payload["data"]
        compatible = (
            old_data.get("split_name", args.split) == args.split
            and old_data.get("n_rows") == new_data["n_rows"]
            and old_data.get("split_full") == new_data["split_full"]
            and old_data.get("smoke", False) == new_data["smoke"]
        )
        if not compatible:
            parser.error(f"cannot merge incompatible split/data into {out}")
        existing.setdefault("models", {}).update(results)
        existing.setdefault("validation", {}).update(val_results)
        existing["data"] = new_data
        payload = existing
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved -> {out}", flush=True)
    print("\n=== DUELS SUMMARY (test) ===")
    print(f"{'model':<14} {'AUC-ROC':>9} {'AUC-PR':>9} {'F2':>8}")
    for name, result in results.items():
        print(f"{name:<14} {result['auc_roc']:>9.4f} {result['auc_pr']:>9.4f} "
              f"{result['fbeta']:>8.4f}")
    print("=== [" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] 5/5: complete ===")
    return payload


if __name__ == "__main__":
    main()
