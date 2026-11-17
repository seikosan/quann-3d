"""Train richer QUANN identity points, then a 256/256 backprop reader.

The expensive 0th-order stage uses all train fraud and a deterministic capped
normal sample. The reader and final evaluation use the full matched QUANN
train/validation/test partitions. Use --smoke for a small end-to-end check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import datasets
import metrics
import quantize
import train_quann
from qcloud_fraud import QCloudFraudClassifier


def _tensors(tokens: np.ndarray, labels: np.ndarray, device):
    t = torch.as_tensor(tokens, dtype=torch.long, device=device)
    m = (t != 0) & (t != 1)
    y = torch.as_tensor(labels.astype(np.float32), device=device)
    return t, m, y


def moderate_train_indices(train_sel: np.ndarray, labels: np.ndarray,
                           normal_cap: int = 450_000,
                           seed: int = config.SEED) -> np.ndarray:
    """Keep every training fraud row and deterministically cap normal rows."""
    fraud = train_sel[labels[train_sel] == 1]
    normal = train_sel[labels[train_sel] == 0]
    rng = np.random.default_rng(seed)
    if len(normal) > normal_cap:
        normal = rng.choice(normal, size=normal_cap, replace=False)
    selected = np.concatenate([fraud, normal])
    rng.shuffle(selected)
    return selected


def train_point_steps(model: QCloudFraudClassifier, tokens: torch.Tensor,
                      labels: torch.Tensor, steps: int,
                      pos_weight: torch.Tensor | None = None,
                      seed: int = config.SEED) -> dict:
    """Run a fixed number of narrowing 0th-order QUANN row updates."""
    if len(tokens) == 0 or steps < 1:
        raise ValueError("point training requires rows and at least one step")
    model.eval()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tokens))
    start = time.perf_counter()
    rewarded = censured = unchanged = done = 0
    for i in range(steps):
        row = int(order[i % len(order)])
        stats = model.update_token_weights(
            tokens[row:row + 1], (tokens[row:row + 1] != 0) & (tokens[row:row + 1] != 1),
            int(labels[row].item()), search_rule="search", pos_weight=pos_weight)
        rewarded += stats["rewarded"]
        censured += stats["censured"]
        unchanged += stats["unchanged"]
        done += 1
    elapsed = max(time.perf_counter() - start, 1e-9)
    conv = model.convergence_info()
    return {"steps": done, "elapsed_sec": elapsed, "steps_per_sec": done / elapsed,
            "rewarded": rewarded, "censured": censured, "unchanged": unchanged,
            "convergence": conv}


def train_points_to_convergence(model: QCloudFraudClassifier,
                                tokens: torch.Tensor, labels: torch.Tensor,
                                pos_weight: torch.Tensor,
                                max_epochs: int = 0, log=print) -> dict:
    """Classic reward/censure passes until all observed identity points converge.

    max_epochs=0 means no epoch cap. A positive cap is useful for an explicitly
    bounded exploratory run; the returned convergence flag reports if it hit.
    """
    n = len(tokens)
    rng = np.random.default_rng(config.SEED)
    total_steps = 0
    start = time.perf_counter()
    epochs = 0
    model.eval()
    while not model.all_converged():
        epochs += 1
        order = rng.permutation(n)
        r = c = u = 0
        for row in order:
            toks = tokens[row:row + 1]
            mask = (toks != 0) & (toks != 1)
            stats = model.update_token_weights(
                toks, mask, int(labels[row].item()), search_rule="search",
                pos_weight=pos_weight)
            r += stats["rewarded"]
            c += stats["censured"]
            u += stats["unchanged"]
        total_steps += n
        info = model.convergence_info()
        elapsed = max(time.perf_counter() - start, 1e-9)
        log(f"  0th-order epoch {epochs}: R={r} C={c} U={u} "
            f"converged={info['converged_tokens']}/{info['total_tokens']} "
            f"width={info['mean_width']:.4f} throughput={total_steps / elapsed:.2f} rows/s")
        if max_epochs and epochs >= max_epochs:
            break
    elapsed = max(time.perf_counter() - start, 1e-9)
    info = model.convergence_info()
    return {"epochs": epochs, "steps": total_steps, "elapsed_sec": elapsed,
            "steps_per_sec": total_steps / elapsed,
            "all_converged": info["all_converged"], **info}


def train_reader(model: QCloudFraudClassifier, tokens: torch.Tensor,
                 labels: torch.Tensor, epochs: int = 3, batch_size: int = 2048,
                 lr: float = 1e-3, pos_weight: torch.Tensor | None = None,
                 max_steps: int | None = None) -> dict:
    """Backprop-train the 256/256 reader while keeping QUANN points fixed."""
    opt = torch.optim.AdamW(
        list(model.cloud_encoder.parameters()) + list(model.class_head.parameters()),
        lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(config.SEED)
    step_count = 0
    start = time.perf_counter()
    model.train()
    for ep in range(epochs):
        order = rng.permutation(len(tokens))
        for s in range(0, len(tokens), batch_size):
            idx = order[s:s + batch_size]
            toks, target = tokens[idx], labels[idx]
            mask = (toks != 0) & (toks != 1)
            points = model.points_for(toks, mask).detach()
            logits, _ = model(toks, mask, points_override=points)
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), target, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_count += 1
            if max_steps and step_count >= max_steps:
                break
        if max_steps and step_count >= max_steps:
            break
    model.eval()
    return {"epochs_requested": epochs, "steps": step_count,
            "elapsed_sec": time.perf_counter() - start}


def predict_array(model: QCloudFraudClassifier, tokens: np.ndarray,
                  batch_size: int = 50_000) -> np.ndarray:
    """Chunked probability inference to bound memory for the 4.7M test set."""
    out = []
    for start in range(0, len(tokens), batch_size):
        batch = torch.as_tensor(tokens[start:start + batch_size], dtype=torch.long)
        out.append(model.predict_proba(batch, use_quann_points=True,
                                       batch=batch_size).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def _smoke_splits(frame: "pd.DataFrame", seed: int = config.SEED):
    """Create small balanced development splits, sampling classes with replacement."""
    import pandas as pd  # noqa: F401 - keeps the annotation optional at import time

    y = datasets.get_label(frame)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        raise ValueError("smoke data must contain both classes")

    def pick(n: int, fraud_frac: float):
        nf = max(1, int(round(n * fraud_frac)))
        nn = n - nf
        return np.concatenate([rng.choice(pos, nf, replace=nf > len(pos)),
                               rng.choice(neg, nn, replace=nn > len(neg))])

    tr, va, te = pick(100, 0.5), pick(100, 0.5), pick(100, 0.1)
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    fq = quantize.FeatureQuantizer().fit(frame.iloc[tr])
    return {"fq": fq, "y": y, "tr_sel": tr, "va_sel": va, "te_sel": te,
            "tok_tr": fq.transform(frame.iloc[tr]),
            "tok_va": fq.transform(frame.iloc[va]),
            "tok_te": fq.transform(frame.iloc[te])}


def run_config(d: int, data: dict, smoke: bool, normal_cap: int,
               max_quann_epochs: int, reader_epochs: int, results_dir: str,
               point_steps: int = 3, reader_max_steps: int | None = None) -> dict:
    device = torch.device("cpu")
    fq, y = data["fq"], data["y"]
    tr_sel, va_sel, te_sel = data["tr_sel"], data["va_sel"], data["te_sel"]
    train_sel = (tr_sel if smoke else
                 moderate_train_indices(tr_sel, y, normal_cap=normal_cap))
    # The full-data builder stores tokens in selection order; map selected source
    # rows back to that order without retaining another full-size token matrix.
    if smoke:
        tr_tokens = data["tok_tr"]
    else:
        tok_by_row = {int(row): i for i, row in enumerate(tr_sel)}
        local = np.fromiter((tok_by_row[int(row)] for row in train_sel),
                            dtype=np.int64, count=len(train_sel))
        tr_tokens = data["tok_tr"][local]
    ttr, mtr, ytr = _tensors(tr_tokens, y[train_sel], device)
    if smoke:
        reader_tokens, reader_labels = ttr, ytr
    else:
        reader_tokens, _, reader_labels = _tensors(data["tok_tr"], y[tr_sel], device)
    if smoke:
        val_tokens, test_tokens = data["tok_va"], data["tok_te"]
    else:
        val_tokens, test_tokens = data["tok_va"], data["tok_te"]
    model = QCloudFraudClassifier(vocab_size=fq.vocab_size, d_model=32,
                                  d_cloud=256, d_cloud_dim=d, hidden=256).to(device)
    raw_pw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    pos_w = torch.tensor(min(raw_pw, 20.0), device=device)
    if smoke:
        point_result = train_point_steps(model, ttr, ytr, point_steps,
                                         pos_weight=pos_w)
    else:
        point_result = train_points_to_convergence(
            model, ttr, ytr, pos_w, max_epochs=max_quann_epochs)
    reader_result = train_reader(
        model, reader_tokens, reader_labels, epochs=1 if smoke else reader_epochs,
        batch_size=256 if smoke else 2048, lr=1e-3, pos_weight=pos_w,
        max_steps=reader_max_steps if smoke else None)

    pva = predict_array(model, val_tokens)
    yva_np = y[va_sel]
    threshold = metrics.best_threshold(yva_np, pva)
    pte = predict_array(model, test_tokens)
    test_metrics = metrics.evaluate(y[te_sel], pte, threshold=threshold)
    test_metrics.update({"config": f"D{d}", "point_train": point_result,
                         "reader_train": reader_result,
                         "quann_points": len(model.token_weights),
                         "converged_points": model.convergence_info()["converged_tokens"],
                         "mean_width": model.convergence_info()["mean_width"]})

    os.makedirs(results_dir, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    checkpoint = os.path.join(results_dir, f"quann_rich_d{d}{suffix}.pt")
    model.save_quann(checkpoint, extra={
        "quantizer": fq.to_dict(), "scale": "smoke" if smoke else "rich_points",
        "rich_points": True, "reader": "256/256",
    })
    print(f"  saved points + reader -> {checkpoint}")
    return {"test": test_metrics, "checkpoint": checkpoint,
            "point_train": point_result, "reader_train": reader_result}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="16,32", help="comma-separated point dimensions")
    ap.add_argument("--d", type=int, action="append", help="point dimension; repeatable")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--normal-cap", type=int, default=450_000,
                    help="normal rows in the 0th-order pool; all training fraud retained")
    ap.add_argument("--max-quann-epochs", type=int, default=0,
                    help="0=continue until all observed points converge")
    ap.add_argument("--reader-epochs", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    dims = args.d or [int(x) for x in args.configs.split(",") if x.strip()]
    if not dims or any(d not in (16, 32) for d in dims):
        ap.error("configs must contain D=16 and/or D=32")
    if args.smoke:
        print("Loading bundled fraud data for smoke ...")
        data = _smoke_splits(datasets.load_small())
        print(f"smoke splits: train={len(data['tr_sel'])} val={len(data['va_sel'])} "
              f"test={len(data['te_sel'])}; K={data['fq'].n_features} "
              f"vocab={data['fq'].vocab_size}")
    else:
        print("Loading full IBM data and matched QUANN splits ...")
        frame = datasets.load_big()
        data = train_quann.build_data(frame, 1_020_000, 100_000, 4_700_000,
                                      train_frac=0.5, test_frac=0.05,
                                      quant_specs=quantize.QUANT_SPECS)
        y = data["y"]
        print(f"full splits: train={len(data['tr_sel']):,} "
              f"(fraud {int(y[data['tr_sel']].sum()):,}), "
              f"val={len(data['va_sel']):,}, test={len(data['te_sel']):,}; "
              f"K={data['fq'].n_features} vocab={data['fq'].vocab_size}")
    results = {}
    for d in dims:
        print(f"\n=== Rich points D={d} ===")
        result = run_config(d, data, args.smoke, args.normal_cap,
                            args.max_quann_epochs, args.reader_epochs,
                            config.RESULTS_DIR, point_steps=3,
                            reader_max_steps=1)
        results[str(d)] = result
        print(f"  test AUC-PR={result['test']['auc_pr']:.4f} "
              f"ROC={result['test']['auc_roc']:.4f} F2={result['test']['fbeta']:.4f} "
              f"lift={result['test']['lift_at_threshold']:.2f}x")
        print(f"  0th-order throughput: {result['point_train']['steps_per_sec']:.3f} rows/s")
    out = args.out or os.path.join(config.RESULTS_DIR,
                                   "rich_points_smoke.json" if args.smoke else "rich_points.json")
    with open(out, "w") as f:
        json.dump({"configs": results,
                   "split": {"train": len(data["tr_sel"]), "val": len(data["va_sel"]),
                             "test": len(data["te_sel"])},
                   "model": {"K": data["fq"].n_features,
                             "vocab": data["fq"].vocab_size,
                             "reader": "256/256"},
                   "smoke": args.smoke}, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
