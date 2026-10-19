"""Strong-reader diagnostic for the QUANN fraud model (option b).

Hypothesis under test: the AUC-PR gap (hybrid 0.258 vs GBT/RF 0.6-0.7) is due
to the READER being too small to combine the point-cloud geometry — NOT the
points being information-poor.

Method: reuse the converged QUANN points (D=8) from the full hybrid checkpoint,
freeze them, and backprop a CloudEncoderND + class_head of VARYING size on the
full train (batched — no QUANN loop, so fast). Report AUC-PR/AUC-ROC vs reader
size. If AUC-PR rises with size -> reader was the bottleneck (scale it). If it
plateaus at ~0.26 -> the D=8 point geometry is the ceiling (needs richer points).

Run:  python3 fraud/src/train_strong_reader.py
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
import metrics
import quantize
import train_quann
from qcloud_fraud import QCloudFraudClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(config.RESULTS_DIR, "quann_hybrid_full.pt"))
    ap.add_argument("--n-train", type=int, default=1020000)
    ap.add_argument("--n-val", type=int, default=100000)
    ap.add_argument("--n-test", type=int, default=4700000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-cloud-dim", type=int, default=None,
                    help="point dim D (default: the D of the checkpoint)")
    ap.add_argument("--sizes", default="64/64,128/128,256/256",
                    help="reader hidden/d_cloud pairs, comma-separated")
    args = ap.parse_args()

    device = torch.device("cpu")  # MPS held free for the live CloudLM run
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    print("Loading full dataset ...")
    df = datasets.load_big()
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    fq = (quantize.FeatureQuantizer.from_dict(checkpoint["quantizer"])
          if "quantizer" in checkpoint else None)
    data = train_quann.build_data(df, args.n_train, args.n_val, args.n_test, 0.5, 0.05,
                                  quantizer=fq)
    fq = data["fq"]; y = data["y"]
    tr, va, te = data["tr_sel"], data["va_sel"], data["te_sel"]
    print(f"train={len(tr):,} val={len(va):,} test={len(te):,} (test fraud {y[te].sum():,}, "
          f"base {y[te].mean():.3%}) | vocab={fq.vocab_size} K={fq.n_features}")

    # Load the converged QUANN points (D is fixed by the checkpoint).
    src = QCloudFraudClassifier.load_quann(args.ckpt, device)
    D = args.d_cloud_dim or src.D
    points = {k: w for k, w in src.token_weights.items()}
    print(f"loaded {len(points)} converged points, D={D}")

    def mk(tok, yy):
        t = torch.tensor(tok, dtype=torch.long, device=device)
        m = (t != 0) & (t != 1)
        return t, m, torch.tensor(yy.astype(np.float32), device=device)

    ttr, mtr, ytr = mk(data["tok_tr"], y[tr])
    tte, mte, yte = mk(data["tok_te"], y[te])
    tva, mva, yva = mk(data["tok_va"], y[va])

    raw_pw = float((y[tr] == 0).sum()) / max(float((y[tr] == 1).sum()), 1.0)
    pos_w = torch.tensor(min(raw_pw, 20.0), device=device)

    results = {}
    for spec in args.sizes.split(","):
        hidden, dcloud = [int(x) for x in spec.split("/")]
        print(f"\n{'='*60}\nreader hidden={hidden} d_cloud={dcloud} (points D={D} fixed)\n{'='*60}")
        m = QCloudFraudClassifier(vocab_size=fq.vocab_size, d_model=32, d_cloud=dcloud,
                                  d_cloud_dim=D, hidden=hidden).to(device)
        m.token_weights = points  # reuse converged points
        opt = torch.optim.AdamW(
            list(m.cloud_encoder.parameters()) + list(m.class_head.parameters()),
            lr=args.lr, weight_decay=1e-4)
        n = ttr.size(0)
        rng = np.random.default_rng(config.SEED)
        t0 = time.time()
        met = None
        for ep in range(1, args.epochs + 1):
            order = rng.permutation(n)
            for s in range(0, n, args.batch):
                idx = order[s:s + args.batch]
                toks, msk, tgt = ttr[idx], mtr[idx], ytr[idx]
                pts = m.points_for(toks, msk).detach()
                m.train()
                logits, _ = m(toks, msk, points_override=pts)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), tgt, pos_weight=pos_w)
                opt.zero_grad(); loss.backward(); opt.step()
            pte = m.predict_proba(tte, use_quann_points=True, batch=50000).cpu().numpy()
            met = metrics.evaluate(yte.cpu().numpy(), pte)
            print(f"  ep{ep}: AUC-ROC={met['auc_roc']:.4f} AUC-PR={met['auc_pr']:.4f} "
                  f"F2={met['fbeta']:.3f} P={met['precision']:.3f} ({time.time()-t0:.0f}s)")
        results[f"{hidden}/{dcloud}"] = {k: met[k] for k in
                                         ("auc_roc", "auc_pr", "fbeta", "precision", "recall",
                                          "lift_at_threshold", "threshold")}

    print(f"\n=== STRONG-READER SUMMARY (fixed D={D} QUANN points, full 4.7M test) ===")
    for k, v in results.items():
        print(f"  reader {k:9s}: AUC-ROC={v['auc_roc']:.4f} AUC-PR={v['auc_pr']:.4f} "
              f"F2={v['fbeta']:.3f} P={v['precision']:.3f} lift={v['lift_at_threshold']:.1f}x")
    out = os.path.join(config.RESULTS_DIR, "strong_reader.json")
    json.dump({"D": D, "results": results,
               "baseline_hybrid": {"auc_roc": 0.9581, "auc_pr": 0.2579}}, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
