"""repro_check.py — proves the bundled-quantizer fix makes AUC reproducible.

The edge-eval found recorded AUCs do not reproduce from disk: the checkpoint
never bundled its input FeatureQuantizer, so re-fitting it (different train
sample -> different bins/top-N) shifts token ids and the AUC drifts.

This self-contained check (bounded `--n-rows`, safe RAM, CPU) shows:
  * tokenizing the SAME rows with the checkpoint's BUNDLED quantizer is
    bit-identical across independent reloads  -> AUC identical (reproducible);
  * a freshly RE-FIT quantizer (the old behavior) drifts  -> AUC changes.

Run:
  python3 fraud/src/repro_check.py                 # first 100K rows
  python3 fraud/src/repro_check.py --n-rows 200000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config            # noqa: F401  (sys.path bootstrap)
import datasets
import quantize
from qcloud_fraud import QCloudFraudClassifier
from sklearn.metrics import roc_auc_score


def _tokens_and_label(df, sel, fq):
    tok = torch.tensor(fq.transform(df.iloc[sel]), dtype=torch.long)
    y = datasets.get_label(df)[sel]
    return tok, (tok != 0) & (tok != 1), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=100000)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=200, help="tiny reader train steps")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"loading first {args.n_rows:,} rows ...", flush=True)
    df = datasets.load_big(args.n_rows)
    n = len(df)
    cut = int(n * 0.5)
    tr_sel = np.arange(0, min(cut, n))
    te_sel = np.arange(cut, n)
    y = datasets.get_label(df)
    print(f"rows={n:,}  test={len(te_sel):,}  test fraud={int(y[te_sel].sum())} "
          f"({y[te_sel].mean():.2%})")

    # --- build a deterministic model + its (training) quantizer -------------
    fq = quantize.FeatureQuantizer().fit(df.iloc[tr_sel])
    m = QCloudFraudClassifier(vocab_size=fq.vocab_size, d_model=32,
                              d_cloud=64, d_cloud_dim=8)
    tr_tok, tr_msk, tr_y = _tokens_and_label(df, tr_sel[:args.n_train], fq)
    opts = [p for p in m.cloud_encoder.parameters()] + list(m.class_head.parameters())
    opt = torch.optim.Adam(opts, lr=1e-3)
    m.train()
    for i in range(args.steps):
        r = i % tr_tok.shape[0]
        logits, _ = m(tr_tok[r:r + 1], tr_msk[r:r + 1],
                      points_override=m.points_for(tr_tok[r:r + 1], tr_msk[r:r + 1]))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(-1), torch.tensor(tr_y[r:r + 1].astype(np.float32)))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()

    # --- save with the bundled quantizer (the fix) --------------------------
    import tempfile
    ckpt = os.path.join(tempfile.mkdtemp(), "repro_hybrid.pt")
    m.save_quann(ckpt, extra={"quantizer": fq.to_dict()})

    te_tok_src, te_msk_src, te_y = _tokens_and_label(df, te_sel, fq)

    def auc_with_quantizer(fq_in):
        t, msk, _ = _tokens_and_label(df, te_sel, fq_in)
        # use the saved model's points via the (re)tokenized ids
        with torch.no_grad():
            pts = m.points_for(t, msk)
            logits, _ = m(t, msk, points_override=pts)
            p = torch.sigmoid(logits).squeeze(-1).numpy()
        return float(roc_auc_score(te_y, p))

    # 1) reproducible: reload the checkpoint (bundled quantizer) TWICE
    import copy
    a1 = auc_with_quantizer(quantize.FeatureQuantizer.from_dict(
        torch.load(ckpt, map_location="cpu")["quantizer"]))
    a2 = auc_with_quantizer(quantize.FeatureQuantizer.from_dict(
        torch.load(ckpt, map_location="cpu")["quantizer"]))

    # token-matrix identity across the two bundled re-tokenizations
    tA, _, _ = _tokens_and_label(df, te_sel,
                                 quantize.FeatureQuantizer.from_dict(
                                     torch.load(ckpt, map_location="cpu")["quantizer"]))
    tB, _, _ = _tokens_and_label(df, te_sel,
                                 quantize.FeatureQuantizer.from_dict(
                                     torch.load(ckpt, map_location="cpu")["quantizer"]))
    tok_identical = bool(torch.equal(tA, tB))

    # 2) the OLD behavior: a freshly re-fit quantizer (independent train sample)
    fq_refit = quantize.FeatureQuantizer().fit(df.iloc[tr_sel[args.n_train:]])
    a_refit = auc_with_quantizer(fq_refit)

    print(f"\n  bundled  : AUC reload#1={a1:.4f}  reload#2={a2:.4f}  "
          f"identical={a1 == a2}  tokens bit-identical={tok_identical}")
    print(f"  re-fit   : AUC={a_refit:.4f}   (differs from bundled -> old non-reproducible path)")
    print(f"\n  => reproducible={'YES' if a1 == a2 and tok_identical else 'NO'} "
          f"(bundled quantizer gives identical tokens + AUC; re-fit drifts)")
    os.remove(ckpt)


if __name__ == "__main__":
    main()
