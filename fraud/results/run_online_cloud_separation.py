"""Bounded CPU experiment: measure and neutralize generic online FP clouds.

This driver intentionally lives outside fraud/src: the model/pipeline code is
unchanged. It rebuilds the established group-disjoint split and fine-tunes only
the existing reader/head from the bundled-quantizer checkpoint.
"""
from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
sys.path.insert(0, os.path.abspath("fraud/src"))
import datasets
import edge_eval
import metrics
import train_quann as tq
from qcloud_fraud import QCloudFraudClassifier
from quantize import FeatureQuantizer


ROOT = os.path.abspath(".")
CKPT = os.path.join(ROOT, "fraud/results/quann_full_hybrid.pt")
OUT = os.path.join(ROOT, "fraud/results/online_cloud_separation.json")
NEW_CKPT = os.path.join(ROOT, "fraud/results/quann_online_neutralized.pt")
SEED = 20261020
USECOLS = ["User", "Is Fraud?", "Amount", "Time", "Month", "Day", "Year",
           "Zip", "Use Chip", "MCC", "Merchant City", "Merchant Name",
           "Merchant State", "Errors?"]


def log(s):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}", flush=True)


def raw_generic(frame):
    amount = (frame["Amount"].astype(str).str.replace("$", "", regex=False)
              .str.replace(",", "", regex=False).str.strip())
    amount = pd.to_numeric(amount, errors="coerce")
    state = frame["Merchant State"].fillna("ONLINE").astype(str)
    return ((frame["Use Chip"].astype(str) == "Online Transaction") &
            (state == "ONLINE") & (frame["MCC"].astype(str) == "5311") &
            (amount > 100.0)).to_numpy()


def safe_generic(frame, tok, fq):
    # Generic motif, excluding merchant identities identified as rare positive
    # contributors in the FP audit. This keeps common names and catch-all names.
    name_col = fq.feature_order.index("Merchant Name")
    rare_name_ids = np.asarray([179, 180, 181, 182, 200, 204, 207], dtype=np.int64)
    return raw_generic(frame) & ~np.isin(tok[:, name_col], rare_name_ids)


def geom_summary(model, tokens_x, tokens_y, max_n=2000):
    rng = np.random.default_rng(SEED)
    if len(tokens_x) > max_n:
        tokens_x = tokens_x[rng.choice(len(tokens_x), max_n, replace=False)]
    if len(tokens_y) > max_n:
        tokens_y = tokens_y[rng.choice(len(tokens_y), max_n, replace=False)]
    def encode(tokens):
        t = torch.as_tensor(tokens, dtype=torch.long)
        mask = (t != 0) & (t != 1)
        with torch.no_grad():
            points = model.points_for(t, mask)
            state = model.cloud_encoder(points, mask)
        return points.cpu().numpy().reshape(len(t), -1), state.cpu().numpy()
    pts_x, states_x = encode(tokens_x)
    pts_y, states_y = encode(tokens_y)

    def pair_stats(x, y):
        x = x.astype(np.float64, copy=False); y = y.astype(np.float64, copy=False)
        xc = x - x.mean(0); yc = y - y.mean(0)
        d = np.linalg.norm(x.mean(0) - y.mean(0))
        vx = np.mean(np.sum(xc * xc, axis=1)); vy = np.mean(np.sum(yc * yc, axis=1))
        # Distances/cosines are sampled to bound O(N^2) memory.
        n = min(1000, len(x), len(y))
        a = x[:n]; b = y[:n]
        dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        den = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
        cos = (a @ b.T) / np.maximum(den, 1e-12)
        return {"n_each": int(len(x)), "centroid_l2": float(d),
                "pooled_within_rms": float(np.sqrt((vx + vy) / 2)),
                "centroid_d_over_within_rms": float(d / max(np.sqrt((vx + vy) / 2), 1e-12)),
                "cross_pair_l2_mean": float(dist.mean()),
                "cross_pair_cos_mean": float(cos.mean())}

    return {"cloud_points_flat": pair_stats(pts_x, pts_y),
            "reader_encoded_state": pair_stats(states_x, states_y)}


def get_scores(model, tok, batch=8192):
    t = torch.as_tensor(tok, dtype=torch.long)
    return model.predict_proba(t, use_quann_points=True, batch=batch).cpu().numpy()


def evaluate(model, tok, y):
    p = get_scores(model, tok)
    out = metrics.evaluate(y, p, threshold=0.9969438910484314)
    out["auc_pr"] = float(average_precision_score(y, p))
    out["auc_roc"] = float(roc_auc_score(y, p))
    return out, p


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED); np.random.seed(SEED)
    log("CPU experiment start; loading IBM frame (MPS untouched)")
    frame = pd.read_csv(datasets.config.BIG_CSV, usecols=USECOLS)
    log(f"loaded rows={len(frame):,}")
    raw_ckpt = torch.load(CKPT, map_location="cpu")
    fq = FeatureQuantizer.from_dict(raw_ckpt["quantizer"])
    data = tq.build_data(frame, 1_020_000, 100_000, 4_700_000, 0.02, 0.0013,
                         seed=datasets.config.SEED, quantizer=fq)
    model = QCloudFraudClassifier.load_quann(CKPT, torch.device("cpu"))
    yall = data["y"]
    tr, va, te = data["tr_sel"], data["va_sel"], data["te_sel"]
    tt, tv, te_tok = data["tok_tr"], data["tok_va"], data["tok_te"]
    ytr, yva, yte = yall[tr], yall[va], yall[te]
    log(f"split train={len(tr):,} val={len(va):,} test={len(te):,} test_fraud={int(yte.sum()):,}")

    # Step 1: sample geometry within the natural held-out test. This is a
    # reader-input comparison, not a claim that centroids alone classify rows.
    te_raw = frame.iloc[te]
    gen_mask = safe_generic(te_raw, te_tok, fq) & (yte == 0)
    online_mask = ((te_raw["Use Chip"].astype(str).to_numpy() == "Online Transaction") &
                   (te_raw["Merchant State"].fillna("ONLINE").astype(str).to_numpy() == "ONLINE") &
                   (yte == 1))
    ng = np.flatnonzero(gen_mask); nf = np.flatnonzero(online_mask)
    if min(len(ng), len(nf)) < 100:
        raise RuntimeError(f"geometry cohorts too small: safe_generic={len(ng)} fraud_online={len(nf)}")
    rng = np.random.default_rng(SEED)
    n_geom = min(2000, len(ng), len(nf))
    gi = rng.choice(ng, n_geom, replace=False); fi = rng.choice(nf, n_geom, replace=False)
    geom = {"definition": "test normals: Online Transaction + ONLINE state + MCC 5311 + amount>100, excluding audit-identified rare positive merchant token IDs [179,180,181,182,200,204,207]; positives: online-chip/state true fraud",
            "candidate_counts": {"safe_generic_normal": int(len(ng)), "online_true_fraud": int(len(nf))},
            "sampled_per_class": int(n_geom),
            "pairwise": geom_summary(model, te_tok[gi], te_tok[fi])}
    log(f"geometry cohorts: safe_generic={len(ng):,}, online_fraud={len(nf):,}, n={n_geom}")
    log("geometry cloud-points d/within=" +
        f"{geom['pairwise']['cloud_points_flat']['centroid_d_over_within_rms']:.3f}; " +
        "reader-state d/within=" +
        f"{geom['pairwise']['reader_encoded_state']['centroid_d_over_within_rms']:.3f}")
    log("geometry cross distance/cos: points=" +
        f"{geom['pairwise']['cloud_points_flat']['cross_pair_l2_mean']:.3f}/" +
        f"{geom['pairwise']['cloud_points_flat']['cross_pair_cos_mean']:.3f}; reader=" +
        f"{geom['pairwise']['reader_encoded_state']['cross_pair_l2_mean']:.3f}/" +
        f"{geom['pairwise']['reader_encoded_state']['cross_pair_cos_mean']:.3f}")

    # Subsample classes for a focused but still label-supervised fine-tune.
    # Hard negatives are the entire operational motif without named merchant
    # identities; positives are authentic online fraud. Random normal controls
    # preserve the non-online part of the primary objective.
    tr_raw = frame.iloc[tr]
    hard = np.flatnonzero(safe_generic(tr_raw, tt, fq) & (ytr == 0))
    pos = np.flatnonzero((ytr == 1) &
        (tr_raw["Use Chip"].astype(str).to_numpy() == "Online Transaction") &
        (tr_raw["Merchant State"].fillna("ONLINE").astype(str).to_numpy() == "ONLINE"))
    normal = np.flatnonzero(ytr == 0)
    log(f"train cohorts hard_generic={len(hard):,} online_fraud={len(pos):,} normal={len(normal):,}")
    if min(len(hard), len(pos)) < 100:
        raise RuntimeError("insufficient training examples for targeted fine-tune")
    rng = np.random.default_rng(SEED)
    nh, np_, nc = min(6000, len(hard)), min(5000, len(pos)), min(3000, len(normal))
    hidx = rng.choice(hard, nh, replace=False)
    pidx = rng.choice(pos, np_, replace=np_ > len(pos))
    cidx = rng.choice(normal, nc, replace=False)
    tr_idx = np.concatenate([hidx, pidx, cidx])
    rng.shuffle(tr_idx)
    x = torch.as_tensor(tt[tr_idx], dtype=torch.long)
    yy = torch.as_tensor(ytr[tr_idx], dtype=torch.float32)

    # Held-out diagnostics from the val users, never used for gradient updates.
    va_raw = frame.iloc[va]
    vh = np.flatnonzero(raw_generic(va_raw) & (yva == 0))
    vp = np.flatnonzero((yva == 1) &
        (va_raw["Use Chip"].astype(str).to_numpy() == "Online Transaction") &
        (va_raw["Merchant State"].fillna("ONLINE").astype(str).to_numpy() == "ONLINE"))
    if min(len(vh), len(vp)) < 50:
        raise RuntimeError(f"validation cohorts too small: {len(vh)}, {len(vp)}")
    vh = rng.choice(vh, min(2000, len(vh)), replace=False)
    vp = rng.choice(vp, min(2000, len(vp)), replace=False)
    v_idx = np.concatenate([vh, vp]); vy = yva[v_idx]; vtok = tv[v_idx]
    before_val, _ = evaluate(model, vtok, vy)
    before_test, before_prob = evaluate(model, te_tok, yte)
    log(f"baseline full test FP={before_test['fp']:,} TP={before_test['tp']:,} "
        f"P={before_test['precision']:.4f} R={before_test['recall']:.4f} "
        f"PR={before_test['auc_pr']:.4f} ROC={before_test['auc_roc']:.4f}")

    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.cloud_encoder.parameters(): p.requires_grad_(True)
    for p in model.class_head.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=2e-5, weight_decay=1e-5)
    bs = 256; train_log = []
    for ep in range(1, 4):
        order = rng.permutation(len(x)); losses = []
        for start in range(0, len(order), bs):
            ix = torch.as_tensor(order[start:start + bs], dtype=torch.long)
            toks = x[ix]; labels = yy[ix]
            mask = (toks != 0) & (toks != 1)
            with torch.no_grad(): pts = model.points_for(toks, mask)
            logits, _ = model(toks, mask, points_override=pts)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), labels, pos_weight=torch.tensor(8.0))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        val, _ = evaluate(model, vtok, vy)
        row = {"epoch": ep, "loss": float(np.mean(losses)),
               "val_fp": val["fp"], "val_tp": val["tp"],
               "val_precision": val["precision"], "val_recall": val["recall"]}
        train_log.append(row)
        log(f"fine-tune ep{ep}/3 loss={row['loss']:.4f} val FP={row['val_fp']} TP={row['val_tp']} "
            f"P={row['val_precision']:.3f} R={row['val_recall']:.3f}")

    after_test, after_prob = evaluate(model, te_tok, yte)
    model.save_quann(NEW_CKPT, extra={"quantizer": raw_ckpt["quantizer"],
        "experiment": "online generic cloud neutralization; reader/head fine-tune",
        "seed": SEED, "threshold": 0.9969438910484314})
    log(f"saved candidate checkpoint {NEW_CKPT}")

    # Edge parity on the same 4.7M tokenized held-out rows using the exact
    # deployment graph and calibrated per-output-channel INT8 proxy.
    dp = edge_eval.DeployPath(model).eval()
    q = edge_eval._to_quant(__import__("copy").deepcopy(dp), weight_per_channel=True)
    ids = torch.as_tensor(te_tok, dtype=torch.long); msk = (ids != 0) & (ids != 1)
    edge_eval._calibrate(q, ids[:50000], msk[:50000])
    p8 = edge_eval._chunked_proba(q, ids, msk, chunk=8192)
    int8 = metrics.evaluate(yte, p8, threshold=0.9969438910484314)
    int8.update({"auc_pr": float(average_precision_score(yte, p8)),
                 "auc_roc": float(roc_auc_score(yte, p8))})
    out = {"date": "2026-10-20", "run": "online-cloud-separation", "checkpoint": NEW_CKPT,
        "threshold": 0.9969438910484314, "geometry": geom,
        "training": {"n_hard_negative": int(len(hard)), "n_online_fraud": int(len(pos)),
                     "n_random_negative": int(len(normal)), "n_train_used": int(len(x)),
                     "epochs": train_log},
        "baseline": {"full_test": before_test, "validation_cohorts": before_val},
        "candidate": {"full_test_fp32": after_test, "full_test_int8_channel": int8},
        "fp32_prob_shift": {"median_abs": float(np.median(np.abs(after_prob-before_prob))),
                            "max_abs": float(np.max(np.abs(after_prob-before_prob)))}}
    with open(OUT, "w") as f: json.dump(out, f, indent=2)
    log(f"FULL TEST FP32: FP={after_test['fp']:,} TP={after_test['tp']:,} "
        f"FN={after_test['fn']:,} P={after_test['precision']:.4f} R={after_test['recall']:.4f} "
        f"PR={after_test['auc_pr']:.4f} ROC={after_test['auc_roc']:.4f}")
    log(f"FULL TEST INT8-channel: FP={int8['fp']:,} TP={int8['tp']:,} "
        f"P={int8['precision']:.4f} R={int8['recall']:.4f} "
        f"PR={int8['auc_pr']:.4f} ROC={int8['auc_roc']:.4f}")
    log(f"done -> {OUT}")
    del frame, data, te_raw, tr_raw, va_raw, ids, msk
    gc.collect()


if __name__ == "__main__":
    main()
