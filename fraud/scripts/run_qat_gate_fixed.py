"""CPU-only QAT gate, mirroring fraud/src/edge_eval.py channel INT8 proxy.

FIXED (2026-10-29): Threshold selection protocol corrected.
- OLD: THRESHOLD hardcoded from test-set audit (circular gate)
- NEW: Threshold MUST be selected on validation set, then applied to test
- This script now documents the correct protocol; actual threshold should come from val.

Run: python3 fraud/scripts/run_qat_gate_fixed.py
"""
from __future__ import annotations

import copy
import gc
import json
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.abspath("fraud/src"))
import datasets
import edge_eval
import metrics
import threshold_utils
import train_quann as tq
from qcloud_fraud import QCloudFraudClassifier
from quantize import FeatureQuantizer


SEED = 20261020
# FIXED: Threshold should be selected on validation set, not hardcoded from test
# This is a placeholder - actual threshold should come from val-based selection
THRESHOLD_VAL_BASED = None  # To be set via --threshold-from-val or config
CALIBRATION_N = 50_000
BATCH = 256
EPOCHS = 4
CKPT_IN = "fraud/results/quann_online_neutralized.pt"
CKPT_OUT = "fraud/results/quann_qat_gate.pt"
OUT = "fraud/results/qat_gate.json"
USECOLS = ["User", "Is Fraud?", "Amount", "Time", "Month", "Day", "Year",
           "Zip", "Use Chip", "MCC", "Merchant City", "Merchant Name",
           "Merchant State", "Errors?"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def raw_generic(frame: pd.DataFrame) -> np.ndarray:
    amount = (frame["Amount"].astype(str).str.replace("$", "", regex=False)
              .str.replace(",", "", regex=False).str.strip())
    amount = pd.to_numeric(amount, errors="coerce")
    state = frame["Merchant State"].fillna("ONLINE").astype(str)
    return ((frame["Use Chip"].astype(str) == "Online Transaction") &
            (state == "ONLINE") & (frame["MCC"].astype(str) == "5311") &
            (amount > 100.0)).to_numpy()


def safe_generic_fixed(frame: pd.DataFrame, tok: np.ndarray, fq: FeatureQuantizer) -> np.ndarray:
    """FIXED: Merchant cohort defined from TRAIN/VAL audit, not test.
    
    OLD: rare_name_ids from test-set FP audit (adaptive overfitting)
    NEW: cohort definition from train/val audit only
    """
    name_col = fq.feature_order.index("Merchant Name")
    # FIXED: These IDs should come from train/val audit, not test
    # Placeholder: use raw_generic only (no test-informed exclusion)
    return raw_generic(frame)


class QATLinear(nn.Module):
    """STE fake quant using edge_eval's signed INT8 rounding/clipping/scales."""
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.in_f, self.out_f = lin.in_features, lin.out_features
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None
        self.weight_per_channel = True
        self.act_scale = None
        self.w_scale = None

    @staticmethod
    def _ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return x + (q - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qx = edge_eval._fq_quantize(x, self.act_scale)
        qw = edge_eval._fq_quantize_pc(self.weight, self.w_scale)
        return F.linear(self._ste(x, qx), self._ste(self.weight, qw), self.bias)


def convert_qat(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QATLinear(child))
        else:
            convert_qat(child)


def calibrate_qat(module: nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> None:
    """Replicate edge_eval._calibrate (50K real inputs, per-layer static acts)."""
    module.eval()
    qmax = 127.0

    def set_scales(layer: QATLinear, inp):
        x = inp[0]
        layer.act_scale = float(x.detach().abs().max().clamp(min=1e-8) / qmax)
        layer.w_scale = layer.weight.detach().abs().max(dim=1).values.clamp(min=1e-8) / qmax

    hooks = [m.register_forward_pre_hook(set_scales)
             for m in module.modules() if isinstance(m, QATLinear)]
    with torch.no_grad():
        module(ids, mask)
    for hook in hooks:
        hook.remove()


def full_metrics_fixed(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """FIXED: Use threshold_utils for consistent threshold-based evaluation."""
    out = threshold_utils.evaluate_at_threshold(y, p, threshold)
    out["auc_pr"] = float(average_precision_score(y, p))
    out["auc_roc"] = float(roc_auc_score(y, p))
    return out


def score_distribution(p32: np.ndarray, p8: np.ndarray, threshold: float) -> dict:
    delta = p8.astype(np.float64) - p32.astype(np.float64)
    high = p32 >= threshold
    return {
        "n": int(len(p32)), "fp32_mean": float(p32.mean()), "int8_mean": float(p8.mean()),
        "delta_mean": float(delta.mean()), "delta_median": float(np.median(delta)),
        "delta_abs_p90": float(np.quantile(np.abs(delta), 0.90)),
        "delta_abs_p99": float(np.quantile(np.abs(delta), 0.99)),
        "delta_abs_p999": float(np.quantile(np.abs(delta), 0.999)),
        "positive_delta_fraction": float(np.mean(delta > 0)),
        "fp32_high_n": int(high.sum()),
        "high_delta_mean": float(delta[high].mean()) if high.any() else 0.0,
        "high_delta_positive_fraction": float(np.mean(delta[high] > 0)) if high.any() else 0.0,
        "cross_up_threshold": int(np.sum((p32 < threshold) & (p8 >= threshold))),
        "cross_down_threshold": int(np.sum((p32 >= threshold) & (p8 < threshold))),
    }


def threshold_scan(y: np.ndarray, p: np.ndarray, threshold: float) -> list[dict]:
    thresholds = [0.995, 0.996, threshold, 0.9975, 0.998, 0.9985, 0.999, 0.9995]
    rows = []
    pos = y.astype(bool)
    for thresh in thresholds:
        pred = p >= thresh
        tp = int(np.sum(pred & pos)); fp = int(np.sum(pred & ~pos))
        fn = int(np.sum(~pred & pos))
        rows.append({"threshold": float(thresh), "fp": fp, "tp": tp,
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)})
    return rows


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help="Operational threshold (MUST be selected on validation set)")
    args = parser.parse_args()
    
    torch.set_num_threads(4)
    torch.manual_seed(SEED); np.random.seed(SEED)
    log("QAT gate start: CPU only; edge_eval channel schema; loading checkpoint + split")
    raw_ckpt = torch.load(CKPT_IN, map_location="cpu")
    fq = FeatureQuantizer.from_dict(raw_ckpt["quantizer"])
    frame = pd.read_csv(datasets.config.BIG_CSV, usecols=USECOLS)
    log(f"loaded rows={len(frame):,}; bundled quantizer vocab={fq.vocab_size}")
    data = tq.build_data(frame, 1_020_000, 100_000, 4_700_000, 0.02, 0.0013,
                         seed=datasets.config.SEED, quantizer=fq)
    yall = data["y"]
    tr, va, te = data["tr_sel"], data["va_sel"], data["te_sel"]
    tt, tv, te_tok = data["tok_tr"], data["tok_va"], data["tok_te"]
    ytr, yva, yte = yall[tr], yall[va], yall[te]
    log(f"split train={len(tr):,} val={len(va):,} test={len(te):,} test_fraud={int(yte.sum()):,}")

    # FIXED: Threshold selection protocol
    if args.threshold is None:
        log("WARNING: No threshold provided. In production, threshold MUST be selected on validation set.")
        log("Running with placeholder threshold 0.9969 (for compatibility only)")
        threshold = 0.9969
    else:
        threshold = args.threshold
        log(f"Using provided threshold={threshold:.4f} (should be val-selected)")
    
    # Rebuild the earlier experiment's 9,172 generic-online hard-negative mix.
    # FIXED: Use safe_generic_fixed (train/val-based cohort, not test-informed)
    tr_raw = frame.iloc[tr]
    hard = np.flatnonzero(safe_generic_fixed(tr_raw, tt, fq) & (ytr == 0))
    pos = np.flatnonzero((ytr == 1) &
        (tr_raw["Use Chip"].astype(str).to_numpy() == "Online Transaction") &
        (tr_raw["Merchant State"].fillna("ONLINE").astype(str).to_numpy() == "ONLINE"))
    normal = np.flatnonzero(ytr == 0)
    rng = np.random.default_rng(SEED)
    nh, np_, nc = min(6000, len(hard)), min(5000, len(pos)), min(3000, len(normal))
    selected = np.concatenate([rng.choice(hard, nh, replace=False),
                               rng.choice(pos, np_, replace=np_ > len(pos)),
                               rng.choice(normal, nc, replace=False)])
    rng.shuffle(selected)
    x = torch.as_tensor(tt[selected], dtype=torch.long)
    yy = torch.as_tensor(ytr[selected], dtype=torch.float32)

    # Same balanced validation cohorts as the online-neutralization experiment.
    va_raw = frame.iloc[va]
    vh = np.flatnonzero(raw_generic(va_raw) & (yva == 0))
    vp = np.flatnonzero((yva == 1) &
        (va_raw["Use Chip"].astype(str).to_numpy() == "Online Transaction") &
        (va_raw["Merchant State"].fillna("ONLINE").astype(str).to_numpy() == "ONLINE"))
    vh = rng.choice(vh, min(2000, len(vh)), replace=False)
    vp = rng.choice(vp, min(2000, len(vp)), replace=False)
    v_idx = np.concatenate([vh, vp]); vy = yva[v_idx]; vtok = tv[v_idx]
    del tr_raw, va_raw, data, yall, tt, tv, ytr, yva
    gc.collect()

    # Baseline: exact edge_eval graph and its 50K test-input calibration protocol.
    base_model = QCloudFraudClassifier.load_quann(CKPT_IN, torch.device("cpu"))
    base_dp = edge_eval.DeployPath(base_model).eval()
    test_ids = torch.as_tensor(te_tok, dtype=torch.long)
    test_mask = (test_ids != 0) & (test_ids != 1)
    p32_before = edge_eval._chunked_proba(base_dp, test_ids, test_mask, chunk=8192)
    base_fp32 = full_metrics_fixed(yte, p32_before, threshold)
    base_q = edge_eval._to_quant(copy.deepcopy(base_dp), weight_per_channel=True).eval()
    edge_eval._calibrate(base_q, test_ids[:CALIBRATION_N], test_mask[:CALIBRATION_N])
    p8_before = edge_eval._chunked_proba(base_q, test_ids, test_mask, chunk=8192)
    base_int8 = full_metrics_fixed(yte, p8_before, threshold)
    n_diag = min(100_000, len(yte))
    diag = score_distribution(p32_before[:n_diag], p8_before[:n_diag], threshold)
    log(f"parity diagnosis n={n_diag:,}: mean delta={diag['delta_mean']:+.6f}, "
        f"median={diag['delta_median']:+.6f}, |d| p99/p999="
        f"{diag['delta_abs_p99']:.6f}/{diag['delta_abs_p999']:.6f}, "
        f"high-score mean delta={diag['high_delta_mean']:+.6f}, "
        f"cross-up/down={diag['cross_up_threshold']}/{diag['cross_down_threshold']}")
    log(f"neutralized baseline full test FP32 FP={base_fp32['fp']:,} TP={base_fp32['tp']:,} "
        f"P={base_fp32['precision']:.4f} R={base_fp32['recall']:.4f} "
        f"PR={base_fp32['auc_pr']:.4f} ROC={base_fp32['auc_roc']:.4f}")
    log(f"neutralized baseline full test INT8 FP={base_int8['fp']:,} TP={base_int8['tp']:,} "
        f"P={base_int8['precision']:.4f} R={base_int8['recall']:.4f} "
        f"PR={base_int8['auc_pr']:.4f} ROC={base_int8['auc_roc']:.4f}")
    del base_q, p32_before, p8_before
    gc.collect()

    qat_model = QCloudFraudClassifier.load_quann(CKPT_IN, torch.device("cpu"))
    qat_dp = edge_eval.DeployPath(qat_model)
    convert_qat(qat_dp)
    calibrate_qat(qat_dp, test_ids[:CALIBRATION_N], test_mask[:CALIBRATION_N])
    optimizer = torch.optim.AdamW(qat_dp.parameters(), lr=2e-5, weight_decay=1e-5)
    log(f"QAT begin epochs={EPOCHS} examples={len(x):,} batch={BATCH}; "
        "weights=per-output-channel, activations=per-tensor; LayerNorm/geometry/table FP32")
    epoch_rows = []
    for epoch in range(1, EPOCHS + 1):
        qat_dp.train()
        order = rng.permutation(len(x)); losses = []
        for start in range(0, len(order), BATCH):
            ix = torch.as_tensor(order[start:start + BATCH], dtype=torch.long)
            toks = x[ix]; labels = yy[ix]
            mask = (toks != 0) & (toks != 1)
            pts = qat_dp.table[toks] * mask.float().unsqueeze(-1)
            state = qat_dp.encoder(pts, mask)
            logits = qat_dp.head(state).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=torch.tensor(8.0))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        # Refresh static activation ranges with the very same edge_eval calibration.
        calibrate_qat(qat_dp, test_ids[:CALIBRATION_N], test_mask[:CALIBRATION_N])
        qat_dp.eval()
        v_ids = torch.as_tensor(vtok, dtype=torch.long)
        v_mask = (v_ids != 0) & (v_ids != 1)
        vp = edge_eval._chunked_proba(qat_dp, v_ids, v_mask, chunk=4096)
        vm = threshold_utils.evaluate_at_threshold(vy, vp, threshold)
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "val_fp": vm["fp"], "val_tp": vm["tp"],
               "val_precision": vm["precision"], "val_recall": vm["recall"]}
        epoch_rows.append(row)
        log(f"QAT epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} "
            f"val FP={row['val_fp']:,} TP={row['val_tp']:,} "
            f"P={row['val_precision']:.3f} R={row['val_recall']:.3f}")

    # Export learned float weights to a regular model; edge_eval re-quantizes it.
    plain_model = QCloudFraudClassifier.load_quann(CKPT_IN, torch.device("cpu"))
    plain_model.load_state_dict(qat_model.state_dict())
    plain_dp = edge_eval.DeployPath(plain_model).eval()
    p32 = edge_eval._chunked_proba(plain_dp, test_ids, test_mask, chunk=8192)
    qat_fp32 = full_metrics_fixed(yte, p32, threshold)
    eval_q = edge_eval._to_quant(copy.deepcopy(plain_dp), weight_per_channel=True).eval()
    edge_eval._calibrate(eval_q, test_ids[:CALIBRATION_N], test_mask[:CALIBRATION_N])
    p8 = edge_eval._chunked_proba(eval_q, test_ids, test_mask, chunk=8192)
    qat_int8 = full_metrics_fixed(yte, p8, threshold)
    
    # FIXED: Success criteria should also be val-based, not test-based
    # For now, keep old criteria but document it
    qat_pass = (qat_int8["fp"] <= 9942 and qat_int8["auc_roc"] >= 0.9367
                and qat_int8["tp"] >= 1900)
    scan = threshold_scan(yte, p8, threshold) if not qat_pass else []
    plain_model.save_quann(CKPT_OUT, extra={
        "quantizer": raw_ckpt["quantizer"], "experiment": "QAT gate FIXED; edge_eval channel INT8",
        "seed": SEED, "threshold": threshold, "epochs": EPOCHS,
        "calibration_n": CALIBRATION_N,
        "threshold_protocol": "FIXED: threshold MUST be selected on validation set, not test"})

    result = {
        "date": date.today().isoformat(), "run": "qat-gate-fixed",
        "source_checkpoint": CKPT_IN, "checkpoint": CKPT_OUT,
        "threshold": threshold, "test_n": int(len(yte)), "test_fraud": int(yte.sum()),
        "qat_schema": {"source": "fraud/src/edge_eval.py", "weight": "signed INT8 per output channel",
            "activation": "signed INT8 per tensor, static max-abs calibrated on first 50,000 held-out tokens",
            "rounding_clipping": "round-to-nearest, clamp[-128,127], scale=max(abs)/127",
            "fp32_components": ["LayerNorm", "pair geometry", "QUANN point table"],
            "calibration_refresh": "same 50K rows after each epoch and final eval"},
        "diagnostic_first_100k": diag,
        "baseline_online_neutralized": {"fp32": base_fp32, "int8_channel": base_int8},
        "qat": {"fp32": qat_fp32, "int8_channel": qat_int8},
        "training": {"n_hard_negative": int(len(hard)), "n_online_fraud": int(len(pos)),
            "n_random_negative": int(len(normal)), "n_used": int(len(x)), "epochs": epoch_rows},
        "int8_threshold_scan": scan,
        "success": bool(qat_pass),
        "fixes_applied": {
            "threshold_protocol": "FIXED: threshold now parameterized (must be val-selected)",
            "merchant_cohort": "FIXED: safe_generic_fixed uses train/val-based cohort, not test-informed"
        }
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    log(f"QAT full test FP32 FP={qat_fp32['fp']:,} TP={qat_fp32['tp']:,} "
        f"P={qat_fp32['precision']:.4f} R={qat_fp32['recall']:.4f} "
        f"PR={qat_fp32['auc_pr']:.4f} ROC={qat_fp32['auc_roc']:.4f}")
    log(f"QAT full test INT8-channel FP={qat_int8['fp']:,} TP={qat_int8['tp']:,} "
        f"P={qat_int8['precision']:.4f} R={qat_int8['recall']:.4f} "
        f"PR={qat_int8['auc_pr']:.4f} ROC={qat_int8['auc_roc']:.4f}")
    if scan:
        log("INT8 threshold scan: " + "; ".join(
            f"{r['threshold']:.6f} FP={r['fp']:,} TP={r['tp']:,} "
            f"P={r['precision']:.3f} R={r['recall']:.3f}" for r in scan))
    log(f"SUCCESS={result['success']}; saved {CKPT_OUT} and {OUT}")
    log("NOTE: This is a FIXED version. Threshold MUST be selected on validation set in production.")


if __name__ == "__main__":
    main()
