"""edge_eval.py — Can QUANN *hybrid* be a small, efficient EDGE model?

Loads the trained hybrid fraud model (`results/quann_hybrid.pt`) and measures the
true DEPLOY graph — token ids -> QUANN point table (gather) -> CloudEncoderND ->
class_head -> P(fraud) — on CPU. The LM path (embedding/to_nd) is NOT deployed.

Sections:
  1. footprint   : reader param count, FP32/INT8 byte size, QUANN point table
  2. latency     : p50/p99 ms, tx/s  (synthetic tokens, K features)
  3. k_scaling   : O(K^2) pair-geometry cost across K = 13/25/50
  4. onnx        : export deploy graph, onnxruntime parity + latency
  5. int8        : FX dynamic quant, FP32 vs INT8 output drift (real QUANN points)
  6. int8_auc    : [--with-auc] reproduce the exact held-out test, FP32 vs INT8 AUC

Runs on CPU (MPS is held free for the live CloudLM 1.5M run).

Run:
  python3 fraud/src/edge_eval.py                 # sections 1-5 (no big data)
  python3 fraud/src/edge_eval.py --with-auc      # + 6 (loads the full 24.4M CSV)

NOTE: latency here is an Apple-Silicon laptop (CPU) figure = a *lower bound*.
Re-run the same script on the target edge device/runtime for the real gate.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                      # noqa: F401  (sys.path bootstrap)
from qcloud_fraud import QCloudFraudClassifier
from quantize import FeatureQuantizer


# ------------------------------------------------------------------ deploy graph
class DeployPath(torch.nn.Module):
    """The deployable hybrid graph (permutation-invariant cloud reader).

    token_ids (B,K) -> gather QUANN point table (V,D) -> CloudEncoderND -> class_head
    -> sigmoid -> P(fraud).  No embedding / no sequence transformer / no backprop.
    """

    def __init__(self, model: QCloudFraudClassifier):
        super().__init__()
        self.encoder = model.cloud_encoder
        self.head = model.class_head
        tab = torch.zeros(model.vocab_size, model.D)
        for tid, w in model.token_weights.items():
            if 0 <= tid < model.vocab_size:
                tab[tid] = torch.tensor(w.value, dtype=torch.float32)
        self.register_buffer("table", tab)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pts = self.table[token_ids.long()] * mask.float().unsqueeze(-1)
        state = self.encoder(pts, mask)
        return torch.sigmoid(self.head(state)).squeeze(-1)


def _log(msg: str, fp):
    line = f"=== [{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg} ==="
    print(line, flush=True)
    if fp:
        fp.write(line + "\n")


def _time_forward(fn, reps: int, warmup: int = 10) -> np.ndarray:
    """Per-call wall time (s) over `reps` calls, after `warmup` warmups."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


# Engine-independent INT8 proxy: fake-quant models the 8-bit precision floor of
# weight+activation quantization without a QNNPACK/X86 engine. The deploy graph
# keeps LayerNorm + the geometry (diff/dist/cos) + the QUANN point table in fp32
# (mixed precision, as CoreML/TFLite do); only the MLP Linears go to 8-bit.
#   mode "tensor"   : per-tensor weight + per-tensor activation  (conservative floor)
#   mode "channel"  : per-output-channel weight + per-tensor act. (standard INT8)
# Scales are CALIBRATED on real inputs (static), matching how runtimes quantize.
def _fq_quantize(t: torch.Tensor, scale, bits: int = 8) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    if scale is None:
        scale = t.detach().abs().max().clamp(min=1e-8) / qmax
    return ((t / scale).clamp(-(qmax + 1), qmax).round()) * scale


def _fq_quantize_pc(w: torch.Tensor, scales, bits: int = 8) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    if scales is None:
        scales = w.detach().abs().max(dim=1).values.clamp(min=1e-8) / qmax
    return ((w / scales.unsqueeze(1)).clamp(-(qmax + 1), qmax).round()) * scales.unsqueeze(1)


class _QuantLinear(torch.nn.Module):
    def __init__(self, lin: torch.nn.Linear, weight_per_channel: bool):
        super().__init__()
        self.in_f, self.out_f = lin.in_features, lin.out_features
        self.weight = torch.nn.Parameter(lin.weight.detach().clone(), requires_grad=False)
        self.bias = (torch.nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
                     if lin.bias is not None else None)
        self.weight_per_channel = weight_per_channel
        self.act_scale = None   # calibrated per-tensor activation scale
        self.w_scale = None     # calibrated scalar (tensor) or (out_f,) (channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = _fq_quantize(x, self.act_scale)
        wq = (_fq_quantize_pc(self.weight, self.w_scale) if self.weight_per_channel
              else _fq_quantize(self.weight, self.w_scale))
        return torch.nn.functional.linear(xq, wq, self.bias)


def _to_quant(m: torch.nn.Module, weight_per_channel: bool) -> torch.nn.Module:
    for name, mod in list(m.named_children()):
        if isinstance(mod, torch.nn.Linear):
            setattr(m, name, _QuantLinear(mod, weight_per_channel))
        else:
            _to_quant(mod, weight_per_channel)
    return m


def _calibrate(m: torch.nn.Module, calib_ids: torch.Tensor, calib_mask: torch.Tensor):
    """Capture static per-Linear activation + weight scales from real inputs."""
    m.eval()
    qmax = 127.0

    def _pre(module, inp):
        x = inp[0]
        module.act_scale = float(x.detach().abs().max().clamp(min=1e-8) / qmax)
        module.w_scale = (module.weight.detach().abs().max(dim=1).values.clamp(min=1e-8) / qmax
                          if module.weight_per_channel
                          else float(module.weight.detach().abs().max().clamp(min=1e-8) / qmax))

    hooks = [mod.register_forward_pre_hook(_pre)
             for mod in m.modules() if isinstance(mod, _QuantLinear)]
    with torch.no_grad():
        m(calib_ids, calib_mask)
    for h in hooks:
        h.remove()


def _free_gb() -> float:
    """Approx reclaimable GB (free + inactive). Guards the big-CSV AUC load."""
    import re
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        free = int(re.search(r"Pages free:\s+(\d+)", out).group(1))
        inact = int(re.search(r"Pages inactive:\s+(\d+)", out).group(1))
        return (free + inact) * page / 1e9
    except Exception:
        return 1e9  # unknown -> proceed


# ------------------------------------------------------------------ section 1
def footprint(model: QCloudFraudClassifier, fp, ckpt_path: str = None):
    _log("1/6 footprint (deploy graph)", fp)
    reader = sum(p.numel() for p in model.cloud_encoder.parameters())
    head = sum(p.numel() for p in model.class_head.parameters())
    lm_dead = sum(p.numel() for n, p in model.named_parameters()
                  if n.startswith("embedding.") or n.startswith("to_nd."))
    table_f = model.vocab_size * model.D
    n_quann = len(model.token_weights)

    fp32 = (reader + head + table_f) * 4
    # INT8 byte size: Linear leaves -> 1B/param; LayerNorm (leaf) + QUANN table stay fp32.
    lin_n = oth_n = 0
    for mod in (model.cloud_encoder, model.class_head):
        for sub in mod.modules():
            if list(sub.children()):
                continue
            n = sum(p.numel() for p in sub.parameters())
            if isinstance(sub, torch.nn.Linear):
                lin_n += n
            elif n:
                oth_n += n
    int8 = lin_n * 1 + oth_n * 4 + table_f * 4

    res = {
        "reader_params": reader, "head_params": head, "lm_dead_params": lm_dead,
        "quann_point_ids": n_quann, "deploy_vocab": model.vocab_size,
        "table_floats": table_f,
        "deploy_fp32_kb": round(fp32 / 1024, 1),
        "deploy_int8_kb": round(int8 / 1024, 1),
        "pt_file_kb": round(os.path.getsize(ckpt_path or os.path.join(config.RESULTS_DIR, "quann_hybrid.pt")) / 1024, 1),
    }
    for k, v in res.items():
        print(f"  {k:18s} {v:,}")
    if fp:
        fp.write(str(res) + "\n")
    return res


# ------------------------------------------------------------------ section 2
def latency(dp: DeployPath, model, n_tx: int, k: int, reps: int, fp) -> dict:
    _log(f"2/6 latency (K={k}, n_tx={n_tx}) — laptop CPU, lower bound", fp)
    t = torch.randint(2, model.vocab_size, (n_tx, k))
    msk = (t != 0) & (t != 1)
    with torch.no_grad():
        ts = _time_forward(lambda: dp(t, msk), reps=reps)
    ms = ts * 1e3                      # per full-batch call
    res = {
        "k": k, "n_tx": n_tx,
        "batch_p50_ms": float(np.percentile(ms, 50)),
        "per_tx_p50_ms": float(np.percentile(ms, 50) / n_tx),   # one auth decision
        "per_tx_p99_ms": float(np.percentile(ms, 99) / n_tx),
        "tx_per_s": float(n_tx * (len(ts) / ts.sum())),
    }
    print(f"  per-tx p50={res['per_tx_p50_ms']:.4f}ms  p99={res['per_tx_p99_ms']:.4f}ms  "
          f"{res['tx_per_s']:.0f} tx/s   (batch {res['batch_p50_ms']:.1f}ms/{n_tx})")
    if fp:
        fp.write(str(res) + "\n")
    return res


# ------------------------------------------------------------------ section 3
def _macs(model, k: int, d_cloud: int, hidden: int) -> int:
    """Analytic MACs/forward for the deploy graph at K points (dense pair cost)."""
    D = model.D
    pt = (D * hidden + hidden) + (hidden * hidden + hidden) + (hidden * d_cloud + d_cloud) * 0 \
        + (hidden * hidden + hidden) + (hidden * d_cloud + d_cloud)
    pt = (D * hidden + hidden) + (hidden * hidden + hidden) + (hidden * d_cloud + d_cloud)
    pr = ((D + 4) * hidden + hidden) + (hidden * hidden + hidden) + (hidden * d_cloud + d_cloud)
    geom = 3 * k * k * D
    fus = (2 * d_cloud * 2 * d_cloud + 2 * d_cloud) + (2 * d_cloud * d_cloud + d_cloud)
    return int(pt * k + pr * k * k + geom + fus)


def k_scaling(dp: DeployPath, model, reps: int, fp, ks=(13, 25, 50)) -> list:
    _log("3/6 K-scaling (pair geometry is O(K^2))", fp)
    out = []
    d_cloud = model.d_cloud
    hidden = model.cloud_encoder.point_mlp[0].out_features
    for k in ks:
        nb = 512
        t = torch.randint(2, model.vocab_size, (nb, k))
        msk = (t != 0) & (t != 1)
        with torch.no_grad():
            ts = _time_forward(lambda: dp(t, msk), reps=reps)
        r = {"k": k, "macs_per_tx": _macs(model, k, d_cloud, hidden),
             "per_tx_p50_ms": float(np.percentile(ts * 1e3, 50) / nb),
             "tx_per_s": float(nb * (len(ts) / ts.sum()))}
        out.append(r)
        print(f"  K={k:2d}  MACs={r['macs_per_tx']:>9,}  "
              f"per-tx p50={r['per_tx_p50_ms']:.4f}ms  {r['tx_per_s']:.0f} tx/s")
    if fp:
        fp.write(str(out) + "\n")
    return out


# ------------------------------------------------------------------ section 4
def onnx_export(dp: DeployPath, model, fp) -> dict:
    _log("4/6 ONNX export + onnxruntime parity/latency", fp)
    f = os.path.join(config.RESULTS_DIR, "edge_deploy.onnx")
    t = torch.randint(2, model.vocab_size, (2, 13))
    msk = (t != 0) & (t != 1)
    with torch.no_grad():
        ref = dp(t, msk).numpy()
    torch.onnx.export(dp, (t, msk), f, input_names=["ids", "mask"],
                      output_names=["p"],
                      dynamic_axes={"ids": {0: "B"}, "mask": {0: "B"}, "p": {0: "B"}},
                      dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(f, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"ids": t.numpy(), "mask": msk.numpy()})[0]
    diff = float(np.abs(ref - o).max())

    big = torch.randint(2, model.vocab_size, (2000, 13))
    bmsk = (big != 0) & (big != 1)
    ts = _time_forward(lambda: sess.run(None, {"ids": big.numpy(), "mask": bmsk.numpy()}),
                       reps=50)
    res = {"onnx_kb": round(os.path.getsize(f) / 1024, 1),
           "ort_max_abs_diff": diff,
           "ort_per_tx_p50_ms": float(np.percentile(ts * 1e3, 50) / 2000),
           "ort_tx_per_s": float(2000 * (len(ts) / ts.sum()))}
    print(f"  {res['onnx_kb']}KB  |torch-ORT| max={diff:.2e}  "
          f"ORT per-tx p50={res['ort_per_tx_p50_ms']:.4f}ms {res['ort_tx_per_s']:.0f} tx/s")
    if fp:
        fp.write(str(res) + "\n")
    return res


# ------------------------------------------------------------------ section 5
def int8_drift(dp: DeployPath, model, n: int, fp) -> dict:
    _log(f"5/6 INT8 (calibrated per-tensor) output drift on {n} random QUANN-point inputs (proxy)", fp)
    import copy
    ids = list(model.token_weights.keys())
    t = torch.tensor(np.random.default_rng(0).choice(ids, size=(n, 13), replace=True),
                     dtype=torch.long)
    msk = (t != 0) & (t != 1)
    q = _to_quant(copy.deepcopy(dp), weight_per_channel=False)
    _calibrate(q, t, msk)
    q.eval(); dp.eval()
    with torch.no_grad():
        pf = dp(t, msk)
        pi = q(t, msk)
    d = (pi - pf)
    res = {"int8_max_abs_dp": float(d.abs().max()),
           "int8_mean_abs_dp": float(d.abs().mean()),
           "int8_p99_abs_dp": float(d.abs().quantile(0.99))}
    print(f"  max|dP|={res['int8_max_abs_dp']:.4f}  mean|dP|={res['int8_mean_abs_dp']:.5f}  "
          f"p99|dP|={res['int8_p99_abs_dp']:.4f}")
    if fp:
        fp.write(str(res) + "\n")
    return res


# ------------------------------------------------------------------ section 6
def _chunked_proba(mod, t: torch.Tensor, msk: torch.Tensor, chunk: int = 50000) -> np.ndarray:
    """Batched proba (bounds the O(K^2) pair tensor) -> (n,) float."""
    mod.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, t.shape[0], chunk):
            outs.append(mod(t[s:s + chunk], msk[s:s + chunk]).numpy())
    return np.concatenate(outs)


def int8_auc(base: DeployPath, fp=None, n_test: int = 300000, test_frac: float = 0.0013,
             calib_n: int = 50000, fq: FeatureQuantizer = None) -> dict:
    """FP32 vs INT8 (tensor / channel) AUC on a held-out test.

    If `fq` (the checkpoint's bundled quantizer) is provided, the test is
    tokenized with it -> AUC is reproducible. Otherwise a quantizer is re-fit
    on the train split (approximate; older checkpoints predate the bundled
    quantizer, so their recorded AUC is not fully reproducible).
    LayerNorm + geometry + QUANN table stay fp32 (mixed precision); only MLP
    Linears go 8-bit. Scales calibrated on a calib slice, AUC on the full test.
    """
    _log("6/6 INT8 AUC — FP32 vs INT8(tensor/channel) on NATURAL-rate held-out test", fp)
    import copy
    import train_quann as tq
    from sklearn.metrics import roc_auc_score

    usecols = ["User", "Is Fraud?", "Amount", "Time", "Month", "Day", "Year",
               "Zip", "Use Chip", "MCC", "Merchant City", "Merchant Name",
               "Merchant State", "Errors?"]
    print(f"  reading {config.BIG_CSV} (usecols={len(usecols)}) ...", flush=True)
    df = _read_big(usecols)
    print(f"  rows={len(df):,}", flush=True)

    data = tq.build_data(df, 50000, 10000, n_test, 0.5, test_frac)
    yte = data["y"][data["te_sel"]]
    if fq is not None:
        tok_te = fq.transform(df.iloc[data["te_sel"]])
    else:
        tok_te = data["tok_te"]
    del df
    gc.collect()
    print(f"  tokenized with {'bundled' if fq is not None else 're-fit'} quantizer", flush=True)

    t = torch.tensor(tok_te, dtype=torch.long)
    msk = (t != 0) & (t != 1)
    calib, calib_msk = t[:calib_n], msk[:calib_n]
    base.eval()

    pf = _chunked_proba(base, t, msk)
    a_fp32 = float(roc_auc_score(yte, pf))
    print(f"  {'FP32':14s} AUC-ROC={a_fp32:.4f}  (n={len(yte):,}, fraud={int(yte.sum())}, "
          f"base={yte.mean():.4%})")

    def _auc_of(mod, tag):
        p = _chunked_proba(mod, t, msk)
        a = float(roc_auc_score(yte, p))
        print(f"  {tag:14s} AUC-ROC={a:.4f}  max|dP| vs FP32={float(np.abs(p - pf).max()):.4f}")
        return a

    qt = _to_quant(copy.deepcopy(base), weight_per_channel=False)
    _calibrate(qt, calib, calib_msk)
    a_tensor = _auc_of(qt, "INT8 tensor")

    qc = _to_quant(copy.deepcopy(base), weight_per_channel=True)
    _calibrate(qc, calib, calib_msk)
    a_channel = _auc_of(qc, "INT8 channel")

    res = {"n_test": int(len(yte)), "n_fraud": int(yte.sum()),
           "base_rate": float(yte.mean()),
           "auc_fp32": a_fp32, "auc_int8_tensor": a_tensor,
           "auc_int8_channel": a_channel,
           "delta_tensor": a_tensor - a_fp32,
           "delta_channel": a_channel - a_fp32,
           "tokenized_with": "bundled quantizer" if fq is not None else "re-fit quantizer",
           "note": ("recorded ref AUC-ROC=0.9581 is NOT reproducible from this checkpoint "
                     "(its input quantizer was not saved at train time); re-fit quantizer "
                     "reproduces FP32 ~0.77 on the 4.7M test. See fraud/REVISION.md."
                     if fq is None else
                     "tokenized with the checkpoint's bundled quantizer -> AUC reproducible.")}
    print(f"  dAUC tensor={res['delta_tensor']:+.4f}  channel={res['delta_channel']:+.4f}")
    if fp:
        fp.write(str(res) + "\n")
    return res


def _read_big(usecols):
    import pandas as pd
    return pd.read_csv(config.BIG_CSV, usecols=usecols)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-auc", action="store_true",
                    help="also load the full 24.4M CSV to measure FP32 vs INT8 AUC")
    ap.add_argument("--n-tx", type=int, default=2000, help="latency sample size")
    ap.add_argument("--reps", type=int, default=100, help="latency reps per call")
    ap.add_argument("--n-test-auc", type=int, default=300000, help="AUC test size (natural rate)")
    ap.add_argument("--test-frac", type=float, default=0.0013, help="AUC test fraud rate (~natural)")
    ap.add_argument("--checkpoint",
                    default=os.path.join(config.RESULTS_DIR, "quann_hybrid_full.pt"))
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    log_path = os.path.join(config.LOGS_DIR, "run_edge_eval.log")
    fp = open(log_path, "a")

    _log("EDGE EVAL — QUANN hybrid as a small edge model", fp)
    print(f"  checkpoint: {args.checkpoint}")
    if fp:
        fp.write(f"  checkpoint: {args.checkpoint}\n")

    model = QCloudFraudClassifier.load_quann(args.checkpoint, torch.device("cpu"))
    print(f"  {model}")
    if fp:
        fp.write(f"  {model}\n")

    # bundled quantizer (self-contained checkpoint) -> reproducible tokenization
    _raw = torch.load(args.checkpoint, map_location="cpu")
    fq = FeatureQuantizer.from_dict(_raw["quantizer"]) if _raw.get("quantizer") else None
    print(f"  bundled quantizer: {'yes (vocab=%d)' % fq.vocab_size if fq else 'no (re-fit fallback)'}")
    if fp:
        fp.write(f"  bundled quantizer: {'yes' if fq else 'no (re-fit fallback)'}\n")

    dp = DeployPath(model).eval()
    results = {}
    results["config"] = {"D": model.D, "d_cloud": model.d_cloud, "d_model": model.d_model,
                         "vocab": model.vocab_size}
    results["footprint"] = footprint(model, fp, ckpt_path=args.checkpoint)
    results["latency"] = latency(dp, model, args.n_tx, 13, args.reps, fp)
    results["k_scaling"] = k_scaling(dp, model, max(30, args.reps // 2), fp)
    results["onnx"] = onnx_export(dp, model, fp)
    results["int8_drift"] = int8_drift(dp, model, 4096, fp)

    if args.with_auc:
        if _free_gb() < 8:
            msg = f"skipped: only {_free_gb():.1f}GB reclaimable (live run priority)"
            print("  " + msg)
            results["int8_auc"] = {"skipped": msg}
        else:
            results["int8_auc"] = int8_auc(dp, fp=fp, n_test=args.n_test_auc,
                                            test_frac=args.test_frac, fq=fq)
    else:
        results["int8_auc"] = {"skipped": "run with --with-auc (needs full 24.4M CSV)"}

    import json
    out = os.path.join(config.RESULTS_DIR, "edge_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    _log(f"done -> {out}", fp)
    fp.close()
    print(f"\n  {json.dumps(results['footprint'])}")
    print(f"  {json.dumps(results['latency'])}")
    print(f"  {json.dumps(results['int8_auc'])}")


if __name__ == "__main__":
    main()
