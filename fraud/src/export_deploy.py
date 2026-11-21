from __future__ import annotations

import argparse
import hashlib
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
from edge_eval import DeployPath
from qcloud_fraud import QCloudFraudClassifier
from quantize import FeatureQuantizer


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_decoder(fq: FeatureQuantizer) -> dict:
    """Full vocabulary token_id -> {feature, label}; 0/1 are reserved specials."""
    dec: dict[str, dict] = {
        "0": {"feature": None, "label": "pad"},
        "1": {"feature": None, "label": "eos"},
    }
    for j, key in enumerate(fq.feature_order):
        lo = fq.offsets[key]
        for local in range(fq.n_per_feature[key]):
            dec[str(lo + local)] = {
                "feature": key,
                "label": explain_pilot.decode_token(fq, j, lo + local),
            }
    return dec


def export_onnx(model: QCloudFraudClassifier, path: str) -> dict:
    dp = DeployPath(model)
    t = torch.randint(2, model.vocab_size, (2, 13))
    msk = (t != 0) & (t != 1)
    torch.onnx.export(dp, (t, msk), path, input_names=["ids", "mask"],
                      output_names=["p"],
                      dynamic_axes={"ids": {0: "B"}, "mask": {0: "B"}, "p": {0: "B"}},
                      dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = dp(t, msk).numpy()
    out = sess.run(None, {"ids": t.numpy(), "mask": msk.numpy()})[0]
    return {
        "ort": sess,
        "onnx_kb": round(os.path.getsize(path) / 1024, 1),
        "export_max_abs_diff": float(np.abs(ref - out).max()),
    }


def deploy_self_test(model: QCloudFraudClassifier, fq: FeatureQuantizer,
                     frame: pd.DataFrame, ort, n_cases: int = 3) -> dict:
    """Real-row parity (torch vs ONNXRuntime) + latency + decoded examples."""
    tok = fq.transform(frame)
    labels = datasets.get_label(frame)
    t = torch.tensor(tok, dtype=torch.long)
    msk = (t != 0) & (t != 1)
    with torch.no_grad():
        p_torch = model.predict_proba(t, use_quann_points=True, batch=5000).cpu().numpy()
    p_ort = ort.run(None, {"ids": tok, "mask": msk.numpy()})[0]
    diff = np.abs(p_torch - p_ort)

    t0 = time.perf_counter()
    n_reps = 10
    for _ in range(n_reps):
        ort.run(None, {"ids": tok, "mask": msk.numpy()})
    tx_per_s = len(tok) * n_reps / (time.perf_counter() - t0)

    cases = []
    fraud = np.flatnonzero(labels == 1)
    normal = np.flatnonzero(labels == 0)
    picks = []
    if len(fraud):
        picks.append(int(fraud[np.argmax(p_torch[fraud])]))
    if len(normal):
        picks.append(int(normal[np.argmax(p_torch[normal])]))
    if len(fraud):
        picks.append(int(fraud[np.argmin(p_torch[fraud])]))
    for i in picks[:n_cases]:
        cases.append({
            "row": int(i),
            "true_class": "fraud" if labels[i] == 1 else "normal",
            "p_fraud": float(p_torch[i]),
            "decoded": explain_pilot.decode_row(fq, tok[i]),
        })
    return {
        "n_rows": int(len(tok)),
        "max_abs_diff": float(diff.max()),
        "p99_abs_diff": float(np.percentile(diff, 99)),
        "tx_per_s": float(tx_per_s),
        "cases": cases,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(config.RESULTS_DIR, "quann_full_hybrid.pt"))
    ap.add_argument("--n-rows", type=int, default=None,
                    help="cap self-test rows (default: all rows of the small CSV)")
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--out-dir", default=os.path.join(config.RESULTS_DIR, "deploy"))
    ap.add_argument("--keep-onnx", action="store_true", default=True,
                    help="write model.onnx into --out-dir (default: on)")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(config.LOGS_DIR, "run_export_deploy.log")
    log_fp = open(log_path, "a")

    def log(msg: str):
        line = f"=== [{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg} ==="
        print(line, flush=True)
        log_fp.write(line + "\n")

    log("FRAUD DEPLOY EXPORT — ONNX + bundled quantizer + token decoder package")
    print(f"  checkpoint: {args.checkpoint}", flush=True)
    log_fp.write(f"  checkpoint: {args.checkpoint}\n")

    raw = torch.load(args.checkpoint, map_location="cpu")
    if not raw.get("quantizer"):
        raise SystemExit("deploy export requires a checkpoint with a bundled quantizer")
    model = QCloudFraudClassifier.load_quann(args.checkpoint, torch.device("cpu"))
    fq = FeatureQuantizer.from_dict(raw["quantizer"])
    print(f"  {model} | vocab={fq.vocab_size} K={fq.n_features}", flush=True)
    log_fp.write(f"  {model} | vocab={fq.vocab_size} K={fq.n_features}\n")

    onnx_path = os.path.join(args.out_dir, "model.onnx")
    log("1/4 ONNX export (dynamic batch)")
    exp = export_onnx(model, onnx_path)
    print(f"  {exp['onnx_kb']}KB | export parity max={exp['export_max_abs_diff']:.2e}", flush=True)
    log_fp.write(f"  {exp['onnx_kb']}KB | export parity max={exp['export_max_abs_diff']:.2e}\n")

    log("2/4 quantizer + token decoder artifacts")
    q_path = os.path.join(args.out_dir, "quantizer.json")
    with open(q_path, "w") as f:
        json.dump(fq.to_dict(), f, indent=2, default=float)
    decoder = build_decoder(fq)
    d_path = os.path.join(args.out_dir, "token_decoder.json")
    with open(d_path, "w") as f:
        json.dump(decoder, f, indent=2)
    print(f"  quantizer {os.path.getsize(q_path) / 1024:.1f}KB | "
          f"decoder {os.path.getsize(d_path) / 1024:.1f}KB ({len(decoder)} entries)", flush=True)
    log_fp.write(f"  quantizer {os.path.getsize(q_path) / 1024:.1f}KB | "
                 f"decoder {os.path.getsize(d_path) / 1024:.1f}KB ({len(decoder)} entries)\n")

    log("3/4 real-row self-test (bundled User0 CSV)")
    frame = datasets.load_small()
    if args.n_rows is not None:
        frame = frame.iloc[:args.n_rows]
    st = deploy_self_test(model, fq, frame, exp["ort"], n_cases=args.n_cases)
    print(f"  n={st['n_rows']:,} max|diff|={st['max_abs_diff']:.2e} "
          f"p99={st['p99_abs_diff']:.2e} {st['tx_per_s']:.0f} tx/s", flush=True)
    log_fp.write(f"  n={st['n_rows']:,} max|diff|={st['max_abs_diff']:.2e} "
                 f"p99={st['p99_abs_diff']:.2e} {st['tx_per_s']:.0f} tx/s\n")

    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": {
            "path": os.path.abspath(args.checkpoint),
            "size_kb": round(os.path.getsize(args.checkpoint) / 1024, 1),
            "sha256": sha256_of(args.checkpoint),
        },
        "model": {
            "D": model.D, "d_model": model.d_model, "d_cloud": model.d_cloud,
            "vocab": model.vocab_size, "quann_points": len(model.token_weights),
            "K": fq.n_features,
        },
        "artifacts": {
            "model_onnx": "model.onnx",
            "model_onnx_kb": exp["onnx_kb"],
            "model_onnx_sha256": sha256_of(onnx_path),
            "quantizer": "quantizer.json",
            "quantizer_kb": round(os.path.getsize(q_path) / 1024, 1),
            "token_decoder": "token_decoder.json",
            "token_decoder_kb": round(os.path.getsize(d_path) / 1024, 1),
            "decoder_entries": len(decoder),
        },
        "self_test": st,
        "usage": {
            "inputs": "ids (B,K) int64 token matrix + mask (B,K) bool (mask = ids not in {0,1})",
            "output": "p (B) = P(fraud)",
            "tokenization": "quantizer.json (FeatureQuantizer.to_dict) maps raw rows -> ids",
            "explanation": "token_decoder.json maps id -> human-readable label",
        },
        "method": {
            "graph": "baked QUANN point table (V,D) + CloudEncoderND + class_head + sigmoid; "
                     "same graph as edge_deploy.onnx, re-exported into the deploy package",
            "parity": "real-row torch vs onnxruntime on the bundled User0 CSV",
        },
    }
    m_path = os.path.join(args.out_dir, "manifest.json")
    with open(m_path, "w") as f:
        json.dump(manifest, f, indent=2, default=float)
    log(f"4/4 done -> {args.out_dir} (manifest {os.path.getsize(m_path) / 1024:.1f}KB)")
    log_fp.close()
    for c in st["cases"]:
        print(f"  [{c['true_class']}] p={c['p_fraud']:.3f} | "
              + "; ".join(c["decoded"][:5]))


if __name__ == "__main__":
    main()
