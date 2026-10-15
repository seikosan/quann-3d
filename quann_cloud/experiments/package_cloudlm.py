"""
CloudLM — Edge Paketleme (tam INT8 + ONNX + int8 GEMM hızı)
============================================================

Eğitilmiş CloudLM'yi edge için paketler:
  A) Boyut: FP32 / weight-only INT8 / tam INT8 (embedding dahil)
  B) Kalite: FP32 vs INT8 PPL
  C) ONNX export (FP32) → onnxruntime hızı (CPU + CoreML)
  D) ONNX INT8 (quantize) → onnxruntime int8 GEMM hızı
  E) Üretim kalitesi (taze model)

Çalıştır:
    python3 quann_cloud/experiments/package_cloudlm.py \
        --ckpt quann_cloud/checkpoints/cloudlm_best.pt \
        --bpe quann_cloud/checkpoints/cloudlm_bpe.json \
        --corpus turkish_sentences.txt
"""

import os
import sys
import math
import time
import random
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
from train_wiki import CloudLM, BPETokenizer, load_txt

try:
    import onnxruntime as ort
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False


class QuantLinear(nn.Module):
    """Weight-only INT8 (per-channel scale); forward'de dequant."""
    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data
        scale = (w.abs().amax(dim=1).clamp(min=1e-8) / 127.0)
        wq = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        self.register_buffer("scale", scale)
        self.register_buffer("weight_int8", wq)
        self.register_buffer("bias", lin.bias.data.clone() if lin.bias is not None
                            else torch.zeros(lin.out_features))

    def forward(self, x):
        w = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype).unsqueeze(1)
        return F.linear(x, w, self.bias)


class QuantEmbedding(nn.Module):
    """INT8 embedding (per-row scale)."""
    def __init__(self, emb: nn.Embedding):
        super().__init__()
        w = emb.weight.data
        scale = (w.abs().amax(dim=1).clamp(min=1e-8) / 127.0)
        wq = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        self.register_buffer("weight_int8", wq)
        self.register_buffer("scale", scale)
        self.padding_idx = emb.padding_idx
        self.embedding_dim = emb.embedding_dim
        self.num_embeddings = emb.num_embeddings

    def forward(self, x):
        w = self.weight_int8.to(torch.float32) * self.scale.unsqueeze(1)
        return F.embedding(x, w, self.padding_idx)


def quantize_full(m):
    """Tüm Linear + embedding'i INT8'e çevir (encoder hariç — .weight'e doğrudan erişir)."""
    for name, mod in list(m.named_modules()):
        if isinstance(mod, nn.Linear) and not name.startswith("encoder"):
            _replace(m, name, QuantLinear(mod))
        elif isinstance(mod, nn.Embedding):
            _replace(m, name, QuantEmbedding(mod))
    return m


def _replace(root, name, new):
    parent = root
    parts = name.split('.')
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def param_bytes(m, int8):
    b = 0
    for _, t in list(m.named_buffers()) + list(m.named_parameters()):
        if int8 and t.dtype == torch.int8:
            b += t.numel() * 1
        else:
            b += t.numel() * 4
    return b


def eval_ppl(model, tok, texts, seq_len, device, max_n=400):
    model.eval()
    eos = tok._tokenizer.token_to_id("<eos>") or 1
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for t in texts[:max_n]:
            ids = tok.encode(t)
            for i in range(len(ids) - seq_len):
                inp = torch.tensor([ids[i:i + seq_len]], device=device)
                tgt = torch.tensor([ids[i + seq_len]], device=device)
                tot += F.cross_entropy(model(inp, inp != 0)[0], tgt).item()
                cnt += 1
    return math.exp(min(tot / max(cnt, 1), 20)), cnt


def onnx_latency(session, input_name, shape, n=20, warmup=3):
    x = np.zeros(shape, dtype=np.int64)
    for _ in range(warmup):
        session.run(None, {input_name: x})
    t0 = time.time()
    for _ in range(n):
        session.run(None, {input_name: x})
    return (time.time() - t0) / n * 1000  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "quann_cloud", "checkpoints", "cloudlm_best.pt"))
    ap.add_argument("--bpe", default=os.path.join(ROOT, "quann_cloud", "checkpoints", "cloudlm_bpe.json"))
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "turkish_sentences.txt"))
    ap.add_argument("--eval-sent", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--gen-len", type=int, default=15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    device = torch.device(args.device or
                          ("mps" if torch.backends.mps.is_available() else "cpu"))

    print("=" * 66)
    print("  CloudLM — Edge Paketleme")
    print("=" * 66)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    if "d_dim" in cfg:  # normalize: CloudLM 'd_cloud_dim' bekler
        cfg["d_cloud_dim"] = cfg.pop("d_dim")
    tok = BPETokenizer.load(args.bpe)
    m = CloudLM(ck["vocab_size"], **cfg).to(device)
    m.load_state_dict(ck["model_state"])
    n = sum(p.numel() for p in m.parameters())
    print(f"  Model: {n:,} param | {cfg} | epoch {ck.get('epoch')}")

    eval_texts = load_txt(args.corpus, max_lines=args.eval_sent + 5000, min_len=15, max_len=300)[500:]
    eos = tok._tokenizer.token_to_id("<eos>") or 1

    # ---- A) Boyut ----
    print(f"\n{'='*66}\n  A) BOYUT\n{'='*66}")
    fp32 = param_bytes(m, int8=False) / 1e6
    mq = quantize_full(m)
    int8_safe = param_bytes(mq, int8=True) / 1e6  # weight-only (encoder FP32)
    full_int8 = n * 1 / 1e6  # tüm ağırlık 1 byte (analitik)
    print(f"  FP32            : {fp32:6.1f} MB")
    print(f"  INT8 (güvenli)  : {int8_safe:6.1f} MB  (encoder+norm FP32)")
    print(f"  INT8 (tam)      : {full_int8:6.1f} MB  (tüm ağırlık 1 byte)")

    # ---- B) Kalite ----
    print(f"\n{'='*66}\n  B) KUANTİZASYON KALİTESİ (PPL)\n{'='*66}")
    cpu = torch.device("cpu")
    ppl_fp32, _ = eval_ppl(m, tok, eval_texts, args.seq_len, device)
    mq_cpu = mq.to(cpu)
    ppl_int8, _ = eval_ppl(mq_cpu, tok, eval_texts, args.seq_len, cpu)
    print(f"  FP32 PPL = {ppl_fp32:.2f} | INT8 PPL = {ppl_int8:.2f} "
          f"(Δ {ppl_int8/ppl_fp32:.3f}x)")

    # ---- C+D) ONNX + onnxruntime ----
    onnx_res = {}
    if HAVE_ORT:
        print(f"\n{'='*66}\n  C+D) ONNX + onnxruntime (int8 GEMM)\n{'='*66}")
        # ONNX export (sadece word_logits; sabit seq_len)
        class Exporter(nn.Module):
            def __init__(s, model):
                super().__init__(); s.model = model
            def forward(s, tokens):
                mask = tokens != 0
                return s.model(tokens, mask)[0]
        exp = Exporter(m.to(cpu).eval())
        dummy = torch.zeros(1, args.seq_len, dtype=torch.long)
        onnx_path = os.path.join(ROOT, "quann_cloud", "checkpoints", "cloudlm.onnx")
        try:
            torch.onnx.export(exp, (dummy,), onnx_path,
                              input_names=["tokens"], output_names=["logits"],
                              dynamic_axes={"tokens": {1: "seq"}, "logits": {0: "batch"}},
                              opset_version=17)
            print(f"  ONNX export: {os.path.basename(onnx_path)}")
            for prov in ["CPUExecutionProvider", "CoreMLExecutionProvider"]:
                try:
                    so = ort.SessionOptions()
                    sess = ort.InferenceSession(onnx_path, so, providers=[prov])
                    ms = onnx_latency(sess, "tokens", (1, args.seq_len))
                    onnx_res[prov] = ms
                    print(f"  {prov:28s} {ms:6.1f} ms/adım  ({1000/ms:.0f} adım/sn)")
                except Exception as e:
                    print(f"  {prov:28s} hata: {str(e)[:60]}")
            # INT8 ONNX
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                qpath = os.path.join(ROOT, "quann_cloud", "checkpoints", "cloudlm_int8.onnx")
                quantize_dynamic(onnx_path, qpath, weight_type=QuantType.QInt8)
                qsize = os.path.getsize(qpath) / 1e6
                sess = ort.InferenceSession(qpath, providers=["CPUExecutionProvider"])
                ms = onnx_latency(sess, "tokens", (1, args.seq_len))
                onnx_res["CPUExecutionProvider_int8"] = ms
                onnx_res["int8_onnx_mb"] = qsize
                print(f"  ONNX INT8: {qsize:.1f} MB | {ms:.1f} ms/adım ({1000/ms:.0f} adım/sn)")
            except Exception as e:
                print(f"  ONNX INT8 hata: {str(e)[:80]}")
        except Exception as e:
            print(f"  ONNX export HATA: {type(e).__name__}: {str(e)[:100]}")
    else:
        print("\n  (onnxruntime yok — int8 GEMM hızı ölçülemedi)")

    # ---- E) Üretim ----
    print(f"\n{'='*66}\n  E) ÜRETİM KALİTESİ\n{'='*66}")
    m.eval()
    prompts = ["Türkiye'nin başkenti", "Bilim ve teknoloji", "Deniz kenarında", "Eğitim sistemi"]
    gen = {}
    for p in prompts:
        ids = torch.tensor([tok.encode(p)], device=device)
        out = tok.decode(m.generate(ids, max_new=args.gen_len, temperature=0.8, eos_id=eos)[0].tolist())
        gen[p] = out
        print(f"    {p:22s} → {out}")

    out = {"params": n, "config": cfg, "epoch": ck.get("epoch"),
           "size_mb": {"fp32": fp32, "int8_safe": int8_safe, "int8_full": full_int8},
           "ppl": {"fp32": ppl_fp32, "int8": ppl_int8, "ratio": ppl_int8 / ppl_fp32},
           "onnx": onnx_res, "generation": gen}
    out_path = os.path.join(ROOT, "quann_cloud", "logs", "package_cloudlm.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
