"""
QUANN-Cloud — Edge Verimlilik Benchmark
=========================================

CloudLM (nokta-bulut mimari) için edge cihaz uygunluk analizi:

  A) Boyut (footprint):  FP32 vs INT8 (dynamic quant) — gerçek 21M + orta config
  B) Gecikme (latency):  inference tokens/sn — FP32 vs INT8, CPU + MPS
  C) Kuantizasyon kalitesi: FP32 PPL vs INT8 PPL (taze eğitilmiş küçük model)

Nokta-bulut mimarinin edge avantajı: bulut durumu D boyutlu (16-32) —
klasik d_model gizli katmandan çok daha küçük; KV/ bellek ayak izi düşük.

Çalıştır:
    python3 quann_cloud/experiments/edge_benchmark.py
    python3 quann_cloud/experiments/edge_benchmark.py --quick
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
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def train_bpe(texts, vocab, path):
    tok = Tokenizer(models.BPE(vocab={}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, min_frequency=2,
                             special_tokens=["<pad>", "<eos>", "<unk>"])
    tok.train_from_iterator(texts, trainer=tr)
    tok.save(path)
    return tok


def model_size_mb(n_params):
    return n_params * 4 / 1e6  # FP32


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def count_int8_bytes(m):
    """INT8 ağırlık boyutu (byte) — int8 ağırlıklar + FP32 scale/bias."""
    b = 0
    for _, t in list(m.named_buffers()) + list(m.named_parameters()):
        b += t.numel() * (1 if t.dtype == torch.int8 else 4)
    return b


def eval_ppl(model, tok, texts, seq_len, device, max_n=500):
    model.eval()
    eos = tok._tokenizer.token_to_id("<eos>") or 1
    tot, n = 0.0, 0
    with torch.no_grad():
        for t in texts[:max_n]:
            ids = tok.encode(t)
            for i in range(len(ids) - seq_len):
                inp = torch.tensor([ids[i:i + seq_len]], device=device)
                tgt = torch.tensor([ids[i + seq_len]], device=device)
                mask = inp != 0
                logits, _, _ = model(inp, mask)
                tot += F.cross_entropy(logits, tgt).item()
                n += 1
    return math.exp(min(tot / max(n, 1), 20)), n


def measure_latency(model, tok, prompt_ids, device, n_tokens=20, warmup=3):
    """tokens/sn (arama tabanlı üretim)."""
    model.eval()
    ids = torch.tensor([prompt_ids], device=device)
    eos = tok._tokenizer.token_to_id("<eos>") or 1
    # warmup
    for _ in range(warmup):
        model.generate(ids, max_new=3, temperature=0.8, eos_id=eos)
    t0 = time.time()
    for _ in range(5):
        model.generate(ids, max_new=n_tokens, temperature=0.8, eos_id=eos)
    dt = (time.time() - t0) / 5
    return n_tokens / dt if dt > 0 else float('inf')


class QuantLinear(nn.Module):
    """Weight-only INT8: ağırlık int8 + per-channel scale; forward'de dequant.
    CPU ve MPS'te çalışır (torch.ao quantized engine bu build'de yok)."""
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


def quantize(m):
    """nn.Linear katmanlarını weight-only INT8'e çevir.
    NOT: encoder (transformer) içi Linear'lar DOKUNULMAZ — MultiheadAttention
    ve FFN, .weight'e doğrudan erişir; QuantLinear bunu bozar. Sadece
    to_nd / cloud / word_head / point_head kuantize edilir."""
    for name, mod in list(m.named_modules()):
        if isinstance(mod, nn.Linear) and not name.startswith("encoder"):
            parent = m
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], QuantLinear(mod))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "turkish_sentences.txt"))
    ap.add_argument("--bpe-vocab", type=int, default=4000)
    ap.add_argument("--bpe-path", default=os.path.join(ROOT, "quann_cloud", "data", "edge_bpe.json"))
    ap.add_argument("--train-sent", type=int, default=12000)
    ap.add_argument("--eval-sent", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--d-cloud", type=int, default=128)
    ap.add_argument("--d-dim", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.train_sent, args.eval_sent, args.epochs = 2000, 200, 2
        args.bpe_vocab, args.d_model, args.layers = 1500, 48, 1

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    mps = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    cpu = torch.device("cpu")

    print("=" * 66)
    print("  QUANN-Cloud — Edge Verimlilik Benchmark")
    print("=" * 66)

    # ---- Veri + BPE ----
    all_texts = load_txt(args.corpus, max_lines=args.train_sent + args.eval_sent + 3000)
    random.shuffle(all_texts)
    train_texts = all_texts[:args.train_sent]
    eval_texts = all_texts[args.train_sent:args.train_sent + args.eval_sent]
    os.makedirs(os.path.dirname(args.bpe_path), exist_ok=True)
    if os.path.exists(args.bpe_path):
        tok = BPETokenizer.load(args.bpe_path)
    else:
        t0 = time.time()
        tok = BPETokenizer(args.bpe_vocab)
        tok.train(all_texts)
        tok.save(args.bpe_path)
        print(f"  BPE: {tok.num_tokens} token ({time.time()-t0:.0f}s)")
    print(f"  Veri: {len(train_texts)} train + {len(eval_texts)} eval | vocab={tok.num_tokens}")

    # =====================================================================
    # A) BOYUT — gerçek 21M + orta config (eğitim gerektirmez)
    # =====================================================================
    print(f"\n{'='*66}\n  A) MODEL BOYUTU (FP32 vs INT8)\n{'='*66}")
    configs = {
        "CloudLM 21M (gerçek)": dict(vocab_size=20000, d_model=256, d_cloud=512, d_cloud_dim=32, n_layers=4),
        f"CloudLM orta (bu dene{''}y)": dict(vocab_size=tok.num_tokens, d_model=args.d_model, d_cloud=args.d_cloud, d_cloud_dim=args.d_dim, n_layers=args.layers),
    }
    size_rows = []
    for name, cfg in configs.items():
        m = CloudLM(**cfg)
        n = count_params(m)
        fp32 = model_size_mb(n)
        mq = quantize(m)
        int8 = count_int8_bytes(mq) / 1e6
        size_rows.append((name, n, fp32, int8))
        print(f"  {name:26s} {n:>12,} param | FP32 {fp32:6.1f} MB | INT8 {int8:6.1f} MB")
        del m, mq

    # =====================================================================
    # B) GECİKME — orta config, FP32 vs INT8, CPU + MPS
    # =====================================================================
    print(f"\n{'='*66}\n  B) GECİKME (tokens/sn, üretim)\n{'='*66}")
    m_med = CloudLM(vocab_size=tok.num_tokens, d_model=args.d_model,
                    d_cloud=args.d_cloud, d_cloud_dim=args.d_dim,
                    n_layers=args.layers)
    prompt = tok.encode("Türkiye'nin başkenti Ankara'dır")
    lat = {}
    for dev_name, dev in [("CPU", cpu), ("MPS", mps)]:
        mm = m_med.to(dev)
        tps = measure_latency(mm, tok, prompt, dev, n_tokens=20)
        lat[(dev_name, "fp32")] = tps
        print(f"  {dev_name:4s} FP32 : {tps:6.1f} tok/sn")
        del mm
    # INT8 (CPU — edge cihaz CPU benzeri)
    m_int8 = quantize(m_med.to(cpu))
    tps_i8 = measure_latency(m_int8, tok, prompt, cpu, n_tokens=20)
    lat[("CPU", "int8")] = tps_i8
    print(f"  CPU  INT8 : {tps_i8:6.1f} tok/sn")
    speedup = lat[("CPU", "fp32")] / tps_i8 if tps_i8 else 0
    print(f"  (INT8/FP32 CPU hız: {speedup:.2f}x)")
    del m_med, m_int8

    # =====================================================================
    # C) KUANTİZASYON KALİTESİ — taze eğitilmiş küçük model
    # =====================================================================
    print(f"\n{'='*66}\n  C) KUANTİZASYON KALİTESİ (FP32 vs INT8 PPL)\n{'='*66}")
    # CPU'da eğit (MPS bellek cap'ı OOM veriyor; CPU cap'sız)
    tr_dev = cpu
    m_small = CloudLM(vocab_size=tok.num_tokens, d_model=args.d_model,
                      d_cloud=args.d_cloud, d_cloud_dim=args.d_dim,
                      n_layers=args.layers).to(tr_dev)
    n_small = count_params(m_small)
    print(f"  Küçük model: {n_small:,} param, {args.epochs} epoch")

    # eğitim örnekleri
    examples = []
    for t in train_texts:
        ids = tok.encode(t)
        for i in range(len(ids) - args.seq_len):
            examples.append((ids[i:i + args.seq_len], ids[i + args.seq_len]))
    eos = tok._tokenizer.token_to_id("<eos>") or 1
    opt = torch.optim.AdamW(m_small.parameters(), lr=2e-4, weight_decay=0.01)
    m_small.train()
    for ep in range(1, args.epochs + 1):
        random.shuffle(examples)
        loss_sum, nb = 0.0, 0
        for b in range(0, len(examples), 32):
            be = examples[b:b + 32]
            if len(be) < 2:
                continue
            inp = torch.tensor([e[0] + [0] * (args.seq_len - len(e[0])) for e in be], device=tr_dev)
            tgt = torch.tensor([e[1] for e in be], device=tr_dev)
            opt.zero_grad()
            logits, _, _ = m_small(inp, inp != 0)
            loss = F.cross_entropy(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m_small.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(be); nb += len(be)
        avg = loss_sum / max(nb, 1)
        print(f"  [FP32] E{ep} loss={avg:.4f} ppl={math.exp(min(avg,20)):.2f}")

    # FP32 PPL (held-out)
    ppl_fp32, _ = eval_ppl(m_small, tok, eval_texts, args.seq_len, tr_dev)
    print(f"  [FP32]  held-out PPL = {ppl_fp32:.2f}")

    # INT8 (CPU)
    m_i8 = quantize(m_small)
    ppl_i8, _ = eval_ppl(m_i8, tok, eval_texts, args.seq_len, cpu)
    print(f"  [INT8]  held-out PPL = {ppl_i8:.2f}  (Δ {ppl_i8/ppl_fp32:.3f}x)")
    del m_small, m_i8

    # =====================================================================
    # ÖZET
    # =====================================================================
    print(f"\n{'='*66}\n  EDGE ÖZET\n{'='*66}")
    print(f"  {'Config':26s} {'Param':>12s} {'FP32 MB':>9s} {'INT8 MB':>9s}")
    for name, n, fp32, int8 in size_rows:
        print(f"  {name:26s} {n:>12,} {fp32:>9.1f} {int8:>9.1f}")
    print(f"\n  Gecikme: CPU FP32 {lat[('CPU','fp32')]:.1f} tok/sn | "
          f"CPU INT8 {lat[('CPU','int8')]:.1f} tok/sn | "
          f"MPS FP32 {lat.get(('MPS','fp32'),0):.1f} tok/sn")
    print(f"  Kuantizasyon kalitesi: FP32 PPL {ppl_fp32:.2f} → INT8 PPL {ppl_i8:.2f}")
    print(f"  Nokta-bulut: bulut durumu D={args.d_dim} boyut (d_model={args.d_model}'den bağımsız, kompakt)")

    out = {
        "sizes": [{"name": n, "params": p, "fp32_mb": f, "int8_mb": i}
                  for n, p, f, i in size_rows],
        "latency_tok_per_s": {f"{k[0]}_{k[1]}": v for k, v in lat.items()},
        "quant_quality": {"fp32_ppl": ppl_fp32, "int8_ppl": ppl_i8,
                          "ratio": ppl_i8 / ppl_fp32},
        "config": {"d_model": args.d_model, "d_cloud": args.d_cloud,
                   "d_dim": args.d_dim, "layers": args.layers,
                   "vocab": tok.num_tokens, "epochs": args.epochs},
    }
    out_path = os.path.join(ROOT, "quann_cloud", "logs", "edge_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
