"""
QUANN-Cloud — Gerçek Veri Deneyi (Backprop'suz vs Backprop)
============================================================

Aynı (küçük) CloudLM mimarisi, aynı Türkçe verisi, aynı BPE:
  - CloudLM  (Backprop, AdamW)  -> PPL_bp
  - QCloud   (QUANN, backprop'suz) -> PPL_quann
  - Random   (eğitilmemiş)       -> PPL_random  (referans taban)

Amaç: QUANN'ın aralık-daraltma mekanizmasının GERÇEK Türkçe verisinde
backprop'a karşı ne kadar rekabet edebildiğini dürüstçe ölçmek.

Çalıştır:
    python3 quann_cloud/experiments/real_data_test.py
    python3 quann_cloud/experiments/real_data_test.py --quick
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
sys.path.insert(0, os.path.join(ROOT, "quann_cloud", "src"))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from qcloud_quann import QUANN, QUANNConfig
from cloudlm import CloudLM


# ----------------------------------------------------------------------
# SLIM CloudLM (cloud hidden'i yapılandırılabilir -> QUANN için uygulanabilir)
# ----------------------------------------------------------------------
class SlimCloudLM(CloudLM):
    """
    Slim CloudLM (111K param) — QUANN/Backprop karşılaştırması için.
    Kanonik cloudlm mimarisi; küçük boyutlar:
    d_model=32, d_cloud=32, D=8, 1 katman, 64 cloud hidden, nhead=d_model//8.
    """

    def __init__(self, vocab_size, d_model=32, d_cloud=32, d_cloud_dim=8,
                 n_layers=1, cloud_hidden=64, max_pos=128):
        super().__init__(vocab_size, d_model, d_cloud, d_cloud_dim,
                         n_layers=n_layers, nhead=max(1, d_model // 8),
                         pos_len=max_pos, hidden=cloud_hidden, point_hidden=128)

# ----------------------------------------------------------------------
# BPE (küçük)
# ----------------------------------------------------------------------
class SmallBPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self._tok = None

    def train(self, texts):
        tok = Tokenizer(models.BPE(vocab={}, merges=[]))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size, min_frequency=2,
            special_tokens=["<pad>", "<eos>", "<unk>"])
        tok.train_from_iterator(texts, trainer=trainer)
        self._tok = tok
        return self

    @property
    def num_tokens(self):
        return len(self._tok.get_vocab())

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=True)

    def save(self, path):
        self._tok.save(path)

    @classmethod
    def load(cls, path):
        c = cls(0)
        c._tok = Tokenizer.from_file(path)
        c.vocab_size = len(c._tok.get_vocab())
        return c


def load_txt(path, max_lines=None, min_len=15, max_len=300):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            t = line.strip()
            if min_len <= len(t) <= max_len:
                texts.append(t)
    return texts


def build_examples(tok, texts, seq_len, device, max_examples=None):
    """(input_tokens, next_token) örnekleri üret (train_wiki ile aynı mantık)."""
    examples = []
    for t in texts:
        ids = tok.encode(t)
        for i in range(len(ids) - seq_len):
            examples.append((ids[i:i + seq_len], ids[i + seq_len]))
            if max_examples and len(examples) >= max_examples:
                return examples, None
    return examples, None


def examples_to_tensors(examples, seq_len, device):
    inputs = torch.tensor(
        [e[0] + [0] * (seq_len - len(e[0])) for e in examples], device=device)
    targets = torch.tensor([e[1] for e in examples], device=device)
    return inputs, targets


def eval_ppl(model, eval_examples, seq_len, device, batch=32, max_n=2000):
    model.eval()
    ex = eval_examples[:max_n]
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(ex), batch):
            batch_ex = ex[b:b + batch]
            inp = torch.tensor(
                [e[0] + [0] * (seq_len - len(e[0])) for e in batch_ex], device=device)
            tgt = torch.tensor([e[1] for e in batch_ex], device=device)
            mask = (inp != 0)
            logits, _, _ = model(inp, mask)
            total_loss += F.cross_entropy(logits, tgt).item() * len(batch_ex)
            total += len(batch_ex)
    return math.exp(min(total_loss / max(total, 1), 20))


def train_backprop(model, train_examples, seq_len, device, epochs, lr=2e-4, batch=32,
                   eval_examples=None, verbose=True):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        random.shuffle(train_examples)
        loss_sum, nb = 0.0, 0
        for b in range(0, len(train_examples), batch):
            be = train_examples[b:b + batch]
            if len(be) < 2:
                continue
            inp = torch.tensor(
                [e[0] + [0] * (seq_len - len(e[0])) for e in be], device=device)
            tgt = torch.tensor([e[1] for e in be], device=device)
            mask = (inp != 0)
            opt.zero_grad()
            logits, _, _ = model(inp, mask)
            loss = F.cross_entropy(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(be)
            nb += len(be)
        avg = loss_sum / max(nb, 1)
        ppl = math.exp(min(avg, 20))
        line = f"  [BP] E{ep:3d} | loss={avg:.4f} ppl={ppl:.2f} | {time.time()-t0:.0f}s"
        if eval_examples:
            vp = eval_ppl(model, eval_examples, seq_len, device, max_n=800)
            line += f" | eval_ppl={vp:.2f}"
        if verbose:
            print(line, flush=True)
    return model


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "turkish_sentences.txt"))
    ap.add_argument("--bpe-vocab", type=int, default=1024)
    ap.add_argument("--bpe-path", default=os.path.join(ROOT, "quann_cloud", "data", "small_bpe.json"))
    ap.add_argument("--train-sent", type=int, default=6000)
    ap.add_argument("--eval-sent", type=int, default=400)
    ap.add_argument("--max-examples", type=int, default=4000)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--d-cloud", type=int, default=32)
    ap.add_argument("--d-dim", type=int, default=8)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--cloud-hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--quann-k", type=int, default=80)
    ap.add_argument("--quann-batch", type=int, default=8,
                    help="ödül sinyali için mini-batch boyutu (gürültü azaltma)")
    ap.add_argument("--quann-sigma", type=float, default=0.25)
    ap.add_argument("--quann-device", default="cpu",
                    help="QUANN forward cihazı (küçük modelde CPU, MPS'ten ~25x hızlı)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="çok küçük ayarlarla hızlı duman testi")
    args = ap.parse_args()

    if args.quick:
        args.train_sent, args.eval_sent = 800, 200
        args.max_examples, args.epochs = 800, 2
        args.quann_k, args.bpe_vocab = 40, 512
        args.quann_device = "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device or
                          ("mps" if torch.backends.mps.is_available() else "cpu"))

    print("=" * 64)
    print("  QUANN-Cloud — Gerçek Veri Deneyi (Backprop'suz vs Backprop)")
    print("=" * 64)
    print(f"  device={device}  seed={args.seed}")

    # ---- Veri ----
    all_texts = load_txt(args.corpus, max_lines=args.train_sent + args.eval_sent + 5000)
    random.shuffle(all_texts)
    train_texts = all_texts[:args.train_sent]
    eval_texts = all_texts[args.train_sent:args.train_sent + args.eval_sent]
    print(f"\n  Veri: {len(train_texts)} train + {len(eval_texts)} eval cümle")

    # ---- BPE ----
    os.makedirs(os.path.dirname(args.bpe_path), exist_ok=True)
    bpe_path = args.bpe_path
    if os.path.exists(bpe_path):
        print(f"  BPE yükleniyor: {bpe_path}")
        tok = SmallBPE.load(bpe_path)
    else:
        print(f"  BPE eğitiliyor (vocab={args.bpe_vocab}) ...")
        t0 = time.time()
        tok = SmallBPE(args.bpe_vocab).train(all_texts)
        tok.save(bpe_path)
        print(f"  BPE hazır: {tok.num_tokens} token ({time.time()-t0:.0f}s)")
    print(f"  Vocab: {tok.num_tokens} | eos={tok._tok.token_to_id('<eos>')}")

    # ---- Örnekler ----
    train_examples, _ = build_examples(tok, train_texts, args.seq_len, device,
                                       max_examples=args.max_examples)
    eval_examples, _ = build_examples(tok, eval_texts, args.seq_len, device,
                                      max_examples=2000)
    print(f"  Örnekler: {len(train_examples)} train + {len(eval_examples)} eval")

    q_device = torch.device(args.quann_device)

    def make_model(dev=None):
        return SlimCloudLM(tok.num_tokens, args.d_model, args.d_cloud,
                           args.d_dim, args.layers, args.cloud_hidden).to(dev or device)

    # ---- Taban: eğitilmemiş (random init) PPL ----
    m0 = make_model()
    ppl_random = eval_ppl(m0, eval_examples, args.seq_len, device)
    print(f"\n  [Random init] eval PPL = {ppl_random:.1f} "
          f"(vocab={tok.num_tokens} => taban ~{tok.num_tokens})")
    del m0
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ---- 1) BACKPROP ----
    print(f"\n{'='*64}\n  1) CloudLM — BACKPROP (AdamW, {args.epochs} ep)\n{'='*64}")
    m_bp = make_model()
    n = sum(p.numel() for p in m_bp.parameters())
    print(f"  Model: {n:,} param | d_model={args.d_model} D={args.d_dim} "
          f"layers={args.layers} d_cloud={args.d_cloud}")
    train_backprop(m_bp, train_examples, args.seq_len, device, args.epochs,
                   eval_examples=eval_examples)
    ppl_bp = eval_ppl(m_bp, eval_examples, args.seq_len, device)
    print(f"  [Backprop]  held-out PPL = {ppl_bp:.2f}")

    # ---- 2) QUANN (backprop'suz) ----
    print(f"\n{'='*64}\n  2) QCloud — QUANN (backprop'suz, {args.epochs} ep, device={q_device})\n{'='*64}")
    m_q = make_model(q_device)
    cfg = QUANNConfig(k_weights=args.quann_k, minibatch=args.quann_batch,
                      sigma0=args.quann_sigma, seed=args.seed)
    quann = QUANN(m_q, q_device, cfg)
    qinfo = quann.info()
    print(f"  QUANN durumu: {qinfo['total_params']:,} ağırlık, "
          f"{qinfo['n_param_tensors']} tensor, σ0={cfg.sigma0}")

    inputs_t, targets_t = examples_to_tensors(train_examples, args.seq_len, q_device)
    eval_inputs, eval_targets = examples_to_tensors(eval_examples, args.seq_len, q_device)

    def q_eval():
        return eval_ppl(m_q, eval_examples, args.seq_len, q_device, max_n=800)

    quann.fit(list(zip(inputs_t, targets_t)), epochs=args.epochs,
              eval_fn=q_eval, verbose=True)
    ppl_q = eval_ppl(m_q, eval_examples, args.seq_len, q_device)
    print(f"  [QUANN]     held-out PPL = {ppl_q:.2f}")
    print(f"  QUANN final: {quann.info()}")

    # ---- Üretim örnekleri ----
    print(f"\n{'='*64}\n  3) Metin Üretimi (aynı prompt)\n{'='*64}")
    prompts = ["Türkiye'nin başkenti", "bilgisayar bilimlerinde",
               "deniz kenarında yürüdük", "ekonomi büyüyor"]
    eos = tok._tok.token_to_id("<eos>") or 1
    for p in prompts:
        ids = torch.tensor([tok.encode(p)], device=device)
        out_bp = tok.decode(m_bp.generate(ids, max_new=10, temperature=0.7, eos_id=eos)[0].tolist())
        ids_q = torch.tensor([tok.encode(p)], device=q_device)
        out_q = tok.decode(m_q.generate(ids_q, max_new=10, temperature=0.7, eos_id=eos)[0].tolist())
        print(f"\n  prompt: \"{p}\"")
        print(f"    BP    → {out_bp}")
        print(f"    QUANN → {out_q}")

    # ---- ÖZET ----
    print(f"\n{'='*64}\n  SONUÇ ÖZETİ\n{'='*64}")
    rows = [
        ("Random (eğitilmemiş)", ppl_random),
        ("Backprop (AdamW)", ppl_bp),
        ("QUANN (backprop'suz)", ppl_q),
    ]
    print(f"  {'Model':28s} {'Held-out PPL':>14s}")
    for name, p in rows:
        print(f"  {name:28s} {p:14.2f}")
    print(f"\n  Vocab={tok.num_tokens} | corpus={len(train_texts)} cümle | "
          f"{n:,} param | {args.epochs} epoch")
    print(f"  QUANN/Backprop PPL oranı: {ppl_q/ppl_bp:.3f} "
          f"(1.0 = eşit, <1 = QUANN daha iyi)")

    # JSON çıktı
    out = {
        "device": str(device), "seed": args.seed,
        "vocab": tok.num_tokens, "params": n,
        "train_sentences": len(train_texts), "eval_sentences": len(eval_texts),
        "train_examples": len(train_examples), "eval_examples": len(eval_examples),
        "epochs": args.epochs, "seq_len": args.seq_len,
        "config": {"d_model": args.d_model, "d_cloud": args.d_cloud,
                   "d_dim": args.d_dim, "layers": args.layers,
                   "cloud_hidden": args.cloud_hidden},
        "ppl_random": ppl_random, "ppl_backprop": ppl_bp, "ppl_quann": ppl_q,
        "quann_ratio": ppl_q / ppl_bp,
        "quann_info": quann.info(),
    }
    out_path = os.path.join(ROOT, "quann_cloud", "logs", "real_data_test.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
