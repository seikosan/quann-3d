"""
Quann-3D — Baseline: Normal Transformer LLM (CloudLM karşılaştırması)
======================================================================
Aynı corpus + aynı BPE + aynı seq_len/epoch ile eğitilen standart
Transformer dil modeli. CloudLM ile elma-elma karşılaştırma.

Mimariler:
  causal : GPT tarzı — causal attention, son pozisyon → head
           (d_model=384, CloudLM'in ~19.5M param ile eşleşir)
  bidir  : CloudLM'in encoder'ı aynen, ama cloud YOK
           (bidirectional attention + mean pool → head) — cloud ablation

Kullanım:
  python3 train_baseline.py --step all --arch causal
  python3 training/train_baseline.py --step train --arch causal \
      --corpus data/turkish_sentences.txt --max-texts 200000 \
      --bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json \
      --save-dir checkpoints/baseline_200k
  python3 training/train_baseline.py --step compare \
      --cloud-ckpt checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt \
      --base-ckpt checkpoints/baseline_200k/baseline_best.pt \
      --bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json
  python3 training/train_baseline.py --step compare \
      --cloud-ckpt checkpoints/wiki_checkpoints_wiki1500k/wiki_cloud_best.pt \
      --base-ckpt checkpoints/baseline_200k/baseline_best.pt \
      --cloud-bpe checkpoints/wiki_checkpoints_wiki1500k/wiki_bpe.json \
      --base-bpe checkpoints/wiki_checkpoints_200k/wiki_bpe.json
"""

import sys
import os

# Repozitör kökü (CWD'den bağımsız)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import json
import argparse
import numpy as np

from train_wiki import BPETokenizer, load_txt, CloudLM


# ─────────────────────────────────────────────
# BASELINE MODELLER
# ─────────────────────────────────────────────

class CausalLM(nn.Module):
    """Standart GPT tarzı: causal attention + son pozisyon → head."""

    def __init__(self, vocab_size, d_model=384, n_layers=4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=max(1, d_model // 16),
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, tokens, mask=None):
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = x + self.pos_enc[:, :tokens.size(1), :]
        T = tokens.size(1)
        causal = torch.triu(torch.ones(T, T, device=tokens.device, dtype=torch.bool), diagonal=1)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.encoder(x, mask=causal,
                         src_key_padding_mask=~mask if mask is not None else None)
        x = self.norm(x)
        return self.head(x)  # (B, T, V)

    def generate(self, tokens, max_new=20, temperature=0.8, eos_id=1, top_k=50):
        self.eval()
        generated = tokens.clone()
        with torch.no_grad():
            for _ in range(max_new):
                mask = (generated != 0) & (generated != eos_id)
                logits = self.forward(generated, mask)
                next_logits = logits[:, -1, :]
                if top_k and top_k < next_logits.size(-1):
                    idx = torch.argsort(next_logits, dim=-1, descending=True)[:, :top_k]
                    mk = torch.ones_like(next_logits, dtype=torch.bool)
                    mk.scatter_(1, idx, False)
                    next_logits = next_logits.masked_fill(mk, float('-inf'))
                probs = F.softmax(next_logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
                if (nxt == eos_id).all():
                    break
                generated = torch.cat([generated, nxt], dim=1)
        return generated


class BidirNoCloud(nn.Module):
    """CloudLM'in encoder'ı aynen, cloud yerine mean pool → head (ablation)."""

    def __init__(self, vocab_size, d_model=256, n_layers=4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=max(1, d_model // 16),
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, tokens, mask=None):
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = x + self.pos_enc[:, :tokens.size(1), :]
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.encoder(x, src_key_padding_mask=~mask if mask is not None else None)
        x = self.norm(x)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
            pooled = x.sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return self.head(pooled.unsqueeze(1)).squeeze(1)  # (B, V)

    def generate(self, tokens, max_new=20, temperature=0.8, eos_id=1, top_k=50):
        self.eval()
        generated = tokens.clone()
        with torch.no_grad():
            for _ in range(max_new):
                mask = (generated != 0) & (generated != eos_id)
                logits = self.forward(generated, mask)
                if top_k and top_k < logits.size(-1):
                    idx = torch.argsort(logits, dim=-1, descending=True)[:, :top_k]
                    mk = torch.ones_like(logits, dtype=torch.bool)
                    mk.scatter_(1, idx, False)
                    logits = logits.masked_fill(mk, float('-inf'))
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
                if (nxt == eos_id).all():
                    break
                generated = torch.cat([generated, nxt], dim=1)
        return generated


def build_model(arch, vocab_size, d_model, n_layers):
    if arch == "causal":
        return CausalLM(vocab_size, d_model=d_model, n_layers=n_layers)
    if arch == "bidir":
        return BidirNoCloud(vocab_size, d_model=d_model, n_layers=n_layers)
    raise ValueError(arch)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────
# ADIMLAR
# ─────────────────────────────────────────────

def step_train(args, tok):
    print(f"\n{'='*60}")
    print(f"  BASELINE EĞİTİM: {args.arch.upper()}")
    print(f"{'='*60}")

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    texts = load_txt(args.corpus, args.max_texts)
    print(f"  {len(texts)} cümle")

    all_tokens = [tok.encode(t) for t in texts]
    all_tokens = [t for t in all_tokens if len(t) >= args.seq_len + 1]
    print(f"  {len(all_tokens)} cümle (≥{args.seq_len+1} token)")
    print(f"  Ort. uzunluk: {np.mean([len(t) for t in all_tokens]):.1f}")

    eos_id = tok._tokenizer.token_to_id("<eos>") or 1

    model = build_model(args.arch, tok.num_tokens, args.d_model, args.layers)
    model.to(device)

    print(f"\n  Model: {count_params(model):,} params")
    print(f"  arch={args.arch} | d_model={args.d_model} | layers={args.layers}")
    print(f"  Vocab={tok.num_tokens} | Device={device} | Epochs={args.epochs}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume: checkpoint epoch'undan devam et, cosine schedule'ı oradan hizala
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"  Resume: epoch {start_epoch} checkpoint'undan devam ({args.resume})")

    os.makedirs(args.save_dir, exist_ok=True)
    best_ppl = float('inf')
    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        examples = []
        for tokens in all_tokens:
            for i in range(len(tokens) - args.seq_len):
                examples.append((tokens[i:i+args.seq_len], tokens[i+args.seq_len]))
        random.shuffle(examples)
        epoch_loss, n_batches = 0, 0

        for b in range(0, len(examples), args.batch):
            batch = examples[b:b+args.batch]
            if len(batch) < 2:
                continue
            inputs = torch.tensor([e[0] for e in batch], device=device)
            targets = torch.tensor([e[1] for e in batch], device=device)
            mask = (inputs != 0) & (inputs != eos_id)

            optimizer.zero_grad()
            if args.arch == "causal":
                # pencere → son pozisyondan TEK sonraki token (CloudLM ile aynı kurulum)
                logits = model(inputs, mask)
                loss = F.cross_entropy(logits[:, -1, :], targets)
            else:
                logits = model(inputs, mask)
                loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        ppl = math.exp(min(avg, 20))
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={avg:.4f} | ppl={ppl:.2f} | {elapsed:.0f}s (cum)", flush=True)

        ckpt = {
            "epoch": epoch + 1, "model_state": model.state_dict(),
            "vocab_size": tok.num_tokens,
            "config": {"arch": args.arch, "d_model": args.d_model, "n_layers": args.layers},
            "ppl": ppl,
        }
        if (epoch + 1) % 5 == 0:
            torch.save(ckpt, os.path.join(args.save_dir, f"baseline_{epoch+1}.pt"))
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(ckpt, os.path.join(args.save_dir, "baseline_best.pt"))

    total = time.time() - t0
    print(f"\n  ✅ {total:.0f}s ({total/60:.1f}dk) | Best PPL: {best_ppl:.2f}")
    return model, best_ppl


def evaluate_ppl(model, tok, texts, device, seq_len, arch, n=2000, seed=0):
    """Held-out benzeri: rastgele N cümle üzerinde PPL."""
    rng = random.Random(seed)
    sample = rng.sample(texts, min(n, len(texts)))
    model.eval()
    total_loss, total_tokens = 0.0, 0
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1
    with torch.no_grad():
        for t in sample:
            ids = tok.encode(t)
            if len(ids) < seq_len + 1:
                continue
            for i in range(0, len(ids) - seq_len, seq_len):
                if i + seq_len >= len(ids):
                    continue  # next token yok (kısa cümle kuyruğu)
                inp = torch.tensor([ids[i:i+seq_len]], device=device)
                mask = (inp != 0) & (inp != eos_id)
                tgt1 = torch.tensor([ids[i+seq_len]], device=device)
                if arch == "causal":
                    # Eğitimle aynı kurulum: pencere → son pozisyondan TEK token
                    logits = model(inp, mask)
                    total_loss += F.cross_entropy(logits[:, -1, :], tgt1, reduction="sum").item()
                    total_tokens += 1
                else:  # cloud ve bidir: pencereden sonraki TEK token
                    if arch == "cloud":
                        logits, _, _ = model(inp, mask)
                    else:
                        logits = model(inp, mask)
                    total_loss += F.cross_entropy(logits, tgt1, reduction="sum").item()
                    total_tokens += 1
    return math.exp(total_loss / max(total_tokens, 1))


def step_compare(args, tok):
    print(f"\n{'='*60}")
    print(f"  KARŞILAŞTIRMA: CloudLM vs Baseline")
    print(f"{'='*60}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # CloudLM
    cckpt = torch.load(args.cloud_ckpt, map_location="cpu", weights_only=False)
    cloud = CloudLM(cckpt["vocab_size"], **{k: v for k, v in cckpt["config"].items()})
    cloud.load_state_dict(cckpt["model_state"])
    cloud.to(device)

    # Baseline
    bckpt = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    bcfg = bckpt["config"]
    base = build_model(bcfg["arch"], bckpt["vocab_size"], bcfg["d_model"], bcfg["n_layers"])
    base.load_state_dict(bckpt["model_state"])
    base.to(device)

    print(f"\n  CloudLM   : {count_params(cloud):,} params (epoch {cckpt['epoch']}, train PPL {cckpt['ppl']:.2f})")
    print(f"  {bcfg['arch']:8s}: {count_params(base):,} params (epoch {bckpt['epoch']}, train PPL {bckpt['ppl']:.2f})")

    cloud_tok = BPETokenizer.load(args.cloud_bpe) if args.cloud_bpe else tok
    base_tok = BPETokenizer.load(args.base_bpe) if args.base_bpe else tok
    if cloud_tok is None or base_tok is None:
        raise SystemExit("--cloud-bpe/--base-bpe eksik tokenizer için --bpe gerekli")

    texts = load_txt(args.corpus, args.max_texts)
    print(f"\n  Ortak held-out PPL ({min(args.eval_n, len(texts))} cümle):")
    print(f"    Tokenizer: CloudLM={args.cloud_bpe or args.bpe or 'varsayılan'} | Baseline={args.base_bpe or args.bpe or 'varsayılan'}")
    c_ppl = evaluate_ppl(cloud, cloud_tok, texts, device, args.seq_len, "cloud", args.eval_n)
    b_ppl = evaluate_ppl(base, base_tok, texts, device, args.seq_len, bcfg["arch"], args.eval_n)
    print(f"    CloudLM  : {c_ppl:.2f}")
    print(f"    Baseline : {b_ppl:.2f}")

    cloud_eos_id = cloud_tok._tokenizer.token_to_id("<eos>") or 1
    base_eos_id = base_tok._tokenizer.token_to_id("<eos>") or 1
    prompts = [
        "Türkiye'nin başkenti",
        "bilgisayar bilimlerinde",
        "sanat tarihi açısından",
        "ekonomik büyüme",
        "demokrasi ve insan",
    ]
    print(f"\n  Üretim (aynı promptlar, temp=0.7):")
    for p in prompts:
        c_ids = torch.tensor([cloud_tok.encode(p)], device=device)
        b_ids = torch.tensor([base_tok.encode(p)], device=device)
        c_out = cloud_tok.decode(cloud.generate(c_ids, max_new=12, temperature=0.7, eos_id=cloud_eos_id)[0].tolist())
        b_out = base_tok.decode(base.generate(b_ids, max_new=12, temperature=0.7, eos_id=base_eos_id)[0].tolist())
        print(f"\n    Prompt: {p!r}")
        print(f"      CloudLM  : {c_out[:80]}")
        print(f"      Baseline : {b_out[:80]}")

    # Sonuç tablosu
    print(f"\n{'='*60}")
    print(f"  ÖZET")
    print(f"{'='*60}")
    print(f"  {'Model':12s} {'Params':>12s} {'Held-out PPL':>14s}")
    print(f"  {'CloudLM':12s} {count_params(cloud):>12,} {c_ppl:>14.2f}")
    print(f"  {bcfg['arch']:12s} {count_params(base):>12,} {b_ppl:>14.2f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=str, default="all", choices=["train", "compare", "all"])
    ap.add_argument("--arch", type=str, default="causal", choices=["causal", "bidir"])
    ap.add_argument("--corpus", type=str,
                    default=os.path.join(ROOT, "data", "turkish_sentences.txt"))
    ap.add_argument("--max-texts", type=int, default=100000)
    ap.add_argument("--bpe", type=str, default=None, help="BPE json yolu (varsayılan: save_dir/wiki_bpe.json)")
    ap.add_argument("--cloud-bpe", type=str, default=None, help="Compare: CloudLM checkpoint'inin BPE json yolu")
    ap.add_argument("--base-bpe", type=str, default=None, help="Compare: baseline checkpoint'inin BPE json yolu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=384, help="causal: 384 (param-eşleşen), bidir: 256")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--save-dir", type=str,
                    default=os.path.join(ROOT, "checkpoints", "baseline_200k"))
    ap.add_argument("--cloud-ckpt", type=str, default=None)
    ap.add_argument("--base-ckpt", type=str, default=None)
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--resume", type=str, default=None, help="Devam edilecek checkpoint yolu (örn. baseline_200k/baseline_40.pt)")
    args = ap.parse_args()

    # d_model varsayılanı arch'e göre
    if args.arch == "bidir" and args.d_model == 384:
        args.d_model = 256

    bpe_path = args.bpe or os.path.join(args.save_dir, "wiki_bpe.json")
    separate_bpes = args.step == "compare" and args.cloud_bpe and args.base_bpe and not args.bpe
    tok = None if separate_bpes else BPETokenizer.load(bpe_path)

    print("=" * 60)
    vocab_label = tok.num_tokens if tok is not None else "per-model BPE"
    print(f"  QUANN-3D — BASELINE ({args.arch}) | vocab={vocab_label}")
    print("=" * 60)

    if args.step in ("train", "all"):
        step_train(args, tok)

    if args.step in ("compare",):
        if not args.cloud_ckpt or not args.base_ckpt:
            raise SystemExit("--cloud-ckpt ve --base-ckpt gerekli")
        step_compare(args, tok)


if __name__ == "__main__":
    main()
