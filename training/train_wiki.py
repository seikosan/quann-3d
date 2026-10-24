"""
Quann-3D — Adım 2+3: Wikipedia Corpus + Büyük Model
=====================================================
- 200K Türkçe cümle (HuggingFace: NumanKaanKaratas/turkish-sentences)
- BPE tokenizer (20K vocab)
- Büyük model: d_model=256, d_cloud=512, 4 katman, 16D bulut

Kullanım:
  python3 train_wiki.py --step all
  python3 train_wiki.py --step tokenizer
  python3 train_wiki.py --step train
  python3 train_wiki.py --step test
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
import re
import json
import argparse
import numpy as np
from typing import List
from collections import Counter

# Ortak mimari (tek kaynak): model + tokenizer'lar cloudlm paketinden
sys.path.insert(0, ROOT)
from cloudlm import BPETokenizer, CloudLM, WordToND, CloudEncoderND  # noqa: F401 (re-export)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EXAMPLE_CHUNK_SIZE = 8192


def iter_shuffled_batches(all_tokens, seq_len, batch_size):
    """Yield shuffled batches without materializing every corpus window."""
    chunks = []
    chunk_start = None
    chunk_count = 0
    chunk_remaining = EXAMPLE_CHUNK_SIZE
    for sentence_idx, tokens in enumerate(all_tokens):
        n_examples = max(0, len(tokens) - seq_len)
        local_start = 0
        while local_start < n_examples:
            take = min(chunk_remaining, n_examples - local_start)
            if chunk_count == 0:
                chunk_start = (sentence_idx, local_start)
            chunk_count += take
            local_start += take
            chunk_remaining -= take
            if chunk_remaining == 0:
                chunks.append((*chunk_start, chunk_count))
                chunk_start = None
                chunk_count = 0
                chunk_remaining = EXAMPLE_CHUNK_SIZE

    if chunk_count:
        chunks.append((*chunk_start, chunk_count))

    # Random chunk order plus per-chunk shuffle approximates a global shuffle,
    # while keeping token-window objects bounded by the chunk size.
    random.shuffle(chunks)
    pending = []
    for sentence_idx, local_start, chunk_count in chunks:
        remaining = chunk_count
        examples = []
        while remaining:
            tokens = all_tokens[sentence_idx]
            n_examples = max(0, len(tokens) - seq_len)
            take = min(remaining, n_examples - local_start)
            if take:
                examples.extend(
                    (tokens[i:i+seq_len], tokens[i+seq_len])
                    for i in range(local_start, local_start + take)
                )
                remaining -= take
            sentence_idx += 1
            local_start = 0

        random.shuffle(examples)
        if pending:
            examples = pending + examples

        full_size = len(examples) // batch_size * batch_size
        for b in range(0, full_size, batch_size):
            yield examples[b:b+batch_size]
        pending = examples[full_size:]

    if pending:
        yield pending


# ─────────────────────────────────────────────
# CORPUS
# ─────────────────────────────────────────────

def load_txt(path, max_lines=None, min_len=20, max_len=300):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            t = line.strip()
            if min_len <= len(t) <= max_len:
                texts.append(t)
    return texts


# ─────────────────────────────────────────────
# ADIMLAR
# ─────────────────────────────────────────────

def step_tokenizer(args):
    print(f"\n{'='*60}")
    print(f"  ADIM 2A: BPE TOKENIZER (Wikipedia)")
    print(f"{'='*60}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    texts = load_txt(args.corpus, args.max_texts)
    print(f"  {len(texts)} cümle")
    
    tok = BPETokenizer(vocab_size=args.bpe_vocab)
    tok.train(texts)
    print(f"  Vocab: {tok.num_tokens} token")
    
    # Örnekler
    print(f"\n  Tokenizasyon:")
    for w in ["bahçede", "güzelleştirmek", "okullarında", "demokrasi"]:
        print(f"    {w:20s} → {tok.sample_tokens(w)}")
    
    tok_path = os.path.join(args.save_dir, "wiki_bpe.json")
    tok.save(tok_path)
    return tok


def step_train(args, tok):
    print(f"\n{'='*60}")
    print(f"  ADIM 2B+3: MODEL EĞİTİM (Wikipedia + Büyük Model)")
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
    
    model = CloudLM(
        tok.num_tokens,
        d_model=args.d_model,
        d_cloud=args.d_cloud,
        d_cloud_dim=args.d_dim,
        n_layers=args.layers,
        pool=args.pool,
        cloud_attn=args.cloud_attn,
    )
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} params")
    print(f"  d_model={args.d_model}, d_cloud={args.d_cloud}, D={args.d_dim}, layers={args.layers}")
    print(f"  Vocab={tok.num_tokens} | Device={device} | Epochs={args.epochs}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume: checkpoint epoch'undan devam et, cosine schedule'ı oradan hizala
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        else:
            for _ in range(start_epoch):
                scheduler.step()
        rng_state = ckpt.get("rng_state", {})
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        if torch.backends.mps.is_available() and "mps" in rng_state:
            torch.mps.set_rng_state(rng_state["mps"])
        print(f"  Resume: epoch {start_epoch} checkpoint'undan devam ({args.resume})")

    os.makedirs(args.save_dir, exist_ok=True)
    best_ppl = ckpt.get("best_ppl", float('inf')) if args.resume else float('inf')

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss, n_batches = 0, 0
        
        for batch in iter_shuffled_batches(all_tokens, args.seq_len, args.batch):
            inputs = torch.tensor([e[0] for e in batch], device=device)
            targets = torch.tensor([e[1] for e in batch], device=device)
            mask = (inputs != 0) & (inputs != eos_id)
            
            optimizer.zero_grad()
            logits, next_point, _ = model(inputs, mask)
            word_loss = F.cross_entropy(logits, targets)
            target_embedding = model.embedding(targets) * math.sqrt(model.embedding.embedding_dim)
            target_points = model.to_nd(model.norm(target_embedding.unsqueeze(1)))[:, 0]
            point_loss = F.mse_loss(next_point, target_points)
            loss = word_loss + 0.1 * point_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += word_loss.item()
            n_batches += 1
        
        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        ppl = math.exp(min(avg, 20))
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={avg:.4f} | ppl={ppl:.2f} | {elapsed:.0f}s", flush=True)
        
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                **({"cuda": torch.cuda.get_rng_state_all()} if torch.cuda.is_available() else {}),
                **({"mps": torch.mps.get_rng_state()} if torch.backends.mps.is_available() else {}),
            },
            "vocab_size": tok.num_tokens,
            "config": {"d_model": args.d_model, "d_cloud": args.d_cloud,
                       "d_cloud_dim": args.d_dim, "n_layers": args.layers,
                       "pool": args.pool, "cloud_attn": args.cloud_attn},
            "ppl": ppl,
            "best_ppl": best_ppl,
        }
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(args.save_dir, f"wiki_cloud_epoch{epoch+1}.pt"))
        
        if ppl < best_ppl:
            best_ppl = ppl
            checkpoint["best_ppl"] = best_ppl
            torch.save(checkpoint, os.path.join(args.save_dir, "wiki_cloud_best.pt"))
    
    total = time.time() - t0
    print(f"\n  ✅ {total:.0f}s ({total/60:.1f}dk) | Best PPL: {best_ppl:.2f}")
    tok.save(os.path.join(args.save_dir, "wiki_bpe.json"))
    return model, tok


def step_test(args, model=None, tok=None):
    print(f"\n{'='*60}")
    print(f"  ADIM 4: TEST")
    print(f"{'='*60}")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    if model is None:
        ckpt = torch.load(os.path.join(args.save_dir, "wiki_cloud_best.pt"),
                          map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = CloudLM(ckpt["vocab_size"], **cfg)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded: epoch {ckpt['epoch']}, PPL {ckpt['ppl']:.2f}")
    
    if tok is None:
        tok = BPETokenizer.load(os.path.join(args.save_dir, "wiki_bpe.json"))
    
    model.to(device)
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1
    
    # Üretim
    print(f"\n  🧪 ÜRETİM:")
    prompts = [
        "Türkiye'nin başkenti",
        "bilgisayar bilimlerinde",
        "sanat tarihi açısından",
        "ekonomik büyüme",
        "demokrasi ve insan",
    ]
    for p in prompts:
        tokens = torch.tensor([tok.encode(p)], device=device)
        result = model.generate(tokens, max_new=12, temperature=0.7, eos_id=eos_id)
        out = tok.decode(result[0].tolist())
        print(f"    {p:30s} → {out[:70]}")
    
    # Geometri
    print(f"\n  🔬 GEOMETRİ:")
    sents = [
        "Türkiye Avrupa ile Asya arasında bir köprüdür",
        "bilim teknolojide önemli rol oynar",
        "sanat toplumun aynası gibidir",
    ]
    for sent in sents:
        tokens = torch.tensor([tok.encode(sent)], device=device)
        mask = (tokens != 0) & (tokens != eos_id)
        pts = model.get_cloud(tokens, mask)
        enc = tok._tokenizer.encode(sent)
        print(f"\n    \"{sent[:50]}\"")
        print(f"    {len(enc.tokens)} token | {pts.shape[1]}D bulut")
        n = len(enc.tokens)
        if n > 1:
            all_d = [float(np.linalg.norm(pts[i]-pts[j])) for i in range(n) for j in range(n) if i!=j]
            print(f"    Min/Max dist: {min(all_d):.3f}/{max(all_d):.3f}")
    
    # Görsel
    sent = "Türkiye Avrupa ile Asya arasında bir köprüdür"
    tokens = torch.tensor([tok.encode(sent)], device=device)
    mask = (tokens != 0) & (tokens != eos_id)
    pts = model.get_cloud(tokens, mask)[:, :3]
    enc = tok._tokenizer.encode(sent)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(enc.tokens), 2)))
    for i, (t, p) in enumerate(zip(enc.tokens, pts)):
        ax.scatter(*p, s=150, color=colors[i%10], edgecolors='black', linewidth=0.5)
        ax.text(p[0], p[1], p[2], f" {t[:8]}", fontsize=8, color=colors[i%10])
    ax.set_title(f"Wikipedia Bulut: {sent[:50]!r}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    path = os.path.join(args.save_dir, "wiki_cloud_test.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📸 {path}")
    print(f"\n  ✅ TEST TAMAM")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all",
                       choices=["tokenizer", "train", "test", "all"])
    parser.add_argument("--corpus", type=str,
                        default=os.path.join(ROOT, "data", "turkish_sentences.txt"))
    parser.add_argument("--max-texts", type=int, default=100000)
    parser.add_argument("--bpe-vocab", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-cloud", type=int, default=512)
    parser.add_argument("--d-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--pool", type=str, default="max", choices=["max", "attn"])
    parser.add_argument("--cloud-attn", action="store_true",
                        help="Enable R2 cloud self-attention (pooling öncesi nokta-üstü attention)")
    parser.add_argument("--save-dir", type=str,
                        default=os.path.join(ROOT, "checkpoints", "wiki_checkpoints"))
    parser.add_argument("--resume", type=str, default=None,
                        help="Devam edilecek checkpoint yolu (örn. wiki_checkpoints_wiki1500k/wiki_cloud_best.pt)")
    args = parser.parse_args()

    print("=" * 60)
    print("  QUANN-3D — WIKIPEDIA + BÜYÜK MODEL")
    print("=" * 60)

    tok = None
    model = None

    if args.step in ("tokenizer", "all"):
        if args.resume:
            # resume: mevcut BPE'yi aynen kullan (yeniden tokenize ETME)
            tok = BPETokenizer.load(os.path.join(args.save_dir, "wiki_bpe.json"))
            print(f"  Resume: mevcut BPE yüklendi ({os.path.join(args.save_dir, 'wiki_bpe.json')})")
        else:
            tok = step_tokenizer(args)
    
    if args.step in ("train", "all"):
        if tok is None:
            tok = BPETokenizer.load(os.path.join(args.save_dir, "wiki_bpe.json"))
        model, tok = step_train(args, tok)
    
    if args.step in ("test", "all"):
        step_test(args, model, tok)


if __name__ == "__main__":
    main()
