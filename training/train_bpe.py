"""
Quann-3D — Adım 1: BPE Tokenizer + Model
==========================================
Word-level yerine BPE (Byte Pair Encoding) subword tokenizer.
Türkçe agglütinasyonu için kritik: "bahçede" → "bahçe" + "de"

Kullanım:
  python3 train_bpe.py --step train_tokenizer   # Sadece BPE train
  python3 train_bpe.py --step train_model       # Model eğit
  python3 train_bpe.py --step test              # Test et
  python3 train_bpe.py --step all               # Hepsini yap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import csv
import re
import json
import argparse
import os
import sys

# Repozitör kökü (CWD'den bağımsız)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cloudlm import CloudLM, WordToND, CloudEncoderND, BPETokenizer
# BPE run'larının checkpoint adları için geriye dönük alias
CloudLMBPE = CloudLM  # noqa: F811

import numpy as np
from typing import List, Optional
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ─────────────────────────────────────────────
# CORPUS
# ─────────────────────────────────────────────

def load_corpus(path=None, max_texts=None):
    if path is None:
        path = os.path.join(ROOT, "data", "bilkent_texts.csv")
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_texts and i >= max_texts:
                break
            text = row.get('text', '')
            if not text or len(text) < 30:
                continue
            text = re.sub(r'\n+', ' ', text)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for s in sentences:
                s = s.strip()
                words = s.split()
                if 3 <= len(words) <= 25:
                    texts.append(s)
    return texts


# ─────────────────────────────────────────────
# ADIM 1: BPE TOKENIZER TRAIN
# ─────────────────────────────────────────────

def step_train_tokenizer(args):
    print(f"\n{'='*60}")
    print(f"  ADIM 1: BPE TOKENIZER")
    print(f"{'='*60}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Corpus
    print(f"\n  Corpus: {args.corpus}")
    texts = load_corpus(args.corpus, args.max_texts)
    print(f"  {len(texts)} cümle")
    
    # BPE train
    print(f"\n  BPE train (vocab={args.bpe_vocab})...")
    tok = BPETokenizer(vocab_size=args.bpe_vocab)
    tok.train(texts)
    print(f"  Vocab: {tok.num_tokens} token")
    
    # Örnekler
    print(f"\n  Tokenizasyon örnekleri:")
    test_words = ["bahçede", "bahçeler", "bahçem", "bahçesinde", 
                  "kedi", "kediler", "kedimi", "kedilerin",
                  "güzel", "güzellik", "güzelleştirmek"]
    for w in test_words:
        tokens = tok.sample_tokens(w)
        print(f"    {w:20s} → {tokens}")
    
    # Kaydet
    tok_path = os.path.join(args.save_dir, "bpe_tokenizer.json")
    tok.save(tok_path)
    
    # İstatistik
    all_tokens = [tok.encode(t) for t in texts[:1000]]
    avg_len = np.mean([len(t) for t in all_tokens])
    print(f"\n  Ort. token/cümle: {avg_len:.1f}")
    print(f"  (Word-level'de ortalama ~12 kelime/cümle idi)")
    
    return tok


# ─────────────────────────────────────────────
# ADIM 2: MODEL EĞİTİM
# ─────────────────────────────────────────────

def step_train_model(args, tok: BPETokenizer):
    print(f"\n{'='*60}")
    print(f"  ADIM 2: MODEL EĞİTİM (BPE)")
    print(f"{'='*60}")
    
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    
    # Corpus + tokenize
    texts = load_corpus(args.corpus, args.max_texts)
    print(f"  {len(texts)} cümle")
    
    all_tokens = [tok.encode(t) for t in texts]
    all_tokens = [t for t in all_tokens if len(t) >= args.seq_len + 1]
    print(f"  {len(all_tokens)} cümle (≥{args.seq_len+1} token)")
    print(f"  Ort. uzunluk: {np.mean([len(t) for t in all_tokens]):.1f} token")
    
    # EOS ID (BPE special tokens: <pad>=0, <eos>=1, <unk>=2)
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1
    
    # Model
    model = CloudLMBPE(
        vocab_size=tok.num_tokens,
        d_model=args.d_model,
        d_cloud=args.d_cloud,
        d_cloud_dim=args.d_dim,
        n_layers=args.layers
    )
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} params")
    print(f"  Config: d_model={args.d_model}, d_cloud={args.d_cloud}, D={args.d_dim}, layers={args.layers}")
    print(f"  Vocab: {tok.num_tokens} | Device: {device}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    
    # Eğitim
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_ppl = float('inf')
    history = []
    
    model.train()
    t0 = time.time()
    
    for epoch in range(args.epochs):
        examples = []
        for tokens in all_tokens:
            for i in range(len(tokens) - args.seq_len):
                examples.append((tokens[i:i+args.seq_len], tokens[i+args.seq_len]))
        
        random.shuffle(examples)
        epoch_loss, n_batches = 0, 0
        
        for b in range(0, len(examples), args.batch):
            batch = examples[b:b+args.batch]
            if len(batch) < 2: continue
            inputs = torch.tensor([e[0] for e in batch], device=device)
            targets = torch.tensor([e[1] for e in batch], device=device)
            mask = (inputs != 0) & (inputs != eos_id)
            
            optimizer.zero_grad()
            logits, _, _ = model(inputs, mask)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0
        history.append({"epoch": epoch+1, "loss": avg_loss, "ppl": ppl})
        
        print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.4f} | ppl={ppl:.2f} | {elapsed:.0f}s")
        
        # Checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt = os.path.join(args.save_dir, f"bpe_cloud_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch+1, "model_state": model.state_dict(),
                "vocab_size": tok.num_tokens,
                "config": {"d_model": args.d_model, "d_cloud": args.d_cloud,
                           "d_cloud_dim": args.d_dim, "n_layers": args.layers},
                "ppl": ppl,
            }, ckpt)
        
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({
                "epoch": epoch+1, "model_state": model.state_dict(),
                "vocab_size": tok.num_tokens,
                "config": {"d_model": args.d_model, "d_cloud": args.d_cloud,
                           "d_cloud_dim": args.d_dim, "n_layers": args.layers},
                "ppl": ppl,
            }, os.path.join(args.save_dir, "bpe_cloud_best.pt"))
    
    total_time = time.time() - t0
    print(f"\n  ✅ Eğitim: {total_time:.0f}s ({total_time/60:.1f}dk) | Best PPL: {best_ppl:.2f}")
    
    # Tokenizer + model kaydet
    tok.save(os.path.join(args.save_dir, "bpe_tokenizer.json"))
    
    return model, tok, history


# ─────────────────────────────────────────────
# ADIM 3: TEST
# ─────────────────────────────────────────────

def step_test(args, model=None, tok=None):
    print(f"\n{'='*60}")
    print(f"  ADIM 3: TEST")
    print(f"{'='*60}")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Yükle
    if model is None:
        ckpt_path = os.path.join(args.save_dir, "bpe_cloud_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.save_dir, "bpe_cloud_epoch5.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = CloudLMBPE(ckpt["vocab_size"], **cfg)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Checkpoint: {ckpt_path} (epoch {ckpt['epoch']}, ppl {ckpt['ppl']:.2f})")
    
    if tok is None:
        tok = BPETokenizer.load(os.path.join(args.save_dir, "bpe_tokenizer.json"))
    
    model.to(device)
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1
    
    # 1. Tokenizasyon testi
    print(f"\n  🔤 Tokenizasyon:")
    test_words = ["bahçede", "bahçeler", "bahçem", "güzellik", "kedi", "okulda"]
    for w in test_words:
        tokens = tok.sample_tokens(w)
        print(f"    {w:15s} → {tokens}")
    
    # 2. Üretim
    print(f"\n  🧪 ÜRETİM (temp=0.7):")
    prompts = [
        "kedi bahçede",
        "çocuk parkta",
        "bu cümleyi",
        "her duygunun",
        "bir gün",
        "okulda arkadaşım",
    ]
    for p in prompts:
        tokens = torch.tensor([tok.encode(p)], device=device)
        mask = (tokens != 0) & (tokens != eos_id)
        result = model.generate(tokens, max_new=10, temperature=0.7, eos_id=eos_id)
        out = tok.decode(result[0].tolist())
        print(f"    {p:25s} → {out}")
    
    # 3. Geometri
    print(f"\n  🔬 GEOMETRİ:")
    test_sents = [
        "kedi bahçede uyudu",
        "çocuk parkta top oynadı",
        "bu cümleyi yazarken düşündüm",
    ]
    for sent in test_sents:
        tokens = torch.tensor([tok.encode(sent)], device=device)
        mask = (tokens != 0) & (tokens != eos_id)
        pts = model.get_cloud(tokens, mask)
        
        # Tokenları decode et (kelime grupları)
        enc = tok._tokenizer.encode(sent)
        tokens_str = enc.tokens
        
        print(f"\n    \"{sent}\"")
        print(f"    Tokens ({len(tokens_str)}): {' '.join(tokens_str[:12])}{'...' if len(tokens_str)>12 else ''}")
        print(f"    Bulut: {pts.shape[1]}D, {len(tokens_str)} nokta")
        
        # Mesafe istatistikleri
        n = len(tokens_str)
        if n > 1:
            all_d = [float(np.linalg.norm(pts[i]-pts[j])) for i in range(n) for j in range(n) if i!=j]
            print(f"    Min/Max dist: {min(all_d):.3f}/{max(all_d):.3f}")
    
    # 4. Bulut görselleştirme
    print(f"\n  📸 GÖRSELLEŞTİRME:")
    sent = "kedi bahçede uyudu"
    tokens = torch.tensor([tok.encode(sent)], device=device)
    mask = (tokens != 0) & (tokens != eos_id)
    pts = model.get_cloud(tokens, mask)[:, :3]
    
    enc = tok._tokenizer.encode(sent)
    tokens_str = enc.tokens
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(tokens_str), 2)))
    
    for i, (t, p) in enumerate(zip(tokens_str, pts)):
        ax.scatter(*p, s=150, color=colors[i % 10], edgecolors='black', linewidth=0.5)
        ax.text(p[0], p[1], p[2], f" {t[:8]}", fontsize=8, color=colors[i % 10])
    
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(pts[i]-pts[j])
            max_d = max(np.linalg.norm(pts[a]-pts[b]) for a in range(len(pts)) for b in range(len(pts)) if a!=b)
            if d < max_d * 0.5:
                ax.plot([pts[i][0],pts[j][0]],[pts[i][1],pts[j][1]],[pts[i][2],pts[j][2]],
                        'gray', alpha=0.4, linewidth=1)
    
    ax.set_title(f"BPE Bulut: {sent!r}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    save_path = os.path.join(args.save_dir, "bpe_cloud_test.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    📸 {save_path}")
    
    print(f"\n  ✅ TEST TAMAM")


# ─────────────────────────────────────────────
# ANA
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all",
                       choices=["train_tokenizer", "train_model", "test", "all"])
    parser.add_argument("--corpus", type=str,
                        default=os.path.join(ROOT, "data", "bilkent_texts.csv"))
    parser.add_argument("--max-texts", type=int, default=5000)
    parser.add_argument("--bpe-vocab", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-cloud", type=int, default=256)
    parser.add_argument("--d-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--save-dir", type=str,
                        default=os.path.join(ROOT, "checkpoints", "bpe_checkpoints"))
    args = parser.parse_args()
    
    # Corpus path (eski workspace fallback'i kaldırıldı)
    if not os.path.exists(args.corpus):
        print(f"  ⚠ Corpus bulunamadı: {args.corpus}")
        print("    Bilkent dataset için README'deki curl komutunu çalıştırın veya")
        print("    --corpus ile başka bir dosya verin.")
    
    print("=" * 60)
    print("  QUANN-3D — BPE PIPELINE")
    print("=" * 60)
    
    tok = None
    model = None
    
    if args.step in ("train_tokenizer", "all"):
        tok = step_train_tokenizer(args)
    
    if args.step in ("train_model", "all"):
        if tok is None:
            tok = BPETokenizer.load(os.path.join(args.save_dir, "bpe_tokenizer.json"))
        model, tok, _ = step_train_model(args, tok)
    
    if args.step in ("test", "all"):
        if model is None:
            model = None  # test fonksiyonu kendisi yükler
        if tok is None:
            tok = None
        step_test(args, model, tok)


if __name__ == "__main__":
    main()
