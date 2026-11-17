"""
Quann-3D — GPU (MPS) Uzun Eğitim
==================================
Tam Bilkent corpus (65K cümle), 16D bulut, 30 epoch.
Checkpoint + değerlendirme + görselleştirme.

Kullanım:
  python3 train_gpu.py [--epochs 30] [--batch 128] [--lr 2e-4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import csv
import re
import argparse
import os
import sys

# Repozitör kökü (CWD'den bağımsız)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cloudlm import CloudLM, WordToND, CloudEncoderND, Tok  # noqa: F401

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
            if not text or len(text) < 50:
                continue
            text = re.sub(r'\n+', ' ', text)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for s in sentences:
                s = s.strip()
                words = s.split()
                if 3 <= len(words) <= 20:
                    clean = [w for w in words if re.match(r"^[a-zA-ZçğıöşüÇĞİÖŞÜ']+$", w)]
                    if len(clean) >= 3:
                        texts.append(' '.join(clean))
    return texts


# ─────────────────────────────────────────────
# EĞİTİM
# ─────────────────────────────────────────────

def train(model, all_tokens, tok, epochs, batch_size, lr, seq_len,
          device, save_dir=None, verbose=True):
    if save_dir is None:
        save_dir = os.path.join(ROOT, "checkpoints", "cloudlm")
    model.to(device)
    all_tokens = [t for t in all_tokens if len(t) >= seq_len + 1]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    os.makedirs(save_dir, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    
    if verbose:
        print(f"  Model: {n_params:,} params | {len(all_tokens)} cümle | vocab={tok.vocab_size}")
        print(f"  Config: d_model={model.embedding.embedding_dim}, d_cloud_dim={model.d_cloud_dim}, seq_len={seq_len}")
        print(f"  Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
        print()
    
    best_ppl = float('inf')
    history = []
    
    model.train()
    for epoch in range(epochs):
        # Örnek havuzu (stride=1 tam kapsama)
        examples = []
        for tokens in all_tokens:
            for i in range(len(tokens) - seq_len):
                examples.append((tokens[i:i+seq_len], tokens[i+seq_len]))
        
        random.shuffle(examples)
        epoch_loss, n_batches = 0, 0
        t0 = time.time()
        
        for b in range(0, len(examples), batch_size):
            batch = examples[b:b+batch_size]
            if len(batch) < 2: continue
            inputs = torch.tensor([e[0] for e in batch], device=device)
            targets = torch.tensor([e[1] for e in batch], device=device)
            mask = (inputs != 0) & (inputs != 1)
            
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
        history.append({"epoch": epoch+1, "loss": avg_loss, "ppl": ppl, "time": elapsed})
        
        if verbose:
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | ppl={ppl:.2f} | {elapsed:.1f}s")
        
        # Her 5 epoch checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt = os.path.join(save_dir, f"cloud_lm_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "vocab_size": tok.vocab_size,
                "config": {
                    "d_model": model.embedding.embedding_dim,
                    "d_cloud": model.cloud.fusion[-2].out_features,
                    "d_cloud_dim": model.d_cloud_dim,
                    "n_layers": len(model.encoder.layers),
                },
                "ppl": ppl,
            }, ckpt)
            if verbose:
                print(f"         💾 {ckpt}")
        
        # En iyi modeli takip et
        if ppl < best_ppl:
            best_ppl = ppl
            best_ckpt = os.path.join(save_dir, "cloud_lm_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "vocab_size": tok.vocab_size,
                "config": {
                    "d_model": model.embedding.embedding_dim,
                    "d_cloud": model.cloud.fusion[-2].out_features,
                    "d_cloud_dim": model.d_cloud_dim,
                    "n_layers": len(model.encoder.layers),
                },
                "ppl": ppl,
            }, best_ckpt)
    
    if verbose:
        print(f"\n  ✅ Eğitim tamam | Best PPL: {best_ppl:.2f}")
    
    return model, history


# ─────────────────────────────────────────────
# DEĞERLENDİRME
# ─────────────────────────────────────────────

def evaluate(model, all_tokens, tok, seq_len=6, device="cpu", n_test=200):
    model.to(device)
    model.eval()
    testable = [t for t in all_tokens if len(t) >= seq_len + 1]
    random.seed(999)
    samples = random.sample(testable, min(n_test, len(testable)))
    
    total_loss, count = 0, 0
    with torch.no_grad():
        for tokens in samples:
            for i in range(0, len(tokens) - seq_len, 2):
                if i + seq_len >= len(tokens): break
                inp = torch.tensor([tokens[i:i+seq_len]], device=device)
                tgt = torch.tensor([tokens[i+seq_len]], device=device)
                mask = (inp != 0) & (inp != 1)
                logits, _, _ = model(inp, mask)
                total_loss += F.cross_entropy(logits, tgt, reduction='sum').item()
                count += 1
    return math.exp(min(total_loss / max(count, 1), 20))


# ─────────────────────────────────────────────
# GÖRSELLEŞTİRME
# ─────────────────────────────────────────────

def plot_cloud(model, text, tok, save_path, device="cpu"):
    model.to(device)
    model.eval()
    tokens = torch.tensor([tok.encode(text)], device=device)
    mask = (tokens != 0) & (tokens != 1)
    with torch.no_grad():
        _, _, points = model(tokens, mask)
    
    pts = points[0].cpu().numpy()[:, :3]
    words = text.lower().split()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(words), 2)))
    
    for i, (w, p) in enumerate(zip(words, pts)):
        ax.scatter(*p, s=200, color=colors[i % 10], edgecolors='black', linewidth=0.5)
        ax.text(p[0], p[1], p[2], f"  {w[:12]}", fontsize=9, fontweight='bold', color=colors[i % 10])
    
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            max_d = max(np.linalg.norm(pts[a]-pts[b]) for a in range(len(pts)) for b in range(len(pts)) if a!=b)
            if d < max_d * 0.5:
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        [pts[i][2], pts[j][2]], 'gray', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"{text[:60]!r}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📸 {save_path}")


def plot_training_history(history, save_path="training_curve.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    ppls = [h["ppl"] for h in history]
    
    ax1.plot(epochs, losses, 'b-', linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, ppls, 'r-', linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 {save_path}")


# ─────────────────────────────────────────────
# ANA
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-cloud", type=int, default=512)
    parser.add_argument("--d-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-vocab", type=int, default=8000)
    parser.add_argument("--max-texts", type=int, default=None)
    parser.add_argument("--corpus", type=str,
                        default=os.path.join(ROOT, "data", "bilkent_texts.csv"))
    parser.add_argument("--save-dir", type=str,
                        default=os.path.join(ROOT, "checkpoints", "cloudlm"))
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("  QUANN-3D — GPU UZUN EĞİTİM")
    print(f"  Device: {device.upper()}")
    print("=" * 60)
    
    # 1. Corpus
    print(f"\n📝 Corpus: {args.corpus}")
    texts = load_corpus(args.corpus, args.max_texts)
    print(f"   {len(texts)} cümle")
    
    # 2. Tokenizer
    print(f"\n🔤 Tokenizer (max_vocab={args.max_vocab})...")
    tok = Tok()
    tok.build(texts, args.max_vocab)
    print(f"   Vocab: {tok.vocab_size}")
    tok.save(os.path.join(args.save_dir, "tokenizer.json"))
    
    # 3. Tokenize
    print(f"\n🔢 Tokenize...")
    all_tokens = [tok.encode(t) for t in texts]
    all_tokens = [t for t in all_tokens if len(t) >= args.seq_len + 1]
    print(f"   {len(all_tokens)} cümle (≥{args.seq_len+1} token)")
    print(f"   Ort. uzunluk: {np.mean([len(t) for t in all_tokens]):.1f}")
    
    # 4. Model
    print(f"\n{'─'*60}")
    print(f"  🌩️  MODEL")
    print(f"{'─'*60}")
    
    model = CloudLM(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        d_cloud=args.d_cloud,
        d_cloud_dim=args.d_dim,
        n_layers=args.layers
    )
    
    # 5. Eğitim
    t0 = time.time()
    model, history = train(
        model, all_tokens, tok,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seq_len=args.seq_len,
        device=device,
        save_dir=args.save_dir,
        verbose=True
    )
    total_time = time.time() - t0
    print(f"\n  ⏱️  Toplam: {total_time:.0f}s ({total_time/60:.1f}dk)")
    
    # 6. Final değerlendirme
    print(f"\n{'─'*60}")
    print(f"  📊 FİNAL DEĞERLENDİRME")
    print(f"{'─'*60}")
    
    ppl = evaluate(model, all_tokens, tok, seq_len=args.seq_len, device=device, n_test=300)
    print(f"   Test PPL: {ppl:.2f}")
    
    # Baseline
    random_model = CloudLM(vocab_size=tok.vocab_size, d_model=args.d_model,
                           d_cloud=args.d_cloud, d_cloud_dim=args.d_dim, n_layers=args.layers)
    rand_ppl = evaluate(random_model, all_tokens, tok, seq_len=args.seq_len, device=device, n_test=300)
    print(f"   Rastgele PPL: {rand_ppl:.2f}")
    print(f"   İyileşme: {rand_ppl/ppl:.1f}x")
    del random_model
    
    # 7. Görselleştirme
    print(f"\n{'─'*60}")
    print(f"  📸 GÖRSELLEŞTİRME")
    print(f"{'─'*60}")
    
    plot_training_history(history, os.path.join(args.save_dir, "training_curve.png"))
    
    # Gerçek cümleler
    random.seed(77)
    test_sents = random.sample(texts, min(5, len(texts)))
    for i, sent in enumerate(test_sents[:3]):
        short = ' '.join(sent.split()[:6])
        plot_cloud(model, short, tok, os.path.join(args.save_dir, f"cloud_{i}.png"), device=device)
    
    # 8. Üretim
    print(f"\n{'─'*60}")
    print(f"  🧪 ÜRETİM (temp=0.7)")
    print(f"{'─'*60}")
    
    model.to(device)
    prompts = random.sample(texts, min(8, len(texts)))
    for p in prompts:
        words = p.split()[:4]
        prompt = ' '.join(words)
        tokens = torch.tensor([tok.encode(prompt)], device=device)
        result = model.generate(tokens, max_new=8, temperature=0.7)
        out = tok.decode(result[0].tolist())
        print(f"   {prompt[:35]:35s} → {out[:60]}")
    
    # 9. Kelime bölgeleri
    print(f"\n{'─'*60}")
    print(f"  📐 KELİME BÖLGELERİ")
    print(f"{'─'*60}")
    
    model.to("cpu")
    word_positions = {}
    sample = random.sample(texts, min(150, len(texts)))
    
    for sent in sample:
        words = sent.split()[:8]
        tokens = torch.tensor([tok.encode(' '.join(words))])
        mask = (tokens != 0) & (tokens != 1)
        with torch.no_grad():
            _, _, points = model(tokens, mask)
        pts = points[0].numpy()[:, :3]
        for w, p in zip(words, pts):
            if w not in word_positions:
                word_positions[w] = []
            word_positions[w].append(p)
    
    word_counts = Counter(w for t in sample for w in t.split()[:8])
    top_words = [w for w, _ in word_counts.most_common(15)]
    
    print(f"\n   {'Kelime':15s} {'X':>8s} {'Y':>8s} {'Z':>8s}  Bölge")
    print(f"   {'-'*15} {'-'*8} {'-'*8} {'-'*8}  {'-'*12}")
    
    for w in top_words:
        if w in word_positions and len(word_positions[w]) > 2:
            avg = np.mean(word_positions[w], axis=0)
            region = ""
            if avg[0] > 0.2: region += "doğu"
            elif avg[0] < -0.2: region += "batı"
            else: region += "merkez"
            if avg[1] > 0.2: region += "+K"
            elif avg[1] < -0.2: region += "+G"
            print(f"   {w:15s} {avg[0]:+8.3f} {avg[1]:+8.3f} {avg[2]:+8.3f}  {region}")
    
    # 10. Son model kaydet
    final_ckpt = os.path.join(args.save_dir, "cloud_lm_final.pt")
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": tok.vocab_size,
        "config": {
            "d_model": args.d_model,
            "d_cloud": args.d_cloud,
            "d_cloud_dim": args.d_dim,
            "n_layers": args.layers,
        },
        "ppl": ppl,
    }, final_ckpt)
    print(f"\n  💾 Final: {final_ckpt}")
    
    print(f"\n{'='*60}")
    print(f"  ✅ TAMAM — {total_time/60:.1f}dk")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
