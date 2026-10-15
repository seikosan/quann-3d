"""
Cümle Bulutu Dil Modeli — GERÇEK CORPUS (Bilkent)
===================================================
9,119 gerçek öğrenci yazısı üzerinde eğitim.
16D bulut, d_model=256, d_cloud=512, 4 katman.

Kullanım:
  python3 cloud_lm_real.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import os
import sys
import csv
import re
import numpy as np
from typing import List, Optional
from collections import Counter

import matplotlib
matplotlib.use("Agg")

# Ortak mimari (tek kaynak)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cloudlm import CloudLM, WordToND, CloudEncoderND, Tok  # noqa: F401
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ─────────────────────────────────────────────
# CORPUS YÜKLEME
# ─────────────────────────────────────────────

def load_bilkent(path="bilkent_texts.csv", max_texts=5000):
    """CSV'den metinleri yükle, cümlelere böl."""
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_texts:
                break
            text = row.get('text', '')
            if not text or len(text) < 50:
                continue
            # Cümlelere böl
            # Türkçe cümle sonu: . ! ? (satır sonu da olabilir)
            text = re.sub(r'\n+', ' ', text)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for s in sentences:
                s = s.strip()
                # 3-20 kelime arası cümleler
                words = s.split()
                if 3 <= len(words) <= 20:
                    # Sadece harf/sayı içeren kelimeler (temiz)
                    clean = [w for w in words if re.match(r'^[a-zA-ZçğıöşüÇĞİÖŞÜ\']+$', w)]
                    if len(clean) >= 3:
                        texts.append(' '.join(clean))
    
    return texts


def build_tokenizer(texts, max_vocab=8000):
    """Kelime tabanlı tokenizer (cloudlm.Tok)."""
    tok = Tok()
    tok.build(texts, max_vocab)
    return tok, tok.itos, tok.stoi

# ─────────────────────────────────────────────
# EĞİTİM
# ─────────────────────────────────────────────

def train(model, all_tokens, tok, epochs=15, batch_size=128, lr=2e-4,
          seq_len=6, device="auto", verbose=True):
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Uzunluk filtresi
    all_tokens = [t for t in all_tokens if len(t) >= seq_len + 1]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  📊 {len(all_tokens)} cümle | vocab={tok.vocab_size} | params={n_params:,} | {device}")
    
    model.train()
    for epoch in range(epochs):
        # Örnek havuzu
        examples = []
        for tokens in all_tokens:
            for i in range(0, len(tokens) - seq_len, 2):  # stride=2 (hız)
                if i + seq_len >= len(tokens): break
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
        if verbose and (epoch + 1) % 3 == 0:
            avg = epoch_loss / max(n_batches, 1)
            ppl = math.exp(min(avg, 20))
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg:.4f} | ppl={ppl:.2f} | {time.time()-t0:.1f}s")
    
    model.eval()
    return model


def evaluate_ppl(model, all_tokens, tok, seq_len=6, device="cpu", n_test=300):
    model.to(device)
    model.eval()
    testable = [t for t in all_tokens if len(t) >= seq_len + 1]
    random.seed(999)
    samples = random.sample(testable, min(n_test, len(testable)))
    
    total_loss, count = 0, 0
    with torch.no_grad():
        for tokens in samples:
            for i in range(0, len(tokens) - seq_len, 3):
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

def plot_cloud(model, text, tok, save_path, title=None, device="cpu"):
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
    
    # Yakın çiftler arası çizgi
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            max_d = np.max([np.linalg.norm(pts[a]-pts[b]) for a in range(len(pts)) for b in range(len(pts)) if a!=b])
            if d < max_d * 0.5:
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        [pts[i][2], pts[j][2]], 'gray', alpha=0.5, linewidth=1.5)
                mid = (pts[i] + pts[j]) / 2
                ax.text(mid[0], mid[1], mid[2], f"{d:.2f}", fontsize=8, color='darkred')
    
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title or f"{text[:50]!r}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📸 {save_path}")


# ─────────────────────────────────────────────
# ANA
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CÜMLE BULUTU — GERÇEK CORPUS (Bilkent)")
    print("=" * 60)
    
    # 1. Corpus yükle
    print("\n📝 Corpus yükleniyor...")
    texts = load_bilkent("bilkent_texts.csv", max_texts=2000)
    print(f"   {len(texts)} cümle (3-20 kelime)")
    
    # Örnekler
    print(f"\n   Örnek cümleler:")
    random.seed(42)
    for t in random.sample(texts, 5):
        print(f"     • {t[:80]}")
    
    # 2. Tokenizer
    print("\n🔤 Tokenizer...")
    tok, itos, stoi = build_tokenizer(texts, max_vocab=3000)
    print(f"   Vocab: {tok.vocab_size} kelime")
    
    # 3. Tokenize
    print("\n🔢 Tokenize...")
    all_tokens = [tok.encode(t) for t in texts]
    all_tokens = [t for t in all_tokens if len(t) >= 7]  # en az 6 kelime + eos
    print(f"   {len(all_tokens)} cümle (≥7 token)")
    
    # Token istatistikleri
    all_words = [w for t in all_tokens for w in t]
    avg_len = np.mean([len(t) for t in all_tokens])
    print(f"   Ort. cümle uzunluğu: {avg_len:.1f} token")
    
    # 4. Model
    print(f"\n{'─'*60}")
    print(f"  🌩️  16D BULUT MODELİ")
    print(f"{'─'*60}")
    
    model = CloudLM(
        vocab_size=tok.vocab_size,
        d_model=128,
        d_cloud=256,
        d_cloud_dim=16,
        n_layers=2
    )
    
    t0 = time.time()
    model = train(model, all_tokens, tok, epochs=10, batch_size=64,
                  lr=3e-4, seq_len=5, verbose=True)
    total_time = time.time() - t0
    print(f"  ⏱️  Toplam: {total_time:.1f}s")
    
    # 5. Test PPL
    print(f"\n📊 Değerlendirme...")
    ppl = evaluate_ppl(model, all_tokens, tok, seq_len=5, n_test=100)
    print(f"   Test Perplexity: {ppl:.2f}")
    
    # Rastgele model PPL
    random_model = CloudLM(vocab_size=tok.vocab_size, d_model=128, d_cloud=256, d_cloud_dim=16, n_layers=2)
    random_ppl = evaluate_ppl(random_model, all_tokens, tok, seq_len=5, n_test=100)
    print(f"   Rastgele (eğitimsiz) PPL: {random_ppl:.2f}")
    print(f"   İyileşme: {random_ppl/ppl:.1f}x")
    
    # 6. Geometri analizi
    print(f"\n\n{'='*60}")
    print(f"  🔬 GEOMETRİ ANALİZİ (gerçek cümleler)")
    print(f"{'='*60}")
    
    model.to("cpu")
    
    # Corpus'tan gerçek cümleler seç
    test_sentences = random.sample(texts, min(8, len(texts)))
    
    for sent in test_sentences[:5]:
        # Sadece ilk 8 kelime (görselleştirme için)
        words = sent.split()[:8]
        short_sent = ' '.join(words)
        tokens = torch.tensor([tok.encode(short_sent)])
        mask = (tokens != 0) & (tokens != 1)
        
        with torch.no_grad():
            _, _, points = model(tokens, mask)
        
        pts = points[0].numpy()
        n = len(words)
        
        # Mesafe analizi
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                dists.append((d, words[i], words[j]))
        dists.sort()
        
        centroid = pts.mean(axis=0)
        spread = float(np.linalg.norm(pts - centroid, axis=1).mean())
        
        print(f"\n  \"{short_sent[:60]}\"")
        print(f"   Yayılım: {spread:.3f} | Noktalar: {n}")
        print(f"   En yakın:  {dists[0][1]} ↔ {dists[0][2]} ({dists[0][0]:.3f})")
        if len(dists) > 1:
            print(f"   En uzak:   {dists[-1][1]} ↔ {dists[-1][2]} ({dists[-1][0]:.3f})")
    
    # 7. Görselleştirme
    print(f"\n\n{'='*60}")
    print(f"  📸 GÖRSELLEŞTİRME")
    print(f"{'='*60}")
    
    # Gerçek cümlelerden görsel
    vis_sentences = [s for s in test_sentences if 3 <= len(s.split()) <= 6][:4]
    for i, sent in enumerate(vis_sentences):
        plot_cloud(model, sent, tok, f"real_cloud_{i}.png",
                  title=f"Bilkent: {sent[:50]!r}")
    
    # 8. Üretim
    print(f"\n\n{'='*60}")
    print(f"  🧪 ÜRETİM (16D, temp=0.7)")
    print(f"{'='*60}")
    
    # Corpus'tan promptlar
    prompts = random.sample(texts, min(6, len(texts)))
    for p in prompts:
        words = p.split()[:4]  # ilk 4 kelime
        prompt = ' '.join(words)
        tokens = torch.tensor([tok.encode(prompt)])
        result = model.generate(tokens, max_new=6, temperature=0.7)
        out = tok.decode(result[0].tolist())
        print(f"   {prompt[:35]:35s} → {out[:50]}")
    
    # 9. Kelime co-kuurtulma analizi
    print(f"\n\n{'='*60}")
    print(f"  📐 KELİME BÖLGELERİ (16D uzayda küme analizi)")
    print(f"{'='*60}")
    
    # Frekanslı kelimelerin ortalama 3D konumu
    model.to("cpu")
    model.eval()
    
    word_positions = {}
    sample_texts = random.sample(texts, min(100, len(texts)))
    
    for sent in sample_texts:
        words = sent.split()[:8]
        tokens = torch.tensor([tok.encode(' '.join(words))])
        mask = (tokens != 0) & (tokens != 1)
        with torch.no_grad():
            _, _, points = model(tokens, mask)
        pts = points[0].cpu().numpy()[:, :3]
        for w, p in zip(words, pts):
            if w not in word_positions:
                word_positions[w] = []
            word_positions[w].append(p)
    
    # En sık görülen 12 kelimenin ortalama konumu
    word_counts = Counter(w for t in sample_texts for w in t.split()[:10])
    top_words = [w for w, _ in word_counts.most_common(12)]
    
    print(f"\n   12 sık kelimenin ortalama 3D konumu:")
    print(f"   {'Kelime':15s} {'X':>8s} {'Y':>8s} {'Z':>8s}  Bölge")
    print(f"   {'-'*15} {'-'*8} {'-'*8} {'-'*8}  {'-'*10}")
    
    for w in top_words:
        if w in word_positions:
            avg = np.mean(word_positions[w], axis=0)
            # Basit bölge sınıflandırması
            if avg[0] > 0.2: region = "doğu"
            elif avg[0] < -0.2: region = "batı"
            else: region = "merkez"
            if avg[1] > 0.2: region += "+kuzey"
            elif avg[1] < -0.2: region += "+güney"
            print(f"   {w:15s} {avg[0]:+8.3f} {avg[1]:+8.3f} {avg[2]:+8.3f}  {region}")
    
    # Bölge kümeleri
    all_avg = np.array([np.mean(word_positions[w], axis=0) for w in top_words if w in word_positions])
    if len(all_avg) > 2:
        spread = float(np.linalg.norm(all_avg - all_avg.mean(axis=0), axis=1).mean())
        print(f"\n   Sık kelimelerin ortalama yayılımı: {spread:.3f}")
        print(f"   (Düşük = kelimeler benzer bölgede, Yüksek = farklı bölgelerde)")
    
    print(f"\n\n{'='*60}")
    print(f"  ✅ TAMAM")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
