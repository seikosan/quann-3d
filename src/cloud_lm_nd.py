"""
Cümle Bulutu Dil Modeli — ND Genelleme + Karşılaştırma
=======================================================
3D vs 7D vs 14D bulut boyutlarını karşılaştırır.
Daha büyük corpus, daha uzun cümleler.

Kullanım:
  python3 cloud_lm_nd.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import sys
from typing import List, Tuple, Optional
from collections import Counter

# Ortak mimari (tek kaynak)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────
# MODEL (ND destekli — cloudlm paketinden, nd boyutları)
# ─────────────────────────────────────────────

from cloudlm import CloudLM as _CloudLM, WordToND, CloudEncoderND, Tok  # noqa: F401


class CloudLMND(_CloudLM):
    """
    ND Cümle Bulutu Dil Modeli (kompakt varyant:
    d_model//2 projeksiyon, 128 cloud hidden, 64 point hidden, 128 pos).
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 d_cloud: int = 256, d_cloud_dim: int = 3):
        super().__init__(vocab_size, d_model, d_cloud, d_cloud_dim,
                         n_layers=2, pos_len=128,
                         hidden=128, proj_hidden=d_model // 2, point_hidden=64)
# ─────────────────────────────────────────────
# BÜYÜK CORPUS (Türkçe, çeşitli cümle kalıpları)
# ─────────────────────────────────────────────

def generate_corpus(n: int = 5000) -> List[str]:
    """
    Şablon tabanlı Türkçe cümle üretici.
    Çeşitli yapılar: özne+yüklem, özne+yüklem+nesne, özne+yüklem+yer, vb.
    """
    ozne = ["kedi", "köpek", "çocuk", "kadın", "adam", "kız", "erkek", 
            "öğrenci", "öğretmen", "doktor", "mühendis", "sanatçı",
            "kuş", "balık", "at", "keçi", "tavuk", "inek", "fare", "yılan"]
    
    fiil = ["koştu", "uyudu", "baktı", "içti", "yedi", "okudu", "yazdı",
            "söyledi", "duydu", "gördü", "aldı", "verdi", "gitti", "geldi",
            "oyun oynadı", "şarkı söyledi", "dans etti", "yemek yaptı",
            "kitap yazdı", "resim çizdi", "müzik dinledi", "fotoğraf çekti"]
    
    nesne = ["topu", "kitabı", "gazeteyi", "mektubu", "haritayı", "listeyi",
             "pencereyi", "kapıyı", "masayı", "sandalyeyi", "arabayı",
             "domatesi", "ekmeği", "sütü", "çayı", "kahveyi", "müziği",
             "şarkıyı", "hikayeyi", "mektubu"]
    
    yer = ["bahçede", "pazara", "okula", "evde", "sokakta", "plajda",
           "dağda", "ormanda", "pazarda", "müze", "sinemaya", "hastaneye",
           "kütüphanede", "restoranda", "parkta", "tarlada", "denizde",
           "pencereden", "kapıdan", "merdivenden", "köprüden"]
    
    zaman = ["sabah", "akşam", "gece", "öğlen", "hafta sonu", "dün",
             "bugün", "yarın", "yazın", "kışın", "baharda", "sonbaharda"]
    
    zarf = ["hızlıca", "yavaşça", "sessizce", "mutlu", "üzgün", "yorgun",
             "hevesle", "dikkatle", "sabırla", "cesaretle", "güzel", "kötü"]
    
    import random
    random.seed(42)
    
    templates = [
        lambda: f"{random.choice(ozne)} {random.choice(yer)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(nesne)} {random.choice(fiil)}",
        lambda: f"{random.choice(zaman)} {random.choice(ozne)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(zarf)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(yer)} {random.choice(nesne)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(zaman)} {random.choice(yer)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(nesne)} {random.choice(yer)} {random.choice(fiil)}",
        lambda: f"{random.choice(zarf)} {random.choice(ozne)} {random.choice(nesne)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(ozne)} ile {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(yer)} {random.choice(zaman)} {random.choice(fiil)}",
    ]
    
    return [random.choice(templates)() for _ in range(n)]


# ─────────────────────────────────────────────
# EĞİTİM
# ─────────────────────────────────────────────

def train(model: CloudLMND, texts: List[str], tok: Tok,
          epochs: int = 30, batch_size: int = 64, lr: float = 5e-4,
          seq_len: int = 5, device: str = "auto", verbose: bool = True):
    
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    all_tokens = [tok.encode(t) for t in texts]
    # Sadece seq_len+1 veya daha uzun olanları al
    all_tokens = [t for t in all_tokens if len(t) >= seq_len + 1]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  📊 {len(all_tokens)} cümle | vocab={tok.vocab_size} | params={n_params:,} | device={device}")
    
    model.train()
    for epoch in range(epochs):
        examples = []
        for tokens in all_tokens:
            for i in range(len(tokens) - seq_len):
                examples.append((tokens[i:i+seq_len], tokens[i+seq_len]))
        
        torch.manual_seed(epoch)  # her epoch farklı shuffle
        import random
        random.shuffle(examples)
        
        epoch_loss, n_batches = 0, 0
        t0 = time.time()
        
        for b in range(0, len(examples), batch_size):
            batch = examples[b:b+batch_size]
            if len(batch) < 2:
                continue
            inputs = torch.tensor([e[0] for e in batch], device=device)
            targets = torch.tensor([e[1] for e in batch], device=device)
            mask = (inputs != 0) & (inputs != 1)
            
            optimizer.zero_grad()
            word_logits, next_point, points = model(inputs, mask)
            loss = F.cross_entropy(word_logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 5 == 0:
            avg = epoch_loss / max(n_batches, 1)
            ppl = math.exp(min(avg, 20))
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg:.4f} | ppl={ppl:.2f} | {elapsed:.1f}s")
    
    model.eval()
    return model


# ─────────────────────────────────────────────
# DEĞERLENDİRME
# ─────────────────────────────────────────────

def evaluate(model: CloudLMND, texts: List[str], tok: Tok, 
             seq_len: int = 5, device: str = "cpu", n_test: int = 500):
    """Perplexity + generation kalitesi."""
    model.to(device)
    model.eval()
    
    all_tokens = [tok.encode(t) for t in texts if len(tok.encode(t)) >= seq_len + 1]
    import random
    random.seed(123)
    test_samples = random.sample(all_tokens, min(n_test, len(all_tokens)))
    
    total_loss, count = 0, 0
    with torch.no_grad():
        for tokens in test_samples:
            for i in range(len(tokens) - seq_len):
                inp = torch.tensor([tokens[i:i+seq_len]], device=device)
                tgt = torch.tensor([tokens[i+seq_len]], device=device)
                mask = (inp != 0) & (inp != 1)
                logits, _, _ = model(inp, mask)
                total_loss += F.cross_entropy(logits, tgt, reduction='sum').item()
                count += 1
    
    avg_loss = total_loss / max(count, 1)
    ppl = math.exp(min(avg_loss, 20))
    return ppl


def show_cloud(model: CloudLMND, text: str, tok: Tok, device: str = "cpu"):
    """Bulut geometrisi gösterimi."""
    model.to(device)
    model.eval()
    tokens = torch.tensor([tok.encode(text)], device=device)
    mask = (tokens != 0) & (tokens != 1)
    
    with torch.no_grad():
        _, _, points = model(tokens, mask)
    
    words = text.lower().replace(".", "").replace(",", "").split()
    pts = points[0].cpu().numpy()
    D = pts.shape[1]
    
    print(f"\n  Cümle: {text!r}")
    print(f"  Boyut: {D}D | Nokta sayısı: {len(words)}")
    print(f"  {'Kelime':15s} {'Koordinatlar':50s}")
    print(f"  {'-'*15} {'-'*50}")
    for w, p in zip(words, pts):
        coords = ", ".join(f"{v:+.3f}" for v in p[:min(D, 6)])
        if D > 6:
            coords += f", ... ({D}D)"
        print(f"  {w:15s} ({coords})")
    
    # Mesafe matrisi
    n = len(words)
    if n <= 8:
        print(f"\n  Mesafe matrisi:")
        header = "  " + " " * 12 + "".join(f"{w[:9]:>11s}" for w in words)
        print(header)
        for i in range(n):
            row = f"  {words[i][:10]:12s}"
            for j in range(n):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                row += f"{d:11.3f}"
            print(row)
    
    # Küme istatistikleri
    centroid = pts.mean(axis=0)
    spread = float(np.linalg.norm(pts - centroid, axis=1).mean())
    max_dist = float(np.max([np.linalg.norm(pts[i]-pts[j]) for i in range(n) for j in range(n)]))
    min_dist = float(np.min([np.linalg.norm(pts[i]-pts[j]) for i in range(n) for j in range(n) if i != j])) if n > 1 else 0
    
    print(f"\n  📐 Küme istatistikleri:")
    print(f"     Merkez: ({', '.join(f'{v:+.3f}' for v in centroid[:min(D,4)])}{'...' if D>4 else ''})")
    print(f"     Ort. yayılım: {spread:.3f}")
    print(f"     Min/Max mesafe: {min_dist:.3f} / {max_dist:.3f}")
    print(f"     Kompaktlık (min/max): {min_dist/max(max_dist,1e-8):.3f}")


import numpy as np


# ─────────────────────────────────────────────
# ANA: Boyut karşılaştırması
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CÜMLE BULUTU DİL MODELİ — ND KARŞILAŞTIRMASI")
    print("=" * 60)
    
    # Corpus
    print("\n📝 Corpus üretiliyor...")
    texts = generate_corpus(n=8000)
    tok = Tok(punct=True)
    vocab_size = tok.build(texts, max_vocab=500)
    print(f"   {len(texts)} cümle, {vocab_size} kelime")
    
    # Farklı boyutlarda modeller
    dims = [3, 7, 14]
    seq_len = 5
    epochs = 25
    
    results = {}
    
    for d in dims:
        print(f"\n{'─'*60}")
        print(f"  🌩️  {d}D BULUT")
        print(f"{'─'*60}")
        
        model = CloudLMND(vocab_size=vocab_size, d_model=96, d_cloud=192, d_cloud_dim=d)
        
        t0 = time.time()
        model = train(model, texts, tok, epochs=epochs, batch_size=64, 
                      lr=5e-4, seq_len=seq_len, verbose=True)
        train_time = time.time() - t0
        
        # Test perplexity
        ppl = evaluate(model, texts, tok, seq_len=seq_len, n_test=300)
        results[d] = {"model": model, "ppl": ppl, "time": train_time}
        print(f"  ✅ Test PPL: {ppl:.2f} | Eğitim: {train_time:.1f}s")
    
    # Özet tablo
    print(f"\n\n{'='*60}")
    print(f"  📊 KARŞILAŞTIRMA")
    print(f"{'='*60}")
    print(f"  {'Boyut':<10} {'PPL':<10} {'Zaman':<10} {'Param':<12}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for d in dims:
        r = results[d]
        n_p = sum(p.numel() for p in r["model"].parameters())
        print(f"  {d}D       {r['ppl']:<10.2f} {r['time']:<10.1f} {n_p:<12,}")
    
    # En iyi boyutla bulut gösterimi
    print(f"\n\n{'='*60}")
    print(f"  🌩️  BULUT GÖRSELLEŞTİRMESİ (7D)")
    print(f"{'='*60}")
    
    test_sentences = [
        "kedi bahçede uyudu",
        "çocuk parkta topu kovaladı",
        "kadın pazara domatesi aldı",
        "öğrenci kütüphanede kitabı okudu",
    ]
    
    model7 = results[7]["model"]
    for sent in test_sentences:
        show_cloud(model7, sent, tok)
        print()
    
    # Üretim
    print(f"\n{'='*60}")
    print(f"  🧪 ÜRETİM (7D, temperature=0.7)")
    print(f"{'='*60}")
    
    prompts = ["kedi bahçede", "çocuk parkta", "kadın pazara", "öğrenci kütüphanede"]
    device7 = next(model7.parameters()).device
    for p in prompts:
        tokens = torch.tensor([tok.encode(p)], device=device7)
        result = model7.generate(tokens, max_new=6, temperature=0.7)
        out = tok.decode(result[0].tolist())
        print(f"  {p:25s} → {out}")
    
    # 3D vs 7D aynı cümle
    print(f"\n\n{'='*60}")
    print(f"  🔬 AYNI CÜMLE, FARKLI BOYUT")
    print(f"{'='*60}")
    
    sent = "kedi bahçede uyudu"
    for d in [3, 7]:
        model_d = results[d]["model"]
        show_cloud(model_d, sent, tok)
        print()
    
    print("\n  Yorum: D arttıkça bulut daha 'açılır' —")
    print("  kelimeler birbirinden ayrılır, semantik roller netleşir.")
    print("  3D'de sıkışan noktalar 7D'de ayrışır.")


if __name__ == "__main__":
    main()
