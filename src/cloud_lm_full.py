"""
Cümle Bulutu Dil Modeli — Tam Boyut
=====================================
- Büyük model (d_model=256, d_cloud=512, 4 katman)
- D=16 ve D=32 karşılaştırması
- 3D scatter plot görselleştirme
- Daha zengin corpus

Kullanım:
  python3 cloud_lm_full.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import sys
import random
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")

# Ortak mimari (tek kaynak)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cloudlm import CloudLM, WordToND, CloudEncoderND, Tok  # noqa: F401
# Geriye dönük ad
CloudLMFull = CloudLM  # noqa: F811
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ─────────────────────────────────────────────
# BÜYÜK CORPUS (30K cümle, zengin şablonlar)
# ─────────────────────────────────────────────

def generate_corpus(n=30000):
    """Zengin Türkçe cümle üretici — 15+ şablon, 50+ kelime havuzu."""
    random.seed(42)
    
    ozne = ["kedi","köpek","çocuk","kadın","adam","kız","erkek","öğrenci",
            "öğretmen","doktor","mühendis","sanatçı","müzisyen","yazar",
            "kuş","balık","at","keçi","tavuk","inek","fare","yılan","arı",
            "balina","kaplumbağa","yengeç","karınca","kelebek","penguen","panda"]
    
    fiil = ["koştu","uyudu","baktı","içti","yedi","okudu","yazdı","söyledi",
            "duydu","gördü","aldı","verdi","gitti","geldi","döndü","oturdu",
            "ayaktaydı","düşündü","hissetti","anladı","öğrendi","keşfetti",
            "yapıyordu","yapacak","yapmış","etmişti","gidiyordu","bakıyordu",
            "şarkı söyledi","dans etti","yemek yaptı","kitap yazdı","resim çizdi",
            "müzik dinledi","fotoğraf çekti","mektup yazdı","harita çizdi",
            "top oynadı","bisiklet sürdü","yüzme yaptı","dağa tırmandı"]
    
    nesne = ["topu","kitabı","gazeteyi","mektubu","haritayı","listeyi",
             "pencereyi","kapıyı","masayı","sandalyeyi","arabayı","bisikleti",
             "domatesi","ekmeği","sütü","çayı","kahveyi","müziği","şarkıyı",
             "hikayeyi","tabloyu","heykeli","arabayı","kutu","çantayı",
             "telefonu","bilgisayarı","televizyonu","radyoyu","kitapları",
             "fotoğrafları","notları","arançayı","kemeri","saatini"]
    
    yer = ["bahçede","pazara","okula","evde","sokakta","plajda","dağda",
           "ormanda","pazarda","müzede","sinemaya","hastaneye","kütüphanede",
           "restoranda","parkta","tarlada","denizde","pencereden","kapıdan",
           "merdivenden","köprüden","tünelden","yoldan","kaldırımda",
           "avluda","çatıda","bodrumda","terasda","sahilde","tepeye",
           "vadiye","kıyıda","ada","yarımada","boğazda","mevkide"]
    
    zaman = ["sabah","akşam","gece","öğlen","hafta sonu","dün","bugün",
             "yarın","yazın","kışın","baharda","sonbaharda","geçen ay",
             "bu hafta","şu an","az önce","birazdan","yılbaşı","ramazanda"]
    
    zarf = ["hızlıca","yavaşça","sessizce","mutlu","üzgün","yorgun",
             "hevesle","dikkatle","sabırla","cesaretle","güzel","kötü",
             "kolayca","zorla","birden","yavaş yavaş","titrek","gür",
             "incelikli","ciddi","neşeli","sakin","heyecanlı","üzüntülü"]
    
    ile = ["annesiyle","babasıyla","arkadaşıyla","kardeşiyle","öğretmeniyle",
           "komşusuyla","ailesiyle","ekibiyle","takımıyla","sınıfıyla"]
    
    templates = [
        lambda: f"{random.choice(ozne)} {random.choice(yer)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(nesne)} {random.choice(fiil)}",
        lambda: f"{random.choice(zaman)} {random.choice(ozne)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(zarf)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(yer)} {random.choice(nesne)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(zaman)} {random.choice(yer)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(nesne)} {random.choice(yer)} {random.choice(fiil)}",
        lambda: f"{random.choice(zarf)} {random.choice(ozne)} {random.choice(nesne)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(ile)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(yer)} {random.choice(zaman)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(zarf)} {random.choice(yer)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(ile)} {random.choice(yer)} {random.choice(fiil)}",
        lambda: f"{random.choice(zaman)} {random.choice(ozne)} {random.choice(nesne)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(ozne)} {random.choice(nesne)} {random.choice(fiil)}",
        lambda: f"{random.choice(ozne)} {random.choice(yer)} {random.choice(ile)} {random.choice(fiil)}",
    ]
    
    return [random.choice(templates)() for _ in range(n)]


# ─────────────────────────────────────────────
# EĞİTİM
# ─────────────────────────────────────────────

def train(model, texts, tok, epochs=30, batch_size=128, lr=3e-4,
          seq_len=5, device="auto", verbose=True):
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    all_tokens = [tok.encode(t) for t in texts]
    all_tokens = [t for t in all_tokens if len(t) >= seq_len + 1]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  📊 {len(all_tokens)} cümle | vocab={tok.vocab_size} | params={n_params:,} | {device}")
    
    model.train()
    for epoch in range(epochs):
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
            word_logits, _, _ = model(inputs, mask)
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
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg:.4f} | ppl={ppl:.2f} | {time.time()-t0:.1f}s")
    
    model.eval()
    return model


def evaluate_ppl(model, texts, tok, seq_len=5, device="cpu", n_test=500):
    model.to(device)
    model.eval()
    all_tokens = [tok.encode(t) for t in texts if len(tok.encode(t)) >= seq_len + 1]
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
    return math.exp(min(total_loss / max(count, 1), 20))


# ─────────────────────────────────────────────
# GÖRSELLEŞTİRME
# ─────────────────────────────────────────────

def plot_cloud_3d(model, text, tok, device="cpu", save_path="cloud_3d.png",
                  title=None, d_cloud_dim=3):
    """3D scatter plot — sadece ilk 3 boyutu gösterir."""
    model.to(device)
    model.eval()
    tokens = torch.tensor([tok.encode(text)], device=device)
    mask = (tokens != 0) & (tokens != 1)
    
    with torch.no_grad():
        _, _, points = model(tokens, mask)
    
    pts = points[0].cpu().numpy()
    words = text.lower().replace(".","").replace(",","").split()
    
    # İlk 3 boyutu kullan (D>3 ise projeksiyon)
    pts3 = pts[:, :3]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(words), 2)))
    
    for i, (w, p) in enumerate(zip(words, pts3)):
        ax.scatter(*p, s=200, color=colors[i % 10], edgecolors='black', 
                   linewidth=0.5, zorder=5)
        ax.text(p[0], p[1], p[2], f"  {w}", fontsize=9, fontweight='bold',
                color=colors[i % 10])
    
    # Çiftler arası çizgiler (mesafe < threshold ise)
    max_dist = np.max([np.linalg.norm(pts3[i]-pts3[j]) for i in range(len(pts3)) for j in range(len(pts3)) if i!=j])
    threshold = max_dist * 0.6
    for i in range(len(pts3)):
        for j in range(i+1, len(pts3)):
            d = np.linalg.norm(pts3[i]-pts3[j])
            if d < threshold:
                ax.plot([pts3[i][0], pts3[j][0]],
                        [pts3[i][1], pts3[j][1]],
                        [pts3[i][2], pts3[j][2]],
                        'gray', alpha=0.4, linewidth=1)
                mid = (pts3[i] + pts3[j]) / 2
                ax.text(mid[0], mid[1], mid[2], f"{d:.2f}", fontsize=7, color='gray')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or f"Cümle Bulutu: {text!r} ({d_cloud_dim}D → 3D projeksiyon)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📸 Kaydedildi: {save_path}")
    return save_path


def plot_cloud_comparison(models_dict, text, tok, device="cpu", save_path="cloud_compare.png"):
    """Aynı cümle, farklı boyutlar — yan yana 3D projeksiyon."""
    dims = list(models_dict.keys())
    n = len(dims)
    
    fig = plt.figure(figsize=(5 * n + 2, 6))
    
    for idx, (d, model) in enumerate(models_dict.items()):
        model.to(device)
        model.eval()
        tokens = torch.tensor([tok.encode(text)], device=device)
        mask = (tokens != 0) & (tokens != 1)
        
        with torch.no_grad():
            _, _, points = model(tokens, mask)
        
        pts = points[0].cpu().numpy()[:, :3]  # ilk 3D
        words = text.lower().replace(".","").replace(",","").split()
        
        ax = fig.add_subplot(1, n, idx+1, projection='3d')
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(words), 2)))
        
        for i, (w, p) in enumerate(zip(words, pts)):
            ax.scatter(*p, s=150, color=colors[i % 10], edgecolors='black', linewidth=0.5)
            ax.text(p[0], p[1], p[2], f" {w[:8]}", fontsize=8, color=colors[i % 10])
        
        ax.set_title(f"{d}D", fontsize=12, fontweight='bold')
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    plt.suptitle(f"Bulut Geometrisi: {text!r}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📸 Kaydedildi: {save_path}")
    return save_path


def plot_cloud_evolution(model, text, tok, device="cpu", save_path="cloud_evolution.png"):
    """Kademeli cümle inşası — bulutun evrimi."""
    model.to(device)
    model.eval()
    words = text.lower().replace(".","").replace(",","").split()
    
    n_steps = min(len(words), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(words), 2)))
    
    for step in range(n_steps):
        partial = " ".join(words[:step+1])
        tokens = torch.tensor([tok.encode(partial)], device=device)
        mask = (tokens != 0) & (tokens != 1)
        
        with torch.no_grad():
            _, _, points = model(tokens, mask)
        
        pts = points[0].cpu().numpy()[:, :3]
        ax = axes[step]
        
        for i, (w, p) in enumerate(zip(words[:step+1], pts)):
            ax.scatter(*p, s=180, color=colors[i % 10], edgecolors='black', linewidth=0.5)
            ax.text(p[0], p[1], p[2], f" {w[:7]}", fontsize=8, color=colors[i % 10])
        
        # Çizgiler
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        [pts[i][2], pts[j][2]], 'gray', alpha=0.3, linewidth=0.8)
        
        ax.set_title(f"Adım {step+1}: {partial!r}", fontsize=9)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    # Boş slotları gizle
    for step in range(n_steps, len(axes)):
        axes[step].set_visible(False)
    
    plt.suptitle(f"Bulut Evrimi: {text!r}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📸 Kaydedildi: {save_path}")
    return save_path


# ─────────────────────────────────────────────
# ANA
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CÜMLE BULUTU — TAM BOYUT (256D model, 512D cloud)")
    print("=" * 60)
    
    # Corpus
    print("\n📝 Corpus...")
    texts = generate_corpus(n=30000)
    tok = Tok(punct=True, fold_i=True)
    vocab_size = tok.build(texts, max_vocab=2000)
    print(f"   {len(texts)} cümle, {vocab_size} kelime")
    
    # Model konfigürasyonları
    configs = {
        16: {"d_model": 256, "d_cloud": 512, "n_layers": 4},
        32: {"d_model": 256, "d_cloud": 512, "n_layers": 4},
    }
    
    seq_len = 5
    epochs = 20
    models = {}
    
    for d, cfg in configs.items():
        print(f"\n{'─'*60}")
        print(f"  🌩️  {d}D BULUT (d_model={cfg['d_model']}, d_cloud={cfg['d_cloud']}, layers={cfg['n_layers']})")
        print(f"{'─'*60}")
        
        model = CloudLMFull(
            vocab_size=vocab_size,
            d_model=cfg["d_model"],
            d_cloud=cfg["d_cloud"],
            d_cloud_dim=d,
            n_layers=cfg["n_layers"]
        )
        
        t0 = time.time()
        model = train(model, texts, tok, epochs=epochs, batch_size=128,
                      lr=3e-4, seq_len=seq_len, verbose=True)
        train_time = time.time() - t0
        
        ppl = evaluate_ppl(model, texts, tok, seq_len=seq_len, n_test=500)
        n_params = sum(p.numel() for p in model.parameters())
        
        models[d] = model
        print(f"  ✅ PPL={ppl:.2f} | params={n_params:,} | time={train_time:.1f}s")
    
    # Karşılaştırma tablosu
    print(f"\n\n{'='*60}")
    print(f"  📊 SONUÇLAR")
    print(f"{'='*60}")
    print(f"  {'Boyut':<8} {'PPL':<10} {'Param':<12} {'Zaman':<8}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*8}")
    for d, m in models.items():
        ppl = evaluate_ppl(m, texts, tok, seq_len=seq_len, n_test=200)
        n_p = sum(p.numel() for p in m.parameters())
        print(f"  {d}D     {ppl:<10.2f} {n_p:<12,} {train_time:<8.1f}")
    
    # ── GÖRSELLEŞTİRME ──
    print(f"\n\n{'='*60}")
    print(f"  📸 GÖRSELLEŞTİRME")
    print(f"{'='*60}")
    
    test_sentences = [
        "kedi bahçede uyudu",
        "çocuk parkta topu kovaladı",
        "kadın pazara domatesi aldı",
        "öğrenci kütüphanede kitabı okudu",
        "müzisyen sahne de şarkı söyledi",
    ]
    
    # 1. Tek cümle 3D scatter
    print("\n  1) Tek cümle 3D:")
    for sent in test_sentences[:3]:
        plot_cloud_3d(models[16], sent, tok, 
                     save_path=f"cloud_{sent.split()[0]}.png",
                     title=f"{sent!r} (16D→3D)", d_cloud_dim=16)
    
    # 2. Boyut karşılaştırması (aynı cümle, 16D vs 32D)
    print("\n  2) Boyut karşılaştırması:")
    plot_cloud_comparison(models, "kedi bahçede uyudu", tok,
                         save_path="cloud_dim_compare.png")
    
    # 3. Bulut evrimi (kademeli inşaa)
    print("\n  3) Bulut evrimi:")
    plot_cloud_evolution(models[16], "çocuk parkta topu kovaladı", tok,
                        save_path="cloud_evolution.png")
    
    # ── GEOMETRİ ANALİZİ ──
    print(f"\n\n{'='*60}")
    print(f"  🔬 GEOMETRİ ANALİZİ (16D)")
    print(f"{'='*60}")
    
    model16 = models[16]
    model16.to("cpu")
    
    for sent in test_sentences:
        tokens = torch.tensor([tok.encode(sent)])
        mask = (tokens != 0) & (tokens != 1)
        with torch.no_grad():
            _, _, points = model16(tokens, mask)
        
        pts = points[0].numpy()
        words = sent.lower().replace(".","").replace(",","").split()
        n = len(words)
        
        # Mesafeler
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                dists.append((d, words[i], words[j]))
        dists.sort()
        
        centroid = pts.mean(axis=0)
        spread = np.linalg.norm(pts - centroid, axis=1).mean()
        
        print(f"\n  {sent!r}")
        print(f"     Yayılım: {spread:.3f} | Noktalar: {n}")
        print(f"     En yakın:  {dists[0][1]} ↔ {dists[0][2]} ({dists[0][0]:.3f})")
        print(f"     En uzak:   {dists[-1][1]} ↔ {dists[-1][2]} ({dists[-1][0]:.3f})")
    
    # ── ÜRETİM ──
    print(f"\n\n{'='*60}")
    print(f"  🧪 ÜRETİM (16D, temp=0.6)")
    print(f"{'='*60}")
    
    prompts = ["kedi bahçede", "çocuk parkta", "kadın pazara", 
               "öğrenci kütüphanede", "müzisyen sahne"]
    for p in prompts:
        tokens = torch.tensor([tok.encode(p)])
        result = model16.generate(tokens, max_new=5, temperature=0.6)
        out = tok.decode(result[0].tolist())
        print(f"  {p:25s} → {out}")
    
    print(f"\n\n{'='*60}")
    print(f"  ✅ TAMAM")
    print(f"{'='*60}")
    print(f"  Görseller: cloud_*.png, cloud_dim_compare.png, cloud_evolution.png")


if __name__ == "__main__":
    main()
