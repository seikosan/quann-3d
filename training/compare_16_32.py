"""
Quann-3D — 16D vs 32D Geometri Karşılaştırması
================================================
Aynı cümleler, iki farklı bulut boyutu.
Kâğıt için ana figür + bölge analizi.

Kullanım:
  python3 compare_16_32.py
"""

import os
import sys

# Repozitör kökü (CWD'den bağımsız)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import numpy as np
import random
from collections import Counter
from sklearn.cluster import KMeans

from tokenizers import Tokenizer

# Model (kanonik tek kaynak): cloudlm paketi
from cloudlm import CloudLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ─────────────────────────────────────────────
# YÜKLEME
# ─────────────────────────────────────────────

def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = CloudLM(ckpt["vocab_size"], **cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def load_tokenizer(path):
    return Tokenizer.from_file(path)


# ─────────────────────────────────────────────
# ANALİZ
# ─────────────────────────────────────────────

def analyze_regions(model, tokenizer, texts, device="cpu", n_clusters_range=range(3, 15)):
    """
    Kelime noktalarının küme analizi.
    Kaç ayrı "bölge" var?
    """
    eos_id = tokenizer.token_to_id("<eos>") or 1
    
    # Tüm kelimelerin ortalama pozisyonlarını topla
    word_positions = {}
    
    for sent in texts:
        tokens = torch.tensor([tokenizer.encode(sent).ids], device=device)
        mask = (tokens != 0) & (tokens != eos_id)
        pts = model.get_cloud(tokens, mask)
        
        # Tokenları decode et (kelime seviyesinde grupla)
        enc = tokenizer.encode(sent)
        token_strs = enc.tokens
        
        # Basit: her token'ı ayrı nokta olarak say
        for i, (t, p) in enumerate(zip(token_strs, pts)):
            if t not in word_positions:
                word_positions[t] = []
            word_positions[t].append(p)
    
    # Sık kelimeleri al
    all_counts = Counter(t for t in word_positions if len(word_positions[t]) > 5)
    top_words = list(all_counts.keys())[:50]
    
    # Ortalama pozisyonlar
    avg_positions = {}
    for w in top_words:
        if w in word_positions:
            avg_positions[w] = np.mean(word_positions[w], axis=0)
    
    if len(avg_positions) < 5:
        return None
    
    pos_matrix = np.array(list(avg_positions.values()))
    words_list = list(avg_positions.keys())
    
    # KMeans ile farklı cluster sayıları dene
    best_k = None
    best_score = float('inf')
    silhouette_scores = {}
    
    from sklearn.metrics import silhouette_score
    
    for k in n_clusters_range:
        if len(pos_matrix) < k + 2:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(pos_matrix)
        
        # Silhouette score
        if len(set(labels)) > 1:
            score = silhouette_score(pos_matrix, labels)
            silhouette_scores[k] = score
            if score > best_score if best_score != float('inf') else True:
                if best_score == float('inf') or score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
    
    # Cluster içi ortalama konumlar
    if best_k:
        cluster_centers = {}
        for k in range(best_k):
            members = [w for w, l in zip(words_list, best_labels) if l == k]
            if members:
                center = np.mean([avg_positions[w] for w in members], axis=0)
                cluster_centers[k] = {"members": members[:8], "center": center, "size": len(members)}
    else:
        cluster_centers = {}
    
    return {
        "n_words": len(avg_positions),
        "best_k": best_k,
        "silhouette": best_score,
        "silhouette_by_k": silhouette_scores,
        "clusters": cluster_centers,
        "avg_positions": avg_positions,
    }


def plot_comparison(model16, model32, tokenizer, texts, save_dir, device="cpu"):
    """16D vs 32D yan yana 3D projeksiyon."""
    eos_id = tokenizer.token_to_id("<eos>") or 1
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, sent in enumerate(texts[:6]):
        ax = axes[idx]
        
        for model, dim_label in [(model16, "16D"), (model32, "32D")]:
            tokens = torch.tensor([tokenizer.encode(sent).ids], device=device)
            mask = (tokens != 0) & (tokens != eos_id)
            pts = model.get_cloud(tokens, mask)[:, :3]  # 3D projeksiyon
            
            enc = tokenizer.encode(sent)
            token_strs = enc.tokens
            
            color = colors[0] if dim_label == "16D" else colors[5]
            offset = 0.3 if dim_label == "32D" else 0  # ayır
            
            for i, (t, p) in enumerate(zip(token_strs, pts)):
                ax.scatter(p[0]+offset, p[1], p[2], s=80, color=color, 
                          alpha=0.7, edgecolors='black', linewidth=0.3)
        
        ax.set_title(f"{sent[:40]!r}", fontsize=9)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    plt.suptitle("16D (mavi) vs 32D (turuncu) — 3D Projeksiyon", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "compare_16_32.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📸 {path}")
    return path


def plot_region_map(model, tokenizer, texts, save_dir, dim_label, device="cpu"):
    """Kelime bölge haritası — 2D projeksiyon (PCA)."""
    from sklearn.decomposition import PCA
    
    eos_id = tokenizer.token_to_id("<eos>") or 1
    
    word_positions = {}
    for sent in texts:
        tokens = torch.tensor([tokenizer.encode(sent).ids], device=device)
        mask = (tokens != 0) & (tokens != eos_id)
        pts = model.get_cloud(tokens, mask)
        enc = tokenizer.encode(sent)
        for t, p in zip(enc.tokens, pts):
            if t not in word_positions:
                word_positions[t] = []
            word_positions[t].append(p)
    
    # Sık kelimeler
    top = [w for w in word_positions if len(word_positions[w]) > 3][:30]
    if len(top) < 5:
        return None
    
    pos_matrix = np.array([np.mean(word_positions[w], axis=0) for w in top])
    
    # PCA → 2D
    pca = PCA(n_components=2)
    pos_2d = pca.fit_transform(pos_matrix)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(pos_2d[:, 0], pos_2d[:, 1], s=100, alpha=0.7,
                         c=range(len(top)), cmap='tab20', edgecolors='black', linewidth=0.5)
    
    for i, w in enumerate(top):
        ax.annotate(w, (pos_2d[i, 0], pos_2d[i, 1]), fontsize=8,
                   ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Kelime Bölge Haritası ({dim_label})")
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(save_dir, f"region_map_{dim_label}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📸 {path}")
    return path


def plot_spread_comparison(model16, model32, tokenizer, texts, save_dir, device="cpu"):
    """Bulut yayılımı karşılaştırması — 32D daha 'açık' mı?"""
    eos_id = tokenizer.token_to_id("<eos>") or 1
    
    spreads_16, spreads_32 = [], []
    
    for sent in texts:
        tokens = torch.tensor([tokenizer.encode(sent).ids], device=device)
        mask = (tokens != 0) & (tokens != eos_id)
        
        pts16 = model16.get_cloud(tokens, mask)
        pts32 = model32.get_cloud(tokens, mask)
        
        # Yayılım: merkezden ortalama uzaklık
        c16 = pts16.mean(axis=0)
        c32 = pts32.mean(axis=0)
        spreads_16.append(float(np.linalg.norm(pts16 - c16, axis=1).mean()))
        spreads_32.append(float(np.linalg.norm(pts32 - c32, axis=1).mean()))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(spreads_16, bins=30, alpha=0.7, color='steelblue', label='16D')
    ax1.hist(spreads_32, bins=30, alpha=0.7, color='coral', label='32D')
    ax1.set_xlabel("Bulut Yayılımı (ortalama merkez uzaklığı)")
    ax1.set_ylabel("Frekans")
    ax1.set_title("Yayılım Dağılımı")
    ax1.legend()
    
    # Box plot
    ax2.boxplot([spreads_16, spreads_32], tick_labels=['16D', '32D'])
    ax2.set_ylabel("Yayılım")
    ax2.set_title("Karşılaştırma")
    
    plt.tight_layout()
    path = os.path.join(save_dir, "spread_compare.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📸 {path}")
    
    print(f"\n  16D ort. yayılım: {np.mean(spreads_16):.3f} ± {np.std(spreads_16):.3f}")
    print(f"  32D ort. yayılım: {np.mean(spreads_32):.3f} ± {np.std(spreads_32):.3f}")
    print(f"  Oran (32D/16D): {np.mean(spreads_32)/max(np.mean(spreads_16),1e-8):.2f}x")


# ─────────────────────────────────────────────
# ANA
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  16D vs 32D — GEOMETRİ KARŞILAŞTIRMASI")
    print("=" * 60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    save_dir = os.path.join(ROOT, "figures", "comparison")
    os.makedirs(save_dir, exist_ok=True)
    
    # Model yükle
    print(f"\n  Model yükleme...")
    
    # 16D
    ckpt16_path = os.path.join(ROOT, "checkpoints", "wiki_checkpoints", "wiki_cloud_best.pt")
    if not os.path.exists(ckpt16_path):
        ckpt16_path = os.path.join(ROOT, "checkpoints", "wiki_checkpoints", "wiki_cloud_epoch50.pt")
    model16, ckpt16 = load_model(ckpt16_path, device)
    print(f"  16D: epoch {ckpt16['epoch']}, PPL {ckpt16['ppl']:.2f}")
    
    # 32D
    ckpt32_path = os.path.join(ROOT, "checkpoints", "wiki_checkpoints_32d", "wiki_cloud_best.pt")
    if not os.path.exists(ckpt32_path):
        ckpt32_path = os.path.join(ROOT, "checkpoints", "wiki_checkpoints_32d", "wiki_cloud_epoch50.pt")
    if not os.path.exists(ckpt32_path):
        # Henüz tamamlanmamış olabilir
        import glob
        files = glob.glob(os.path.join(ROOT, "checkpoints", "wiki_checkpoints_32d", "wiki_cloud_epoch*.pt"))
        if files:
            # Sayısal sıralama (leksikografik sorted ile epoch10 < epoch2 olurdu)
            def _epoch_num(p):
                base = os.path.basename(p)
                try:
                    return int(base.replace("wiki_cloud_epoch", "").replace(".pt", ""))
                except ValueError:
                    return -1
            ckpt32_path = max(files, key=_epoch_num)
        else:
            print("  ⚠️  32D checkpoint bulunamadı! Eğitim tamamlanmadı.")
            print("  Sadece 16D analizi yapılıyor.")
            model32 = None
            ckpt32 = None
    if os.path.exists(ckpt32_path):
        model32, ckpt32 = load_model(ckpt32_path, device)
        print(f"  32D: epoch {ckpt32['epoch']}, PPL {ckpt32['ppl']:.2f}")
    
    # Tokenizer
    tok_path = os.path.join(ROOT, "checkpoints", "wiki_checkpoints", "wiki_bpe.json")
    tokenizer = load_tokenizer(tok_path)
    
    # Test cümleleri
    texts = []
    with open(os.path.join(ROOT, "data", "turkish_sentences.txt"), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 500: break
            t = line.strip()
            if 30 <= len(t) <= 200:
                texts.append(t)
    
    random.seed(42)
    test_texts = random.sample(texts, min(100, len(texts)))
    print(f"  {len(test_texts)} test cümlesi")
    
    # ── 1. Bölge Analizi ──
    print(f"\n{'='*60}")
    print(f"  1. BÖLGE ANALİZİ (KMeans)")
    print(f"{'='*60}")
    
    results = {}
    
    for label, model in [("16D", model16), ("32D", model32)]:
        if model is None:
            continue
        print(f"\n  {label}:")
        res = analyze_regions(model, tokenizer, test_texts, device)
        if res:
            results[label] = res
            print(f"     Kelime sayısı: {res['n_words']}")
            print(f"     Optimal cluster: k={res['best_k']}")
            print(f"     Silhouette: {res['silhouette']:.3f}")
            print(f"     Silhouette by k: {dict(sorted(res['silhouette_by_k'].items()))}")
            
            # Cluster örnekleri
            for k, c in sorted(res['clusters'].items())[:5]:
                members = ", ".join(c['members'][:5])
                print(f"     Bölge {k} ({c['size']} kelime): {members}")
    
    # ── 2. Karşılaştırma ──
    if model32 is not None:
        print(f"\n{'='*60}")
        print(f"  2. YAN YANA KARŞILAŞTIRMA")
        print(f"{'='*60}")
        
        # 3D projeksiyon
        vis_texts = [t for t in test_texts if 15 <= len(t) <= 80][:6]
        plot_comparison(model16, model32, tokenizer, vis_texts, save_dir, device)
        
        # Yayılım
        print(f"\n  3. YAYILIM KARŞILAŞTIRMASI:")
        plot_spread_comparison(model16, model32, tokenizer, test_texts, save_dir, device)
    
    # ── 3. Bölge Haritaları ──
    print(f"\n{'='*60}")
    print(f"  4. BÖLGE HARİTASI (PCA 2D)")
    print(f"{'='*60}")
    
    plot_region_map(model16, tokenizer, test_texts, save_dir, "16D", device)
    if model32 is not None:
        plot_region_map(model32, tokenizer, test_texts, save_dir, "32D", device)
    
    # ── ÖZET ──
    print(f"\n\n{'='*60}")
    print(f"  📊 ÖZET")
    print(f"{'='*60}")
    
    if "16D" in results and "32D" in results:
        r16, r32 = results["16D"], results["32D"]
        print(f"  {'':10s} {'16D':>10s} {'32D':>10s}")
        print(f"  {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'Bölge (k)':>10s} {r16['best_k']:>10d} {r32['best_k']:>10d}")
        print(f"  {'Silhouette':>10s} {r16['silhouette']:>10.3f} {r32['silhouette']:>10.3f}")
        print(f"  {'Kelime':>10s} {r16['n_words']:>10d} {r32['n_words']:>10d}")
        print(f"\n  Yorum: 32D'de {r32['best_k']} bölge (16D'de {r16['best_k']})")
        print(f"  → Boyut arttıkça daha ince ayrışma")
    elif "16D" in results:
        r16 = results["16D"]
        print(f"  16D: {r16['best_k']} bölge, silhouette={r16['silhouette']:.3f}")
        print(f"  (32D henüz hazır değil)")
    
    print(f"\n  ✅ TAMAM")
    print(f"  Görseller: {save_dir}/")


if __name__ == "__main__":
    main()
