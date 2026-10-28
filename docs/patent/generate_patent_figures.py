"""
Patent Şekilleri Üretici
========================
v2.4 için 6 patent şekli:
1. Şekil 1: Evrensel mimari ve modül akışı
2. Şekil 2: Çift-geometri ve simetrik birleştirme
3. Şekil 3: Aynı buluttan skor, ablation, prototip ve drift
4. Şekil 4: D=64 + Nokta=20 optimizasyonu
5. Şekil 5: Max-pool vs attention-pool karşılaştırması
6. Şekil 6: Bulut geometri vizualizasyonu (3D)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG_DIR = Path("docs/patent/fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# ŞEKİL 1: Evrensel Mimari
# ─────────────────────────────────────────────

def fig1_architecture():
    """Evrensel mimari ve modül akışı (100-290)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Basit blok diyagram
    boxes = [
        ("100 Girdi/Bağlam", 0.1, 0.5),
        ("110 Nokta Projeksiyonu\n(D-boyutlu)", 0.3, 0.5),
        ("120 Çift-Geometri\nKodlayıcı", 0.5, 0.5),
        ("123 Simetrik\nBirleştirme", 0.7, 0.5),
        ("130 Okuyucu\n(Skor/Çıktı)", 0.85, 0.5),
    ]
    
    # Alt satır - açıklama modülleri
    bottom_boxes = [
        ("150 Yorumlama", 0.3, 0.2),
        ("260 Ablation", 0.45, 0.2),
        ("270 Decoder", 0.6, 0.2),
        ("280 Karşı-Faktürel", 0.75, 0.2),
        ("290 Dağıtım\nPaketleyici", 0.9, 0.2),
    ]
    
    y_box = 0.65
    box_width = 0.15
    box_height = 0.08
    
    for i, (text, x, y) in enumerate(boxes + bottom_boxes):
        rect = plt.Rectangle((x, y), box_width, box_height, 
                            facecolor='#4A90D9' if y > 0.3 else '#5CB85C',
                            edgecolor='#333', linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + box_width/2, y + box_height/2, text,
               ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Oklar
    arrows = [
        (0.25, 0.55, 0.3, 0.55),
        (0.45, 0.55, 0.5, 0.55),
        (0.65, 0.55, 0.7, 0.55),
        (0.82, 0.55, 0.85, 0.55),
        (0.5, 0.45, 0.5, 0.35),  # 120 → 150
        (0.7, 0.45, 0.7, 0.35),  # 123 → 270
        (0.85, 0.45, 0.85, 0.35),  # 130 → 290
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#666'))
    
    ax.set_title('Şekil 1: Evrensel Mimari ve Modül Akışı', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_mimari.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Şekil 1: fig1_mimari.png")

# ─────────────────────────────────────────────
# ŞEKİL 2: Çift-Geometri
# ─────────────────────────────────────────────

def fig2_pair_geometry():
    """Çift-geometri ve simetrik birleştirme"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Nokta bulutu
    ax = axes[0]
    np.random.seed(42)
    points = np.random.randn(10, 3) * 0.5
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='#4A90D9', alpha=0.8)
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            ax.plot([points[i, 0], points[j, 0]], 
                   [points[i, 1], points[j, 1]], 
                   [points[i, 2], points[j, 2]], 
                   'gray', alpha=0.3, lw=0.5)
    ax.set_title('Nokta Bulutu', fontweight='bold')
    
    # Çift-geometri vektörü
    ax = axes[1]
    ax.axis('off')
    
    vec_text = """
    x_ij = [pᵢ - pⱼ, ‖pᵢ - pⱼ‖, cos(θᵢⱼ), ‖pᵢ‖, ‖pⱼ‖]
    
    Bileşenler:
    • Fark vektörü: pᵢ - pⱼ (D boyut)
    • Mesafe: ‖pᵢ - pⱼ‖ (1 boyut)
    • Kosinüs açı: cos(θᵢⱼ) (1 boyut)
    • Norm: ‖pᵢ‖ (1 boyut)
    • Norm: ‖pⱼ‖ (1 boyut)
    
    Toplam: D + 4 boyut
    """
    ax.text(0.1, 0.9, vec_text, fontsize=11, family='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.8))
    
    # Simetrik agregasyon
    ax = axes[2]
    ax.axis('off')
    
    agg_text = """
    Simetrik Küme Agregasyonu:
    
    Nokta Özellikleri → max-pool
    Çift Özellikleri  → mean-pool
    
    Fusion = [max_pooled_point, mean_pooled_pair]
    
    ↓
    
    cloud_state (512 boyut)
    """
    ax.text(0.1, 0.9, agg_text, fontsize=11, family='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#D4EDDA', alpha=0.8))
    
    fig.suptitle('Şekil 2: Çift-Geometri ve Simetrik Birleştirme', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_ciftgeometri.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Şekil 2: fig2_ciftgeometri.png")

# ─────────────────────────────────────────────
# ŞEKİL 3: Aynı Buluttan Açıklama
# ─────────────────────────────────────────────

def fig3_explanation():
    """Aynı buluttan skor, ablation, prototip ve drift"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 1. Bulut
    ax = axes[0]
    points = np.random.randn(15, 3) * 0.4
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='#4A90D9', alpha=0.8)
    ax.set_title('Nokta Bulutu', fontweight='bold')
    ax.axis('off')
    
    # 2. Ablation
    ax = axes[1]
    ax.axis('off')
    ablation_text = """
    Leave-One-Point-Out
    
    p_k çıkarıldığında:
    logit_change = logit_full - logit_without_p_k
    
    Örnek:
    "bahçe" noktası çıkarıldığında
    logit değişimi: -2.34
    
    → "bahçe" kritik katkı
    """
    ax.text(0.1, 0.9, ablation_text, fontsize=10, family='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#F8D7DA', alpha=0.8))
    
    # 3. Prototip
    ax = axes[2]
    ax.axis('off')
    proto_text = """
    Bölge/Prototip İsimlendirme
    
    En yakın prototip: "Türkiye"
    Mesafe: 0.42
    Destek: %78
    
    → Benzer anlam bölgesi
    """
    ax.text(0.1, 0.9, proto_text, fontsize=10, family='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#D1ECF1', alpha=0.8))
    
    # 4. Drift
    ax = axes[3]
    ax.axis('off')
    drift_text = """
    Bölge Kayması / Drift
    
    Zaman t1: Bölge 7 = %21
    Zaman t2: Bölge 7 = %26 (+5%)
    
    → Anlam kayması algılandı
    """
    ax.text(0.1, 0.9, drift_text, fontsize=10, family='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.8))
    
    fig.suptitle('Şekil 3: Aynı Buluttan Native Açıklama', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_aciklama.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Şekil 3: fig3_aciklama.png")

# ─────────────────────────────────────────────
# ŞEKİL 4: Nokta Sayısı Optimizasyonu
# ─────────────────────────────────────────────

def fig4_point_count_optimization():
    """D=64 + Nokta sayısı optimizasyonu"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    point_counts = [10, 15, 20, 25, 30]
    ppl_scores = [57.04, 45.2, 37.63, 42.1, 134.31]
    
    ax.plot(point_counts, ppl_scores, 'o-', linewidth=2, markersize=10, color='#4A90D9')
    ax.fill_between(point_counts, ppl_scores, alpha=0.3, color='#4A90D9')
    
    # Optimum vurgusu
    ax.axvline(20, color='#28A745', linestyle='--', linewidth=2, alpha=0.7)
    ax.annotate('OPTIMUM\n(37.63 PPL)', xy=(20, 37.63), xytext=(25, 50),
               arrowprops=dict(arrowstyle='->', color='#28A745', lw=2),
               fontsize=12, fontweight='bold', color='#28A745')
    
    ax.set_xlabel('Nokta Sayısı', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perplexity (PPL)', fontsize=12, fontweight='bold')
    ax.set_title('Şekil 4: Nokta Sayısı Optimizasyonu (D=64)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(point_counts)
    
    # İyileşme metni
    ax.text(0.5, 0.95, '20 nokta: %34 iyileşme (57.04 → 37.63)\nBaseline a göre %30 daha iyi',
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#D4EDDA', alpha=0.9))
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_nokta_optimizasyon.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Şekil 4: fig4_nokta_optimizasyon.png")

# ─────────────────────────────────────────────
# ŞEKİL 5: Max-Pool vs Attention-Pool
# ─────────────────────────────────────────────

def fig5_max_vs_attn_pool():
    """Max-pool vs attention-pool karşılaştırması"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 20K sonuçları
    ax = axes[0]
    pools = ['Max-Pool', 'Attention-Pool']
    ppl_20k = [315.99, 334.89]
    colors = ['#28A745', '#DC3545']
    
    bars = ax.bar(pools, ppl_20k, color=colors, alpha=0.8)
    ax.set_ylabel('Held-Out PPL (20K)', fontsize=12, fontweight='bold')
    ax.set_title('20K Korpus', fontsize=13, fontweight='bold')
    
    for bar, val in zip(bars, ppl_20k):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
               f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 200K sonuçları
    ax = axes[1]
    ppl_200k = [61.91, 66.85]
    
    bars = ax.bar(pools, ppl_200k, color=colors, alpha=0.8)
    ax.set_ylabel('Held-Out PPL (200K)', fontsize=12, fontweight='bold')
    ax.set_title('200K Korpus', fontsize=13, fontweight='bold')
    
    for bar, val in zip(bars, ppl_200k):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle('Şekil 5: Max-Pool vs Attention-Pool Karşılaştırması', fontsize=14, fontweight='bold', y=1.02)
    
    # Açıklama
    fig.text(0.5, 0.02, 'Max-pool her iki korpusa da kazandı (20K: Δ18.90, 200K: Δ5.91)\nAttention-pool küçük korpusta overfitting eğiliminde',
            ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round', facecolor='#F8D7DA', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_max_vs_attn_pool.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Şekil 5: fig5_max_vs_attn_pool.png")

# ─────────────────────────────────────────────
# ŞEKİL 6: Bulut Geometri Vizualizasyonu
# ─────────────────────────────────────────────

def fig6_cloud_visualization():
    """Bulut geometri vizualizasyonu (3D)"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Örnek cümle noktaları
    np.random.seed(42)
    n_points = 12
    points = np.random.randn(n_points, 3) * 0.5
    
    # Renk paleti
    colors = plt.cm.tab10(np.linspace(0, 1, n_points))
    
    # Noktaları çiz
    for i, pt in enumerate(points):
        ax.scatter(pt[0], pt[1], pt[2], s=120, color=colors[i], 
                  edgecolors='black', linewidth=0.8, alpha=0.9)
    
    # Mesafe çizgileri (bazı çiftler)
    for i in range(n_points):
        for j in range(i+1, min(i+3, n_points)):
            dist = np.linalg.norm(points[i] - points[j])
            ax.plot([points[i, 0], points[j, 0]],
                   [points[i, 1], points[j, 1]],
                   [points[i, 2], points[j, 2]],
                   'gray', alpha=0.3, lw=0.8)
    
    ax.set_title('Şekil 6: Bulut Geometri Vizualizasyonu (3D)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # İstatistik
    all_dists = [np.linalg.norm(points[i]-points[j]) 
                for i in range(n_points) for j in range(i+1, n_points)]
    ax.text2D(0.05, 0.95, f'Nokta sayısı: {n_points}\nMin mesafe: {min(all_dists):.3f}\nMax mesafe: {max(all_dists):.3f}',
             transform=ax.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_bulut_vizualizasyon.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Şekil 6: fig6_bulut_vizualizasyon.png")

# ─────────────────────────────────────────────
# TÜM ŞEKİLLERİ ÜRET
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PATENT ŞEKİLLERİ ÜRETİMİ (v2.4)")
    print("=" * 60)
    
    fig1_architecture()
    fig2_pair_geometry()
    fig3_explanation()
    fig4_point_count_optimization()
    fig5_max_vs_attn_pool()
    fig6_cloud_visualization()
    
    print("\n" + "=" * 60)
    print("  ✅ TÜM ŞEKİLLER TAMAM: docs/patent/fig/")
    print("=" * 60)
